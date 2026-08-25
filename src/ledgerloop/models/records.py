"""Canonical records -- the normalised form of the three sources.

Four entity types from three files (PLAN.md §5.1). The PSP file yields both
settlements and the payments nested inside them, and they are separate
entities because the whole N:1 problem is about which payments compose a
settlement.

A deliberate non-decision: **none of these models validate the settlement
arithmetic**. ``net = gross - fee - tax + adjustments`` is exactly the
invariant that anomaly A03 ``FEE_TAX_MISMATCH`` breaks on purpose. A Pydantic
validator enforcing it would make 4% of the corpus unparseable and delete the
anomaly the system is supposed to detect. The check lives in
:meth:`CanonicalSettlement.net_delta_minor` and its result is *evidence*, not a
gate.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import Currency, OrderStatus, RecordType, SourceName
from ledgerloop.models.refs import RecordRef

__all__ = [
    "CanonicalBankTxn",
    "CanonicalOrder",
    "CanonicalPayment",
    "CanonicalRecord",
    "CanonicalSettlement",
    "RawRecord",
]


class RawRecord(FrozenLedgerModel):
    """One unparsed row/object, kept so every canonical record can point home.

    The audit trail has to be able to show a controller the original line from
    the bank statement, not just the system's interpretation of it.
    """

    source: SourceName
    source_line: int = Field(ge=0, description="0-based position within the source file")
    payload: dict[str, object]


class _CanonicalBase(FrozenLedgerModel):
    """Fields shared by every canonical record."""

    source: SourceName
    raw: RawRecord | None = Field(
        default=None,
        description="Provenance back to the unparsed source row; None for synthesised records.",
    )

    @property
    def ref(self) -> RecordRef:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError


class CanonicalOrder(_CanonicalBase):
    """Source A -- the internal ledger. Our system of record."""

    record_type: Literal[RecordType.ORDER] = RecordType.ORDER
    source: Literal[SourceName.LEDGER] = SourceName.LEDGER

    order_id: str
    merchant_id: str
    customer_ref: str
    amount_minor: MinorUnits
    currency: Currency = Currency.INR
    booked_at: datetime
    status: OrderStatus

    @property
    def ref(self) -> RecordRef:
        return RecordRef(record_type=RecordType.ORDER, record_id=self.order_id)


class CanonicalPayment(_CanonicalBase):
    """Source B, inner -- one payment inside a settlement batch.

    ``order_ref_raw`` preserves whatever the PSP actually wrote, including the
    deliberate corruptions of PLAN.md §5.1 (``null``, ``"ord 2026 004821"``,
    or a non-ASCII hyphen). ``order_ref_normalized`` is what normalisation
    recovered, or ``None`` when it could not. Keeping both is what lets the
    audit trail explain *why* a T0 exact join missed.
    """

    record_type: Literal[RecordType.PAYMENT] = RecordType.PAYMENT
    source: Literal[SourceName.PSP] = SourceName.PSP

    payment_id: str
    settlement_id: str | None = Field(
        default=None,
        description="Declared parent settlement. When present, T2 anchors on it "
        "and verifies one subset instead of searching all of them.",
    )
    order_ref_raw: str | None = None
    order_ref_normalized: str | None = None
    amount_minor: MinorUnits
    currency: Currency = Currency.INR
    captured_at: datetime

    @property
    def ref(self) -> RecordRef:
        return RecordRef(record_type=RecordType.PAYMENT, record_id=self.payment_id)


class CanonicalSettlement(_CanonicalBase):
    """Source B, outer -- a payout batch. Fees and tax live here.

    This is the middle layer that makes the problem three-way rather than
    two-way: gross never equals net, so nothing joins on amount alone.
    """

    record_type: Literal[RecordType.SETTLEMENT] = RecordType.SETTLEMENT
    source: Literal[SourceName.PSP] = SourceName.PSP

    settlement_id: str
    utr: str | None = None
    settled_on: date
    gross_minor: MinorUnits
    fee_minor: MinorUnits
    tax_minor: MinorUnits
    adjustments_minor: MinorUnits = Field(
        default=0,
        description="Signed. Negative for refunds and chargebacks netted off the payout "
        "(anomalies A06 and A08).",
    )
    net_minor: MinorUnits
    payment_ids: tuple[str, ...] = ()
    currency: Currency = Currency.INR

    @property
    def ref(self) -> RecordRef:
        return RecordRef(record_type=RecordType.SETTLEMENT, record_id=self.settlement_id)

    @property
    def expected_net_minor(self) -> int:
        """What ``net_minor`` should be if the settlement is internally consistent."""
        return self.gross_minor - self.fee_minor - self.tax_minor + self.adjustments_minor

    @property
    def net_delta_minor(self) -> int:
        """Declared net minus expected net. Non-zero signals A03.

        Reported as evidence, never enforced -- see the module docstring.
        """
        return self.net_minor - self.expected_net_minor


class CanonicalBankTxn(_CanonicalBase):
    """Source C -- the bank statement. The messy one.

    ``narration_raw`` is free text with no structured reference field.
    ``extracted_utr`` / ``extracted_merchant`` are what the regex-first parser
    recovered (LLM fallback only on regex miss, PLAN.md §7.3). Both are
    ``None`` for anomaly A07 ``MISSING_REFERENCE``, which is the point.

    Credits and debits are separate non-negative fields, mirroring how bank
    statements are actually published, so a sign convention error cannot turn
    an outgoing payment into incoming money.
    """

    record_type: Literal[RecordType.BANK_TXN] = RecordType.BANK_TXN
    source: Literal[SourceName.BANK] = SourceName.BANK

    txn_id: str
    value_date: date
    narration_raw: str
    narration_normalized: str | None = None
    extracted_utr: str | None = None
    extracted_merchant: str | None = None
    credit_minor: MinorUnits = 0
    debit_minor: MinorUnits = 0
    balance_minor: MinorUnits | None = None
    currency: Currency = Currency.INR

    @property
    def ref(self) -> RecordRef:
        return RecordRef(record_type=RecordType.BANK_TXN, record_id=self.txn_id)

    @property
    def signed_amount_minor(self) -> int:
        """Credit positive, debit negative."""
        return self.credit_minor - self.debit_minor

    @property
    def is_credit(self) -> bool:
        """Whether this row is incoming money.

        Only credits are settlement candidates. Debits and the unrelated noise
        rows of PLAN.md §5.1 (rent, salary, vendor payments) must match nothing,
        and are counted as true negatives by the evaluator.
        """
        return self.credit_minor > 0


#: Discriminated union over the four entity types.
#:
#: The discriminator makes ``model_validate`` pick the right class from JSON
#: without a hand-written dispatch, and makes ``match`` statements over records
#: exhaustive for mypy.
CanonicalRecord = Annotated[
    CanonicalOrder | CanonicalPayment | CanonicalSettlement | CanonicalBankTxn,
    Field(discriminator="record_type"),
]
