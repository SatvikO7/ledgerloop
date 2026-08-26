"""Builders for small, hand-checkable matching corpora.

The matching tests need datasets whose every number can be verified by eye --
one settlement, two payments, a credit that is three paise short. Generating
those with the real generator would be indirect and slow, and reading them off
the committed fixture would tie a unit test to a corpus it does not control.

So these builders assemble :class:`~ledgerloop.ingest.dataset.IngestResult`
objects directly from canonical records, with defaults chosen so the common
case is one line:

    corpus(settlements=[batch()])

Everything is consistent by default -- the settlement's gross equals its
payments, the credit equals the declared net, the dates are one day apart -- so
a test states only the thing it is varying, and a reader can tell what is being
tested from the arguments alone.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.ingest.dates import DateOrder, DateOrderEvidence
from ledgerloop.models.enums import OrderStatus, SourceName
from ledgerloop.models.records import (
    CanonicalBankTxn,
    CanonicalOrder,
    CanonicalPayment,
    CanonicalSettlement,
    RawRecord,
)

#: The day everything hangs off. Settlements settle here, credits land T+1.
SETTLED_ON = date(2026, 3, 10)
CREDITED_ON = SETTLED_ON + timedelta(days=1)
BOOKED_AT = datetime(2026, 3, 9, 12, 0, 0)

__all__ = [
    "BOOKED_AT",
    "CREDITED_ON",
    "SETTLED_ON",
    "Batch",
    "bank_credit",
    "batch",
    "corpus",
    "debit_row",
    "make_order",
    "noise_credit",
]


def _raw(source: SourceName, line: int) -> RawRecord:
    """Provenance for a synthesised record. Present so traceability tests hold."""
    return RawRecord(source=source, source_line=line, payload={"synthetic": True})


def make_order(
    order_id: str = "ORD-2026-000001",
    *,
    amount_minor: int = 100_000,
    line: int = 0,
    merchant_id: str = "MRCH_0001",
    customer_ref: str = "CUST_10001",
    status: OrderStatus = OrderStatus.CAPTURED,
) -> CanonicalOrder:
    return CanonicalOrder(
        raw=_raw(SourceName.LEDGER, line),
        order_id=order_id,
        merchant_id=merchant_id,
        customer_ref=customer_ref,
        amount_minor=amount_minor,
        booked_at=BOOKED_AT,
        status=status,
    )


def _payment(
    payment_id: str,
    settlement_id: str,
    amount_minor: int,
    *,
    order_ref: str | None,
    line: int,
) -> CanonicalPayment:
    return CanonicalPayment(
        raw=_raw(SourceName.PSP, line),
        payment_id=payment_id,
        settlement_id=settlement_id,
        order_ref_raw=order_ref,
        order_ref_normalized=order_ref,
        amount_minor=amount_minor,
        captured_at=BOOKED_AT,
    )


class Batch:
    """A settlement, its payments and the orders behind them, built together.

    Grouped because they are only meaningful together: a settlement's gross is
    the sum of its payments, and a payment's amount is what its order was booked
    for. Building them separately would make every test restate those two
    invariants, and a test that got one wrong would be testing the wrong thing.
    """

    def __init__(
        self,
        settlement_id: str = "SETL-0001",
        *,
        utr: str | None = "UTR2026031012345",
        amounts: tuple[int, ...] = (60_000, 40_000),
        fee_minor: int = 0,
        tax_minor: int = 0,
        adjustments_minor: int = 0,
        net_minor: int | None = None,
        gross_minor: int | None = None,
        settled_on: date = SETTLED_ON,
        order_refs: tuple[str | None, ...] | None = None,
        first_index: int = 1,
        merchant_id: str = "MRCH_0001",
        customer_refs: tuple[str, ...] | None = None,
    ) -> None:
        gross = sum(amounts)
        self.merchant_id = merchant_id
        self.orders = tuple(
            make_order(
                f"ORD-2026-{first_index + i:06d}",
                amount_minor=amount,
                line=first_index + i,
                merchant_id=merchant_id,
                customer_ref=(
                    customer_refs[i] if customer_refs is not None else f"CUST_{10001 + i}"
                ),
            )
            for i, amount in enumerate(amounts)
        )
        refs: tuple[str | None, ...] = (
            order_refs
            if order_refs is not None
            else tuple(order.order_id for order in self.orders)
        )
        self.payments = tuple(
            _payment(
                f"PAY-{first_index + i:05d}",
                settlement_id,
                amount,
                order_ref=refs[i],
                line=first_index + i,
            )
            for i, amount in enumerate(amounts)
        )
        self.settlement = CanonicalSettlement(
            raw=_raw(SourceName.PSP, first_index),
            settlement_id=settlement_id,
            utr=utr,
            settled_on=settled_on,
            gross_minor=gross if gross_minor is None else gross_minor,
            fee_minor=fee_minor,
            tax_minor=tax_minor,
            adjustments_minor=adjustments_minor,
            net_minor=(
                gross - fee_minor - tax_minor + adjustments_minor
                if net_minor is None
                else net_minor
            ),
            payment_ids=tuple(payment.payment_id for payment in self.payments),
        )

    @property
    def net_minor(self) -> int:
        return self.settlement.net_minor

    def credit(
        self,
        txn_id: str = "BNK-00001",
        *,
        delta_minor: int = 0,
        days_after: int = 1,
        utr: str | None = "",
    ) -> CanonicalBankTxn:
        """A bank credit for this batch, offset by ``delta_minor`` if asked."""
        return bank_credit(
            txn_id,
            amount_minor=self.net_minor + delta_minor,
            utr=self.settlement.utr if utr == "" else utr,
            value_date=self.settlement.settled_on + timedelta(days=days_after),
        )


def batch(settlement_id: str = "SETL-0001", **kwargs: object) -> Batch:
    """Shorthand for :class:`Batch`."""
    return Batch(settlement_id, **kwargs)  # type: ignore[arg-type]


def bank_credit(
    txn_id: str = "BNK-00001",
    *,
    amount_minor: int = 100_000,
    utr: str | None = "UTR2026031012345",
    value_date: date = CREDITED_ON,
    merchant: str | None = "RAZORPAY SOFTWARE PVT",
    line: int = 0,
) -> CanonicalBankTxn:
    narration = f"NEFT CR-{merchant}-{utr or ''}-SETTLEMENT"
    return CanonicalBankTxn(
        raw=_raw(SourceName.BANK, line),
        txn_id=txn_id,
        value_date=value_date,
        narration_raw=narration,
        narration_normalized=narration.replace("-", " "),
        extracted_utr=utr,
        extracted_merchant=merchant,
        credit_minor=amount_minor,
        debit_minor=0,
    )


def noise_credit(txn_id: str = "BNK-09001", *, amount_minor: int = 100_000) -> CanonicalBankTxn:
    """An unrelated incoming row. No reference, must match nothing."""
    return CanonicalBankTxn(
        raw=_raw(SourceName.BANK, 900),
        txn_id=txn_id,
        value_date=CREDITED_ON,
        narration_raw="INTEREST CREDIT SAVINGS",
        narration_normalized="INTEREST CREDIT SAVINGS",
        credit_minor=amount_minor,
        debit_minor=0,
    )


def debit_row(
    txn_id: str = "BNK-09002",
    *,
    amount_minor: int = 100_000,
    utr: str | None = "UTR2026031012345",
) -> CanonicalBankTxn:
    """Outgoing money carrying a settlement's UTR. Must never be matched."""
    narration = f"NEFT DR-RAZORPAY SOFTWARE PVT-{utr or ''}-SETTLEMENT"
    return CanonicalBankTxn(
        raw=_raw(SourceName.BANK, 901),
        txn_id=txn_id,
        value_date=CREDITED_ON,
        narration_raw=narration,
        narration_normalized=narration.replace("-", " "),
        extracted_utr=utr,
        credit_minor=0,
        debit_minor=amount_minor,
    )


def corpus(
    *,
    batches: tuple[Batch, ...] | list[Batch] = (),
    bank_txns: tuple[CanonicalBankTxn, ...] | list[CanonicalBankTxn] = (),
    extra_orders: tuple[CanonicalOrder, ...] | list[CanonicalOrder] = (),
) -> IngestResult:
    """Assemble an :class:`IngestResult` from batches and bank rows."""
    return IngestResult(
        orders=tuple(order for b in batches for order in b.orders) + tuple(extra_orders),
        payments=tuple(payment for b in batches for payment in b.payments),
        settlements=tuple(b.settlement for b in batches),
        bank_txns=tuple(bank_txns),
        problems=(),
        date_order=DateOrderEvidence(
            order=DateOrder.DAY_FIRST,
            proven=True,
            day_first_witnesses=1,
            month_first_witnesses=0,
            ambiguous_values=0,
            unparsable_values=0,
            total_values=len(tuple(bank_txns)),
        ),
    )


@pytest.fixture
def simple() -> IngestResult:
    """One clean batch of two payments, credited in full. The happy path."""
    only = batch()
    return corpus(batches=[only], bank_txns=[only.credit()])
