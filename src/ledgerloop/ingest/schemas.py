"""Declared input schemas for the three sources.

Each source gets a schema object naming its file, its fields, and which of them
are required. The parsers validate against it before reading a single record.

WHY THIS IS SEPARATE FROM THE PARSERS
-------------------------------------
Three reasons, in ascending order of importance:

1. A schema mismatch produces one clear message naming the missing columns,
   rather than a ``KeyError`` from somewhere in the middle of a row loop.
2. The three sources are heterogeneous on purpose (PLAN.md 5.1) -- CSV, nested
   JSON, and CSV-with-free-text. Writing down what each one *is* keeps that
   heterogeneity legible instead of scattering it across three parsers.
3. It draws the line the whole ingest layer exists to draw. Everything to the
   left of these schemas is somebody else's format; everything to the right is
   :mod:`ledgerloop.models.records`. When a new source arrives, this file is
   the specification of what has to be produced.

A missing **required** field is a file-level failure and raises. A missing
optional field is absent data, and the parsers handle it as such -- the PSP's
``adjustments_paise`` is genuinely optional, and defaulting it to zero is
correct rather than a guess.

Extra fields are **allowed and ignored**. A bank adding a column should not
stop the month-end run, and the raw payload keeps the value anyway, so nothing
is lost -- unlike the model layer, where ``extra="forbid"`` guards against
hallucinated LLM keys and the trade-off runs the other way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from ledgerloop.generator.emitters import BANK_FILE, LEDGER_FILE, PSP_FILE
from ledgerloop.models.enums import SourceName

__all__ = [
    "BANK_SCHEMA",
    "LEDGER_SCHEMA",
    "PSP_PAYMENT_SCHEMA",
    "PSP_SETTLEMENT_SCHEMA",
    "SourceSchema",
    "value_of",
]


@dataclass(frozen=True)
class SourceSchema:
    """The fields one record of one source is expected to carry."""

    source: SourceName
    entity: str
    filename: str | None
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def fields(self) -> tuple[str, ...]:
        return self.required + self.optional

    def missing_from(self, present: Iterable[str]) -> tuple[str, ...]:
        """Required fields absent from ``present``, in declaration order."""
        available = {str(name).strip() for name in present}
        return tuple(name for name in self.required if name not in available)

    def describe_mismatch(self, present: Iterable[str]) -> str | None:
        """A message naming what is missing, or ``None`` when the shape is fine."""
        missing = self.missing_from(present)
        if not missing:
            return None
        return (
            f"{self.filename or self.entity}: {self.entity} is missing required "
            f"field(s) {', '.join(missing)}; expected {', '.join(self.required)}"
        )


def value_of(record: Mapping[str, object], field: str) -> object | None:
    """Read a field, treating an empty or whitespace-only string as absent.

    CSV has no null. An empty cell and a missing column are the same absence to
    everything downstream, and collapsing them here means the parsers do not
    each have to remember that.
    """
    raw = record.get(field)
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return raw


#: Source A -- the internal ledger. Clean, structured, our system of record.
LEDGER_SCHEMA: Final[SourceSchema] = SourceSchema(
    source=SourceName.LEDGER,
    entity="order",
    filename=LEDGER_FILE,
    required=(
        "order_id",
        "merchant_id",
        "customer_ref",
        "amount_gross_paise",
        "booked_at",
        "status",
    ),
    # Absent in some exports; INR is the only supported currency anyway, and a
    # missing column is not licence to skip the check on the rows that have it.
    optional=("currency",),
)

#: Source B, outer -- the payout batch. Fees and tax live here.
PSP_SETTLEMENT_SCHEMA: Final[SourceSchema] = SourceSchema(
    source=SourceName.PSP,
    entity="settlement",
    filename=PSP_FILE,
    required=("settlement_id", "settled_on", "gross_paise", "fee_paise", "net_paise"),
    # `utr` is optional because a settlement genuinely may not publish one, and
    # `tax_paise` / `adjustments_paise` because zero is the honest default for a
    # batch that incurred neither.
    optional=("utr", "tax_paise", "adjustments_paise", "merchant_id", "payments"),
)

#: Source B, inner -- one payment inside a batch.
PSP_PAYMENT_SCHEMA: Final[SourceSchema] = SourceSchema(
    source=SourceName.PSP,
    entity="payment",
    filename=PSP_FILE,
    required=("payment_id", "amount_paise", "captured_at"),
    # `order_ref` is optional because the PSP writes `null` for roughly one
    # payment in fourteen (PLAN.md 5.1). That is expected input, not a defect.
    optional=("order_ref",),
)

#: Source C -- the bank statement. Free text, DD/MM/YYYY dates.
BANK_SCHEMA: Final[SourceSchema] = SourceSchema(
    source=SourceName.BANK,
    entity="bank_txn",
    filename=BANK_FILE,
    required=("txn_id", "value_date", "narration", "credit_paise", "debit_paise"),
    optional=("balance_paise",),
)
