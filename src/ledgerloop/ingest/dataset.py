"""Read a whole dataset directory: three files in, one canonical corpus out.

The seam between "somebody else's format" and this system. Everything upstream
is CSV and JSON written by three parties that never agreed on anything;
everything downstream is :mod:`ledgerloop.models.records` and can be reasoned
about uniformly.

WHAT THIS IS NOT
----------------
It is **not** the matcher, and it deliberately builds no cross-source links.
:attr:`CanonicalPayment.order_ref_normalized` is populated; whether it names a
real order is T0's question, at step 4. Ingest that quietly joined would move
the first tier's work into an unmeasured layer, and the first defensible number
would then be measuring two things at once.

It is also **not** what B0 reads. The naive readers in
:mod:`ledgerloop.eval.baselines` stay naive on purpose: a baseline sharing the
system's normalisation measures the system twice.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ledgerloop.generator.emitters import BANK_FILE, LEDGER_FILE, PSP_FILE
from ledgerloop.ingest.bank import BankRecords, parse_bank_csv
from ledgerloop.ingest.dates import DateOrder, DateOrderEvidence
from ledgerloop.ingest.ledger import parse_ledger_csv
from ledgerloop.ingest.narration import NarrationParse
from ledgerloop.ingest.normalize import is_order_ref_shaped
from ledgerloop.ingest.problems import IngestError, IngestProblem, ProblemLog
from ledgerloop.ingest.psp import parse_psp_json
from ledgerloop.models.enums import SourceName
from ledgerloop.models.records import (
    CanonicalBankTxn,
    CanonicalOrder,
    CanonicalPayment,
    CanonicalRecord,
    CanonicalSettlement,
    RawRecord,
)

__all__ = ["IngestResult", "ingest_dataset"]


@dataclass(frozen=True)
class IngestResult:
    """One dataset, normalised, with the evidence for how it was read."""

    orders: tuple[CanonicalOrder, ...]
    payments: tuple[CanonicalPayment, ...]
    settlements: tuple[CanonicalSettlement, ...]
    bank_txns: tuple[CanonicalBankTxn, ...]

    problems: tuple[IngestProblem, ...]
    date_order: DateOrderEvidence
    narrations: dict[str, NarrationParse] = field(default_factory=dict)
    wall_clock_ms: int = 0

    @property
    def record_count(self) -> int:
        return (
            len(self.orders) + len(self.payments) + len(self.settlements) + len(self.bank_txns)
        )

    @property
    def normalized(self) -> list[CanonicalRecord]:
        """Every canonical record, in source order. Feeds ``ReconState.normalized``."""
        records: list[CanonicalRecord] = []
        records.extend(self.orders)
        records.extend(self.settlements)
        records.extend(self.payments)
        records.extend(self.bank_txns)
        return records

    @property
    def raw_by_source(self) -> dict[SourceName, list[RawRecord]]:
        """Provenance grouped the way ``ReconState.raw`` holds it.

        Every canonical record points home, so this is derived rather than
        collected separately -- there is no way for the two to disagree.
        """
        grouped: dict[SourceName, list[RawRecord]] = {source: [] for source in SourceName}
        for record in self.normalized:
            if record.raw is not None:
                grouped[record.source].append(record.raw)
        return grouped

    # -- diagnostics the report and the step-4 tiers care about ----------

    @property
    def credits(self) -> tuple[CanonicalBankTxn, ...]:
        """Incoming money. Only these are settlement candidates."""
        return tuple(txn for txn in self.bank_txns if txn.is_credit)

    @property
    def credits_with_utr(self) -> int:
        return sum(1 for txn in self.credits if txn.extracted_utr is not None)

    @property
    def credits_with_merchant(self) -> int:
        return sum(1 for txn in self.credits if txn.extracted_merchant is not None)

    @property
    def credits_with_no_reference(self) -> int:
        """A07 plus the noise rows: nothing recoverable in the narration at all."""
        return sum(
            1
            for txn in self.credits
            if txn.extracted_utr is None and txn.extracted_merchant is None
        )

    @property
    def payments_with_usable_ref(self) -> int:
        """Payments whose normalised reference matches the published grammar.

        The gap between this and the payment count is the ceiling T0 cannot
        exceed on reference alone, and it is baseline mess rather than an
        anomaly class -- see :func:`~ledgerloop.ingest.normalize.normalize_order_ref`.
        """
        return sum(
            1 for payment in self.payments if is_order_ref_shaped(payment.order_ref_normalized)
        )

    @property
    def payments_with_recovered_ref(self) -> int:
        """Payments whose reference was mangled in the file but recovered here.

        The measured value of normalisation: these are exact-join matches that
        a parser reading ``order_ref`` literally would have missed.
        """
        return sum(
            1
            for payment in self.payments
            if payment.order_ref_raw is not None
            and payment.order_ref_raw != payment.order_ref_normalized
            and is_order_ref_shaped(payment.order_ref_normalized)
        )

    @property
    def payments_with_no_ref(self) -> int:
        """Payments the PSP published without any reference at all."""
        return sum(1 for payment in self.payments if payment.order_ref_raw is None)

    def problems_by_source(self, source: SourceName) -> tuple[IngestProblem, ...]:
        return tuple(problem for problem in self.problems if problem.source is source)

    def __iter__(self) -> Iterator[CanonicalRecord]:
        return iter(self.normalized)


def ingest_dataset(
    directory: Path,
    *,
    strict: bool = False,
    default_date_order: DateOrder = DateOrder.DAY_FIRST,
) -> IngestResult:
    """Parse and normalise the three sources in a generated dataset directory.

    ``strict=True`` turns every quarantined record into an
    :class:`~ledgerloop.ingest.problems.IngestError`. The test suite uses it so
    that a fixture which starts losing rows fails the build rather than quietly
    shrinking the corpus every later metric is computed over.
    """
    started_ns = time.perf_counter_ns()

    for filename in (LEDGER_FILE, PSP_FILE, BANK_FILE):
        if not (directory / filename).is_file():
            raise IngestError(f"{directory}: missing required source file {filename}")

    log = ProblemLog(strict=strict)
    orders = parse_ledger_csv(directory / LEDGER_FILE, log)
    psp = parse_psp_json(directory / PSP_FILE, log)
    bank: BankRecords = parse_bank_csv(
        directory / BANK_FILE, log, default_date_order=default_date_order
    )

    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
    return IngestResult(
        orders=orders,
        payments=psp.payments,
        settlements=psp.settlements,
        bank_txns=bank.transactions,
        problems=log.problems,
        date_order=bank.date_order,
        narrations=bank.narrations,
        wall_clock_ms=int(elapsed_ms),
    )
