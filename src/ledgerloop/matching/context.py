"""The indexes and the residual pool every tier shares.

PLAN.md 6.1: "Cheapest and most certain first. **A record leaves the pool once
decided.**" That sentence is the whole reason this module exists. A tier ladder
without a shared pool is not a ladder -- it is six independent matchers whose
outputs contradict each other, and the contradiction surfaces as a precision
loss nobody can attribute.

WHAT "DECIDED" MEANS
--------------------
Deliberately not the same as "matched". A settlement leaves the pool when a
tier has **reached a conclusion** about it, and there are two ways to conclude:

* **Resolved** -- exactly one qualifying counterpart. The pair is consumed.
* **Contested** -- several equally qualified counterparts. The tier cannot
  choose between them, and a later tier applying a *looser* rule to the same
  evidence could only choose arbitrarily. So the settlement leaves the pool
  carrying its contenders, and the decision policy routes it for review.

A settlement with **no** qualifying counterpart has not been decided at all. It
stays in the pool and falls through to the next tier, which is exactly how
anomaly A02 (a credit three paise off the declared net) reaches T1.

This distinction is what makes "T1 cannot silently override a stronger T0
result" structural rather than a convention each tier has to remember: T1 never
sees a pair T0 ruled on.

EXCLUSIVITY
-----------
Consumption is two-sided. A bank credit consumed by one settlement cannot be
claimed by another, and a settlement resolved against one credit cannot be
re-matched. One payout, one credit -- and where the sources say otherwise, that
is an anomaly to report rather than a pair to match twice.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.models.records import (
    CanonicalBankTxn,
    CanonicalOrder,
    CanonicalPayment,
    CanonicalSettlement,
)
from ledgerloop.money import sum_minor

__all__ = ["MatchContext", "SettlementView"]


@dataclass(frozen=True)
class SettlementView:
    """A settlement together with the payments the PSP nested inside it.

    The pairing is read straight off the source document -- ``payment_ids`` is
    the file's own nesting -- so it is asserted, not inferred, and no tier has
    to earn it. What a tier earns is the *bank* end of the chain.
    """

    settlement: CanonicalSettlement
    payments: tuple[CanonicalPayment, ...]

    @property
    def settlement_id(self) -> str:
        return self.settlement.settlement_id

    @property
    def utr(self) -> str | None:
        return self.settlement.utr

    @property
    def net_minor(self) -> int:
        return self.settlement.net_minor

    @property
    def payment_gross_minor(self) -> int:
        """What the nested payments actually sum to.

        Compared against the settlement's declared ``gross_minor`` by
        :meth:`gross_reconciles`. They agree on every well-formed batch; a
        disagreement means a payment was quarantined at ingest or the report is
        internally inconsistent, and either way the batch must not be
        auto-matched on the strength of a total that does not hold.
        """
        return sum_minor(
            (payment.amount_minor for payment in self.payments),
            field=f"{self.settlement_id}.payments",
        )

    @property
    def gross_reconciles(self) -> bool:
        """Whether the declared gross equals the sum of the nested payments.

        The arithmetic gate for T0 and T1. **Not** the settlement identity
        ``net == gross - fee - tax + adjustments``, which anomaly A03 breaks on
        purpose and which is reported as evidence rather than enforced.
        """
        return bool(self.payments) and self.payment_gross_minor == self.settlement.gross_minor


@dataclass
class MatchContext:
    """Indexes over one ingested dataset, plus the tier ladder's residual pool.

    Built once and threaded through every tier. Iteration order follows source
    order throughout, so a run is reproducible without sorting at each use.
    """

    orders: tuple[CanonicalOrder, ...]
    settlements: tuple[SettlementView, ...]
    bank_txns: tuple[CanonicalBankTxn, ...]

    orders_by_id: dict[str, CanonicalOrder] = field(default_factory=dict)
    settlements_by_id: dict[str, SettlementView] = field(default_factory=dict)
    credits_by_utr: dict[str, tuple[CanonicalBankTxn, ...]] = field(default_factory=dict)
    settlements_by_utr: dict[str, tuple[SettlementView, ...]] = field(default_factory=dict)

    consumed_settlements: set[str] = field(default_factory=set)
    consumed_credits: set[str] = field(default_factory=set)

    @classmethod
    def from_ingest(cls, ingest: IngestResult) -> MatchContext:
        """Build the indexes from an :class:`IngestResult`.

        Payments are grouped by the ``settlement_id`` the parser recorded rather
        than by the settlement's ``payment_ids`` list, so a payment quarantined
        at ingest simply does not appear -- and :attr:`SettlementView.
        gross_reconciles` then fails, which is the intended consequence.
        """
        payments_by_settlement: dict[str, list[CanonicalPayment]] = {}
        for payment in ingest.payments:
            if payment.settlement_id is not None:
                payments_by_settlement.setdefault(payment.settlement_id, []).append(payment)

        views = tuple(
            SettlementView(
                settlement=settlement,
                payments=tuple(payments_by_settlement.get(settlement.settlement_id, ())),
            )
            for settlement in ingest.settlements
        )

        credits_by_utr: dict[str, list[CanonicalBankTxn]] = {}
        for txn in ingest.bank_txns:
            if txn.is_credit and txn.extracted_utr is not None:
                credits_by_utr.setdefault(txn.extracted_utr, []).append(txn)

        settlements_by_utr: dict[str, list[SettlementView]] = {}
        for view in views:
            if view.utr is not None:
                settlements_by_utr.setdefault(view.utr, []).append(view)

        return cls(
            orders=ingest.orders,
            settlements=views,
            bank_txns=ingest.bank_txns,
            orders_by_id={order.order_id: order for order in ingest.orders},
            settlements_by_id={view.settlement_id: view for view in views},
            credits_by_utr={utr: tuple(rows) for utr, rows in credits_by_utr.items()},
            settlements_by_utr={utr: tuple(rows) for utr, rows in settlements_by_utr.items()},
        )

    # -- the residual pool ------------------------------------------------

    def open_settlements(self) -> Iterator[SettlementView]:
        """Settlements no tier has ruled on yet, in source order."""
        for view in self.settlements:
            if view.settlement_id not in self.consumed_settlements:
                yield view

    def open_credits_for(self, utr: str) -> tuple[CanonicalBankTxn, ...]:
        """Unconsumed credits whose narration published ``utr``, in source order."""
        return tuple(
            txn
            for txn in self.credits_by_utr.get(utr, ())
            if txn.txn_id not in self.consumed_credits
        )

    def open_credits(self) -> tuple[CanonicalBankTxn, ...]:
        """Every unclaimed incoming row, keyed or not, in source order.

        The keyed indexes cannot answer "is there another row that could be
        this payout?", because the row that could be is exactly the one whose
        reference went missing. See the mutual-uniqueness check in
        :mod:`ledgerloop.matching.bank_leg`.
        """
        return tuple(
            txn
            for txn in self.bank_txns
            if txn.is_credit and txn.txn_id not in self.consumed_credits
        )

    def open_settlements_for(self, utr: str) -> tuple[SettlementView, ...]:
        """Unconsumed settlements publishing ``utr``. More than one is a collision."""
        return tuple(
            view
            for view in self.settlements_by_utr.get(utr, ())
            if view.settlement_id not in self.consumed_settlements
        )

    def consume(self, settlement_id: str, credit_ids: Sequence[str] = ()) -> None:
        """Remove a settlement, and any credits it claimed, from the pool.

        Called for a resolved pair *and* for a contested one. See the module
        docstring: reaching a conclusion is what removes a record, not reaching
        a match.
        """
        self.consumed_settlements.add(settlement_id)
        self.consumed_credits.update(credit_ids)

    # -- diagnostics the report explains the score with -------------------

    @property
    def credits(self) -> tuple[CanonicalBankTxn, ...]:
        return tuple(txn for txn in self.bank_txns if txn.is_credit)

    @property
    def credits_with_utr(self) -> int:
        return sum(1 for txn in self.credits if txn.extracted_utr is not None)

    @property
    def settlements_with_utr(self) -> int:
        return sum(1 for view in self.settlements if view.utr is not None)
