"""The mutable draft world, and the clean baseline it starts from.

Generation is two-phase:

1. **Build a world that reconciles perfectly.** Every order has a payment, every
   payment sits in a settlement, every settlement lands as one bank credit for
   its net.
2. **Apply scenarios** that break it in the eleven specific, labelled ways of
   :class:`~ledgerloop.models.enums.AnomalyClass`.

The draft types here are mutable dataclasses because phase 2 edits them. The
frozen Pydantic contracts are produced at the end, once the world has settled --
which is also what keeps the anomalies from having to fight
``FrozenLedgerModel``.

**Ground truth is derived from these effects, never inferred from the data.**
Each scenario appends a :class:`ScenarioEffect` recording what it did, how much
money it put at stake, and by how much it deliberately broke bank-vs-settlement
conservation. The conservation property test is written against those declared
deltas, so "money is conserved modulo declared anomalies" is a checkable
statement rather than a slogan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ledgerloop.models.enums import AnomalyClass, ExpectedStatus, OrderStatus
from ledgerloop.models.refs import RecordRef

__all__ = [
    "DraftBankTxn",
    "DraftOrder",
    "DraftPayment",
    "DraftSettlement",
    "DraftWorld",
    "ScenarioEffect",
]

#: PSP fee, in basis points of gross. 2%.
FEE_BPS = 200
#: Tax on the fee, in basis points. 18% GST.
TAX_BPS = 1_800


def _bps(amount_minor: int, bps: int) -> int:
    """Integer basis-point share, rounded up. No float in the money path."""
    return -(-amount_minor * bps // 10_000)


@dataclass
class DraftOrder:
    order_id: str
    merchant_id: str
    customer_ref: str
    amount_minor: int
    booked_at: datetime
    status: OrderStatus = OrderStatus.CAPTURED


@dataclass
class DraftPayment:
    payment_id: str
    order_id: str
    settlement_id: str
    amount_minor: int
    captured_at: datetime
    order_ref_raw: str | None
    """What the PSP actually wrote. Deliberately corrupted for some payments:
    ``None``, a space-separated form, or a non-ASCII hyphen. Normalisation has to
    recover ``order_id`` from this, and cannot always."""


@dataclass
class DraftSettlement:
    settlement_id: str
    merchant_id: str
    utr: str
    settled_on: date
    payment_ids: list[str] = field(default_factory=list)
    adjustments_minor: int = 0
    """Signed. Refunds (A06) and chargebacks (A08) net off here."""
    net_mismatch_minor: int = 0
    """Deliberate inconsistency between declared and computed net -- anomaly A03.
    Kept separate from ``adjustments_minor`` so the generator can always
    reconstruct what the settlement *should* have said."""

    def gross_minor(self, payments: dict[str, DraftPayment]) -> int:
        return sum(payments[pid].amount_minor for pid in self.payment_ids)

    def fee_minor(self, payments: dict[str, DraftPayment]) -> int:
        return _bps(self.gross_minor(payments), FEE_BPS)

    def tax_minor(self, payments: dict[str, DraftPayment]) -> int:
        return _bps(self.fee_minor(payments), TAX_BPS)

    def computed_net_minor(self, payments: dict[str, DraftPayment]) -> int:
        """What the identity says the net should be."""
        return (
            self.gross_minor(payments)
            - self.fee_minor(payments)
            - self.tax_minor(payments)
            + self.adjustments_minor
        )

    def declared_net_minor(self, payments: dict[str, DraftPayment]) -> int:
        """What the PSP file actually publishes. Differs from computed under A03."""
        return self.computed_net_minor(payments) + self.net_mismatch_minor


@dataclass
class DraftBankTxn:
    txn_id: str
    value_date: date
    narration: str
    credit_minor: int = 0
    debit_minor: int = 0
    balance_minor: int = 0
    settlement_id: str | None = None
    """``None`` for orphan credits (A10) and noise rows. Those are excluded from
    the conservation sum precisely because they belong to no settlement."""
    covered_payment_ids: list[str] = field(default_factory=list)
    """Which payments' money this credit carries. Empty for orphans and noise.
    Under A09 a settlement's payments are partitioned across two credits; under
    A08 the charged-back payment is absent from every credit, because its money
    never arrived."""


@dataclass(frozen=True)
class ScenarioEffect:
    """The generator's own record of what one scenario did.

    Ground truth is assembled from these, so nothing is ever back-inferred from
    the emitted files.
    """

    anomaly: AnomalyClass
    primary_ref: RecordRef
    expected_status: ExpectedStatus
    impact_minor: int
    note: str
    bank_delta_minor: int = 0
    """Signed amount by which this scenario deliberately breaks the identity
    ``sum(settlement-linked bank credits) == sum(declared nets)``. A02 contributes
    its drift; A05 contributes a whole duplicated credit; everything else is 0."""


@dataclass
class DraftWorld:
    """The world under construction."""

    orders: list[DraftOrder] = field(default_factory=list)
    payments: list[DraftPayment] = field(default_factory=list)
    settlements: list[DraftSettlement] = field(default_factory=list)
    bank_txns: list[DraftBankTxn] = field(default_factory=list)
    effects: list[ScenarioEffect] = field(default_factory=list)
    draws: dict[AnomalyClass, int] = field(default_factory=dict)

    claims: set[tuple[str, str]] = field(default_factory=set)
    """``(settlement_id, aspect)`` pairs already taken by a scenario.

    Claims are per *aspect*, not per settlement, because the anomaly classes
    fail along genuinely independent dimensions. A batch can credit the wrong
    amount **and** arrive late **and** lose its reference from the narration --
    those compose, and real reconciliation queues are full of items where two
    things went wrong at once.

    What does not compose is two scenarios fighting over the same dimension: a
    payout that is both split in two and duplicated has no single correct truth.
    Claiming per aspect blocks exactly those collisions while leaving the
    orthogonal combinations available, which is also what gives a small dev
    dataset enough room to contain every class at all."""

    claimed_payments: set[str] = field(default_factory=set)
    """Payments already consumed by a scenario, so no payment is both charged
    back and refunded."""

    def claim(self, settlement_id: str, aspect: str) -> bool:
        """Take a claim, returning False if it was already held."""
        key = (settlement_id, aspect)
        if key in self.claims:
            return False
        self.claims.add(key)
        return True

    def is_claimed(self, settlement_id: str, aspect: str) -> bool:
        return (settlement_id, aspect) in self.claims

    # -- indexes --
    #
    # Identity mappings (order->payment, id->object) are stable once the
    # baseline is built: scenarios mutate amounts, dates and narrations, and
    # they append bank rows, but they never re-parent a payment or renumber an
    # id. So these are built once rather than rebuilt per lookup.
    #
    # This matters at scale. Rebuilding a 5,000-entry dict inside a loop that
    # runs once per settlement per draw is the difference between a scale run
    # taking seconds and taking a minute.

    _payment_index: dict[str, DraftPayment] = field(default_factory=dict, repr=False)
    _settlement_index: dict[str, DraftSettlement] = field(default_factory=dict, repr=False)
    _order_index: dict[str, DraftOrder] = field(default_factory=dict, repr=False)
    _payment_by_order: dict[str, DraftPayment] = field(default_factory=dict, repr=False)
    _credit_index: dict[str, list[DraftBankTxn]] = field(default_factory=dict, repr=False)

    def reindex(self) -> None:
        """Rebuild every index. Called once after the baseline world is built."""
        self._payment_index = {payment.payment_id: payment for payment in self.payments}
        self._settlement_index = {s.settlement_id: s for s in self.settlements}
        self._order_index = {order.order_id: order for order in self.orders}
        self._payment_by_order = {payment.order_id: payment for payment in self.payments}
        self._credit_index = {}
        for txn in self.bank_txns:
            if txn.settlement_id is not None:
                self._credit_index.setdefault(txn.settlement_id, []).append(txn)

    def add_bank_txn(self, txn: DraftBankTxn) -> None:
        """Append a bank row, keeping the credit index consistent.

        Scenarios must use this rather than ``bank_txns.append`` -- a row added
        behind the index would be invisible to ``credits_for_settlement`` and
        the resulting truth would disagree with the emitted data.
        """
        self.bank_txns.append(txn)
        if txn.settlement_id is not None:
            self._credit_index.setdefault(txn.settlement_id, []).append(txn)

    def payments_by_id(self) -> dict[str, DraftPayment]:
        return self._payment_index

    def settlements_by_id(self) -> dict[str, DraftSettlement]:
        return self._settlement_index

    def orders_by_id(self) -> dict[str, DraftOrder]:
        return self._order_index

    def payment_for_order(self, order_id: str) -> DraftPayment | None:
        return self._payment_by_order.get(order_id)

    def credits_for_settlement(self, settlement_id: str) -> list[DraftBankTxn]:
        return self._credit_index.get(settlement_id, [])

    def record_draw(self, anomaly: AnomalyClass) -> None:
        self.draws[anomaly] = self.draws.get(anomaly, 0) + 1

    def settled_credit_total_minor(self) -> int:
        """Total bank credit attributable to settlements.

        Excludes orphans and noise, which belong to no settlement by definition.
        """
        return sum(txn.credit_minor for txn in self.bank_txns if txn.settlement_id is not None)

    def declared_net_total_minor(self) -> int:
        payments = self.payments_by_id()
        return sum(
            settlement.declared_net_minor(payments) for settlement in self.settlements
        )

    def declared_bank_delta_minor(self) -> int:
        return sum(effect.bank_delta_minor for effect in self.effects)
