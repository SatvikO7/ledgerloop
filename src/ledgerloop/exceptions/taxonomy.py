"""What the system concluded about an item it could not resolve.

PLAN.md 8.1 called the exception taxonomy a mirror of the anomaly taxonomy.
It is not, and ARCHITECTURE.md 6 decision 5 records why: the two vocabularies
answer different questions and the mapping between them is many-to-many. This
module is the half that answers *"what did the system conclude?"* and it never
reads an anomaly label, because it cannot -- ground truth is not an input to a
reconciliation run.

THE CLASSIFIER IS A CASCADE, AND THE ORDER IS THE ARGUMENT
-----------------------------------------------------------
Every rule below tests **arithmetic or structure that is present in the three
sources**. They are tried most-specific-first, and the first that fires wins:

* A settlement whose own declared identity does not close is a
  ``FEE_TAX_MISMATCH`` before it is anything else -- the discrepancy is stated
  by the source document, so no weaker explanation is needed.
* A negative adjustment equal to exactly one nested payment's gross is a
  ``CHARGEBACK_NETTED``: the arithmetic names the payment.
* Only when nothing specific fires does an item become ``UNKNOWN_RESIDUAL``,
  which is a **system state, not an anomaly** -- it means the ladder ran out of
  explanations, and it is reported as such rather than being smoothed into the
  nearest plausible class.

An unclassifiable item getting the nearest-looking label is the failure mode
this ordering exists to prevent: it would make the confusion matrix look good
while telling a controller the wrong thing to do next.

WHAT COUNTS AS AN ITEM
----------------------
Exceptions are raised over **records**, not over decisions. A decision is about
a *pair*; a controller's queue is about a payout that did not arrive or a credit
nobody can explain. Two kinds of item, and both come off the residual pool the
ladder left behind:

* an **open settlement** -- a payout the ladder could not credit;
* an **unclaimed credit** -- incoming money no settlement claimed;
* an **uncredited payment** -- a payment whose batch *was* credited while its
  own money was not.

Decisions supply evidence and hypotheses for those items; they do not create
them. That is what stops one contested settlement from producing four
exceptions because four candidates were routed.

A MATCHED RECORD IS NOT NECESSARILY A CLEAN ONE
------------------------------------------------
The third kind above, and one more rule, exist because "the ladder matched it"
and "there is nothing to tell a controller" are different statements. Two shapes
prove it, and both were found by measuring against ground truth *after* the
classifier was written:

* **A08 CHARGEBACK_NETTED.** T0 credits the batch correctly by excluding the
  charged-back payment -- that is the right match. But a payment whose money
  never arrived is still a problem, and it appears in no link, so nothing else
  would ever surface it.
* **A03 FEE_TAX_MISMATCH.** A settlement can be credited in full while its own
  declared identity does not close. The bank agrees with the net; the PSP's
  arithmetic does not agree with itself. The exception is about the document,
  not about the match.

In both cases the impact is the **discrepancy**, not the payout: the money that
did arrive is not at stake.

NO LLM REACHES THIS MODULE
--------------------------
Classification, severity and impact are deterministic functions of the sources.
Step 9's LLM may *rewrite the prose* on an exception that already exists, and
:attr:`ReconException.root_cause_source` records when it did. It may never
choose the class, the severity or the rupee figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from ledgerloop.config import MatchingTolerances, SeverityThresholds
from ledgerloop.matching.bank_leg import ClawBack, attribute_clawback
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.models.enums import ExceptionClass, OrderStatus, Severity
from ledgerloop.models.records import CanonicalBankTxn, CanonicalOrder, CanonicalPayment
from ledgerloop.money import sum_minor, tolerance_minor, within_tolerance

__all__ = [
    "AGENT_RESOLVABLE_CLASSES",
    "ClawbackItem",
    "CreditItem",
    "PaymentItem",
    "ResidualItem",
    "SettlementItem",
    "classify_credit",
    "classify_payment",
    "classify_settlement",
    "clawback_items",
    "residual_items",
    "severity_for",
]

#: The only classes a bounded rule may act on (PLAN.md 8.3). Everything else is
#: proposal-only, and ``UNMATCHABLE`` is refused by the model itself.
AGENT_RESOLVABLE_CLASSES: frozenset[ExceptionClass] = frozenset(
    {
        ExceptionClass.ROUNDING_DRIFT,
        ExceptionClass.TIMING_SHIFT,
        ExceptionClass.DUPLICATE_CREDIT,
    }
)


@dataclass(frozen=True)
class SettlementItem:
    """A payout the ladder could not credit, with what the sources say about it."""

    view: SettlementView
    keyed_credits: tuple[CanonicalBankTxn, ...] = ()
    near_amount_credits: tuple[CanonicalBankTxn, ...] = ()
    consumed_keyed_credits: tuple[CanonicalBankTxn, ...] = ()
    credited: bool = False
    """Whether the ladder credited this payout.

    ``True`` here means the item exists only because the settlement's own
    arithmetic does not close -- the money arrived, the document disagrees with
    itself, and the impact is the discrepancy rather than the payout.
    """

    @property
    def key(self) -> str:
        return self.view.settlement_id

    @property
    def clawback(self) -> ClawBack:
        return attribute_clawback(self.view)

    @property
    def impact_minor(self) -> int:
        """Money that did not arrive: the whole declared net."""
        return self.view.net_minor

    @property
    def as_of(self) -> date:
        return self.view.settlement.settled_on


@dataclass(frozen=True)
class CreditItem:
    """Incoming money no settlement claimed."""

    credit: CanonicalBankTxn
    keyed_settlements: tuple[SettlementView, ...] = ()
    twin_credits: tuple[CanonicalBankTxn, ...] = ()
    matched_twins: tuple[CanonicalBankTxn, ...] = ()
    """Twins the ladder credited. Empty when the reference is on two credits and
    the ladder refused both -- in which case *which* one is the duplicate is not
    established by anything, and the prose must not claim it is."""
    has_merchant_profile: bool = False

    reposting_of: CanonicalBankTxn | None = None
    """The payout this credit is a re-posting of, when the Phase 2.3 pass said so.

    A **separate** fact from :attr:`twin_credits`, which is keyed on the
    reference alone. The composed A05+A07 case is exactly the difference: two
    identical credits whose narrations both lost their UTR are twins of nothing,
    because there is no reference to twin on -- and before Phase 2.3 the queue
    reported the later one as an orphan, which is the wrong instruction to give
    a controller. The pass identifies it from the amount and the narration, and
    this field carries that finding into the classification.
    """

    @property
    def key(self) -> str:
        return self.credit.txn_id

    @property
    def impact_minor(self) -> int:
        return self.credit.credit_minor

    @property
    def as_of(self) -> date:
        return self.credit.value_date


@dataclass(frozen=True)
class PaymentItem:
    """A payment left uncredited by a batch that was otherwise credited."""

    payment: CanonicalPayment
    view: SettlementView
    clawed_back: bool = False

    @property
    def key(self) -> str:
        return self.payment.payment_id

    @property
    def impact_minor(self) -> int:
        return self.payment.amount_minor

    @property
    def as_of(self) -> date:
        return self.view.settlement.settled_on


@dataclass(frozen=True)
class ClawbackItem:
    """A negative adjustment that accounts for money from **outside** its batch.

    A06 ``POST_SETTLEMENT_REFUND`` is the shape this exists for, and Phase 2
    is why it exists now. An order is refunded *after* its payout has already
    left, so the refund cannot be netted off the batch it belonged to -- it is
    clawed back from a **later** batch instead, whose adjustments are short by
    exactly the refunded amount.

    Nothing else in the queue reaches it. The later batch reconciles perfectly
    (its credit equals its declared net, adjustment included), so it raises no
    exception; the earlier batch was paid in full, so it raises none either; and
    the refunded order appears in no unresolved link. Before Phase 2 those
    orders were reported only **by accident** -- they happened to sit inside a
    batch that some other anomaly had left contested, and the contested
    settlement's evidence chain named them. Removing that accident (the
    duplicate-posting pass matched those batches) is what made the gap visible,
    and the gap was always there.

    THE ATTRIBUTION IS ARITHMETIC, NOT A GUESS
    ------------------------------------------
    Two source facts have to agree before anything is said:

    1. this settlement's ``adjustments_minor`` is negative and matches **no**
       payment nested in it -- so the money it accounts for is not this batch's;
    2. the ledger holds an order marked ``REFUNDED`` whose payment is worth
       exactly that amount and was paid out in an *earlier* batch.

    Where exactly one order satisfies (2) the claw-back is attributed to it and
    the order is the exception's subject. Where several do, the amount cannot
    name one of them and the **settlement** is the subject instead, with every
    candidate named in the chain -- the same refusal-to-guess the tiers apply,
    at the queue's own granularity.
    """

    view: SettlementView
    """The settlement whose adjustments carry the claw-back."""

    amount_minor: int
    refunded_orders: tuple[CanonicalOrder, ...] = ()
    source_settlements: tuple[SettlementView, ...] = ()
    """The batches the refunded orders were originally paid out in."""

    @property
    def attributed(self) -> CanonicalOrder | None:
        """The one refunded order this claw-back names, or ``None``."""
        return self.refunded_orders[0] if len(self.refunded_orders) == 1 else None

    @property
    def key(self) -> str:
        attributed = self.attributed
        return attributed.order_id if attributed is not None else self.view.settlement_id

    @property
    def impact_minor(self) -> int:
        """The refund. Not the payout -- the rest of the batch arrived."""
        return self.amount_minor

    @property
    def as_of(self) -> date:
        return self.view.settlement.settled_on


#: Any kind of residual item.
ResidualItem = SettlementItem | CreditItem | PaymentItem


@dataclass
class _Residual:
    settlements: list[SettlementItem] = field(default_factory=list)
    credits: list[CreditItem] = field(default_factory=list)


def residual_items(
    context: MatchContext,
    *,
    matched_settlements: frozenset[str],
    matched_credits: frozenset[str],
    matched_payments: frozenset[str] = frozenset(),
    merchant_profiles: frozenset[str] = frozenset(),
    tolerances: MatchingTolerances | None = None,
) -> tuple[tuple[SettlementItem, ...], tuple[CreditItem, ...], tuple[PaymentItem, ...]]:
    """Everything the ladder left unexplained, with its supporting context.

    ``matched_*`` are what the run actually **auto-matched** -- not what it
    consumed. A settlement the pool consumed because two credits contested it
    was *decided*, but it was not matched, and it is exactly the item a
    controller has to look at. Reading consumption instead would make every
    contested payout disappear from the queue.

    Debits never become items. Outgoing money is not a payout this system
    reconciles, and a debit carrying a settlement's UTR is a source quirk the
    matcher already refuses -- ``ARCHITECTURE.md`` 6, 19. The count of them is
    reported separately rather than being quietly absorbed.
    """
    bands = tolerances or MatchingTolerances()
    settlements: list[SettlementItem] = []
    payments: list[PaymentItem] = []

    for view in context.settlements:
        credited = view.settlement_id in matched_settlements
        clawback = attribute_clawback(view)
        if credited:
            # The money arrived. Two things can still be wrong with it, and both
            # are properties of the PSP's own document rather than of the match.
            if view.settlement.net_delta_minor != 0:
                settlements.append(SettlementItem(view=view, credited=True))
            for payment in view.payments:
                if payment.payment_id in matched_payments:
                    continue
                payments.append(
                    PaymentItem(
                        payment=payment,
                        view=view,
                        clawed_back=(
                            clawback.excluded is not None
                            and clawback.excluded.payment_id == payment.payment_id
                        ),
                    )
                )
            continue

        keyed = context.credits_by_utr.get(view.utr, ()) if view.utr else ()
        band = tolerance_minor(
            view.net_minor,
            floor_minor=bands.amount_floor_minor,
            bps=bands.amount_bps,
        )
        near = tuple(
            txn
            for txn in context.credits
            if txn.txn_id not in matched_credits
            and within_tolerance(txn.credit_minor, view.net_minor, band)
        )
        settlements.append(
            SettlementItem(
                view=view,
                keyed_credits=tuple(
                    txn for txn in keyed if txn.txn_id not in matched_credits
                ),
                near_amount_credits=near,
                consumed_keyed_credits=tuple(
                    txn for txn in keyed if txn.txn_id in matched_credits
                ),
            )
        )

    credits: list[CreditItem] = []
    for txn in context.credits:
        if txn.txn_id in matched_credits:
            continue
        named_settlements = (
            context.settlements_by_utr.get(txn.extracted_utr, ())
            if txn.extracted_utr
            else ()
        )
        twins = tuple(
            other
            for other in context.credits_by_utr.get(txn.extracted_utr or "", ())
            if other.txn_id != txn.txn_id
        )
        group = context.duplicates.group_for(txn.txn_id)
        reposting_of = (
            group.original
            if group is not None and txn.txn_id != group.original.txn_id
            else None
        )
        credits.append(
            CreditItem(
                credit=txn,
                keyed_settlements=named_settlements,
                twin_credits=twins,
                matched_twins=tuple(
                    other for other in twins if other.txn_id in matched_credits
                ),
                has_merchant_profile=(
                    txn.extracted_merchant is not None
                    and txn.extracted_merchant in merchant_profiles
                ),
                reposting_of=reposting_of,
            )
        )
    return tuple(settlements), tuple(credits), tuple(payments)


def _refund_shaped(item: SettlementItem) -> CanonicalBankTxn | None:
    """A keyed credit short of the declared net by a *positive* amount.

    A06 ``POST_SETTLEMENT_REFUND`` nets a refund off a payout that has already
    been declared, so the money that arrives is less than the document says. The
    shortfall is evidence of a refund; an *excess* is not, and is left to fall
    through to a weaker rule rather than being called a refund with a negative
    sign.
    """
    for credit in item.keyed_credits:
        if item.view.net_minor - credit.credit_minor > 0:
            return credit
    return None


def classify_settlement(
    item: SettlementItem, *, date_window_days: int, ambiguous: bool = False
) -> ExceptionClass:
    """Which class explains an uncredited payout. Deterministic, sources only.

    ``ambiguous`` is passed in rather than re-derived: T2 already established
    that two subsets fit, and re-running the search here could reach a different
    conclusion from the tier that actually declined -- which would put one
    verdict in the queue and another in the decision log.
    """
    if ambiguous:
        return ExceptionClass.AMBIGUOUS_AGGREGATION

    settlement = item.view.settlement
    clawback = item.clawback

    # A payout that was credited is only here because its own identity does not
    # close. Nothing about the bank statement is in question, so none of the
    # rules below -- all of which reason about a missing credit -- applies.
    if item.credited:
        return ExceptionClass.FEE_TAX_MISMATCH

    # 1. The document does not close on its own terms. A03 breaks the identity
    #    net = gross - fee - tax + adjustments on purpose, and the settlement
    #    itself is the evidence -- no bank row needed.
    if settlement.net_delta_minor != 0:
        return ExceptionClass.FEE_TAX_MISMATCH

    # 2. A negative adjustment equal to exactly one nested payment's gross. The
    #    arithmetic names the payment that was clawed back.
    if clawback.present and clawback.excluded is not None:
        return ExceptionClass.CHARGEBACK_NETTED

    # 3. A negative adjustment matching no payment of this batch is a refund
    #    netted off from somewhere else (A06's ordinary shape).
    if clawback.present:
        return ExceptionClass.POST_SETTLEMENT_REFUND

    # 4. The key is on credits, but they do not add up to the payout.
    if item.keyed_credits:
        short = _refund_shaped(item)
        if len(item.keyed_credits) > 1:
            return ExceptionClass.SPLIT_PAYOUT_INCOMPLETE
        if short is not None:
            return ExceptionClass.POST_SETTLEMENT_REFUND
        gap = abs((item.keyed_credits[0].value_date - settlement.settled_on).days)
        if gap > date_window_days:
            return ExceptionClass.TIMING_SHIFT
        return ExceptionClass.UNKNOWN_RESIDUAL

    # 5. No keyed credit, but an unclaimed credit carries the right amount. The
    #    reference is what is missing, not the money.
    if item.near_amount_credits:
        outside = [
            txn
            for txn in item.near_amount_credits
            if abs((txn.value_date - settlement.settled_on).days) > date_window_days
        ]
        if len(outside) == len(item.near_amount_credits):
            return ExceptionClass.TIMING_SHIFT
        return ExceptionClass.MISSING_REFERENCE

    # 6. Nothing arrived at all. A payout whose money is not in the statement is
    #    either still in flight or genuinely absent, and the sources cannot tell
    #    those apart -- so it is named as the system state it is.
    return ExceptionClass.LATE_ARRIVAL


def classify_credit(item: CreditItem) -> ExceptionClass:
    """Which class explains an unclaimed credit. Deterministic, sources only."""
    credit = item.credit

    # 0. The duplicate-posting pass already identified this row as a re-posting
    #    of an earlier identical credit. That is the strongest statement
    #    available about it and it does not need a reference to hold, which is
    #    why it is tried before the keyed rule below: the composed A05+A07 case
    #    has no reference to key on, and used to land at rule 5 as an orphan.
    if item.reposting_of is not None:
        return ExceptionClass.DUPLICATE_CREDIT

    # 1. The same reference on more than one credit **for the same amount**.
    #    A05: the money arrived twice and only one arrival is real.
    #
    #    The amount test is what separates A05 from A09. Two credits sharing a
    #    reference are duplicates only if they are *copies*; where they carry
    #    different amounts they are tranches of one payout that was split, and
    #    calling those "the same payout twice" tells a controller to chase a
    #    reversal that does not exist. Found in Phase 2.5 on the one settlement
    #    the ladder still refuses -- its two tranches differ by Rs 52,543.57 and
    #    were both being reported as duplicates of each other.
    if item.keyed_settlements and item.twin_credits:
        if any(other.credit_minor == credit.credit_minor for other in item.twin_credits):
            return ExceptionClass.DUPLICATE_CREDIT
        return ExceptionClass.SPLIT_PAYOUT_INCOMPLETE

    # 2. A reference that names a settlement nobody credited. The pair exists in
    #    the sources; something about the amount or the date stopped the match.
    if item.keyed_settlements:
        return ExceptionClass.UNKNOWN_RESIDUAL

    # 3. A reference naming nothing at all is money from outside this ledger.
    if credit.extracted_utr is not None:
        return ExceptionClass.ORPHAN_BANK_CREDIT

    # 4. No reference, but a merchant the statement has used elsewhere: the
    #    narration lost its UTR (A07) and the name is what survived.
    if item.has_merchant_profile:
        return ExceptionClass.MISSING_REFERENCE

    # 5. No reference and no merchant this corpus has ever seen. Nothing in the
    #    three sources can relate it, which is the definition of the floor --
    #    not a failure to try.
    if credit.extracted_merchant is None:
        return ExceptionClass.UNMATCHABLE

    return ExceptionClass.ORPHAN_BANK_CREDIT


def classify_payment(item: PaymentItem) -> ExceptionClass:
    """Why a payment inside a credited batch carries no money of its own.

    One rule and one honest fallback. A negative adjustment equal to exactly
    this payment's gross is a claw-back and the arithmetic says so; anything
    else is a payment the ladder credited its batch without accounting for,
    which is a gap in the system rather than a named anomaly.
    """
    if item.clawed_back:
        return ExceptionClass.CHARGEBACK_NETTED
    return ExceptionClass.UNKNOWN_RESIDUAL


def severity_for(
    impact_minor: int,
    *,
    age_days: int,
    thresholds: SeverityThresholds,
) -> Severity:
    """Severity from rupee impact, escalated by age (PLAN.md 8.1).

    Money sets the band and age can raise it by one step, never more and never
    down. An old ₹12 drift is still a ₹12 drift; a fortnight-old ₹4 lakh payout
    is a different conversation from yesterday's.

    Age is measured against the dataset's own latest date, not the wall clock,
    so two runs over the same data produce the same queue -- the report carries
    no timestamp anywhere and this must not be the exception.
    """
    if impact_minor >= thresholds.critical_minor:
        base = Severity.CRITICAL
    elif impact_minor >= thresholds.high_minor:
        base = Severity.HIGH
    elif impact_minor >= thresholds.medium_minor:
        base = Severity.MEDIUM
    else:
        base = Severity.LOW

    if age_days < thresholds.escalate_after_days:
        return base
    ladder = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    return ladder[min(ladder.index(base) + 1, len(ladder) - 1)]


def payment_gross_minor(payments: Sequence[CanonicalPayment], *, field_name: str) -> int:
    """Sum of a payment set's gross, through the money gate."""
    return sum_minor((payment.amount_minor for payment in payments), field=field_name)


def clawback_items(
    context: MatchContext,
    *,
    matched_settlements: frozenset[str],
) -> tuple[ClawbackItem, ...]:
    """Claw-backs whose money belongs to a batch other than the one carrying them.

    Runs over **every** settlement, matched or not, because that is precisely
    the case the rest of the queue cannot see: a batch that reconciled to the
    paise can still be the place somebody else's refund was taken from. See
    :class:`ClawbackItem` for the two source facts that have to agree.

    ``matched_settlements`` is not used to decide whether to report -- a
    claw-back is a fact about the document -- but it is what tells the pair
    apart in the evidence: the *source* batch is named only when the ladder
    credited it, so the chain never asserts a payout the run did not make.

    Ordering follows source order over the carrying settlement, so the queue is
    reproducible.
    """
    refunded = _refunded_orders_by_amount(context)
    settlement_of_order = _settlement_by_order(context)

    items: list[ClawbackItem] = []
    for view in context.settlements:
        clawback = attribute_clawback(view)
        if not clawback.present or clawback.excluded is not None:
            continue
        if not clawback.attributable:
            # Several nested payments share the amount: the claw-back is about
            # this batch after all, and `classify_settlement` reports it as the
            # ambiguity it is. Not this function's item.
            continue

        candidates = tuple(
            order
            for order in refunded.get(clawback.amount_minor, ())
            # Refunded *after* its own payout, so the batch it belonged to was
            # settled no later than the one absorbing the claw-back -- and never
            # the same batch, which would be an ordinary in-batch chargeback.
            if (origin := settlement_of_order.get(order.order_id)) is not None
            and origin.settlement_id != view.settlement_id
            and origin.settlement.settled_on <= view.settlement.settled_on
        )
        if not candidates:
            continue

        origins = tuple(
            origin
            for order in candidates
            if (origin := settlement_of_order.get(order.order_id)) is not None
            and origin.settlement_id in matched_settlements
        )
        items.append(
            ClawbackItem(
                view=view,
                amount_minor=clawback.amount_minor,
                refunded_orders=candidates,
                source_settlements=origins,
            )
        )
    return tuple(items)


def _refunded_orders_by_amount(
    context: MatchContext,
) -> dict[int, tuple[CanonicalOrder, ...]]:
    """Ledger orders marked ``REFUNDED``, indexed by their amount.

    The ledger's own status field, read and not inferred. An order the system of
    record calls refunded is the only kind of order whose money is expected to
    come back, so it is the only kind a negative adjustment can be about.
    """
    grouped: dict[int, list[CanonicalOrder]] = {}
    for order in context.orders:
        if order.status is OrderStatus.REFUNDED:
            grouped.setdefault(order.amount_minor, []).append(order)
    return {amount: tuple(orders) for amount, orders in grouped.items()}


def _settlement_by_order(context: MatchContext) -> dict[str, SettlementView]:
    """Which batch each order was paid out in, via its payment.

    Orders whose payment reference did not normalise are absent, which is the
    honest outcome: an order the PSP could not point at is not one this rule can
    claim to have located.
    """
    located: dict[str, SettlementView] = {}
    for view in context.settlements:
        for payment in view.payments:
            if payment.order_ref_normalized is not None:
                located[payment.order_ref_normalized] = view
    return located
