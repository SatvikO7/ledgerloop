"""Phase 2 -- the eleven anomaly classes.

One function per class. Each takes the world and the order whose scenario draw
selected it, mutates the world, and appends a
:class:`~ledgerloop.generator.world.ScenarioEffect` describing exactly what it
did. Ground truth is built from those effects, so nothing is ever inferred back
out of the emitted files.

Scenarios claim one **aspect** of a settlement -- amount, structure, date or
narration -- rather than the whole settlement. Orthogonal anomalies therefore
compose: a batch can credit the wrong amount *and* arrive late *and* lose its
reference, which is what real reconciliation queues actually look like. Two
scenarios contending for the same aspect do not compose (a payout that is both
split and duplicated has no single correct truth), so the second one declines
and reports that it did.

A scenario that cannot place itself returns ``False``. The draw is still
counted: prevalence describes what was *drawn*, the effect list describes what
*happened*, and keeping those two numbers apart is what makes both honest.

A11 ``FX_MULTICURRENCY`` is absent: it is cut from the MVP.
"""

from __future__ import annotations

import random
from datetime import timedelta

from ledgerloop.generator.vocab import MERCHANTS, NARRATION_WITHOUT_UTR
from ledgerloop.generator.world import (
    DraftBankTxn,
    DraftPayment,
    DraftSettlement,
    DraftWorld,
    ScenarioEffect,
)
from ledgerloop.models.enums import AnomalyClass, ExpectedStatus, OrderStatus
from ledgerloop.models.refs import bank_ref, order_ref, payment_ref, settlement_ref
from ledgerloop.money import allocate_minor

__all__ = ["SCENARIOS", "apply_scenario"]


def _next_bank_id(world: DraftWorld) -> str:
    """Allocate the next bank id from the current high-water mark.

    Derived from existing ids rather than a counter, so injected rows number
    correctly no matter what order the scenarios ran in.
    """
    highest = max((int(txn.txn_id.split("-")[1]) for txn in world.bank_txns), default=0)
    return f"BNK-{highest + 1:05d}"


def _merchant_variant(rng: random.Random, merchant_id: str) -> str:
    merchant = next(m for m in MERCHANTS if m.merchant_id == merchant_id)
    return merchant.variants[rng.randrange(len(merchant.variants))]


def _settlement_of(
    world: DraftWorld, order_id: str
) -> tuple[DraftPayment | None, DraftSettlement | None]:
    """The payment placed for an order, and the batch it settled in."""
    payment = world.payment_for_order(order_id)
    if payment is None:
        return None, None
    return payment, world.settlements_by_id().get(payment.settlement_id)


#: The four independent dimensions along which a settlement can go wrong.
#: Scenarios claim one each, so orthogonal anomalies compose and conflicting
#: ones do not. See ``DraftWorld.claims``.
ASPECT_AMOUNT = "amount"
ASPECT_STRUCTURE = "structure"
ASPECT_DATE = "date"
ASPECT_NARRATION = "narration"

#: A06 is the one scenario whose money lands on a *different* settlement than
#: the one it originates from. The source batch is claimed under its own aspect
#: -- it only loses a payment, its credit is untouched -- while the ``amount``
#: claim goes to the target batch, which is where the credit actually changes.
#: Claiming ``amount`` on the source instead would leave the target unprotected
#: and let a claw-back silently overwrite an A02 drift.
ASPECT_REFUND_SOURCE = "refund_source"

#: Claimed by every scenario whose ``primary_ref`` is the settlement itself or
#: its existing credit -- A02, A03, A04, A07, A12.
#:
#: A :class:`GroundTruthRecord` carries exactly one ``anomaly_class``, so two
#: effects naming the same primary record would silently overwrite each other's
#: label and one anomaly class would vanish from the truth set. Per-class recall
#: would then be measured against a corpus that claims the class is not there.
#:
#: A05, A09 and A10 do not need this: each creates a *new* bank row, so their
#: primaries are unique by construction. A06 and A08 name an order or payment
#: they have already claimed exclusively.
ASPECT_PRIMARY = "primary_label"


def _claim_settlement(
    world: DraftWorld,
    order_id: str,
    aspect: str,
    *,
    min_payments: int = 1,
    needs_later_sibling: bool = False,
    require_single_credit: bool = True,
    claims_primary_label: bool = False,
) -> DraftSettlement | None:
    """Find and claim a settlement this scenario can safely mutate.

    The drawn order's own settlement is preferred; if it is already claimed or
    fails the scenario's preconditions, the search falls through to any other
    eligible settlement.

    **The draw selects *what* anomaly to inject, not *where*.** Binding a
    scenario to its drawn order would make coverage a lottery: with ~10 payments
    per batch, most settlement-level draws would land on an already-mutated
    settlement and decline, and the dataset would be missing whole anomaly
    classes at dev size. Placement is a generator concern; the effect records
    exactly where it landed, so ground truth stays precise either way.
    """
    _, own = _settlement_of(world, order_id)
    ordered = ([own] if own is not None else []) + [
        candidate
        for candidate in world.settlements
        if own is None or candidate.settlement_id != own.settlement_id
    ]

    for candidate in ordered:
        if world.is_claimed(candidate.settlement_id, aspect):
            continue
        if claims_primary_label and world.is_claimed(candidate.settlement_id, ASPECT_PRIMARY):
            continue
        if len(candidate.payment_ids) < min_payments:
            continue
        credits = world.credits_for_settlement(candidate.settlement_id)
        if require_single_credit and len(credits) != 1:
            continue
        if not credits:
            continue
        if needs_later_sibling and not _later_sibling(world, candidate):
            continue
        world.claim(candidate.settlement_id, aspect)
        if claims_primary_label:
            world.claim(candidate.settlement_id, ASPECT_PRIMARY)
        return candidate
    return None


def _later_sibling(world: DraftWorld, settlement: DraftSettlement) -> DraftSettlement | None:
    """The same merchant's next batch, which is where a claw-back lands.

    Excludes batches whose ``amount`` aspect is already claimed: the claw-back
    rewrites the target's credit, so landing on an already-adjusted batch would
    overwrite the earlier scenario's money and break conservation.
    """
    later = [
        candidate
        for candidate in world.settlements
        if candidate.merchant_id == settlement.merchant_id
        and candidate.settled_on > settlement.settled_on
        and len(world.credits_for_settlement(candidate.settlement_id)) == 1
        and not world.is_claimed(candidate.settlement_id, ASPECT_AMOUNT)
    ]
    if not later:
        return None
    return min(later, key=lambda s: s.settled_on)


def _claim_payment(world: DraftWorld, settlement: DraftSettlement) -> str | None:
    """Take an unclaimed payment from a settlement, leaving at least one behind.

    A batch stripped of every payment would produce an empty credit, which is a
    degenerate case rather than a modelled anomaly.
    """
    available = [pid for pid in settlement.payment_ids if pid not in world.claimed_payments]
    if len(available) < 1 or len(settlement.payment_ids) < 2:
        return None
    chosen = available[0]
    world.claimed_payments.add(chosen)
    return chosen


# ----------------------------------------------------------------------
# A01
# ----------------------------------------------------------------------


def clean(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """Nothing happens. Recorded as a draw, produces no effect."""
    del world, rng, order_id
    return True


# ----------------------------------------------------------------------
# A02 -- ±1 to 3 paise from FX or fee rounding
# ----------------------------------------------------------------------


def rounding_drift(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The bank credits a few paise more or less than the declared net.

    Still matchable: the drift is far inside the T1 tolerance floor of ₹1. This
    class exists to prove the system tolerates noise without either rejecting
    good matches or silently absorbing real discrepancies.
    """
    settlement = _claim_settlement(
        world, order_id, ASPECT_AMOUNT, claims_primary_label=True
    )
    if settlement is None:
        return False

    drift = rng.choice([-3, -2, -1, 1, 2, 3])
    credit = world.credits_for_settlement(settlement.settlement_id)[0]
    credit.credit_minor += drift

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.ROUNDING_DRIFT,
            primary_ref=bank_ref(credit.txn_id),
            expected_status=ExpectedStatus.MATCHED,
            impact_minor=abs(drift),
            bank_delta_minor=drift,
            note=(
                f"bank credited {drift:+d} paise against declared net "
                f"on {settlement.settlement_id}"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A03 -- net != gross - fee - tax + adjustments
# ----------------------------------------------------------------------


def fee_tax_mismatch(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The settlement's own arithmetic does not add up.

    The bank pays the *declared* net, so bank-to-settlement still reconciles;
    the inconsistency is internal to the PSP report. That is what makes this
    class interesting -- it is invisible to any two-way check and only the
    three-way structure surfaces it.
    """
    settlement = _claim_settlement(
        world, order_id, ASPECT_AMOUNT, claims_primary_label=True
    )
    if settlement is None:
        return False

    mismatch = rng.choice([-1, 1]) * rng.randrange(500, 25_000, 100)
    settlement.net_mismatch_minor = mismatch

    # The credit must follow the declared net, or this would masquerade as A02.
    payments = world.payments_by_id()
    for credit in world.credits_for_settlement(settlement.settlement_id):
        credit.credit_minor = settlement.declared_net_minor(payments)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.FEE_TAX_MISMATCH,
            primary_ref=settlement_ref(settlement.settlement_id),
            expected_status=ExpectedStatus.EXCEPTION,
            impact_minor=abs(mismatch),
            note=(
                f"declared net differs from gross - fee - tax + adjustments by "
                f"{mismatch:+d} paise on {settlement.settlement_id}"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A04 -- credit lands T+2 or T+3 instead of T+1
# ----------------------------------------------------------------------


def timing_shift(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The money arrives late but intact. Inside the ±3 day T1 window."""
    settlement = _claim_settlement(
        world,
        order_id,
        ASPECT_DATE,
        require_single_credit=False,
        claims_primary_label=True,
    )
    if settlement is None:
        return False
    credits = world.credits_for_settlement(settlement.settlement_id)

    extra_days = rng.choice([1, 2])
    for credit in credits:
        credit.value_date += timedelta(days=extra_days)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.TIMING_SHIFT,
            primary_ref=bank_ref(credits[0].txn_id),
            expected_status=ExpectedStatus.MATCHED,
            impact_minor=0,
            note=f"credit landed T+{1 + extra_days} instead of T+1 for {settlement.settlement_id}",
        )
    )
    return True


# ----------------------------------------------------------------------
# A05 -- the same UTR credited twice
# ----------------------------------------------------------------------


def duplicate_credit(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """A second, identical credit appears. It must match nothing.

    The duplicate carries ``settlement_id`` -- it genuinely purports to pay that
    batch, and the bank really did credit the money twice -- but
    ``covered_payment_ids`` is empty, so it gets **no** ground-truth link. A
    system that links it has made a false positive. The doubled money is
    declared as a ``bank_delta_minor``, so the conservation law records "the
    bank paid this settlement twice" rather than quietly balancing.
    """
    del rng
    settlement = _claim_settlement(world, order_id, ASPECT_STRUCTURE)
    if settlement is None:
        return False

    original = world.credits_for_settlement(settlement.settlement_id)[0]
    duplicate = DraftBankTxn(
        txn_id=_next_bank_id(world),
        value_date=original.value_date + timedelta(days=1),
        narration=original.narration,
        credit_minor=original.credit_minor,
        settlement_id=settlement.settlement_id,
        covered_payment_ids=[],
    )
    world.add_bank_txn(duplicate)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.DUPLICATE_CREDIT,
            primary_ref=bank_ref(duplicate.txn_id),
            expected_status=ExpectedStatus.EXCEPTION,
            impact_minor=duplicate.credit_minor,
            bank_delta_minor=duplicate.credit_minor,
            note=(
                f"{duplicate.txn_id} repeats UTR {settlement.utr} already credited as "
                f"{original.txn_id}"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A06 -- refund issued after payout, netted into a later batch
# ----------------------------------------------------------------------


def post_settlement_refund(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The order is refunded after its payout, and claws back from a later batch.

    This is why the reconciliation is a path problem rather than a join: the
    money left in a batch the refunded order has nothing to do with.
    """
    del rng
    settlement = _claim_settlement(
        world,
        order_id,
        ASPECT_REFUND_SOURCE,
        min_payments=2,
        needs_later_sibling=True,
        require_single_credit=False,
    )
    if settlement is None:
        return False
    target = _later_sibling(world, settlement)
    if target is None:  # pragma: no cover - guaranteed by needs_later_sibling
        return False

    payment_id = _claim_payment(world, settlement)
    if payment_id is None:
        return False

    payments = world.payments_by_id()
    refund = payments[payment_id].amount_minor
    # Decline rather than emit a negative payout: a batch that cannot absorb the
    # claw-back is a different anomaly than the one being modelled.
    if target.declared_net_minor(payments) - refund <= 0:
        return False

    # The credit that changes is the TARGET's, so that is what gets claimed.
    world.claim(target.settlement_id, ASPECT_AMOUNT)
    refunded_order_id = payments[payment_id].order_id
    world.orders_by_id()[refunded_order_id].status = OrderStatus.REFUNDED
    target.adjustments_minor -= refund

    for credit in world.credits_for_settlement(target.settlement_id):
        credit.credit_minor = target.declared_net_minor(payments)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.POST_SETTLEMENT_REFUND,
            primary_ref=order_ref(refunded_order_id),
            expected_status=ExpectedStatus.EXCEPTION,
            impact_minor=refund,
            note=(
                f"{refunded_order_id} refunded after payout {settlement.settlement_id}; "
                f"clawed back inside adjustments on {target.settlement_id}"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A07 -- no reference anywhere in the narration
# ----------------------------------------------------------------------


def missing_reference(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The UTR is stripped from the narration; only a merchant variant remains.

    Still resolvable -- through T3 on the name plus the amount -- which is the
    point. It is the class that justifies fuzzy matching existing at all.
    """
    settlement = _claim_settlement(
        world,
        order_id,
        ASPECT_NARRATION,
        require_single_credit=False,
        claims_primary_label=True,
    )
    if settlement is None:
        return False

    credit = world.credits_for_settlement(settlement.settlement_id)[0]
    template = NARRATION_WITHOUT_UTR[rng.randrange(len(NARRATION_WITHOUT_UTR))]
    credit.narration = template.format(
        variant=_merchant_variant(rng, settlement.merchant_id)
    )

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.MISSING_REFERENCE,
            primary_ref=bank_ref(credit.txn_id),
            expected_status=ExpectedStatus.MATCHED,
            impact_minor=0,
            note=f"UTR {settlement.utr} absent from narration on {credit.txn_id}",
        )
    )
    return True


# ----------------------------------------------------------------------
# A08 -- chargeback debited inside adjustments
# ----------------------------------------------------------------------


def chargeback_netted(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """A payment is charged back and silently netted off its own payout.

    The payment sits in the settlement, but its money never reaches the bank, so
    it gets **no** ``PAYMENT_CREDITED_AS`` link. The batch total is short by
    exactly the chargeback, and nothing in the bank statement says why. This is
    the flagship exception -- "₹4,312 short because a chargeback was netted off
    SETL-0091" is the sentence the whole project exists to produce.
    """
    del rng
    settlement = _claim_settlement(world, order_id, ASPECT_AMOUNT, min_payments=2)
    if settlement is None:
        return False
    payment_id = _claim_payment(world, settlement)
    if payment_id is None:
        return False

    payments = world.payments_by_id()
    chargeback = payments[payment_id].amount_minor
    if settlement.declared_net_minor(payments) - chargeback <= 0:
        return False
    settlement.adjustments_minor -= chargeback

    credit = world.credits_for_settlement(settlement.settlement_id)[0]
    credit.covered_payment_ids = [
        pid for pid in credit.covered_payment_ids if pid != payment_id
    ]
    credit.credit_minor = settlement.declared_net_minor(payments)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.CHARGEBACK_NETTED,
            primary_ref=payment_ref(payment_id),
            expected_status=ExpectedStatus.EXCEPTION,
            impact_minor=chargeback,
            note=(
                f"chargeback of {chargeback} paise on {payment_id} netted off "
                f"{settlement.settlement_id}; money never reached the bank"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A09 -- one settlement arrives as two bank credits
# ----------------------------------------------------------------------


def split_payout(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The batch is paid out in two tranches.

    The payments are partitioned across the two credits and the net is allocated
    by the partition's sums, so the parts sum to exactly the whole. This is the
    class the plan's flat ground-truth row could not express.
    """
    settlement = _claim_settlement(world, order_id, ASPECT_STRUCTURE, min_payments=2)
    if settlement is None:
        return False
    original = world.credits_for_settlement(settlement.settlement_id)[0]
    if len(original.covered_payment_ids) < 2:
        return False

    cut = rng.randint(1, len(original.covered_payment_ids) - 1)
    first_ids = original.covered_payment_ids[:cut]
    second_ids = original.covered_payment_ids[cut:]

    payments = world.payments_by_id()
    weights = [
        sum(payments[pid].amount_minor for pid in first_ids),
        sum(payments[pid].amount_minor for pid in second_ids),
    ]
    first_amount, second_amount = allocate_minor(original.credit_minor, weights)

    original.credit_minor = first_amount
    original.covered_payment_ids = first_ids

    second = DraftBankTxn(
        txn_id=_next_bank_id(world),
        value_date=original.value_date + timedelta(days=rng.randint(1, 2)),
        narration=original.narration,
        credit_minor=second_amount,
        settlement_id=settlement.settlement_id,
        covered_payment_ids=second_ids,
    )
    world.add_bank_txn(second)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.SPLIT_PAYOUT,
            # The second credit is the row that would not otherwise exist, and it
            # is unique -- so this needs no primary-label claim.
            primary_ref=bank_ref(second.txn_id),
            expected_status=ExpectedStatus.MATCHED,
            impact_minor=0,
            note=(
                f"{settlement.settlement_id} paid out as {original.txn_id} + "
                f"{second.txn_id} ({first_amount} + {second_amount} paise)"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A10 -- a credit with no settlement behind it
# ----------------------------------------------------------------------


def orphan_bank_credit(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """A direct transfer lands with nothing in the PSP file to explain it.

    Marked ``UNMATCHABLE``, not ``EXCEPTION``. No system could resolve this
    without data that does not exist in the three sources, and reporting that
    ceiling honestly is the point of the distinction. These are excluded from
    the match-rate denominator.
    """
    del order_id
    txn = DraftBankTxn(
        txn_id=_next_bank_id(world),
        value_date=world.bank_txns[rng.randrange(len(world.bank_txns))].value_date,
        narration=f"NEFT CR-DIRECT TRANSFER-{rng.randint(100_000, 999_999)}-INWARD",
        credit_minor=rng.randrange(80_000, 6_000_000, 100),
        settlement_id=None,
        covered_payment_ids=[],
    )
    world.add_bank_txn(txn)

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.ORPHAN_BANK_CREDIT,
            primary_ref=bank_ref(txn.txn_id),
            expected_status=ExpectedStatus.UNMATCHABLE,
            impact_minor=txn.credit_minor,
            note=(
                f"{txn.txn_id} has no settlement in the PSP report; "
                "irreconcilable by construction"
            ),
        )
    )
    return True


# ----------------------------------------------------------------------
# A12 -- the settlement record appears a batch late
# ----------------------------------------------------------------------


def late_arrival(world: DraftWorld, rng: random.Random, order_id: str) -> bool:
    """The PSP reports the batch a week after its payments were captured.

    This pushes the payment-to-credit date gap outside the ±3 day T1 window, so
    the link has to be recovered by aggregation or graph inference instead. It
    is the class that stops date proximity from being sufficient.
    """
    settlement = _claim_settlement(
        world,
        order_id,
        ASPECT_DATE,
        require_single_credit=False,
        claims_primary_label=True,
    )
    if settlement is None:
        return False
    credits = world.credits_for_settlement(settlement.settlement_id)

    delay = timedelta(days=rng.randint(6, 9))
    settlement.settled_on += delay
    for credit in credits:
        credit.value_date += delay

    world.effects.append(
        ScenarioEffect(
            anomaly=AnomalyClass.LATE_ARRIVAL,
            primary_ref=settlement_ref(settlement.settlement_id),
            expected_status=ExpectedStatus.MATCHED,
            impact_minor=0,
            note=f"{settlement.settlement_id} reported {delay.days} days after capture",
        )
    )
    return True


#: Dispatch table. Keys cover every member of AnomalyClass, and a test asserts it.
SCENARIOS = {
    AnomalyClass.CLEAN: clean,
    AnomalyClass.ROUNDING_DRIFT: rounding_drift,
    AnomalyClass.FEE_TAX_MISMATCH: fee_tax_mismatch,
    AnomalyClass.TIMING_SHIFT: timing_shift,
    AnomalyClass.DUPLICATE_CREDIT: duplicate_credit,
    AnomalyClass.POST_SETTLEMENT_REFUND: post_settlement_refund,
    AnomalyClass.MISSING_REFERENCE: missing_reference,
    AnomalyClass.CHARGEBACK_NETTED: chargeback_netted,
    AnomalyClass.SPLIT_PAYOUT: split_payout,
    AnomalyClass.ORPHAN_BANK_CREDIT: orphan_bank_credit,
    AnomalyClass.LATE_ARRIVAL: late_arrival,
}


def apply_scenario(
    world: DraftWorld, rng: random.Random, anomaly: AnomalyClass, order_id: str
) -> bool:
    """Apply one drawn scenario. Returns whether it took effect.

    A scenario declines when its preconditions do not hold -- the settlement was
    already mutated by a conflicting class, or the batch is too small. The draw
    is still recorded, because prevalence describes what was *drawn*; the effect
    list describes what actually happened. Keeping those two numbers separate is
    what makes both of them honest.
    """
    return SCENARIOS[anomaly](world, rng, order_id)
