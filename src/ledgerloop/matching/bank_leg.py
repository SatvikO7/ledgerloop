"""Resolving the bank leg: which credit paid which settlement, and therefore
which payments each credit carries.

This is where the reconciliation is actually earned. The order and payment ends
of the chain are asserted by the sources -- the PSP nests its payments inside a
settlement and names the order each one paid -- but nothing in any file says
which bank row is which payout. That edge has to be inferred, and it is the
edge the evaluation unit hangs off: ``PAYMENT_CREDITED_AS`` is a
``(payment, bank_txn)`` pair, and the only way to reach it is through the
settlement.

T0 AND T1 ARE THE SAME ALGORITHM WITH DIFFERENT BANDS
-----------------------------------------------------
T0 is exact-key plus exact amount; T1 is exact-key plus a tolerance band and a
date window. Written as two modules they would be a hundred duplicated lines
whose behaviours could drift apart -- and the one place drift would show up is
T1 quietly accepting something T0 had already declined for a reason.

So there is one resolver, parameterised by a :class:`BankLegRule`, and **T0 is
the degenerate case with a zero-width band**: ``tolerance_minor(net, floor=0,
bps=0)`` is ``0``, and ``within_tolerance(a, b, 0)`` is exact equality. The two
tiers differ in their rule, not in their reasoning.

MUTUAL UNIQUENESS
-----------------
A match is safe only when it is unique **from both sides**. Counting the
credits that carry the settlement's key answers one half -- *is this settlement
explained by only one credit?* -- and it is the half a keyed join can see.

The other half is *is this credit explained by only one settlement, and is this
settlement's payout not also sitting somewhere else in the statement?* It has
to be asked separately, because the row that would answer it is precisely the
row whose reference went missing.

The case that forces it is real and was found by a property test rather than by
inspection. Anomalies compose: a settlement can be hit by A05
``DUPLICATE_CREDIT`` **and** A07 ``MISSING_REFERENCE``, and when the narration
rewrite lands on the original while the duplicate keeps its copied narration,
the *duplicate* is the only row carrying the key. A keyed uniqueness check sees
one clean contender and matches the entire batch to the duplicate -- every link
wrong, at ``p = 1.0``. Checking that no other unclaimed credit could be this
payout catches it, and costs a scan.

A KEY ON SEVERAL CREDITS MEANS THE PAYOUT WAS SPLIT
----------------------------------------------------
A09 ``SPLIT_PAYOUT`` delivers one settlement as two tranches, and the generator
copies the narration onto both -- so the settlement's key appears twice while
only the tranches' *sum* equals the net. Neither tranche is usually within
0.5% of the whole, so the amount test declines both and the batch falls through
to T2, which is correct.

It is not always so. A lopsided split -- 213,691 paise against 43,499,522 --
leaves the larger tranche **inside** a proportional band computed on the whole
net, because the missing tranche is smaller than 0.5% of it. The tier then
matches the entire batch to a tranche that carries only part of it, and every
payment the other tranche took is asserted wrongly. This too was found by a
property test rather than by inspection.

So a settlement whose key appears on more than one unclaimed credit is **not
resolvable as a whole batch**, however well one of them fits on amount. The
disposition matters: this is left **undecided** rather than contested, so the
settlement stays in the pool. The question a split raises is *how do the
payments partition*, and that is T2's subset arithmetic to answer -- consuming
the settlement here would take it away from the tier built to solve it. Only
where several credits each claim the *whole* net (A05) is the question
unanswerable by any later tier, and only then is the settlement consumed.

The key is not always there to see it. Anomalies compose again: A07 can strip
the reference from the tranche A09 left behind, and then the surviving tranche
is the only keyed row and its amount is inside the band. What remains is
arithmetic. **A tolerance band exists to absorb rounding drift, and a shortfall
that exactly equals another unclaimed credit is not drift** -- it is the rest of
the payout, sitting in the statement. That test is exact, needs no search, and
only ever *declines*: finding the partition stays T2's job, and this tier's job
is to refuse to over-assert before it.

THE THREE OUTCOMES, AND WHY CONTESTED IS NOT REJECTED
------------------------------------------------------
For each settlement still in the pool, the qualifying credits are counted:

* **none** -- no conclusion. The settlement stays in the pool and falls through
  to the next tier. This is how A02 ``ROUNDING_DRIFT`` (a credit three paise
  off the declared net) survives T0 and is caught by T1.
* **one** -- resolved. The settlement and the credit are consumed, and the
  credit is allocated across the settlement's payments.
* **more than one** -- contested. A settlement's net is paid **once**; two
  credits both claiming to be that payment in full is a contradiction, not a
  choice. Anomaly A05 ``DUPLICATE_CREDIT`` is exactly this, and its ground-truth
  verdict is ``EXCEPTION`` -- so the honest output is the contenders plus the
  reason, not a coin flip. A lone keyed contender is contested on the same
  grounds when an unkeyed credit could equally be the payout.

An exact-looking key is therefore **not** sufficient on its own. This is the
single largest behavioural difference from baseline B0, which credits the whole
batch to every row carrying the UTR and turns one duplicated credit into a full
batch of false positives.

CONTESTED EDGES ARE NOT EXPANDED
--------------------------------
A resolved settlement edge is expanded into one ``PAYMENT_CREDITED_AS``
candidate per payment. A contested one is not. The payment links are a
*consequence* of believing the settlement edge; manufacturing them for an edge
the system does not believe would assert payment-level links on evidence that
does not support them, and would multiply one ambiguous settlement into twenty
ambiguous links in the review queue.

THE PROBABILITY OF A CONTESTED PAIR
------------------------------------
``Tier.is_deterministic_certain`` says T0 and T1 bypass the blender at
``p = 1.0``. That certainty is about *which* link, and it holds only where the
key resolves uniquely. Where ``n`` contenders are indistinguishable under the
tier's own rule, the honest statement is a uniform prior: ``p = 1/n``. At
``n = 1`` this is exactly the 1.0 the design specifies, so the common path is
unchanged; at ``n = 2`` it is 0.5, which the *existing* thresholds route to an
exception without a single new constant being invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.records import CanonicalBankTxn, CanonicalPayment
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.money import (
    allocate_minor,
    delta_ratio,
    format_minor,
    sum_minor,
    tolerance_minor,
    within_tolerance,
)

__all__ = [
    "BankLegOutcome",
    "BankLegRule",
    "ClawBack",
    "allocated_share_minor",
    "attribute_clawback",
    "candidate_id",
    "resolve_bank_leg",
]


@dataclass(frozen=True)
class BankLegRule:
    """One tier's admission test for a settlement/credit pair.

    ``date_window_days`` is ``None`` for T0 on purpose. PLAN.md 6.1 gives T0 as
    "exact join, exact amount" and introduces the date window only at T1, which
    reads backwards until you see why: an exact UTR **and** an exact net to the
    paise is already overwhelming evidence, and adding a date constraint to it
    would reject A04 ``TIMING_SHIFT`` and A12 ``LATE_ARRIVAL`` -- money that
    arrived intact but late -- for no gain in precision. T1's amount test is
    looser, so T1 needs the date to carry the constraint T0 got from the amount.
    """

    tier: Tier
    amount_floor_minor: int
    amount_bps: int
    date_window_days: int | None

    def band_for(self, amount_minor: int) -> int:
        """The tolerance band this rule judges ``amount_minor`` against."""
        return tolerance_minor(
            amount_minor, floor_minor=self.amount_floor_minor, bps=self.amount_bps
        )


@dataclass(frozen=True)
class BankLegOutcome:
    """What one tier's pass over the pool produced."""

    tier: Tier
    candidates: tuple[MatchCandidate, ...]
    resolved_settlements: int = 0
    contested_settlements: int = 0
    undecided_settlements: int = 0
    resolved_credits: int = 0
    settlements_without_key: int = 0
    key_collisions: int = 0
    split_suspected: int = 0

    @property
    def settlement_links(self) -> int:
        return sum(
            1
            for candidate in self.candidates
            if candidate.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )

    @property
    def payment_links(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.is_evaluable)


_SEPARATOR: Final[str] = "|"


def candidate_id(tier: Tier, link_type: LinkType, source_key: str, target_key: str) -> str:
    """A stable, human-readable candidate identifier.

    Derived from the candidate's own content rather than from a counter, so two
    runs over the same data produce the same ids -- which is what lets a
    reproducibility test compare decision logs directly instead of comparing
    them modulo renumbering.
    """
    return _SEPARATOR.join((tier.name, link_type.value, source_key, target_key))


@dataclass(frozen=True)
class ClawBack:
    """A negative adjustment on a settlement, and the payment it accounts for.

    Anomaly A08 ``CHARGEBACK_NETTED`` is the reason this exists. A payment is
    charged back and silently netted off its own payout: it is still nested in
    the settlement, but **its money never reaches the bank**, so ground truth
    gives it no ``PAYMENT_CREDITED_AS`` link. A closure that credited every
    nested payment would assert one link too many on every affected batch.

    The claw-back is not inferred from a pattern -- it is read off the
    settlement's own declared ``adjustments_minor``, which is negative by
    exactly the charged-back amount. Where that amount equals exactly one
    payment's gross, the arithmetic identifies which payment did not arrive.

    Three outcomes, and the distinction matters:

    * **No claw-back** -- adjustments are zero or positive. Every payment
      travelled with the credit.
    * **Attributed** -- exactly one payment matches. That payment is excluded
      from the closure and named in the evidence.
    * **Unattributable** -- the amount matches several payments. Some payment
      did not arrive, so crediting all of them guarantees a false positive, and
      excluding an arbitrary one would be a guess dressed as a deduction. This
      is the only case that clears :attr:`attributable`, which drops
      ``arithmetic_verified`` and lets the decision policy demote the payment
      links to review rather than auto-match them on arithmetic that does not
      close.

    A claw-back matching **no** payment is the ordinary A06 case: a refund on an
    *earlier* batch, netted off this one. The adjustment belongs to a payment
    that was never in this settlement, so every payment here did arrive and the
    closure stays verified.
    """

    amount_minor: int = 0
    excluded: CanonicalPayment | None = None
    attributable: bool = True

    @property
    def present(self) -> bool:
        return self.amount_minor != 0


def attribute_clawback(view: SettlementView) -> ClawBack:
    """Identify the payment a negative adjustment netted off, if it can be."""
    adjustments = view.settlement.adjustments_minor
    if adjustments >= 0:
        return ClawBack()

    magnitude = -adjustments
    matches = [payment for payment in view.payments if payment.amount_minor == magnitude]
    if len(matches) == 1:
        return ClawBack(amount_minor=magnitude, excluded=matches[0], attributable=True)
    if not matches:
        return ClawBack(amount_minor=magnitude, excluded=None, attributable=True)
    return ClawBack(amount_minor=magnitude, excluded=None, attributable=False)


def allocated_share_minor(candidate: MatchCandidate) -> int:
    """Read back the share a ``PAYMENT_CREDITED_AS`` candidate was allocated.

    The share is written into the candidate's first ``ARITHMETIC_CHECK``
    evidence item by :func:`_payment_candidates`, and this is the sanctioned way
    to get it out. Reading it positionally, or taking whichever evidence item
    happens to carry an amount, would break the moment an item is inserted --
    and this value is what the evaluator charges false positives at, so a silent
    break would move a rupee figure rather than raise.

    The link type is checked first, and that check is the point. A
    ``SETTLEMENT_CREDITED_AS`` candidate also carries an ``ARITHMETIC_CHECK``,
    holding the batch's **gross**; returning it here would hand the evaluator a
    plausible number that is wrong by the whole fee. Only payment links have an
    allocated share, so only payment links are accepted.
    """
    if candidate.link_type is not LinkType.PAYMENT_CREDITED_AS:
        raise ValueError(
            f"{candidate.candidate_id} is a {candidate.link_type.value} link; "
            "only expanded payment links carry an allocated share"
        )
    for item in candidate.evidence:
        if item.kind is EvidenceKind.ARITHMETIC_CHECK and item.amount_minor is not None:
            return item.amount_minor
    raise ValueError(  # pragma: no cover - every payment link is built with one
        f"{candidate.candidate_id} carries no allocation evidence"
    )


def _rivals(
    view: SettlementView,
    chosen: CanonicalBankTxn,
    context: MatchContext,
    rule: BankLegRule,
) -> tuple[CanonicalBankTxn, ...]:
    """Unclaimed credits, other than ``chosen``, that could also be this payout.

    The second half of mutual uniqueness. Judged on the tier's own band: if two
    rows are indistinguishable under the rule the tier is applying, that tier
    cannot tell them apart.

    A credit publishing a **different** settlement's reference is not a rival.
    The bank wrote a reference and it is not this batch's, so the row is already
    explained and its amount agreeing is a coincidence. Only rows carrying no
    reference at all -- the ones A07 stripped, which is exactly where the
    dangerous case lives -- or this settlement's own key can compete for it.
    """
    return tuple(
        txn
        for txn in context.open_credits()
        if txn.txn_id != chosen.txn_id
        and txn.extracted_utr in (None, view.utr)
        and _qualifies(view, txn, rule)
    )


def _shortfall_sits_in_the_statement(
    view: SettlementView,
    chosen: CanonicalBankTxn,
    context: MatchContext,
) -> CanonicalBankTxn | None:
    """The unclaimed credit that makes up exactly what ``chosen`` is short.

    A band absorbs drift of a few paise. Where the gap between the declared net
    and the credit is instead the whole amount of another row in the statement,
    the payout was split and this credit carries only part of the batch --
    however comfortably the gap fits inside a proportional tolerance computed on
    a large net.

    Exact equality, over unclaimed credits only, and only for a credit that came
    up *short*. An overpayment is not a split.
    """
    shortfall = view.net_minor - chosen.credit_minor
    if shortfall <= 0:
        return None
    for txn in context.open_credits():
        if txn.txn_id != chosen.txn_id and txn.credit_minor == shortfall:
            return txn
    return None


def _qualifies(view: SettlementView, txn: CanonicalBankTxn, rule: BankLegRule) -> bool:
    """Whether this credit could be this settlement's payout under ``rule``."""
    if not within_tolerance(txn.credit_minor, view.net_minor, rule.band_for(view.net_minor)):
        return False
    if rule.date_window_days is None:
        return True
    gap = (txn.value_date - view.settlement.settled_on).days
    return abs(gap) <= rule.date_window_days


def _features(
    view: SettlementView, txn: CanonicalBankTxn, rule: BankLegRule
) -> FeatureVector:
    delta = txn.credit_minor - view.net_minor
    return FeatureVector(
        tier=rule.tier,
        amount_delta_minor=delta,
        tolerance_band_minor=rule.band_for(view.net_minor),
        amount_delta_ratio=delta_ratio(delta, view.net_minor),
        date_delta_days=(txn.value_date - view.settlement.settled_on).days,
    )


def _key_evidence(view: SettlementView, txn: CanonicalBankTxn) -> Evidence:
    return Evidence(
        kind=EvidenceKind.EXACT_KEY,
        detail=(
            f"UTR {view.utr} published by {view.settlement_id} appears in the "
            f"narration of {txn.txn_id}"
        ),
        refs=(settlement_ref(view.settlement_id), bank_ref(txn.txn_id)),
    )


def _amount_evidence(view: SettlementView, txn: CanonicalBankTxn, band: int) -> Evidence:
    delta = txn.credit_minor - view.net_minor
    if delta == 0:
        detail = (
            f"credit {format_minor(txn.credit_minor)} equals the net declared by "
            f"{view.settlement_id} exactly"
        )
    else:
        detail = (
            f"credit {format_minor(txn.credit_minor)} differs from the net declared by "
            f"{view.settlement_id} by {format_minor(delta)}, inside the "
            f"{format_minor(band)} tolerance band"
        )
    return Evidence(
        kind=EvidenceKind.AMOUNT_MATCH,
        detail=detail,
        refs=(settlement_ref(view.settlement_id), bank_ref(txn.txn_id)),
        amount_minor=txn.credit_minor,
    )


def _date_evidence(view: SettlementView, txn: CanonicalBankTxn, window: int) -> Evidence:
    gap = (txn.value_date - view.settlement.settled_on).days
    return Evidence(
        kind=EvidenceKind.DATE_PROXIMITY,
        detail=(
            f"credited {gap:+d} day(s) from the {view.settlement.settled_on.isoformat()} "
            f"settlement date, inside the +/-{window} day window"
        ),
        refs=(settlement_ref(view.settlement_id), bank_ref(txn.txn_id)),
        score=None,
    )


def _arithmetic_evidence(view: SettlementView) -> Evidence:
    if view.gross_reconciles:
        detail = (
            f"declared gross {format_minor(view.settlement.gross_minor)} equals the sum of "
            f"{len(view.payments)} nested payment(s)"
        )
    else:
        detail = (
            f"declared gross {format_minor(view.settlement.gross_minor)} does NOT equal the "
            f"{format_minor(view.payment_gross_minor)} its {len(view.payments)} nested "
            "payment(s) sum to"
        )
    return Evidence(
        kind=EvidenceKind.ARITHMETIC_CHECK,
        detail=detail,
        refs=(settlement_ref(view.settlement_id),),
        amount_minor=view.settlement.gross_minor,
    )


def _contest_evidence(
    view: SettlementView, chosen: CanonicalBankTxn, contenders: tuple[CanonicalBankTxn, ...]
) -> Evidence:
    others = [txn.txn_id for txn in contenders if txn.txn_id != chosen.txn_id]
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{len(contenders)} credits qualify for {view.settlement_id} on the same "
            f"evidence ({', '.join(txn.txn_id for txn in contenders)}); a settlement's net "
            f"is paid once, so {chosen.txn_id} cannot be distinguished from "
            f"{', '.join(others)}"
        ),
        refs=(settlement_ref(view.settlement_id), *(bank_ref(t.txn_id) for t in contenders)),
    )


def _rival_evidence(
    view: SettlementView, chosen: CanonicalBankTxn, rivals: tuple[CanonicalBankTxn, ...]
) -> Evidence:
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{chosen.txn_id} carries the key for {view.settlement_id}, but "
            f"{', '.join(txn.txn_id for txn in rivals)} would settle it just as well on "
            "amount and date; the payout is not uniquely identified, and a reference can "
            "go missing from the row that was really the payment"
        ),
        refs=(
            settlement_ref(view.settlement_id),
            bank_ref(chosen.txn_id),
            *(bank_ref(txn.txn_id) for txn in rivals),
        ),
    )


def _collision_evidence(view: SettlementView, peers: tuple[SettlementView, ...]) -> Evidence:
    others = [peer.settlement_id for peer in peers if peer.settlement_id != view.settlement_id]
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"UTR {view.utr} is published by {len(peers)} settlements "
            f"({', '.join(peer.settlement_id for peer in peers)}); the key does not "
            f"identify {view.settlement_id} against {', '.join(others)}"
        ),
        refs=tuple(settlement_ref(peer.settlement_id) for peer in peers),
    )


def _clawback_evidence(view: SettlementView, clawback: ClawBack) -> Evidence | None:
    """Explain, on every affected link, why a nested payment was left out."""
    if not clawback.present:
        return None
    if clawback.excluded is not None:
        return Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{view.settlement_id} declares adjustments of "
                f"{format_minor(-clawback.amount_minor)}, exactly the gross of "
                f"{clawback.excluded.payment_id}; that payment was netted off and its "
                "money never reached the bank, so it is excluded from this credit"
            ),
            refs=(
                settlement_ref(view.settlement_id),
                payment_ref(clawback.excluded.payment_id),
            ),
            amount_minor=clawback.amount_minor,
        )
    if not clawback.attributable:
        return Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{view.settlement_id} declares adjustments of "
                f"{format_minor(-clawback.amount_minor)}, which several of its payments "
                "match; the arithmetic cannot say which one was netted off, so the "
                "payments this credit carries are not established"
            ),
            refs=(settlement_ref(view.settlement_id),),
            amount_minor=clawback.amount_minor,
        )
    return Evidence(
        kind=EvidenceKind.ARITHMETIC_CHECK,
        detail=(
            f"{view.settlement_id} declares adjustments of "
            f"{format_minor(-clawback.amount_minor)}, matching no payment of its own; "
            "the claw-back belongs to another batch and every payment here was credited"
        ),
        refs=(settlement_ref(view.settlement_id),),
        amount_minor=clawback.amount_minor,
    )


def _settlement_candidate(
    view: SettlementView,
    txn: CanonicalBankTxn,
    rule: BankLegRule,
    *,
    contenders: tuple[CanonicalBankTxn, ...],
    probability: float,
    extra_evidence: tuple[Evidence, ...] = (),
) -> MatchCandidate:
    band = rule.band_for(view.net_minor)
    evidence = [_key_evidence(view, txn), _amount_evidence(view, txn, band)]
    if rule.date_window_days is not None:
        evidence.append(_date_evidence(view, txn, rule.date_window_days))
    evidence.append(_arithmetic_evidence(view))
    if len(contenders) > 1:
        evidence.append(_contest_evidence(view, txn, contenders))
    evidence.extend(extra_evidence)

    return MatchCandidate(
        candidate_id=candidate_id(
            rule.tier,
            LinkType.SETTLEMENT_CREDITED_AS,
            settlement_ref(view.settlement_id).key,
            bank_ref(txn.txn_id).key,
        ),
        link_type=LinkType.SETTLEMENT_CREDITED_AS,
        source_ref=settlement_ref(view.settlement_id),
        target_ref=bank_ref(txn.txn_id),
        tier=rule.tier,
        features=_features(view, txn, rule),
        evidence=tuple(evidence),
        calibrated_p=probability,
        arithmetic_verified=view.gross_reconciles,
    )


def _payment_candidates(
    view: SettlementView, txn: CanonicalBankTxn, rule: BankLegRule
) -> list[MatchCandidate]:
    """Expand a resolved settlement edge into the evaluation unit.

    The credit is allocated across the payments by their gross amounts, using
    the same :func:`~ledgerloop.money.allocate_minor` the generator used to
    build the truth links. Largest-remainder allocation conserves exactly, so
    the asserted link amounts sum to the credit -- no paise created by the
    matcher, none destroyed.

    This is a real improvement over baseline B0, which asserts each payment's
    full *gross*: B0's reconciled-rupee figure runs above the truth even where
    its links are right, because it has no fee model and no allocation.
    """
    if not view.payments:
        return []

    clawback = attribute_clawback(view)
    covered = tuple(
        payment
        for payment in view.payments
        if clawback.excluded is None or payment.payment_id != clawback.excluded.payment_id
    )
    if not covered:
        return []

    shares = allocate_minor(txn.credit_minor, [p.amount_minor for p in covered])
    conserved = sum_minor(shares, field=f"{view.settlement_id}.allocation") == txn.credit_minor
    verified = view.gross_reconciles and conserved and clawback.attributable
    features = _features(view, txn, rule)
    clawback_note = _clawback_evidence(view, clawback)

    candidates: list[MatchCandidate] = []
    for payment, share in zip(covered, shares, strict=True):
        candidates.append(
            MatchCandidate(
                candidate_id=candidate_id(
                    rule.tier,
                    LinkType.PAYMENT_CREDITED_AS,
                    payment_ref(payment.payment_id).key,
                    bank_ref(txn.txn_id).key,
                ),
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(payment.payment_id),
                target_ref=bank_ref(txn.txn_id),
                tier=rule.tier,
                features=features,
                evidence=(
                    Evidence(
                        kind=EvidenceKind.EXACT_KEY,
                        detail=(
                            f"{payment.payment_id} is one of {len(covered)} payment(s) "
                            f"in {view.settlement_id} whose money travelled with "
                            f"{txn.txn_id}, credited on UTR {view.utr}"
                        ),
                        refs=(
                            payment_ref(payment.payment_id),
                            settlement_ref(view.settlement_id),
                            bank_ref(txn.txn_id),
                        ),
                    ),
                    Evidence(
                        kind=EvidenceKind.ARITHMETIC_CHECK,
                        detail=(
                            f"allocated {format_minor(share)} of the "
                            f"{format_minor(txn.credit_minor)} credit, by gross weight; "
                            f"the {len(covered)} share(s) sum to the credit exactly"
                        ),
                        refs=(payment_ref(payment.payment_id), bank_ref(txn.txn_id)),
                        amount_minor=share,
                    ),
                    *((clawback_note,) if clawback_note is not None else ()),
                ),
                calibrated_p=1.0,
                arithmetic_verified=verified,
            )
        )
    return candidates


def resolve_bank_leg(context: MatchContext, rule: BankLegRule) -> BankLegOutcome:
    """Run one tier's pass over the settlements still in the pool.

    Iterates in source order and consumes as it goes, so the result is a pure
    function of the ingested corpus and the rule.
    """
    candidates: list[MatchCandidate] = []
    resolved = contested = undecided = resolved_credits = 0
    without_key = collisions = split_suspected = 0

    for view in list(context.open_settlements()):
        if view.settlement_id in context.consumed_settlements:
            continue  # consumed by a collision group earlier in this same pass
        if view.utr is None:
            without_key += 1
            continue

        peers = context.open_settlements_for(view.utr)
        if len(peers) > 1:
            # A key naming several settlements identifies none of them. The whole
            # group leaves the pool together, or the survivors would look
            # unambiguous on the next iteration of this same pass.
            collisions += 1
            open_credits = context.open_credits_for(view.utr)
            for peer in peers:
                for txn in open_credits:
                    if _qualifies(peer, txn, rule):
                        candidates.append(
                            _settlement_candidate(
                                peer,
                                txn,
                                rule,
                                contenders=(txn,),
                                probability=1.0 / len(peers),
                                extra_evidence=(_collision_evidence(peer, peers),),
                            )
                        )
                contested += 1
                context.consume(peer.settlement_id)
            continue

        keyed = context.open_credits_for(view.utr)
        contenders = tuple(txn for txn in keyed if _qualifies(view, txn, rule))

        if len(keyed) > 1 and len(contenders) <= 1:
            # The key is on several credits but the batch total is on at most
            # one: the payout was split. A whole-batch match would credit this
            # tranche with payments the other one carried. Left in the pool --
            # partitioning payments across tranches is T2's arithmetic.
            split_suspected += 1
            undecided += 1
            continue

        if not contenders:
            undecided += 1
            continue

        if len(contenders) == 1:
            remainder = _shortfall_sits_in_the_statement(view, contenders[0], context)
            if remainder is not None:
                split_suspected += 1
                undecided += 1
                continue

        rivals: tuple[CanonicalBankTxn, ...] = ()
        if len(contenders) == 1:
            rivals = _rivals(view, contenders[0], context, rule)

        # Mutual uniqueness: a lone keyed contender is only a match when nothing
        # else in the statement could be the same payout.
        field_size = len(contenders) + len(rivals)
        probability = 1.0 / field_size
        for txn in contenders:
            candidates.append(
                _settlement_candidate(
                    view,
                    txn,
                    rule,
                    contenders=contenders,
                    probability=probability,
                    extra_evidence=(
                        (_rival_evidence(view, txn, rivals),) if rivals else ()
                    ),
                )
            )

        if field_size == 1:
            winner = contenders[0]
            candidates.extend(_payment_candidates(view, winner, rule))
            context.consume(view.settlement_id, [winner.txn_id])
            resolved += 1
            resolved_credits += 1
        else:
            # The credits stay in the pool: a contested settlement has claimed
            # none of them, and one may yet be resolved against another batch.
            context.consume(view.settlement_id)
            contested += 1

    return BankLegOutcome(
        tier=rule.tier,
        candidates=tuple(candidates),
        resolved_settlements=resolved,
        contested_settlements=contested,
        undecided_settlements=undecided,
        resolved_credits=resolved_credits,
        settlements_without_key=without_key,
        key_collisions=collisions,
        split_suspected=split_suspected,
    )
