"""T2 -- aggregation. Which payments compose this credit.

PLAN.md 6.2, the tier the plan calls the algorithmic core. T0 and T1 ask
whether *one* credit is *one* settlement's whole payout. T2 asks the N:1
question underneath it: when a batch is paid out in tranches, which payments
travelled in which tranche?

It is the tier that takes anomaly A09 ``SPLIT_PAYOUT``, which T0 and T1
deliberately leave in the pool -- a whole-batch match against one tranche would
credit it with payments the other tranche carried, so they decline and count it
as ``split_suspected`` rather than consuming it.

THE SEARCH IS ANCHORED, NOT GLOBAL
-----------------------------------
PLAN.md 6.2 step 2: "Anchor on the declared ``settlement_id`` when present --
this collapses the search to verifying one subset rather than searching all of
them." Every payment here carries that anchor, because the PSP nests it inside
its settlement, so the candidate pool for a credit is never "all open payments"
but "this settlement's open payments". A twenty-item search instead of a
three-hundred-item one, and the difference is not speed but *correctness*: a
global search would happily explain a credit with payments from a batch that
has nothing to do with it.

The credits are anchored the same way, by the UTR the settlement published.
PLAN.md 6.2 step 1 also describes bucketing credits by a ±3-day window; that is
**not** applied on top, and the omission is deliberate. The key already
identifies the settlement exactly, so a date filter can only remove true
tranches -- and A04 ``TIMING_SHIFT`` composed with A09 puts a legitimate second
tranche up to five days out. The date gap is recorded as evidence instead of
being used as a gate.

WHAT IS SEARCHED, AND IN WHAT SPACE
------------------------------------
Payments carry **gross** amounts; a bank credit carries **net**. The bridge is
proportional: a tranche covering gross ``g`` of a batch whose gross is ``G``
carries ``allocate_minor(N, [g, G - g])[0]`` of the net -- the same
largest-remainder split the generator used to build the truth links, and the
same one T0/T1 use to allocate a whole batch.

So the subset-sum runs in gross space over a window derived from the credit,
and **every candidate is then verified by re-deriving the credit exactly**. The
window is deliberately widened by a paise at each end and the verification is
what actually decides, so no rounding assumption in the window derivation can
admit a subset that does not reconcile.

REFUSING TO CHOOSE
------------------
PLAN.md 6.2 step 5: "if two different subsets both fit within ε, **do not
match** -- emit an ``AMBIGUOUS_AGGREGATION`` exception with both hypotheses.
Silently picking one is exactly the dishonesty the track brief warns against."

That is enforced by the solver counting rather than finding, and by this module
treating "unique" as *exhaustively* unique: a greedy fallback that found a
subset but cannot prove it is alone gets the same treatment as two subsets. The
probability convention is Step 4's, unchanged -- ``p = 1/n`` over the
hypotheses that could not be told apart -- so a two-way ambiguity lands at 0.5,
below the configured ``tau_low``, and routes to an exception through the policy
rather than through a special case written into this tier.

TRANSACTIONAL PER SETTLEMENT
-----------------------------
A batch is solved as a whole or not at all. Credits are taken largest first
(most constrained), each search runs over what the previous ones left, and when
the credits account for the whole net the last one is *verified against the
remainder* rather than searched -- which both removes a search and tightens
uniqueness. Nothing is emitted until the entire assignment holds, so a batch
can never end up half-explained with its remaining payments silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ledgerloop.config import LexicalMatching, MatchingTolerances
from ledgerloop.matching.bank_leg import attribute_clawback, candidate_id
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.matching.subset_sum import (
    Accept,
    SubsetSearch,
    SubsetSolution,
    find_subsets,
)
from ledgerloop.matching.tier3_lexical import MerchantProfile, score_names
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.records import CanonicalBankTxn, CanonicalPayment
from ledgerloop.models.refs import RecordRef, bank_ref, payment_ref, settlement_ref
from ledgerloop.money import (
    allocate_minor,
    delta_ratio,
    format_minor,
    sum_minor,
    within_tolerance,
)

__all__ = [
    "AggregationOutcome",
    "Assignment",
    "accept_for",
    "credit_bucket",
    "expected_credit_minor",
    "features_for",
    "find_tranche_set",
    "lexical_credit_bucket",
    "payment_bucket",
    "run_split_completion",
    "run_tier2",
    "search_window",
]


@dataclass(frozen=True)
class Assignment:
    """One credit and the payments T2 concluded travelled in it."""

    credit: CanonicalBankTxn
    payments: tuple[CanonicalPayment, ...]
    gross_minor: int
    expected_minor: int
    search: SubsetSearch

    @property
    def residual_minor(self) -> int:
        """What the subset leaves unexplained. Zero on a clean split."""
        return self.credit.credit_minor - self.expected_minor

    @property
    def proven_unique(self) -> bool:
        """Whether the search is entitled to say this subset is the only one."""
        return self.search.is_unique


@dataclass(frozen=True)
class AggregationOutcome:
    """What one T2 pass over the residual pool produced."""

    candidates: tuple[MatchCandidate, ...] = ()
    settlements_seen: int = 0
    settlements_resolved: int = 0
    settlements_ambiguous: int = 0
    settlements_unsolved: int = 0
    settlements_without_key: int = 0
    credits_matched: int = 0
    payments_matched: int = 0
    subsets_examined: int = 0
    greedy_fallbacks: int = 0
    timeouts: int = 0

    @property
    def tier(self) -> Tier:
        return Tier.T2_AGGREGATION

    @property
    def settlement_links(self) -> int:
        return sum(
            1 for c in self.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )

    @property
    def payment_links(self) -> int:
        return sum(1 for c in self.candidates if c.is_evaluable)


def payment_bucket(view: SettlementView, context: MatchContext) -> tuple[CanonicalPayment, ...]:
    """The payments T2 may draw a subset from, for one settlement.

    The settlement's own nested payments -- the anchor -- minus any payment a
    negative adjustment identifies as charged back, whose money never reached
    the bank and which therefore belongs to no tranche. That exclusion is the
    same exact arithmetic T0/T1 apply; reusing it keeps one rule in one place.
    """
    del context  # anchoring makes the settlement its own bucket
    clawback = attribute_clawback(view)
    excluded = clawback.excluded.payment_id if clawback.excluded is not None else None
    return tuple(p for p in view.payments if p.payment_id != excluded)


def credit_bucket(view: SettlementView, context: MatchContext) -> tuple[CanonicalBankTxn, ...]:
    """The unclaimed credits publishing this settlement's key, largest first.

    Largest first because the biggest tranche is the most constrained: it has
    the fewest subsets that can reach it, so searching it first prunes hardest
    and leaves the smallest residue for the tranches after it. Ties break on
    ``txn_id`` so the order is total and the run reproducible.
    """
    if view.utr is None:
        return ()
    return tuple(
        sorted(
            context.open_credits_for(view.utr),
            key=lambda txn: (-txn.credit_minor, txn.txn_id),
        )
    )


def expected_credit_minor(gross_minor: int, view_gross: int, net_minor: int) -> int:
    """The net a tranche covering ``gross_minor`` of a batch carries.

    ``allocate_minor(N, [g, G - g])[0]`` -- the project's own conservation-
    preserving split, and the same one the generator used to build the truth
    links. Integers throughout; the money path never sees a ratio.
    """
    if view_gross <= 0:
        return 0
    if gross_minor >= view_gross:
        return net_minor
    return allocate_minor(net_minor, [gross_minor, view_gross - gross_minor])[0]


def search_window(
    credit_minor: int, view_gross: int, net_minor: int, epsilon: int
) -> tuple[int, int]:
    """The gross range whose implied credit could land within ``epsilon``.

    Derived by inverting the proportional share, then **widened by one paise at
    each end** to swallow every rounding assumption in the inversion. The window
    only narrows the search; the exact re-derivation in
    :func:`expected_credit_minor` is what decides.
    """
    low = ((credit_minor - epsilon - 1) * view_gross) // net_minor
    high = -(-((credit_minor + epsilon + 1) * view_gross) // net_minor)
    return max(0, low), min(view_gross, high)


def _subset_refs(payments: tuple[CanonicalPayment, ...]) -> tuple[RecordRef, ...]:
    return tuple(payment_ref(p.payment_id) for p in payments)


def features_for(assignment: Assignment, view: SettlementView, epsilon: int) -> FeatureVector:
    residual = assignment.residual_minor
    return FeatureVector(
        tier=Tier.T2_AGGREGATION,
        amount_delta_minor=residual,
        tolerance_band_minor=epsilon,
        amount_delta_ratio=delta_ratio(residual, assignment.credit.credit_minor),
        date_delta_days=(
            assignment.credit.value_date - view.settlement.settled_on
        ).days,
        subset_size=len(assignment.payments),
    )


def _subset_evidence(assignment: Assignment, view: SettlementView, epsilon: int) -> Evidence:
    members = ", ".join(p.payment_id for p in assignment.payments)
    return Evidence(
        kind=EvidenceKind.SUBSET_SUM,
        detail=(
            f"{len(assignment.payments)} of {len(view.payments)} payment(s) in "
            f"{view.settlement_id} ({members}) carry gross "
            f"{format_minor(assignment.gross_minor)} of the batch's "
            f"{format_minor(view.settlement.gross_minor)}, which allocates to "
            f"{format_minor(assignment.expected_minor)} of the "
            f"{format_minor(view.net_minor)} net -- matching "
            f"{assignment.credit.txn_id} within {format_minor(epsilon)} "
            f"(residual {format_minor(assignment.residual_minor)})"
        ),
        refs=(
            settlement_ref(view.settlement_id),
            bank_ref(assignment.credit.txn_id),
            *_subset_refs(assignment.payments),
        ),
        amount_minor=assignment.credit.credit_minor,
    )


def _uniqueness_evidence(assignment: Assignment, view: SettlementView) -> Evidence:
    search = assignment.search
    if search.is_unique:
        return Evidence(
            kind=EvidenceKind.ARITHMETIC_CHECK,
            detail=(
                f"exhaustive search over {len(view.payments)} payment(s) found exactly "
                f"one subset reaching {assignment.credit.txn_id} "
                f"({search.examined} combination(s) examined by "
                f"{search.method.replace('_', ' ')})"
            ),
            refs=(settlement_ref(view.settlement_id), bank_ref(assignment.credit.txn_id)),
        )
    if search.method == "greedy":
        return Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{view.settlement_id} has {len(view.payments)} payments, past the "
                "exhaustive cap, so this subset was found greedily; it fits, but the "
                "search cannot prove no other subset does"
            ),
            refs=(settlement_ref(view.settlement_id), bank_ref(assignment.credit.txn_id)),
        )
    # A search that hit its time bound never becomes an assignment -- `_solve`
    # abandons the whole settlement -- so the only remaining case is a search
    # that stopped early because a rival subset turned up.
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"the search stopped as soon as a {search.found}th subset reaching "
            f"{assignment.credit.txn_id} appeared; knowing there is more than one is "
            "enough to decline, and counting the rest would buy nothing"
        ),
        refs=(settlement_ref(view.settlement_id), bank_ref(assignment.credit.txn_id)),
    )


def _ambiguity_evidence(
    view: SettlementView,
    credit: CanonicalBankTxn,
    hypotheses: tuple[SubsetSolution, ...],
    payments: tuple[CanonicalPayment, ...],
) -> Evidence:
    rendered = "; ".join(
        "{" + ", ".join(payments[i].payment_id for i in solution.indices) + "}"
        for solution in hypotheses
    )
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{len(hypotheses)} different subsets of {view.settlement_id} reach "
            f"{credit.txn_id} within tolerance ({rendered}); picking one would be a "
            "coin flip, so none is asserted"
        ),
        refs=(settlement_ref(view.settlement_id), bank_ref(credit.txn_id)),
        amount_minor=credit.credit_minor,
    )


def _conservation_evidence(
    view: SettlementView, assignments: tuple[Assignment, ...]
) -> Evidence:
    """The end-to-end money check on a committed partition.

    Only ever built for a partition that accounts for the whole net, because
    that is the only kind the tier commits -- so it can state the strong form:
    the tranches add up, and every payment is in exactly one of them.
    """
    total = sum_minor(
        (a.credit.credit_minor for a in assignments), field=f"{view.settlement_id}.tranches"
    )
    covered = sum(len(a.payments) for a in assignments)
    detail = (
        f"{len(assignments)} tranche(s) totalling {format_minor(total)} account for "
        f"the whole {format_minor(view.net_minor)} net of {view.settlement_id}, and "
        f"the partition covers all {covered} payment(s) exactly once"
    )
    return Evidence(
        kind=EvidenceKind.ARITHMETIC_CHECK,
        detail=detail,
        refs=(settlement_ref(view.settlement_id),),
        amount_minor=total,
    )


def _settlement_candidate(
    view: SettlementView,
    assignment: Assignment,
    *,
    probability: float,
    verified: bool,
    epsilon: int,
    extra: tuple[Evidence, ...] = (),
) -> MatchCandidate:
    return MatchCandidate(
        candidate_id=candidate_id(
            Tier.T2_AGGREGATION,
            LinkType.SETTLEMENT_CREDITED_AS,
            settlement_ref(view.settlement_id).key,
            bank_ref(assignment.credit.txn_id).key,
        ),
        link_type=LinkType.SETTLEMENT_CREDITED_AS,
        source_ref=settlement_ref(view.settlement_id),
        target_ref=bank_ref(assignment.credit.txn_id),
        tier=Tier.T2_AGGREGATION,
        features=features_for(assignment, view, epsilon),
        evidence=(
            _subset_evidence(assignment, view, epsilon),
            _uniqueness_evidence(assignment, view),
            *extra,
        ),
        subset_members=_subset_refs(assignment.payments),
        calibrated_p=probability,
        arithmetic_verified=verified,
    )


def _payment_candidates(
    view: SettlementView,
    assignment: Assignment,
    *,
    probability: float,
    verified: bool,
    epsilon: int,
    extra: tuple[Evidence, ...] = (),
) -> list[MatchCandidate]:
    """Expand one tranche into the evaluation unit.

    The tranche is allocated across the payments it carries, by gross weight,
    with the same conserving split the truth links were built from -- so the
    shares sum to the credit exactly and a correct link is correct in rupees.
    """
    credit = assignment.credit
    shares = allocate_minor(credit.credit_minor, [p.amount_minor for p in assignment.payments])
    conserved = (
        sum_minor(shares, field=f"{credit.txn_id}.allocation") == credit.credit_minor
    )
    features = features_for(assignment, view, epsilon)
    subset = _subset_refs(assignment.payments)
    subset_evidence = _subset_evidence(assignment, view, epsilon)

    candidates: list[MatchCandidate] = []
    for payment, share in zip(assignment.payments, shares, strict=True):
        candidates.append(
            MatchCandidate(
                candidate_id=candidate_id(
                    Tier.T2_AGGREGATION,
                    LinkType.PAYMENT_CREDITED_AS,
                    payment_ref(payment.payment_id).key,
                    bank_ref(credit.txn_id).key,
                ),
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(payment.payment_id),
                target_ref=bank_ref(credit.txn_id),
                tier=Tier.T2_AGGREGATION,
                features=features,
                evidence=(
                    subset_evidence,
                    Evidence(
                        kind=EvidenceKind.ARITHMETIC_CHECK,
                        detail=(
                            f"allocated {format_minor(share)} of the "
                            f"{format_minor(credit.credit_minor)} tranche, by gross "
                            f"weight; the {len(assignment.payments)} share(s) sum to "
                            "the tranche exactly"
                        ),
                        refs=(payment_ref(payment.payment_id), bank_ref(credit.txn_id)),
                        amount_minor=share,
                    ),
                    *extra,
                ),
                subset_members=subset,
                calibrated_p=probability,
                arithmetic_verified=verified and conserved,
            )
        )
    return candidates


def _remainder_search(indices: tuple[int, ...], total: int) -> SubsetSearch:
    """A search result standing for "the leftovers, verified".

    The final tranche of a partition that accounts for the whole net is not
    searched at all -- whatever the earlier tranches did not take must be its,
    and that is checked rather than looked for. Recording it as an exhaustive
    result of one is accurate: there is exactly one remainder.
    """
    return SubsetSearch(
        solutions=(SubsetSolution(indices=indices, total_minor=total),),
        exhaustive=True,
        method="remainder",
    )


@dataclass
class _Attempt:
    """Working state while one settlement is solved. Committed or discarded whole."""

    assignments: list[Assignment]
    ambiguity: tuple[CanonicalBankTxn, SubsetSearch, tuple[CanonicalPayment, ...]] | None = None
    unproven: Assignment | None = None
    timed_out: bool = False
    failed: bool = False
    examined: int = 0
    greedy: int = 0


def accept_for(target: int, gross: int, net: int, epsilon: int) -> Accept:
    """The exact re-derivation a candidate subset has to survive.

    Bound as a factory rather than written inline in the loop, so the credit it
    closes over is the one it was built for. A closure capturing a loop variable
    would silently re-target itself on the next iteration.
    """

    def accept(indices: tuple[int, ...], total: int) -> bool:
        del indices
        return within_tolerance(expected_credit_minor(total, gross, net), target, epsilon)

    return accept


def _solve(
    view: SettlementView,
    payments: tuple[CanonicalPayment, ...],
    credits: tuple[CanonicalBankTxn, ...],
    tolerances: MatchingTolerances,
) -> _Attempt:
    """Assign every credit a subset, or explain why it could not be done."""
    epsilon = tolerances.aggregation_epsilon_minor
    # **The denominator is the bucket, not the batch.** `payments` is what
    # `payment_bucket` decided actually travelled -- the nested payments minus
    # any a negative adjustment identifies as charged back, whose money never
    # reached the bank. Allocating the net across *all* nested payments while
    # partitioning only the ones that arrived puts a payment in the denominator
    # that is in no tranche, and every tranche then comes out short by its
    # share of money that was never paid.
    #
    # Found in Phase 2.5, on A08 composed with A09: SETL-0004 on `test` seed 42
    # expects Rs 2,02,486.64 for its larger tranche and the batch-gross
    # denominator predicted Rs 1,83,195.85 -- out by Rs 19,290.79, far outside
    # any epsilon, so the batch was refused. With the bucket's gross the
    # prediction is exact on every tranche of every case, delta = 0.
    #
    # For a batch with no claw-back the two are equal, so this is a correction
    # to a composed case and not a change of policy. `run_tier2`'s behaviour on
    # every corpus in this project is unchanged by it, which is measured rather
    # than assumed -- see `.local/steps/phase-2-5.md` section 5.
    gross = sum_minor(
        (payment.amount_minor for payment in payments),
        field=f"{view.settlement_id}.bucket_gross",
    )
    net = view.net_minor
    attempt = _Attempt(assignments=[])
    remaining = list(payments)

    for position, credit in enumerate(credits):
        is_last = position == len(credits) - 1
        if not remaining:
            attempt.failed = True
            return attempt

        if is_last:
            # The remainder is the only thing this tranche can be. Verified, not
            # searched -- one fewer search, and a tighter uniqueness claim.
            taken = tuple(remaining)
            subtotal = sum_minor(
                (p.amount_minor for p in taken), field=f"{view.settlement_id}.remainder"
            )
            expected = expected_credit_minor(subtotal, gross, net)
            if not within_tolerance(expected, credit.credit_minor, epsilon):
                attempt.failed = True
                return attempt
            search = _remainder_search(tuple(range(len(remaining))), subtotal)
        else:
            low, high = search_window(credit.credit_minor, gross, net, epsilon)
            search = find_subsets(
                [p.amount_minor for p in remaining],
                low,
                high,
                want=2,
                max_exact_items=tolerances.max_subset_size,
                timeout_ms=tolerances.subset_solver_timeout_ms,
                accept=accept_for(credit.credit_minor, gross, net, epsilon),
            )
            attempt.examined += search.examined
            if search.method == "greedy":
                attempt.greedy += 1
            if search.timed_out:
                attempt.timed_out = True
                return attempt
            if search.is_ambiguous:
                attempt.ambiguity = (credit, search, tuple(remaining))
                return attempt
            if not search.solutions:
                attempt.failed = True
                return attempt
            solution = search.solutions[0]
            taken = tuple(remaining[i] for i in solution.indices)
            subtotal = solution.total_minor
            expected = expected_credit_minor(subtotal, gross, net)

        assignment = Assignment(
            credit=credit,
            payments=taken,
            gross_minor=subtotal,
            expected_minor=expected,
            search=search,
        )
        attempt.assignments.append(assignment)
        if not assignment.proven_unique:
            attempt.unproven = assignment
        chosen = {p.payment_id for p in taken}
        remaining = [p for p in remaining if p.payment_id not in chosen]

    # No leftover check is needed: the final tranche takes the whole remainder
    # by construction, so a completed loop has assigned every payment exactly
    # once. The partition is total or the loop returned early.
    return attempt


def run_tier2(context: MatchContext, tolerances: MatchingTolerances) -> AggregationOutcome:
    """Run T2 over whatever T0 and T1 left in the pool.

    Only settlements whose **keyed credits together account for the whole net**
    are attempted. That restriction is what makes the tier safe: a partition
    covering the batch conserves money exactly and assigns every payment once,
    so the arithmetic can be checked end to end. A lone tranche whose sibling is
    missing from the pool -- because A07 stripped its reference -- is a
    different problem, and it belongs to the tiers that can match an
    unreferenced row rather than to this one.
    """
    epsilon = tolerances.aggregation_epsilon_minor
    candidates: list[MatchCandidate] = []
    seen = resolved = ambiguous = unsolved = without_key = 0
    credits_matched = payments_matched = examined = greedy = timeouts = 0

    for view in list(context.open_settlements()):
        credits = credit_bucket(view, context)
        if not credits:
            without_key += 1
            continue
        if len(credits) < 2:
            # One credit is a whole-batch question, which T0/T1 own. T2 exists
            # for the N:1 case and must not re-litigate a one-to-one match.
            continue

        payments = payment_bucket(view, context)
        gross = sum_minor(
            (payment.amount_minor for payment in payments),
            field=f"{view.settlement_id}.bucket_gross",
        )
        if not payments or gross <= 0 or view.net_minor <= 0:
            continue

        total = sum_minor(
            (txn.credit_minor for txn in credits), field=f"{view.settlement_id}.credits"
        )
        if not within_tolerance(total, view.net_minor, epsilon):
            # The tranches do not add up to the payout, so this is not a split
            # of this batch. Left in the pool rather than forced.
            continue

        seen += 1
        attempt = _solve(view, payments, credits, tolerances)
        examined += attempt.examined
        greedy += attempt.greedy

        if attempt.timed_out:
            # A resource failure, not a data ambiguity. Nothing is concluded and
            # the settlement stays in the pool for a later tier.
            timeouts += 1
            unsolved += 1
            continue

        if attempt.ambiguity is not None:
            credit, search, pool = attempt.ambiguity
            first = search.solutions[0]
            hypothesis = Assignment(
                credit=credit,
                payments=tuple(pool[i] for i in first.indices),
                gross_minor=first.total_minor,
                expected_minor=expected_credit_minor(first.total_minor, gross, view.net_minor),
                search=search,
            )
            candidates.append(
                _settlement_candidate(
                    view,
                    hypothesis,
                    probability=1.0 / search.found,
                    verified=False,
                    epsilon=epsilon,
                    extra=(_ambiguity_evidence(view, credit, search.solutions, pool),),
                )
            )
            context.consume(view.settlement_id)
            ambiguous += 1
            continue

        if attempt.failed or not attempt.assignments:
            unsolved += 1
            continue

        if attempt.unproven is not None:
            # A subset was found but its uniqueness was never established. Same
            # treatment as an outright ambiguity: proposed, never asserted.
            candidates.append(
                _settlement_candidate(
                    view,
                    attempt.unproven,
                    probability=0.5,
                    verified=False,
                    epsilon=epsilon,
                )
            )
            context.consume(view.settlement_id)
            ambiguous += 1
            continue

        conservation = _conservation_evidence(view, tuple(attempt.assignments))
        for assignment in attempt.assignments:
            candidates.append(
                _settlement_candidate(
                    view,
                    assignment,
                    probability=1.0,
                    verified=True,
                    epsilon=epsilon,
                    extra=(conservation,),
                )
            )
            candidates.extend(
                _payment_candidates(
                    view,
                    assignment,
                    probability=1.0,
                    verified=True,
                    epsilon=epsilon,
                    extra=(conservation,),
                )
            )
            payments_matched += len(assignment.payments)
        credits_matched += len(attempt.assignments)
        context.consume(view.settlement_id, [a.credit.txn_id for a in attempt.assignments])
        resolved += 1

    return AggregationOutcome(
        candidates=tuple(candidates),
        settlements_seen=seen,
        settlements_resolved=resolved,
        settlements_ambiguous=ambiguous,
        settlements_unsolved=unsolved,
        settlements_without_key=without_key,
        credits_matched=credits_matched,
        payments_matched=payments_matched,
        subsets_examined=examined,
        greedy_fallbacks=greedy,
        timeouts=timeouts,
    )


# ---------------------------------------------------------------------------
# Split completion (Phase 2.5) -- T2 reaching a split payout with no reference
#
# `run_tier2` finds tranches through the settlement's key. When A09 composes
# with A07 and every tranche loses its narration reference, `credit_bucket`
# returns nothing, T2 never sees the batch, and no later tier can help: T3
# matches ONE credit against the WHOLE net, and a tranche is by definition not
# the whole net. The batch falls through the ladder entirely.
#
# What follows supplies the tranche set from T3's merchant evidence and then
# runs `_solve` -- T2's own partition machinery, unchanged and uncopied -- over
# it. Everything below this line is candidate *generation*; not one line of the
# arithmetic, the uniqueness rule or the conservation check is duplicated.
# ---------------------------------------------------------------------------


def lexical_credit_bucket(
    view: SettlementView,
    context: MatchContext,
    lexical: LexicalMatching,
    profiles: Mapping[str, MerchantProfile],
) -> tuple[CanonicalBankTxn, ...]:
    """Unreferenced credits that could be a tranche of this settlement's payout.

    The same three tests T3 applies to a whole-batch candidate, minus the one
    that cannot hold for a tranche:

    * **no reference of its own.** A credit publishing some *other* settlement's
      UTR is already explained and its amount agreeing is a coincidence; one
      publishing *this* settlement's is T2's business through the keyed path.
      Only the rows A07 stripped can compete here -- which is exactly where the
      case this pass exists for lives.
    * **the merchant the batch belongs to**, scored against the master T3
      derived from the statement's own keyed narrations. No new gate: the
      configured :attr:`LexicalMatching.min_score`.
    * **inside T3's date window** of the settlement date.

    What is deliberately *not* applied is T3's amount test. T3 requires a
    candidate to be within tolerance of the **whole net**, and a tranche never
    is -- that single filter is why the composed A09+A07 case is invisible to
    every existing tier.

    Largest first, then ``txn_id``: a total order, so the run is reproducible
    and the subset search enumerates identically every time.
    """
    merchant = context.merchant_of(view)
    if merchant is None:
        return ()
    profile = profiles.get(merchant)
    if profile is None or not profile.spellings:
        return ()

    picked: list[CanonicalBankTxn] = []
    for txn in context.open_credits():
        if txn.extracted_utr is not None or txn.extracted_merchant is None:
            continue
        best = max(score_names(txn.extracted_merchant, s) for s in profile.spellings)
        if best < lexical.min_score:
            continue
        gap = (txn.value_date - view.settlement.settled_on).days
        if abs(gap) > lexical.date_window_days:
            continue
        picked.append(txn)
    return tuple(sorted(picked, key=lambda txn: (-txn.credit_minor, txn.txn_id)))


def find_tranche_set(
    view: SettlementView,
    credits: tuple[CanonicalBankTxn, ...],
    tolerances: MatchingTolerances,
) -> tuple[tuple[CanonicalBankTxn, ...] | None, SubsetSearch]:
    """The one set of credits summing **exactly** to the net, or nothing.

    Three refusals, and each is the reason the pass can be trusted rather than a
    tuning choice:

    * **Exact, not banded.** The target window is ``[net, net]``. A split payout
      conserves money by construction -- the tranches *are* the payout -- so a
      tolerance here would not absorb rounding drift, it would admit sets that
      are merely close, and "close" over a pool of similar amounts is where
      false positives live. T2's ``epsilon`` still governs the *partition*
      below, where per-payment rounding genuinely accumulates.
    * **At least two credits.** A single credit equal to the whole net is a
      one-to-one match and belongs to T0/T1/T3. This pass must not re-litigate
      one, and a "split" of one tranche is not a split.
    * **Exhaustively unique.** ``SubsetSearch.is_unique`` is
      ``exhaustive and len(solutions) == 1``. A greedy fallback on a pool too
      wide to enumerate can find a set but can never prove it is alone, so it
      is refused -- the same distinction T2 already draws.
    """
    if len(credits) < 2:
        return None, SubsetSearch(solutions=(), exhaustive=True, method="meet_in_the_middle")

    search = find_subsets(
        [txn.credit_minor for txn in credits],
        view.net_minor,
        view.net_minor,
        want=2,
        max_exact_items=tolerances.max_subset_size,
        timeout_ms=tolerances.subset_solver_timeout_ms,
        accept=_at_least_two_members,
    )
    if not search.is_unique:
        return None, search
    solution = search.solutions[0]
    return tuple(credits[index] for index in solution.indices), search


def _at_least_two_members(indices: tuple[int, ...], total: int) -> bool:
    """A one-credit "split" is a whole-batch match. Refused here, not counted."""
    del total
    return len(indices) >= 2


def _tranche_set_evidence(
    view: SettlementView,
    credits: tuple[CanonicalBankTxn, ...],
    search: SubsetSearch,
) -> Evidence:
    members = ", ".join(
        f"{txn.txn_id} {format_minor(txn.credit_minor)}" for txn in credits
    )
    return Evidence(
        kind=EvidenceKind.SUBSET_SUM,
        detail=(
            f"{len(credits)} unreferenced credit(s) ({members}) sum exactly to "
            f"{format_minor(view.net_minor)}, the net {view.settlement_id} "
            f"declares -- and no other subset of the {search.examined} "
            f"combination(s) examined reaches it, so the payout was split "
            f"across these rows and no others"
        ),
        refs=(
            settlement_ref(view.settlement_id),
            *(bank_ref(txn.txn_id) for txn in credits),
        ),
        amount_minor=view.net_minor,
    )


def _lexical_grounding_evidence(
    view: SettlementView,
    credits: tuple[CanonicalBankTxn, ...],
    profile: MerchantProfile,
) -> Evidence:
    names = ", ".join(sorted({txn.extracted_merchant or "" for txn in credits}))
    return Evidence(
        kind=EvidenceKind.LEXICAL_SIMILARITY,
        detail=(
            f"none of these rows carries a reference (A07 stripped it); they "
            f"name {names!r}, which the statement's own keyed credits use for "
            f"the merchant behind {view.settlement_id} "
            f"({profile.witnesses} spelling(s) on file)"
        ),
        refs=tuple(bank_ref(txn.txn_id) for txn in credits),
    )


def run_split_completion(
    context: MatchContext,
    tolerances: MatchingTolerances,
    lexical: LexicalMatching,
    profiles: Mapping[str, MerchantProfile],
) -> AggregationOutcome:
    """Reach the split payouts whose tranches lost their reference.

    Runs over settlements **still in the pool** after T2 and T3 have had their
    pass, and only over those with no keyed credit -- a settlement T2 could see
    through its UTR has already been ruled on, and re-litigating it here would
    let a looser candidate rule overturn a stricter tier's refusal.

    The result is a :class:`AggregationOutcome` carrying ``T2_AGGREGATION``
    candidates, because that is what they are: ``subset_members`` is a
    type-level invariant of T2 (``MatchCandidate`` refuses them on any other
    tier), and every figure in them is produced by T2's own code.

    **The pass needs T2 and T3 both enabled**, and the pipeline gates it on
    exactly that. So the published ``T0-T2`` ablation row is unchanged -- with
    T3 off there is no merchant master and no pool -- and the contribution lands
    on ``T0-T3``, which is the honest place for a result that needs both.
    """
    epsilon = tolerances.aggregation_epsilon_minor
    candidates: list[MatchCandidate] = []
    seen = resolved = ambiguous = unsolved = no_pool = 0
    credits_matched = payments_matched = examined = timeouts = 0

    for view in list(context.open_settlements()):
        if context.open_credits_for(view.utr or ""):
            continue  # keyed: T2's, through the path it already has
        payments = payment_bucket(view, context)
        if not payments or view.payment_gross_minor <= 0 or view.net_minor <= 0:
            continue

        pool = lexical_credit_bucket(view, context, lexical, profiles)
        if len(pool) < 2:
            no_pool += 1
            continue

        seen += 1
        tranches, search = find_tranche_set(view, pool, tolerances)
        examined += search.examined
        if search.timed_out:
            timeouts += 1
            unsolved += 1
            continue
        if tranches is None:
            # Either nothing sums to the net, or more than one set does. Both
            # are refusals and neither is asserted: the settlement stays in the
            # pool and the exception queue reports it as it did before.
            unsolved += 1
            continue

        # --- from here down it is T2, unchanged --------------------------
        attempt = _solve(view, payments, tranches, tolerances)
        examined += attempt.examined
        if attempt.timed_out:
            timeouts += 1
            unsolved += 1
            continue
        if attempt.ambiguity is not None or attempt.unproven is not None:
            # The tranche set is right but the payments do not partition
            # uniquely across it. Refused for the same reason T2 refuses: two
            # partitions fitting is a coin flip, not an answer.
            ambiguous += 1
            continue
        if attempt.failed or not attempt.assignments:
            unsolved += 1
            continue

        merchant = context.merchant_of(view)
        profile = profiles[merchant] if merchant is not None else None
        grounding: tuple[Evidence, ...] = (
            _tranche_set_evidence(view, tranches, search),
        )
        if profile is not None:
            grounding += (_lexical_grounding_evidence(view, tranches, profile),)
        conservation = _conservation_evidence(view, tuple(attempt.assignments))

        for assignment in attempt.assignments:
            candidates.append(
                _settlement_candidate(
                    view,
                    assignment,
                    probability=1.0,
                    verified=True,
                    epsilon=epsilon,
                    extra=(conservation, *grounding),
                )
            )
            candidates.extend(
                _payment_candidates(
                    view,
                    assignment,
                    probability=1.0,
                    verified=True,
                    epsilon=epsilon,
                    extra=(conservation, *grounding),
                )
            )
            payments_matched += len(assignment.payments)
        credits_matched += len(attempt.assignments)
        context.consume(view.settlement_id, [a.credit.txn_id for a in attempt.assignments])
        resolved += 1

    return AggregationOutcome(
        candidates=tuple(candidates),
        settlements_seen=seen,
        settlements_resolved=resolved,
        settlements_ambiguous=ambiguous,
        settlements_unsolved=unsolved,
        settlements_without_key=no_pool,
        credits_matched=credits_matched,
        payments_matched=payments_matched,
        subsets_examined=examined,
        timeouts=timeouts,
    )
