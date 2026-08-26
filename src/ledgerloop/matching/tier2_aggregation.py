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

from dataclasses import dataclass

from ledgerloop.config import MatchingTolerances
from ledgerloop.matching.bank_leg import attribute_clawback, candidate_id
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.matching.subset_sum import (
    Accept,
    SubsetSearch,
    SubsetSolution,
    find_subsets,
)
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
    "credit_bucket",
    "expected_credit_minor",
    "payment_bucket",
    "run_tier2",
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


def _search_window(
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


def _features(assignment: Assignment, view: SettlementView, epsilon: int) -> FeatureVector:
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
        features=_features(assignment, view, epsilon),
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
    features = _features(assignment, view, epsilon)
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


def _accept_for(target: int, gross: int, net: int, epsilon: int) -> Accept:
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
    gross = view.payment_gross_minor
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
            low, high = _search_window(credit.credit_minor, gross, net, epsilon)
            search = find_subsets(
                [p.amount_minor for p in remaining],
                low,
                high,
                want=2,
                max_exact_items=tolerances.max_subset_size,
                timeout_ms=tolerances.subset_solver_timeout_ms,
                accept=_accept_for(credit.credit_minor, gross, net, epsilon),
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
        gross = view.payment_gross_minor
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
