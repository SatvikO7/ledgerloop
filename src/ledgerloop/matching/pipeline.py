"""Run the tier ladder over one ingested dataset.

Ingest in, :class:`~ledgerloop.state.ReconState` and predictions out. Two tiers
today; the shape does not change when T2-T5 arrive, because the pool in
:mod:`ledgerloop.matching.context` already carries the residual between them.

WHY CANDIDATE YIELD AND AUTO-MATCH RATE ARE REPORTED SEPARATELY
----------------------------------------------------------------
They answer different questions and a single number conflates them:

* **Candidate yield** -- what the tier *found*. It measures reach.
* **Auto-match** -- what survived the decision policy. It measures conviction.

A tier that proposes a hundred candidates and auto-matches forty has not
performed like a tier that proposes forty and auto-matches forty, and the
difference is exactly the review queue a finance team has to staff. Reporting
one figure would let a tier hide either an over-eager proposal stage or an
over-cautious policy behind the other. :class:`~ledgerloop.models.metrics.
TierContribution` has carried both fields since Step 0; this is the step that
populates them.

WHAT FEEDS THE EVALUATOR
------------------------
Only ``AUTO_MATCHED`` decisions on ``PAYMENT_CREDITED_AS`` links become
:class:`~ledgerloop.eval.metrics.PredictedLink` records. Both filters matter:

* ``NEEDS_REVIEW`` and ``EXCEPTION`` are not predictions. Counting a referral
  as a match is the precision-inflating trap.
* ``ORDER_PAID_BY`` and ``SETTLEMENT_CREDITED_AS`` are structural or
  intermediate edges, excluded from the metrics by ``ARCHITECTURE.md`` §2. They
  are matched, decided and audited -- they simply are not scored.

The asserted amount on each predicted link is the payment's **allocated share**
of the credit, not its gross. That is the same largest-remainder allocation the
generator used to build the truth links, so a correct match asserts exactly the
right rupee figure -- unlike B0, whose reconciled total runs above the truth
even where its links are right.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ledgerloop.config import RunConfig
from ledgerloop.eval.metrics import PredictedLink
from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.matching.bank_leg import BankLegOutcome, allocated_share_minor
from ledgerloop.matching.calibration import BlendOutcome, CalibrationBundle, apply_bundle
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.policy import decide_all
from ledgerloop.matching.tier0_exact import OrderLegOutcome, run_tier0
from ledgerloop.matching.tier1_tolerance import run_tier1
from ledgerloop.matching.tier2_aggregation import AggregationOutcome, run_tier2
from ledgerloop.matching.tier3_lexical import LexicalOutcome, build_profiles, run_tier3
from ledgerloop.matching.tier4_graph import GraphOutcome, run_tier4
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, LinkType, Tier
from ledgerloop.models.metrics import TierContribution
from ledgerloop.models.refs import bank_ref
from ledgerloop.state import ReconState

__all__ = [
    "MATCHER_DESCRIPTION",
    "MATCHER_NAME",
    "TIER_BY_INDEX",
    "MatchRun",
    "ResidualAdjudicator",
    "ladder_description",
    "ladder_name",
    "run_matching",
]


class ResidualAdjudicator(Protocol):
    """T5's shape, declared where the ladder can see it and nowhere else.

    The LLM tier is injected rather than imported. ``matching`` must not depend
    on :mod:`ledgerloop.llm`, because the moment it does, ``--no-llm`` stops
    being the same code path with one branch taken and starts being a second
    implementation -- and a second implementation is one nobody measures.

    An adjudicator returns *candidates*. It cannot decide, cannot set a
    probability, and cannot bypass ``verify_arithmetic``: everything it returns
    goes through the same policy as T2's, and the
    :class:`~ledgerloop.models.decisions.MatchDecision` validator refuses to
    auto-match anything whose arithmetic did not close.
    """

    def __call__(
        self, context: MatchContext, established: Sequence[MatchCandidate]
    ) -> tuple[Sequence[MatchCandidate], object]: ...

MATCHER_NAME = "T0-T4"
MATCHER_DESCRIPTION = (
    "Exact key, tolerance, aggregation, lexical, graph "
    "(LedgerLoop deterministic tiers)"
)

#: The tiers this step implements. T5 is a later step and is absent rather than
#: present-and-empty, so the report cannot show a zero for an unbuilt tier.
IMPLEMENTED_TIERS: tuple[Tier, ...] = (
    Tier.T0_EXACT,
    Tier.T1_TOLERANCE,
    Tier.T2_AGGREGATION,
    Tier.T3_FUZZY,
    Tier.T4_GRAPH,
)

#: ``RunConfig.enabled_tiers`` index -> the tier it switches off.
#:
#: The mapping is explicit rather than derived from ``Tier``'s ordinal so that
#: renumbering the enum cannot silently repoint an ablation row at a different
#: tier -- the row labels in ``EVALUATION.md`` would still say ``T0-T2`` while
#: measuring something else.
TIER_BY_INDEX: dict[int, Tier] = {
    0: Tier.T0_EXACT,
    1: Tier.T1_TOLERANCE,
    2: Tier.T2_AGGREGATION,
    3: Tier.T3_FUZZY,
    4: Tier.T4_GRAPH,
    5: Tier.T5_LLM,
}


def ladder_name(enabled_tiers: Sequence[int]) -> str:
    """``T0``, ``T0-T2``, ``T0-T5`` -- what the run actually ran.

    A contiguous range renders as its endpoints and anything else lists its
    members, because an ablation row labelled ``T0-T4`` that skipped T2 would
    be a mislabelled measurement rather than a terse one.
    """
    tiers = tuple(enabled_tiers)
    if not tiers:  # pragma: no cover - RunConfig refuses an empty ladder
        return "none"
    contiguous = tuple(range(tiers[0], tiers[-1] + 1)) == tiers
    if not contiguous:
        return "+".join(f"T{index}" for index in tiers)
    if len(tiers) == 1:
        return f"T{tiers[0]}"
    return f"T{tiers[0]}-T{tiers[-1]}"


def ladder_description(enabled_tiers: Sequence[int]) -> str:
    """The human-readable name of every tier that ran, in ladder order."""
    labels = {
        0: "exact key",
        1: "tolerance",
        2: "aggregation",
        3: "lexical",
        4: "graph",
        5: "LLM adjudication",
    }
    named = ", ".join(labels[index] for index in enabled_tiers if index in labels)
    return f"{named} (LedgerLoop {ladder_name(enabled_tiers)})"


@dataclass(frozen=True)
class MatchRun:
    """One matching run: everything decided, plus why it scored that way."""

    name: str
    description: str
    state: ReconState
    predictions: tuple[PredictedLink, ...]
    tier_contributions: tuple[TierContribution, ...]
    wall_clock_ms: int

    order_leg: OrderLegOutcome
    bank_legs: tuple[BankLegOutcome, ...] = field(default=())
    aggregation: AggregationOutcome = field(default_factory=AggregationOutcome)
    lexical: LexicalOutcome = field(default_factory=LexicalOutcome)
    graph: GraphOutcome = field(default_factory=GraphOutcome)
    passes: int = 1
    blend: BlendOutcome = field(default_factory=BlendOutcome)
    calibrated: bool = False
    adjudication: object | None = None
    """Whatever T5 returned beside its candidates, for the report to render.

    Typed as ``object`` on purpose: the ladder does not know what an LLM outcome
    looks like and must not learn. The CLI, which built the adjudicator, is what
    narrows it again.
    """

    #: The indexes and residual pool the run finished with. Carried so Step 8's
    #: classifier can see *what was left* rather than rebuilding a pool that
    #: never went through the ladder -- a rebuilt context would show every
    #: settlement as open and every credit as unclaimed.
    context: MatchContext | None = None
    #: The narration spellings T3 learned from the statement's own references.
    #: What separates "this credit lost its reference" from "this credit is from
    #: outside the ledger".
    merchant_spellings: frozenset[str] = field(default_factory=frozenset)

    @property
    def out_of_scope_refs(self) -> frozenset[str]:
        """Records the exception queue does not cover: the outgoing rows.

        A debit is money leaving the account, not a payout being reconciled.
        Exposed as a set rather than a count so the evaluator can exclude
        exactly these records from its denominators instead of subtracting a
        number and hoping the two agree.
        """
        if self.context is None:  # pragma: no cover - always set by run_matching
            return frozenset()
        return frozenset(
            bank_ref(txn.txn_id).key
            for txn in self.context.bank_txns
            if not txn.is_credit
        )

    credits_seen: int = 0
    credits_with_utr: int = 0
    settlements_seen: int = 0
    settlements_with_utr: int = 0
    settlements_resolved: int = 0
    settlements_contested: int = 0
    settlements_unresolved: int = 0

    # -- shapes the report shares with BaselineRun -----------------------

    @property
    def credits_without_utr(self) -> int:
        """Credits carrying no recoverable reference. A07, plus the noise rows."""
        return self.credits_seen - self.credits_with_utr

    @property
    def credits_joined(self) -> int:
        return (
            sum(leg.resolved_credits for leg in self.bank_legs)
            + self.aggregation.credits_matched
            + self.lexical.credits_matched
        )

    @property
    def settlements_joined(self) -> int:
        return self.settlements_resolved

    @property
    def settlements_unjoined(self) -> int:
        return self.settlements_seen - self.settlements_resolved

    # -- candidate yield versus conviction --------------------------------

    @property
    def candidates(self) -> tuple[MatchCandidate, ...]:
        return tuple(self.state.candidates)

    @property
    def decisions(self) -> tuple[MatchDecision, ...]:
        return tuple(self.state.decisions)

    def decisions_with(self, outcome: DecisionOutcome) -> tuple[MatchDecision, ...]:
        return tuple(d for d in self.state.decisions if d.outcome is outcome)

    @property
    def auto_matched(self) -> int:
        return len(self.decisions_with(DecisionOutcome.AUTO_MATCHED))

    @property
    def needs_review(self) -> int:
        return len(self.decisions_with(DecisionOutcome.NEEDS_REVIEW))

    @property
    def exceptions(self) -> int:
        return len(self.decisions_with(DecisionOutcome.EXCEPTION))

    @property
    def evaluable_candidates(self) -> int:
        """Candidates on the scored link type, before the policy ruled."""
        return sum(1 for candidate in self.state.candidates if candidate.is_evaluable)


def _tier_contributions(
    outcomes: tuple[BankLegOutcome, ...],
    order_leg: OrderLegOutcome,
    aggregation: AggregationOutcome,
    lexical: LexicalOutcome,
    graph: GraphOutcome,
    decisions: tuple[MatchDecision, ...],
    elapsed_by_tier: dict[Tier, int],
    enabled: frozenset[Tier],
    llm_candidates: tuple[MatchCandidate, ...] = (),
    llm_ran: bool = False,
) -> tuple[TierContribution, ...]:
    """One row per implemented tier: proposed, auto-matched, and marginal.

    ``marginal_auto_matched`` is what this tier added *over the ladder without
    it*. For a strictly residual ladder -- each tier only sees what the previous
    ones left in the pool -- every auto-match a tier produces is by construction
    an auto-match no earlier tier could have made, so marginal equals total.
    That equality is a property of the pool, not a shortcut, and it stops
    holding at T4, whose re-run loop can revisit earlier tiers' residue.

    T2 depends on it twice over: it only ever sees settlements T0 and T1 left
    *undecided*, and the splits it exists for are exactly the ones they declined
    to consume.

    ``enabled`` restricts the table to the tiers this run actually ran. An
    ablation row that switched T3 and T4 off must not print a T3 row at zero:
    a zero for a tier that did not run is a false measurement, which is the
    same rule the absent-T5-row case already followed.
    """
    ran = (*IMPLEMENTED_TIERS, Tier.T5_LLM) if llm_ran else IMPLEMENTED_TIERS
    tiers = tuple(tier for tier in ran if tier in enabled)

    # Tallied over every tier, then projected onto the ones that ran. A disabled
    # tier contributes nothing by construction, so the two orders agree -- but
    # accumulating into a dict keyed only on the enabled tiers would raise on
    # the first candidate from a tier the caller switched off, which is a
    # crash rather than a measurement.
    proposed: dict[Tier, int] = dict.fromkeys(ran, 0)
    matched: dict[Tier, int] = dict.fromkeys(ran, 0)

    proposed[Tier.T0_EXACT] += len(order_leg.candidates)
    for outcome in outcomes:
        proposed[outcome.tier] += len(outcome.candidates)
    proposed[Tier.T2_AGGREGATION] += len(aggregation.candidates)
    proposed[Tier.T3_FUZZY] += len(lexical.candidates)
    proposed[Tier.T4_GRAPH] += len(graph.candidates)
    if llm_ran:
        proposed[Tier.T5_LLM] += len(llm_candidates)
    for decision in decisions:
        if decision.outcome is DecisionOutcome.AUTO_MATCHED:
            matched[decision.tier] += 1

    return tuple(
        TierContribution(
            tier=tier,
            candidates_proposed=proposed[tier],
            auto_matched=matched[tier],
            marginal_auto_matched=matched[tier],
            llm_calls=0,
            wall_clock_ms=elapsed_by_tier.get(tier, 0),
        )
        for tier in tiers
    )


def _predictions(
    decisions: tuple[MatchDecision, ...], candidates: dict[str, MatchCandidate]
) -> tuple[PredictedLink, ...]:
    """Turn auto-matched evaluation links into the evaluator's input contract."""
    links: list[PredictedLink] = []
    for decision in decisions:
        if not decision.is_positive_prediction:
            continue
        if decision.link_type is not LinkType.PAYMENT_CREDITED_AS:
            continue
        candidate = candidates[decision.candidate_id]
        links.append(
            PredictedLink(
                source_ref=decision.source_ref,
                target_ref=decision.target_ref,
                amount_minor=allocated_share_minor(candidate),
            )
        )
    return tuple(links)


def _merge_aggregation(
    left: AggregationOutcome, right: AggregationOutcome
) -> AggregationOutcome:
    """Accumulate two T2 passes. Counters add; candidates concatenate."""
    return AggregationOutcome(
        candidates=left.candidates + right.candidates,
        settlements_seen=left.settlements_seen + right.settlements_seen,
        settlements_resolved=left.settlements_resolved + right.settlements_resolved,
        settlements_ambiguous=left.settlements_ambiguous + right.settlements_ambiguous,
        settlements_unsolved=right.settlements_unsolved,
        settlements_without_key=right.settlements_without_key,
        credits_matched=left.credits_matched + right.credits_matched,
        payments_matched=left.payments_matched + right.payments_matched,
        subsets_examined=left.subsets_examined + right.subsets_examined,
        greedy_fallbacks=left.greedy_fallbacks + right.greedy_fallbacks,
        timeouts=left.timeouts + right.timeouts,
    )


def _merge_lexical(left: LexicalOutcome, right: LexicalOutcome) -> LexicalOutcome:
    """Accumulate two T3 passes.

    Totals add; the *residual* counts (unsolved, without-profile) are taken
    from the last pass, because they describe what is still open rather than
    what happened -- summing them would double-count a settlement that stayed
    open across passes.
    """
    return LexicalOutcome(
        candidates=left.candidates + right.candidates,
        profiles_built=right.profiles_built,
        profile_witnesses=right.profile_witnesses,
        settlements_seen=left.settlements_seen + right.settlements_seen,
        settlements_resolved=left.settlements_resolved + right.settlements_resolved,
        settlements_ambiguous=left.settlements_ambiguous + right.settlements_ambiguous,
        settlements_unsolved=right.settlements_unsolved,
        settlements_without_profile=right.settlements_without_profile,
        credits_matched=left.credits_matched + right.credits_matched,
        payments_matched=left.payments_matched + right.payments_matched,
        names_scored=left.names_scored + right.names_scored,
        rejected_below_score=left.rejected_below_score + right.rejected_below_score,
        rejected_on_margin=left.rejected_on_margin + right.rejected_on_margin,
    )


def run_matching(
    ingest: IngestResult,
    config: RunConfig,
    *,
    decided_at: datetime | None = None,
    bundle: CalibrationBundle | None = None,
    adjudicator: ResidualAdjudicator | None = None,
) -> MatchRun:
    """Run the tier ladder over an ingested dataset.

    ``decided_at`` stamps every decision in the run. One timestamp for the whole
    run rather than one per decision: the ordering that matters for replay is
    the audit sequence, not the clock, and a single stamp makes a run's decision
    log byte-comparable to a rerun of the same data.

    ``adjudicator`` is T5, injected by the caller that owns the model. Absent --
    which is every ``--no-llm`` run -- the ladder is exactly the deterministic
    T0-T4 it was at Step 8, down to the tier table having no T5 row: a zero for
    a tier that did not run is a false measurement.

    ``bundle`` is the fitted blender, isotonic calibrator and threshold from
    Step 7. Without it the residual tiers keep the provisional probabilities
    they set themselves -- which is what every step up to Step 6 measured, and
    what the ``--no-calibration`` ablation row still measures. With it, each
    residual pass is scored **as it is produced**, before T4 reads the pass as
    its premise set: T4 admits only premises at or above ``tau_high``, so
    calibrating afterwards would let it infer from links the policy was about to
    refuse. The threshold itself comes from ``config.thresholds``, which
    :func:`~ledgerloop.matching.calibration.configure_for` fills from the same
    bundle -- so a fitted threshold is inside ``config_hash`` rather than
    beside it.

    ``config.enabled_tiers`` is what the **ablation** turns: a tier not listed
    does not run, contributes no candidates, and gets **no row in the tier
    table** rather than a row of zeros. The field has carried that description
    since Step 0 and Step 10 is where the ladder starts reading it.

    Switching a tier off is not the same as it finding nothing. Every tier here
    consumes from the shared pool, so removing T1 leaves its settlements
    *undecided* and T2 sees them -- which is precisely the marginal contribution
    an ablation row is asking about. That is why the rows are produced by
    re-running the ladder rather than by subtracting tier counters from a single
    full run.
    """
    started_ns = time.perf_counter_ns()
    stamp = decided_at or datetime.now()
    enabled = frozenset(config.enabled_tiers)
    enabled_tiers = frozenset(
        TIER_BY_INDEX[index] for index in config.enabled_tiers if index in TIER_BY_INDEX
    )

    context = MatchContext.from_ingest(ingest)

    t0_started = time.perf_counter_ns()
    if 0 in enabled:
        order_leg, t0_bank = run_tier0(context)
    else:
        order_leg = OrderLegOutcome(candidates=())
        t0_bank = BankLegOutcome(tier=Tier.T0_EXACT, candidates=())
    t0_ms = (time.perf_counter_ns() - t0_started) // 1_000_000

    t1_started = time.perf_counter_ns()
    t1_bank = (
        run_tier1(context, config.tolerances)
        if 1 in enabled
        else BankLegOutcome(tier=Tier.T1_TOLERANCE, candidates=())
    )
    t1_ms = (time.perf_counter_ns() - t1_started) // 1_000_000

    # The residual loop (PLAN.md 6.1). T0 and T1 are exact and run once --
    # nothing a later tier does can unlock a key that was not there. T2, T3 and
    # T4 each consume records the others may have been waiting on, so they
    # repeat until a pass changes nothing or the configured cap is reached.
    #
    # The merchant master is built once: it is derived from the *references* in
    # the statement, which no amount of matching alters.
    profiles = build_profiles(context)
    aggregation = AggregationOutcome()
    lexical = LexicalOutcome()
    graph = GraphOutcome()
    residual: list[MatchCandidate] = []
    t2_ms = t3_ms = t4_ms = 0
    passes = 0

    # The loop exists to let T2/T3/T4 unlock each other. With none of them
    # enabled there is nothing to iterate, and running an empty pass would
    # report `passes = 1` for a ladder that has no residual stage at all.
    residual_passes = (
        config.graph.max_rerun_passes if enabled & {2, 3, 4} else 0
    )
    for _ in range(residual_passes):
        passes += 1
        before = len(residual)

        started = time.perf_counter_ns()
        pass_aggregation = (
            run_tier2(context, config.tolerances) if 2 in enabled else AggregationOutcome()
        )
        t2_ms += (time.perf_counter_ns() - started) // 1_000_000

        started = time.perf_counter_ns()
        pass_lexical = (
            run_tier3(context, config.tolerances, config.lexical, profiles=profiles)
            if 3 in enabled
            else LexicalOutcome()
        )
        t3_ms += (time.perf_counter_ns() - started) // 1_000_000

        if bundle is not None:
            # Scored here, before T4 reads this pass as its premise set: T4
            # admits only premises at or above tau_high, so calibrating
            # afterwards would let it infer from links the policy was about to
            # refuse. The authoritative count is taken once at the end.
            apply_bundle((*pass_aggregation.candidates, *pass_lexical.candidates), bundle)

        established = (
            *order_leg.candidates,
            *t0_bank.candidates,
            *t1_bank.candidates,
            *residual,
            *pass_aggregation.candidates,
            *pass_lexical.candidates,
        )
        started = time.perf_counter_ns()
        pass_graph = (
            run_tier4(context, established, config.graph, config.thresholds)
            if 4 in enabled
            else GraphOutcome()
        )
        t4_ms += (time.perf_counter_ns() - started) // 1_000_000

        if bundle is not None:
            apply_bundle(pass_graph.candidates, bundle)

        residual.extend(pass_aggregation.candidates)
        residual.extend(pass_lexical.candidates)
        residual.extend(pass_graph.candidates)
        aggregation = _merge_aggregation(aggregation, pass_aggregation)
        lexical = _merge_lexical(lexical, pass_lexical)
        graph = pass_graph

        if len(residual) == before:
            break

    # T5 last, and only over what everything before it left. It sees the
    # established candidates so its evidence packs describe the real residual,
    # and its output rejoins the ordinary flow: scored by the blender if one is
    # fitted, then routed by the same policy as every other tier.
    adjudication: object | None = None
    t5_ms = 0
    llm_candidates: tuple[MatchCandidate, ...] = ()
    if adjudicator is not None and 5 in enabled:
        started = time.perf_counter_ns()
        established = (
            *order_leg.candidates,
            *t0_bank.candidates,
            *t1_bank.candidates,
            *residual,
        )
        proposed, adjudication = adjudicator(context, established)
        llm_candidates = tuple(proposed)
        if bundle is not None:
            apply_bundle(llm_candidates, bundle)
        t5_ms = (time.perf_counter_ns() - started) // 1_000_000

    bank_legs = (t0_bank, t1_bank)
    candidates: list[MatchCandidate] = [
        *order_leg.candidates,
        *t0_bank.candidates,
        *t1_bank.candidates,
        *residual,
        *llm_candidates,
    ]
    # One authoritative pass over every candidate, T0 and T1 included, so the
    # reported counters describe the whole run rather than the residual passes.
    # Re-scoring an already-scored candidate is idempotent: the same features go
    # through the same fitted model.
    blend = (
        apply_bundle(candidates, bundle) if bundle is not None else BlendOutcome()
    )
    decisions = decide_all(candidates, config.thresholds, decided_at=stamp)

    state = ReconState(run_id=config.run_id, config=config)
    state.raw = ingest.raw_by_source
    state.normalized = ingest.normalized
    state.candidates = candidates
    state.decisions = list(decisions)

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000

    # The name describes what *ran*, not what was permitted. A config listing
    # T5 on a machine with no key ran T0-T4, and labelling that row `T0-T5`
    # would credit the ladder with a tier it never invoked.
    ran_tiers = tuple(
        index
        for index in config.enabled_tiers
        if index != 5 or (adjudicator is not None)
    )
    return MatchRun(
        name=ladder_name(ran_tiers),
        description=ladder_description(ran_tiers),
        state=state,
        predictions=_predictions(decisions, by_id),
        tier_contributions=_tier_contributions(
            bank_legs,
            order_leg,
            aggregation,
            lexical,
            graph,
            decisions,
            {
                Tier.T0_EXACT: t0_ms,
                Tier.T1_TOLERANCE: t1_ms,
                Tier.T2_AGGREGATION: t2_ms,
                Tier.T3_FUZZY: t3_ms,
                Tier.T4_GRAPH: t4_ms,
                Tier.T5_LLM: t5_ms,
            },
            enabled=enabled_tiers,
            llm_candidates=llm_candidates,
            llm_ran=adjudicator is not None and 5 in enabled,
        ),
        wall_clock_ms=int(elapsed_ms),
        order_leg=order_leg,
        bank_legs=bank_legs,
        aggregation=aggregation,
        lexical=lexical,
        graph=graph,
        passes=passes,
        blend=blend,
        calibrated=bundle is not None,
        adjudication=adjudication,
        context=context,
        merchant_spellings=frozenset(
            spelling for profile in profiles.values() for spelling in profile.spellings
        ),
        credits_seen=len(context.credits),
        credits_with_utr=context.credits_with_utr,
        settlements_seen=len(context.settlements),
        settlements_with_utr=context.settlements_with_utr,
        settlements_resolved=sum(leg.resolved_settlements for leg in bank_legs)
        + aggregation.settlements_resolved
        + lexical.settlements_resolved,
        settlements_contested=sum(leg.contested_settlements for leg in bank_legs)
        + aggregation.settlements_ambiguous
        + lexical.settlements_ambiguous,
        settlements_unresolved=len(context.settlements)
        - len(context.consumed_settlements),
    )
