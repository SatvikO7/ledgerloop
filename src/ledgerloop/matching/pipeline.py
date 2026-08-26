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
from dataclasses import dataclass, field
from datetime import datetime

from ledgerloop.config import RunConfig
from ledgerloop.eval.metrics import PredictedLink
from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.matching.bank_leg import BankLegOutcome, allocated_share_minor
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.policy import decide_all
from ledgerloop.matching.tier0_exact import OrderLegOutcome, run_tier0
from ledgerloop.matching.tier1_tolerance import run_tier1
from ledgerloop.matching.tier2_aggregation import AggregationOutcome, run_tier2
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, LinkType, Tier
from ledgerloop.models.metrics import TierContribution
from ledgerloop.state import ReconState

__all__ = ["MATCHER_DESCRIPTION", "MATCHER_NAME", "MatchRun", "run_matching"]

MATCHER_NAME = "T0+T1+T2"
MATCHER_DESCRIPTION = (
    "Exact key, tolerance, then aggregation (LedgerLoop deterministic tiers)"
)

#: The tiers this step implements. T3-T5 are later steps and are absent rather
#: than present-and-empty, so the report cannot show a zero for an unbuilt tier.
IMPLEMENTED_TIERS: tuple[Tier, ...] = (
    Tier.T0_EXACT,
    Tier.T1_TOLERANCE,
    Tier.T2_AGGREGATION,
)


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
    decisions: tuple[MatchDecision, ...],
    elapsed_by_tier: dict[Tier, int],
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
    """
    proposed: dict[Tier, int] = dict.fromkeys(IMPLEMENTED_TIERS, 0)
    matched: dict[Tier, int] = dict.fromkeys(IMPLEMENTED_TIERS, 0)

    proposed[Tier.T0_EXACT] += len(order_leg.candidates)
    for outcome in outcomes:
        proposed[outcome.tier] += len(outcome.candidates)
    proposed[Tier.T2_AGGREGATION] += len(aggregation.candidates)
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
        for tier in IMPLEMENTED_TIERS
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


def run_matching(
    ingest: IngestResult,
    config: RunConfig,
    *,
    decided_at: datetime | None = None,
) -> MatchRun:
    """Run T0 then T1 over an ingested dataset.

    ``decided_at`` stamps every decision in the run. One timestamp for the whole
    run rather than one per decision: the ordering that matters for replay is
    the audit sequence, not the clock, and a single stamp makes a run's decision
    log byte-comparable to a rerun of the same data.
    """
    started_ns = time.perf_counter_ns()
    stamp = decided_at or datetime.now()

    context = MatchContext.from_ingest(ingest)

    t0_started = time.perf_counter_ns()
    order_leg, t0_bank = run_tier0(context)
    t0_ms = (time.perf_counter_ns() - t0_started) // 1_000_000

    t1_started = time.perf_counter_ns()
    t1_bank = run_tier1(context, config.tolerances)
    t1_ms = (time.perf_counter_ns() - t1_started) // 1_000_000

    t2_started = time.perf_counter_ns()
    aggregation = run_tier2(context, config.tolerances)
    t2_ms = (time.perf_counter_ns() - t2_started) // 1_000_000

    bank_legs = (t0_bank, t1_bank)
    candidates: list[MatchCandidate] = [
        *order_leg.candidates,
        *t0_bank.candidates,
        *t1_bank.candidates,
        *aggregation.candidates,
    ]
    decisions = decide_all(candidates, config.thresholds, decided_at=stamp)

    state = ReconState(run_id=config.run_id, config=config)
    state.raw = ingest.raw_by_source
    state.normalized = ingest.normalized
    state.candidates = candidates
    state.decisions = list(decisions)

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000

    return MatchRun(
        name=MATCHER_NAME,
        description=MATCHER_DESCRIPTION,
        state=state,
        predictions=_predictions(decisions, by_id),
        tier_contributions=_tier_contributions(
            bank_legs,
            order_leg,
            aggregation,
            decisions,
            {
                Tier.T0_EXACT: t0_ms,
                Tier.T1_TOLERANCE: t1_ms,
                Tier.T2_AGGREGATION: t2_ms,
            },
        ),
        wall_clock_ms=int(elapsed_ms),
        order_leg=order_leg,
        bank_legs=bank_legs,
        aggregation=aggregation,
        credits_seen=len(context.credits),
        credits_with_utr=context.credits_with_utr,
        settlements_seen=len(context.settlements),
        settlements_with_utr=context.settlements_with_utr,
        settlements_resolved=sum(leg.resolved_settlements for leg in bank_legs)
        + aggregation.settlements_resolved,
        settlements_contested=sum(leg.contested_settlements for leg in bank_legs)
        + aggregation.settlements_ambiguous,
        settlements_unresolved=len(context.settlements)
        - len(context.consumed_settlements),
    )
