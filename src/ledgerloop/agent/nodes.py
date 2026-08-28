"""The graph's nodes. Each one calls a function that existed before Step 11.

PLAN.md §4.3 is honest about what LangGraph buys: *"for the happy path, a
function chain would do."* So these nodes are deliberately thin. Every one of
them is a call into a tested module plus the audit events that call deserves —
and there is **no reconciliation logic in this file at all**. Grep it for
arithmetic and you will find none; grep it for a threshold comparison and you
will find none. That is the property that keeps the graph from becoming a
second implementation of the system it orchestrates.

WHERE THE PLAN'S BOXES MAP
--------------------------
PLAN.md §4.1 draws nine boxes. Eight map one-to-one; one is merged, and the
merge is stated rather than hidden:

===============================  ===================================
PLAN.md §4.1                     node here
===============================  ===================================
``ingest_sources``               :func:`ingest_sources`
``normalize_records``            :func:`normalize_records`
``build_entity_graph``           :func:`build_entity_graph`
``tier_ladder`` (looping)        :func:`tier_ladder` + the cycle edge
``llm_adjudicate``               :func:`llm_adjudicate`
``calibrate_confidence``         }  :func:`calibrate_and_decide`
``apply_decision_policy``        }
``classify_exceptions``          :func:`classify_exceptions_node`
``generate_report``              :func:`generate_report`
===============================  ===================================

The blend and the policy are one node because
:func:`~ledgerloop.matching.pipeline.close_ladder` does both and splitting it
would mean a second assembly path for :class:`~ledgerloop.matching.pipeline.
MatchRun` — two ways to build the object every metric is computed from. The
node's audit events report the two stages separately, so the merge costs
nothing in the replay.

``explain_exceptions`` is a tenth node the plan does not draw. It is LLM call
site 3 (PLAN.md §7.2), and it is separate from classification precisely because
the class, the severity and the money are settled before it runs and are not
sent back for revision.
"""

from __future__ import annotations

import time

from ledgerloop.agent.audit import AuditLog
from ledgerloop.agent.state import GraphState, RunResources
from ledgerloop.eval.harness import assemble_system_run
from ledgerloop.eval.reliability import measure_calibration, score_contenders
from ledgerloop.exceptions import classify_exceptions, mark_resolvable, resolve_bounded
from ledgerloop.ingest import ingest_dataset
from ledgerloop.llm.integration import LLMRunSummary, explain_queue, repair_narrations
from ledgerloop.llm.tasks import AdjudicationOutcome, ExplanationOutcome, NarrationOutcome
from ledgerloop.matching.harvest import harvest
from ledgerloop.matching.pipeline import (
    adjudicate_residual,
    close_ladder,
    open_ladder,
    run_residual_pass,
    should_run_residual_pass,
)
from ledgerloop.models.audit import AuditEventType
from ledgerloop.models.metrics import CostLedger

__all__ = [
    "NODE_SEQUENCE",
    "build_entity_graph",
    "calibrate_and_decide",
    "classify_exceptions_node",
    "explain_exceptions",
    "generate_report",
    "ingest_sources",
    "llm_adjudicate",
    "measure_quality",
    "normalize_records",
    "record_failure",
    "should_loop",
    "tier_ladder",
]

#: Node names in the order the happy path visits them, ``tier_ladder`` once.
#:
#: Used by the tests to assert the graph's topology matches the documented one,
#: and by the UI to render a progress list. Declared as data so the two cannot
#: disagree with the builder.
NODE_SEQUENCE: tuple[str, ...] = (
    "ingest_sources",
    "normalize_records",
    "build_entity_graph",
    "tier_ladder",
    "llm_adjudicate",
    "calibrate_and_decide",
    "classify_exceptions",
    "explain_exceptions",
    "measure_quality",
    "generate_report",
)


def _log(state: GraphState) -> AuditLog:
    """A log positioned to continue the run's existing sequence.

    Rebuilt per node rather than threaded, because LangGraph owns the state
    between nodes and a mutable object living across them would be the hidden
    global PLAN.md §4.2 forbids. The counter is recovered from what is already
    in the state, so the total order survives the round trip.
    """
    log = AuditLog(run_id=state.get("run_id", "unknown"))
    log._sequence = len(state.get("audit", []))
    return log


def _entered(log: AuditLog, node: str) -> int:
    log.emit(AuditEventType.NODE_ENTERED, node)
    return time.perf_counter_ns()


def _completed(
    log: AuditLog, node: str, started_ns: int, message: str = "", **payload: object
) -> None:
    log.emit(
        AuditEventType.NODE_COMPLETED,
        node,
        message=message,
        payload=dict(payload),
        latency_ms=(time.perf_counter_ns() - started_ns) // 1_000_000,
    )


def ingest_sources(state: GraphState, resources: RunResources) -> GraphState:
    """Parse the three sources into canonical records.

    ``strict=False``: a malformed row is quarantined and counted rather than
    taking the run down. The count reaches the report; the row does not reach
    the matcher.
    """
    log = _log(state)
    started = _entered(log, "ingest_sources")
    ingest = ingest_dataset(resources.setup.directory, strict=False)
    _completed(
        log,
        "ingest_sources",
        started,
        message=(
            f"{len(ingest.orders)} orders, {len(ingest.payments)} payments, "
            f"{len(ingest.settlements)} settlements, {len(ingest.bank_txns)} bank rows"
        ),
        orders=len(ingest.orders),
        payments=len(ingest.payments),
        settlements=len(ingest.settlements),
        bank_txns=len(ingest.bank_txns),
        quarantined=len(ingest.problems),
        date_basis=ingest.date_order.basis,
    )
    return GraphState(
        ingest=ingest, node_log=["ingest_sources"], audit=list(log.events)
    )


def normalize_records(state: GraphState, resources: RunResources) -> GraphState:
    """LLM call site 1: re-read the narrations the regex layer could not.

    Gated on T5 as well as on the client. An ablation row that ran the
    deterministic ladder while a model quietly repaired its inputs would credit
    the deterministic tiers with the model's contribution.

    An accepted repair only ever *fills a gap* -- it cannot overwrite a
    reference the regex layer read, and it cannot touch an amount or a date.
    """
    log = _log(state)
    started = _entered(log, "normalize_records")
    ingest = state["ingest"]
    outcome = NarrationOutcome()
    if resources.llm_active and resources.client is not None:
        ingest, outcome = repair_narrations(
            resources.client,
            ingest,
            batch_size=resources.setup.config.llm.narration_batch_size,
        )
        for repair in outcome.repairs:
            log.emit(
                AuditEventType.LLM_CALL,
                "normalize_records",
                message=f"narration repaired for {repair.item_id}",
                payload={
                    "utr": repair.utr,
                    "merchant": repair.merchant,
                    "model_confidence": repair.confidence,
                },
            )
    _completed(
        log,
        "normalize_records",
        started,
        message=(
            f"{outcome.accepted} repaired of {outcome.attempted} attempted"
            if resources.llm_active
            else "deterministic: no model in this run"
        ),
        attempted=outcome.attempted,
        accepted=outcome.accepted,
        rejected_ungrounded=outcome.rejected_ungrounded,
    )
    return GraphState(
        ingest=ingest,
        narration=outcome,
        node_log=["normalize_records"],
        audit=list(log.events),
    )


def build_entity_graph(state: GraphState, resources: RunResources) -> GraphState:
    """Index the corpus, run the exact tiers, learn the merchant master.

    T0 and T1 sit here rather than in :func:`tier_ladder` because they are
    exact and run **once**: nothing a later tier does can unlock a key that was
    not in the source. Only T2/T3/T4 are inside the loop, which is what makes
    the loop's cycle meaningful rather than decorative.
    """
    log = _log(state)
    started = _entered(log, "build_entity_graph")
    ladder = open_ladder(
        state["ingest"], resources.setup.config, bundle=resources.bundle
    )
    exact = len(ladder.order_leg.candidates) + len(ladder.t0_bank.candidates)
    _completed(
        log,
        "build_entity_graph",
        started,
        message=(
            f"{len(ladder.context.settlements)} settlements, "
            f"{len(ladder.context.credits)} credits, "
            f"{len(ladder.profiles)} merchant profiles"
        ),
        t0_candidates=exact,
        t1_candidates=len(ladder.t1_bank.candidates),
        merchant_profiles=len(ladder.profiles),
        residual_cap=ladder.residual_cap,
    )
    return GraphState(
        ladder=ladder, node_log=["build_entity_graph"], audit=list(log.events)
    )


def tier_ladder(state: GraphState, resources: RunResources) -> GraphState:
    """One residual pass: T2 → T3 → T4 over what the ladder has left.

    **This node is the cycle.** :func:`should_loop` sends the graph back here
    while a pass keeps adding candidates, which is the behaviour PLAN.md §4.3
    names as LangGraph's justification: resolving one bank credit constrains the
    remaining ones, so a later pass can match what an earlier one could not.
    """
    del resources
    log = _log(state)
    started = _entered(log, "tier_ladder")
    ladder = state["ladder"]
    run_residual_pass(ladder)
    _completed(
        log,
        "tier_ladder",
        started,
        message=(
            f"pass {ladder.passes}: {ladder.last_pass_added} new candidate(s), "
            f"{len(ladder.residual)} residual in all"
        ),
        pass_number=ladder.passes,
        added=ladder.last_pass_added,
        residual_total=len(ladder.residual),
        aggregation_resolved=ladder.aggregation.settlements_resolved,
        aggregation_ambiguous=ladder.aggregation.settlements_ambiguous,
        lexical_resolved=ladder.lexical.settlements_resolved,
        graph_candidates=len(ladder.graph.candidates),
    )
    return GraphState(
        ladder=ladder,
        passes=ladder.passes,
        node_log=["tier_ladder"],
        audit=list(log.events),
    )


def should_loop(state: GraphState) -> str:
    """The conditional edge. ``tier_ladder`` again, or on to T5.

    Reads :func:`~ledgerloop.matching.pipeline.should_run_residual_pass` rather
    than reimplementing the condition, so the graph's cycle and the direct
    path's ``while`` are the same predicate. A run that took two passes through
    the CLI takes two passes through the graph, and there is a test that says so.
    """
    ladder = state.get("ladder")
    if ladder is None:  # pragma: no cover - build_entity_graph always sets it
        return "llm_adjudicate"
    return "tier_ladder" if should_run_residual_pass(ladder) else "llm_adjudicate"


def llm_adjudicate(state: GraphState, resources: RunResources) -> GraphState:
    """T5, over what everything before it left. It proposes; it never decides.

    Three gates stand between the model and a candidate and all three live
    outside this file: the schema, the grounding check against the evidence
    pack, and ``verify_arithmetic``. A proposal whose money does not close is
    **demoted, not dropped** -- it becomes a candidate the policy routes to a
    human, because "the model suggested this and the arithmetic disagrees" is
    information a controller wants.
    """
    log = _log(state)
    started = _entered(log, "llm_adjudicate")
    ladder = state["ladder"]
    adjudicate_residual(ladder, resources.adjudicator if resources.llm_active else None)

    outcome = ladder.adjudication
    if not isinstance(outcome, AdjudicationOutcome):
        outcome = AdjudicationOutcome()
    for candidate in ladder.llm_candidates:
        log.emit(
            AuditEventType.CANDIDATE_PROPOSED,
            "llm_adjudicate",
            message="T5 proposal, pending arithmetic verification",
            candidate_id=candidate.candidate_id,
            refs=(candidate.source_ref, candidate.target_ref),
            payload={
                "arithmetic_verified": candidate.arithmetic_verified,
                "model_confidence": candidate.features.llm_confidence,
            },
        )
        log.emit(
            AuditEventType.ARITHMETIC_VERIFIED
            if candidate.arithmetic_verified
            else AuditEventType.ARITHMETIC_FAILED,
            "llm_adjudicate",
            message=(
                "money re-derived from the sources and it closes"
                if candidate.arithmetic_verified
                else "money re-derived from the sources and it does not close; demoted"
            ),
            candidate_id=candidate.candidate_id,
        )
    _completed(
        log,
        "llm_adjudicate",
        started,
        message=(
            f"{len(ladder.llm_candidates)} proposal(s), {outcome.demoted} demoted"
            if ladder.adjudicator_ran
            else "skipped: no model in this run"
        ),
        ran=ladder.adjudicator_ran,
        proposals=len(ladder.llm_candidates),
        demoted=outcome.demoted,
        rejected_ungrounded=outcome.rejected_ungrounded,
    )
    return GraphState(
        ladder=ladder,
        adjudication=outcome,
        node_log=["llm_adjudicate"],
        audit=list(log.events),
    )


def calibrate_and_decide(state: GraphState, resources: RunResources) -> GraphState:
    """PLAN.md's ``calibrate_confidence`` and ``apply_decision_policy``, together.

    One node because :func:`~ledgerloop.matching.pipeline.close_ladder` does
    both and splitting it would create a second way to assemble the
    :class:`~ledgerloop.matching.pipeline.MatchRun` every metric is computed
    from. The audit events below report the two stages separately, so nothing
    is lost in the replay.

    **The LLM cannot reach either stage.** A T5 candidate is scored by the same
    blender and routed by the same policy as T2's, and
    :class:`~ledgerloop.models.decisions.MatchDecision` refuses to auto-match
    anything whose arithmetic did not close.
    """
    del resources
    log = _log(state)
    started = _entered(log, "calibrate_and_decide")
    ladder = state["ladder"]
    matched = close_ladder(ladder)

    blend = matched.blend
    log.emit(
        AuditEventType.NODE_COMPLETED,
        "calibrate_confidence",
        message=(
            f"{blend.scored} scored, {blend.bypassed_deterministic} bypassed at T0/T1, "
            f"{blend.refusals_kept} tier refusals kept, "
            f"{blend.abstained_uncovered} abstained on an unfitted tier"
            if matched.calibrated
            else "no fitted bundle: the tiers keep their provisional probabilities"
        ),
        payload={
            "calibrated": matched.calibrated,
            "scored": blend.scored,
            "bypassed_deterministic": blend.bypassed_deterministic,
            "refusals_kept": blend.refusals_kept,
            "abstained_uncovered": blend.abstained_uncovered,
        },
    )
    for decision in matched.decisions:
        log.emit(
            AuditEventType.DECISION_MADE,
            "apply_decision_policy",
            message=f"{decision.outcome.value} at {decision.tier.name}",
            decision_id=decision.decision_id,
            candidate_id=decision.candidate_id,
            refs=(decision.source_ref, decision.target_ref),
            payload={
                "outcome": decision.outcome.value,
                "tier": decision.tier.name,
                "link_type": decision.link_type.value,
                "calibrated_p": decision.calibrated_p,
            },
        )
    _completed(
        log,
        "calibrate_and_decide",
        started,
        message=(
            f"{matched.auto_matched} auto-matched, {matched.needs_review} to review, "
            f"{matched.exceptions} exception"
        ),
        auto_matched=matched.auto_matched,
        needs_review=matched.needs_review,
        exceptions=matched.exceptions,
        evaluable_candidates=matched.evaluable_candidates,
        passes=matched.passes,
    )
    return GraphState(
        matched=matched, node_log=["calibrate_and_decide"], audit=list(log.events)
    )


def classify_exceptions_node(state: GraphState, resources: RunResources) -> GraphState:
    """The typed exception queue, plus bounded auto-resolution.

    Ground truth is **not** an input here, and the ordering in the graph is what
    guarantees it: :func:`measure_quality` is the only node that labels a
    candidate and it runs strictly after this one. A matched record is not
    necessarily a clean one, so the classifier reads the sources and the run's
    own decisions rather than only what the ladder failed to match.

    The resolver **proposes** and never posts. A proposal past its bound is
    emitted as refused, with the bound named.
    """
    log = _log(state)
    started = _entered(log, "classify_exceptions")
    matched = state["matched"]
    config = resources.setup.config
    assert matched.context is not None  # close_ladder always sets it

    queue = classify_exceptions(
        matched.context,
        matched.decisions,
        matched.candidates,
        config,
        merchant_profiles=matched.merchant_spellings,
    )
    resolutions = resolve_bounded(queue.exceptions, config.auto_resolution)
    exceptions = mark_resolvable(queue.exceptions, resolutions)

    for exception in exceptions:
        log.emit(
            AuditEventType.EXCEPTION_RAISED,
            "classify_exceptions",
            message=exception.root_cause,
            exception_id=exception.exception_id,
            refs=exception.involved_refs,
            payload={
                "exception_class": exception.exception_class.value,
                "severity": exception.severity.value,
                "impact_minor": exception.impact_minor,
                "resolvable_by_agent": exception.resolvable_by_agent,
                "suggested_action": exception.suggested_action,
            },
        )
    for resolution in resolutions.resolutions:
        log.emit(
            AuditEventType.AUTO_RESOLUTION_APPLIED
            if resolution.applied
            else AuditEventType.AUTO_RESOLUTION_REFUSED,
            "classify_exceptions",
            message=f"{resolution.rule} (bound: {resolution.bound})",
            payload={
                "exception_class": resolution.exception_class.value,
                "rule": resolution.rule,
                "bound": resolution.bound,
                "applied": resolution.applied,
            },
        )
    _completed(
        log,
        "classify_exceptions",
        started,
        message=(
            f"{len(exceptions)} exception(s), {len(queue.unmatchable)} unmatchable, "
            f"{sum(1 for r in resolutions.resolutions if r.applied)} auto-resolvable"
        ),
        raised=len(exceptions),
        unmatchable=len(queue.unmatchable),
        rounding_spent_minor=resolutions.rounding_spent_minor,
    )
    return GraphState(
        exceptions=tuple(exceptions),
        resolutions=resolutions,
        node_log=["classify_exceptions"],
        audit=list(log.events),
    )


def explain_exceptions(state: GraphState, resources: RunResources) -> GraphState:
    """LLM call site 3: better prose on a queue that is already complete.

    The class, the severity and the rupee figure are given to the model as
    settled facts and are never sent back for revision. An exception whose
    rewrite is refused keeps its template prose and its ``TEMPLATE`` marker, so
    the queue never has a hole in it.
    """
    log = _log(state)
    started = _entered(log, "explain_exceptions")
    exceptions = state["exceptions"]
    outcome = ExplanationOutcome(exceptions=tuple(exceptions))
    if resources.llm_active and resources.client is not None:
        exceptions, outcome = explain_queue(resources.client, list(exceptions))
    _completed(
        log,
        "explain_exceptions",
        started,
        message=(
            f"{outcome.rewritten} of {len(exceptions)} rewritten"
            if resources.llm_active
            else "deterministic: every root cause is template prose"
        ),
        rewritten=outcome.rewritten,
        rejected_ungrounded=outcome.rejected_ungrounded,
    )
    return GraphState(
        exceptions=tuple(exceptions),
        explanation=outcome,
        node_log=["explain_exceptions"],
        audit=list(log.events),
    )


def measure_quality(state: GraphState, resources: RunResources) -> GraphState:
    """Calibration quality on this corpus. **The first node to touch truth.**

    It runs after :func:`classify_exceptions_node` on purpose:
    ``measure_calibration`` writes ``is_truth_positive`` in place onto the
    candidate objects the classifier receives, so measuring first would put
    ground-truth labels inside the classifier's input. Not a leak today,
    because the classifier does not read the field -- a leak that would arrive
    silently the day a rule did.
    """
    log = _log(state)
    started = _entered(log, "measure_quality")
    view = None
    if resources.bundle is not None and resources.measure_calibration_quality:
        contenders = score_contenders(
            resources.bundle,
            harvest(state["ingest"], resources.setup.truth, resources.setup.config).rows,
        )
        view = measure_calibration(
            state["matched"].candidates,
            resources.setup.truth,
            contender_probabilities=contenders.probabilities,
            contender_labels=contenders.labels,
        )
    _completed(
        log,
        "measure_quality",
        started,
        message=(
            f"ECE {view.asserted.ece:.4f} over {view.asserted.sample_count} residual links"
            if view is not None
            else "skipped: no fitted bundle, or measurement not requested"
        ),
        measured=view is not None,
    )
    return GraphState(
        calibration=view, node_log=["measure_quality"], audit=list(log.events)
    )


def generate_report(state: GraphState, resources: RunResources) -> GraphState:
    """Score the finished run. The only node that produces a metric.

    Delegates to :func:`~ledgerloop.eval.harness.assemble_system_run`, which the
    direct CLI path also calls -- so a number cannot differ between the graph
    and the chain by construction rather than by a passing test.
    """
    log = _log(state)
    started = _entered(log, "generate_report")
    cost = (
        resources.client.ledger() if resources.client is not None else CostLedger()
    )
    system_run = assemble_system_run(
        resources.setup,
        ingested=state["ingest"],
        matched=state["matched"],
        exceptions=state["exceptions"],
        resolutions=state["resolutions"],
        calibration=state.get("calibration"),
        cost=cost,
        llm=LLMRunSummary(
            narration=state.get("narration") or NarrationOutcome(),
            adjudication=state.get("adjudication") or AdjudicationOutcome(),
            explanation=state.get("explanation")
            or ExplanationOutcome(exceptions=tuple(state["exceptions"])),
        ),
    )
    metrics = system_run.metrics
    _completed(
        log,
        "generate_report",
        started,
        message=(
            f"precision {metrics.auto_match_precision:.4f}, "
            f"match rate {metrics.match_rate:.4f}, "
            f"exception recall {metrics.exception_recall:.4f}"
        ),
        auto_match_precision=metrics.auto_match_precision,
        match_rate=metrics.match_rate,
        exception_recall=metrics.exception_recall,
        llm_calls=cost.llm_calls,
        total_tokens=cost.total_tokens,
    )
    log.emit(
        AuditEventType.RUN_COMPLETED,
        "generate_report",
        message=f"run {resources.setup.config.run_id} complete",
        payload={"config_hash": resources.setup.config.config_hash},
    )
    return GraphState(
        system_run=system_run, node_log=["generate_report"], audit=list(log.events)
    )


def record_failure(state: GraphState, node: str, error: BaseException) -> GraphState:
    """Turn a node's exception into state the graph can route on.

    A reconciliation that dies halfway must still leave a replayable log. The
    error is recorded as an audit event and as state; the graph then routes to
    ``END`` rather than letting the traceback escape, so the partial run is
    inspectable and resumable instead of merely lost.
    """
    log = _log(state)
    log.emit(
        AuditEventType.RUN_FAILED,
        node,
        message=f"{type(error).__name__}: {error}",
        payload={"exception_type": type(error).__name__},
    )
    return GraphState(
        failed_node=node,
        error=f"{type(error).__name__}: {error}",
        node_log=[f"{node}:failed"],
        audit=list(log.events),
    )
