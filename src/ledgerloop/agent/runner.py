"""Run the graph, and read a run's history back.

The composition layer for Step 11, sitting where :mod:`ledgerloop.fitting` sits:
:mod:`ledgerloop.agent.graph` knows how to wire nodes, :mod:`ledgerloop.eval.
harness` knows how to prepare and score a run, and this is the twenty lines that
put the two together.

WHAT RESUMABILITY MEANS HERE, EXACTLY
-------------------------------------
Two different guarantees, and conflating them would be the easy lie:

**Replay is durable.** ``audit.jsonl`` survives the process. Any completed or
*failed* run can be walked event by event afterwards, from a different process,
on a different day -- which is what PLAN.md D7 asks for and what the UI's Audit
Replay screen reads.

**Resume is in-process.** LangGraph's ``InMemorySaver`` snapshots state after
every node, so :func:`resume_run` can pick a failed run up at the node that
failed without redoing the ones before it. That checkpoint dies with the
process. Making it survive would mean serialising a ``MatchContext``, an
``IngestResult`` and a live ``LLMClient`` -- the client is the reason it is not
attempted: a checkpoint on disk carrying an API key is a secret in an artefact
directory.

The boundary is stated rather than papered over. A cross-process resume today
means re-running from the top, which on a 300-record corpus is under a second --
and the audit log of the failed attempt is still there to explain why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ledgerloop.agent.audit import AuditLog
from ledgerloop.agent.graph import LangGraphUnavailable, build_recon_graph
from ledgerloop.agent.state import GraphState, RunResources, initial_state
from ledgerloop.agent.store import RUNS_ROOT, save_run
from ledgerloop.eval.harness import SystemRun, prepare_run
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.integration import adjudicator_for
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.models.audit import AuditEvent, AuditEventType

__all__ = ["GraphRun", "resume_run", "run_graph"]


@dataclass(frozen=True)
class GraphRun:
    """One graph execution: the scored run, its log, and how it got there."""

    system: SystemRun | None
    audit: AuditLog
    node_log: tuple[str, ...]
    passes: int
    thread_id: str
    failed_node: str | None = None
    error: str | None = None
    wall_clock_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.system is not None

    @property
    def residual_iterations(self) -> int:
        """How many times the cycle edge fired. The loop, counted."""
        return sum(1 for node in self.node_log if node == "tier_ladder")

    def require(self) -> SystemRun:
        """The scored run, or the failure that stopped it -- never ``None``."""
        if self.system is None:
            raise RuntimeError(
                f"the run failed at {self.failed_node}: {self.error}. "
                f"Its audit log has {len(self.audit.events)} event(s) and is "
                "replayable up to the failure."
            )
        return self.system


def _resources(
    directory: Path,
    *,
    bundle: CalibrationBundle | None,
    client: LLMClient | None,
    enabled_tiers: tuple[int, ...] | None,
    run_id: str | None,
    measure_calibration_quality: bool,
) -> RunResources:
    setup = prepare_run(
        directory,
        bundle=bundle,
        client=client,
        enabled_tiers=enabled_tiers,
        run_id=run_id,
    )
    # T5 is injected as the Protocol the ladder declares, so `matching` still
    # does not import `llm` (ARCHITECTURE.md §6, decision 43). `--no-llm` and a
    # missing key both land here as `None`.
    adjudicator = (
        adjudicator_for(client, setup.config)
        if (client is not None and setup.llm_active)
        else None
    )
    return RunResources(
        setup=setup,
        bundle=bundle,
        client=client,
        adjudicator=adjudicator,
        measure_calibration_quality=measure_calibration_quality,
    )


def _collect(state: GraphState, resources: RunResources, elapsed_ms: int, thread: str) -> GraphRun:
    log = AuditLog(run_id=resources.setup.config.run_id)
    events: list[AuditEvent] = [
        event for event in state.get("audit", []) if isinstance(event, AuditEvent)
    ]
    log.events = events
    log._sequence = len(events)
    return GraphRun(
        system=state.get("system_run"),
        audit=log,
        node_log=tuple(state.get("node_log", [])),
        passes=int(state.get("passes", 0)),
        thread_id=thread,
        failed_node=state.get("failed_node"),
        error=state.get("error"),
        wall_clock_ms=elapsed_ms,
    )


def run_graph(
    directory: Path,
    *,
    bundle: CalibrationBundle | None = None,
    client: LLMClient | None = None,
    enabled_tiers: tuple[int, ...] | None = None,
    run_id: str | None = None,
    measure_calibration_quality: bool = True,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
    store: Path | None = RUNS_ROOT,
    recursion_limit: int = 50,
) -> GraphRun:
    """Execute the reconciliation graph over one dataset directory.

    Produces a :class:`~ledgerloop.eval.harness.SystemRun` identical to the one
    :func:`~ledgerloop.eval.harness.run_system` produces for the same inputs --
    both call the same node functions and the same
    :func:`~ledgerloop.eval.harness.assemble_system_run`. A test asserts the
    equality on predictions, decisions, the tier table and every headline metric.

    ``store`` writes the run's four files for the UI to read. Pass ``None`` to
    run without leaving a record -- which is what the equivalence tests do, so
    they cannot pollute ``reports/runs`` with hundreds of fixtures.

    ``recursion_limit`` bounds LangGraph's own step count. The residual loop is
    already capped by ``config.graph.max_rerun_passes``; this is the second
    bound, on the graph rather than on the tier, because an unbounded loop in a
    reconciliation run is a hang and not a slow answer.
    """
    resources = _resources(
        directory,
        bundle=bundle,
        client=client,
        enabled_tiers=enabled_tiers,
        run_id=run_id,
        measure_calibration_quality=measure_calibration_quality,
    )
    graph = build_recon_graph(resources, checkpointer=checkpointer)
    thread = thread_id or resources.setup.config.run_id

    log = AuditLog(run_id=resources.setup.config.run_id)
    log.emit(
        AuditEventType.RUN_STARTED,
        "__start__",
        message=f"reconciling {directory}",
        payload={
            "config_hash": resources.setup.config.config_hash,
            "enabled_tiers": list(resources.setup.tiers),
            "llm_active": resources.llm_active,
        },
    )
    state = initial_state(resources.setup.config.run_id)
    state["audit"] = list(log.events)

    started = time.perf_counter_ns()
    final = cast(
        GraphState,
        graph.invoke(
            state,
            config={
                "configurable": {"thread_id": thread},
                "recursion_limit": recursion_limit,
            },
        ),
    )
    elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000

    result = _collect(final, resources, elapsed_ms, thread)
    if store is not None and result.system is not None:
        save_run(result.system, result.audit, root=store)
    return result


def resume_run(
    graph: Any,
    thread_id: str,
    *,
    resources: RunResources,
    recursion_limit: int = 50,
) -> GraphRun:
    """Continue a checkpointed run from wherever it stopped.

    In-process only -- see the module docstring. ``graph`` must be the compiled
    graph the original run used, because the checkpointer lives inside it.

    Invoking with ``None`` is LangGraph's resume signal: it reloads the last
    checkpoint for the thread rather than starting a fresh state, so the nodes
    that already succeeded are not re-run.
    """
    started = time.perf_counter_ns()
    final = cast(
        GraphState,
        graph.invoke(
            None,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            },
        ),
    )
    elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    return _collect(final, resources, elapsed_ms, thread_id)


__all__ += ["LangGraphUnavailable"]
