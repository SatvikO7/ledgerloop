"""The LangGraph state, and the resources that deliberately stay out of it.

PLAN.md §4.2 fixes the shape: *every node takes state, returns state; no hidden
globals.* This module holds that state for the graph, and draws one line the
plan does not: **the model client and the fitted bundle are not state.**

WHY RESOURCES ARE NOT STATE
---------------------------
A checkpointer serialises state. An :class:`~ledgerloop.llm.client.LLMClient`
holds an API key; a :class:`~ledgerloop.matching.calibration.CalibrationBundle`
is a fitted model that already lives on disk with its own provenance. Neither
belongs in a file written after every node:

* a checkpoint carrying a key is a secret in an artefact directory;
* a checkpoint carrying a copy of the bundle could be resumed against a
  *different* bundle than the one its provenance names, which is precisely the
  stale-calibration failure ``load_bundle_for`` exists to refuse.

So :class:`RunResources` is captured by the graph builder and closed over by the
nodes. The state holds data; the resources hold capability. Resuming a run means
supplying the resources again — which is correct, because they are exactly the
things that should be re-checked rather than restored.

WHY A TypedDict
---------------
LangGraph merges each node's returned mapping into the state. A ``TypedDict``
with ``total=False`` says exactly that: a node returns the keys it changed, and
nothing else. A Pydantic model would demand every field on every return, or a
custom reducer per field, for no gain — the models that matter
(:class:`~ledgerloop.models.metrics.RunMetrics`,
:class:`~ledgerloop.models.recon_exception.ReconException`) are already typed
and are carried *inside* the state rather than replaced by it.

``audit`` is the one key with a reducer: events **append**, they never replace.
That is the append-only guarantee made mechanical — a node that tried to
overwrite the log would find its events added to the end instead.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from ledgerloop.eval.harness import RunSetup
from ledgerloop.llm.client import LLMClient
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.matching.pipeline import ResidualAdjudicator

__all__ = ["GraphState", "RunResources", "initial_state"]


@dataclass(frozen=True)
class RunResources:
    """Capability, not data. Supplied at build time, never checkpointed.

    ``adjudicator`` is T5 as the ladder declares it -- a Protocol, so
    :mod:`ledgerloop.matching` still does not import :mod:`ledgerloop.llm`
    (ARCHITECTURE.md §6, decision 43). The graph is assembled by the CLI, which
    owns the model, and hands the ladder a callable or ``None``.
    """

    setup: RunSetup
    bundle: CalibrationBundle | None = None
    client: LLMClient | None = None
    adjudicator: ResidualAdjudicator | None = None
    measure_calibration_quality: bool = True

    @property
    def llm_active(self) -> bool:
        return self.setup.llm_active


class GraphState(TypedDict, total=False):
    """What flows between nodes.

    Every value here is produced by a function that existed before Step 11. The
    graph moves them; it does not compute them.
    """

    run_id: str
    node_log: Annotated[list[str], operator.add]
    """Node names in execution order, appended. The graph's own trace.

    Separate from ``audit`` because it answers a different question: the audit
    log is what a controller replays, this is what a *developer* reads to see
    which edges fired -- including the residual loop's repeats, which are the
    whole reason the graph is a graph.
    """

    audit: Annotated[list[Any], operator.add]
    """:class:`~ledgerloop.models.audit.AuditEvent` records, append-only.

    Typed ``Any`` because a ``TypedDict`` annotation is evaluated by LangGraph
    at build time and the concrete model would pull a heavier import into the
    state schema for no checking benefit -- the producer in
    :mod:`ledgerloop.agent.audit` is typed, which is where it matters.
    """

    ingest: Any
    """:class:`~ledgerloop.ingest.dataset.IngestResult`, possibly narration-repaired."""

    ladder: Any
    """:class:`~ledgerloop.matching.pipeline.LadderRun` -- the accumulating ladder."""

    matched: Any
    """:class:`~ledgerloop.matching.pipeline.MatchRun`, once the ladder is closed."""

    exceptions: Any
    resolutions: Any
    calibration: Any
    system_run: Any
    """:class:`~ledgerloop.eval.harness.SystemRun` -- the finished, scored run."""

    narration: Any
    adjudication: Any
    explanation: Any

    passes: int
    """Residual passes taken. Written by the loop node, read by the edge."""

    failed_node: str | None
    error: str | None
    """Set when a node raised. The graph routes to END rather than propagating.

    A reconciliation that dies halfway must still leave a replayable log --
    that is the whole point of an append-only audit trail, and it is what makes
    the failure resumable rather than merely reported.
    """


def initial_state(run_id: str) -> GraphState:
    """The empty state one run starts from."""
    return GraphState(
        run_id=run_id,
        node_log=[],
        audit=[],
        passes=0,
        failed_node=None,
        error=None,
    )
