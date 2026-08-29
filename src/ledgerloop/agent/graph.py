"""The LangGraph state machine (PLAN.md §4.1), assembled from tested functions.

WHY LANGGRAPH IS HERE AT ALL
----------------------------
PLAN.md §4.3 answers this without overselling it, and the answer is repeated
here because it is the honest one: *for the happy path, a function chain would
do.* Three things earn the framework its place, and only three:

1. **The loop.** ``tier_ladder`` re-runs while a pass keeps adding candidates --
   resolving one bank credit constrains the remaining ones, so a later pass can
   match what an earlier one could not. That is a real cycle in the graph, not a
   ``while`` hidden inside a node.
2. **The conditional branch.** Whether to loop again, and whether T5 runs at
   all, are edges rather than ``if`` statements buried in a function.
3. **Checkpointing.** A snapshot after every node, which is what makes a failed
   run resumable and what gives the Audit Replay screen something to walk.

Nothing else is claimed for it. The reconciliation logic is unchanged and lives
where it lived: :mod:`ledgerloop.matching`, :mod:`ledgerloop.exceptions`,
:mod:`ledgerloop.llm`, :mod:`ledgerloop.eval`.

WHY THE IMPORT IS OPTIONAL
--------------------------
``langgraph`` is an extra (``pip install 'ledgerloop[graph]'``), not a runtime
dependency. The project's claim is that the deterministic core runs on
``pydantic`` and ``rapidfuzz`` alone, and a state machine wrapped around a
finished pipeline must not be able to take that away: ``ledgerloop eval``,
``ablation``, ``sweep`` and every metric in ``EVALUATION.md`` are produced
without importing this module. Absent the extra, :func:`build_recon_graph`
raises a message naming the install rather than an ``ImportError`` from four
frames down.

FAILURE IS ROUTED, NOT RAISED
-----------------------------
Every node is wrapped by :func:`_guarded`, which catches, records a
``RUN_FAILED`` audit event, and sets ``error`` in state. The graph then routes
to ``END``. A reconciliation that dies halfway leaves a **replayable** log and a
checkpoint to resume from; a traceback out of ``invoke`` would leave neither.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ledgerloop.agent.nodes import (
    build_entity_graph,
    calibrate_and_decide,
    classify_exceptions_node,
    complete_splits_node,
    explain_exceptions,
    generate_report,
    ingest_sources,
    llm_adjudicate,
    measure_quality,
    normalize_records,
    record_failure,
    should_loop,
    tier_ladder,
)
from ledgerloop.agent.state import GraphState, RunResources

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.graph.state import CompiledStateGraph

__all__ = [
    "GRAPH_EDGES",
    "LangGraphUnavailable",
    "build_recon_graph",
    "langgraph_available",
]

_INSTALL_HINT = (
    "LangGraph is an optional extra. Install it with "
    "`uv pip install -e \".[graph]\"` (or `pip install 'ledgerloop[graph]'`). "
    "Every metric in EVALUATION.md is produced without it -- `ledgerloop eval`, "
    "`ablation` and `sweep` do not import this module."
)

#: The graph's topology, as data, so a test can assert it rather than infer it.
#:
#: ``(from, to, condition)``. A condition of ``None`` is an unconditional edge.
#: The two entries out of ``tier_ladder`` are the cycle and its exit -- the pair
#: that PLAN.md §4.3 names as LangGraph's justification.
GRAPH_EDGES: tuple[tuple[str, str, str | None], ...] = (
    ("__start__", "ingest_sources", None),
    ("ingest_sources", "normalize_records", None),
    ("normalize_records", "build_entity_graph", None),
    ("build_entity_graph", "tier_ladder", "residual passes remain"),
    ("build_entity_graph", "complete_splits", "no residual tier is enabled"),
    ("tier_ladder", "tier_ladder", "the last pass added candidates"),
    ("tier_ladder", "complete_splits", "the last pass added nothing, or the cap is hit"),
    ("complete_splits", "llm_adjudicate", None),
    ("llm_adjudicate", "calibrate_and_decide", None),
    ("calibrate_and_decide", "classify_exceptions", None),
    ("classify_exceptions", "explain_exceptions", None),
    ("explain_exceptions", "measure_quality", None),
    ("measure_quality", "generate_report", None),
    ("generate_report", "__end__", None),
)


class LangGraphUnavailable(RuntimeError):
    """``langgraph`` is not installed. Raised with the install command."""


def langgraph_available() -> bool:
    """Whether the optional extra is importable. Used by the CLI and the UI."""
    try:  # pragma: no cover - trivially one branch per environment
        import langgraph.graph  # noqa: F401
    except ImportError:
        return False
    return True


def _guarded(
    name: str, node: Callable[[GraphState, RunResources], GraphState], resources: RunResources
) -> Callable[[GraphState], GraphState]:
    """Bind a node to its resources and turn any failure into routable state.

    The binding is a closure rather than a state field on purpose: an
    :class:`~ledgerloop.llm.client.LLMClient` holds an API key and a
    checkpointer writes state to disk. See :mod:`ledgerloop.agent.state`.
    """

    def invoke(state: GraphState) -> GraphState:
        if state.get("error"):
            # An earlier node already failed. Downstream nodes are skipped
            # rather than run against half-built state, which would turn one
            # honest failure into a second, misleading one.
            return GraphState()
        try:
            return node(state, resources)
        except Exception as error:
            return record_failure(state, name, error)

    invoke.__name__ = name
    return invoke


def _route_after(name: str, chooser: Callable[[GraphState], str]) -> Callable[[GraphState], str]:
    """Wrap a conditional edge so a failed run short-circuits to the end."""

    def choose(state: GraphState) -> str:
        if state.get("error"):
            return "__end__"
        return chooser(state)

    choose.__name__ = f"route_after_{name}"
    return choose


def build_recon_graph(
    resources: RunResources,
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[GraphState, Any, GraphState, GraphState]:
    """Compile the reconciliation graph for one set of resources.

    ``checkpointer`` defaults to LangGraph's ``InMemorySaver``, which snapshots
    the state after every node and gives ``get_state_history()`` -- enough to
    resume a failed run in-process and to walk the node sequence afterwards.

    **The durable record is the JSONL audit log, not the checkpointer.** A
    checkpoint holds live Python objects (a ``MatchContext``, an
    ``IngestResult``); the audit log holds facts about what happened, in a
    format that survives a process, a version bump and a text editor. Cross-
    process resume would need a serialising checkpointer over those objects and
    is not built -- :mod:`ledgerloop.agent.runner` documents the boundary.
    """
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:  # pragma: no cover - environment dependent
        raise LangGraphUnavailable(_INSTALL_HINT) from error

    builder: Any = StateGraph(GraphState)
    for name, node in (
        ("ingest_sources", ingest_sources),
        ("normalize_records", normalize_records),
        ("build_entity_graph", build_entity_graph),
        ("tier_ladder", tier_ladder),
        ("complete_splits", complete_splits_node),
        ("llm_adjudicate", llm_adjudicate),
        ("calibrate_and_decide", calibrate_and_decide),
        ("classify_exceptions", classify_exceptions_node),
        ("explain_exceptions", explain_exceptions),
        ("measure_quality", measure_quality),
        ("generate_report", generate_report),
    ):
        builder.add_node(name, _guarded(name, node, resources))

    builder.add_edge(START, "ingest_sources")
    builder.add_edge("ingest_sources", "normalize_records")
    builder.add_edge("normalize_records", "build_entity_graph")

    # The conditional branch and the cycle. Both read
    # `should_run_residual_pass` through `should_loop`, so the graph's loop and
    # the direct path's `while` are the same predicate rather than two copies.
    # The loop's exit lands on `complete_splits`, not on T5: the split-completion
    # stage is the last thing the deterministic ladder does, and it has to see a
    # pool nothing else will touch again. `should_loop` still returns the name
    # `llm_adjudicate` for "stop looping" -- that is `should_run_residual_pass`'s
    # own vocabulary and is left alone -- so the mapping is what redirects it.
    loop_targets = {
        "tier_ladder": "tier_ladder",
        "llm_adjudicate": "complete_splits",
        "__end__": END,
    }
    builder.add_conditional_edges(
        "build_entity_graph", _route_after("build_entity_graph", should_loop), loop_targets
    )
    builder.add_conditional_edges(
        "tier_ladder", _route_after("tier_ladder", should_loop), loop_targets
    )

    for source, target in (
        ("complete_splits", "llm_adjudicate"),
        ("llm_adjudicate", "calibrate_and_decide"),
        ("calibrate_and_decide", "classify_exceptions"),
        ("classify_exceptions", "explain_exceptions"),
        ("explain_exceptions", "measure_quality"),
        ("measure_quality", "generate_report"),
    ):
        builder.add_edge(source, target)
    builder.add_edge("generate_report", END)

    compiled: CompiledStateGraph[GraphState, Any, GraphState, GraphState] = (
        builder.compile(checkpointer=checkpointer or InMemorySaver())
    )
    return compiled
