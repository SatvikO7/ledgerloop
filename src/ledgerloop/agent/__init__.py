"""The LangGraph assembly (Step 11) and the run store the UI reads (Step 12).

PLAN.md §4.1 draws the state machine; this package is it. Two properties matter
more than anything in the modules themselves:

**No reconciliation logic lives here.** Every node in :mod:`ledgerloop.agent.
nodes` calls a function that existed and was tested before Step 11 --
:mod:`ledgerloop.matching`, :mod:`ledgerloop.exceptions`,
:mod:`ledgerloop.llm`, :mod:`ledgerloop.eval`. The graph moves data between
them and records what happened. It computes nothing.

**LangGraph is an optional extra.** ``ledgerloop eval``, ``ablation``, ``sweep``
and every number in ``EVALUATION.md`` are produced without importing it, so the
deterministic core keeps the two-dependency footprint the project claims. Only
:mod:`ledgerloop.agent.graph` and :mod:`ledgerloop.agent.runner` need it, and
they say so with an install hint rather than an ``ImportError`` from four frames
down. :mod:`ledgerloop.agent.audit` and :mod:`ledgerloop.agent.store` need
nothing beyond the core, which is why the UI can read a run on a machine that
cannot execute one.
"""

from __future__ import annotations

from ledgerloop.agent.audit import AUDIT_FILE, AuditLog, read_audit_jsonl
from ledgerloop.agent.state import GraphState, RunResources, initial_state
from ledgerloop.agent.store import (
    RUNS_ROOT,
    StoredRun,
    list_runs,
    load_run,
    save_run,
)

__all__ = [
    "AUDIT_FILE",
    "RUNS_ROOT",
    "AuditLog",
    "GraphState",
    "RunResources",
    "StoredRun",
    "initial_state",
    "list_runs",
    "load_run",
    "read_audit_jsonl",
    "save_run",
]
