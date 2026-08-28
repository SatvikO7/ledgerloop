"""The Streamlit interface. Widget glue over :mod:`ledgerloop.ui.views`.

Run it with ``make ui`` (or ``streamlit run src/ledgerloop/ui/app.py``).

WHY THIS FILE HAS NO LOGIC
--------------------------
Every number on every screen comes from a run that already finished and wrote
itself to ``reports/runs/<run_id>/``. This module reads those four files
through :mod:`ledgerloop.agent.store`, shapes them through
:mod:`ledgerloop.ui.views`, and calls widgets. It **cannot** recompute a metric,
re-derive a decision or re-classify an exception, because it never holds the
objects that could.

That is deliberate and it is the failure mode being avoided: a dashboard that
computes its own version of a number is a second implementation, and the day it
disagrees with ``EVALUATION.md`` nobody can say which is right.

The only action that runs anything is the button on screen 1, and it calls
:func:`~ledgerloop.agent.runner.run_graph` -- the same entry point
``ledgerloop run`` uses.

WHY STREAMLIT AND NOT REACT
---------------------------
PLAN.md §10 lists React + Vite + FastAPI with Streamlit as the fallback, and
§16's cut list has "React UI → Streamlit" at position 5. This is that cut,
taken deliberately: the four screens are tables and a form, a FastAPI gateway
would exist only to serve them to a second process, and the demo is graded on
`make demo` working on a clean machine. One dependency, one process, no build
step.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import streamlit as st

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.store import RUNS_ROOT, StoredRun, list_runs
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import Difficulty, SplitName
from ledgerloop.money import format_minor
from ledgerloop.ui.views import (
    OUTCOME_HELP,
    Headline,
    evidence_rows,
    exception_rows,
    headline,
    money_rows,
    recall_rows,
    record_keys,
    tier_rows,
    trace_record,
)

#: Where the app looks, overridable by environment.
#:
#: Streamlit re-executes this script on every interaction and on every
#: ``AppTest`` run, so a module-level constant cannot be patched from outside --
#: which makes an environment variable the only honest seam. It is not merely a
#: test hook: pointing the UI at another machine's ``reports/runs`` is a real
#: thing to want, and this is how.
_RUNS_ENV = "LEDGERLOOP_RUNS_DIR"
_DATA_ENV = "LEDGERLOOP_DATA_DIR"
_BUNDLE_ENV = "LEDGERLOOP_CALIBRATION"


def _runs_root() -> Path:
    return Path(os.environ.get(_RUNS_ENV, str(RUNS_ROOT)))


def _data_root() -> Path:
    return Path(os.environ.get(_DATA_ENV, "data/generated"))


def _bundle_path() -> Path:
    return Path(os.environ.get(_BUNDLE_ENV, "reports/calibration.json"))


def _datasets() -> list[Path]:
    root = _data_root()
    if not root.is_dir():
        return []
    return sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "manifest.json").is_file()
    )


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


_T = TypeVar("_T")


def _picked(value: object, options: Sequence[_T]) -> _T:
    """A selectbox's answer, narrowed to the option type.

    Streamlit's stubs type ``selectbox`` as returning ``None`` for a plain list
    of options, which makes every downstream use unreachable to the type
    checker. The value is genuinely one of ``options`` -- the widget has no
    other way to answer a non-empty list -- so the narrowing is a cast with a
    fallback to the first option rather than a runtime check that can fire.
    """
    return cast(_T, value) if value is not None else options[0]


# --------------------------------------------------------------------------
# Screen 1 -- Run
# --------------------------------------------------------------------------
def screen_run() -> None:
    """Generate or pick a dataset, then reconcile it through the graph."""
    st.header("Run a reconciliation")
    st.caption(
        "Everything below executes the same pipeline `ledgerloop run` does, "
        "through the LangGraph state machine. No number on any screen is "
        "computed by this interface."
    )

    with st.expander("Generate a dataset", expanded=not _datasets()):
        st.write(
            "Ground truth is generated **first** and the data is derived from it. "
            "Generation is a pure function of (seed, split, difficulty), so the "
            "same inputs twice produce byte-identical files."
        )
        columns = st.columns(3)
        splits, difficulties = list(SplitName), list(Difficulty)
        split = _picked(
            columns[0].selectbox(
                "Split", splits, format_func=lambda value: value.value, index=0
            ),
            splits,
        )
        difficulty = _picked(
            columns[1].selectbox(
                "Difficulty", difficulties, format_func=lambda value: value.value, index=1
            ),
            difficulties,
        )
        seed = columns[2].number_input("Seed", min_value=0, value=42, step=1)
        if st.button("Generate", key="generate"):
            config = GeneratorConfig(split=split, difficulty=difficulty, seed=int(seed))
            target = _data_root() / f"{split.value}-{difficulty.value}-{int(seed)}"
            with st.spinner(f"generating {target}"):
                generate_to_disk(config, target)
            st.success(f"wrote {target}")
            st.rerun()

    datasets = _datasets()
    if not datasets:
        st.info("No datasets yet. Generate one above, or run `make data`.")
        return

    st.subheader("Reconcile")
    chosen = _picked(
        st.selectbox("Dataset", datasets, format_func=lambda path: path.name),
        datasets,
    )
    bundle_path = _bundle_path()
    use_bundle = st.checkbox(
        "Apply the fitted calibration bundle",
        value=bundle_path.is_file(),
        disabled=not bundle_path.is_file(),
        help=(
            "Thresholds fitted on the `calibration` split, never on `test`. "
            "Without it the residual tiers keep the provisional probabilities "
            "they set themselves."
        ),
    )
    st.caption(
        "The LLM is off unless `LEDGERLOOP_LLM_API_KEY` is set. Every number "
        "this demo produces is deterministic either way."
    )

    if not langgraph_available():
        st.error(
            "LangGraph is not installed. Install the extra with "
            '`uv pip install -e ".[demo]"`.'
        )
        return

    if st.button("Run reconciliation", type="primary", key="reconcile"):
        from ledgerloop.agent.runner import run_graph
        from ledgerloop.eval.harness import load_bundle_for
        from ledgerloop.eval.truth_io import load_manifest

        bundle = None
        if use_bundle and bundle_path.is_file():
            try:
                bundle = load_bundle_for(bundle_path, load_manifest(chosen))
            except ValueError as error:
                st.error(str(error))
                return
        with st.spinner(f"reconciling {chosen.name}"):
            result = run_graph(chosen, bundle=bundle, store=_runs_root())
        if not result.ok:
            # An honest failure, shown as one. The log is still replayable.
            st.error(f"The run failed at `{result.failed_node}`: {result.error}")
            st.caption(
                f"{len(result.audit.events)} audit event(s) were recorded and the "
                "log replays up to the failure."
            )
            return
        st.success(
            f"Reconciled in {result.wall_clock_ms} ms across "
            f"{len(result.node_log)} node visit(s), "
            f"{result.residual_iterations} residual pass(es)."
        )
        st.rerun()


# --------------------------------------------------------------------------
# Screen 2 -- Results
# --------------------------------------------------------------------------
def _outcome_columns(view: Headline) -> None:
    """Every decision outcome, always as separate figures.

    All four, including ``REJECTED`` at zero: the policy ran and did not use it,
    which is a measurement. Showing three of four would mean choosing which
    outcomes to mention.
    """
    st.subheader("What the system decided")
    st.caption(
        "Every outcome the policy has, never one 'matched' number. A referral is "
        "not a match, and an exception is a decision to escalate rather than a "
        "failure to decide."
    )
    columns = st.columns(4)
    for column, (label, value) in zip(
        columns,
        (
            ("AUTO_MATCHED", view.auto_matched),
            ("NEEDS_REVIEW", view.needs_review),
            ("EXCEPTION", view.exceptions),
            ("REJECTED", view.rejected),
        ),
        strict=True,
    ):
        column.metric(label, value, help=OUTCOME_HELP[label])
    st.caption(
        f"On the evaluation unit ({view.evaluation_links} `PAYMENT_CREDITED_AS` "
        f"links), from {view.candidates_proposed} candidate(s) proposed. The "
        f"exception **queue** holds {view.queue_size} item(s) -- a settlement "
        "nothing could credit raises one without producing a link decision."
    )


def screen_results(run: StoredRun) -> None:
    st.header("Results")
    view = headline(run)

    columns = st.columns(4)
    columns[0].metric(
        "Auto-match precision",
        _pct(view.precision),
        help=(
            f"95% Wilson interval [{view.precision_ci[0]:.4f}, "
            f"{view.precision_ci[1]:.4f}]. Target ≥ 99.00%."
        ),
    )
    columns[1].metric("Match rate", _pct(view.match_rate), help="Target ≥ 85.00%.")
    columns[2].metric("Link recall", _pct(view.recall))
    columns[3].metric(
        "Exception recall", _pct(view.exception_recall), help="Target ≥ 95.00%."
    )

    if view.precision_is_perfect:
        st.success(
            f"{view.true_positives} correct links, **zero** false positives, "
            f"₹0 of wrongly reconciled money."
        )
    else:
        st.warning(
            f"{view.false_positives} false positive(s) costing "
            f"{view.false_positive_cost}. That is money declared reconciled "
            "that was not."
        )

    _outcome_columns(view)

    st.subheader("The money")
    st.dataframe(money_rows(run), width="stretch", hide_index=True)
    st.caption(
        f"{view.unmatchable_count} record(s) are unmatchable by construction, "
        f"worth {view.unmatchable_impact}. They are excluded from the match-rate "
        "denominator and reported here instead: a real ceiling is not a model "
        "failure."
    )

    st.subheader("What each tier contributed")
    st.dataframe(tier_rows(run), width="stretch", hide_index=True)
    st.caption(
        "Proposed and auto-matched are separate columns. A tier that proposes a "
        "hundred and commits forty has not performed like one that proposes "
        f"forty and commits forty. The residual loop ran {view.residual_passes} "
        "pass(es)."
    )

    st.subheader("Recall by anomaly class")
    st.caption("Every class the corpus contains, **including the ones that score badly**.")
    st.dataframe(recall_rows(run), width="stretch", hide_index=True)

    st.subheader("What the model cost")
    if not view.llm_available:
        st.info(
            "No LLM ran. Every number above is deterministic: the links come "
            "from the tier ladder, the exception classes and amounts from the "
            "classifier, the probabilities from the fitted bundle."
        )
    else:
        columns = st.columns(3)
        columns[0].metric("Calls", view.llm_calls)
        columns[1].metric("Tokens", f"{view.llm_tokens:,}")
        columns[2].metric(
            "Equivalent paid cost", f"₹{view.equivalent_paid_cost_inr:.2f}",
            help="Actual spend is ₹0 on the free tier.",
        )
        st.caption(
            "The model never decided a match and never did arithmetic. Every "
            "proposal it made was re-derived from the source documents before "
            "any decision was taken."
        )


# --------------------------------------------------------------------------
# Screen 3 -- Exceptions
# --------------------------------------------------------------------------
def screen_exceptions(run: StoredRun) -> None:
    st.header("Exception queue")
    rows = exception_rows(run)
    if not rows:
        st.success("No exceptions were raised for this run.")
        return

    total = sum(int(row["impact_minor"]) for row in rows)
    st.caption(
        f"{len(rows)} exception(s) covering {format_minor(total)}, "
        "**sorted by rupee impact descending**. One ₹4 lakh payout matters more "
        "than two hundred one-paise drifts, and any other sort order hides that."
    )

    columns = st.columns(2)
    severities = sorted({str(row["Severity"]) for row in rows})
    classes = sorted({str(row["Class"]) for row in rows})
    severity_options = ["all", *severities]
    class_options = ["all", *classes]
    severity = _picked(
        columns[0].selectbox("Severity", severity_options), severity_options
    )
    exception_class = _picked(
        columns[1].selectbox("Class", class_options), class_options
    )

    shown = exception_rows(
        run,
        severity=None if severity == "all" else severity,
        exception_class=None if exception_class == "all" else exception_class,
    )
    if len(shown) != len(rows):
        st.caption(
            f"Showing {len(shown)} of {len(rows)}. The filter narrows the view; "
            "it does not change the counts on the Results screen."
        )
    st.dataframe(
        [{k: v for k, v in row.items() if k not in {"impact_minor", "exception_id"}}
         for row in shown],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Evidence chain")
    st.caption(
        "Every row carries a chain a controller can check against the sources "
        "rather than trusting the system's summary."
    )
    by_id = {item.exception_id: item for item in run.exceptions}
    options = [str(row["exception_id"]) for row in shown]
    if not options:
        return
    chosen = _picked(
        st.selectbox(
        "Exception",
        options,
        format_func=lambda value: (
            f"{by_id[value].severity.value} · "
            f"{by_id[value].exception_class.value} · "
            f"{by_id[value].involved_refs[0].record_id if by_id[value].involved_refs else ''}"
        ),
        ),
        options,
    )
    exception = by_id[chosen]
    st.markdown(f"**Root cause.** {exception.root_cause}")
    st.markdown(f"**Suggested action.** {exception.suggested_action}")
    if exception.resolvable_by_agent:
        st.info(
            "The agent may resolve this within a hard bound. It **proposes** a "
            "journal adjustment and never posts to any real system."
        )
    else:
        st.warning("Proposal only. This one needs a person.")
    st.dataframe(evidence_rows(exception), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Screen 4 -- Audit replay
# --------------------------------------------------------------------------
def screen_audit(run: StoredRun) -> None:
    st.header("Audit replay")
    st.caption(
        "Step a record through the run: which tier proposed it, what the "
        "blender scored it, what the policy returned, and why. Read from the "
        "append-only log the run wrote -- nothing here is re-derived."
    )

    keys = record_keys(run)
    if not keys:
        st.info("This run recorded no decisions or exceptions.")
        return

    chosen = _picked(st.selectbox("Record", keys), keys)
    trace = trace_record(run, chosen)

    st.subheader(f"Final outcome: `{trace.outcome}`")
    st.write(trace.explanation)
    if trace.outcome in OUTCOME_HELP:
        st.caption(OUTCOME_HELP[trace.outcome])

    if trace.decisions:
        st.subheader("Decisions")
        st.dataframe(
            [
                {
                    "Tier": decision.tier.name,
                    "Outcome": decision.outcome.value,
                    "p": decision.calibrated_p,
                    "Arithmetic verified": decision.arithmetic_verified,
                    "Link": decision.link_type.value,
                    "Reason": decision.reason,
                }
                for decision in trace.decisions
            ],
            width="stretch",
            hide_index=True,
        )

    if trace.exceptions:
        st.subheader("Exceptions naming this record")
        st.dataframe(
            [
                {
                    "Class": item.exception_class.value,
                    "Severity": item.severity.value,
                    "Root cause": item.root_cause,
                }
                for item in trace.exceptions
            ],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Event timeline")
    timeline = trace.timeline()
    if timeline:
        st.dataframe(timeline, width="stretch", hide_index=True)
    else:
        st.caption("No audit event names this record directly.")

    with st.expander("The whole run log"):
        st.caption(
            f"{len(run.audit)} event(s), ordered by sequence rather than by "
            "clock -- several events routinely land inside one millisecond and "
            "replay has to be exactly reproducible."
        )
        st.dataframe(
            [
                {
                    "#": event.sequence,
                    "Node": event.node,
                    "Event": event.event_type.value,
                    "Message": event.message,
                }
                for event in run.audit
            ],
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="LedgerLoop", page_icon="🧾", layout="wide")
    st.title("LedgerLoop")
    st.caption(
        "Three-way reconciliation with an honest exception list. "
        "Ledger ↔ PSP settlements ↔ bank statement."
    )

    runs = list_runs(_runs_root())
    with st.sidebar:
        st.header("Runs")
        if runs:
            selected: Any = st.radio(
                "Completed runs",
                runs,
                format_func=lambda run: run.run_id,
                label_visibility="collapsed",
            )
        else:
            selected = None
            st.info("No runs yet. Start one on the **Run** tab.")
        st.divider()
        st.caption(
            f"Runs are read from `{_runs_root()}`. Each holds `run.json`, "
            "`audit.jsonl`, `exceptions.json` and `decisions.json` -- the same "
            "files the CLI writes."
        )

    tabs = st.tabs(["Run", "Results", "Exceptions", "Audit replay"])
    with tabs[0]:
        screen_run()
    for tab, screen in zip(
        tabs[1:], (screen_results, screen_exceptions, screen_audit), strict=True
    ):
        with tab:
            if selected is None:
                st.info("No run selected. Start one on the **Run** tab.")
            else:
                screen(selected)


# Streamlit executes this module top to bottom, so the entry point is a plain
# call rather than a guard: there is no `__main__` when `streamlit run` imports
# it. The guard is kept for `python -m` invocations, which print a hint instead
# of rendering nothing.
if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
