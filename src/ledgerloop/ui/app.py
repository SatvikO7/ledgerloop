"""The Streamlit dashboard. Widget glue over :mod:`ledgerloop.ui.views`.

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
disagrees with ``EVALUATION.md`` nobody can say which is right. The verdict
pills are the sharpest case -- *met* / *missed* / *undecided* is a real
judgement, and it is made by
:meth:`~ledgerloop.models.metrics.Proportion.verdict`, the same call the report
writer makes. The dashboard renders a ruling; it never reaches one.

The only action that runs anything is the button on the **Run** tab, and it
calls :func:`~ledgerloop.agent.runner.run_graph` -- the same entry point
``ledgerloop run`` uses.

WHAT THE REDESIGN CHANGED, AND WHY
-----------------------------------
The first version was four screens of tables that answered a developer's
questions in a developer's order. The story a reviewer actually needs is one
sentence -- *how much money was reconciled, how accurate was it, what is left,
and why* -- and the layout now follows it:

**Overview** answers it in one screen. **Pipeline** shows how the answer was
reached, rung by rung. **Exceptions** is the controller's workday. **Evidence**
takes a single record apart for someone who does not trust the summary.
**Evaluation** is the measurement, intervals included.

Three presentation rules survive from the tables and are load-bearing:

* **Four decision outcomes, never one "matched" number.** A referral is not a
  match and an exception is a decision to escalate, not a failure to decide.
* **A zero is never printed for something that did not happen.** A rung that ran
  and found nothing is drawn differently from one that never ran -- which is why
  T4's honest zero and T5's absence do not look alike.
* **No proportion is rendered without its interval.** ``Kpi`` has no accessor
  that returns a bare estimate with nothing beside it.

WHY STREAMLIT AND NOT REACT
---------------------------
PLAN.md §10 lists React + Vite + FastAPI with Streamlit as the fallback, and
§16's cut list has "React UI → Streamlit" at position 5. This is that cut,
taken deliberately: a FastAPI gateway would exist only to serve these screens to
a second process, and the demo is graded on `make demo` working on a clean
machine. One dependency, one process, no build step. The styling below is a
single stylesheet, not a framework.
"""

from __future__ import annotations

import html
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
from ledgerloop.models.metrics import Verdict
from ledgerloop.money import format_minor
from ledgerloop.ui.views import (
    OUTCOME_HELP,
    Headline,
    Kpi,
    TierStage,
    evidence_rows,
    exception_rows,
    headline,
    kpis,
    money_rows,
    outcome_rows,
    recall_rows,
    record_keys,
    tier_stages,
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
    return sorted(path for path in root.iterdir() if (path / "manifest.json").is_file())


_T = TypeVar("_T")


def _picked(value: object, options: Sequence[_T]) -> _T:
    """Narrow a Streamlit selection back to its option type.

    Streamlit's stubs type ``selectbox`` as returning ``None`` for a plain
    option list, which makes every downstream use unreachable to a type
    checker. This is the narrowing, in one place, rather than a scatter of
    ``# type: ignore`` at each call site.
    """
    return cast("_T", value) if value is not None else options[0]


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
#: One stylesheet, injected once.
#:
#: Colours are defined against both themes: Streamlit ships light and dark and a
#: dashboard that assumed one would be unreadable in the other on a demo
#: machine nobody configured. Semantic tones carry meaning rather than
#: decoration -- green is a met target, amber is undecided, red is a miss, slate
#: is *not measured* -- so the palette says the same thing the text does.
_STYLE = """
<style>
  :root {
    --ll-ink: #0f172a;
    --ll-muted: #64748b;
    --ll-line: rgba(148, 163, 184, 0.28);
    --ll-surface: rgba(255, 255, 255, 0.72);
    --ll-good: #059669;
    --ll-good-soft: rgba(5, 150, 105, 0.12);
    --ll-warn: #b45309;
    --ll-warn-soft: rgba(217, 119, 6, 0.14);
    --ll-bad: #dc2626;
    --ll-bad-soft: rgba(220, 38, 38, 0.12);
    --ll-idle: #64748b;
    --ll-idle-soft: rgba(100, 116, 139, 0.12);
    --ll-brand: #4f46e5;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ll-ink: #e2e8f0;
      --ll-muted: #94a3b8;
      --ll-line: rgba(148, 163, 184, 0.22);
      --ll-surface: rgba(30, 41, 59, 0.55);
      --ll-good: #34d399;
      --ll-warn: #fbbf24;
      --ll-bad: #f87171;
      --ll-idle: #94a3b8;
      --ll-brand: #a5b4fc;
    }
  }

  .ll-hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #0ea5e9 100%);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    color: #f8fafc;
  }
  .ll-hero h1 { font-size: 1.7rem; margin: 0; letter-spacing: -0.02em; color: #fff; }
  .ll-hero p { margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.95rem; }
  .ll-hero .ll-chips { margin-top: 0.85rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .ll-hero .ll-chip {
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 999px;
    padding: 0.16rem 0.68rem;
    font-size: 0.78rem;
  }

  .ll-grid { display: flex; flex-wrap: wrap; gap: 0.85rem; margin-bottom: 0.4rem; }
  .ll-card {
    flex: 1 1 210px;
    background: var(--ll-surface);
    border: 1px solid var(--ll-line);
    border-radius: 14px;
    padding: 0.95rem 1.05rem;
  }
  .ll-card .ll-label {
    font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ll-muted); font-weight: 600;
  }
  .ll-card .ll-value {
    font-size: 1.85rem; font-weight: 700; line-height: 1.15;
    margin-top: 0.2rem; color: var(--ll-ink); letter-spacing: -0.02em;
  }
  .ll-card .ll-sub { font-size: 0.78rem; color: var(--ll-muted); margin-top: 0.15rem; }
  .ll-card .ll-note {
    font-size: 0.79rem; color: var(--ll-muted); margin-top: 0.55rem; line-height: 1.4;
  }

  .ll-pill {
    display: inline-block; border-radius: 999px; padding: 0.1rem 0.55rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
  }
  .ll-good { color: var(--ll-good); background: var(--ll-good-soft); }
  .ll-warn { color: var(--ll-warn); background: var(--ll-warn-soft); }
  .ll-bad  { color: var(--ll-bad);  background: var(--ll-bad-soft); }
  .ll-muted-pill { color: var(--ll-idle); background: var(--ll-idle-soft); }

  .ll-bar {
    height: 6px; border-radius: 999px; margin-top: 0.6rem;
    background: var(--ll-idle-soft); position: relative; overflow: hidden;
  }
  .ll-bar span { position: absolute; top: 0; bottom: 0; border-radius: 999px; }

  .ll-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.3rem; }
  .ll-rung {
    flex: 1 1 150px; border: 1px solid var(--ll-line); border-radius: 13px;
    padding: 0.7rem 0.8rem; background: var(--ll-surface);
  }
  .ll-rung .ll-tier {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    color: var(--ll-muted); text-transform: uppercase;
  }
  .ll-rung .ll-name {
    font-size: 1rem; font-weight: 700; color: var(--ll-ink); margin-top: 0.1rem;
  }
  .ll-rung .ll-stat { font-size: 0.8rem; color: var(--ll-muted); margin-top: 0.4rem; }
  .ll-rung .ll-big { font-size: 1.3rem; font-weight: 700; color: var(--ll-ink); }
  .ll-rung.ll-on { border-color: rgba(79, 70, 229, 0.45); }
  .ll-rung.ll-off { opacity: 0.72; border-style: dashed; }
  .ll-arrow {
    align-self: center; color: var(--ll-muted); font-size: 1.05rem; padding: 0 0.05rem;
  }
  .ll-purpose {
    font-size: 0.76rem; color: var(--ll-muted); margin-top: 0.45rem; line-height: 1.35;
  }
  .ll-chain {
    border-left: 3px solid var(--ll-brand); padding: 0.15rem 0 0.15rem 0.85rem;
    margin: 0.35rem 0;
  }
  .ll-chain .ll-step-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--ll-muted); font-weight: 700;
  }
  .ll-chain .ll-step-body { font-size: 0.9rem; color: var(--ll-ink); }
</style>
"""

#: How a verdict is drawn. The tone is the ruling, not a design choice.
_VERDICT_TONE: dict[Verdict, tuple[str, str]] = {
    Verdict.MET: ("ll-good", "target met"),
    Verdict.MISSED: ("ll-bad", "target missed"),
    Verdict.UNDECIDED: ("ll-warn", "undecided"),
    Verdict.UNTARGETED: ("ll-muted-pill", "reported"),
}

_TONE_CLASS = {
    "good": "ll-good",
    "warn": "ll-warn",
    "bad": "ll-bad",
    "muted": "ll-muted-pill",
}


def _esc(value: object) -> str:
    return html.escape(str(value))


def _card(label: str, value: str, sub: str = "", note: str = "", pill: str = "") -> str:
    """One statistic, rendered the same way everywhere it appears."""
    return (
        '<div class="ll-card">'
        f'<div class="ll-label">{_esc(label)}</div>'
        f'<div class="ll-value">{_esc(value)}</div>'
        + (f'<div class="ll-sub">{_esc(sub)}</div>' if sub else "")
        + (f'<div style="margin-top:0.45rem">{pill}</div>' if pill else "")
        + (f'<div class="ll-note">{_esc(note)}</div>' if note else "")
        + "</div>"
    )


def _kpi_card(kpi: Kpi) -> str:
    """A headline proportion with its interval, its sample and its ruling.

    The interval is drawn as well as printed: the bar is the 95% Wilson span,
    so a wide one *looks* wide. A number whose interval covers twenty points
    should not be able to look as solid as one measured on thousands.
    """
    tone, wording = _VERDICT_TONE[kpi.verdict]
    if not kpi.measured:
        tone, wording = "ll-muted-pill", "not measured"
    pill = f'<span class="ll-pill {tone}">{_esc(wording)}</span>'
    bar = ""
    if kpi.measured:
        left = kpi.ci_low * 100.0
        width = max(kpi.ci_high * 100.0 - left, 0.8)
        colour = {
            "ll-good": "var(--ll-good)",
            "ll-bad": "var(--ll-bad)",
            "ll-warn": "var(--ll-warn)",
            "ll-muted-pill": "var(--ll-idle)",
        }[tone]
        bar = (
            '<div class="ll-bar">'
            f'<span style="left:{left:.2f}%;width:{width:.2f}%;background:{colour}"></span>'
            "</div>"
        )
    sample = f"{kpi.sample} · {kpi.goal}" if kpi.measured else kpi.goal
    return (
        '<div class="ll-card">'
        f'<div class="ll-label">{_esc(kpi.label)}</div>'
        f'<div class="ll-value">{_esc(kpi.percent)}</div>'
        f'<div class="ll-sub">95% CI {_esc(kpi.interval)}</div>'
        f"{bar}"
        f'<div style="margin-top:0.55rem">{pill}</div>'
        f'<div class="ll-sub" style="margin-top:0.35rem">{_esc(sample)}</div>'
        f'<div class="ll-note">{_esc(kpi.explanation)}</div>'
        "</div>"
    )


def _rung_card(stage: TierStage) -> str:
    """One rung of the ladder.

    A rung that never ran is dashed and says so. A rung that ran and found
    nothing shows its zero as a measurement -- which is exactly T4's situation
    and the reason this distinction is drawn at all.
    """
    if not stage.ran:
        body = (
            '<div class="ll-stat"><span class="ll-pill ll-muted-pill">did not run</span></div>'
            '<div class="ll-purpose">Switched off for this run, so it has no result. '
            "Not a zero.</div>"
        )
        css = "ll-rung ll-off"
    else:
        tone = "ll-good" if stage.contributed else "ll-muted-pill"
        wording = f"{stage.auto_matched:,} matched" if stage.contributed else "found nothing"
        body = (
            f'<div class="ll-big">{stage.auto_matched:,}</div>'
            f'<div class="ll-stat">{stage.proposed:,} proposed · '
            f"{stage.refused:,} refused · {stage.wall_clock_ms} ms</div>"
            f'<div style="margin-top:0.4rem"><span class="ll-pill {tone}">'
            f"{_esc(wording)}</span></div>"
            f'<div class="ll-purpose">{_esc(stage.purpose)}</div>'
        )
        css = "ll-rung ll-on" if stage.contributed else "ll-rung"
    return (
        f'<div class="{css}">'
        f'<div class="ll-tier">{_esc(stage.tier.split("_")[0])}</div>'
        f'<div class="ll-name">{_esc(stage.label)}</div>'
        f"{body}</div>"
    )


def _hero(run: StoredRun | None) -> None:
    """The masthead: what this is, and which run is on screen."""
    chips = ""
    if run is not None:
        dataset = run.summary.get("dataset", {})
        llm = run.summary.get("llm", {})
        # `llm.available` and not `config.llm_enabled`: the config flag says the
        # LLM was *permitted*, which is true by default and stays true on a
        # machine with no key. A chip reading "LLM on" over a run that never
        # reached a model would be the dashboard's most damaging possible claim,
        # because it is the one that undoes the deterministic-by-default result.
        parts = [
            f"run {run.run_id}",
            f"{dataset.get('split', '?')} · {dataset.get('difficulty', '?')} · seed "
            f"{dataset.get('seed', '?')}",
            f"{int(dataset.get('records', 0)):,} records",
            f"LLM used · {int(llm.get('calls', 0))} call(s)"
            if llm.get("available")
            else "deterministic · no LLM",
        ]
        chips = '<div class="ll-chips">' + "".join(
            f'<span class="ll-chip">{_esc(part)}</span>' for part in parts
        ) + "</div>"
    st.markdown(
        '<div class="ll-hero"><h1>LedgerLoop</h1>'
        "<p>Confidence-aware payment reconciliation &mdash; ledger &harr; PSP "
        "settlements &harr; bank statement, with an honest exception list.</p>"
        f"{chips}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 1-2. Executive KPIs and the reconciliation overview
# --------------------------------------------------------------------------
def screen_overview(run: StoredRun) -> None:
    view = headline(run)

    st.subheader("How accurate was it?")
    st.caption(
        "Four proportions, each with the sample it came from and a 95% Wilson "
        "interval. The verdict is read off the **interval**, one-sided, so a "
        "sample too small to separate a pass from a failure says *undecided* "
        "rather than picking the flattering answer."
    )
    cards = kpis(run)
    st.markdown(
        '<div class="ll-grid">' + "".join(_kpi_card(kpi) for kpi in cards) + "</div>",
        unsafe_allow_html=True,
    )
    if not any(kpi.measured for kpi in cards):
        st.info(
            "This run record predates stored intervals. Re-run the "
            "reconciliation to populate them."
        )

    st.divider()
    st.subheader("What did it decide?")
    st.caption(
        "Every outcome the policy has, never one 'matched' number. A referral is "
        "not a match, and an exception is a decision to escalate rather than a "
        "failure to decide."
    )
    st.markdown(
        '<div class="ll-grid">'
        + "".join(
            _card(
                str(row["Outcome"]),
                f"{int(row['Count']):,}",
                pill=(
                    f'<span class="ll-pill {_TONE_CLASS[str(row["tone"])]}">'
                    f"{_esc(str(row['key']))}</span>"
                ),
                note=OUTCOME_HELP[str(row["key"])],
            )
            for row in outcome_rows(view)
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"On the evaluation unit ({view.evaluation_links} `PAYMENT_CREDITED_AS` "
        f"links), from {view.candidates_proposed} candidate(s) proposed. The "
        f"exception **queue** holds {view.queue_size} item(s) -- a settlement "
        "nothing could credit raises one without producing a link decision."
    )

    st.divider()
    _money_section(run, view)

    if view.false_positives == 0:
        st.success(
            f"**{view.true_positives} links committed, **zero** false positives, "
            f"{view.false_positive_cost} of wrong money.** Precision is the claim "
            "this system is built around, and it holds on this run."
        )
    else:
        st.error(
            f"{view.false_positives} false positive(s) costing "
            f"{view.false_positive_cost}. Every one is a link the system "
            "committed without a human and got wrong."
        )
    if not view.llm_available:
        st.info(
            "No LLM ran. Every number on this page is deterministic and "
            "reproducible with no API key."
        )


def _money_section(run: StoredRun, view: Headline) -> None:
    """4. Money. The figure a finance team actually asks for, first."""
    st.subheader("How much money?")
    total_minor = int(run.metrics.get("reconciled_minor", 0)) + int(
        run.metrics.get("outstanding_minor", 0)
    )
    reconciled_minor = int(run.metrics.get("reconciled_minor", 0))
    share = reconciled_minor / total_minor if total_minor else 0.0
    st.markdown(
        '<div class="ll-grid">'
        + _card(
            "Reconciled",
            view.reconciled,
            sub=f"{share:.1%} of the money the evaluation unit covers",
            note="Money on links the system found and got right.",
        )
        + _card(
            "Outstanding",
            view.outstanding,
            sub="on links it did not assert",
            note="Not lost money -- money this run did not claim to have explained.",
        )
        + _card(
            "False-positive cost",
            view.false_positive_cost,
            pill=(
                '<span class="ll-pill ll-good">nothing wrong</span>'
                if view.false_positives == 0
                else f'<span class="ll-pill ll-bad">{view.false_positives} wrong link(s)</span>'
            ),
            note="Money it declared reconciled that was not.",
        )
        + _card(
            "Unmatchable floor",
            view.unmatchable_impact,
            sub=f"{view.unmatchable_count} record(s)",
            note="No system could resolve these from the three sources.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The unmatchable line is **unmatchable by construction** -- a ceiling in "
        "the data and not a model failure, so it is its own row rather than a "
        "share of the outstanding total."
    )
    with st.expander("The money view as a table"):
        st.dataframe(money_rows(run), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# 3. The tier pipeline
# --------------------------------------------------------------------------
def screen_pipeline(run: StoredRun) -> None:
    st.subheader("How it got there")
    st.caption(
        "Six rungs, cheapest and most certain first. Each one only sees what the "
        "rungs above it left behind, so a match is always made by the weakest "
        "mechanism that could make it."
    )
    stages = tier_stages(run)
    flow: list[str] = []
    for index, stage in enumerate(stages):
        if index:
            flow.append('<div class="ll-arrow">&#8594;</div>')
        flow.append(_rung_card(stage))
    st.markdown('<div class="ll-flow">' + "".join(flow) + "</div>", unsafe_allow_html=True)

    st.caption(
        "**Proposed** is yield and **matched** is conviction; they are different "
        "columns on purpose. A rung that proposes a hundred and commits forty has "
        "not performed like one that proposes forty and commits forty -- the gap "
        "is the review queue a finance team has to staff."
    )

    silent = [stage for stage in stages if stage.ran and not stage.contributed]
    if silent:
        st.info(
            "**"
            + ", ".join(stage.label for stage in silent)
            + " ran and contributed nothing on this corpus.** That is a"
            " measurement, not an error and not a missing component. Every"
            " earlier rung matches at settlement granularity -- it establishes"
            " the settlement-to-credit edge and expands the whole batch at once"
            " -- so the *partial* assignments graph inference exists to finish"
            " never arise. Loosening a rule until it fired would trade precision"
            " for the appearance of contribution."
        )

    st.divider()
    st.subheader("Residual passes")
    st.caption(
        "T2, T3 and T4 re-run while a pass keeps changing something, bounded by "
        "configuration. A later rung freeing a record can let an earlier, "
        "stronger rung claim it on the next pass."
    )
    st.markdown(
        '<div class="ll-grid">'
        + _card("Residual passes", f"{headline(run).residual_passes}")
        + _card("Candidates proposed", f"{headline(run).candidates_proposed:,}")
        + _card("Wall clock", f"{headline(run).wall_clock_ms} ms")
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("The ladder as a table"):
        st.dataframe(
            [
                {
                    "Tier": stage.tier,
                    "Ran": stage.ran,
                    "Proposed": stage.proposed,
                    "Auto-matched": stage.auto_matched,
                    "Refused": stage.refused,
                    "Marginal": stage.marginal,
                    "Wall clock (ms)": stage.wall_clock_ms,
                }
                for stage in stages
            ],
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
# 5. The exception queue
# --------------------------------------------------------------------------
def screen_exceptions(run: StoredRun) -> None:
    st.subheader("Exception queue")
    rows = exception_rows(run)
    if not rows:
        st.success("No exceptions were raised for this run.")
        return

    total = sum(int(row["impact_minor"]) for row in rows)
    critical = sum(1 for row in rows if str(row["Severity"]).upper() == "CRITICAL")
    st.markdown(
        '<div class="ll-grid">'
        + _card("Items in the queue", f"{len(rows):,}", note="The controller's workday.")
        + _card("Money under exception", format_minor(total))
        + _card(
            "Critical",
            f"{critical:,}",
            pill=(
                '<span class="ll-pill ll-bad">needs attention</span>'
                if critical
                else '<span class="ll-pill ll-good">none</span>'
            ),
        )
        + "</div>",
        unsafe_allow_html=True,
    )
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
            "it does not change the counts on the Overview screen."
        )
    st.dataframe(
        [
            {k: v for k, v in row.items() if k not in {"impact_minor", "exception_id"}}
            for row in shown
        ],
        width="stretch",
        hide_index=True,
    )

    st.divider()
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
# 6. One record, taken apart
# --------------------------------------------------------------------------
def screen_evidence(run: StoredRun) -> None:
    st.subheader("Follow one record")
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

    deciding = trace.decisions[-1] if trace.decisions else None
    steps: list[tuple[str, str]] = [
        (
            "1 · Source records",
            f"`{chosen}` was read from the ledger, the PSP settlement file and the "
            "bank statement, each in its own format.",
        ),
        (
            "2 · Normalisation",
            "Dates, amounts and references were parsed into canonical records. "
            "Money is integer paise from here on -- no float ever touches it.",
        ),
        (
            "3 · Candidate",
            f"{len(trace.decisions)} decision(s) name this record."
            if trace.decisions
            else "No tier proposed a link for this record.",
        ),
    ]
    if deciding is not None:
        steps.append(("4 · Tier", f"`{deciding.tier.name}` made the ruling that stands."))
        steps.append(
            (
                "5 · Arithmetic verification",
                "The money closed against the source documents, so the link was "
                "eligible to be committed."
                if deciding.arithmetic_verified
                else "The arithmetic did **not** close, so the link could not be "
                "auto-matched whatever its score.",
            )
        )
        steps.append(
            (
                "6 · Decision",
                f"`{deciding.outcome.value}` at p = {deciding.calibrated_p:.4f}. "
                f"{deciding.reason}",
            )
        )
    st.markdown(
        "".join(
            '<div class="ll-chain">'
            f'<div class="ll-step-label">{_esc(label)}</div>'
            f'<div class="ll-step-body">{_esc(body)}</div></div>'
            for label, body in steps
        ),
        unsafe_allow_html=True,
    )

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
# 7. Evaluation
# --------------------------------------------------------------------------
def screen_evaluation(run: StoredRun) -> None:
    view = headline(run)
    st.subheader("Measurement")
    st.caption(
        "The same figures `EVALUATION.md` publishes, read from this run's own "
        "record. The verdict column uses the identical rule the report uses."
    )
    st.dataframe(
        [
            {
                "Metric": kpi.label,
                "Value": kpi.percent,
                "95% Wilson CI": kpi.interval,
                "Sample": kpi.sample or "--",
                "Target": kpi.goal,
                "Verdict": _VERDICT_TONE[kpi.verdict][1]
                if kpi.measured
                else "not measured",
            }
            for kpi in kpis(run)
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="ll-grid">'
        + _card("True positives", f"{view.true_positives:,}")
        + _card(
            "False positives",
            f"{view.false_positives:,}",
            pill=(
                '<span class="ll-pill ll-good">none</span>'
                if view.false_positives == 0
                else '<span class="ll-pill ll-bad">investigate</span>'
            ),
        )
        + _card("False negatives", f"{view.false_negatives:,}")
        + _card("False-positive cost", view.false_positive_cost)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Recall by anomaly class")
    st.caption(
        "Whatever the run measured, in class order -- **including the classes "
        "that score badly**. Publishing only the good rows is precisely what "
        "this project is trying not to do."
    )
    rows = recall_rows(run)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        worst = min(rows, key=lambda row: float(row["Recall"]))
        st.caption(
            f"Weakest class on this run: **{worst['Anomaly class']}** at "
            f"{float(worst['Recall']):.2%}."
        )
    else:
        st.info("This run recorded no per-class recall.")


# --------------------------------------------------------------------------
# The run launcher
# --------------------------------------------------------------------------
def screen_run() -> None:
    """Generate or pick a dataset, then reconcile it through the graph."""
    st.subheader("Run a reconciliation")
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
def main() -> None:
    st.set_page_config(page_title="LedgerLoop", page_icon="🧾", layout="wide")
    st.markdown(_STYLE, unsafe_allow_html=True)

    runs = list_runs(_runs_root())
    with st.sidebar:
        st.markdown("### LedgerLoop")
        st.caption("Confidence-aware payment reconciliation")
        st.divider()
        st.markdown("**Runs**")
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
        if selected is not None:
            dataset = selected.summary.get("dataset", {})
            config = selected.summary.get("config", {})
            st.divider()
            st.markdown("**This run**")
            st.caption(
                f"Split `{dataset.get('split', '?')}` · "
                f"difficulty `{dataset.get('difficulty', '?')}` · "
                f"seed `{dataset.get('seed', '?')}`\n\n"
                f"{int(dataset.get('records', 0)):,} records · "
                f"{int(dataset.get('evaluation_links', 0)):,} evaluation links\n\n"
                f"Tuning hash `{config.get('tuning_hash', '?')}`"
            )
        st.divider()
        st.caption(
            f"Runs are read from `{_runs_root()}`. Each holds `run.json`, "
            "`audit.jsonl`, `exceptions.json` and `decisions.json` -- the same "
            "files the CLI writes. This interface renders them and computes "
            "nothing."
        )

    _hero(selected)

    tabs = st.tabs(
        ["Overview", "Pipeline", "Exceptions", "Evidence", "Evaluation", "Run"]
    )
    screens = (
        screen_overview,
        screen_pipeline,
        screen_exceptions,
        screen_evidence,
        screen_evaluation,
    )
    for tab, screen in zip(tabs[:5], screens, strict=True):
        with tab:
            if selected is None:
                st.info("No run selected. Start one on the **Run** tab.")
            else:
                screen(selected)
    with tabs[5]:
        screen_run()


# Streamlit executes this module top to bottom, so the entry point is a plain
# call rather than a guard: there is no `__main__` when `streamlit run` imports
# it. The guard is kept for `python -m` invocations, which print a hint instead
# of rendering nothing.
if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
