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

The only action that runs anything is the button on the **New report** tab, and it
calls :func:`~ledgerloop.agent.runner.run_graph` -- the same entry point
``ledgerloop run`` uses.

WHO THE SCREENS ARE FOR
-----------------------
The layout answers a **non-technical** reader's questions, in the order they ask
them, and puts the developer's vocabulary one tab away:

* **Overview** -- how many matched, how much money, what needs a person, was
  anything matched wrongly. Four numbers, about five seconds.
* **Needs review** -- the queue, led by the rupee figure, with the
  classifier's own plain-English cause and action.
* **Transactions** -- every decision, filterable and searchable.
* **Why it matched** -- one record, explained without jargon.
* **Accuracy & details** -- the measurement in ``EVALUATION.md``'s vocabulary:
  KPIs with their intervals and verdicts, the six-rung ladder, per-class recall,
  the money table and the full audit replay. **Nothing was deleted to make the
  other screens clean**; it was moved, and a glossary sits at the top of it
  translating every term it uses.

The plain wording lives in :mod:`ledgerloop.ui.plain`, which translates the
stored run and computes nothing -- so the two vocabularies cannot become two
sets of numbers.

Three presentation rules survive from the tables and are load-bearing:

* **Four decision outcomes, never one "matched" number.** A referral is not a
  match and an exception is a decision to escalate, not a failure to decide.
* **A zero is never printed for something that did not happen.** A rung that ran
  and found nothing is drawn differently from one that never ran -- which is why
  T4's honest zero and T5's absence do not look alike.
* **No proportion is rendered without its interval.** ``Kpi`` has no accessor
  that returns a bare estimate with nothing beside it.
* **The plain screens never claim more than was measured.** The overview says
  *0 incorrect matches*, which is a count. Whether that clears a 99% target at
  this sample size is a statistical ruling, and it stays in the report beside
  its interval where it can be read properly.

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
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict, TypeVar, cast

import streamlit as st

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.store import RUNS_ROOT, StoredRun, list_runs
from ledgerloop.config import GeneratorConfig, LLMConfig
from ledgerloop.envfile import load_env_file
from ledgerloop.eval.harness import ReconcileResult, reconcile_only
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest.problems import IngestError
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.providers import build_ladder
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, Difficulty, SplitName
from ledgerloop.models.metrics import Verdict
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.money import format_minor
from ledgerloop.ui.plain import (
    AttentionItem,
    Bucket,
    JourneyStep,
    assistant_activity,
    attention_items,
    attention_items_from,
    buckets,
    glossary,
    journey,
    match_story_from,
    report_labels,
    safety_note,
    snapshot,
    status_of,
    transaction_rows,
    transaction_rows_from,
    transaction_search,
)
from ledgerloop.ui.uploads import (
    SourceKind,
    UploadProblem,
    assess,
    detect,
    row_count,
    upload_snapshot,
    upload_tier_stages,
    validate,
)
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
    --ll-line: rgba(148, 163, 184, 0.30);
    --ll-surface: #ffffff;
    --ll-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 14px rgba(15, 23, 42, 0.05);
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
      --ll-shadow: none;
      --ll-good: #34d399;
      --ll-warn: #fbbf24;
      --ll-bad: #f87171;
      --ll-idle: #94a3b8;
      --ll-brand: #a5b4fc;
    }
  }

  .ll-hero {
    background: linear-gradient(115deg, #4338ca 0%, #4f46e5 55%, #0284c7 100%);
    border-radius: 18px;
    padding: 1.05rem 1.5rem;
    margin-bottom: 0.9rem;
    color: #f8fafc;
  }
  .ll-hero h1 { font-size: 1.5rem; margin: 0; letter-spacing: -0.02em; color: #fff; }
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

  /* --- the plain-language screens ------------------------------------- */
  /* Bigger, fewer, and colour-coded by meaning. A reconciliation lead should
     be able to read the four numbers that matter from across a room, so these
     cards are deliberately not the same size as the technical ones. */
  .ll-big-grid { display: flex; flex-wrap: wrap; gap: 0.9rem; margin: 0.2rem 0 0.6rem; }
  .ll-big-card {
    flex: 1 1 210px; border-radius: 16px; padding: 1.2rem 1.3rem;
    border: 1px solid var(--ll-line); background: var(--ll-surface);
    box-shadow: var(--ll-shadow);
  }
  .ll-big-card .ll-top {
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.01em;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .ll-big-card .ll-num {
    font-size: 2.5rem; font-weight: 800; line-height: 1.05;
    margin: 0.35rem 0 0.1rem; color: var(--ll-ink); letter-spacing: -0.03em;
  }
  .ll-big-card .ll-money {
    font-size: 1.75rem; font-weight: 800; line-height: 1.15;
    margin: 0.35rem 0 0.1rem; color: var(--ll-ink); letter-spacing: -0.02em;
    overflow-wrap: anywhere;
  }
  .ll-big-card .ll-cap { font-size: 0.84rem; color: var(--ll-muted); line-height: 1.35; }
  .ll-big-card.ll-t-good { border-color: rgba(5, 150, 105, 0.42); background: var(--ll-good-soft); }
  .ll-big-card.ll-t-good .ll-top { color: var(--ll-good); }
  .ll-big-card.ll-t-warn { border-color: rgba(217, 119, 6, 0.42); background: var(--ll-warn-soft); }
  .ll-big-card.ll-t-warn .ll-top { color: var(--ll-warn); }
  .ll-big-card.ll-t-bad  { border-color: rgba(220, 38, 38, 0.42); background: var(--ll-bad-soft); }
  .ll-big-card.ll-t-bad  .ll-top { color: var(--ll-bad); }
  .ll-big-card.ll-t-brand { border-color: rgba(79, 70, 229, 0.42); }
  .ll-big-card.ll-t-brand .ll-top { color: var(--ll-brand); }
  .ll-big-card.ll-t-muted .ll-top { color: var(--ll-muted); }

  /* The where-did-everything-go picture. One row, three destinations. */
  .ll-split { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.3rem; }
  .ll-dest {
    flex: 1 1 190px; border-radius: 14px; padding: 0.85rem 1rem;
    border: 1px solid var(--ll-line); background: var(--ll-surface);
  }
  .ll-dest .ll-dest-head { font-size: 0.9rem; font-weight: 700; }
  .ll-dest .ll-dest-num {
    font-size: 2rem; font-weight: 800; color: var(--ll-ink); line-height: 1.1;
    margin: 0.2rem 0 0.25rem; letter-spacing: -0.02em;
  }
  .ll-dest .ll-dest-note { font-size: 0.8rem; color: var(--ll-muted); line-height: 1.35; }
  .ll-dest.ll-t-good { border-color: rgba(5, 150, 105, 0.4); }
  .ll-dest.ll-t-good .ll-dest-head { color: var(--ll-good); }
  .ll-dest.ll-t-warn { border-color: rgba(217, 119, 6, 0.4); }
  .ll-dest.ll-t-warn .ll-dest-head { color: var(--ll-warn); }
  .ll-dest.ll-t-muted .ll-dest-head { color: var(--ll-muted); }

  /* The safety banner. The project's strongest property, stated once, large. */
  .ll-safe {
    border-radius: 16px; padding: 1.1rem 1.3rem; margin: 0.5rem 0 0.2rem;
    border: 1px solid rgba(5, 150, 105, 0.42); background: var(--ll-good-soft);
  }
  .ll-safe.ll-t-bad { border-color: rgba(220, 38, 38, 0.45); background: var(--ll-bad-soft); }
  .ll-safe .ll-safe-title {
    font-size: 1.45rem; font-weight: 800; color: var(--ll-good); letter-spacing: -0.02em;
  }
  .ll-safe.ll-t-bad .ll-safe-title { color: var(--ll-bad); }
  .ll-safe .ll-safe-body {
    font-size: 0.95rem; color: var(--ll-ink); margin-top: 0.3rem; line-height: 1.45;
  }

  /* One queue item, led by the money and closed by the action. */
  .ll-item {
    border: 1px solid var(--ll-line); border-left: 4px solid var(--ll-warn);
    border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.55rem;
    background: var(--ll-surface);
  }
  .ll-item .ll-item-amt {
    font-size: 1.35rem; font-weight: 800; color: var(--ll-ink); letter-spacing: -0.02em;
  }
  .ll-item .ll-item-sub { font-size: 0.8rem; color: var(--ll-muted); }
  .ll-item .ll-item-body {
    font-size: 0.92rem; color: var(--ll-ink); margin-top: 0.5rem; line-height: 1.5;
  }
  .ll-item .ll-item-act {
    font-size: 0.9rem; margin-top: 0.5rem; padding-top: 0.5rem;
    border-top: 1px dashed var(--ll-line); color: var(--ll-ink); line-height: 1.45;
  }
  .ll-item.ll-sev-critical { border-left-color: var(--ll-bad); }
  .ll-item.ll-sev-low { border-left-color: var(--ll-idle); }

  /* The "why we matched it" reason list. Ticks, not bullet points. */
  .ll-reasons { list-style: none; padding-left: 0; margin: 0.4rem 0 0; }
  .ll-reasons li {
    font-size: 0.95rem; color: var(--ll-ink); line-height: 1.5;
    padding: 0.25rem 0 0.25rem 1.6rem; position: relative;
  }
  .ll-reasons li::before {
    content: "✓"; position: absolute; left: 0; top: 0.25rem;
    color: var(--ll-good); font-weight: 800;
  }
  .ll-reasons.ll-open li::before { content: "•"; color: var(--ll-muted); }

  /* The verdict, first thing on the page and larger than anything else. */
  .ll-status {
    border-radius: 18px; padding: 1.35rem 1.5rem; margin: 0.1rem 0 1.1rem;
    border: 1px solid rgba(5, 150, 105, 0.35); background: var(--ll-good-soft);
  }
  .ll-status.ll-t-bad { border-color: rgba(220, 38, 38, 0.4); background: var(--ll-bad-soft); }
  /* Amber is a real state here: "can be read, but not reconciled" is neither a
     success nor a failure, and rendering it in success green -- which is what
     happened before this rule existed -- told the reader the opposite of the
     sentence they were reading. */
  .ll-status.ll-t-warn {
    border-color: rgba(217, 119, 6, 0.42); background: var(--ll-warn-soft);
  }
  .ll-status.ll-t-warn .ll-status-title { color: var(--ll-warn); }
  .ll-status .ll-status-title {
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.025em; color: var(--ll-good);
  }
  .ll-status.ll-t-bad .ll-status-title { color: var(--ll-bad); }
  .ll-status .ll-status-money {
    font-size: 1.5rem; font-weight: 800; color: var(--ll-ink);
    margin-top: 0.6rem; letter-spacing: -0.02em;
  }
  .ll-status .ll-status-money span {
    font-size: 0.85rem; font-weight: 600; color: var(--ll-muted);
    letter-spacing: 0; margin-left: 0.3rem;
  }
  .ll-status .ll-status-body {
    font-size: 1rem; color: var(--ll-ink); margin-top: 0.35rem;
    line-height: 1.5; max-width: 70ch;
  }

  /* Payment -> bank -> settlement -> reconciled, as a vertical path. */
  .ll-journey {
    border: 1px solid var(--ll-line); border-radius: 14px; padding: 1rem 1.1rem;
    background: var(--ll-surface); box-shadow: var(--ll-shadow); height: 100%;
  }
  .ll-journey .ll-jhead {
    font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ll-muted); font-weight: 700; margin-bottom: 0.7rem;
  }
  .ll-jstep {
    border-radius: 10px; padding: 0.5rem 0.75rem;
    border: 1px solid var(--ll-line); background: var(--ll-idle-soft);
  }
  .ll-jstep .ll-jlabel { font-size: 0.95rem; font-weight: 700; color: var(--ll-ink); }
  .ll-jstep .ll-jnote { font-size: 0.78rem; color: var(--ll-muted); margin-top: 0.1rem; }
  .ll-jstep.ll-t-good {
    border-color: rgba(5, 150, 105, 0.42); background: var(--ll-good-soft);
  }
  .ll-jstep.ll-t-good .ll-jlabel { color: var(--ll-good); }
  .ll-jstep.ll-t-warn {
    border-color: rgba(217, 119, 6, 0.42); background: var(--ll-warn-soft);
  }
  .ll-jstep.ll-t-warn .ll-jlabel { color: var(--ll-warn); }
  .ll-jarrow {
    text-align: center; color: var(--ll-muted); font-size: 1rem; line-height: 1.2;
    padding: 0.15rem 0;
  }

  /* An optional source. Present or not; never a requirement. */
  .ll-drop {
    border: 1px dashed var(--ll-line); border-radius: 14px;
    padding: 0.85rem 1rem; background: var(--ll-surface); margin-bottom: 0.5rem;
  }
  .ll-drop.ll-have { border-style: solid; border-color: rgba(5, 150, 105, 0.45); }
  .ll-drop .ll-drop-title { font-size: 1rem; font-weight: 700; color: var(--ll-ink); }
  .ll-drop .ll-drop-note {
    font-size: 0.82rem; color: var(--ll-muted); margin-top: 0.2rem; line-height: 1.4;
  }
  .ll-drop .ll-drop-type {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--ll-muted); font-weight: 700; margin-top: 0.45rem;
  }

  .ll-lede {
    font-size: 1.02rem; color: var(--ll-ink); line-height: 1.55;
    max-width: 62ch; margin: 0.2rem 0 0.9rem;
  }
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


#: Why a rung that ran found nothing, per rung. A zero explained by the wrong
#: reason is worse than an unexplained one.
_SILENT_REASON: dict[str, str] = {
    "T4_GRAPH": (
        "That is a measurement, not an error and not a missing component. Every "
        "earlier rung matches at settlement granularity -- it establishes the "
        "settlement-to-credit edge and expands the whole batch at once -- so the "
        "*partial* assignments graph inference exists to finish never arise. "
        "Loosening a rule until it fired would trade precision for the "
        "appearance of contribution."
    ),
    "T5_LLM": (
        "It is only ever offered **the settlements the ladder could not credit** "
        "-- not the review queue as a whole. Almost everything in that queue is a "
        "*finding* rather than a missing link: a payout the bank posted twice, a "
        "chargeback whose money never arrived, a record nothing in the three "
        "files could resolve. There is no link for a model to find in any of "
        "those, and the right answer to a duplicate credit is to raise it with "
        "the bank, not to reason about it. So a zero here usually means the "
        "residual it *can* work on was empty or tiny -- and when it is not, a "
        "proposal still has to survive the grounding check and `verify_"
        "arithmetic` before it counts."
    ),
}

_SILENT_DEFAULT = (
    "That is a measurement rather than an error: the rung ran and found nothing "
    "it could assert."
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


def _hero(run: StoredRun | None, *, chip: str | None = None) -> None:
    """The masthead: what this is, and which run is on screen.

    ``chip`` overrides the provenance line, for a subject that is not a stored
    run. The chip must never describe a different subject from the one on the
    tabs -- a masthead reading "no AI model used" over an upload that called one
    is the same lie in the other direction.
    """
    chips = ""
    if chip is not None:
        chips = f'<div class="ll-chips"><span class="ll-chip">{_esc(chip)}</span></div>'
    elif run is not None:
        llm = run.summary.get("llm", {})
        # ONE chip, and it is about provenance rather than identity.
        #
        # No run id, no split, no seed: those identify the report for someone
        # reproducing a figure and mean nothing to anyone else, so they live in
        # the sidebar's details expander. No record count either -- "742 read"
        # sat a few centimetres above the Overview's "345 checked", both true,
        # different denominators, and confusing side by side.
        #
        # What survives is `llm.available`, not `config.llm_enabled`: the config
        # flag says the LLM was *permitted*, which is true by default and stays
        # true on a machine with no key. A chip claiming a model was involved in
        # a run that never reached one would be the most damaging thing this
        # dashboard could say, because it is the claim that undoes the
        # deterministic-by-default result.
        parts = [
            f"AI assistant used on {int(llm.get('calls', 0))} item(s)"
            if llm.get("available")
            else "No AI model used - every figure is repeatable",
        ]
        chips = '<div class="ll-chips">' + "".join(
            f'<span class="ll-chip">{_esc(part)}</span>' for part in parts
        ) + "</div>"
    st.markdown(
        '<div class="ll-hero"><h1>LedgerLoop</h1>'
        "<p>Automatic payment reconciliation — it compares your payments, "
        "your bank transactions and your settlement records, and only matches "
        "what it can prove.</p>"
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

    # One reason per rung, not one reason for all of them.
    #
    # This used to print T4's argument -- "partial assignments never arise" --
    # over whichever rungs happened to be silent, T5 included. That is a true
    # sentence about graph inference and a false one about the model, and a
    # panel that explains a zero with the wrong reason is worse than one that
    # says nothing: it invites a reader to distrust the rungs that *did* fire.
    silent = [stage for stage in stages if stage.ran and not stage.contributed]
    for stage in silent:
        st.info(f"**{stage.label} ran and contributed nothing on this corpus.** "
                + _SILENT_REASON.get(stage.tier, _SILENT_DEFAULT))

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
        columns[0].selectbox("Severity", severity_options, key="q_sev"), severity_options
    )
    exception_class = _picked(
        columns[1].selectbox("Class", class_options, key="q_class"), class_options
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

    chosen = _picked(st.selectbox("Record", keys, key="audit_record"), keys)
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
# The plain-language screens
#
# Everything above this line is the technical view and is unchanged. These
# screens render the *same stored run* for a reader who has never met a tier, a
# Wilson interval or a residual pass. Nothing here recomputes anything: the
# translation lives in `ui/plain.py`, and every number comes off the run record.
# --------------------------------------------------------------------------
_TONE_SUFFIX = {
    "good": "ll-t-good",
    "warn": "ll-t-warn",
    "bad": "ll-t-bad",
    "brand": "ll-t-brand",
    "muted": "ll-t-muted",
}

_TICK = "✓"
_WARN = "⚠"
_RING = "○"
_SHIELD = "\U0001f6e1"
_RUPEE = "₹"
_FOLDER = "📁"


def _big_card(
    icon: str, label: str, value: str, caption: str, tone: str, *, money: bool = False
) -> str:
    """One headline number, large enough to read from across a room."""
    size = "ll-money" if money else "ll-num"
    return (
        f'<div class="ll-big-card {_TONE_SUFFIX[tone]}">'
        f'<div class="ll-top">{_esc(icon)} {_esc(label)}</div>'
        f'<div class="{size}">{_esc(value)}</div>'
        f'<div class="ll-cap">{_esc(caption)}</div>'
        "</div>"
    )


def _destination(bucket: Bucket) -> str:
    icon = {"check": _TICK, "review": _WARN, "open": _RING}[bucket.icon]
    return (
        f'<div class="ll-dest {_TONE_SUFFIX[bucket.tone]}">'
        f'<div class="ll-dest-head">{_esc(icon)} {_esc(bucket.label)}</div>'
        f'<div class="ll-dest-num">{bucket.count:,}</div>'
        f'<div class="ll-dest-note">{_esc(bucket.note)}</div>'
        "</div>"
    )


#: The ladder in a reader's words. Deliberately five steps and not six: T4 and
#: T5 are collapsed into "left for review" here because a non-technical reader
#: cares where an item *ends up*, not which of two internal rungs declined it.
#: The full six-rung measurement is one tab away and is not softened there.
_PLAIN_STEPS: tuple[tuple[str, str, str], ...] = (
    (
        "1",
        "Exact match",
        "The payment reference is on the bank transaction and the amounts agree exactly.",
    ),
    (
        "2",
        "Close match",
        "The reference agrees and the amounts line up once the processor's fee is "
        "allowed for.",
    ),
    (
        "3",
        "Grouped payout",
        "Many payments paid out as one bank credit, worked back to the individual "
        "payments.",
    ),
    (
        "4",
        "Name and amount",
        "No reference on the bank line, so the business name and the exact amount "
        "are used instead.",
    ),
    (
        "5",
        "Left for review",
        "Not enough evidence. It goes to a person with a reason, rather than being "
        "guessed.",
    ),
)


def _journey_row(steps: list[JourneyStep], heading: str) -> str:
    """One path through the three files, drawn as arrows."""
    cells: list[str] = []
    for index, step in enumerate(steps):
        if index:
            cells.append('<div class="ll-jarrow">&#8595;</div>')
        cells.append(
            f'<div class="ll-jstep {_TONE_SUFFIX[step.tone]}">'
            f'<div class="ll-jlabel">{_esc(step.label)}</div>'
            f'<div class="ll-jnote">{_esc(step.note)}</div>'
            "</div>"
        )
    return (
        '<div class="ll-journey">'
        f'<div class="ll-jhead">{_esc(heading)}</div>'
        + "".join(cells)
        + "</div>"
    )


def screen_home(run: StoredRun) -> None:
    """The hero screen. What happened, was it safe, and what to do next."""
    view = snapshot(run)
    status = status_of(run)

    # 1. The verdict, before any number. A reader who reads nothing else should
    #    still leave knowing whether this reconciliation can be relied on.
    st.markdown(
        f'<div class="ll-status {_TONE_SUFFIX[status.tone]}">'
        f'<div class="ll-status-title">{_esc(_TICK if status.tone == "good" else _WARN)} '
        f"{_esc(status.title)}</div>"
        f'<div class="ll-status-body">{_esc(status.body)}</div>'
        f'<div class="ll-status-money">{_esc(view.reconciled)} '
        "<span>reconciled</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 2. Four numbers. Where the run stored the match-rate denominator these are
    #    one unit and genuinely add up -- checked = matched + needs review --
    #    which is why `counts_add_up` is checked rather than assumed. An older
    #    report without it falls back to what it does have, and says so.
    if view.counts_add_up:
        checked, resolved, unresolved = view.checked, view.resolved, view.unresolved
        assert checked is not None and resolved is not None and unresolved is not None
        cards = (
            _big_card("", "Transactions checked", f"{checked:,}",
                      "records that could be reconciled from your files", "brand")
            + _big_card(_TICK, "Successfully matched", f"{resolved:,}",
                        "settled automatically, with evidence", "good")
            + _big_card(_WARN, "Need review", f"{unresolved:,}",
                        "left for a person to decide", "warn")
            + _big_card(_SHIELD, "Incorrect matches", f"{view.incorrect:,}",
                        "matched when they should not have been",
                        "good" if view.is_clean else "bad")
        )
    else:
        cards = (
            _big_card("", "Transactions checked", f"{view.records:,}",
                      "records read from your three files", "brand")
            + _big_card(_TICK, "Successfully matched", f"{view.matched:,}",
                        "settled automatically, with evidence", "good")
            + _big_card(_WARN, "Need review", f"{view.needs_attention:,}",
                        "left for a person to decide", "warn")
            + _big_card(_SHIELD, "Incorrect matches", f"{view.incorrect:,}",
                        "matched when they should not have been",
                        "good" if view.is_clean else "bad")
        )
    st.markdown(f'<div class="ll-big-grid">{cards}</div>', unsafe_allow_html=True)
    if not view.counts_add_up:
        st.caption(
            "This report predates stored transaction counts, so the first two "
            "figures are read from different totals and do not add up. Newer "
            "reports show a single consistent count."
        )

    # 3. Why that zero is a zero. The verdict says the run finished safely; this
    #    says *how* -- by refusing rather than guessing -- which is the product's
    #    actual argument. A caption rather than a second banner: two stacked
    #    green boxes said nearly the same thing twice and pushed the numbers
    #    below the fold.
    title, body = safety_note(run)
    st.caption(f"**{title}.** {body}")

    # 4. What to do next. A pointer, not a button: Streamlit cannot switch tabs
    #    programmatically, and a control that looked clickable and did nothing
    #    would be worse than a sentence.
    if view.needs_attention:
        st.info(
            f"**Next:** open **Needs review** to work through the queue -- "
            f"{view.needs_attention:,} flagged item(s), largest amount first. "
            "Each one says what was found, what it is worth, and what to do "
            "about it. The queue also covers records outside the count above, "
            "such as a payout the bank posted twice."
        )
    else:
        st.success("Nothing is waiting for you on this report.")

    st.divider()
    st.subheader("What LedgerLoop actually does")
    st.caption(
        "Three systems describe the same money: your ledger, your bank, and your "
        "payment processor. LedgerLoop lines them up."
    )
    settled, stuck = journey(run)
    columns = st.columns(2)
    with columns[0]:
        st.markdown(_journey_row(settled, "When the evidence agrees"),
                    unsafe_allow_html=True)
    with columns[1]:
        st.markdown(_journey_row(stuck, "When it does not"), unsafe_allow_html=True)

    st.divider()
    st.subheader("How the matching works")
    st.caption("Each step only looks at what the step before it could not settle.")
    flow: list[str] = []
    for index, (number, name, why) in enumerate(_PLAIN_STEPS):
        if index:
            flow.append('<div class="ll-arrow">&#8594;</div>')
        css = "ll-rung ll-on" if index < len(_PLAIN_STEPS) - 1 else "ll-rung"
        flow.append(
            f'<div class="{css}">'
            f'<div class="ll-tier">Step {_esc(number)}</div>'
            f'<div class="ll-name">{_esc(name)}</div>'
            f'<div class="ll-purpose">{_esc(why)}</div>'
            "</div>"
        )
    st.markdown('<div class="ll-flow">' + "".join(flow) + "</div>", unsafe_allow_html=True)
    st.caption(
        "The rung-by-rung measurement, with what each step proposed and what it "
        "refused, is in **Accuracy & details**."
    )

    if view.unmatchable:
        st.info(
            f"{view.unmatchable:,} record(s), worth {view.unmatchable_impact}, "
            "cannot be resolved from the three source files at all -- the "
            "information simply is not in them. They are reported separately so "
            "that a real limit is never mistaken for a mistake."
        )


def _attention_card(item: AttentionItem) -> None:
    severity = item.severity.lower()
    css = f"ll-item ll-sev-{severity}" if severity in {"critical", "low"} else "ll-item"
    st.markdown(
        f'<div class="{css}">'
        f'<div class="ll-item-amt">{_esc(item.amount)}</div>'
        f'<div class="ll-item-sub">{_esc(item.subject)} &middot; '
        f"{_esc(item.severity)} priority</div>"
        f'<div class="ll-item-body"><strong>What we found.</strong> '
        f"{_esc(item.found)}</div>"
        f'<div class="ll-item-act"><strong>What to do.</strong> '
        f"{_esc(item.action)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Technical details for {item.subject}"):
        st.write(
            {
                "Exception class": item.technical_class,
                "Severity": item.severity,
                "Impact (paise)": item.amount_minor,
                "Evidence items": item.evidence_count,
                "Agent may resolve": item.agent_may_resolve,
                "Exception id": item.exception_id,
            }
        )
        if not item.agent_may_resolve:
            st.warning("Proposal only. This one needs a person.")


#: How many queue items are drawn as cards before falling back to the table.
#: The queue is sorted by money, so the cap keeps the most expensive items in
#: the readable format and never hides one silently -- the count is printed.
_CARD_LIMIT = 40


def _attention_body(
    items: list[AttentionItem],
    *,
    key_prefix: str,
    empty: str,
    table: Callable[[], None] | None = None,
) -> None:
    """The queue, drawn once for both a stored report and an upload.

    ``key_prefix`` keeps the two filter widgets apart -- Streamlit derives a
    widget id from its label and options, and two identically-labelled
    selectboxes on one run collide.
    """
    if not items:
        st.success(empty)
        return

    total = sum(item.amount_minor for item in items)
    st.subheader(f"{len(items):,} item(s) need your attention")
    st.markdown(
        '<p class="ll-lede">LedgerLoop did not find enough evidence to safely '
        "settle these, or it found something that looks wrong. "
        "<strong>Nothing was guessed.</strong> Each item says what was found and "
        "what to do about it, largest amount first.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ll-big-grid">'
        + _big_card(
            _WARN,
            "Items to review",
            f"{len(items):,}",
            "each with a reason and an action",
            "warn",
        )
        + _big_card(
            _RUPEE,
            "Money involved",
            format_minor(total),
            "across every item in the queue",
            "brand",
            money=True,
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    severities = sorted({item.severity for item in items})
    options = ["All", *severities]
    chosen = _picked(
        st.selectbox("Show", options, key=f"{key_prefix}_attn_sev"), options
    )
    shown = [item for item in items if chosen == "All" or item.severity == chosen]
    if len(shown) != len(items):
        st.caption(
            f"Showing {len(shown)} of {len(items)}. Filtering changes this list "
            "only -- the counts on Overview are read from the run itself."
        )

    for item in shown[:_CARD_LIMIT]:
        _attention_card(item)
    if len(shown) > _CARD_LIMIT:
        st.caption(
            f"Showing the {_CARD_LIMIT} largest of {len(shown)}. The rest are in "
            "the full queue below."
        )
    if table is not None:
        with st.expander("The whole queue as a table (technical)"):
            table()


def screen_attention(run: StoredRun) -> None:
    """The review queue for a bundled report."""
    _attention_body(
        attention_items(run),
        key_prefix="report",
        empty="Nothing needs your attention in this run.",
        table=lambda: screen_exceptions(run),
    )


def _transactions_body(
    rows: list[dict[str, object]],
    filtered: Callable[[str | None], list[dict[str, object]]],
    *,
    key_prefix: str,
) -> None:
    """The decision table, drawn once for both a stored report and an upload."""
    if not rows:
        st.info("No decisions were recorded.")
        return

    st.subheader("Transactions")
    st.markdown(
        '<p class="ll-lede">Every payment LedgerLoop reached a decision on, and '
        "how it reached it.</p>",
        unsafe_allow_html=True,
    )
    statuses = sorted({str(row["Status"]) for row in rows})
    options = ["All", *statuses]
    columns = st.columns([1, 2])
    chosen = _picked(
        columns[0].selectbox("Status", options, key=f"{key_prefix}_tx_status"), options
    )
    query = columns[1].text_input(
        "Search",
        placeholder="Payment or bank reference, e.g. PAY-00041",
        key=f"{key_prefix}_tx_search",
    )

    shown = filtered(None if chosen == "All" else chosen)
    shown = transaction_search(shown, query or "")
    st.caption(
        f"Showing {len(shown):,} of {len(rows):,}. There is no amount column "
        "because no amount is recorded per match -- the money totals are on "
        "Overview, and every amount under review is in Needs review."
    )
    st.dataframe(
        [
            {
                key: row[key]
                for key in (
                    "Status",
                    "Payment",
                    "Bank transaction",
                    "How it was matched",
                    "Confidence",
                )
            }
            for row in shown
        ],
        width="stretch",
        hide_index=True,
    )


def screen_transactions(run: StoredRun) -> None:
    """Every decision a bundled report committed."""
    _transactions_body(
        transaction_rows(run),
        lambda status: transaction_rows(run, status=status),
        key_prefix="report",
    )


def _why_body(
    decisions: Sequence[MatchDecision],
    exceptions: Sequence[ReconException],
    keys: Sequence[str],
    *,
    key_prefix: str,
) -> None:
    """One record explained, drawn once for both a report and an upload."""
    if not keys:
        st.info("No decisions or exceptions were recorded.")
        return

    def naming(key: str) -> list[MatchDecision]:
        return [
            decision
            for decision in decisions
            if key in (decision.source_ref.key, decision.target_ref.key)
        ]

    st.subheader("Why it matched")
    st.markdown(
        '<p class="ll-lede">Pick any record and see exactly why LedgerLoop '
        "reached the conclusion it did, or why it refused to.</p>",
        unsafe_allow_html=True,
    )
    # Matched records first. The tab is called "Why it matched", and opening a
    # reviewer on a refusal buries the strongest thing the screen has to show.
    # Every refusal is still one selection away and is explained just as fully.
    matched_first = [
        key
        for key in keys
        if any(
            decision.outcome is DecisionOutcome.AUTO_MATCHED
            for decision in naming(key)
        )
    ]
    seen = set(matched_first)
    ordered = matched_first + [key for key in keys if key not in seen]
    chosen = _picked(
        st.selectbox("Record", ordered, key=f"{key_prefix}_why_record"), ordered
    )
    story = match_story_from(naming(chosen), exceptions, chosen)

    if story.matched:
        st.success(f"{_TICK} {story.headline}")
    else:
        st.warning(f"{_WARN} {story.headline}")

    if story.partner:
        st.markdown(f"**{chosen}** to **{story.partner}**")
    if story.stage:
        st.markdown(f"**How it was matched.** {story.stage}")

    st.markdown("**Why:**")
    css = "ll-reasons" if story.matched else "ll-reasons ll-open"
    st.markdown(
        f'<ul class="{css}">'
        + "".join(f"<li>{_esc(reason)}</li>" for reason in story.reasons)
        + "</ul>",
        unsafe_allow_html=True,
    )
    if story.confidence:
        st.markdown(f"**Confidence.** {story.confidence}")
    if story.caveat:
        st.info(story.caveat)

    with st.expander("Technical details"):
        st.write(dict(story.technical))
        st.caption(
            "Read from the append-only log this run wrote. Nothing here is "
            "recomputed. The full event-by-event replay is in **Technical "
            "report**."
        )


def screen_why(run: StoredRun) -> None:
    """One record from a bundled report, explained without jargon."""
    _why_body(
        run.decisions, run.exceptions, record_keys(run), key_prefix="report"
    )


def _assistant_panel(run: StoredRun) -> None:
    """What the AI assistant did, and what was taken off it.

    Shown even when it did nothing, because "nothing" has three different
    meanings here and a reader cannot tell them apart otherwise: no model was
    configured, a model was configured and the provider refused every call, or a
    model answered and every answer was thrown out by the gates. The third is
    the system working; the first two are not failures either.
    """
    activity = assistant_activity(run)
    st.subheader("What the AI assistant did")

    if not activity.available:
        st.info(
            "**No AI model was used for this report.** Every figure above came "
            "from the deterministic rules, and re-running it produces the same "
            "numbers with no network and no key."
        )
        return

    if not activity.used:
        st.warning(
            "**A model was configured, but no call completed.** The provider "
            "declined every attempt -- a rate limit or an outage -- and the "
            "ladder fell back to the deterministic result, which is exactly "
            "what it exists to do. Nothing above depends on the model."
        )
        return

    st.markdown(
        '<div class="ll-big-grid">'
        + _big_card(
            "",
            "Calls made",
            f"{activity.calls:,}",
            f"{activity.tokens:,} tokens, about "
            f"{_RUPEE}{activity.cost_inr:.2f} at paid-API rates",
            "brand",
        )
        + _big_card(
            _TICK,
            "Suggestions accepted",
            f"{activity.accepted:,}",
            "survived every check and became candidates",
            "good",
        )
        + _big_card(
            _SHIELD,
            "Suggestions refused",
            f"{activity.refused:,}",
            "thrown out before they could count",
            "good" if activity.refused else "muted",
        )
        + _big_card(
            "",
            "Explanations reworded",
            f"{activity.prose_rewritten:,}",
            "wording only -- never the class or the amount",
            "muted",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if activity.refused:
        st.success(
            f"**{activity.refused} suggestion(s) were refused, and that is the "
            "point.** "
            + (
                f"{activity.refused_ungrounded} cited a record that was not in "
                "the evidence pack it was given. "
                if activity.refused_ungrounded
                else ""
            )
            + (
                f"{activity.demoted_unverified} did not add up when the money "
                "was re-derived from your files, and were handed to a person "
                "rather than dropped. "
                if activity.demoted_unverified
                else ""
            )
            + "The model proposes; deterministic code decides."
        )
    st.caption(
        "The assistant may read a narration the rules could not parse, suggest a "
        "link for what is left over, and reword an explanation. It may **never** "
        "decide a match, do arithmetic, set a confidence or classify an "
        "exception -- and `verify_arithmetic` has no parameter for where a "
        "proposal came from."
    )


def screen_report(run: StoredRun) -> None:
    """Everything a judge or an engineer needs, out of the normal user's way."""
    st.subheader("Accuracy and details")
    st.markdown(
        '<p class="ll-lede">The measurement, in the vocabulary '
        "<code>EVALUATION.md</code> uses. Every figure is read from this run's "
        "own record.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("What the technical terms mean", expanded=True):
        st.dataframe(
            [
                {
                    "Term": entry.term,
                    "In plain words": entry.plain,
                    "This run": entry.value,
                }
                for entry in glossary(headline(run))
            ],
            width="stretch",
            hide_index=True,
        )

    # The same three destinations the Overview leads with, counted in the
    # *decision* unit rather than the record unit. Both are true and they are
    # different numbers, so they must not share a screen: on the Overview they
    # would read as a contradiction rather than as two views of one run.
    st.subheader("Where everything went, by decision")
    st.caption(
        "Committed links, queue items and missing links. These count decisions "
        "and exceptions, not records, so they do not match the Overview's "
        "figures -- which count records a perfect system could have resolved."
    )
    st.markdown(
        '<div class="ll-split">'
        + "".join(_destination(bucket) for bucket in buckets(run))
        + "</div>",
        unsafe_allow_html=True,
    )
    view = snapshot(run)
    if view.referred:
        st.caption(
            f"{view.referred:,} of these were found and deliberately **not** "
            "committed. LedgerLoop hands a case to a person rather than choose "
            "between two possibilities."
        )

    st.divider()
    _assistant_panel(run)
    st.divider()
    screen_overview(run)
    st.divider()
    screen_pipeline(run)
    st.divider()
    screen_evaluation(run)
    st.divider()
    screen_evidence(run)


# --------------------------------------------------------------------------
# The same five screens, for files the reader brought themselves
#
# Not a second dashboard. Each of these calls the shaping functions the
# report-backed screens call -- `attention_items_from`, `transaction_rows_from`,
# `match_story_from`, `tier_stages_from` -- and differs only where an upload
# genuinely cannot answer the question a report can.
#
# There is exactly one such place, and it is the accuracy figures. Precision,
# recall and the match rate are scored against a hand-checked answer key. An
# upload has none, so those are absent and *said to be absent*, rather than
# rendered as a zero that would read as "nothing went wrong".
# --------------------------------------------------------------------------
def screen_home_upload(result: ReconcileResult) -> None:
    """Overview, for uploaded files."""
    view = upload_snapshot(result)
    st.markdown(
        '<p class="ll-lede">This is what LedgerLoop did with the records you '
        "uploaded. It matched what it could prove and left the rest for a "
        "person.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ll-big-grid">'
        + _big_card(
            "",
            "Transactions read",
            f"{view.records:,}",
            "records across the files you supplied",
            "brand",
        )
        + _big_card(
            _TICK,
            "Confidently matched",
            f"{view.matched:,}",
            "payments linked to a bank credit",
            "good",
        )
        + _big_card(
            _WARN,
            "Need your attention",
            f"{view.queue:,}",
            "LedgerLoop would not guess at these",
            "warn",
        )
        + _big_card(
            _RUPEE,
            "Value matched",
            view.matched_value,
            "on the links it committed",
            "brand",
            money=True,
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "**How many of those matches were correct is not shown, because it "
        "cannot be known from your files alone.** Precision and recall are "
        "measured against a hand-checked answer key -- somebody's finished "
        "reconciliation of the same period -- and an export does not come with "
        "one. Every figure above is a statement about what LedgerLoop *did*."
    )

    if result.llm_used:
        st.success(
            f"The AI assistant was called {result.llm_calls} time(s). Everything "
            "it suggested was re-checked against your files before being "
            "accepted."
        )
    else:
        st.caption("Deterministic: no AI model was called for this result.")

    st.divider()
    st.subheader("What was in your files")
    st.markdown(
        '<div class="ll-split">'
        + _destination(
            Bucket(
                "open",
                "Payments",
                view.payments,
                "from your processor report",
                "muted",
            )
        )
        + _destination(
            Bucket(
                "open",
                "Payouts",
                view.payouts,
                "batches the processor reported",
                "muted",
            )
        )
        + _destination(
            Bucket(
                "open",
                "Bank rows",
                view.bank_rows,
                "credits and debits your bank posted",
                "muted",
            )
        )
        + _destination(
            Bucket(
                "open",
                "Orders",
                view.orders,
                "from your ledger, if you supplied one",
                "muted",
            )
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if view.quarantined:
        st.warning(
            f"{view.quarantined:,} row(s) could not be read and were set aside "
            "rather than guessed at. They are listed under **Accuracy & "
            "details**."
        )

    if view.queue:
        st.divider()
        st.info(
            f"**Next:** open **Needs review** -- {view.queue:,} item(s) worth "
            f"{view.queue_value}, largest first. Each says what was found and "
            "what to do about it."
        )


def screen_attention_upload(result: ReconcileResult) -> None:
    """Needs review, for uploaded files."""
    _attention_body(
        attention_items_from(result.exceptions),
        key_prefix="up",
        empty="Nothing in your files needs attention.",
    )


def screen_transactions_upload(result: ReconcileResult) -> None:
    """Transactions, for uploaded files."""
    _transactions_body(
        transaction_rows_from(result.matched.decisions),
        lambda status: transaction_rows_from(result.matched.decisions, status=status),
        key_prefix="up",
    )


def screen_why_upload(result: ReconcileResult) -> None:
    """Why it matched, for uploaded files."""
    decisions = list(result.matched.decisions)
    keys = sorted(
        {ref.key for d in decisions for ref in (d.source_ref, d.target_ref)}
        | {ref.key for item in result.exceptions for ref in item.involved_refs}
    )
    _why_body(decisions, list(result.exceptions), keys, key_prefix="up")


def screen_report_upload(result: ReconcileResult) -> None:
    """Accuracy & details, for uploaded files.

    The accuracy half is genuinely unavailable and says so. The details half --
    which rung matched what, what the ingest layer had to set aside, what the
    assistant did -- needs no answer key at all, and withholding it would have
    been laziness dressed up as rigour.
    """
    view = upload_snapshot(result)
    st.subheader("Accuracy and details")
    st.warning(
        "**No accuracy figures for your own files.** Precision, recall and the "
        "match rate each compare what LedgerLoop found against what is *really* "
        "there, and knowing what is really there means somebody reconciling the "
        "period by hand first. Your files do not carry that, so those numbers "
        "are absent rather than estimated. The bundled sample reports do carry "
        "it, and this screen reports all of it for them."
    )

    st.divider()
    st.subheader("How it got there")
    st.caption(
        "Six rungs, cheapest and most certain first. Each only sees what the "
        "rungs above it left behind."
    )
    stages = upload_tier_stages(result)
    flow: list[str] = []
    for index, stage in enumerate(stages):
        if index:
            flow.append('<div class="ll-arrow">&#8594;</div>')
        flow.append(_rung_card(stage))
    st.markdown('<div class="ll-flow">' + "".join(flow) + "</div>", unsafe_allow_html=True)
    st.caption(
        "**Proposed** is yield and **matched** is conviction. The gap is the "
        "review queue a finance team has to staff -- and it needs no answer key "
        "to measure, because it counts what the system did rather than whether "
        "it was right."
    )

    st.divider()
    _assistant_panel_upload(result)

    st.divider()
    st.subheader("What was read")
    st.dataframe(
        [
            {"Source": "Orders (your ledger)", "Records": view.orders},
            {"Source": "Payments (processor report)", "Records": view.payments},
            {"Source": "Payouts (processor report)", "Records": view.payouts},
            {"Source": "Bank transactions", "Records": view.bank_rows},
            {"Source": "Rows set aside as unreadable", "Records": view.quarantined},
        ],
        width="stretch",
        hide_index=True,
    )
    if result.ingest.problems:
        with st.expander(f"The {view.quarantined} row(s) that could not be read"):
            st.dataframe(
                [
                    {
                        "Source": problem.source.value,
                        "Record": problem.record_id or "",
                        "Why": problem.detail,
                    }
                    for problem in result.ingest.problems[:200]
                ],
                width="stretch",
                hide_index=True,
            )


def _assistant_panel_upload(result: ReconcileResult) -> None:
    """What the model did on an upload, read from the run's own cost ledger."""
    st.subheader("What the AI assistant did")
    if not result.llm_used:
        st.info(
            "**No AI model was called for this result.** Everything above came "
            "from the deterministic rules. Tick the box on **Your files** before "
            "running to involve one."
        )
        return
    st.markdown(
        '<div class="ll-big-grid">'
        + _big_card(
            "",
            "Calls made",
            f"{result.llm_calls:,}",
            "each one re-checked before it counted",
            "brand",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The assistant may read a narration the rules could not parse and "
        "suggest a link for what is left over. It may **never** decide a match, "
        "do arithmetic or set a confidence."
    )


def _model_available() -> bool:
    """Whether a model could be reached, without reaching for one.

    `.env` is loaded here as well as in the CLI, because ``streamlit run``
    starts the script directly and never passes through ``ledgerloop.cli``. An
    already-exported variable still wins; see :mod:`ledgerloop.envfile`.
    """
    load_env_file()
    return build_ladder(LLMConfig(enabled=True)) is not None


def _ui_client(*, wanted: bool) -> LLMClient:
    """The model client this interface should use, if any.

    ``wanted`` is the person's explicit choice, and it defaults to **off**.
    Three reasons, in order:

    * the deterministic path is the one every published number came from, and it
      should stay what happens when nobody asked for anything else;
    * a live run took roughly two and a half minutes of provider time in Phase
      2.8, against under a second deterministically, and a reviewer clicking a
      button deserves to know which they are getting;
    * a key in `.env` is a *capability*, not an instruction. Spending someone's
      quota because they once saved a credential is not a decision this
      interface gets to make for them.

    Returns a **disabled** client when nothing is reachable or nothing was
    asked for. Nothing here reports that a model ran: that is read from the cost
    ledger afterwards, because a key existing is not a call being made.
    """
    if not wanted or not _model_available():
        return LLMClient(config=LLMConfig(enabled=False), provider=None)
    # Its own cache directory, not the default.
    #
    # `LLMConfig.cache_dir` defaults to `tests/fixtures/llm_cache`, which is
    # committed and is supposed to stay empty until a deliberate live run fills
    # it -- Step 10 leaked five stand-in responses in there once and they had to
    # be removed. A dashboard writing real model answers into the test fixtures
    # would do it again, quietly, every time somebody ticked the box. B2 solved
    # this the same way with `reports/llm_cache_b2`.
    config = LLMConfig(enabled=True, cache_dir=Path("reports/llm_cache_ui"))
    ladder = build_ladder(config)
    if ladder is None:  # pragma: no cover - guarded by _model_available
        return LLMClient(config=LLMConfig(enabled=False), provider=None)
    return LLMClient(config=config, provider=ladder)


class _Held(TypedDict):
    """One accepted upload, as the screen needs to describe it."""

    filename: str
    rows: int
    bytes: int


def _session_upload_dir() -> Path:
    """A private scratch directory for this browser session.

    Created under the system temp root, never inside the repository: an upload
    that landed in ``data/`` could overwrite a committed fixture, and a demo
    would then display someone's real statement. The path is kept in session
    state so a rerun reuses it rather than scattering directories.
    """
    key = "upload_dir"
    existing = st.session_state.get(key)
    if existing and Path(existing).is_dir():
        return Path(existing)
    created = Path(tempfile.mkdtemp(prefix="ledgerloop-upload-"))
    st.session_state[key] = str(created)
    return created


def _uploaded() -> dict[SourceKind, _Held]:
    """What has been accepted so far, by source."""
    store = st.session_state.setdefault("uploads", {})
    return cast("dict[SourceKind, _Held]", store)


def _accept(kind: SourceKind, name: str, payload: bytes) -> UploadProblem | None:
    """Validate and store one file. Returns the problem, or ``None`` on success.

    The bytes are written to the session directory under the name the ingester
    expects, so nothing downstream needs to know an upload happened.
    """
    problem = validate(kind, name, payload)
    if problem is not None:
        return problem
    target = _session_upload_dir() / kind.value
    target.write_bytes(payload)
    _uploaded()[kind] = {
        "filename": name,
        "rows": row_count(kind, payload),
        "bytes": len(payload),
    }
    return None


def _upload_card(kind: SourceKind) -> None:
    """One optional source: drop a file, or do not."""
    held = _uploaded().get(kind)
    st.markdown(
        f'<div class="ll-drop{" ll-have" if held else ""}">'
        f'<div class="ll-drop-title">{_esc(kind.label)}</div>'
        f'<div class="ll-drop-note">{_esc(kind.blurb)}</div>'
        f'<div class="ll-drop-type">{_esc(kind.file_type)} &middot; optional</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    chosen = st.file_uploader(
        kind.label,
        type=["json"] if kind is SourceKind.PROCESSOR else ["csv"],
        key=f"upload_{kind.name}",
        label_visibility="collapsed",
    )
    if chosen is not None:
        payload = chosen.getvalue()
        detected = detect(chosen.name, payload)
        if detected is not None and detected is not kind:
            st.warning(
                f"This looks like a **{detected.label.lower()}**, not a "
                f"{kind.label.lower()}. Upload it in the "
                f"**{detected.label}** box instead."
            )
        else:
            problem = _accept(kind, chosen.name, payload)
            if problem is not None:
                st.error(f"**{problem.reason}.** {problem.detail}")

    held = _uploaded().get(kind)
    if held:
        unit = "payout(s)" if kind is SourceKind.PROCESSOR else "row(s)"
        st.success(f"{held['filename']} - {held['rows']:,} {unit}")
        if st.button("Remove", key=f"remove_{kind.name}"):
            _uploaded().pop(kind, None)
            path = _session_upload_dir() / kind.value
            if path.is_file():
                path.unlink()
            st.rerun()


def screen_upload() -> None:
    """Bring your own files. The first thing a new visitor sees."""
    st.subheader("Upload your files")
    st.markdown(
        '<p class="ll-lede">Add whichever records you have. '
        "<strong>None of them is required</strong> on its own, and LedgerLoop "
        "will tell you what it can do with the combination you give it.</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(3)
    for column, kind in zip(columns, SourceKind, strict=True):
        with column:
            _upload_card(kind)

    supplied = set(_uploaded())
    verdict = assess(supplied)
    tone = "good" if verdict.can_reconcile else "warn"
    st.markdown(
        f'<div class="ll-status {_TONE_SUFFIX[tone]}" style="margin-top:1rem">'
        f'<div class="ll-status-title">{_esc(verdict.headline)}</div>'
        f'<div class="ll-status-body">{_esc(verdict.detail)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if verdict.missing_hint:
        st.caption(verdict.missing_hint)

    if not supplied:
        st.caption(
            "No files yet. The reports already on this machine are in the "
            "sidebar, if you would rather look at one of those."
        )
        return

    if _model_available():
        st.checkbox(
            "Also ask the AI assistant about anything the rules could not settle",
            key="use_llm",
            help=(
                "Off by default. The rules alone are deterministic and take under "
                "a second; a model adds a network round trip and can take minutes. "
                "Whatever it suggests is still re-checked against your files "
                "before anything is accepted."
            ),
        )
    else:
        st.caption(
            "No AI model is configured on this machine, so this runs on the "
            "deterministic rules alone. That is the normal case and needs no "
            "setup."
        )

    disabled = not verdict.can_reconcile
    if st.button(
        "Start reconciliation",
        type="primary",
        disabled=disabled,
        key="start_upload_run",
    ):
        _reconcile_uploads()

    result = st.session_state.get("upload_result")
    if result is not None:
        _upload_results(cast("ReconcileResult", result))


def _reconcile_uploads() -> None:
    """Run the ladder over the uploaded files and keep the result."""
    directory = _session_upload_dir()
    client = _ui_client(wanted=bool(st.session_state.get("use_llm", False)))
    with st.spinner("Reading your files and matching what can be proved..."):
        try:
            st.session_state["upload_result"] = reconcile_only(
                directory, client=client
            )
        except IngestError as error:
            st.session_state["upload_result"] = None
            st.error(f"**Your files could not be read.** {error}")


def _upload_results(result: ReconcileResult) -> None:
    """The receipt for a run, and a pointer to the screens that read it.

    Deliberately short. Everything a reader wants to know about their files is
    answered by the five tabs beside this one, on the same screens the bundled
    reports use -- so this says the run finished and what it found, and sends
    them there rather than growing a second dashboard inside one tab.
    """
    view = upload_snapshot(result)
    st.divider()
    st.subheader("Done")
    st.markdown(
        '<div class="ll-big-grid">'
        + _big_card(
            "",
            "Transactions read",
            f"{view.records:,}",
            "records across the files you supplied",
            "brand",
        )
        + _big_card(
            _TICK,
            "Confidently matched",
            f"{view.matched:,}",
            "payments linked to a bank credit",
            "good",
        )
        + _big_card(
            _WARN,
            "Need your attention",
            f"{view.queue:,}",
            "LedgerLoop would not guess at these",
            "warn",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if view.quarantined:
        st.warning(
            f"{view.quarantined:,} row(s) could not be read and were set aside "
            "rather than guessed at. They are listed under "
            "**Accuracy & details**."
        )
    st.success(
        "**Every tab now describes your files.** Overview for the money, "
        "Needs review for the queue, Transactions for every decision, Why it "
        "matched to check one of them, Accuracy & details for the rungs and "
        "what was read. The sidebar switches back to a sample report."
    )

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
    labels = report_labels(runs)
    with st.sidebar:
        st.markdown("### LedgerLoop")
        st.caption("Automatic payment reconciliation")
        st.divider()
        st.markdown("**Reports**")
        # An upload sits in the same list as the bundled samples, because to a
        # reader it *is* another report -- the one they brought. Putting it in a
        # separate control would have kept the split that made the tabs confusing
        # in the first place.
        upload = st.session_state.get("upload_result")
        if upload is not None:
            picked_scope = st.radio(
                "Showing",
                ("Your files", "A sample report"),
                key="scope",
                label_visibility="collapsed",
            )
            st.session_state["show_upload"] = picked_scope == "Your files"
            st.caption(
                f"{cast('ReconcileResult', upload).ingest.record_count:,} records "
                "you uploaded"
                if st.session_state["show_upload"]
                else "the bundled corpus below"
            )
        else:
            st.session_state["show_upload"] = False
        if runs:
            # Labelled by what the report *is*, not by the run id. `t0t4-test-42`
            # names a ladder, a split and a seed: exactly what someone
            # reproducing a figure needs, and exactly what someone reading one
            # does not. The full id is in the details expander below.
            selected: Any = st.radio(
                "Reports",
                runs,
                format_func=lambda run: labels.get(run.run_id, run.run_id),
                label_visibility="collapsed",
            )
        else:
            selected = None
            st.info("No reports yet. Create one on the **New report** tab.")
        if selected is not None:
            dataset = selected.summary.get("dataset", {})
            config = selected.summary.get("config", {})
            st.divider()
            view = snapshot(selected)
            st.markdown("**Current report**")
            # The same figure the Overview leads with, so the two never
            # disagree on screen. Falls back to the raw record count on an older
            # report that did not store the reconcilable total.
            checked = view.checked if view.checked is not None else view.records
            st.caption(f"{checked:,} transactions checked")
            if view.is_clean:
                st.success("Reconciliation complete")
            else:
                st.error(f"{view.incorrect} incorrect match(es)")
            # Identifiers belong to whoever is reproducing a figure, and to
            # nobody else. They stay one click away rather than greeting every
            # reader with a tuning hash.
            with st.expander("Report details"):
                st.caption(
                    f"Run id `{selected.run_id}`\n\n"
                    f"Split `{dataset.get('split', '?')}` · "
                    f"difficulty `{dataset.get('difficulty', '?')}` · "
                    f"seed `{dataset.get('seed', '?')}`\n\n"
                    f"{int(dataset.get('evaluation_links', 0)):,} evaluation links · "
                    f"{view.matched:,} committed\n\n"
                    f"Tuning hash `{config.get('tuning_hash', '?')}`\n\n"
                    f"Read from `{_runs_root()}` -- `run.json`, `audit.jsonl`, "
                    "`exceptions.json` and `decisions.json`, the same files the CLI "
                    "writes. This interface renders them and computes nothing."
                )

    upload = st.session_state.get("upload_result")
    show_upload = bool(st.session_state.get("show_upload")) and upload is not None
    if show_upload:
        held = cast("ReconcileResult", upload)
        _hero(
            None,
            chip=f"AI assistant used on {held.llm_calls} item(s)"
            if held.llm_used
            else "No AI model used — every figure is repeatable",
        )
    else:
        _hero(selected)

    # Five reader-facing screens, then the launcher. The order is the order a
    # person asks the questions in: what happened, what do I do, show me
    # everything, prove one of them, and only then the measurement.
    tabs = st.tabs(
        [
            "Your files",
            "Overview",
            "Needs review",
            "Transactions",
            "Why it matched",
            "Accuracy & details",
            "Sample data",
        ]
    )
    # Upload first. A visitor arrives with their own records, not with a
    # curiosity about a bundled corpus, so the thing they came to do is the
    # first thing on the page.
    with tabs[0]:
        screen_upload()
    # The same five questions, answered about whichever subject the sidebar is
    # pointing at. Piling an upload's whole result onto the first tab was the
    # earlier design and it was wrong twice over: the screens built to answer
    # these questions sat unused, and a reader had to learn that "your files"
    # meant something different from every other tab.
    report_screens = (
        screen_home,
        screen_attention,
        screen_transactions,
        screen_why,
        screen_report,
    )
    upload_screens = (
        screen_home_upload,
        screen_attention_upload,
        screen_transactions_upload,
        screen_why_upload,
        screen_report_upload,
    )
    for tab, report_screen, upload_screen in zip(
        tabs[1:6], report_screens, upload_screens, strict=True
    ):
        with tab:
            if show_upload:
                st.caption(
                    f"{_FOLDER} Your files — "
                    f"{cast('ReconcileResult', upload).ingest.record_count:,} "
                    "records you uploaded. Switch subject in the sidebar."
                )
                upload_screen(cast("ReconcileResult", upload))
                continue
            if selected is None:
                st.info(
                    "These screens describe a finished report. Upload your files "
                    "on **Your files**, or create a sample one on **Sample data**."
                )
                continue
            if upload is not None:
                # Only reachable when the reader has deliberately switched to a
                # sample while holding a result of their own, so this says which
                # subject they are looking at rather than warning them off.
                st.caption(
                    f"{_FOLDER} Sample report — "
                    f"'{labels.get(selected.run_id, selected.run_id)}', not your "
                    "uploaded files."
                )
            report_screen(selected)
    with tabs[6]:
        screen_run()


# Streamlit executes this module top to bottom, so the entry point is a plain
# call rather than a guard: there is no `__main__` when `streamlit run` imports
# it. The guard is kept for `python -m` invocations, which print a hint instead
# of rendering nothing.
if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
