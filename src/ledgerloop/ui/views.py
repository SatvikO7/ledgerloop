"""What the four screens display, as pure functions over a stored run.

**No Streamlit import in this module, and no reconciliation logic either.**
Every function here takes a :class:`~ledgerloop.agent.store.StoredRun` -- four
JSON/JSONL files a completed run wrote -- and returns rows of plain data. That
split is the whole design of the UI:

* :mod:`ledgerloop.ui.app` is widget glue that renders these rows. It has no
  branches worth testing.
* This module has every branch and no widgets, so it is tested like any other
  module in the project.
* Neither can recompute a metric, because neither holds the objects that could.
  The dashboard and ``EVALUATION.md`` cannot disagree; they read the same
  numbers from the same run.

THE FOUR SCREENS (PLAN.md §14.1)
--------------------------------
1. **Run** -- generate or pick a dataset, start a reconciliation.
2. **Results** -- the money view: match rate, precision, recall, ₹ reconciled
   against ₹ outstanding, and the tier waterfall.
3. **Exceptions** -- the controller's workday, sorted by rupee impact
   descending, each row carrying a class, a severity, a price, an evidence
   chain and an action.
4. **Audit replay** -- step a record through the tiers and see why the final
   decision happened.

PLAN.md's screen 3 is a Cytoscape lineage graph. It is not built: a graph
rendering library is a dependency the demo does not need, and the lineage a
reviewer actually asks about -- *which tier decided this record, on what
evidence* -- is what the Audit Replay screen answers directly. Stated here
rather than left as a silent omission.

WHAT THIS MODULE WILL NOT DO
----------------------------
It never invents a number and never hides one. An unresolved exception is
displayed as unresolved; an ``UNMATCHABLE`` record is displayed as the honest
floor rather than folded into a failure; and the three decision outcomes are
three columns, never one "matched" figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ledgerloop.agent.store import StoredRun
from ledgerloop.models.audit import AuditEvent
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.money import format_minor

__all__ = [
    "OUTCOME_HELP",
    "DecisionTrace",
    "Headline",
    "exception_rows",
    "headline",
    "money_rows",
    "recall_rows",
    "record_keys",
    "tier_rows",
    "trace_record",
]

#: What each decision outcome means, in a controller's words.
#:
#: The three are displayed separately everywhere in the UI. Collapsing them into
#: one "matched" number is the single most misleading thing a reconciliation
#: dashboard can do: a referral is not a match, and an exception is not a
#: failure to decide -- it is a decision to escalate.
OUTCOME_HELP: dict[str, str] = {
    DecisionOutcome.AUTO_MATCHED.value: (
        "Committed without a human. The probability cleared the fitted threshold "
        "and the arithmetic closed against the source documents."
    ),
    DecisionOutcome.NEEDS_REVIEW.value: (
        "Referred to a person. The system found something and declined to commit "
        "it -- this is the precision-first design working, not a miss."
    ),
    DecisionOutcome.EXCEPTION.value: (
        "Escalated with a typed reason. Something is wrong with the record and "
        "the queue says what, what it costs, and what to do next."
    ),
    DecisionOutcome.REJECTED.value: (
        "Considered and refused outright. The policy's fourth outcome, shown "
        "even at zero because the policy did run and did not use it -- unlike "
        "a component that was never built, this zero is a measurement."
    ),
}


@dataclass(frozen=True)
class Headline:
    """The numbers screen 2 leads with, each already decided by the run."""

    precision: float
    precision_ci: tuple[float, float]
    recall: float
    match_rate: float
    exception_recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    false_positive_cost: str
    reconciled: str
    outstanding: str
    unmatchable_count: int
    unmatchable_impact: str
    auto_matched: int
    needs_review: int
    exceptions: int
    """Evaluation-unit decisions routed to an exception.

    **Not** the size of the exception queue, and the UI labels both. A
    settlement nothing could credit produces a queue entry without producing a
    ``PAYMENT_CREDITED_AS`` decision at all, so the two numbers differ and
    showing one as the other would understate the work a controller faces.
    """

    rejected: int
    """The policy's fourth outcome. Displayed rather than folded into another.

    A dashboard showing three of four outcomes has chosen which failures to
    mention, and the count would have to go somewhere -- silently into
    "unmatched", where nobody could tell it apart from a record no tier reached.
    """

    queue_size: int
    """Exceptions in the queue. The controller's actual workday."""

    candidates_proposed: int
    residual_passes: int
    llm_available: bool
    llm_calls: int
    llm_tokens: int
    equivalent_paid_cost_inr: float
    records: int
    evaluation_links: int
    wall_clock_ms: int

    @property
    def precision_is_perfect(self) -> bool:
        """Whether the run made **no** wrong auto-match. The headline claim."""
        return self.false_positives == 0


def headline(run: StoredRun) -> Headline:
    """Read the headline off the stored run. Nothing here is computed."""
    metrics = run.metrics
    decisions = run.summary.get("decisions", {})
    llm = run.summary.get("llm", {})
    dataset = run.summary.get("dataset", {})
    return Headline(
        precision=float(metrics.get("auto_match_precision", 0.0)),
        precision_ci=(
            float(metrics.get("precision_ci_low", 0.0)),
            float(metrics.get("precision_ci_high", 1.0)),
        ),
        recall=float(metrics.get("recall", 0.0)),
        match_rate=float(metrics.get("match_rate", 0.0)),
        exception_recall=float(metrics.get("exception_recall", 0.0)),
        true_positives=int(metrics.get("true_positives", 0)),
        false_positives=int(metrics.get("false_positives", 0)),
        false_negatives=int(metrics.get("false_negatives", 0)),
        false_positive_cost=format_minor(int(metrics.get("false_positive_cost_minor", 0))),
        reconciled=format_minor(int(metrics.get("reconciled_minor", 0))),
        outstanding=format_minor(int(metrics.get("outstanding_minor", 0))),
        unmatchable_count=int(metrics.get("unmatchable_count", 0)),
        unmatchable_impact=format_minor(int(metrics.get("unmatchable_impact_minor", 0))),
        auto_matched=int(decisions.get(DecisionOutcome.AUTO_MATCHED.value, 0)),
        needs_review=int(decisions.get(DecisionOutcome.NEEDS_REVIEW.value, 0)),
        exceptions=int(decisions.get(DecisionOutcome.EXCEPTION.value, 0)),
        rejected=int(decisions.get(DecisionOutcome.REJECTED.value, 0)),
        queue_size=len(run.exceptions),
        candidates_proposed=int(decisions.get("candidates_proposed", 0)),
        residual_passes=int(run.summary.get("residual_passes", 0)),
        llm_available=bool(llm.get("available", False)),
        llm_calls=int(llm.get("calls", 0)),
        llm_tokens=int(llm.get("total_tokens", 0)),
        equivalent_paid_cost_inr=float(llm.get("equivalent_paid_cost_inr", 0.0)),
        records=int(dataset.get("records", 0)),
        evaluation_links=int(dataset.get("evaluation_links", 0)),
        wall_clock_ms=int(metrics.get("wall_clock_ms", 0)),
    )


def money_rows(run: StoredRun) -> list[dict[str, str]]:
    """The money view: what was reconciled, what is outstanding, what is a floor.

    ``Unmatchable`` is its own row rather than a share of the outstanding total,
    because it is a real ceiling and not a model failure -- no system can
    resolve those records without data outside the three sources.
    """
    metrics = run.metrics
    reconciled = int(metrics.get("reconciled_minor", 0))
    outstanding = int(metrics.get("outstanding_minor", 0))
    return [
        {
            "Measure": "Reconciled",
            "Amount": format_minor(reconciled),
            "Meaning": "money on links the system found and got right",
        },
        {
            "Measure": "Outstanding",
            "Amount": format_minor(outstanding),
            "Meaning": "money on links it did not assert",
        },
        {
            "Measure": "Total across evaluation links",
            "Amount": format_minor(reconciled + outstanding),
            "Meaning": "the money the evaluation unit covers",
        },
        {
            "Measure": "False-positive cost",
            "Amount": format_minor(int(metrics.get("false_positive_cost_minor", 0))),
            "Meaning": "money it declared reconciled that was not",
        },
        {
            "Measure": "Unmatchable impact (the honest floor)",
            "Amount": format_minor(int(metrics.get("unmatchable_impact_minor", 0))),
            "Meaning": (
                f"{int(metrics.get('unmatchable_count', 0))} records no system could "
                "resolve from these three sources"
            ),
        },
    ]


def tier_rows(run: StoredRun) -> list[dict[str, Any]]:
    """The tier waterfall: what each rung proposed and what it committed.

    Yield and conviction are separate columns. A tier that proposes a hundred
    and commits forty has not performed like one that proposes forty and commits
    forty -- the gap is the review queue a finance team has to staff.
    """
    rows = run.summary.get("tiers", [])
    return [
        {
            "Tier": row.get("tier", "?"),
            "Proposed": int(row.get("candidates_proposed", 0)),
            "Auto-matched": int(row.get("auto_matched", 0)),
            "Marginal": int(row.get("marginal_auto_matched", 0)),
            "Wall clock (ms)": int(row.get("wall_clock_ms", 0)),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def recall_rows(run: StoredRun) -> list[dict[str, Any]]:
    """Per-anomaly-class recall, **including the classes that score badly**.

    Publishing only the good rows is precisely what this project is trying not
    to do, so the table is whatever the run measured, in class order.
    """
    rows = run.summary.get("recall_by_anomaly_class", {})
    if not isinstance(rows, dict):
        return []
    return [
        {"Anomaly class": name, "Recall": float(value)}
        for name, value in sorted(rows.items())
    ]


def exception_rows(
    run: StoredRun,
    *,
    severity: str | None = None,
    exception_class: str | None = None,
) -> list[dict[str, Any]]:
    """The queue, **sorted by rupee impact descending** (PLAN.md §8.2.3).

    Never by count and never by class: one ₹4 lakh payout matters more than two
    hundred one-paise drifts, and any other sort order hides that.

    The filters narrow what is shown and the caller is expected to say so. They
    cannot make an exception disappear from the counts on screen 2, which are
    read from the run's own summary rather than from this list.
    """
    items = sorted(run.exceptions, key=lambda item: (-item.impact_minor, item.exception_id))
    return [
        {
            "Severity": item.severity.value,
            "Impact": format_minor(item.impact_minor),
            "impact_minor": item.impact_minor,
            "Class": item.exception_class.value,
            "Subject": item.involved_refs[0].record_id if item.involved_refs else "",
            "Root cause": item.root_cause,
            "Suggested action": item.suggested_action,
            "Agent may resolve": item.resolvable_by_agent,
            "Confidence": item.classification_confidence,
            "exception_id": item.exception_id,
        }
        for item in items
        if (severity is None or item.severity.value == severity)
        and (exception_class is None or item.exception_class.value == exception_class)
    ]


def evidence_rows(exception: ReconException) -> list[dict[str, str]]:
    """One exception's evidence chain, pointing back at source records."""
    return [
        {
            "Kind": item.kind.value,
            "Detail": item.detail,
            "Records": ", ".join(ref.record_id for ref in item.refs),
            "Amount": format_minor(item.amount_minor) if item.amount_minor else "",
        }
        for item in exception.evidence
    ]


__all__ += ["evidence_rows"]


@dataclass(frozen=True)
class DecisionTrace:
    """How one record moved through the ladder, and why it ended where it did.

    Screen 4. Assembled purely by *selecting* from the stored decisions and the
    stored audit log -- nothing is re-derived, which is what makes the replay a
    replay rather than a second run.
    """

    record_key: str
    decisions: tuple[MatchDecision, ...]
    events: tuple[AuditEvent, ...]
    exceptions: tuple[ReconException, ...]

    @property
    def final(self) -> MatchDecision | None:
        """The decision that stands. The log is append-only, so it is the last."""
        return self.decisions[-1] if self.decisions else None

    @property
    def outcome(self) -> str:
        decision = self.final
        if decision is not None:
            return decision.outcome.value
        if self.exceptions:
            return DecisionOutcome.EXCEPTION.value
        return "NO DECISION"

    @property
    def explanation(self) -> str:
        """Why it ended there, in a sentence, from what the run recorded."""
        decision = self.final
        if decision is None:
            if self.exceptions:
                return self.exceptions[0].root_cause
            return (
                "No tier proposed a link for this record and no exception names "
                "it. It is outside the evaluation unit, or it is one of the "
                "unmatchable records the report counts as the honest floor."
            )
        return (
            f"{decision.tier.name} proposed it, the blender scored it at "
            f"p = {decision.calibrated_p:.4f}, and the policy returned "
            f"{decision.outcome.value}. {decision.reason}"
        ).strip()

    def timeline(self) -> list[dict[str, Any]]:
        """The audit events naming this record, in sequence order."""
        return [
            {
                "#": event.sequence,
                "Node": event.node,
                "Event": event.event_type.value,
                "Message": event.message,
                "Tier": str(event.payload.get("tier", "")),
                "Outcome": str(event.payload.get("outcome", "")),
            }
            for event in self.events
        ]


def trace_record(run: StoredRun, record_key: str) -> DecisionTrace:
    """Everything the run recorded about one record. Screen 4's whole input."""
    return DecisionTrace(
        record_key=record_key,
        decisions=run.decisions_for(record_key),
        events=run.audit_for(record_key),
        exceptions=tuple(
            exception
            for exception in run.exceptions
            if any(ref.key == record_key for ref in exception.involved_refs)
        ),
    )


def record_keys(run: StoredRun) -> list[str]:
    """Every record the run decided or raised an exception about, sorted.

    The Audit Replay screen's picker. Sorted so the list is stable between
    reloads -- a picker whose order moved would make the demo unrepeatable.
    """
    keys = {
        ref
        for decision in run.decisions
        for ref in (decision.source_ref.key, decision.target_ref.key)
    }
    keys |= {
        ref.key for exception in run.exceptions for ref in exception.involved_refs
    }
    return sorted(keys)
