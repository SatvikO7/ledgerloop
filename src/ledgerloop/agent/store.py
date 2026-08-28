"""Where a finished run is written, and how it is read back.

One directory per run under ``reports/runs/<run_id>/``:

===================  ==========================================================
``run.json``         headline metrics, config, tier table, counts, LLM cost
``audit.jsonl``      the append-only event log -- the replay's source
``exceptions.json``  the queue, in rupee-impact order, with evidence chains
``decisions.json``   every decision on the evaluation unit, with its tier
===================  ==========================================================

WHY FILES AND NOT A DATABASE
----------------------------
PLAN.md §3.1 draws Postgres. It is not here for the same reason Neo4j and
Chroma are not (ARCHITECTURE.md §5): the project's claim is that it runs on
nothing, and a run record that a judge can `cat`, diff and commit is worth more
at this scale than one behind a connection string. JSONL is already the audit
format Step 0 defined (``AuditEvent.to_jsonl``), so the durable layer costs no
new dependency and no new contract.

The trade is named: no concurrent writers, no query language, and a directory
listing instead of an index. At a few hundred runs of a few hundred records
that is not a constraint anyone feels.

WHY THE UI READS THIS AND NOT THE PIPELINE
------------------------------------------
Step 12's Streamlit app renders **only** what is in these files. It cannot
recompute a metric, re-derive a decision or re-classify an exception, because
it never holds the objects that could. That is what stops a UI from quietly
becoming a second implementation of the system -- the failure mode where a
dashboard and a report disagree and nobody can say which is right.

Everything here is derived from a :class:`~ledgerloop.eval.harness.SystemRun`.
Nothing is computed that the run did not already decide.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ledgerloop.agent.audit import AUDIT_FILE, AuditLog, read_audit_jsonl
from ledgerloop.eval.harness import SystemRun
from ledgerloop.models.audit import AuditEvent
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, LinkType
from ledgerloop.models.recon_exception import ReconException

__all__ = [
    "AUDIT_FILE",
    "DECISIONS_FILE",
    "EXCEPTIONS_FILE",
    "RUNS_ROOT",
    "RUN_FILE",
    "StoredRun",
    "list_runs",
    "load_run",
    "save_run",
]

RUNS_ROOT = Path("reports/runs")
RUN_FILE = "run.json"
EXCEPTIONS_FILE = "exceptions.json"
DECISIONS_FILE = "decisions.json"


@dataclass(frozen=True)
class StoredRun:
    """One run, read back off disk. The UI's entire input.

    ``summary`` is the parsed ``run.json``; the other three are the parsed
    files beside it. Nothing here is recomputed -- see the module docstring.
    """

    run_id: str
    directory: Path
    summary: dict[str, Any]
    audit: tuple[AuditEvent, ...]
    exceptions: tuple[ReconException, ...]
    decisions: tuple[MatchDecision, ...]

    @property
    def dataset(self) -> str:
        return str(self.summary.get("dataset", {}).get("directory", ""))

    @property
    def metrics(self) -> dict[str, Any]:
        value = self.summary.get("metrics", {})
        return value if isinstance(value, dict) else {}

    def decisions_for(self, record_key: str) -> tuple[MatchDecision, ...]:
        """Every decision naming a record. The Audit Replay screen's lookup."""
        return tuple(
            decision
            for decision in self.decisions
            if record_key in (decision.source_ref.key, decision.target_ref.key)
        )

    def audit_for(self, record_key: str) -> tuple[AuditEvent, ...]:
        return tuple(
            event
            for event in self.audit
            if any(ref.key == record_key for ref in event.refs)
        )


def _decision_rows(run: SystemRun) -> list[MatchDecision]:
    """Evaluation-unit decisions only, in the order the policy made them.

    Restricted to ``PAYMENT_CREDITED_AS``: the structural ``ORDER_PAID_BY`` and
    intermediate ``SETTLEMENT_CREDITED_AS`` edges are decided and audited but
    never scored (ARCHITECTURE.md §2), and putting 283 of them in the UI's
    decision table would bury the 130 that the metrics are about.
    """
    return [
        decision
        for decision in run.matched.decisions
        if decision.link_type is LinkType.PAYMENT_CREDITED_AS
    ]


def _summary(run: SystemRun, log: AuditLog) -> dict[str, Any]:
    """The headline record. Every number is read off the run, never recomputed."""
    metrics = run.metrics
    links = metrics.link_metrics
    coverage = run.coverage
    decisions = _decision_rows(run)
    outcomes = {outcome.value: 0 for outcome in DecisionOutcome}
    for decision in decisions:
        outcomes[decision.outcome.value] += 1

    return {
        "run_id": run.config.run_id,
        "engine": "langgraph",
        "dataset": {
            "directory": str(run.directory),
            "split": run.manifest.split.value,
            "difficulty": run.manifest.difficulty.value,
            "seed": run.manifest.seed,
            "generator_version": run.manifest.generator_version,
            "records": len(run.truth.records),
            "evaluation_links": len(run.truth.evaluation_pairs),
        },
        "config": {
            "config_hash": run.config.config_hash,
            "tuning_hash": run.config.tuning_hash,
            "enabled_tiers": list(run.config.enabled_tiers),
            "tau_high": run.config.thresholds.tau_high,
            "tau_low": run.config.thresholds.tau_low,
            "tau_high_is_fitted": run.config.thresholds.tau_high_is_fitted,
            "llm_enabled": run.config.llm.enabled,
        },
        "metrics": {
            "auto_match_precision": metrics.auto_match_precision,
            "precision_ci_low": links.precision_ci_low if links else 0.0,
            "precision_ci_high": links.precision_ci_high if links else 1.0,
            "recall": links.recall if links else 0.0,
            "f1": links.f1 if links else 0.0,
            "match_rate": metrics.match_rate,
            "exception_recall": metrics.exception_recall,
            "true_positives": links.true_positives if links else 0,
            "false_positives": links.false_positives if links else 0,
            "false_negatives": links.false_negatives if links else 0,
            "false_positive_cost_minor": links.false_positive_cost_minor if links else 0,
            "reconciled_minor": metrics.reconciled_minor,
            "outstanding_minor": metrics.outstanding_minor,
            "unmatchable_count": metrics.unmatchable_count,
            "unmatchable_impact_minor": metrics.unmatchable_impact_minor,
            "records_per_second": metrics.records_per_second,
            "wall_clock_ms": run.matched.wall_clock_ms,
        },
        "recall_by_anomaly_class": {
            anomaly.value: value
            for anomaly, value in sorted(
                metrics.recall_by_anomaly_class.items(), key=lambda item: item[0].value
            )
        },
        "exception_confusion": metrics.exception_confusion,
        "decisions": {
            "evaluation_unit_total": len(decisions),
            **outcomes,
            "candidates_proposed": run.candidates_proposed,
            "structural_decisions": len(run.matched.decisions) - len(decisions),
        },
        "coverage": {
            "expected": len(coverage.expected),
            "covered_expected": len(coverage.covered_expected),
            "missed": sorted(coverage.missed),
            "unmatchable": len(coverage.unmatchable),
            "covered_unmatchable": len(coverage.covered_unmatchable),
            "unmatchable_recall": coverage.unmatchable_recall,
            "out_of_scope": coverage.out_of_scope,
        },
        "tiers": [
            {
                "tier": row.tier.name,
                "candidates_proposed": row.candidates_proposed,
                "auto_matched": row.auto_matched,
                "marginal_auto_matched": row.marginal_auto_matched,
                "wall_clock_ms": row.wall_clock_ms,
            }
            for row in metrics.tier_contributions
        ],
        "residual_passes": run.matched.passes,
        "llm": {
            "available": run.llm_available,
            "calls": run.cost.llm_calls,
            "cache_hits": run.cost.cache_hits,
            "total_tokens": run.cost.total_tokens,
            "actual_cost_inr": run.cost.actual_cost_inr,
            "equivalent_paid_cost_inr": run.cost.equivalent_paid_cost_inr,
            "accepted": run.llm.accepted,
            "rejected_ungrounded": run.llm.rejected_ungrounded,
            "rejected_unverified": run.llm.rejected_unverified,
            "prose_rewritten": run.llm.explanation.rewritten,
        },
        "calibration": (
            run.metrics.calibration.model_dump(mode="json")
            if run.metrics.calibration is not None
            else None
        ),
        "ingest": {
            "quarantined": len(run.ingest.problems),
            "date_basis": run.ingest.date_order.basis,
            "credits_with_utr": run.matched.credits_with_utr,
            "credits_seen": run.matched.credits_seen,
        },
        "audit": {
            "events": len(log.events),
            "by_node": log.by_node(),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")


def save_run(
    run: SystemRun,
    log: AuditLog,
    *,
    root: Path = RUNS_ROOT,
) -> Path:
    """Write one run's four files and return its directory.

    The directory is named for the ``run_id``, so a re-run of the same
    configuration over the same corpus overwrites its own record rather than
    accumulating near-duplicates the UI would have to disambiguate.
    """
    directory = root / run.config.run_id
    directory.mkdir(parents=True, exist_ok=True)

    _write_json(directory / RUN_FILE, _summary(run, log))
    log.write_jsonl(directory / AUDIT_FILE)
    _write_json(
        directory / EXCEPTIONS_FILE,
        [exception.model_dump(mode="json") for exception in run.exceptions],
    )
    _write_json(
        directory / DECISIONS_FILE,
        [decision.model_dump(mode="json") for decision in _decision_rows(run)],
    )
    return directory


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def load_run(directory: Path) -> StoredRun | None:
    """Read one run back. ``None`` when the directory holds no ``run.json``.

    A run whose ``run.json`` is missing or unparseable is **not** a run: the
    other three files describe it but cannot be interpreted without the
    headline. Returning ``None`` lets the UI list what it can read and say so,
    rather than render a half-populated screen.
    """
    summary = _read_json(directory / RUN_FILE)
    if not isinstance(summary, dict):
        return None

    exceptions_raw = _read_json(directory / EXCEPTIONS_FILE) or []
    decisions_raw = _read_json(directory / DECISIONS_FILE) or []
    return StoredRun(
        run_id=str(summary.get("run_id", directory.name)),
        directory=directory,
        summary=summary,
        audit=read_audit_jsonl(directory / AUDIT_FILE),
        exceptions=_parse_all(exceptions_raw, ReconException),
        decisions=_parse_all(decisions_raw, MatchDecision),
    )


def _parse_all(rows: Sequence[Any], model: Any) -> tuple[Any, ...]:
    """Parse what parses and drop what does not.

    A single unreadable row -- from a model change between the write and the
    read -- must not make the whole run unopenable. The UI shows the count it
    got; it never claims a queue is empty when the file was merely stale.
    """
    parsed = []
    for row in rows:
        try:
            parsed.append(model.model_validate(row))
        except ValueError:
            continue
    return tuple(parsed)


def list_runs(root: Path = RUNS_ROOT) -> tuple[StoredRun, ...]:
    """Every readable run under ``root``, newest first by directory mtime."""
    if not root.is_dir():
        return ()
    found = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        stored = load_run(directory)
        if stored is not None:
            found.append((directory.stat().st_mtime, stored))
    return tuple(run for _, run in sorted(found, key=lambda item: -item[0]))
