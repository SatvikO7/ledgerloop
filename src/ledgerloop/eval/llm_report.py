"""One controlled run of the **production** LLM path, measured end to end.

WHAT THIS ANSWERS THAT B2 DOES NOT
-----------------------------------
B2 asks *what happens with no gates*. This asks *what the gated path costs and
what it is refused*, on the residual the deterministic ladder actually leaves.
They are different questions and they need different artefacts: B2's numbers are
about accuracy without verification, and these are about machinery under
verification.

Every column here is measured on the real code path -- the same prompts, the
same content-hash cache, the same budget, the same schema validation, the same
grounding gate, the same ``verify_arithmetic``, the same provider ladder and the
same cost ledger.

THE CONTROL, AND WHY IT IS THE POINT
-------------------------------------
The run is executed **twice on the same corpus**: once with the model and once
with ``--no-llm``. Both are scored, and the artefact records both sets of
metrics side by side. That is the control the project's central claim needs --
"the LLM proposes, deterministic code decides" is a claim about *authority*, and
the way to measure authority is to check whether the answer moves when the
model is removed.

:attr:`LLMReportArtifact.metrics_unchanged` is that check. It is ``True`` when
precision, recall, match rate, exception recall and the auto-match count are
identical across the two runs, and the report prints it as a statement rather
than as a footnote.

``False`` is **not** a failure and must not be read as one. One call site --
narration repair -- is allowed to change what the ladder reads: an accepted
repair writes a reference onto a bank row exactly as the regex layer would
have, and everything downstream then decides on it deterministically. So a
moved metric means the model improved an *input*, never that it made a
decision. What would be a failure is a proposal reaching ``AUTO_MATCHED``
without ``arithmetic_verified``, and that is refused at construction by
:class:`~ledgerloop.models.decisions.MatchDecision` rather than measured here.

LIVE VERSUS OFFLINE
-------------------
:attr:`LLMReportArtifact.live` says which. A live run reached a provider on the
ladder and :attr:`provider_used` names the rung that answered. An offline run
was driven by :class:`~ledgerloop.llm.offline_analyst.OfflineAnalyst`, a
documented rule that reads the prompt and nothing else, and **no claim is made
about any model's answer quality** on the strength of it. The report banners the
difference; a reader who cannot tell the two apart has been misled, and that is
the failure this field exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ledgerloop.eval.artifacts import LLMReportArtifact, RunScore
from ledgerloop.eval.harness import DETERMINISTIC_TIERS, SystemRun, run_system
from ledgerloop.llm.client import LLMClient
from ledgerloop.matching.calibration import CalibrationBundle

__all__ = ["run_llm_report", "score_of"]


def score_of(run: SystemRun) -> RunScore:
    """The four headline figures of one run, so two runs can be compared.

    A function rather than a classmethod on :class:`RunScore`: the artefact
    models live in :mod:`ledgerloop.eval.artifacts` precisely so a document
    renderer can import them without importing the harness, and a constructor
    taking a ``SystemRun`` would put the harness back in their import graph.
    """
    links = run.metrics.link_metrics
    return RunScore(
        precision=run.metrics.auto_match_precision,
        recall=links.recall if links is not None else 0.0,
        match_rate=run.metrics.match_rate,
        exception_recall=run.metrics.exception_recall,
        auto_matched=run.auto_matched,
        false_positives=links.false_positives if links is not None else 0,
    )


@dataclass(frozen=True)
class _Timed:
    run: SystemRun
    wall_clock_ms: int


def _timed(**kwargs: object) -> _Timed:
    started = time.perf_counter_ns()
    run = run_system(**kwargs)  # type: ignore[arg-type]
    return _Timed(run=run, wall_clock_ms=(time.perf_counter_ns() - started) // 1_000_000)


def run_llm_report(
    directory: Path,
    *,
    client: LLMClient,
    bundle: CalibrationBundle | None = None,
    live: bool = False,
) -> LLMReportArtifact:
    """Run the corpus with the model and again without it, and score both.

    ``client`` is whatever the caller built -- a real ladder or the offline
    analyst -- and ``live`` says which, because this function cannot tell and
    must not guess. The caller knows whether a key was present; inferring it
    from a provider name would make the artefact's most important field a
    heuristic.
    """
    if not client.enabled:
        return LLMReportArtifact(
            ran=False,
            reason="no provider was reachable and none was asked for",
        )

    with_llm = _timed(
        directory=directory,
        bundle=bundle,
        client=client,
        measure_calibration_quality=False,
        run_id="llm-report-with",
    )
    without_llm = _timed(
        directory=directory,
        bundle=bundle,
        enabled_tiers=DETERMINISTIC_TIERS,
        measure_calibration_quality=False,
        run_id="llm-report-without",
    )

    run = with_llm.run
    summary = run.llm
    return LLMReportArtifact(
        ran=True,
        live=live,
        provider_used=run.cost.provider_used,
        ladder=tuple(getattr(client.provider, "ladder", ()) or ()),
        fallback_depth=run.cost.fallback_depth,
        provider_failures=client.provider_failure_detail,
        split=run.manifest.split.value,
        difficulty=run.manifest.difficulty.value,
        seed=run.manifest.seed,
        generator_version=run.manifest.generator_version,
        record_count=run.metrics.record_count,
        narrations_offered=summary.narration.attempted,
        narrations_accepted=summary.narration.accepted,
        proposals_returned=len(summary.adjudication.hypotheses),
        proposals_accepted=summary.adjudication.accepted,
        rejected_ungrounded=summary.rejected_ungrounded,
        rejected_unverified=summary.rejected_unverified,
        demoted=summary.adjudication.demoted,
        explanations_accepted=summary.explanation.accepted,
        calls_refused=summary.calls_refused,
        validation_failures=client.validation_failures,
        cost=run.cost,
        wall_clock_ms=with_llm.wall_clock_ms + without_llm.wall_clock_ms,
        with_llm=score_of(run),
        without_llm=score_of(without_llm.run),
    )
