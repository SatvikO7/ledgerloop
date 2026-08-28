"""Evaluation: the scoreboard, built before any matcher exists.

Step 2 of the implementation order, and deliberately so. Building the
evaluator first means every later change is measured rather than argued about:
when T2 lands, the question "did the aggregation solver help?" already has a
number waiting for it, produced by code written before anyone had a stake in
the answer.

The package has four parts:

* :mod:`ledgerloop.eval.truth_io` -- read a generated dataset's ground truth
  back off disk. The exact inverse of the generator's emitters.
* :mod:`ledgerloop.eval.metrics` -- link-level precision/recall/F1 with a
  confidence interval, match rate, per-anomaly-class recall, and the money view.
* :mod:`ledgerloop.eval.baselines` -- B0, the exact-join-on-UTR baseline. The
  "why not just SQL" answer, and the harness's first real input.
* :mod:`ledgerloop.eval.reliability` -- labels a finished run's candidates and
  measures ECE, Brier and the reliability bins. Reads ground truth to *label*,
  never to fit: the fitting lives in :mod:`ledgerloop.matching.calibration` and
  never touches the test split.
* :mod:`ledgerloop.eval.report` -- renders ``EVALUATION.md``. Nothing in that
  document is hand-typed.

Step 10 added the comparison the plan's §9 is built around, in five modules:

* :mod:`ledgerloop.eval.harness` -- one scored run of the production pipeline
  over one dataset. The headline, every ablation row and every sweep row go
  through it, so none of them can drift into a second implementation.
* :mod:`ledgerloop.eval.summary` -- the scalars a table averages, and mean ± std.
* :mod:`ledgerloop.eval.artifacts` -- the serialised results, models only.
* :mod:`ledgerloop.eval.ablation` and :mod:`ledgerloop.eval.sweep` -- the
  runners that produce them.
* :mod:`ledgerloop.eval.llm_baseline` -- B2, plus
  :mod:`ledgerloop.eval.offline_provider`, the stand-in that lets it run where
  no provider key exists.

**The models are split from the runners on purpose.** ``report`` renders the
tables and the runners import ``llm``; ``matching`` imports ``eval.metrics`` for
one contract type. Without the split those three facts close into an import
cycle and hand ``matching`` a transitive dependency on ``llm`` -- the one
dependency ARCHITECTURE.md §6, decision 43 forbids.
"""

from __future__ import annotations

from ledgerloop.eval.artifacts import (
    AblationArtifact,
    LLMBaselineArtifact,
    SweepArtifact,
)
from ledgerloop.eval.baselines import BaselineRun, run_b0, run_b1
from ledgerloop.eval.metrics import (
    EVALUATED_RECORD_TYPES,
    LinkConfusion,
    MatchRateResult,
    MoneyView,
    PredictedLink,
    confusion,
    evaluate,
    link_metrics,
    match_rate,
    money_view,
    recall_by_anomaly_class,
    wilson_interval,
)
from ledgerloop.eval.reliability import (
    CalibrationEvaluation,
    label_candidates,
    measure_calibration,
)
from ledgerloop.eval.report import EvaluatedRun, render_report, write_report
from ledgerloop.eval.summary import Aggregate, RunSummary, aggregate, summarise
from ledgerloop.eval.truth_io import DatasetManifest, load_ground_truth, load_manifest

__all__ = [
    "EVALUATED_RECORD_TYPES",
    "AblationArtifact",
    "Aggregate",
    "BaselineRun",
    "CalibrationEvaluation",
    "DatasetManifest",
    "EvaluatedRun",
    "LLMBaselineArtifact",
    "LinkConfusion",
    "MatchRateResult",
    "MoneyView",
    "PredictedLink",
    "RunSummary",
    "SweepArtifact",
    "aggregate",
    "confusion",
    "evaluate",
    "label_candidates",
    "link_metrics",
    "load_ground_truth",
    "load_manifest",
    "match_rate",
    "measure_calibration",
    "money_view",
    "recall_by_anomaly_class",
    "render_report",
    "run_b0",
    "run_b1",
    "summarise",
    "wilson_interval",
    "write_report",
]
