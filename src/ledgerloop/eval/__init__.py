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
"""

from __future__ import annotations

from ledgerloop.eval.baselines import BaselineRun, run_b0
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
from ledgerloop.eval.truth_io import DatasetManifest, load_ground_truth, load_manifest

__all__ = [
    "EVALUATED_RECORD_TYPES",
    "BaselineRun",
    "CalibrationEvaluation",
    "DatasetManifest",
    "EvaluatedRun",
    "LinkConfusion",
    "MatchRateResult",
    "MoneyView",
    "PredictedLink",
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
    "wilson_interval",
    "write_report",
]
