"""Measuring calibration on the test split -- labelling, never fitting.

:mod:`ledgerloop.matching.calibration` *fits* the calibrator on the calibration
split. This module *measures* the result on the test split, which is a
different act with a different rule attached: ground truth is read here to
attach a label to a probability the run already produced, and never to change
one.

The distinction is enforced by where the code lives. Nothing in ``matching``
imports this module, so no fitted artefact can be derived from test data even
by accident; and this module never returns anything a decision consumes.

WHAT IS MEASURED, AND OVER WHICH POPULATION
-------------------------------------------
Two populations, reported side by side, because neither answers the other's
question:

* **Asserted** -- the residual candidates the run actually scored. This is what
  the deployed system's probabilities meant on this corpus. It is also small
  and, on a corpus the tiers refuse to guess on, almost entirely correct, so
  its reliability diagram tends to have one populated bin.
* **Contender** -- every pairing the residual tiers considered, harvested top-k
  and labelled the same way the training set was. Larger, and it contains the
  wrong pairings, so it is the population where a reliability diagram has
  anything to show. It is a diagnostic of the calibrator, not a claim about the
  system: the system never asserted most of these.

``CalibrationMetrics.residual_only`` is ``True`` for both. T0 and T1 are
excluded for the reason that field documents -- ~70% of volume at p = 1.0
produces an ECE that measures the shape of the corpus rather than the quality
of the calibrator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ledgerloop.matching.calibration import (
    DEFAULT_BIN_COUNT,
    CalibrationBundle,
    ReliabilityDiagram,
    reliability,
    residual_rows,
)
from ledgerloop.matching.harvest import LabelledCandidate
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.truth import GroundTruth

__all__ = [
    "CalibrationEvaluation",
    "ContenderScores",
    "label_candidates",
    "measure_calibration",
    "score_contenders",
]


@dataclass(frozen=True)
class ContenderScores:
    """A harvested contender population, scored by a bundle.

    ``abstained`` counts the contenders whose tier the bundle was never fitted
    for. They are dropped rather than scored as the reference tier, and counted
    rather than dropped silently -- a diagnostic computed over half a population
    while claiming to describe all of it is worse than no diagnostic.
    """

    probabilities: tuple[float, ...]
    labels: tuple[bool, ...]
    abstained: int = 0


def score_contenders(
    bundle: CalibrationBundle, rows: Sequence[LabelledCandidate]
) -> ContenderScores:
    """Score every harvested contender the bundle covers."""
    probabilities: list[float] = []
    labels: list[bool] = []
    abstained = 0
    for row in rows:
        probability = bundle.probability_for(row.candidate)
        if probability is None:
            abstained += 1
            continue
        probabilities.append(probability)
        labels.append(row.is_positive)
    return ContenderScores(
        probabilities=tuple(probabilities),
        labels=tuple(labels),
        abstained=abstained,
    )


def label_candidates(
    candidates: Sequence[MatchCandidate], truth: GroundTruth
) -> int:
    """Attach ground-truth labels to evaluation-unit candidates, in place.

    Returns how many were labelled. Only ``PAYMENT_CREDITED_AS`` candidates get
    a label, because ``evaluation_pairs`` is the only truth set defined over
    them -- ARCHITECTURE.md 2.

    Called **after** the run has decided everything. ``MatchCandidate.
    is_truth_positive`` exists for exactly this, and its own docstring records
    the rule: populated only when building a training or calibration set, or
    when scoring a finished run. A tier reading it would make every metric in
    the project meaningless.
    """
    labelled = 0
    pairs = truth.evaluation_pairs
    for candidate in candidates:
        if not candidate.is_evaluable:
            continue
        candidate.is_truth_positive = candidate.pair in pairs
        labelled += 1
    return labelled


@dataclass(frozen=True)
class CalibrationEvaluation:
    """The two diagrams, plus the counts that say how much to trust them."""

    asserted: ReliabilityDiagram
    contenders: ReliabilityDiagram | None = None

    @property
    def sample_count(self) -> int:
        return self.asserted.sample_count


def measure_calibration(
    candidates: Sequence[MatchCandidate],
    truth: GroundTruth,
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
    contender_probabilities: Sequence[float] = (),
    contender_labels: Sequence[bool] = (),
) -> CalibrationEvaluation:
    """Label a finished run's candidates and measure their calibration.

    ``contender_probabilities`` and ``contender_labels`` are optional and come
    from a harvest over the same dataset. When absent, only the asserted
    population is reported -- an absent diagram rather than an empty one, on
    the same principle as the rest of the report: a zero for something that was
    not measured is a false measurement.
    """
    label_candidates(candidates, truth)
    probabilities, labels = residual_rows(candidates)
    asserted = reliability(
        probabilities, labels, bin_count=bin_count, residual_only=True
    )
    contenders = None
    if contender_probabilities:
        contenders = reliability(
            contender_probabilities,
            contender_labels,
            bin_count=bin_count,
            residual_only=True,
        )
    return CalibrationEvaluation(asserted=asserted, contenders=contenders)
