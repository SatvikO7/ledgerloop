"""Isotonic calibration, threshold selection, and the reliability numbers.

PLAN.md 6.5 in three parts, in the order they have to happen:

1. **Isotonic regression** maps the blender's raw score to a probability. Fitted
   on the ``calibration`` split, never on ``train`` and never on ``test``.
2. **Threshold selection** picks ``tau_high`` as the *lowest* threshold whose
   auto-match precision reaches ``target_auto_match_precision`` on the
   calibration split. PLAN.md 6.5: "Thresholds are not hand-picked."
3. **Reliability** -- ECE, Brier and the bin table -- is what says whether step
   1 worked. Measured over the residual tiers only, for the reason
   ``CalibrationMetrics.residual_only`` documents.

WHY ISOTONIC RATHER THAN PLATT
------------------------------
A logistic already produced the score. Fitting a second logistic (Platt) on top
of it can only apply a monotone reparameterisation from the same two-parameter
family, so it cannot fix the shape of a miscalibration -- and the shape is
exactly what is wrong here: the corpus is dominated by candidates that are
right, so the raw scores pile up near one and the middle of the range is
almost empty. Isotonic is non-parametric, needs only monotonicity, and is
allowed to map a whole band of raw scores to one probability -- which is the
honest answer when a band contains too few samples to say anything finer.

The cost of isotonic is that it can only produce values it saw, so a
calibration split with no errors in its top block produces a probability of
exactly 1.0 there. That is not the calibrator being overconfident; it is the
calibration split being too small to distinguish 1.0 from 0.995, and
:attr:`ReliabilityDiagram.populated_bins` and the sample counts are what make
that visible instead of impressive.

THE THREE SPLITS, AND WHAT MAY TOUCH WHICH
------------------------------------------
+----------------+------------------------------------------------------------+
| ``train``      | logistic coefficients only                                 |
| ``calibration``| isotonic map and ``tau_high`` only                          |
| ``test``       | nothing is fitted; every published number is measured here |
+----------------+------------------------------------------------------------+

Nothing in this module reads the test split. The bundle records which splits it
was fitted on so a report cannot claim a discipline it did not follow.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator

from ledgerloop.config import DecisionThresholds, RunConfig
from ledgerloop.matching.blender import DEFAULT_L2, LogisticBlender, fit_logistic
from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.candidates import FeatureVector, MatchCandidate
from ledgerloop.models.enums import SplitName, Tier
from ledgerloop.models.metrics import CalibrationMetrics
from ledgerloop.stats import wilson_interval

__all__ = [
    "DEFAULT_BIN_COUNT",
    "BlendOutcome",
    "CalibrationBundle",
    "CalibrationProvenance",
    "IsotonicCalibrator",
    "ReliabilityBin",
    "ReliabilityDiagram",
    "ThresholdSelection",
    "apply_bundle",
    "brier_score",
    "configure_for",
    "expected_calibration_error",
    "fit_bundle",
    "fit_isotonic",
    "reliability",
    "select_tau_high",
    "thresholds_from",
]

#: Reliability bins. Ten equal-width bins is the convention every ECE paper
#: uses; the count travels in :class:`~ledgerloop.models.metrics.
#: CalibrationMetrics` so a reader never has to assume it.
DEFAULT_BIN_COUNT = 10


class IsotonicCalibrator(FrozenLedgerModel):
    """A monotone, piecewise-constant map from raw score to probability.

    Stored as the pooled blocks that Pool-Adjacent-Violators produced:
    ``thresholds[i]`` is the lowest raw score in block ``i`` and ``values[i]``
    is that block's fitted probability. Prediction is a right-continuous step
    function.

    **Deliberately not interpolated between blocks.** Isotonic regression's
    fitted value is constant on a pooled block; interpolating across the gap
    between two blocks would invent intermediate probabilities the fit never
    produced, and on a corpus this size those gaps are wide. A step is the
    honest reading of what was measured.
    """

    thresholds: tuple[float, ...]
    values: tuple[float, ...]
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _monotone_and_aligned(self) -> IsotonicCalibrator:
        if len(self.thresholds) != len(self.values):
            raise ValueError(
                f"{len(self.thresholds)} block starts against {len(self.values)} values"
            )
        if not self.thresholds:
            raise ValueError("an isotonic calibrator needs at least one block")
        if any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:], strict=False)):
            raise ValueError("block starts must be strictly increasing")
        if any(b < a for a, b in zip(self.values, self.values[1:], strict=False)):
            raise ValueError("fitted values must be non-decreasing -- that is the fit")
        if any(value < 0.0 or value > 1.0 for value in self.values):
            raise ValueError("fitted values must be probabilities")
        if self.positive_count > self.sample_count:
            raise ValueError("positive_count cannot exceed sample_count")
        return self

    @property
    def block_count(self) -> int:
        return len(self.values)

    def predict(self, raw_score: float) -> float:
        """Calibrated probability for one raw score.

        Scores below the first block take the first block's value and scores
        above the last take the last: the calibrator has no evidence outside
        the range it was fitted on, and extrapolating a trend it never observed
        is how a calibrator becomes overconfident precisely where it is least
        informed.
        """
        chosen = self.values[0]
        for threshold, value in zip(self.thresholds, self.values, strict=True):
            if raw_score >= threshold:
                chosen = value
            else:
                break
        return chosen


def fit_isotonic(
    scores: Sequence[float], labels: Sequence[bool]
) -> IsotonicCalibrator:
    """Fit an isotonic regression by Pool-Adjacent-Violators.

    PAVA is exact and deterministic: sort by raw score, then repeatedly merge
    any block whose mean exceeds its right neighbour's until the sequence is
    non-decreasing. The result is the least-squares monotone fit, and it is
    reached in one linear pass with a stack.

    Ties in the raw score are pooled into one block before the sweep. Two
    candidates the blender scored identically cannot be given different
    probabilities without the calibrator inventing an ordering the model never
    expressed.
    """
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")
    if not scores:
        raise ValueError("cannot fit an isotonic calibrator on an empty set")

    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])

    # (block start, sum of labels, count). Ties pooled up front.
    blocks: list[list[float]] = []
    for score, label in ordered:
        value = 1.0 if label else 0.0
        if blocks and blocks[-1][0] == score:
            blocks[-1][1] += value
            blocks[-1][2] += 1.0
        else:
            blocks.append([score, value, 1.0])

    merged: list[list[float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and (
            merged[-2][1] / merged[-2][2] > merged[-1][1] / merged[-1][2]
        ):
            right = merged.pop()
            left = merged.pop()
            merged.append([left[0], left[1] + right[1], left[2] + right[2]])

    return IsotonicCalibrator(
        thresholds=tuple(block[0] for block in merged),
        values=tuple(block[1] / block[2] for block in merged),
        sample_count=len(scores),
        positive_count=sum(1 for label in labels if label),
    )


class ThresholdSelection(FrozenLedgerModel):
    """The fitted ``tau_high`` and the evidence for it.

    Every field here exists so a report can state *how* the threshold was
    chosen. A bare number would be indistinguishable from the hand-picked
    default it replaces, which is the distinction
    :attr:`DecisionThresholds.tau_high_is_fitted` exists to preserve.
    """

    tau_high: float = Field(ge=0.0, le=1.0)
    target_precision: float = Field(ge=0.0, le=1.0)
    achieved_precision: float = Field(ge=0.0, le=1.0)
    precision_ci_low: float = Field(ge=0.0, le=1.0)
    precision_ci_high: float = Field(ge=0.0, le=1.0)
    attained: bool = Field(
        description="Whether any threshold reached the target on the calibration "
        "split. False means the fallback was used and the report must say so."
    )
    auto_matched: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    candidates_considered: int = Field(ge=0)
    positives_available: int = Field(ge=0)

    @property
    def coverage(self) -> float:
        """Share of the calibration candidates this threshold would auto-match."""
        if self.candidates_considered == 0:
            return 0.0
        return self.auto_matched / self.candidates_considered

    @property
    def recall(self) -> float:
        """Share of the calibration split's true links this threshold would keep."""
        if self.positives_available == 0:
            return 0.0
        return self.true_positives / self.positives_available


def select_tau_high(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    target_precision: float,
) -> ThresholdSelection:
    """Lowest threshold reaching ``target_precision`` on the calibration split.

    *Lowest* rather than *safest*: among the thresholds that meet the precision
    target, the lowest is the one that auto-matches the most, and precision is
    already guaranteed by the constraint. Optimising precision further would
    trade away coverage that costs nothing.

    **The point estimate decides; the interval is reported.** PLAN.md 6.5 states
    the rule in terms of achieved precision, and a Wilson lower bound at this
    sample size would demand roughly three hundred consecutive correct
    predictions before any threshold qualified -- the calibration split has a
    few hundred candidates in total. So the selection follows the plan and
    :attr:`precision_ci_low` travels with the answer, which is the same
    discipline every other precision figure in this project follows.

    When no threshold attains the target, ``tau_high`` is **1.0** and
    ``attained`` is ``False``: the precision-first answer to "this model cannot
    be trusted to auto-match at the target" is to auto-match only what the
    calibrator called certain, not to lower the target.
    """
    if len(probabilities) != len(labels):
        raise ValueError(f"{len(probabilities)} probabilities against {len(labels)} labels")

    positives_available = sum(1 for label in labels if label)
    pairs = sorted(zip(probabilities, labels, strict=True), key=lambda pair: -pair[0])

    best: tuple[float, int, int] | None = None
    true_positives = false_positives = 0
    for index, (probability, label) in enumerate(pairs):
        if label:
            true_positives += 1
        else:
            false_positives += 1
        is_last_of_tie = index + 1 == len(pairs) or pairs[index + 1][0] != probability
        if not is_last_of_tie:
            continue
        predicted = true_positives + false_positives
        precision = true_positives / predicted
        if precision >= target_precision:
            # Descending order, so every later qualifying threshold is lower and
            # auto-matches more. Keep overwriting: the last one wins.
            best = (probability, true_positives, false_positives)

    if best is None:
        selected, kept_tp, kept_fp = 1.0, 0, 0
        for probability, label in pairs:
            if probability >= 1.0:
                if label:
                    kept_tp += 1
                else:
                    kept_fp += 1
        best = (selected, kept_tp, kept_fp)
        attained = False
    else:
        attained = True

    tau_high, kept_tp, kept_fp = best
    predicted = kept_tp + kept_fp
    achieved = kept_tp / predicted if predicted else 0.0
    ci_low, ci_high = wilson_interval(kept_tp, predicted)
    return ThresholdSelection(
        tau_high=tau_high,
        target_precision=target_precision,
        achieved_precision=achieved,
        precision_ci_low=ci_low,
        precision_ci_high=ci_high,
        attained=attained,
        auto_matched=predicted,
        true_positives=kept_tp,
        false_positives=kept_fp,
        candidates_considered=len(pairs),
        positives_available=positives_available,
    )


class ReliabilityBin(FrozenLedgerModel):
    """One row of the reliability diagram."""

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_probability: float = Field(ge=0.0, le=1.0)
    empirical_rate: float = Field(ge=0.0, le=1.0)

    @property
    def gap(self) -> float:
        """Signed miscalibration: positive means the bin was overconfident."""
        return self.mean_probability - self.empirical_rate


class ReliabilityDiagram(FrozenLedgerModel):
    """The bin table plus the two scalar summaries.

    Kept together because neither number is interpretable alone. An ECE of
    0.01 over one populated bin says nothing at all, which is why
    :attr:`~ledgerloop.models.metrics.CalibrationMetrics.populated_bins` is part
    of the reported contract.
    """

    bins: tuple[ReliabilityBin, ...]
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    ece: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)
    residual_only: bool = True

    @property
    def populated_bins(self) -> int:
        return sum(1 for item in self.bins if item.count > 0)

    def metrics(self) -> CalibrationMetrics:
        """The reported contract, for :class:`~ledgerloop.models.metrics.RunMetrics`."""
        return CalibrationMetrics(
            ece=self.ece,
            brier=self.brier,
            bin_count=len(self.bins),
            populated_bins=self.populated_bins,
            sample_count=self.sample_count,
            residual_only=self.residual_only,
        )


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error of the probabilities against the outcomes.

    Zero on an empty set, which is the same convention
    :mod:`ledgerloop.eval.metrics` uses for an empty denominator: no evidence
    is reported as no score, never as a perfect one.
    """
    if len(probabilities) != len(labels):
        raise ValueError(f"{len(probabilities)} probabilities against {len(labels)} labels")
    if not probabilities:
        return 0.0
    total = sum(
        (probability - (1.0 if label else 0.0)) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    )
    return total / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> float:
    """Sample-weighted mean gap between confidence and accuracy across bins."""
    return reliability(probabilities, labels, bin_count=bin_count).ece


def reliability(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
    residual_only: bool = True,
) -> ReliabilityDiagram:
    """Bin the probabilities, then measure the gap in each bin.

    Bins are equal width over ``[0, 1]`` and the last one is closed at the top,
    so a probability of exactly 1.0 -- which isotonic produces routinely -- lands
    in the top bin rather than falling out of the table.
    """
    if len(probabilities) != len(labels):
        raise ValueError(f"{len(probabilities)} probabilities against {len(labels)} labels")
    if bin_count < 1:
        raise ValueError(f"bin_count must be at least 1, got {bin_count}")

    width = 1.0 / bin_count
    sums = [0.0] * bin_count
    hits = [0.0] * bin_count
    counts = [0] * bin_count
    for probability, label in zip(probabilities, labels, strict=True):
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"{probability} is not a probability")
        index = min(int(probability / width), bin_count - 1)
        sums[index] += probability
        hits[index] += 1.0 if label else 0.0
        counts[index] += 1

    bins: list[ReliabilityBin] = []
    error = 0.0
    total = len(probabilities)
    for index in range(bin_count):
        count = counts[index]
        mean_probability = sums[index] / count if count else 0.0
        empirical = hits[index] / count if count else 0.0
        bins.append(
            ReliabilityBin(
                lower=index * width,
                upper=(index + 1) * width,
                count=count,
                mean_probability=mean_probability,
                empirical_rate=empirical,
            )
        )
        if count and total:
            error += (count / total) * abs(mean_probability - empirical)

    return ReliabilityDiagram(
        bins=tuple(bins),
        sample_count=total,
        positive_count=sum(1 for label in labels if label),
        ece=error,
        brier=brier_score(probabilities, labels),
        residual_only=residual_only,
    )


class CalibrationProvenance(FrozenLedgerModel):
    """Which data produced this bundle.

    A calibrated probability is only meaningful against the corpus it was
    fitted on, so the bundle carries its own lineage: which splits, which
    seeds, which generator version, and how many rows of each class. A bundle
    fitted on generator ``0.2.0`` is not applicable to ``0.3.0``, and this is
    what makes that checkable rather than assumed.
    """

    train_split: SplitName
    train_seeds: tuple[int, ...]
    calibration_split: SplitName
    calibration_seeds: tuple[int, ...]
    generator_version: str
    top_k: int = Field(ge=1)
    train_rows: int = Field(ge=0)
    train_positives: int = Field(ge=0)
    calibration_rows: int = Field(ge=0)
    calibration_positives: int = Field(ge=0)
    calibration_abstained: int = Field(
        default=0,
        ge=0,
        description="Calibration rows whose tier the fitted blender does not cover, "
        "so they could not be scored and took no part in the isotonic fit.",
    )
    train_rows_by_tier: dict[str, int] = Field(default_factory=dict)
    calibration_rows_by_tier: dict[str, int] = Field(default_factory=dict)

    @property
    def train_corpora(self) -> tuple[tuple[SplitName, int], ...]:
        return tuple((self.train_split, seed) for seed in self.train_seeds)

    @property
    def calibration_corpora(self) -> tuple[tuple[SplitName, int], ...]:
        return tuple((self.calibration_split, seed) for seed in self.calibration_seeds)

    @model_validator(mode="after")
    def _splits_are_disjoint(self) -> CalibrationProvenance:
        """The three-way discipline, enforced rather than trusted.

        Fitting the logistic and the isotonic on the same rows lets the
        calibrator see in-sample scores and report a calibration quality the
        system does not have -- which is the exact reason ARCHITECTURE.md 6
        decision 1 added the ``train`` split. Checked per *corpus*, because both
        halves take several seeds: overlapping on one seed out of five is the
        same leak in a form that would be easy to miss.

        And neither half may ever be ``test``. That is not a convention this
        model trusts a caller to observe -- a bundle that broke it could not be
        constructed at all.
        """
        if not self.train_seeds or not self.calibration_seeds:
            raise ValueError("both halves of the fit need at least one corpus")
        shared = set(self.train_corpora) & set(self.calibration_corpora)
        if shared:
            named = ", ".join(f"{split.value} seed {seed}" for split, seed in sorted(shared))
            raise ValueError(
                f"the logistic and the isotonic must be fitted on different data; "
                f"both name {named}"
            )
        if SplitName.TEST in (self.train_split, self.calibration_split):
            raise ValueError(
                "the test split may not be fitted on -- every published number "
                "comes from it"
            )
        return self


class CalibrationBundle(FrozenLedgerModel):
    """Everything the decision policy needs, fitted and serialisable.

    One object rather than three loose ones, because the three are only valid
    together: an isotonic map applies to the raw scores of *one* logistic, and
    a threshold is a cut on the probabilities of *that* isotonic. Loading them
    separately is how a run ends up applying last week's threshold to this
    week's model.
    """

    blender: LogisticBlender
    calibrator: IsotonicCalibrator
    thresholds: ThresholdSelection
    provenance: CalibrationProvenance
    fit_reliability: ReliabilityDiagram = Field(
        description="Reliability of the calibrated probabilities on the calibration "
        "split itself. In-sample by construction, so it is a fit diagnostic and "
        "never the reported calibration -- that comes from the test split."
    )

    def covers(self, candidate: MatchCandidate) -> bool:
        """Whether this bundle was fitted for the candidate's tier."""
        return self.blender.covers(candidate.tier)

    def probability(self, candidate: MatchCandidate) -> float:
        """Blend then calibrate, in one step.

        Raises for a tier the blender never saw. The strict form is the right
        default -- a probability for an unfitted tier would be the reference
        tier's probability wearing someone else's name -- and
        :meth:`probability_for` is the form for callers that would rather
        abstain than fail.
        """
        return self.calibrator.predict(self.blender.score_candidate(candidate))

    def probability_for(self, candidate: MatchCandidate) -> float | None:
        """The calibrated probability, or ``None`` where the bundle abstains."""
        if not self.covers(candidate):
            return None
        return self.probability(candidate)

    def save(self, path: Path) -> Path:
        """Write the bundle as indented JSON, with ``\\n`` endings everywhere.

        Indented and key-sorted so two fits diff line by line: a coefficient
        that moved is the interesting part of a rerun, and a single-line dump
        would hide it inside one changed line.
        """
        payload = json.loads(self.model_dump_json())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path) -> CalibrationBundle:
        with path.open("r", encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())


@dataclass(frozen=True)
class BlendOutcome:
    """What applying a bundle to one run's candidates actually did.

    Four counters rather than one, because "the blender ran" is not a single
    event: most candidates never reach it, and the ones that do not are the
    interesting part of the story.
    """

    scored: int = 0
    bypassed_deterministic: int = 0
    refusals_kept: int = 0
    abstained_uncovered: int = 0

    @property
    def considered(self) -> int:
        return (
            self.scored
            + self.bypassed_deterministic
            + self.refusals_kept
            + self.abstained_uncovered
        )

    def merge(self, other: BlendOutcome) -> BlendOutcome:
        return BlendOutcome(
            scored=self.scored + other.scored,
            bypassed_deterministic=self.bypassed_deterministic
            + other.bypassed_deterministic,
            refusals_kept=self.refusals_kept + other.refusals_kept,
            abstained_uncovered=self.abstained_uncovered + other.abstained_uncovered,
        )


def apply_bundle(
    candidates: Sequence[MatchCandidate], bundle: CalibrationBundle
) -> BlendOutcome:
    """Score and calibrate in place, respecting both structural exclusions.

    Three populations are left exactly as the tiers set them, and each for a
    different reason:

    * **T0 and T1** -- ``Tier.is_deterministic_certain``. Their probability is
      not an estimate but a count: ``1/n`` over indistinguishable contenders,
      and 1.0 where the key resolved uniquely. There is nothing for a model to
      improve, and including them would swamp the calibration measurement.
    * **Refusals** -- any candidate a tier emitted with
      ``arithmetic_verified=False``. That flag marks a conclusion the tier
      reached on evidence the feature vector does not carry: a second subset
      that also fits, a runner-up the name score could not beat. A model that
      cannot see the rival must not overturn the refusal, and the policy would
      demote it to review anyway, so the tier's ``1/n`` stands and the routing
      it produces (an ambiguity at 0.5 falling below ``tau_low``) is preserved.
    * **Tiers the model was never fitted for** -- ``covers()`` is ``False``.
      Scoring them would silently treat them as the reference tier. The
      abstention is counted and reported.

    Mutates the candidates because :class:`~ledgerloop.models.candidates.
    MatchCandidate` is the one mutable contract in the system, filled in
    progressively by the stage that knows each field. The immutable record of
    what was concluded is the decision.
    """
    outcome = BlendOutcome()
    for candidate in candidates:
        if candidate.tier.is_deterministic_certain:
            outcome = outcome.merge(BlendOutcome(bypassed_deterministic=1))
            continue
        if not candidate.arithmetic_verified:
            outcome = outcome.merge(BlendOutcome(refusals_kept=1))
            continue
        if not bundle.blender.covers(candidate.tier):
            outcome = outcome.merge(BlendOutcome(abstained_uncovered=1))
            continue
        raw = bundle.blender.score_candidate(candidate)
        candidate.raw_score = raw
        candidate.calibrated_p = bundle.calibrator.predict(raw)
        outcome = outcome.merge(BlendOutcome(scored=1))
    return outcome


def thresholds_from(
    selection: ThresholdSelection, base: DecisionThresholds
) -> DecisionThresholds:
    """The routing thresholds a fitted selection implies.

    ``tau_low`` is **not** fitted -- PLAN.md 6.5 gives it as a policy constant,
    and there is no objective to select it against the way precision selects
    ``tau_high``. It is only ever lowered here, and only when a fitted
    ``tau_high`` lands beneath it: an inverted pair is invalid, and narrowing
    the exception band preserves the fitted threshold rather than adjusting it
    to fit a constant that was never measured.
    """
    return DecisionThresholds(
        tau_high=selection.tau_high,
        tau_low=min(base.tau_low, selection.tau_high),
        target_auto_match_precision=base.target_auto_match_precision,
        tau_high_is_fitted=True,
    )


def configure_for(config: RunConfig, bundle: CalibrationBundle) -> RunConfig:
    """A copy of ``config`` carrying the bundle's fitted thresholds.

    The threshold has to live on the :class:`~ledgerloop.config.RunConfig`
    rather than being passed beside it, because ``config_hash`` is what
    identifies a result -- a run under a fitted ``tau_high`` must not hash
    identically to a run under the placeholder default.
    """
    return config.model_copy(
        update={"thresholds": thresholds_from(bundle.thresholds, config.thresholds)}
    )


def fit_bundle(
    train_features: Sequence[FeatureVector],
    train_labels: Sequence[bool],
    calibration_features: Sequence[FeatureVector],
    calibration_labels: Sequence[bool],
    *,
    provenance: CalibrationProvenance,
    target_precision: float,
    l2: float = DEFAULT_L2,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> CalibrationBundle:
    """Fit the logistic, then the isotonic, then the threshold -- in that order.

    The order is the discipline. Each stage consumes the previous stage's
    *output on data the previous stage never saw*: the isotonic is fitted on
    calibration-split raw scores from a logistic fitted on train, and the
    threshold is a cut on calibration-split probabilities from that isotonic.
    Fitting them together, or on one split, is how a calibration curve ends up
    describing the fit rather than the system.

    Calibration rows whose tier the fitted blender does not cover are dropped
    rather than scored -- see :meth:`LogisticBlender.covers` -- and counted in
    the provenance, because a calibration split that is mostly abstentions
    would otherwise look like a calibration split that was mostly agreement.
    """
    blender = fit_logistic(train_features, train_labels, l2=l2)

    scored: list[float] = []
    kept_labels: list[bool] = []
    abstained = 0
    for features, label in zip(calibration_features, calibration_labels, strict=True):
        if not blender.covers(features.tier):
            abstained += 1
            continue
        scored.append(blender.score(features))
        kept_labels.append(label)

    if not scored:
        raise ValueError(
            "no calibration row could be scored by the fitted blender; the two "
            "halves of the fit have no tier in common"
        )

    calibrator = fit_isotonic(scored, kept_labels)
    probabilities = [calibrator.predict(score) for score in scored]
    selection = select_tau_high(
        probabilities, kept_labels, target_precision=target_precision
    )
    return CalibrationBundle(
        blender=blender,
        calibrator=calibrator,
        thresholds=selection,
        provenance=provenance.model_copy(update={"calibration_abstained": abstained}),
        fit_reliability=reliability(
            probabilities, kept_labels, bin_count=bin_count, residual_only=True
        ),
    )


def residual_rows(
    candidates: Sequence[MatchCandidate],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Calibrated probabilities and truth labels for the residual tiers.

    The population the calibration report is measured over: evaluation-unit
    candidates from T2 upwards that carry both a probability and a ground-truth
    label. T0 and T1 are excluded here for the same reason they bypass the
    blender, and :attr:`CalibrationMetrics.residual_only` records that the
    exclusion happened.
    """
    probabilities: list[float] = []
    labels: list[bool] = []
    for candidate in candidates:
        if candidate.tier.is_deterministic_certain or not candidate.is_evaluable:
            continue
        if candidate.calibrated_p is None or candidate.is_truth_positive is None:
            continue
        probabilities.append(candidate.calibrated_p)
        labels.append(candidate.is_truth_positive)
    return tuple(probabilities), tuple(labels)


def rows_by_tier(candidates: Sequence[MatchCandidate]) -> dict[str, int]:
    """Candidate counts keyed by tier name, for the provenance record."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.tier.name] = counts.get(candidate.tier.name, 0) + 1
    return {tier.name: counts[tier.name] for tier in Tier if tier.name in counts}


__all__ += ["residual_rows", "rows_by_tier"]
