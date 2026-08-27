"""The blender -- feature vector to raw score, by logistic regression.

PLAN.md 6.5: "A **logistic regression** (deliberately simple, inspectable, and
fast) maps features -> raw score." This module is that regression, its design
matrix, and nothing else. The mapping from raw score to a *probability* is
isotonic and lives in :mod:`ledgerloop.matching.calibration`, because the two
are fitted on different splits and conflating them is exactly the mistake the
``train`` split was added to prevent.

WHO IS SCORED, AND WHO IS NOT
-----------------------------
Two populations never reach this model, and both exclusions are structural
rather than a filter someone has to remember:

* **T0 and T1 bypass it entirely.** ``Tier.is_deterministic_certain`` is the
  gate. An exact-key join's correctness follows from the match itself, and
  ~70% of volume at ``p = 1.0`` would dominate the fit, flatten every other
  coefficient and produce a reliability diagram with one populated bin.
* **A tier's *refusal* is never re-scored.** Where T2 found two subsets that
  both fit, or T3 could not separate the best name from the runner-up, the
  tier emits the candidate with ``arithmetic_verified=False`` and a ``1/n``
  probability. That refusal rests on evidence the feature vector does not
  carry -- the existence of a rival -- so a model that cannot see the rival
  must not overturn the conclusion. See :func:`~ledgerloop.matching.
  calibration.apply_bundle`.

TIER IS CATEGORICAL, NEVER ORDINAL
----------------------------------
ARCHITECTURE.md 6 decision 3. ``Tier`` is an ``IntEnum`` so tiers order
naturally in a ladder, but feeding that integer to a logistic regression would
let a near-perfectly predictive variable dominate every coefficient and collapse
the model into a tier lookup. The design matrix therefore carries **one
indicator per fitted tier above the reference level**, and the reference level
is the lowest tier present in the training data.

Only tiers actually seen in training get a level. A tier absent from the fit is
not silently treated as the reference level -- ``covers()`` returns ``False``
and the caller abstains, leaving the tier's own provisional probability in
place. T4 fires zero times on this corpus and T5 does not exist yet, so this is
the common case rather than a defensive flourish.

WHY NOT scikit-learn
--------------------
PLAN.md 10 names scikit-learn for "logistic + isotonic regression only". The
same argument that cut NetworkX (ARCHITECTURE.md 6 decision 29) and pandas
(decision 9) applies with more force here: this is a ridge-penalised IRLS over
about a dozen columns and a few hundred rows, which is the sixty lines below,
against a NumPy + SciPy transitive tree two orders of magnitude larger than the
whole project. Two further reasons specific to this step:

* **Reproducibility.** A hand-written Newton iteration with a fixed tolerance
  reproduces byte for byte; an L-BFGS implementation whose convergence path can
  change between library versions does not.
* **Inspectability.** The pitch says the blender is inspectable. A coefficient
  table the report prints, produced by an optimiser written in the same file,
  is inspectable in a way ``LogisticRegression().fit()`` is not.

The trade is real and named: scikit-learn is better tested than this file, so
this file is tested against closed-form and separable cases whose answers are
known independently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.candidates import FeatureVector, MatchCandidate
from ledgerloop.models.enums import Tier

__all__ = [
    "BASE_FEATURE_NAMES",
    "DEFAULT_L2",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "BlenderError",
    "LogisticBlender",
    "encode_base",
    "feature_names",
    "fit_logistic",
    "sigmoid",
    "solve_symmetric",
]

#: Ridge strength. Small enough to leave the fit data-driven, large enough that
#: perfectly separable training data -- which this corpus produces easily, a
#: lexical score of 1.0 against one of 0.6 -- yields finite coefficients instead
#: of a divergence. Without it a separable column drives its weight to infinity
#: and every candidate lands at exactly 0 or 1, which is not a score.
DEFAULT_L2 = 1.0

#: Newton iterations. IRLS on a problem this size converges in well under ten;
#: the cap exists so a pathological input ends rather than hangs.
DEFAULT_MAX_ITERATIONS = 50

#: Convergence tolerance on the largest coefficient step.
DEFAULT_TOLERANCE = 1e-9

#: Cap on ``amount_delta_ratio``. ``delta_ratio`` returns ``inf`` when the base
#: is zero, and one infinite design-matrix entry destroys the whole fit rather
#: than one row. One is already "the discrepancy is the size of the amount", so
#: everything past it is the same statement about a worse pairing.
_RATIO_CAP = 1.0

#: Cap on how many tolerance bands the amount gap may span. Past a few bands a
#: pairing is simply wrong, and the unbounded tail would let one absurd row set
#: the scale of the column.
_BAND_CAP = 5.0

#: Date gaps are scaled by a month. T1 works in +/-3 days and T3 in +/-7, so a
#: month sits comfortably outside every gate and keeps the column in [0, 1].
_DAY_SCALE = 30.0

#: Subset sizes are scaled by the configured exhaustive-search cap.
_SUBSET_SCALE = 40.0

#: The columns that do not depend on which tiers were fitted, in order.
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "amount_delta_ratio",
    "amount_delta_bands",
    "date_delta_days",
    "lexical_score",
    "semantic_score",
    "graph_support",
    "subset_size",
    "llm_confidence",
    "llm_confidence_present",
)


class BlenderError(ValueError):
    """Raised when a fit cannot be performed or a model cannot be applied."""


def sigmoid(z: float) -> float:
    """Logistic function, written so neither tail overflows.

    ``math.exp(800)`` raises ``OverflowError``, and even a ridge-penalised fit
    can produce a linear predictor in the hundreds on a confident row. The
    branch keeps the exponent negative in both directions.
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _clip(value: float, low: float, high: float) -> float:
    """Bound one column.

    No NaN guard, deliberately. Every float field on
    :class:`~ledgerloop.models.candidates.FeatureVector` carries a ``ge``
    bound, and pydantic rejects NaN against one -- so a NaN cannot reach this
    function without the model having been bypassed, and a branch that can only
    be reached by breaking a contract is untested code pretending to be safety.
    ``inf`` *can* arrive (``delta_ratio`` returns it on a zero base) and is
    handled by the comparison below.
    """
    return low if value < low else high if value > high else value


def encode_base(features: FeatureVector) -> tuple[float, ...]:
    """The tier-independent half of a design row, in :data:`BASE_FEATURE_NAMES` order.

    Every column is bounded and roughly unit-scaled. That is not cosmetic: an
    unscaled column with a long tail makes the Hessian ill-conditioned, and a
    Newton step on an ill-conditioned Hessian is where a hand-written optimiser
    actually fails. Each cap is stated as a constant above and each is a claim
    about the domain rather than a number that made some fit converge.

    ``semantic_score``, ``llm_confidence`` and ``llm_confidence_present`` are
    structurally zero in the MVP -- ChromaDB is cut and T5 does not exist. They
    are carried anyway, because PLAN.md 6.5's feature list is the contract and a
    column that is always zero contributes exactly zero to the gradient, so its
    coefficient stays 0.0 and says so in the printed table. Adding the columns
    later would change the design matrix and silently invalidate every model
    fitted before the change.
    """
    band = max(features.tolerance_band_minor, 1)
    return (
        _clip(features.amount_delta_ratio, 0.0, _RATIO_CAP),
        _clip(abs(features.amount_delta_minor) / band, 0.0, _BAND_CAP),
        _clip(abs(features.date_delta_days) / _DAY_SCALE, 0.0, 1.0),
        _clip(features.lexical_score, 0.0, 1.0),
        _clip(features.semantic_score, 0.0, 1.0),
        _clip(features.graph_support, 0.0, 1.0),
        _clip(features.subset_size / _SUBSET_SCALE, 0.0, 1.0),
        0.0 if features.llm_confidence is None else _clip(features.llm_confidence, 0.0, 1.0),
        0.0 if features.llm_confidence is None else 1.0,
    )


def feature_names(tier_levels: Sequence[Tier]) -> tuple[str, ...]:
    """Column names for a model fitted over ``tier_levels``.

    The first level is the reference and gets no column: with an intercept in
    the model a full one-hot is collinear, and the ridge would then split one
    effect arbitrarily across two coefficients that mean something only when
    added together. Naming the reference is what makes the printed table
    readable -- ``tier=T3_FUZZY`` is a shift *relative to* the reference tier.
    """
    return tuple(f"tier={tier.name}" for tier in tier_levels[1:]) + BASE_FEATURE_NAMES


class LogisticBlender(FrozenLedgerModel):
    """A fitted logistic regression over candidate features.

    Frozen and serialisable: the model travels in the calibration bundle beside
    the isotonic map and the selected threshold, and a report that prints a
    number has to be able to print the coefficients that produced it.
    """

    tier_levels: tuple[Tier, ...] = Field(
        description="Tiers seen during fitting, ascending. The first is the "
        "reference level and has no coefficient of its own."
    )
    coefficients: tuple[float, ...]
    intercept: float
    l2: float = Field(ge=0.0)
    iterations: int = Field(ge=0)
    converged: bool
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    log_likelihood: float

    @model_validator(mode="after")
    def _shapes_agree(self) -> LogisticBlender:
        if not self.tier_levels:
            raise ValueError("a blender must be fitted over at least one tier")
        if len(self.coefficients) != len(self.feature_names):
            raise ValueError(
                f"{len(self.coefficients)} coefficients against "
                f"{len(self.feature_names)} feature names"
            )
        if self.positive_count > self.sample_count:
            raise ValueError("positive_count cannot exceed sample_count")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        return feature_names(self.tier_levels)

    @property
    def negative_count(self) -> int:
        return self.sample_count - self.positive_count

    @property
    def single_class(self) -> bool:
        """Whether the fit saw only one label.

        Not an error -- the ridge keeps the coefficients finite -- but a model
        fitted on one class has learned a base rate and nothing else, and every
        report that prints its output says so rather than presenting it as a
        discrimination it never demonstrated.
        """
        return self.sample_count > 0 and self.positive_count in (0, self.sample_count)

    def covers(self, tier: Tier) -> bool:
        """Whether this model was fitted for ``tier``.

        A tier absent from the fit has no indicator column, so scoring it would
        silently treat it as the reference level -- a T5 candidate scored as if
        it were a T2 aggregation. The caller abstains instead, and the number of
        abstentions is reported rather than absorbed.
        """
        return tier in self.tier_levels

    def encode(self, features: FeatureVector) -> tuple[float, ...]:
        """Full design row: tier indicators, then the base columns."""
        if not self.covers(features.tier):
            raise BlenderError(
                f"blender was fitted over {[t.name for t in self.tier_levels]} and "
                f"cannot encode a {features.tier.name} candidate"
            )
        indicators = tuple(
            1.0 if features.tier is level else 0.0 for level in self.tier_levels[1:]
        )
        return indicators + encode_base(features)

    def linear(self, features: FeatureVector) -> float:
        row = self.encode(features)
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, row, strict=True)
        )

    def score(self, features: FeatureVector) -> float:
        """The raw score: ``sigmoid(w . x + b)``, in ``(0, 1)``.

        Deliberately *not* called a probability. It is monotone in the evidence
        and nothing more; the isotonic step on the calibration split is what
        turns it into a number meaning "right this often".
        """
        return sigmoid(self.linear(features))

    def score_candidate(self, candidate: MatchCandidate) -> float:
        return self.score(candidate.features)

    def coefficient_table(self) -> tuple[tuple[str, float], ...]:
        """``(name, coefficient)`` pairs including the intercept, for the report."""
        pairs = zip(self.feature_names, self.coefficients, strict=True)
        return (("intercept", self.intercept), *pairs)


def solve_symmetric(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Written out rather than imported because the systems here are (p+1) square
    with p around a dozen. Partial pivoting rather than plain elimination: the
    Hessian of a logistic likelihood becomes near-singular as fitted
    probabilities approach 0 or 1, which is precisely the regime a confident
    reconciliation corpus produces, and unpivoted elimination divides by the
    small pivot that regime creates.

    Raises :class:`BlenderError` on a singular system rather than returning
    coefficients that are silently wrong.
    """
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise BlenderError(
            f"cannot solve a {len(matrix)}-row system against {size} targets"
        )

    augmented = [[*row, target] for row, target in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            raise BlenderError(
                f"singular system at column {column}: the design matrix has a "
                "degenerate direction the ridge did not remove"
            )
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * size
    for row in reversed(range(size)):
        total = augmented[row][size] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, size)
        )
        solution[row] = total / augmented[row][row]
    return solution


def _log_likelihood(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    weights: Sequence[float],
    intercept: float,
) -> float:
    """Unpenalised log-likelihood of the fitted model on its own training rows.

    Written in the numerically stable form: ``log sigmoid(z)`` is
    ``-log1p(exp(-|z|))`` plus ``-|z|`` when the sign of ``z`` disagrees with
    the label, which never evaluates ``exp`` of a large positive number.
    """
    total = 0.0
    for row, target in zip(rows, targets, strict=True):
        z = intercept + sum(w * x for w, x in zip(weights, row, strict=True))
        agrees = (z >= 0.0) == (target > 0.5)
        total += -math.log1p(math.exp(-abs(z))) + (0.0 if agrees else -abs(z))
    return total


def fit_logistic(
    features: Sequence[FeatureVector],
    labels: Sequence[bool],
    *,
    l2: float = DEFAULT_L2,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> LogisticBlender:
    """Fit a ridge-penalised logistic regression by IRLS (Newton-Raphson).

    Deterministic: no random initialisation, no shuffling, no early stop on a
    clock. The same rows in the same order always produce the same
    coefficients, which is what lets a fitted bundle be hashed and a rerun be
    proved rather than assumed.

    The penalty does **not** apply to the intercept. Shrinking an intercept
    towards zero shrinks the prior towards a 50% base rate, which is a claim
    about the corpus nobody intended to make.
    """
    if len(features) != len(labels):
        raise BlenderError(f"{len(features)} feature rows against {len(labels)} labels")
    if not features:
        raise BlenderError("cannot fit a blender on an empty training set")
    if l2 < 0.0:
        raise BlenderError(f"l2 must be non-negative, got {l2}")

    tier_levels = tuple(sorted({vector.tier for vector in features}))
    template = LogisticBlender(
        tier_levels=tier_levels,
        coefficients=(0.0,) * len(feature_names(tier_levels)),
        intercept=0.0,
        l2=l2,
        iterations=0,
        converged=False,
        sample_count=len(features),
        positive_count=sum(1 for label in labels if label),
        log_likelihood=0.0,
    )
    rows = [template.encode(vector) for vector in features]
    targets = [1.0 if label else 0.0 for label in labels]
    width = len(rows[0])

    weights = [0.0] * width
    intercept = 0.0
    converged = False
    iterations = 0

    for step_number in range(1, max_iterations + 1):
        iterations = step_number
        # Gradient and Hessian of the penalised negative log-likelihood, with
        # the intercept carried as an implicit leading column of ones.
        gradient = [0.0] * (width + 1)
        hessian = [[0.0] * (width + 1) for _ in range(width + 1)]
        for row, target in zip(rows, targets, strict=True):
            z = intercept + sum(w * x for w, x in zip(weights, row, strict=True))
            probability = sigmoid(z)
            residual = target - probability
            variance = probability * (1.0 - probability)
            full = (1.0, *row)
            for i, xi in enumerate(full):
                if xi == 0.0:
                    continue
                gradient[i] += residual * xi
                for j, xj in enumerate(full):
                    if xj != 0.0:
                        hessian[i][j] += variance * xi * xj
        for index in range(1, width + 1):
            gradient[index] -= 2.0 * l2 * weights[index - 1]
            hessian[index][index] += 2.0 * l2

        try:
            step = solve_symmetric(hessian, gradient)
        except BlenderError:
            # The Hessian degenerated. On this corpus that means one thing:
            # the rows are perfectly separable (or single-class), so every
            # fitted probability has run to 0 or 1, the IRLS weights
            # ``p(1 - p)`` have vanished and the intercept -- which the ridge
            # deliberately does not penalise -- is diverging.
            #
            # Stopping is the correct answer, not a rescue. The last iterate
            # already classifies every training row correctly; continuing would
            # only push the coefficients further out in a direction the data
            # cannot bound. ``converged`` stays False and ``single_class`` says
            # which case it was, so a report can state that the fit ran out of
            # contrast rather than presenting an arbitrary large coefficient as
            # a measurement.
            break
        intercept += step[0]
        weights = [w + delta for w, delta in zip(weights, step[1:], strict=True)]
        if max(abs(delta) for delta in step) < tolerance:
            converged = True
            break

    return template.model_copy(
        update={
            "coefficients": tuple(weights),
            "intercept": intercept,
            "iterations": iterations,
            "converged": converged,
            "log_likelihood": _log_likelihood(rows, targets, weights, intercept),
        }
    )
