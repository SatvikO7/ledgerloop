"""Property tests for the blender and the calibrator (PLAN.md §13).

The unit tests check numbers a reader can verify by hand. These check the
statements that have to hold for *every* input -- and for this step that matters
more than usual, because a calibrator is exactly the kind of component that can
be subtly wrong on inputs nobody thought to write down while looking perfectly
reasonable on the ones they did.

Four properties carry the weight:

* **Isotonic output is monotone.** If it were not, the fit would not be an
  isotonic regression at all, and every probability downstream would be a
  number with no defensible meaning.
* **A selected threshold achieves what it claims.** ``tau_high`` is the single
  most consequential number in the project.
* **ECE and Brier stay inside [0, 1].** They are typed as probabilities on
  :class:`~ledgerloop.models.metrics.CalibrationMetrics`, so a value outside the
  range is not a bad metric -- it is an unconstructable model.
* **Every design-matrix column stays inside its cap**, for any feature vector
  the models permit. One unbounded entry destroys a whole fit, not one row.
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ledgerloop.matching.blender import (
    encode_base,
    fit_logistic,
    sigmoid,
    solve_symmetric,
)
from ledgerloop.matching.calibration import (
    brier_score,
    fit_isotonic,
    reliability,
    select_tau_high,
)
from ledgerloop.models.candidates import FeatureVector
from ledgerloop.models.enums import Tier

RESIDUAL_TIERS = st.sampled_from(
    [Tier.T2_AGGREGATION, Tier.T3_FUZZY, Tier.T4_GRAPH, Tier.T5_LLM]
)
scores = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False)
probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
labels = st.booleans()


def scored_sets(min_size: int = 1, max_size: int = 40):
    return st.lists(st.tuples(scores, labels), min_size=min_size, max_size=max_size)


def probability_sets(min_size: int = 1, max_size: int = 40):
    return st.lists(
        st.tuples(probabilities, labels), min_size=min_size, max_size=max_size
    )


feature_vectors = st.builds(
    FeatureVector,
    tier=RESIDUAL_TIERS,
    amount_delta_minor=st.integers(min_value=-(10**12), max_value=10**12),
    tolerance_band_minor=st.integers(min_value=0, max_value=10**9),
    amount_delta_ratio=st.one_of(
        st.floats(min_value=0.0, max_value=10**6, allow_nan=False),
        st.just(math.inf),
    ),
    date_delta_days=st.integers(min_value=-4000, max_value=4000),
    lexical_score=probabilities,
    semantic_score=probabilities,
    graph_support=probabilities,
    subset_size=st.integers(min_value=0, max_value=5000),
    llm_confidence=st.one_of(st.none(), probabilities),
)


class TestTheSigmoid:
    @given(scores)
    def test_it_stays_inside_zero_and_one(self, z):
        assert 0.0 <= sigmoid(z) <= 1.0

    @given(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False))
    def test_it_is_strictly_interior_before_float64_saturates(self, z):
        """Past |z| ~ 37 the true value is within one ulp of a bound and rounds to it.

        Worth stating rather than papering over: a raw score of exactly 1.0 is a
        *representable* answer, not a bug, and the calibrated probability field
        accepts it. The strict interior only holds where float64 can express it.
        """
        assert 0.0 < sigmoid(z) < 1.0

    @given(scores, st.floats(min_value=0.0, max_value=50.0, allow_nan=False))
    def test_it_never_decreases(self, z, step):
        assert sigmoid(z + step) >= sigmoid(z)

    @given(st.floats(min_value=-1e300, max_value=1e300, allow_nan=False))
    def test_no_magnitude_overflows(self, z):
        assert 0.0 <= sigmoid(z) <= 1.0


class TestTheDesignRow:
    @given(feature_vectors)
    def test_every_column_is_finite_and_capped(self, vector):
        """One infinite entry destroys the whole fit, not one row."""
        row = encode_base(vector)
        assert all(math.isfinite(value) for value in row)
        assert all(0.0 <= value <= 5.0 for value in row)

    @given(feature_vectors)
    def test_the_row_width_never_changes(self, vector):
        """A design matrix with ragged rows is an unfittable one."""
        assert len(encode_base(vector)) == 9

    @given(feature_vectors)
    def test_the_llm_indicator_agrees_with_the_llm_column(self, vector):
        row = encode_base(vector)
        assert (row[8] == 1.0) is (vector.llm_confidence is not None)


class TestIsotonicRegression:
    @given(scored_sets())
    def test_the_fitted_values_never_decrease(self, samples):
        fitted = fit_isotonic([s for s, _ in samples], [label for _, label in samples])
        assert list(fitted.values) == sorted(fitted.values)

    @given(scored_sets())
    def test_every_fitted_value_is_a_probability(self, samples):
        fitted = fit_isotonic([s for s, _ in samples], [label for _, label in samples])
        assert all(0.0 <= value <= 1.0 for value in fitted.values)

    @given(scored_sets(), scores, st.floats(min_value=0.0, max_value=20.0,
                                            allow_nan=False))
    def test_prediction_never_decreases_with_the_score(self, samples, point, step):
        fitted = fit_isotonic([s for s, _ in samples], [label for _, label in samples])
        assert fitted.predict(point + step) >= fitted.predict(point)

    @given(scored_sets())
    def test_the_fit_preserves_the_base_rate(self, samples):
        """PAVA is a least-squares projection: pooled means average to the mean."""
        fitted = fit_isotonic([s for s, _ in samples], [label for _, label in samples])
        assert fitted.positive_count == sum(1 for _, label in samples if label)
        assert fitted.sample_count == len(samples)

    @given(scored_sets())
    def test_the_block_starts_are_scores_that_were_observed(self, samples):
        """A block start the blender never produced is a boundary nobody measured."""
        observed = {s for s, _ in samples}
        fitted = fit_isotonic([s for s, _ in samples], [label for _, label in samples])
        assert set(fitted.thresholds) <= observed


class TestThresholdSelection:
    @given(probability_sets(), st.floats(min_value=0.0, max_value=1.0,
                                         allow_nan=False))
    @settings(max_examples=200)
    def test_an_attained_threshold_really_achieves_the_target(self, samples, target):
        selection = select_tau_high(
            [p for p, _ in samples],
            [label for _, label in samples],
            target_precision=target,
        )
        if not selection.attained:
            return
        kept = [label for p, label in samples if p >= selection.tau_high]
        assert kept
        assert sum(1 for label in kept if label) / len(kept) >= target

    @given(probability_sets())
    def test_the_counts_agree_with_the_threshold(self, samples):
        selection = select_tau_high(
            [p for p, _ in samples],
            [label for _, label in samples],
            target_precision=0.99,
        )
        kept = [label for p, label in samples if p >= selection.tau_high]
        assert selection.auto_matched == len(kept)
        assert selection.true_positives == sum(1 for label in kept if label)
        assert selection.true_positives + selection.false_positives == (
            selection.auto_matched
        )

    @given(probability_sets())
    def test_coverage_and_recall_are_proportions(self, samples):
        selection = select_tau_high(
            [p for p, _ in samples],
            [label for _, label in samples],
            target_precision=0.5,
        )
        assert 0.0 <= selection.coverage <= 1.0
        assert 0.0 <= selection.recall <= 1.0

    @given(probability_sets())
    def test_a_lower_target_is_never_harder_to_meet(self, samples):
        strict = select_tau_high(
            [p for p, _ in samples],
            [label for _, label in samples],
            target_precision=1.0,
        )
        lenient = select_tau_high(
            [p for p, _ in samples],
            [label for _, label in samples],
            target_precision=0.5,
        )
        if strict.attained:
            assert lenient.attained
            assert lenient.tau_high <= strict.tau_high


class TestTheReportedNumbers:
    @given(probability_sets())
    def test_ece_and_brier_stay_inside_the_unit_interval(self, samples):
        """Both are typed as probabilities; outside the range is unconstructable."""
        values = [p for p, _ in samples]
        marks = [label for _, label in samples]
        diagram = reliability(values, marks)
        assert 0.0 <= diagram.ece <= 1.0
        assert 0.0 <= diagram.brier <= 1.0
        assert diagram.brier == brier_score(values, marks)

    @given(probability_sets())
    def test_the_bins_partition_the_sample(self, samples):
        diagram = reliability([p for p, _ in samples], [x for _, x in samples])
        assert sum(item.count for item in diagram.bins) == len(samples)
        assert diagram.populated_bins <= len(diagram.bins)

    @given(probability_sets())
    def test_the_metrics_contract_is_always_constructable(self, samples):
        diagram = reliability([p for p, _ in samples], [x for _, x in samples])
        metrics = diagram.metrics()
        assert metrics.sample_count == len(samples)
        assert metrics.populated_bins == diagram.populated_bins

    @given(probability_sets(min_size=2))
    def test_perfect_predictions_score_zero_on_both(self, samples):
        marks = [label for _, label in samples]
        perfect = [1.0 if label else 0.0 for label in marks]
        diagram = reliability(perfect, marks)
        assert diagram.ece == 0.0
        assert diagram.brier == 0.0


class TestFitting:
    @given(
        st.lists(
            st.tuples(RESIDUAL_TIERS, probabilities, labels),
            min_size=2,
            max_size=30,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_a_fit_always_produces_finite_coefficients(self, samples):
        """Separable data, single-class data, collinear data -- all bounded."""
        vectors = [
            FeatureVector(tier=tier, lexical_score=score) for tier, score, _ in samples
        ]
        model = fit_logistic(vectors, [label for _, _, label in samples])
        assert math.isfinite(model.intercept)
        assert all(math.isfinite(value) for value in model.coefficients)
        assert 0.0 < model.score(vectors[0]) < 1.0

    @given(
        st.lists(
            st.tuples(RESIDUAL_TIERS, probabilities, labels),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_the_fitted_tiers_are_exactly_the_tiers_present(self, samples):
        vectors = [
            FeatureVector(tier=tier, lexical_score=score) for tier, score, _ in samples
        ]
        model = fit_logistic(vectors, [label for _, _, label in samples])
        assert set(model.tier_levels) == {tier for tier, _, _ in samples}
        assert list(model.tier_levels) == sorted(model.tier_levels)
        for tier in Tier:
            assert model.covers(tier) is (tier in model.tier_levels)


class TestTheLinearSolver:
    @given(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=3,
            max_size=3,
        ),
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=2,
            max_size=2,
        ),
    )
    def test_a_solution_satisfies_the_system_it_came_from(self, diagonal, target):
        """Diagonally dominant by construction, so a solution must exist."""
        matrix = [
            [10.0 + abs(diagonal[0]), diagonal[1]],
            [diagonal[1], 10.0 + abs(diagonal[2])],
        ]
        solution = solve_symmetric(matrix, target)
        for row, value in zip(matrix, target, strict=True):
            residual = sum(a * x for a, x in zip(row, solution, strict=True)) - value
            assume(math.isfinite(residual))
            assert abs(residual) < 1e-6
