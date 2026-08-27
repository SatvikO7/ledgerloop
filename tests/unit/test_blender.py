"""The blender -- design matrix, optimiser, and the tier encoding.

Three things are being checked, and they fail in different ways:

* **The encoding**, because a design matrix that quietly mis-scales one column
  produces a model that is wrong without being broken.
* **The optimiser**, against problems whose answers are known independently of
  this code -- a symmetric solve with a hand-checkable solution, a separable
  fit whose sign is not in doubt.
* **The tier encoding is categorical**, which is a settled decision
  (`ARCHITECTURE.md` §6, 3) rather than an implementation detail: an ordinal
  tier would let a near-perfectly predictive column dominate every coefficient.
"""

from __future__ import annotations

import math

import pytest

from ledgerloop.matching.blender import (
    BASE_FEATURE_NAMES,
    BlenderError,
    LogisticBlender,
    encode_base,
    feature_names,
    fit_logistic,
    sigmoid,
    solve_symmetric,
)
from ledgerloop.models.candidates import FeatureVector
from ledgerloop.models.enums import Tier


def features(tier: Tier = Tier.T2_AGGREGATION, **kwargs: object) -> FeatureVector:
    return FeatureVector(tier=tier, **kwargs)  # type: ignore[arg-type]


class TestTheSigmoid:
    def test_it_is_a_half_at_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_it_is_symmetric(self):
        assert sigmoid(2.5) + sigmoid(-2.5) == pytest.approx(1.0)

    @pytest.mark.parametrize("z", [-1000.0, -800.0, 800.0, 1000.0])
    def test_neither_tail_overflows(self, z):
        """``math.exp(800)`` raises. A confident row must not take the run down."""
        value = sigmoid(z)
        assert 0.0 <= value <= 1.0

    def test_it_is_monotone(self):
        values = [sigmoid(z) for z in (-5.0, -1.0, 0.0, 1.0, 5.0)]
        assert values == sorted(values)


class TestTheLinearSolver:
    def test_it_solves_a_system_checkable_by_hand(self):
        # 2x + y = 5, x + 3y = 10  ->  x = 1, y = 3
        solution = solve_symmetric([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
        assert solution[0] == pytest.approx(1.0)
        assert solution[1] == pytest.approx(3.0)

    def test_it_pivots_past_a_zero_leading_entry(self):
        """Unpivoted elimination divides by zero here; partial pivoting does not."""
        solution = solve_symmetric([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])
        assert solution == pytest.approx([3.0, 2.0])

    def test_a_singular_system_raises_rather_than_returning_nonsense(self):
        with pytest.raises(BlenderError, match="singular"):
            solve_symmetric([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_a_mismatched_shape_raises(self):
        with pytest.raises(BlenderError, match="system"):
            solve_symmetric([[1.0, 2.0]], [1.0, 2.0])


class TestTheDesignRow:
    def test_the_base_columns_are_in_the_declared_order(self):
        row = encode_base(
            features(
                amount_delta_minor=50,
                tolerance_band_minor=100,
                amount_delta_ratio=0.25,
                date_delta_days=-3,
                lexical_score=0.9,
                graph_support=0.5,
                subset_size=20,
            )
        )
        named = dict(zip(BASE_FEATURE_NAMES, row, strict=True))
        assert named["amount_delta_ratio"] == pytest.approx(0.25)
        assert named["amount_delta_bands"] == pytest.approx(0.5)
        assert named["date_delta_days"] == pytest.approx(3 / 30)
        assert named["lexical_score"] == pytest.approx(0.9)
        assert named["graph_support"] == pytest.approx(0.5)
        assert named["subset_size"] == pytest.approx(0.5)

    def test_an_infinite_ratio_is_capped_rather_than_poisoning_the_fit(self):
        """``delta_ratio`` returns inf on a zero base. One inf destroys every row."""
        row = encode_base(features(amount_delta_ratio=math.inf))
        assert row[0] == 1.0

    def test_a_zero_tolerance_band_does_not_divide_by_zero(self):
        """T0's band is zero by construction, and T4 sets no band at all."""
        row = encode_base(features(amount_delta_minor=7, tolerance_band_minor=0))
        assert row[1] == pytest.approx(5.0)  # capped

    def test_the_date_column_is_unsigned(self):
        """Three days early and three days late are equally far from the settlement."""
        early = encode_base(features(date_delta_days=-3))
        late = encode_base(features(date_delta_days=3))
        assert early[2] == late[2]

    def test_the_llm_columns_are_absent_rather_than_zero_when_there_is_no_llm(self):
        without = dict(zip(BASE_FEATURE_NAMES, encode_base(features()), strict=True))
        assert without["llm_confidence"] == 0.0
        assert without["llm_confidence_present"] == 0.0

        with_llm = dict(
            zip(
                BASE_FEATURE_NAMES,
                encode_base(features(llm_confidence=0.4)),
                strict=True,
            )
        )
        assert with_llm["llm_confidence"] == pytest.approx(0.4)
        assert with_llm["llm_confidence_present"] == 1.0

    def test_a_nan_cannot_reach_the_design_matrix_at_all(self):
        """The bound on the field is the guard, so ``encode_base`` needs none."""
        with pytest.raises(ValueError, match="greater than or equal"):
            features(amount_delta_ratio=math.nan)

    def test_every_column_stays_inside_its_cap(self):
        row = encode_base(
            features(
                amount_delta_minor=10**9,
                tolerance_band_minor=1,
                amount_delta_ratio=1000.0,
                date_delta_days=9999,
                subset_size=10_000,
            )
        )
        assert max(row) <= 5.0
        assert min(row) >= 0.0


class TestTierIsCategorical:
    """`ARCHITECTURE.md` §6, 3: one-hot, never the ordinal."""

    @pytest.fixture
    def fitted(self):
        rows = [features(Tier.T2_AGGREGATION)] * 4 + [features(Tier.T3_FUZZY)] * 4
        labels = [True, True, False, False, True, False, False, False]
        return fit_logistic(rows, labels)

    def test_the_lowest_fitted_tier_is_the_reference_level(self, fitted):
        assert fitted.tier_levels[0] is Tier.T2_AGGREGATION
        assert "tier=T2_AGGREGATION" not in fitted.feature_names
        assert "tier=T3_FUZZY" in fitted.feature_names

    def test_the_reference_tier_encodes_as_all_zero_indicators(self, fitted):
        row = fitted.encode(features(Tier.T2_AGGREGATION))
        assert row[0] == 0.0

    def test_a_non_reference_tier_encodes_as_an_indicator_not_its_number(self, fitted):
        """T3 is ``Tier == 3``. The design row must carry 1.0, never 3.0."""
        row = fitted.encode(features(Tier.T3_FUZZY))
        assert row[0] == 1.0

    def test_names_pair_one_to_one_with_coefficients(self, fitted):
        assert len(fitted.feature_names) == len(fitted.coefficients)

    def test_the_intercept_leads_the_printed_table(self, fitted):
        table = fitted.coefficient_table()
        assert table[0] == ("intercept", fitted.intercept)
        assert len(table) == len(fitted.coefficients) + 1


class TestTiersTheModelNeverSaw:
    """A tier absent from the fit must not be scored as the reference level."""

    @pytest.fixture
    def t2_only(self):
        rows = [features(Tier.T2_AGGREGATION) for _ in range(4)]
        return fit_logistic(rows, [True, False, True, False])

    def test_it_reports_which_tiers_it_covers(self, t2_only):
        assert t2_only.covers(Tier.T2_AGGREGATION)
        assert not t2_only.covers(Tier.T3_FUZZY)
        assert not t2_only.covers(Tier.T5_LLM)

    def test_encoding_an_uncovered_tier_raises_rather_than_guessing(self, t2_only):
        with pytest.raises(BlenderError, match="cannot encode"):
            t2_only.encode(features(Tier.T5_LLM))

    def test_a_single_tier_model_has_no_indicator_columns_at_all(self, t2_only):
        assert t2_only.feature_names == BASE_FEATURE_NAMES


class TestFitting:
    def test_it_learns_the_direction_the_data_states(self):
        """A clean lexical separation must produce a positive lexical coefficient."""
        good = [features(lexical_score=1.0) for _ in range(20)]
        bad = [features(lexical_score=0.0) for _ in range(20)]
        model = fit_logistic(good + bad, [True] * 20 + [False] * 20)
        named = dict(zip(model.feature_names, model.coefficients, strict=True))
        assert named["lexical_score"] > 0.0
        assert model.score(features(lexical_score=1.0)) > model.score(
            features(lexical_score=0.0)
        )

    def test_it_is_deterministic(self):
        rows = [features(lexical_score=score / 10) for score in range(10)]
        labels = [score >= 5 for score in range(10)]
        first = fit_logistic(rows, labels)
        second = fit_logistic(rows, labels)
        assert first.coefficients == second.coefficients
        assert first.intercept == second.intercept

    def test_a_stronger_ridge_shrinks_the_coefficients(self):
        rows = [features(lexical_score=1.0)] * 10 + [features(lexical_score=0.0)] * 10
        labels = [True] * 10 + [False] * 10
        light = fit_logistic(rows, labels, l2=0.1)
        heavy = fit_logistic(rows, labels, l2=50.0)
        assert abs(heavy.coefficients[3]) < abs(light.coefficients[3])

    def test_the_ridge_does_not_shrink_the_intercept(self):
        """Shrinking an intercept shrinks the prior towards a 50% base rate."""
        rows = [features(lexical_score=0.5) for _ in range(30)]
        labels = [True] * 27 + [False] * 3
        model = fit_logistic(rows, labels, l2=1000.0)
        assert model.score(features(lexical_score=0.5)) > 0.7

    def test_a_single_class_fit_says_so_rather_than_looking_confident(self):
        rows = [features(lexical_score=0.9) for _ in range(10)]
        model = fit_logistic(rows, [True] * 10)
        assert model.single_class
        assert model.positive_count == 10
        assert model.negative_count == 0

    def test_a_single_class_fit_stops_instead_of_diverging(self):
        """The Hessian degenerates once every fitted probability has run to 1."""
        rows = [features(lexical_score=0.9) for _ in range(10)]
        model = fit_logistic(rows, [True] * 10)
        assert not model.converged
        assert math.isfinite(model.intercept)
        assert all(math.isfinite(value) for value in model.coefficients)

    def test_it_reports_how_many_iterations_it_took(self):
        rows = [features(lexical_score=0.2), features(lexical_score=0.8)]
        model = fit_logistic(rows, [False, True])
        assert model.iterations >= 1
        assert model.converged

    def test_the_log_likelihood_is_never_positive(self):
        rows = [features(lexical_score=0.2), features(lexical_score=0.8)]
        model = fit_logistic(rows, [False, True])
        assert model.log_likelihood <= 0.0

    @pytest.mark.parametrize(
        ("rows", "labels", "message"),
        [
            ([], [], "empty"),
            ([features()], [], "against"),
        ],
    )
    def test_it_refuses_an_input_it_cannot_fit(self, rows, labels, message):
        with pytest.raises(BlenderError, match=message):
            fit_logistic(rows, labels)

    def test_a_negative_ridge_is_refused(self):
        with pytest.raises(BlenderError, match="non-negative"):
            fit_logistic([features()], [True], l2=-1.0)

    def test_a_score_is_a_probability_shaped_number(self):
        model = fit_logistic(
            [features(lexical_score=0.0), features(lexical_score=1.0)], [False, True]
        )
        for score in (0.0, 0.5, 1.0):
            value = model.score(features(lexical_score=score))
            assert 0.0 < value < 1.0


class TestTheFittedModelAsAContract:
    @pytest.fixture
    def model(self):
        rows = [features(lexical_score=0.1), features(lexical_score=0.9)]
        return fit_logistic(rows, [False, True])

    def test_it_round_trips_through_json(self, model):
        restored = LogisticBlender.model_validate_json(model.model_dump_json())
        assert restored == model
        assert restored.score(features(lexical_score=0.9)) == model.score(
            features(lexical_score=0.9)
        )

    def test_it_is_frozen(self, model):
        with pytest.raises(Exception):  # noqa: B017 - pydantic's frozen error type
            model.intercept = 1.0

    def test_a_coefficient_count_that_disagrees_with_the_names_is_refused(self):
        with pytest.raises(ValueError, match="coefficients against"):
            LogisticBlender(
                tier_levels=(Tier.T2_AGGREGATION,),
                coefficients=(0.0,),
                intercept=0.0,
                l2=1.0,
                iterations=1,
                converged=True,
                sample_count=1,
                positive_count=1,
                log_likelihood=0.0,
            )

    def test_a_model_over_no_tiers_is_refused(self):
        with pytest.raises(ValueError, match="at least one tier"):
            LogisticBlender(
                tier_levels=(),
                coefficients=(),
                intercept=0.0,
                l2=1.0,
                iterations=1,
                converged=True,
                sample_count=0,
                positive_count=0,
                log_likelihood=0.0,
            )

    def test_more_positives_than_samples_is_refused(self):
        with pytest.raises(ValueError, match="positive_count"):
            LogisticBlender(
                tier_levels=(Tier.T2_AGGREGATION,),
                coefficients=(0.0,) * len(feature_names((Tier.T2_AGGREGATION,))),
                intercept=0.0,
                l2=1.0,
                iterations=1,
                converged=True,
                sample_count=1,
                positive_count=2,
                log_likelihood=0.0,
            )
