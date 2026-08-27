"""Isotonic calibration, threshold selection, and the reliability numbers.

The tests are organised the way the fit is: isotonic, then the threshold that
cuts its output, then the diagram that says whether either worked, then the
bundle that carries all three, then the rules about who may be scored at all.

Two of these are guarantees rather than behaviours, and they get their own
classes:

* A selected ``tau_high`` **must** achieve the target precision on the data it
  was selected from, or report that it could not. A selector that silently
  returned a threshold missing its target would make every headline in the
  project a claim about nothing.
* The three splits **must** stay separate. That one is enforced in the type, so
  it is tested by trying to build a bundle that breaks it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgerloop.config import DecisionThresholds, RunConfig
from ledgerloop.matching.calibration import (
    BlendOutcome,
    CalibrationBundle,
    CalibrationProvenance,
    IsotonicCalibrator,
    apply_bundle,
    brier_score,
    configure_for,
    expected_calibration_error,
    fit_bundle,
    fit_isotonic,
    reliability,
    residual_rows,
    rows_by_tier,
    select_tau_high,
    thresholds_from,
)
from ledgerloop.models.candidates import FeatureVector, MatchCandidate
from ledgerloop.models.enums import LinkType, SplitName, Tier
from ledgerloop.models.refs import bank_ref, payment_ref


def features(tier: Tier = Tier.T2_AGGREGATION, **kwargs: object) -> FeatureVector:
    return FeatureVector(tier=tier, **kwargs)  # type: ignore[arg-type]


def candidate(
    identifier: str = "c1",
    *,
    tier: Tier = Tier.T2_AGGREGATION,
    verified: bool = True,
    probability: float | None = 0.5,
    lexical: float = 0.0,
) -> MatchCandidate:
    return MatchCandidate(
        candidate_id=identifier,
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref(f"PAY-{identifier}"),
        target_ref=bank_ref(f"BNK-{identifier}"),
        tier=tier,
        features=features(tier, lexical_score=lexical),
        calibrated_p=probability,
        arithmetic_verified=verified,
    )


def provenance(**kwargs: object) -> CalibrationProvenance:
    defaults: dict[str, object] = {
        "train_split": SplitName.TRAIN,
        "train_seeds": (42,),
        "calibration_split": SplitName.CALIBRATION,
        "calibration_seeds": (43,),
        "generator_version": "0.2.0",
        "top_k": 3,
        "train_rows": 0,
        "train_positives": 0,
        "calibration_rows": 0,
        "calibration_positives": 0,
    }
    return CalibrationProvenance(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestIsotonicFitting:
    def test_it_pools_a_violation_into_its_neighbour(self):
        """The textbook PAVA case. 1, 0, 1 at rising scores fits to 0.5, 0.5, 1."""
        fitted = fit_isotonic([0.1, 0.2, 0.3], [True, False, True])
        assert fitted.block_count == 2
        assert fitted.values == pytest.approx((0.5, 1.0))
        assert fitted.predict(0.1) == fitted.predict(0.2) == pytest.approx(0.5)

    def test_an_already_monotone_sequence_is_left_alone(self):
        fitted = fit_isotonic([0.1, 0.2, 0.3], [False, False, True])
        assert fitted.values == pytest.approx((0.0, 0.0, 1.0))

    def test_the_output_is_always_non_decreasing(self):
        fitted = fit_isotonic(
            [0.9, 0.1, 0.5, 0.3, 0.7], [True, False, True, False, False]
        )
        assert list(fitted.values) == sorted(fitted.values)

    def test_ties_in_the_raw_score_are_pooled_into_one_block(self):
        """The blender expressed no ordering, so the calibrator invents none."""
        fitted = fit_isotonic([0.5, 0.5, 0.5, 0.5], [True, True, True, False])
        assert fitted.block_count == 1
        assert fitted.predict(0.5) == pytest.approx(0.75)

    def test_it_records_the_sample_it_was_fitted_on(self):
        fitted = fit_isotonic([0.1, 0.9], [False, True])
        assert fitted.sample_count == 2
        assert fitted.positive_count == 1

    def test_input_order_does_not_matter(self):
        forwards = fit_isotonic([0.1, 0.5, 0.9], [False, True, True])
        backwards = fit_isotonic([0.9, 0.5, 0.1], [True, True, False])
        assert forwards.values == backwards.values

    @pytest.mark.parametrize(
        ("scores", "labels", "message"),
        [([], [], "empty"), ([0.1], [], "against")],
    )
    def test_it_refuses_what_it_cannot_fit(self, scores, labels, message):
        with pytest.raises(ValueError, match=message):
            fit_isotonic(scores, labels)


class TestIsotonicPrediction:
    @pytest.fixture
    def fitted(self):
        return fit_isotonic([0.2, 0.4, 0.6, 0.8], [False, False, True, True])

    def test_it_is_a_step_function_not_an_interpolation(self, fitted):
        """Interpolating between blocks invents probabilities the fit never made."""
        assert fitted.predict(0.5) == fitted.predict(0.4)
        assert fitted.predict(0.5) != fitted.predict(0.6)

    def test_a_score_below_the_first_block_takes_the_first_block(self, fitted):
        assert fitted.predict(-5.0) == fitted.values[0]

    def test_a_score_above_the_last_block_takes_the_last_block(self, fitted):
        """No extrapolation: the calibrator has no evidence out there."""
        assert fitted.predict(99.0) == fitted.values[-1]

    def test_it_never_decreases(self, fitted):
        seen = [fitted.predict(x / 20) for x in range(21)]
        assert seen == sorted(seen)

    def test_non_monotone_values_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            IsotonicCalibrator(
                thresholds=(0.1, 0.2), values=(0.9, 0.1), sample_count=2,
                positive_count=1,
            )

    def test_unordered_block_starts_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            IsotonicCalibrator(
                thresholds=(0.2, 0.1), values=(0.1, 0.9), sample_count=2,
                positive_count=1,
            )

    def test_an_empty_calibrator_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="at least one block"):
            IsotonicCalibrator(
                thresholds=(), values=(), sample_count=0, positive_count=0
            )

    def test_mismatched_block_lengths_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="block starts against"):
            IsotonicCalibrator(
                thresholds=(0.1, 0.2), values=(0.5,), sample_count=1, positive_count=1
            )

    def test_more_positives_than_samples_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="positive_count"):
            IsotonicCalibrator(
                thresholds=(0.1,), values=(0.5,), sample_count=1, positive_count=2
            )

    def test_a_value_outside_zero_to_one_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="probabilities"):
            IsotonicCalibrator(
                thresholds=(0.1,), values=(1.5,), sample_count=1, positive_count=1
            )


class TestThresholdSelection:
    def test_it_takes_the_lowest_threshold_that_meets_the_target(self):
        """Lowest, because precision is already guaranteed and coverage is free."""
        probabilities = [0.99, 0.98, 0.97, 0.10]
        labels = [True, True, True, False]
        selection = select_tau_high(probabilities, labels, target_precision=0.99)
        assert selection.tau_high == pytest.approx(0.97)
        assert selection.auto_matched == 3
        assert selection.attained

    def test_the_chosen_threshold_actually_achieves_the_target(self):
        probabilities = [0.95, 0.94, 0.93, 0.92]
        labels = [True, True, False, True]
        selection = select_tau_high(probabilities, labels, target_precision=0.99)
        kept = [
            label
            for probability, label in zip(probabilities, labels, strict=True)
            if probability >= selection.tau_high
        ]
        if selection.attained:
            assert sum(kept) / len(kept) >= 0.99

    def test_it_stops_above_a_wrong_prediction(self):
        selection = select_tau_high(
            [0.9, 0.8, 0.7], [True, False, True], target_precision=0.99
        )
        assert selection.tau_high == pytest.approx(0.9)
        assert selection.false_positives == 0

    def test_an_unattainable_target_refuses_rather_than_lowering_the_bar(self):
        selection = select_tau_high([0.9, 0.8], [False, False], target_precision=0.99)
        assert not selection.attained
        assert selection.tau_high == 1.0
        assert selection.auto_matched == 0

    def test_a_tie_is_resolved_at_the_group_boundary(self):
        """Two candidates the calibrator gave one probability cannot be split."""
        selection = select_tau_high(
            [0.9, 0.9, 0.5], [True, False, True], target_precision=0.5
        )
        assert selection.tau_high == pytest.approx(0.5)
        assert selection.auto_matched == 3

    def test_it_reports_the_interval_alongside_the_point_estimate(self):
        selection = select_tau_high([1.0] * 20, [True] * 20, target_precision=0.99)
        assert selection.achieved_precision == 1.0
        assert selection.precision_ci_low < 1.0
        assert selection.precision_ci_high == 1.0

    def test_it_reports_coverage_and_recall_over_the_calibration_set(self):
        selection = select_tau_high(
            [0.9, 0.9, 0.1, 0.1], [True, True, True, False], target_precision=0.99
        )
        assert selection.candidates_considered == 4
        assert selection.positives_available == 3
        assert selection.coverage == pytest.approx(0.5)
        assert selection.recall == pytest.approx(2 / 3)

    def test_an_empty_calibration_set_yields_no_threshold(self):
        selection = select_tau_high([], [], target_precision=0.99)
        assert not selection.attained
        assert selection.coverage == 0.0
        assert selection.recall == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="against"):
            select_tau_high([0.5], [], target_precision=0.99)


class TestReliability:
    def test_a_perfectly_calibrated_set_has_no_error(self):
        probabilities = [0.5] * 10
        labels = [True] * 5 + [False] * 5
        assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)

    def test_confident_and_wrong_is_the_largest_error(self):
        assert expected_calibration_error([1.0] * 4, [False] * 4) == pytest.approx(1.0)

    def test_a_probability_of_exactly_one_lands_in_the_top_bin(self):
        """Isotonic produces 1.0 routinely; it must not fall out of the table."""
        diagram = reliability([1.0], [True], bin_count=10)
        assert diagram.bins[-1].count == 1

    def test_empty_bins_are_reported_as_empty_rather_than_as_zero_accuracy(self):
        diagram = reliability([0.95, 0.96], [True, True], bin_count=10)
        assert diagram.populated_bins == 1
        assert all(item.count == 0 for item in diagram.bins[:-1])

    def test_the_gap_says_which_way_the_miscalibration_runs(self):
        overconfident = reliability([0.9] * 10, [True] * 5 + [False] * 5)
        populated = [item for item in overconfident.bins if item.count]
        assert populated[0].gap == pytest.approx(0.4)

    def test_the_brier_score_is_the_mean_squared_error(self):
        assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
        assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)

    def test_an_empty_set_scores_zero_rather_than_perfect(self):
        assert brier_score([], []) == 0.0

    def test_a_brier_score_over_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="against"):
            brier_score([0.5], [])

    def test_a_diagram_over_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="against"):
            reliability([0.5], [])

    def test_the_metrics_contract_carries_the_bin_counts(self):
        diagram = reliability([0.95, 0.05], [True, False], bin_count=10)
        metrics = diagram.metrics()
        assert metrics.bin_count == 10
        assert metrics.populated_bins == 2
        assert metrics.sample_count == 2
        assert metrics.residual_only

    def test_a_probability_outside_zero_to_one_raises(self):
        with pytest.raises(ValueError, match="not a probability"):
            reliability([1.5], [True])

    def test_a_bin_count_below_one_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            reliability([0.5], [True], bin_count=0)


class TestTheSplitDiscipline:
    """`ARCHITECTURE.md` §6, 1 -- enforced in the type, not left to a convention."""

    def test_overlapping_corpora_are_refused(self):
        """One shared seed out of five is the same leak in a form easy to miss."""
        with pytest.raises(ValueError, match="different data"):
            provenance(
                train_seeds=(42, 43),
                calibration_split=SplitName.TRAIN,
                calibration_seeds=(43,),
            )

    def test_the_same_seed_in_two_different_splits_is_not_an_overlap(self):
        record = provenance(train_seeds=(42,), calibration_seeds=(42,))
        assert record.calibration_seeds == (42,)

    def test_the_same_split_at_different_seeds_is_allowed(self):
        record = provenance(
            calibration_split=SplitName.CALIBRATION, calibration_seeds=(99,)
        )
        assert record.calibration_seeds == (99,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_split": SplitName.TEST},
            {"calibration_split": SplitName.TEST},
        ],
    )
    def test_the_test_split_may_never_be_fitted_on(self, kwargs):
        with pytest.raises(ValueError, match="test split"):
            provenance(**kwargs)

    def test_a_half_with_no_corpus_is_refused(self):
        with pytest.raises(ValueError, match="at least one corpus"):
            provenance(train_seeds=())

    def test_it_names_the_corpora_it_was_fitted_on(self):
        record = provenance(train_seeds=(1, 2))
        assert record.train_corpora == (
            (SplitName.TRAIN, 1),
            (SplitName.TRAIN, 2),
        )


class TestFittingABundle:
    @pytest.fixture
    def bundle(self):
        train = [features(lexical_score=1.0)] * 10 + [features(lexical_score=0.0)] * 10
        train_labels = [True] * 10 + [False] * 10
        calib = [features(lexical_score=1.0)] * 8 + [features(lexical_score=0.0)] * 8
        calib_labels = [True] * 7 + [False] + [False] * 8
        return fit_bundle(
            train,
            train_labels,
            calib,
            calib_labels,
            provenance=provenance(),
            target_precision=0.8,
        )

    def test_the_pieces_travel_together(self, bundle):
        assert bundle.blender.sample_count == 20
        assert bundle.calibrator.sample_count == 16
        assert bundle.thresholds.candidates_considered == 16

    def test_the_probability_is_the_blender_then_the_calibrator(self, bundle):
        row = candidate(lexical=1.0)
        expected = bundle.calibrator.predict(bundle.blender.score_candidate(row))
        assert bundle.probability(row) == pytest.approx(expected)

    def test_it_reports_the_in_sample_reliability_as_a_fit_diagnostic(self, bundle):
        assert bundle.fit_reliability.sample_count == 16
        assert bundle.fit_reliability.residual_only

    def test_calibration_rows_on_an_unfitted_tier_are_counted_not_scored(self):
        train = [features(Tier.T2_AGGREGATION, lexical_score=v) for v in (0.0, 1.0)]
        calib = [
            features(Tier.T2_AGGREGATION, lexical_score=1.0),
            features(Tier.T3_FUZZY, lexical_score=1.0),
        ]
        bundle = fit_bundle(
            train,
            [False, True],
            calib,
            [True, True],
            provenance=provenance(),
            target_precision=0.5,
        )
        assert bundle.provenance.calibration_abstained == 1
        assert bundle.calibrator.sample_count == 1

    def test_two_halves_with_no_tier_in_common_are_refused(self):
        with pytest.raises(ValueError, match="no calibration row"):
            fit_bundle(
                [features(Tier.T2_AGGREGATION)],
                [True],
                [features(Tier.T3_FUZZY)],
                [True],
                provenance=provenance(),
                target_precision=0.99,
            )

    def test_it_round_trips_through_disk(self, bundle, tmp_path):
        path = bundle.save(tmp_path / "nested" / "bundle.json")
        restored = CalibrationBundle.load(path)
        assert restored == bundle

    def test_the_saved_file_is_diffable(self, bundle, tmp_path):
        """A coefficient that moved is the interesting part of a refit."""
        path = bundle.save(tmp_path / "bundle.json")
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") > 10
        assert json.loads(text)["provenance"]["generator_version"] == "0.2.0"

    def test_it_writes_newline_endings_on_every_platform(self, bundle, tmp_path):
        path = bundle.save(tmp_path / "bundle.json")
        assert b"\r\n" not in path.read_bytes()


class TestApplyingABundle:
    @pytest.fixture
    def bundle(self):
        train = [features(lexical_score=1.0)] * 6 + [features(lexical_score=0.0)] * 6
        return fit_bundle(
            train,
            [True] * 6 + [False] * 6,
            [features(lexical_score=1.0), features(lexical_score=0.0)],
            [True, False],
            provenance=provenance(),
            target_precision=0.5,
        )

    @pytest.mark.parametrize("tier", [Tier.T0_EXACT, Tier.T1_TOLERANCE])
    def test_the_deterministic_tiers_are_left_exactly_as_they_were(self, bundle, tier):
        row = candidate(tier=tier, probability=0.5)
        outcome = apply_bundle([row], bundle)
        assert row.calibrated_p == 0.5
        assert row.raw_score is None
        assert outcome.bypassed_deterministic == 1

    def test_a_tier_refusal_is_never_overturned(self, bundle):
        """`arithmetic_verified=False` marks evidence the features cannot carry."""
        row = candidate(verified=False, probability=0.5, lexical=1.0)
        outcome = apply_bundle([row], bundle)
        assert row.calibrated_p == 0.5
        assert outcome.refusals_kept == 1

    def test_a_verified_residual_candidate_is_scored_and_calibrated(self, bundle):
        row = candidate(lexical=1.0)
        outcome = apply_bundle([row], bundle)
        assert row.raw_score is not None
        assert row.calibrated_p == pytest.approx(
            bundle.calibrator.predict(row.raw_score)
        )
        assert outcome.scored == 1

    def test_an_unfitted_tier_abstains_rather_than_being_scored_as_the_reference(
        self, bundle
    ):
        row = candidate(tier=Tier.T4_GRAPH, probability=0.8)
        outcome = apply_bundle([row], bundle)
        assert row.calibrated_p == 0.8
        assert outcome.abstained_uncovered == 1

    def test_applying_twice_changes_nothing(self, bundle):
        row = candidate(lexical=1.0)
        apply_bundle([row], bundle)
        first = row.calibrated_p
        apply_bundle([row], bundle)
        assert row.calibrated_p == first

    def test_the_counters_account_for_every_candidate(self, bundle):
        rows = [
            candidate("a", tier=Tier.T0_EXACT),
            candidate("b", verified=False),
            candidate("c"),
            candidate("d", tier=Tier.T4_GRAPH),
        ]
        outcome = apply_bundle(rows, bundle)
        assert outcome.considered == len(rows)


class TestTheRoutingThresholds:
    def test_a_fitted_threshold_is_marked_as_fitted(self):
        thresholds = thresholds_from(
            select_tau_high([0.9], [True], target_precision=0.5), DecisionThresholds()
        )
        assert thresholds.tau_high == pytest.approx(0.9)
        assert thresholds.tau_high_is_fitted

    def test_tau_low_is_narrowed_rather_than_left_inverted(self):
        """An inverted pair is invalid; the fitted threshold is what survives."""
        thresholds = thresholds_from(
            select_tau_high([0.3], [True], target_precision=0.5), DecisionThresholds()
        )
        assert thresholds.tau_high == pytest.approx(0.3)
        assert thresholds.tau_low == pytest.approx(0.3)

    def test_tau_low_is_otherwise_untouched(self):
        base = DecisionThresholds(tau_low=0.6)
        thresholds = thresholds_from(
            select_tau_high([0.95], [True], target_precision=0.5), base
        )
        assert thresholds.tau_low == pytest.approx(0.6)

    def test_the_run_config_carries_the_fitted_threshold_into_its_hash(self):
        """A fitted run must not hash identically to a placeholder one."""
        train = [features(lexical_score=1.0), features(lexical_score=0.0)]
        bundle = fit_bundle(
            train,
            [True, False],
            train,
            [True, False],
            provenance=provenance(),
            target_precision=0.5,
        )
        base = RunConfig(run_id="r")
        configured = configure_for(base, bundle)
        assert configured.thresholds.tau_high_is_fitted
        assert configured.config_hash != base.config_hash
        assert configured.run_id == base.run_id


class TestTheReportedPopulation:
    def test_only_labelled_residual_evaluation_links_are_measured(self):
        rows = [
            candidate("a", tier=Tier.T0_EXACT, probability=1.0),
            candidate("b", probability=0.9),
            candidate("c", probability=None),
        ]
        rows[0].is_truth_positive = True
        rows[1].is_truth_positive = True
        probabilities, labels = residual_rows(rows)
        assert probabilities == (0.9,)
        assert labels == (True,)

    def test_an_unlabelled_candidate_is_skipped_rather_than_assumed_wrong(self):
        rows = [candidate("a", probability=0.9)]
        assert residual_rows(rows) == ((), ())

    def test_rows_are_counted_by_tier_in_ladder_order(self):
        rows = [
            candidate("a", tier=Tier.T3_FUZZY),
            candidate("b", tier=Tier.T2_AGGREGATION),
            candidate("c", tier=Tier.T2_AGGREGATION),
        ]
        assert list(rows_by_tier(rows)) == ["T2_AGGREGATION", "T3_FUZZY"]
        assert rows_by_tier(rows)["T2_AGGREGATION"] == 2


class TestBlendOutcomeArithmetic:
    def test_merging_adds_every_counter(self):
        merged = BlendOutcome(scored=1, refusals_kept=2).merge(
            BlendOutcome(scored=3, bypassed_deterministic=4, abstained_uncovered=5)
        )
        assert merged.scored == 4
        assert merged.refusals_kept == 2
        assert merged.bypassed_deterministic == 4
        assert merged.abstained_uncovered == 5
        assert merged.considered == 15


def test_a_bundle_path_is_returned_so_a_caller_can_report_it(tmp_path):
    train = [features(lexical_score=1.0), features(lexical_score=0.0)]
    bundle = fit_bundle(
        train,
        [True, False],
        train,
        [True, False],
        provenance=provenance(),
        target_precision=0.5,
    )
    destination = tmp_path / "b.json"
    assert bundle.save(destination) == destination
    assert Path(destination).is_file()
