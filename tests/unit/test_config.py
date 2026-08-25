"""RunConfig contract tests.

Config is where the project's stated commitments become enforceable: the
precision target, the tolerance bands, the auto-resolution leash, the LLM
budget, and the ablation's tier selection.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledgerloop.config import (
    SPLIT_SIZES,
    STANDARD_PREVALENCE,
    AutoResolutionBounds,
    DecisionThresholds,
    GeneratorConfig,
    LLMConfig,
    MatchingTolerances,
    RunConfig,
)
from ledgerloop.models import AnomalyClass, SplitName


class TestPrevalence:
    def test_standard_prevalence_sums_to_one(self):
        assert sum(STANDARD_PREVALENCE.values()) == pytest.approx(1.0)

    def test_covers_every_anomaly_class(self):
        assert set(STANDARD_PREVALENCE) == set(AnomalyClass)

    def test_fx_reassignment_is_reflected_in_clean(self):
        """A11 is cut; its 2% went to CLEAN, so CLEAN is 0.67 not the plan's 0.65."""
        assert STANDARD_PREVALENCE[AnomalyClass.CLEAN] == pytest.approx(0.67)

    def test_generator_rejects_a_distribution_that_does_not_sum_to_one(self):
        """Silently normalising would make prevalence unverifiable, and every
        downstream metric is conditioned on prevalence."""
        broken = dict(STANDARD_PREVALENCE)
        broken[AnomalyClass.CLEAN] = 0.5
        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            GeneratorConfig(prevalence=broken)

    def test_generator_rejects_negative_weights(self):
        broken = dict(STANDARD_PREVALENCE)
        broken[AnomalyClass.CLEAN] = 0.72
        broken[AnomalyClass.LATE_ARRIVAL] = -0.04
        with pytest.raises(ValidationError, match="non-negative"):
            GeneratorConfig(prevalence=broken)


class TestSplits:
    def test_train_split_exists(self):
        """The correction to PLAN.md §5.4: the blender needs its own fitting data,
        so the calibrator never sees in-sample scores."""
        assert SplitName.TRAIN in SPLIT_SIZES
        assert SPLIT_SIZES[SplitName.TRAIN] > 0

    def test_every_split_has_a_size(self):
        assert set(SPLIT_SIZES) == set(SplitName)

    def test_dev_meets_the_challenge_floor_of_fifty(self):
        assert SPLIT_SIZES[SplitName.DEV] >= 50

    def test_test_split_visibly_exceeds_the_floor(self):
        assert SPLIT_SIZES[SplitName.TEST] >= 300

    def test_order_count_override(self):
        assert GeneratorConfig(split=SplitName.TEST).effective_order_count == 300
        assert GeneratorConfig(split=SplitName.TEST, order_count=17).effective_order_count == 17


class TestThresholds:
    def test_defaults_match_the_plan(self):
        thresholds = DecisionThresholds()
        assert thresholds.tau_high == 0.95
        assert thresholds.tau_low == 0.60
        assert thresholds.target_auto_match_precision == 0.99

    def test_tau_high_is_not_fitted_by_default(self):
        """A report must never present a hand-picked threshold as a fitted one."""
        assert DecisionThresholds().tau_high_is_fitted is False

    def test_ordering_is_enforced(self):
        with pytest.raises(ValidationError, match="must not exceed"):
            DecisionThresholds(tau_high=0.5, tau_low=0.9)

    def test_equal_thresholds_are_permitted(self):
        """A degenerate policy with no review band is a legitimate ablation."""
        assert DecisionThresholds(tau_high=0.8, tau_low=0.8).tau_low == 0.8


class TestTolerances:
    def test_defaults_encode_one_rupee_or_half_a_percent(self):
        tolerances = MatchingTolerances()
        assert tolerances.amount_floor_minor == 100
        assert tolerances.amount_bps == 50
        assert tolerances.date_window_days == 3

    def test_aggregation_epsilon_exceeds_the_single_record_floor(self):
        """A subset accumulates per-payment rounding drift across its members."""
        tolerances = MatchingTolerances()
        assert tolerances.aggregation_epsilon_minor > tolerances.amount_floor_minor

    def test_solver_timeout_is_bounded(self):
        assert MatchingTolerances().subset_solver_timeout_ms == 200


class TestAutoResolutionBounds:
    def test_defaults_are_tight(self):
        bounds = AutoResolutionBounds()
        assert bounds.rounding_per_record_minor == 500
        assert bounds.rounding_per_run_minor == 50_000

    def test_run_bound_exceeds_the_per_record_bound(self):
        bounds = AutoResolutionBounds()
        assert bounds.rounding_per_run_minor > bounds.rounding_per_record_minor

    def test_can_be_disabled_entirely(self):
        assert AutoResolutionBounds(enabled=False).enabled is False


class TestLLMConfig:
    def test_call_budget_matches_the_stated_target(self):
        """< 30 calls per 300 records is the cost-discipline headline."""
        assert LLMConfig().max_calls_per_run == 30

    def test_temperature_is_zero_for_reproducibility(self):
        assert LLMConfig().temperature == 0.0

    def test_validation_retries_are_bounded(self):
        """One retry, then fall through to an exception. Never a crash."""
        assert LLMConfig().validation_retries == 1


class TestRunConfig:
    def test_default_enables_every_tier(self):
        assert RunConfig(run_id="RUN-1").enabled_tiers == (0, 1, 2, 3, 4, 5)

    def test_ablation_prefixes_are_valid(self):
        """The ablation table walks (0,), (0,1), (0,1,2) ... each row priced."""
        for cut in range(1, 6):
            config = RunConfig(run_id=f"ABL-{cut}", enabled_tiers=tuple(range(cut)))
            assert config.enabled_tiers == tuple(range(cut))

    def test_tiers_must_be_ascending_and_unique(self):
        with pytest.raises(ValidationError, match="ascending"):
            RunConfig(run_id="RUN-1", enabled_tiers=(2, 1, 0))
        with pytest.raises(ValidationError, match="must not repeat"):
            RunConfig(run_id="RUN-1", enabled_tiers=(0, 0, 1))

    def test_tiers_must_be_in_range(self):
        with pytest.raises(ValidationError, match=r"range 0\.\.5"):
            RunConfig(run_id="RUN-1", enabled_tiers=(0, 9))

    def test_at_least_one_tier_required(self):
        with pytest.raises(ValidationError, match="at least one tier"):
            RunConfig(run_id="RUN-1", enabled_tiers=())

    def test_t5_without_llm_is_rejected(self):
        """Otherwise the ablation would report an LLM contribution that never happened."""
        with pytest.raises(ValidationError, match="never happened"):
            RunConfig(run_id="RUN-1", llm=LLMConfig(enabled=False))

    def test_no_llm_mode_runs_tiers_zero_to_four(self):
        config = RunConfig(
            run_id="RUN-1", enabled_tiers=(0, 1, 2, 3, 4), llm=LLMConfig(enabled=False)
        )
        assert 5 not in config.enabled_tiers


class TestConfigHash:
    def test_identical_configs_hash_identically(self):
        """Proves a rerun reproduced a result rather than merely resembling it."""
        a = RunConfig(run_id="RUN-A")
        b = RunConfig(run_id="RUN-B")
        assert a.config_hash == b.config_hash

    def test_run_identity_and_paths_are_excluded(self):
        from pathlib import Path

        a = RunConfig(run_id="RUN-A")
        b = RunConfig(run_id="RUN-B", data_dir=Path("elsewhere"))
        assert a.config_hash == b.config_hash

    def test_a_threshold_change_changes_the_hash(self):
        a = RunConfig(run_id="RUN-A")
        b = RunConfig(run_id="RUN-A", thresholds=DecisionThresholds(tau_high=0.90))
        assert a.config_hash != b.config_hash

    def test_a_tier_change_changes_the_hash(self):
        a = RunConfig(run_id="RUN-A")
        b = RunConfig(
            run_id="RUN-A", enabled_tiers=(0, 1, 2), llm=LLMConfig(enabled=False)
        )
        assert a.config_hash != b.config_hash

    def test_hash_is_stable_across_calls(self):
        config = RunConfig(run_id="RUN-A")
        assert config.config_hash == config.config_hash

    def test_config_is_immutable(self):
        config = RunConfig(run_id="RUN-A")
        with pytest.raises(ValidationError):
            config.seed = 99
