"""Metrics and cost-ledger contract tests.

The cost ledger's derived figures are the ones quoted in the pitch, so they get
tests rather than trust.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledgerloop.models import (
    AnomalyClass,
    CalibrationMetrics,
    CostLedger,
    LinkMetrics,
    RunMetrics,
    Tier,
    TierContribution,
)


class TestCostLedger:
    def test_total_tokens(self):
        ledger = CostLedger(prompt_tokens=1200, completion_tokens=180)
        assert ledger.total_tokens == 1380

    def test_calls_per_100_records(self):
        """The stated target is under 30 calls per 300 records."""
        ledger = CostLedger(llm_calls=28)
        assert ledger.calls_per_100_records(300) == pytest.approx(9.333, abs=1e-3)

    def test_calls_per_100_records_handles_an_empty_run(self):
        assert CostLedger(llm_calls=5).calls_per_100_records(0) == 0.0

    def test_cache_hit_rate_reaches_one_on_a_repeat_run(self):
        """A second identical run must consume zero live API calls."""
        assert CostLedger(llm_calls=0, cache_hits=28).cache_hit_rate == 1.0

    def test_cache_hit_rate_on_a_cold_run(self):
        assert CostLedger(llm_calls=28, cache_hits=0).cache_hit_rate == 0.0

    def test_cache_hit_rate_with_no_attempts_is_zero(self):
        """--no-llm makes no attempts; that is 0.0, not a division by zero."""
        assert CostLedger().cache_hit_rate == 0.0

    def test_actual_cost_defaults_to_zero(self):
        """₹0 by architecture, not by luck."""
        assert CostLedger().actual_cost_inr == 0.0

    def test_fallback_depth_records_a_rate_limited_run(self):
        """A run that had to walk the provider ladder must be visible, not silent."""
        assert CostLedger(provider_used="gemini", fallback_depth=1).fallback_depth == 1


class TestLinkMetrics:
    def test_carries_a_confidence_interval(self):
        """A point estimate from ~250 auto-matched links cannot distinguish
        0.99 from 0.97; the interval is what makes the claim measured."""
        metrics = LinkMetrics(
            true_positives=248,
            false_positives=2,
            false_negatives=20,
            precision=0.992,
            recall=0.925,
            f1=0.957,
            precision_ci_low=0.972,
            precision_ci_high=0.998,
        )
        assert metrics.precision_ci_low <= metrics.precision <= metrics.precision_ci_high

    def test_false_positive_cost_is_money(self):
        with pytest.raises(ValidationError, match="float is forbidden"):
            LinkMetrics(
                true_positives=1,
                false_positives=0,
                false_negatives=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                precision_ci_low=1.0,
                precision_ci_high=1.0,
                false_positive_cost_minor=4312.0,
            )


class TestCalibrationMetrics:
    def test_defaults_to_residual_only(self):
        """Including T0/T1 would measure the shape of the corpus, not the calibrator."""
        metrics = CalibrationMetrics(
            ece=0.031, brier=0.042, bin_count=10, populated_bins=8, sample_count=91
        )
        assert metrics.residual_only is True

    def test_populated_bin_count_is_reported(self):
        """A low count invalidates the ECE, so the report must show it."""
        metrics = CalibrationMetrics(
            ece=0.004, brier=0.001, bin_count=10, populated_bins=1, sample_count=250
        )
        assert metrics.populated_bins < metrics.bin_count


class TestRunMetrics:
    def test_defaults_are_zero_not_absent(self):
        metrics = RunMetrics(run_id="RUN-1", record_count=0)
        assert metrics.auto_match_precision == 0.0
        assert metrics.match_rate == 0.0
        assert metrics.link_metrics is None

    def test_per_class_recall_can_hold_every_anomaly_class(self):
        """The 11-row table INCLUDING the classes that do badly."""
        metrics = RunMetrics(
            run_id="RUN-1",
            record_count=300,
            recall_by_anomaly_class=dict.fromkeys(AnomalyClass, 0.0),
        )
        assert len(metrics.recall_by_anomaly_class) == 11

    def test_confusion_matrix_is_rectangular(self):
        """11 anomaly classes against 13 exception classes."""
        metrics = RunMetrics(
            run_id="RUN-1",
            record_count=300,
            exception_confusion={
                AnomalyClass.CHARGEBACK_NETTED.value: {"E_CHARGEBACK_NETTED": 9},
            },
        )
        assert metrics.exception_confusion["A08_CHARGEBACK_NETTED"] == {
            "E_CHARGEBACK_NETTED": 9
        }

    def test_unmatchable_ceiling_is_reported_separately(self):
        metrics = RunMetrics(
            run_id="RUN-1",
            record_count=300,
            unmatchable_count=6,
            unmatchable_impact_minor=250_000,
        )
        assert metrics.unmatchable_count == 6

    def test_tier_contributions_price_each_rung(self):
        metrics = RunMetrics(
            run_id="RUN-1",
            record_count=300,
            tier_contributions=(
                TierContribution(
                    tier=Tier.T2_AGGREGATION,
                    candidates_proposed=48,
                    auto_matched=36,
                    marginal_auto_matched=36,
                ),
            ),
        )
        assert metrics.tier_contributions[0].llm_calls == 0
