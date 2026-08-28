"""Mean ± std, and the two cases a naive implementation gets wrong.

Aggregation looks like the least interesting code in Step 10 and is where the
two most misleading cells in an evaluation report come from: a spread of 0.0
printed for a single observation, and a population standard deviation printed
for a sample. Both are tested here rather than trusted.
"""

from __future__ import annotations

from math import isclose

import pytest

from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.summary import Aggregate, aggregate, summarise
from ledgerloop.models.enums import Difficulty, SplitName
from ledgerloop.models.metrics import CostLedger
from ledgerloop.models.truth import GroundTruth


class TestAggregate:
    def test_the_mean_is_the_mean(self):
        result = aggregate("precision", [1.0, 0.5, 0.0])
        assert isclose(result.mean, 0.5)
        assert result.count == 3

    def test_it_is_the_sample_standard_deviation_not_the_population_one(self):
        """`ddof = 1`. Five seeds are a sample of the generator's distribution,
        not the population of every corpus it can produce, and the population
        formula reports a spread narrower than the evidence supports.

        For [0, 2] the sample deviation is sqrt(2) = 1.4142 and the population
        deviation is 1.0 -- far enough apart that a wrong choice is visible.
        """
        result = aggregate("recall", [0.0, 2.0])
        assert result.std is not None
        assert isclose(result.std, 2.0**0.5)

    def test_one_observation_has_an_undefined_spread_not_a_zero_one(self):
        """A spread of zero claims two runs agreed. One run has no spread at
        all, and rendering that as 0.0000 is a claim nothing measured."""
        result = aggregate("precision", [0.99])
        assert result.std is None
        assert result.rendered() == "0.9900"

    def test_identical_observations_do_report_a_spread_of_zero(self):
        """The other side of the same rule: five runs that agreed *did* agree,
        and that is a measurement rather than an absence."""
        result = aggregate("precision", [1.0] * 5)
        assert result.std == 0.0
        assert result.rendered() == "1.0000 ± 0.0000"

    def test_no_observations_render_as_na(self):
        assert aggregate("precision", []).rendered() == "n/a"
        assert Aggregate(metric="x", count=0).rendered() == "n/a"

    def test_the_range_travels_with_the_mean(self):
        """A mean of 0.58 over seeds spanning 0.41 to 0.90 is a different claim
        from one over seeds spanning 0.57 to 0.59, and the table prints both."""
        result = aggregate("recall", [0.41, 0.58, 0.90])
        assert result.minimum == 0.41
        assert result.maximum == 0.90

    @pytest.mark.parametrize("digits", [0, 2, 4])
    def test_the_digit_count_applies_to_both_halves(self, digits):
        rendered = aggregate("n", [1.0, 3.0]).rendered(digits=digits)
        mean, _, std = rendered.partition(" ± ")
        assert len(mean.partition(".")[2]) == digits
        assert len(std.partition(".")[2]) == digits


class TestSummarise:
    def test_it_reads_what_it_can_from_the_metrics_rather_than_being_told(self):
        """A summary that took its precision as an argument could disagree with
        the report section beside it. It takes the RunMetrics instead."""
        truth = GroundTruth(
            split=SplitName.TEST,
            difficulty=Difficulty.STANDARD,
            seed=42,
            generator_version="0.2.0",
        )
        metrics = evaluate((), truth, run_id="empty")
        row = summarise(
            "T0-T4",
            metrics,
            split=SplitName.TEST,
            difficulty=Difficulty.STANDARD,
            seed=42,
        )
        assert row.precision == metrics.auto_match_precision
        assert row.match_rate == metrics.match_rate
        assert row.record_count == metrics.record_count

    def test_an_explicit_ledger_overrides_the_metrics_one(self):
        """Ablation rows carry a per-row client, so the cost that belongs to the
        row is passed in rather than read off a RunMetrics several rows share."""
        truth = GroundTruth(
            split=SplitName.DEV,
            difficulty=Difficulty.STANDARD,
            seed=1,
            generator_version="0.2.0",
        )
        row = summarise(
            "T0-T5",
            evaluate((), truth, run_id="x"),
            split=SplitName.DEV,
            difficulty=Difficulty.STANDARD,
            seed=1,
            cost=CostLedger(llm_calls=7, prompt_tokens=100, completion_tokens=20),
            llm_available=True,
        )
        assert row.llm_calls == 7
        assert row.llm_tokens == 120
        assert row.llm_available is True

    def test_llm_available_defaults_to_false_so_a_zero_is_never_ambiguous(self):
        """A zero in the LLM columns means one of two things -- the tier ran and
        contributed nothing, or no model was reachable -- and they are not the
        same finding. The flag is what separates them."""
        truth = GroundTruth(
            split=SplitName.DEV,
            difficulty=Difficulty.STANDARD,
            seed=1,
            generator_version="0.2.0",
        )
        row = summarise(
            "T0-T4",
            evaluate((), truth, run_id="x"),
            split=SplitName.DEV,
            difficulty=Difficulty.STANDARD,
            seed=1,
        )
        assert row.llm_available is False
        assert row.llm_calls == 0
