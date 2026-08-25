"""Metric correctness, on truth sets small enough to verify by hand.

The evaluator is the one component whose bugs are invisible: a matcher that
breaks fails loudly, but a scorer that quietly counts wrong produces a number
that looks exactly like a real one. So every metric here is checked against a
set whose expected value can be computed in a sentence.

Two properties get disproportionate attention because they are the ones a
reader of ``EVALUATION.md`` will rely on hardest: the behaviour of the interval
at a perfect score, and the refusal to report a zero denominator as a zero.
"""

from __future__ import annotations

import math

import pytest

from ledgerloop.eval.metrics import (
    PredictedLink,
    confusion,
    evaluate,
    evaluation_links_by_class,
    link_metrics,
    match_rate,
    money_view,
    recall_by_anomaly_class,
    wilson_interval,
)
from ledgerloop.models import (
    AnomalyClass,
    Difficulty,
    ExpectedStatus,
    GroundTruth,
    GroundTruthLink,
    GroundTruthRecord,
    LinkType,
    SplitName,
    bank_ref,
    order_ref,
    payment_ref,
    settlement_ref,
)

PAY_1 = "payment:PAY-1"
PAY_2 = "payment:PAY-2"
PAY_3 = "payment:PAY-3"
BNK_1 = "bank_txn:BNK-1"
BNK_2 = "bank_txn:BNK-2"
BNK_3 = "bank_txn:BNK-3"


def _truth(**overrides) -> GroundTruth:
    kwargs = {
        "split": SplitName.DEV,
        "difficulty": Difficulty.STANDARD,
        "seed": 42,
        "generator_version": "0.2.0",
    }
    kwargs.update(overrides)
    return GroundTruth(**kwargs)


def _credited(payment: str, bank: str, amount: int) -> GroundTruthLink:
    return GroundTruthLink(
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref(payment),
        target_ref=bank_ref(bank),
        amount_minor=amount,
    )


def _verdict(
    ref,
    anomaly: AnomalyClass = AnomalyClass.CLEAN,
    status: ExpectedStatus = ExpectedStatus.MATCHED,
    impact: int = 0,
) -> GroundTruthRecord:
    return GroundTruthRecord(
        record_ref=ref, expected_status=status, anomaly_class=anomaly, impact_minor=impact
    )


def _predict(payment: str, bank: str, amount: int = 0) -> PredictedLink:
    return PredictedLink(
        source_ref=payment_ref(payment), target_ref=bank_ref(bank), amount_minor=amount
    )


class TestWilsonInterval:
    def test_no_trials_yields_total_ignorance(self):
        """No evidence must widen the interval, not narrow it."""
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_a_perfect_score_still_has_a_lower_bound_below_one(self):
        """The reason this project uses Wilson rather than the normal approximation.

        At 250/250 the normal approximation computes ``p ± z·sqrt(p(1-p)/n)``
        with ``p(1-p) == 0`` and returns ``[1.0, 1.0]`` -- certainty of
        perfection from 250 samples. The headline claim of this project is a
        precision figure, so the estimator that breaks exactly at a clean run is
        the one estimator it cannot use.
        """
        low, high = wilson_interval(250, 250)
        assert high == 1.0
        assert low < 1.0
        assert 0.98 < low < 0.99

    def test_one_error_in_250_moves_the_lower_bound_visibly(self):
        """PLAN.md's sample-size warning, made checkable."""
        clean_low, _ = wilson_interval(250, 250)
        one_wrong_low, _ = wilson_interval(249, 250)
        assert one_wrong_low < clean_low

    def test_interval_brackets_the_point_estimate(self):
        low, high = wilson_interval(50, 100)
        assert low < 0.5 < high

    def test_interval_stays_inside_the_unit_range(self):
        for successes, trials in ((0, 1), (1, 1), (1, 3), (7, 9)):
            low, high = wilson_interval(successes, trials)
            assert 0.0 <= low <= high <= 1.0

    def test_more_samples_narrow_the_interval(self):
        narrow_low, narrow_high = wilson_interval(900, 1000)
        wide_low, wide_high = wilson_interval(9, 10)
        assert (narrow_high - narrow_low) < (wide_high - wide_low)

    def test_impossible_counts_are_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed trials"):
            wilson_interval(5, 3)
        with pytest.raises(ValueError, match="non-negative"):
            wilson_interval(-1, 3)


class TestConfusion:
    def test_partitions_predictions_and_truth(self):
        matrix = confusion(
            [(PAY_1, BNK_1), (PAY_2, BNK_2)], {(PAY_1, BNK_1), (PAY_3, BNK_3)}
        )
        assert matrix.true_positives == {(PAY_1, BNK_1)}
        assert matrix.false_positives == {(PAY_2, BNK_2)}
        assert matrix.false_negatives == {(PAY_3, BNK_3)}

    def test_a_repeated_prediction_is_one_claim(self):
        """Asserting the same link twice must not let a system move its own
        denominator."""
        matrix = confusion([(PAY_1, BNK_1)] * 5, {(PAY_1, BNK_1)})
        assert matrix.predicted_count == 1
        assert matrix.precision == 1.0


class TestLinkMetricsOnHandBuiltSets:
    def test_two_of_three_correct_with_one_missed(self):
        metrics = link_metrics(
            [(PAY_1, BNK_1), (PAY_2, BNK_2), (PAY_3, BNK_3)],
            {(PAY_1, BNK_1), (PAY_2, BNK_2), ("payment:PAY-4", "bank_txn:BNK-4")},
        )
        assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (
            2,
            1,
            1,
        )
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.recall == pytest.approx(2 / 3)
        assert metrics.f1 == pytest.approx(2 / 3)

    def test_f1_is_the_harmonic_mean(self):
        metrics = link_metrics(
            [(PAY_1, BNK_1), (PAY_2, BNK_2)], {(PAY_1, BNK_1), (PAY_3, BNK_3), (PAY_2, BNK_1)}
        )
        expected = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        assert metrics.f1 == pytest.approx(expected)

    def test_false_positive_cost_sums_only_the_wrong_links(self):
        """A rupee figure, not a ratio -- and it must ignore the correct links."""
        metrics = link_metrics(
            [(PAY_1, BNK_1), (PAY_2, BNK_2)],
            {(PAY_1, BNK_1)},
            asserted_amount_by_pair={(PAY_1, BNK_1): 500_000, (PAY_2, BNK_2): 431_200},
        )
        assert metrics.false_positive_cost_minor == 431_200

    def test_cost_is_zero_when_no_amounts_are_supplied(self):
        metrics = link_metrics([(PAY_2, BNK_2)], {(PAY_1, BNK_1)})
        assert metrics.false_positive_cost_minor == 0


class TestDegenerateDenominators:
    """A zero denominator is not a score. It is the absence of one."""

    def test_predicting_nothing_is_not_perfect_precision(self):
        metrics = link_metrics([], {(PAY_1, BNK_1)})
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_predicting_nothing_widens_the_interval_to_everything(self):
        metrics = link_metrics([], {(PAY_1, BNK_1)})
        assert (metrics.precision_ci_low, metrics.precision_ci_high) == (0.0, 1.0)

    def test_empty_truth_gives_no_recall_credit(self):
        metrics = link_metrics([(PAY_1, BNK_1)], set())
        assert metrics.recall == 0.0
        assert metrics.false_positives == 1

    def test_both_empty(self):
        metrics = link_metrics([], set())
        assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
        assert (metrics.precision_ci_low, metrics.precision_ci_high) == (0.0, 1.0)


class TestMatchRate:
    def test_denominator_excludes_unmatchable_records(self):
        truth = _truth(
            records=(
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1")),
                _verdict(
                    bank_ref("BNK-9"),
                    AnomalyClass.ORPHAN_BANK_CREDIT,
                    ExpectedStatus.UNMATCHABLE,
                ),
            )
        )
        result = match_rate([(PAY_1, BNK_1)], truth)
        assert result.denominator_refs == {PAY_1, BNK_1}
        assert result.resolved_refs == {PAY_1, BNK_1}
        assert result.rate == 1.0

    def test_denominator_excludes_orders_and_settlements(self):
        """Structural records are attached by edges the sources assert. Charging
        the matcher for them would understate it as dishonestly as crediting it
        would overstate it."""
        truth = _truth(
            records=(
                _verdict(order_ref("ORD-1")),
                _verdict(settlement_ref("SETL-1")),
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1")),
            )
        )
        result = match_rate([], truth)
        assert result.denominator_refs == {PAY_1, BNK_1}

    def test_an_incorrect_assertion_still_counts_as_coverage(self):
        """Match rate measures reach; precision measures correctness. Folding
        the second into the first would double-count it."""
        truth = _truth(
            records=(_verdict(payment_ref("PAY-1")), _verdict(bank_ref("BNK-2")))
        )
        result = match_rate([(PAY_1, "bank_txn:BNK-2")], truth)
        assert result.rate == 1.0

    def test_no_reconcilable_records_is_not_a_perfect_rate(self):
        assert match_rate([], _truth()).rate == 0.0


class TestRecallByAnomalyClass:
    def test_attribution_comes_from_the_records_not_the_link_label(self):
        """The generator labels records, never links: every emitted
        ``GroundTruthLink`` carries the default ``A01_CLEAN``. Grouping on the
        link field would report one all-clean row and hide every other class.
        """
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 100),),
            records=(
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1"), AnomalyClass.MISSING_REFERENCE),
            ),
        )
        assert truth.links[0].anomaly_class is AnomalyClass.CLEAN
        assert set(evaluation_links_by_class(truth)) == {AnomalyClass.MISSING_REFERENCE}

    def test_a_link_broken_two_ways_appears_in_both_rows(self):
        """Settled decision 14: anomalies compose along independent aspects, so
        forcing a multi-class link into one row would misreport whichever class
        lost the tiebreak."""
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 100),),
            records=(
                _verdict(payment_ref("PAY-1"), AnomalyClass.CHARGEBACK_NETTED),
                _verdict(bank_ref("BNK-1"), AnomalyClass.SPLIT_PAYOUT),
            ),
        )
        grouped = evaluation_links_by_class(truth)
        assert set(grouped) == {AnomalyClass.CHARGEBACK_NETTED, AnomalyClass.SPLIT_PAYOUT}

    def test_clean_collects_only_links_clean_at_both_ends(self):
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 100), _credited("PAY-2", "BNK-2", 100)),
            records=(
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1")),
                _verdict(payment_ref("PAY-2")),
                _verdict(bank_ref("BNK-2"), AnomalyClass.TIMING_SHIFT),
            ),
        )
        grouped = evaluation_links_by_class(truth)
        assert grouped[AnomalyClass.CLEAN] == {(PAY_1, BNK_1)}
        assert grouped[AnomalyClass.TIMING_SHIFT] == {(PAY_2, BNK_2)}

    def test_classes_that_score_zero_are_reported_not_omitted(self):
        """PLAN.md §9.1: the table exists to publish the bad rows."""
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 100), _credited("PAY-2", "BNK-2", 100)),
            records=(
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1")),
                _verdict(payment_ref("PAY-2")),
                _verdict(bank_ref("BNK-2"), AnomalyClass.MISSING_REFERENCE),
            ),
        )
        recalls = recall_by_anomaly_class([(PAY_1, BNK_1)], truth)
        assert recalls[AnomalyClass.CLEAN] == 1.0
        assert recalls[AnomalyClass.MISSING_REFERENCE] == 0.0

    def test_rows_are_ordered_by_class_value(self):
        """Stable ordering keeps two runs of the report diffable."""
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 100), _credited("PAY-2", "BNK-2", 100)),
            records=(
                _verdict(payment_ref("PAY-1"), AnomalyClass.SPLIT_PAYOUT),
                _verdict(bank_ref("BNK-1")),
                _verdict(payment_ref("PAY-2"), AnomalyClass.ROUNDING_DRIFT),
                _verdict(bank_ref("BNK-2")),
            ),
        )
        assert list(recall_by_anomaly_class([], truth)) == [
            AnomalyClass.ROUNDING_DRIFT,
            AnomalyClass.SPLIT_PAYOUT,
        ]


class TestMoneyView:
    def test_reconciled_and_outstanding_partition_the_link_money(self):
        """No paise created by over-asserting, none destroyed by under-asserting."""
        truth = _truth(
            links=(
                _credited("PAY-1", "BNK-1", 500_000),
                _credited("PAY-2", "BNK-2", 431_200),
            )
        )
        view = money_view([(PAY_1, BNK_1)], truth)
        assert view.reconciled_minor == 500_000
        assert view.outstanding_minor == 431_200
        assert view.total_minor == 931_200

    def test_structural_links_carry_no_money_in_this_view(self):
        truth = _truth(
            links=(
                _credited("PAY-1", "BNK-1", 500_000),
                GroundTruthLink(
                    link_type=LinkType.ORDER_PAID_BY,
                    source_ref=order_ref("ORD-1"),
                    target_ref=payment_ref("PAY-1"),
                    amount_minor=500_000,
                ),
            )
        )
        assert money_view([(PAY_1, BNK_1)], truth).total_minor == 500_000

    def test_asserting_a_link_that_does_not_exist_reconciles_nothing(self):
        truth = _truth(links=(_credited("PAY-1", "BNK-1", 500_000),))
        view = money_view([(PAY_2, BNK_2)], truth)
        assert view.reconciled_minor == 0
        assert view.outstanding_minor == 500_000


class TestEvaluate:
    def _dataset(self) -> GroundTruth:
        return _truth(
            links=(
                _credited("PAY-1", "BNK-1", 500_000),
                _credited("PAY-2", "BNK-2", 431_200),
            ),
            records=(
                _verdict(payment_ref("PAY-1")),
                _verdict(bank_ref("BNK-1")),
                _verdict(payment_ref("PAY-2"), AnomalyClass.MISSING_REFERENCE),
                _verdict(bank_ref("BNK-2"), AnomalyClass.MISSING_REFERENCE),
                _verdict(
                    bank_ref("BNK-9"),
                    AnomalyClass.ORPHAN_BANK_CREDIT,
                    ExpectedStatus.UNMATCHABLE,
                    impact=250_000,
                ),
            ),
        )

    def test_assembles_a_complete_run_metrics(self):
        truth = self._dataset()
        metrics = evaluate(
            [_predict("PAY-1", "BNK-1", 500_000), _predict("PAY-3", "BNK-9", 900_000)],
            truth,
            run_id="b0-dev-42",
        )
        assert metrics.run_id == "b0-dev-42"
        assert metrics.record_count == 5
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.true_positives == 1
        assert metrics.link_metrics.false_positives == 1
        assert metrics.link_metrics.false_negatives == 1
        assert metrics.link_metrics.false_positive_cost_minor == 900_000

    def test_record_count_comes_from_the_verdict_list(self):
        """One verdict per record, so the count cannot disagree with the
        denominators computed beside it."""
        truth = self._dataset()
        assert evaluate([], truth, run_id="x").record_count == len(truth.records)

    def test_unmatchable_ceiling_is_reported_separately(self):
        metrics = evaluate([], self._dataset(), run_id="x")
        assert metrics.unmatchable_count == 1
        assert metrics.unmatchable_impact_minor == 250_000

    def test_throughput_is_zero_without_a_measured_duration(self):
        metrics = evaluate([], self._dataset(), run_id="x", wall_clock_ms=0)
        assert metrics.records_per_second == 0.0

    def test_throughput_is_records_over_seconds(self):
        metrics = evaluate([], self._dataset(), run_id="x", wall_clock_ms=500)
        assert metrics.records_per_second == pytest.approx(10.0)

    def test_a_repeated_prediction_does_not_double_the_cost(self):
        """`evaluate` de-duplicates the same way `confusion` does, so the cost
        table and the counts describe the same set of links."""
        truth = self._dataset()
        metrics = evaluate(
            [_predict("PAY-3", "BNK-9", 900_000)] * 3, truth, run_id="x"
        )
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.false_positives == 1
        assert metrics.link_metrics.false_positive_cost_minor == 900_000

    def test_perfect_predictions_score_perfectly(self):
        truth = self._dataset()
        metrics = evaluate(
            [_predict("PAY-1", "BNK-1", 500_000), _predict("PAY-2", "BNK-2", 431_200)],
            truth,
            run_id="x",
        )
        assert metrics.auto_match_precision == 1.0
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.recall == 1.0
        assert metrics.outstanding_minor == 0
        # Even at a perfect score the interval refuses to claim certainty.
        assert metrics.link_metrics.precision_ci_low < 1.0
        assert not math.isclose(metrics.link_metrics.precision_ci_low, 1.0)
