"""Property tests for the evaluator (PLAN.md §13).

The unit tests check specific numbers the reader can verify by hand. These
check the statements that must hold for *every* input, which is the class of
bug that would otherwise only surface once a real matcher started producing
prediction sets nobody hand-wrote.

The interval properties get the most attention. An evaluator whose confidence
interval could exclude its own point estimate, or could widen as evidence
accumulates, would be worse than reporting no interval at all -- it would make
the headline claim look rigorous while being wrong.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ledgerloop.eval.metrics import (
    confusion,
    link_metrics,
    match_rate,
    money_view,
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
    payment_ref,
)

# Small identifier pools: the interesting behaviour is in how predicted and
# truth sets overlap, and overlap only happens if the pools collide.
_PAYMENTS = st.sampled_from([f"PAY-{index}" for index in range(6)])
_BANKS = st.sampled_from([f"BNK-{index}" for index in range(6)])

pairs = st.tuples(
    _PAYMENTS.map(lambda value: f"payment:{value}"),
    _BANKS.map(lambda value: f"bank_txn:{value}"),
)
pair_sets = st.lists(pairs, max_size=20)
amounts = st.integers(min_value=0, max_value=10**9)


def _truth_from(pair_list, amount_list) -> GroundTruth:
    seen: dict[tuple[str, str], int] = {}
    for pair, amount in zip(pair_list, amount_list, strict=False):
        seen.setdefault(pair, amount)
    return GroundTruth(
        split=SplitName.DEV,
        difficulty=Difficulty.STANDARD,
        seed=42,
        generator_version="0.2.0",
        links=tuple(
            GroundTruthLink(
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(pair[0].split(":", 1)[1]),
                target_ref=bank_ref(pair[1].split(":", 1)[1]),
                amount_minor=amount,
            )
            for pair, amount in seen.items()
        ),
        records=tuple(
            GroundTruthRecord(
                record_ref=ref,
                expected_status=ExpectedStatus.MATCHED,
                anomaly_class=AnomalyClass.CLEAN,
            )
            for ref in sorted(
                {payment_ref(pair[0].split(":", 1)[1]) for pair in seen}
                | {bank_ref(pair[1].split(":", 1)[1]) for pair in seen},
                key=lambda item: item.key,
            )
        ),
    )


class TestConfusionPartitions:
    @given(pair_sets, pair_sets)
    def test_predictions_split_into_true_and_false_positives(self, predicted, truth):
        matrix = confusion(predicted, set(truth))
        assert matrix.predicted_count == len(set(predicted))
        assert not (matrix.true_positives & matrix.false_positives)

    @given(pair_sets, pair_sets)
    def test_truth_splits_into_true_positives_and_false_negatives(self, predicted, truth):
        matrix = confusion(predicted, set(truth))
        assert matrix.truth_count == len(set(truth))
        assert not (matrix.true_positives & matrix.false_negatives)

    @given(pair_sets, pair_sets)
    def test_the_three_sets_are_disjoint(self, predicted, truth):
        matrix = confusion(predicted, set(truth))
        assert not (matrix.false_positives & matrix.false_negatives)


class TestScoresStayInRange:
    @given(pair_sets, pair_sets)
    def test_every_ratio_is_a_probability(self, predicted, truth):
        metrics = link_metrics(predicted, set(truth))
        for value in (metrics.precision, metrics.recall, metrics.f1):
            assert 0.0 <= value <= 1.0

    @given(pair_sets, pair_sets)
    def test_f1_lies_between_precision_and_recall(self, predicted, truth):
        """The harmonic mean is bounded by its arguments."""
        metrics = link_metrics(predicted, set(truth))
        low = min(metrics.precision, metrics.recall)
        high = max(metrics.precision, metrics.recall)
        assert low - 1e-9 <= metrics.f1 <= high + 1e-9

    @given(pair_sets, pair_sets)
    def test_predicting_exactly_the_truth_scores_perfectly(self, predicted, _unused):
        metrics = link_metrics(predicted, set(predicted))
        if predicted:
            assert metrics.precision == 1.0
            assert metrics.recall == 1.0
            assert metrics.false_positives == 0
            assert metrics.false_negatives == 0


class TestIntervalInvariants:
    @given(st.integers(min_value=0, max_value=5000), st.integers(min_value=0, max_value=5000))
    def test_bounds_are_ordered_and_inside_the_unit_range(self, a: int, b: int):
        successes, trials = min(a, b), max(a, b)
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0

    @given(st.integers(min_value=0, max_value=5000), st.integers(min_value=1, max_value=5000))
    def test_the_interval_contains_its_own_point_estimate(self, a: int, b: int):
        """An interval that excluded the estimate it accompanies would make the
        report look rigorous while being wrong."""
        successes, trials = min(a, b), max(a, b)
        low, high = wilson_interval(successes, trials)
        assert low - 1e-12 <= successes / trials <= high + 1e-12

    @given(st.integers(min_value=1, max_value=200))
    def test_more_evidence_never_widens_the_interval(self, scale: int):
        """Same proportion, more samples: the interval must tighten."""
        small_low, small_high = wilson_interval(scale, 2 * scale)
        large_low, large_high = wilson_interval(10 * scale, 20 * scale)
        assert (large_high - large_low) <= (small_high - small_low)

    @given(st.integers(min_value=1, max_value=5000))
    def test_a_flawless_run_still_admits_doubt(self, trials: int):
        """No finite sample proves perfection."""
        low, high = wilson_interval(trials, trials)
        assert high == 1.0
        assert low < 1.0


class TestMoneyAndCoverage:
    @given(pair_sets, st.lists(amounts, max_size=20), pair_sets)
    def test_reconciled_and_outstanding_conserve_the_link_money(
        self, truth_pairs, truth_amounts, predicted
    ):
        """No paise created by over-asserting, none destroyed by under-asserting."""
        truth = _truth_from(truth_pairs, truth_amounts)
        view = money_view(predicted, truth)
        total = sum(
            link.amount_minor
            for link in truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        )
        assert view.reconciled_minor + view.outstanding_minor == total

    @given(pair_sets, st.lists(amounts, max_size=20), pair_sets)
    def test_match_rate_is_a_probability(self, truth_pairs, truth_amounts, predicted):
        truth = _truth_from(truth_pairs, truth_amounts)
        assert 0.0 <= match_rate(predicted, truth).rate <= 1.0

    @given(pair_sets, st.lists(amounts, max_size=20))
    def test_resolving_every_link_reaches_full_coverage(self, truth_pairs, truth_amounts):
        truth = _truth_from(truth_pairs, truth_amounts)
        result = match_rate(truth.evaluation_pairs, truth)
        assert result.resolved_refs == result.denominator_refs

    @given(pair_sets, st.lists(amounts, max_size=20), pair_sets)
    def test_false_positive_cost_never_charges_for_a_correct_link(
        self, truth_pairs, truth_amounts, predicted
    ):
        truth = _truth_from(truth_pairs, truth_amounts)
        asserted = dict.fromkeys(predicted, 1_000)
        metrics = link_metrics(
            predicted, truth.evaluation_pairs, asserted_amount_by_pair=asserted
        )
        assert metrics.false_positive_cost_minor == metrics.false_positives * 1_000
