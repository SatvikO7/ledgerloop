"""Property tests for T2.

Three families.

**Solver correctness** -- the meet-in-the-middle search agrees with brute force
on sets small enough to enumerate both ways, for arbitrary amounts and windows.
That is the only way to be confident the counting is right, and the counting is
what licenses every T2 match.

**Money** -- the proportional bridge between gross and net conserves, and every
tranche's allocated shares sum to the tranche exactly. Integers throughout.

**The ladder** -- across generated corpora at several splits, seeds and
difficulties: T2 never asserts a wrong link, never touches a settlement an
earlier tier ruled on, never assigns a payment to two tranches, and decides
identically on a rerun.
"""

from __future__ import annotations

import tempfile
from functools import cache
from itertools import combinations
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ledgerloop.config import GeneratorConfig, RunConfig
from ledgerloop.eval.metrics import confusion
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import IngestResult, ingest_dataset
from ledgerloop.matching import MatchRun, run_matching
from ledgerloop.matching.bank_leg import allocated_share_minor
from ledgerloop.matching.subset_sum import find_subsets, meet_in_the_middle
from ledgerloop.matching.tier2_aggregation import expected_credit_minor
from ledgerloop.models.enums import Difficulty, LinkType, SplitName, Tier
from ledgerloop.money import allocate_minor

amounts = st.lists(
    st.integers(min_value=1, max_value=10_000), min_size=1, max_size=10
)
targets = st.integers(min_value=0, max_value=60_000)


def _brute_force(values: list[int], low: int, high: int) -> set[tuple[int, ...]]:
    return {
        combo
        for size in range(1, len(values) + 1)
        for combo in combinations(range(len(values)), size)
        if low <= sum(values[i] for i in combo) <= high
    }


class TestTheSolverAgreesWithBruteForce:
    @given(amounts, targets, st.integers(min_value=0, max_value=500))
    @settings(max_examples=150, deadline=None)
    def test_it_finds_a_solution_exactly_when_one_exists(
        self, values: list[int], target: int, slack: int
    ):
        low, high = max(0, target - slack), target + slack
        expected = _brute_force(values, low, high)
        search = meet_in_the_middle(values, low, high, want=1)
        assert bool(search.solutions) == bool(expected)
        if search.solutions:
            assert search.solutions[0].indices in expected

    @given(amounts, targets, st.integers(min_value=0, max_value=500))
    @settings(max_examples=150, deadline=None)
    def test_uniqueness_is_never_claimed_wrongly(
        self, values: list[int], target: int, slack: int
    ):
        """The claim that licenses every T2 auto-match, checked against the truth."""
        low, high = max(0, target - slack), target + slack
        expected = _brute_force(values, low, high)
        search = find_subsets(values, low, high, want=2)
        if search.is_unique:
            assert len(expected) == 1
            assert search.solutions[0].indices in expected
        if len(expected) > 1:
            assert not search.is_unique

    @given(amounts, targets, st.integers(min_value=0, max_value=500))
    @settings(max_examples=100, deadline=None)
    def test_every_reported_solution_really_lands_in_the_window(
        self, values: list[int], target: int, slack: int
    ):
        low, high = max(0, target - slack), target + slack
        for solution in find_subsets(values, low, high, want=3).solutions:
            assert solution.indices
            assert sum(values[i] for i in solution.indices) == solution.total_minor
            assert low <= solution.total_minor <= high

    @given(amounts, targets, st.integers(min_value=0, max_value=500))
    @settings(max_examples=80, deadline=None)
    def test_the_search_is_deterministic(
        self, values: list[int], target: int, slack: int
    ):
        low, high = max(0, target - slack), target + slack
        assert (
            find_subsets(values, low, high).solutions
            == find_subsets(values, low, high).solutions
        )

    @given(amounts, targets, st.integers(min_value=0, max_value=200))
    @settings(max_examples=80, deadline=None)
    def test_a_wider_window_never_finds_less(
        self, values: list[int], target: int, extra: int
    ):
        narrow = find_subsets(values, max(0, target - 10), target + 10, want=1)
        wide = find_subsets(
            values, max(0, target - 10 - extra), target + 10 + extra, want=1
        )
        assert wide.found >= narrow.found


class TestTheProportionalBridge:
    @given(
        st.integers(min_value=1, max_value=10_000_000),
        st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=200, deadline=None)
    def test_the_two_halves_of_a_split_account_for_the_net(
        self, gross: int, net: int
    ):
        assume(net <= gross)
        part = gross // 3 or 1
        first = expected_credit_minor(part, gross, net)
        second = expected_credit_minor(gross - part, gross, net)
        assert isinstance(first, int) and not isinstance(first, bool)
        assert abs(first + second - net) <= 1

    @given(
        st.integers(min_value=1, max_value=1_000_000),
        st.integers(min_value=1, max_value=1_000_000),
    )
    @settings(max_examples=200, deadline=None)
    def test_a_share_is_never_negative_or_larger_than_the_net(
        self, gross: int, net: int
    ):
        assume(net <= gross)
        for part in (1, gross // 2 or 1, gross):
            share = expected_credit_minor(part, gross, net)
            assert 0 <= share <= net

    @given(
        st.lists(st.integers(min_value=1, max_value=500_000), min_size=1, max_size=12),
        st.integers(min_value=1, max_value=5_000_000),
    )
    @settings(max_examples=150, deadline=None)
    def test_allocating_a_tranche_conserves_it_exactly(
        self, weights: list[int], credit: int
    ):
        shares = allocate_minor(credit, weights)
        assert sum(shares) == credit
        assert all(isinstance(s, int) and not isinstance(s, bool) for s in shares)


@cache
def _matched(
    split: SplitName, difficulty: Difficulty, seed: int
) -> tuple[Path, IngestResult, MatchRun]:
    directory = Path(tempfile.mkdtemp(prefix="ll-t2-"))
    generate_to_disk(
        GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
    )
    ingested = ingest_dataset(directory, strict=True)
    run = run_matching(ingested, RunConfig(run_id=f"{split.value}-{seed}"))
    return directory, ingested, run


@pytest.mark.parametrize("split", [SplitName.DEV, SplitName.CALIBRATION])
@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", [7, 42])
class TestTheLadderWithT2:
    def test_no_t2_prediction_is_wrong(self, split, difficulty, seed):
        """The precision-first claim, extended to the aggregation tier."""
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        t2_pairs = {
            (c.source_ref.key, c.target_ref.key)
            for c in run.candidates
            if c.tier is Tier.T2_AGGREGATION and c.is_evaluable
        }
        predicted = {p.pair for p in run.predictions} & t2_pairs
        assert predicted <= truth.evaluation_pairs

    def test_the_whole_ladder_still_asserts_nothing_wrong(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
        assert matrix.false_positives == frozenset()

    def test_t2_never_revisits_a_settlement_an_earlier_tier_ruled_on(
        self, split, difficulty, seed
    ):
        _, _, run = _matched(split, difficulty, seed)
        earlier: set[str] = set()
        later: set[str] = set()
        for candidate in run.candidates:
            if candidate.link_type is not LinkType.SETTLEMENT_CREDITED_AS:
                continue
            target = later if candidate.tier is Tier.T2_AGGREGATION else earlier
            target.add(candidate.source_ref.key)
        assert earlier & later == set()

    def test_no_payment_is_assigned_to_two_tranches(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        seen: dict[str, str] = {}
        for prediction in run.predictions:
            source = prediction.source_ref.key
            assert seen.setdefault(source, prediction.target_ref.key) == (
                prediction.target_ref.key
            )

    def test_every_t2_match_conserves_its_tranche(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        totals: dict[str, int] = {}
        for candidate in run.candidates:
            if candidate.tier is not Tier.T2_AGGREGATION or not candidate.is_evaluable:
                continue
            if candidate.calibrated_p != 1.0:
                continue
            key = candidate.target_ref.record_id
            totals[key] = totals.get(key, 0) + allocated_share_minor(candidate)
        credits = {t.txn_id: t.credit_minor for t in run.state.normalized if _is_credit(t)}
        for txn_id, total in totals.items():
            assert total == credits[txn_id]

    def test_predicted_amounts_equal_the_truth_amounts(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        truth_amounts = {
            link.pair: link.amount_minor
            for link in truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        for prediction in run.predictions:
            assert prediction.amount_minor == truth_amounts[prediction.pair]

    def test_the_time_bound_never_fires(self, split, difficulty, seed):
        """A run whose solver timed out is a run that is not reproducible."""
        _, _, run = _matched(split, difficulty, seed)
        assert run.aggregation.timeouts == 0

    def test_rerunning_decides_identically(self, split, difficulty, seed):
        _, ingested, run = _matched(split, difficulty, seed)
        again = run_matching(
            ingested,
            RunConfig(run_id=f"{split.value}-{seed}"),
            decided_at=run.decisions[0].decided_at,
        )
        assert again.predictions == run.predictions
        assert [d.decision_id for d in again.decisions] == [
            d.decision_id for d in run.decisions
        ]


def _is_credit(record) -> bool:
    return getattr(record, "credit_minor", 0) > 0


@settings(deadline=None, max_examples=6)
@given(seed=st.integers(min_value=0, max_value=400))
def test_the_ladder_is_precise_at_an_arbitrary_seed(seed: int):
    directory, _, run = _matched(SplitName.DEV, Difficulty.STANDARD, seed)
    truth = load_ground_truth(directory)
    matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
    assert matrix.false_positives == frozenset()
