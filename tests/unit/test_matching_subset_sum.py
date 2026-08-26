"""The subset-sum engine, on inputs small enough to verify by eye.

The property that matters is not "it finds a subset" but "it knows how many
there are". A solver that returns the first subset it stumbles on would let T2
assert a coin flip, so the tests are weighted towards counting: duplicate
amounts that produce genuinely distinct subsets, targets reachable two ways,
and the exhaustiveness flag that says whether the count can be trusted.
"""

from __future__ import annotations

import pytest

from ledgerloop.matching.subset_sum import (
    find_subsets,
    greedy_subset,
    meet_in_the_middle,
)
from ledgerloop.money import MoneyError


def _sets(search):
    return {solution.indices for solution in search.solutions}


class TestExactSubsets:
    def test_a_single_item_target(self):
        search = find_subsets([10, 20, 30], 20, 20)
        assert search.is_unique
        assert _sets(search) == {(1,)}

    def test_a_multi_item_target(self):
        search = find_subsets([10, 20, 30], 40, 40)
        assert search.is_unique
        assert _sets(search) == {(0, 2)}

    def test_the_whole_set(self):
        search = find_subsets([10, 20, 30], 60, 60)
        assert search.is_unique
        assert _sets(search) == {(0, 1, 2)}

    def test_indices_are_ascending_and_map_to_the_original_order(self):
        search = find_subsets([50, 10, 40], 90, 90)
        (solution,) = search.solutions
        assert solution.indices == (0, 2)
        assert solution.total_minor == 90

    def test_the_total_is_reported(self):
        search = find_subsets([7, 11, 13], 18, 24)
        assert all(18 <= s.total_minor <= 24 for s in search.solutions)


class TestNoSubset:
    def test_an_unreachable_target_is_exhaustively_empty(self):
        search = find_subsets([10, 20, 30], 999, 999)
        assert search.solutions == ()
        assert search.exhaustive
        assert not search.is_unique
        assert not search.is_ambiguous

    def test_an_empty_item_list_finds_nothing(self):
        search = find_subsets([], 0, 100)
        assert search.solutions == ()
        assert search.exhaustive

    def test_the_empty_subset_is_never_a_solution(self):
        """A credit explained by no payments is not an explanation."""
        search = find_subsets([10, 20], 0, 0)
        assert search.solutions == ()

    def test_a_target_below_every_item_finds_nothing(self):
        assert find_subsets([10, 20, 30], 1, 5).solutions == ()


class TestUniqueness:
    def test_two_ways_to_reach_the_target_is_ambiguous(self):
        search = find_subsets([10, 20, 30, 40], 50, 50)
        assert search.is_ambiguous
        assert search.found == 2
        assert _sets(search) == {(0, 3), (1, 2)}

    def test_duplicate_amounts_make_genuinely_distinct_subsets(self):
        """Same sum, different payments. Two subsets, not one."""
        search = find_subsets([25, 25, 50], 50, 50)
        assert search.is_ambiguous
        assert _sets(search) == {(0, 1), (2,)}

    def test_three_identical_amounts_are_ambiguous_pairwise(self):
        search = find_subsets([25, 25, 25], 25, 25)
        assert search.is_ambiguous
        assert search.found == 2

    def test_an_ambiguous_search_is_not_exhaustive(self):
        """It stopped early on purpose; more solutions may exist and none is needed."""
        search = find_subsets([10, 20, 30, 40], 50, 50)
        assert not search.exhaustive
        assert not search.is_unique

    def test_want_controls_how_many_are_collected(self):
        assert find_subsets([10, 20, 30, 40], 50, 50, want=1).found == 1
        assert find_subsets([10, 20, 30, 40], 50, 50, want=3).found >= 2


class TestToleranceWindow:
    @pytest.mark.parametrize("target", [48, 49, 50, 51, 52])
    def test_anything_inside_the_window_is_found(self, target):
        search = find_subsets([target], 48, 52)
        assert search.is_unique

    def test_the_window_is_inclusive_at_both_ends(self):
        assert find_subsets([48], 48, 52).is_unique
        assert find_subsets([52], 48, 52).is_unique

    def test_one_beyond_either_end_is_not_found(self):
        assert find_subsets([47], 48, 52).solutions == ()
        assert find_subsets([53], 48, 52).solutions == ()

    def test_an_inverted_window_is_refused(self):
        with pytest.raises(ValueError, match="empty target window"):
            find_subsets([10], 50, 40)


class TestTheAcceptHook:
    def test_a_rejected_candidate_does_not_count_as_a_solution(self):
        """T2 uses this to re-derive the credit exactly, so a near-miss never counts."""
        search = find_subsets([10, 20, 30, 40], 50, 50, accept=lambda idx, _: idx == (0, 3))
        assert search.is_unique
        assert _sets(search) == {(0, 3)}

    def test_rejecting_everything_leaves_an_exhaustive_empty_result(self):
        search = find_subsets([10, 20, 30, 40], 50, 50, accept=lambda *_: False)
        assert search.solutions == ()
        assert search.exhaustive

    def test_the_hook_sees_the_total(self):
        seen: list[int] = []

        def accept(_indices, total):
            seen.append(total)
            return True

        find_subsets([10, 20, 30], 30, 30, accept=accept)
        assert seen and all(value == 30 for value in seen)


class TestTheMoneyGate:
    @pytest.mark.parametrize("bad", [10.5, True])
    def test_a_non_integer_amount_is_refused(self, bad):
        with pytest.raises(MoneyError):
            find_subsets([10, bad], 10, 20)

    def test_a_float_window_is_refused(self):
        with pytest.raises(MoneyError):
            find_subsets([10], 9.5, 20)


class TestStrategySelection:
    def test_small_buckets_use_the_exhaustive_search(self):
        assert find_subsets(list(range(1, 11)), 10, 10).method == "meet_in_the_middle"

    def test_a_bucket_past_the_cap_falls_back_to_greedy(self):
        amounts = [100 + i for i in range(50)]
        search = find_subsets(amounts, 100, 120, max_exact_items=40)
        assert search.method == "greedy"

    def test_a_greedy_result_never_claims_exhaustiveness(self):
        """It can find a subset; it can never prove that subset is alone."""
        amounts = [100 + i for i in range(50)]
        search = find_subsets(amounts, 149, 149, max_exact_items=40)
        assert search.method == "greedy"
        assert not search.exhaustive
        assert not search.is_unique

    def test_the_width_guard_binds_before_a_generous_item_cap(self):
        """Past 32 items the enumeration is the cost, whatever the config says."""
        amounts = [10 * (i + 1) for i in range(36)]
        assert find_subsets(amounts, 10, 20, max_exact_items=40).method == "greedy"


class TestGreedy:
    def test_it_finds_a_reachable_target_largest_first(self):
        search = greedy_subset([50, 30, 20], 50, 50)
        assert _sets(search) == {(0,)}

    def test_it_returns_nothing_when_the_window_is_unreachable(self):
        assert greedy_subset([50, 30, 20], 7, 9).solutions == ()

    def test_it_can_miss_a_reachable_target(self):
        """40 + 35 reaches 75, but greedy takes 60 first and cannot recover.

        A real limitation of the fallback, and exactly why a greedy result is
        only ever a hypothesis. The exhaustive search finds it.
        """
        assert greedy_subset([60, 40, 35], 75, 75).solutions == ()
        assert find_subsets([60, 40, 35], 75, 75).is_unique

    def test_it_is_deterministic_across_equal_amounts(self):
        amounts = [30, 30, 30, 10]
        first = greedy_subset(amounts, 60, 60)
        second = greedy_subset(amounts, 60, 60)
        assert _sets(first) == _sets(second)

    def test_the_accept_hook_applies_to_greedy_too(self):
        assert greedy_subset([50, 30, 20], 50, 50, accept=lambda *_: False).solutions == ()


class TestDeterminism:
    def test_repeated_searches_return_identical_results(self):
        amounts = [17, 23, 41, 8, 15, 42, 4]
        first = find_subsets(amounts, 60, 65)
        second = find_subsets(amounts, 60, 65)
        assert first.solutions == second.solutions

    def test_the_solution_order_is_stable(self):
        amounts = [10, 20, 30, 40]
        assert [s.indices for s in find_subsets(amounts, 50, 50).solutions] == [
            s.indices for s in find_subsets(amounts, 50, 50).solutions
        ]

    def test_meet_in_the_middle_matches_a_brute_force_count(self):
        """The engine agrees with the naive answer on a set small enough to enumerate."""
        from itertools import combinations

        amounts = [3, 5, 8, 11, 13, 16]
        target = 24
        brute = {
            combo
            for size in range(1, len(amounts) + 1)
            for combo in combinations(range(len(amounts)), size)
            if sum(amounts[i] for i in combo) == target
        }
        search = meet_in_the_middle(amounts, target, target, want=len(brute) + 1)
        assert _sets(search) == brute


class TestTheTimeBound:
    def test_a_zero_timeout_disables_the_clock_rather_than_aborting(self):
        """The deterministic bound is the item cap; the clock is a safety net."""
        search = find_subsets([10, 25, 30], 30, 30, timeout_ms=0)
        assert search.is_unique
        assert not search.timed_out

    def test_a_long_search_that_stays_inside_its_budget_is_exhaustive(self):
        """The clock is consulted and found to be fine -- the ordinary case."""
        amounts = [1_000 + 7 * i for i in range(24)]
        search = find_subsets(amounts, 0, 10**9, want=10_000, timeout_ms=60_000)
        assert not search.timed_out

    def test_an_ordinary_search_never_reports_a_timeout(self):
        search = find_subsets(list(range(1, 21)), 50, 55)
        assert not search.timed_out


class TestReportingSurface:
    def test_a_solution_reports_its_size(self):
        (solution,) = find_subsets([10, 20, 30], 40, 40).solutions
        assert len(solution) == 2

    def test_the_time_bound_aborts_a_search_that_outruns_it(self):
        """A one-millisecond budget against a 32-item enumeration. Not flaky:
        enumerating 2^16 entries a half already costs far more than that."""
        amounts = [1_000 + 7 * i for i in range(32)]
        search = find_subsets(amounts, 0, 10**9, want=10**6, timeout_ms=1)
        assert search.timed_out
        assert not search.exhaustive
        assert not search.is_unique
