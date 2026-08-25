"""Property tests for the money invariants (PLAN.md §13).

These are the tests that make "no floats in the money path" and "money is
conserved" claims rather than hopes. Every finance engineer who reads the repo
looks for exactly this.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ledgerloop.money import (
    MoneyError,
    allocate_minor,
    assert_minor,
    delta_ratio,
    format_minor,
    parse_major_to_minor,
    parse_minor_units,
    sum_minor,
    tolerance_minor,
    within_tolerance,
)

# Bounded to a realistic range: ±₹100 crore in paise. Unbounded ints would test
# Python's bignum implementation rather than this module.
amounts = st.integers(min_value=-10**12, max_value=10**12)
positive_amounts = st.integers(min_value=0, max_value=10**12)


class TestNoFloatsEverEnter:
    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_assert_minor_rejects_every_float(self, value: float):
        with pytest.raises(MoneyError):
            assert_minor(value)

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_sum_minor_rejects_a_float_anywhere_in_the_sequence(self, value: float):
        with pytest.raises(MoneyError):
            sum_minor([1, 2, value, 3])

    @given(amounts)
    def test_integral_floats_are_still_rejected(self, value: int):
        """`499900.0` is arithmetically equal but still a breach of the invariant."""
        with pytest.raises(MoneyError):
            assert_minor(float(value))


class TestSumConservation:
    @given(st.lists(amounts, max_size=50))
    def test_sum_matches_builtin_for_ints(self, values: list[int]):
        assert sum_minor(values) == sum(values)

    @given(st.lists(amounts, min_size=1, max_size=50))
    def test_sum_is_order_independent(self, values: list[int]):
        """Matching must not depend on the order records were read in."""
        assert sum_minor(values) == sum_minor(list(reversed(values)))

    @given(st.lists(amounts, max_size=30), st.lists(amounts, max_size=30))
    def test_sum_is_additive_across_partitions(self, left: list[int], right: list[int]):
        """A subset's total plus its complement's total is the whole."""
        assert sum_minor(left) + sum_minor(right) == sum_minor([*left, *right])


class TestAllocationConservesMoney:
    @given(
        amounts,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    )
    def test_parts_sum_exactly_to_the_total(self, total: int, weights: list[int]):
        """The A09 split-payout guarantee: no paise created, none destroyed."""
        assume(sum(weights) > 0)
        parts = allocate_minor(total, weights)
        assert sum(parts) == total

    @given(
        amounts,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    )
    def test_part_count_matches_weight_count(self, total: int, weights: list[int]):
        assume(sum(weights) > 0)
        assert len(allocate_minor(total, weights)) == len(weights)

    @given(
        amounts,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    )
    def test_allocation_is_deterministic(self, total: int, weights: list[int]):
        """Seeded regeneration must be byte-identical, so allocation is a pure function."""
        assume(sum(weights) > 0)
        assert allocate_minor(total, weights) == allocate_minor(total, weights)

    @given(
        positive_amounts,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    )
    def test_no_part_exceeds_a_positive_total(self, total: int, weights: list[int]):
        assume(sum(weights) > 0)
        parts = allocate_minor(total, weights)
        assert all(0 <= part <= total for part in parts)

    @given(
        amounts,
        st.lists(st.integers(min_value=0, max_value=10**9), min_size=1, max_size=20),
    )
    def test_sign_of_every_part_follows_the_total(self, total: int, weights: list[int]):
        """A negative adjustments line must never allocate a positive part."""
        assume(sum(weights) > 0)
        parts = allocate_minor(total, weights)
        if total >= 0:
            assert all(part >= 0 for part in parts)
        else:
            assert all(part <= 0 for part in parts)


class TestToleranceProperties:
    @given(amounts, positive_amounts, st.integers(min_value=0, max_value=10_000))
    def test_band_never_falls_below_the_floor(self, amount: int, floor: int, bps: int):
        """A band silently stricter than configured would look like recall loss."""
        assert tolerance_minor(amount, floor_minor=floor, bps=bps) >= floor

    @given(amounts, st.integers(min_value=0, max_value=10_000))
    def test_band_covers_the_proportional_share(self, amount: int, bps: int):
        band = tolerance_minor(amount, floor_minor=0, bps=bps)
        assert band * 10_000 >= abs(amount) * bps

    @given(amounts, st.integers(min_value=0, max_value=10_000))
    def test_band_ignores_sign(self, amount: int, bps: int):
        assert tolerance_minor(amount, floor_minor=0, bps=bps) == tolerance_minor(
            -amount, floor_minor=0, bps=bps
        )

    @given(amounts, amounts, positive_amounts)
    def test_within_tolerance_is_symmetric(self, left: int, right: int, band: int):
        assert within_tolerance(left, right, band) == within_tolerance(right, left, band)

    @given(amounts, positive_amounts)
    def test_a_value_always_matches_itself(self, amount: int, band: int):
        assert within_tolerance(amount, amount, band)


class TestDeltaRatioStaysInFeatureSpace:
    @given(amounts, amounts)
    def test_never_negative(self, delta: int, base: int):
        assert delta_ratio(delta, base) >= 0.0

    @given(amounts, amounts)
    def test_ignores_sign_of_both_arguments(self, delta: int, base: int):
        assert delta_ratio(delta, base) == delta_ratio(-delta, -base)

    @given(amounts)
    def test_zero_delta_is_always_a_perfect_ratio(self, base: int):
        assert delta_ratio(0, base) == 0.0


class TestParsingRoundTrips:
    @given(amounts)
    def test_minor_units_round_trip_through_text(self, amount: int):
        assert parse_minor_units(str(amount)) == amount

    @given(amounts)
    def test_format_then_reparse_recovers_the_amount(self, amount: int):
        """format_minor is presentation, but it must not lose information."""
        rendered = format_minor(amount, symbol="")
        assert parse_major_to_minor(rendered.replace(",", "")) == amount

    @given(amounts)
    def test_decimal_parsing_matches_integer_parsing(self, amount: int):
        """Both entry points agree, so ingest can pick either per source format."""
        as_major = Decimal(amount) / 100
        assert parse_major_to_minor(str(as_major)) == parse_minor_units(str(amount))
