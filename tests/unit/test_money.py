"""Unit tests for the money module -- the invariant everything else rests on."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ledgerloop.money import (
    RUPEE,
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


class TestAssertMinor:
    def test_accepts_int(self):
        assert assert_minor(499900) == 499900
        assert assert_minor(-431200) == -431200
        assert assert_minor(0) == 0

    def test_rejects_float(self):
        with pytest.raises(MoneyError, match="float is forbidden"):
            assert_minor(499900.0)

    def test_rejects_bool(self):
        """bool subclasses int, so `True + 499900 == 499901` would pass silently."""
        with pytest.raises(MoneyError, match="bool is not a money value"):
            assert_minor(True)
        with pytest.raises(MoneyError, match="bool is not a money value"):
            assert_minor(False)

    def test_rejects_decimal(self):
        """Decimal must be converted deliberately, never coerced."""
        with pytest.raises(MoneyError, match="converted explicitly"):
            assert_minor(Decimal("4999.00"))

    def test_rejects_str(self):
        with pytest.raises(MoneyError, match="expected int minor units"):
            assert_minor("499900")

    def test_error_names_the_field(self):
        with pytest.raises(MoneyError, match="net_paise"):
            assert_minor(1.5, field="net_paise")


class TestParseMinorUnits:
    def test_int_passthrough(self):
        assert parse_minor_units(3680323) == 3680323

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("3680323", 3680323), ("  499900 ", 499900), ("-431200", -431200), ("1_000", 1000)],
    )
    def test_digit_strings(self, text, expected):
        assert parse_minor_units(text) == expected

    def test_rejects_decimal_point(self):
        """A decimal point means the caller picked the wrong parser."""
        with pytest.raises(MoneyError, match="parse_major_to_minor"):
            parse_minor_units("36803.23")

    def test_rejects_empty(self):
        with pytest.raises(MoneyError, match="empty string"):
            parse_minor_units("   ")

    def test_rejects_garbage(self):
        with pytest.raises(MoneyError, match="not an integer"):
            parse_minor_units("UTR2026030412345")

    def test_rejects_bool(self):
        with pytest.raises(MoneyError, match="bool is not a money value"):
            parse_minor_units(True)


class TestParseMajorToMinor:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("36803.23", 3680323),
            ("4999", 499900),
            ("0.01", 1),
            ("-4312.00", -431200),
            ("1,234.56", 123456),
        ],
    )
    def test_exact_conversion(self, text, expected):
        assert parse_major_to_minor(text) == expected

    def test_no_float_rounding_error(self):
        """The float path would give 1114.9999999999998 -> 111499. Decimal gives 111500."""
        assert parse_major_to_minor("1115.00") == 111500
        assert parse_major_to_minor("0.29") == 29
        assert parse_major_to_minor("8.87") == 887

    def test_rejects_sub_minor_precision(self):
        """Refuse to round silently -- the file or the scale assumption is wrong."""
        with pytest.raises(MoneyError, match="sub-minor-unit precision"):
            parse_major_to_minor("36803.234")

    def test_rejects_float_input(self):
        with pytest.raises(MoneyError, match="float input"):
            parse_major_to_minor(36803.23)

    def test_rejects_non_finite(self):
        with pytest.raises(MoneyError, match="not a finite amount"):
            parse_major_to_minor("Infinity")

    def test_rejects_garbage(self):
        with pytest.raises(MoneyError, match="not a decimal amount"):
            parse_major_to_minor("NEFT CR")

    def test_rejects_bool(self):
        with pytest.raises(MoneyError, match="bool is not a money value"):
            parse_major_to_minor(True)

    def test_rejects_empty(self):
        with pytest.raises(MoneyError, match="empty string"):
            parse_major_to_minor("  ")


class TestFormatMinor:
    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, "₹0.00"),
            (1, "₹0.01"),
            (99999, "₹999.99"),
            (-431200, "-₹4,312.00"),
        ],
    )
    def test_rendering(self, amount, expected):
        assert format_minor(amount) == expected

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (3680323, "₹36,803.23"),  # 36 thousand
            (100_000_000, "₹10,00,000.00"),  # 10 lakh
            (10_000_000_000, "₹10,00,00,000.00"),  # 10 crore
        ],
    )
    def test_indian_digit_grouping(self, amount, expected):
        """Lakh/crore grouping -- ₹12,34,567 is what an Indian controller reads."""
        assert format_minor(amount) == expected

    def test_rejects_float(self):
        with pytest.raises(MoneyError):
            format_minor(1.5)


class TestSumMinor:
    def test_sums_ints(self):
        assert sum_minor([499900, 320100, 1]) == 820001

    def test_empty_is_zero(self):
        assert sum_minor([]) == 0

    def test_rejects_a_single_float_in_the_list(self):
        """A float deep in a subset-sum candidate list is the failure this prevents."""
        with pytest.raises(MoneyError, match=r"amounts\[1\]"):
            sum_minor([499900, 320100.0, 1])


class TestTolerance:
    def test_floor_wins_for_small_amounts(self):
        # 0.5% of ₹100 is 50 paise, below the ₹1 floor.
        assert tolerance_minor(10_000, floor_minor=RUPEE, bps=50) == 100

    def test_proportional_wins_for_large_amounts(self):
        # 0.5% of ₹10,000 is ₹50.
        assert tolerance_minor(1_000_000, floor_minor=RUPEE, bps=50) == 5_000

    def test_proportional_part_rounds_up(self):
        """A band that rounds down would be stricter than advertised."""
        # 0.5% of 201 paise = 1.005 paise -> 2, not 1. Floor of 0 isolates the effect.
        assert tolerance_minor(201, floor_minor=0, bps=50) == 2

    def test_negative_amount_uses_magnitude(self):
        assert tolerance_minor(-1_000_000, floor_minor=RUPEE, bps=50) == 5_000

    def test_rejects_negative_floor(self):
        with pytest.raises(MoneyError, match="non-negative"):
            tolerance_minor(1000, floor_minor=-1, bps=50)

    def test_rejects_negative_bps(self):
        with pytest.raises(MoneyError, match="bps must be non-negative"):
            tolerance_minor(1000, floor_minor=0, bps=-50)

    def test_within_tolerance_rejects_negative_band(self):
        with pytest.raises(MoneyError, match="band must be non-negative"):
            within_tolerance(1000, 1000, -1)

    def test_within_tolerance_is_inclusive(self):
        assert within_tolerance(1000, 1100, 100) is True
        assert within_tolerance(1000, 1101, 100) is False

    def test_within_tolerance_is_symmetric(self):
        assert within_tolerance(1000, 1100, 100) == within_tolerance(1100, 1000, 100)


class TestDeltaRatio:
    def test_basic(self):
        assert delta_ratio(50, 1000) == pytest.approx(0.05)

    def test_uses_magnitude(self):
        assert delta_ratio(-50, 1000) == delta_ratio(50, 1000)

    def test_zero_over_zero_is_zero(self):
        assert delta_ratio(0, 0) == 0.0

    def test_nonzero_over_zero_is_inf(self):
        """A discrepancy against nothing must never look like a perfect match."""
        assert delta_ratio(50, 0) == float("inf")

    def test_rejects_float_inputs(self):
        with pytest.raises(MoneyError):
            delta_ratio(50.0, 1000)


class TestAllocateMinor:
    def test_even_split(self):
        assert allocate_minor(1000, [1, 1]) == [500, 500]

    def test_conserves_total_when_it_does_not_divide(self):
        """The A09 split-payout case: no paise created or destroyed."""
        parts = allocate_minor(1000, [1, 1, 1])
        assert sum(parts) == 1000
        assert parts == [334, 333, 333]

    def test_weighted(self):
        parts = allocate_minor(3680323, [499900, 320100])
        assert sum(parts) == 3680323

    def test_negative_total_allocates_symmetrically(self):
        """Adjustments lines and chargebacks are negative."""
        parts = allocate_minor(-1000, [1, 1, 1])
        assert sum(parts) == -1000
        assert parts == [-334, -333, -333]

    def test_ties_broken_by_position_for_reproducibility(self):
        assert allocate_minor(1000, [1, 1, 1]) == allocate_minor(1000, [1, 1, 1])

    def test_rejects_empty_weights(self):
        with pytest.raises(MoneyError, match="non-empty"):
            allocate_minor(1000, [])

    def test_rejects_zero_weight_sum(self):
        with pytest.raises(MoneyError, match="must not sum to zero"):
            allocate_minor(1000, [0, 0])

    def test_rejects_negative_weights(self):
        with pytest.raises(MoneyError, match="non-negative"):
            allocate_minor(1000, [1, -1])

    def test_rejects_float_total(self):
        with pytest.raises(MoneyError):
            allocate_minor(1000.0, [1, 1])
