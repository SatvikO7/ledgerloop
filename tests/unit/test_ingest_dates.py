"""Date parsing and the DD/MM ambiguity.

PLAN.md Phase 2 acceptance says ambiguous ``DD/MM`` dates must be resolved
correctly and a test must assert it. This is that test file, and the assertion
it makes is stronger than the plan asks for: not only that the fixture reads
day-first, but that the reading is *derived from the column* and that the two
ways it can fail -- a contradictory column and a genuinely undecidable one --
are each handled distinctly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from ledgerloop.ingest.dates import (
    IST,
    DateOrder,
    infer_date_order,
    parse_iso_date,
    parse_slash_date,
    parse_timestamp,
)


class TestInference:
    def test_a_day_past_the_twelfth_proves_day_first(self):
        evidence = infer_date_order(["06/03/2026", "14/03/2026", "09/03/2026"])
        assert evidence.order is DateOrder.DAY_FIRST
        assert evidence.proven
        assert evidence.day_first_witnesses == 1
        assert evidence.ambiguous_values == 2

    def test_a_second_component_past_the_twelfth_proves_month_first(self):
        evidence = infer_date_order(["03/14/2026", "03/06/2026"])
        assert evidence.order is DateOrder.MONTH_FIRST
        assert evidence.proven
        assert evidence.month_first_witnesses == 1

    def test_a_column_readable_both_ways_falls_back_to_the_convention(self):
        evidence = infer_date_order(["06/03/2026", "09/04/2026"])
        assert evidence.order is DateOrder.DAY_FIRST
        assert not evidence.proven
        assert evidence.ambiguous_values == 2
        assert "convention" in evidence.basis

    def test_the_fallback_convention_is_configurable(self):
        evidence = infer_date_order(["06/03/2026"], default=DateOrder.MONTH_FIRST)
        assert evidence.order is DateOrder.MONTH_FIRST
        assert not evidence.proven

    def test_evidence_beats_the_configured_default(self):
        """A default is what to do absent evidence, never what to do despite it."""
        evidence = infer_date_order(
            ["14/03/2026", "06/03/2026"], default=DateOrder.MONTH_FIRST
        )
        assert evidence.order is DateOrder.DAY_FIRST
        assert evidence.proven

    def test_a_column_witnessing_both_orders_is_refused(self):
        with pytest.raises(ValueError, match="both orders"):
            infer_date_order(["14/03/2026", "03/14/2026"])

    def test_a_value_with_two_impossible_components_is_refused(self):
        with pytest.raises(ValueError, match="not a date"):
            infer_date_order(["14/25/2026"])

    def test_unparsable_values_are_counted_not_raised(self):
        evidence = infer_date_order(["14/03/2026", "", "2026-03-14", "rubbish"])
        assert evidence.proven
        assert evidence.unparsable_values == 3
        assert evidence.total_values == 4

    def test_an_empty_column_is_ambiguous_rather_than_an_error(self):
        evidence = infer_date_order([])
        assert not evidence.proven
        assert evidence.total_values == 0

    def test_the_basis_names_the_witness_count(self):
        evidence = infer_date_order(["14/03/2026", "06/03/2026"])
        assert "proven by 1 of 2" in evidence.basis


class TestParsing:
    def test_the_ambiguous_example_reads_both_ways(self):
        """``06/03/2026`` is the whole point: 6 March or 3 June."""
        assert parse_slash_date("06/03/2026", DateOrder.DAY_FIRST) == date(2026, 3, 6)
        assert parse_slash_date("06/03/2026", DateOrder.MONTH_FIRST) == date(2026, 6, 3)

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_slash_date("  14/03/2026 ", DateOrder.DAY_FIRST) == date(2026, 3, 14)

    def test_a_hyphen_separator_is_accepted(self):
        assert parse_slash_date("14-03-2026", DateOrder.DAY_FIRST) == date(2026, 3, 14)

    @pytest.mark.parametrize("raw", ["", "2026-03-14", "14/03/26", "14/3", "not a date"])
    def test_malformed_input_raises(self, raw):
        with pytest.raises(ValueError):
            parse_slash_date(raw, DateOrder.DAY_FIRST)

    def test_an_impossible_calendar_date_is_not_rolled_forward(self):
        """``31/02`` is a data error. Silently returning 3 March would be a fabrication."""
        with pytest.raises(ValueError):
            parse_slash_date("31/02/2026", DateOrder.DAY_FIRST)

    def test_a_leap_day_parses(self):
        assert parse_slash_date("29/02/2024", DateOrder.DAY_FIRST) == date(2024, 2, 29)

    def test_iso_dates_need_no_inference(self):
        assert parse_iso_date(" 2026-03-06 ") == date(2026, 3, 6)


class TestTimestamps:
    def test_naive_input_is_taken_as_ist(self):
        assert parse_timestamp("2026-03-04T11:22:11") == datetime(2026, 3, 4, 11, 22, 11)

    def test_an_offset_aware_timestamp_is_converted_to_naive_ist(self):
        """PLAN.md 5.1 shows ``+05:30``; the generator writes naive. Both must compare."""
        assert parse_timestamp("2026-03-04T11:22:11+05:30") == datetime(2026, 3, 4, 11, 22, 11)

    def test_a_foreign_offset_is_shifted_not_truncated(self):
        assert parse_timestamp("2026-03-04T06:00:00+00:00") == datetime(2026, 3, 4, 11, 30)

    def test_mixed_awareness_stays_comparable(self):
        naive = parse_timestamp("2026-03-04T11:22:11")
        aware = parse_timestamp("2026-03-04T05:52:11+00:00")
        assert naive == aware  # would raise TypeError if one kept its tzinfo

    def test_ist_is_five_and_a_half_hours(self):
        assert IST.utcoffset(None) == timedelta(hours=5, minutes=30)
        assert IST != UTC
