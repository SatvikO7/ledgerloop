"""The deterministic narration parser.

Three things are being asserted, in order of how much they matter:

1. Every narration shape the generator can emit is parsed exactly -- all
   twelve merchants, all three variants each, through all nine templates.
2. Rows that must match nothing yield nothing. A parser that invents a
   counterparty for a rent debit has produced a false positive before the
   matcher is even involved.
3. Shapes the generator does *not* emit -- a truncated UTR, a spaced UTR, an
   unreadable row -- degrade rather than crash.
"""

from __future__ import annotations

import pytest

from ledgerloop.generator.vocab import (
    MERCHANTS,
    NARRATION_WITH_UTR,
    NARRATION_WITHOUT_UTR,
    NOISE_NARRATIONS,
)
from ledgerloop.ingest.narration import CANONICAL_UTR_DIGITS, parse_narration

UTR = "UTR2026030412345"
VARIANT = "RAZORPAY SOFTWARE PVT"


def _render(template: str, variant: str) -> str:
    return template.format(variant=variant, utr=UTR) if "{utr}" in template else (
        template.format(variant=variant)
    )


class TestGeneratedShapes:
    @pytest.mark.parametrize("template", NARRATION_WITH_UTR)
    def test_every_utr_template_yields_both_fields(self, template):
        parsed = parse_narration(_render(template, VARIANT))
        assert parsed.utr == UTR
        assert parsed.merchant == VARIANT
        assert not parsed.utr_is_truncated
        assert parsed.is_credit_shaped

    @pytest.mark.parametrize("template", NARRATION_WITHOUT_UTR)
    def test_the_a07_templates_yield_a_merchant_and_no_utr(self, template):
        """A07 ``MISSING_REFERENCE`` strips the reference; the name must survive.

        This is the class that justifies T3 existing at all: the row is still
        resolvable, but only through the merchant name plus the amount.
        """
        parsed = parse_narration(_render(template, VARIANT))
        assert parsed.utr is None
        assert parsed.merchant == VARIANT
        assert parsed.resolved_by_regex

    def test_the_whole_vocabulary_round_trips_through_every_template(self):
        """324 narrations: 12 merchants x 3 variants x 9 templates, exact.

        Compared against the variant with its separators flattened, because
        that is what recovery means here -- ``NYKAA E-RETAIL`` and
        ``NYKAA E RETAIL`` are the same name and the parser is entitled to
        return either. Recovering *fewer tokens* than the variant contains
        would be the failure, and this catches it.
        """
        checked = 0
        for merchant in MERCHANTS:
            for variant in merchant.variants:
                expected = variant.replace("-", " ")
                for template in (*NARRATION_WITH_UTR, *NARRATION_WITHOUT_UTR):
                    parsed = parse_narration(_render(template, variant))
                    assert parsed.merchant == expected, (variant, template, parsed.merchant)
                    checked += 1
        assert checked == 324

    def test_a_hyphen_inside_the_name_does_not_truncate_it(self):
        """``NYKAA E-RETAIL PRIVATE`` -- the separator falls inside the name.

        Consecutive surviving segments are one field, not two. Splitting on the
        hyphen and keeping only the first fragment would silently discard
        ``RETAIL``, the token that identifies the merchant.
        """
        parsed = parse_narration(f"NEFT CR-NYKAA E-RETAIL PRIVATE-{UTR}-SETTLEMENT")
        assert parsed.merchant == "NYKAA E-RETAIL PRIVATE".replace("-", " ")
        assert parsed.merchant_normalized == "NYKAA E RETAIL"


class TestRowsThatMustMatchNothing:
    @pytest.mark.parametrize("narration", NOISE_NARRATIONS)
    def test_noise_rows_yield_no_merchant_and_no_utr(self, narration):
        parsed = parse_narration(narration)
        assert parsed.utr is None
        assert parsed.merchant is None
        assert not parsed.is_credit_shaped
        assert not parsed.resolved_by_regex

    def test_the_a10_orphan_credit_names_a_rail_but_no_counterparty(self):
        """``DIRECT TRANSFER`` is a descriptor a bank writes *instead* of a name."""
        parsed = parse_narration("NEFT CR-DIRECT TRANSFER-370162-INWARD")
        assert parsed.rail == "NEFT"
        assert parsed.utr is None
        assert parsed.merchant is None
        assert parsed.reference_tokens == ("370162",)

    def test_a_bare_digit_run_is_not_mistaken_for_a_utr(self):
        assert parse_narration("NEFT CR-ACME LTD-370162-INWARD").utr is None

    def test_the_rail_gate_records_what_it_refused(self):
        """The gate is auditable: an operator can see the name it declined."""
        parsed = parse_narration("SALARY CREDIT PAYROLL BATCH")
        assert parsed.merchant is None
        assert parsed.discarded_segments


class TestDegradedInput:
    def test_a_truncated_utr_is_kept_and_flagged(self):
        parsed = parse_narration("NEFT CR-ACME LTD-UTR202603-SETTLEMENT")
        assert parsed.utr == "UTR202603"
        assert parsed.utr_is_truncated
        assert len("202603") < CANONICAL_UTR_DIGITS

    def test_a_spaced_utr_is_still_found(self):
        assert parse_narration("NEFT CR-ACME LTD-UTR 2026030412345").utr == UTR

    def test_a_utr_alone_passes_the_rail_gate(self):
        """A reference is itself proof the row is a transfer."""
        parsed = parse_narration(f"ACME TRADING-{UTR}")
        assert parsed.rail is None
        assert parsed.utr == UTR
        assert parsed.merchant == "ACME TRADING"

    @pytest.mark.parametrize("narration", ["", "   ", "///", "---", "12345"])
    def test_an_unreadable_narration_never_raises(self, narration):
        parsed = parse_narration(narration)
        assert parsed.utr is None
        assert parsed.merchant is None

    def test_parsing_is_deterministic(self):
        narration = f"IMPS CR/{UTR}/ZOMATO HYPERPURE PVT/PAYOUT"
        assert parse_narration(narration) == parse_narration(narration)

    def test_the_first_utr_wins_when_a_row_carries_two(self):
        parsed = parse_narration(f"NEFT CR-{UTR}-ACME-UTR2026030499999-SETTLEMENT")
        assert parsed.utr == UTR

    def test_the_longest_free_text_run_is_preferred(self):
        """Between two separated candidates, the merchant is the wider field."""
        parsed = parse_narration(f"NEFT CR-XY-{UTR}-ACME TRADING COMPANY-SETTLEMENT")
        assert parsed.merchant == "ACME TRADING COMPANY"
        assert "XY" in parsed.discarded_segments

    def test_the_normalised_narration_is_carried_through(self):
        parsed = parse_narration(f"NEFT CR-{VARIANT}-{UTR}-SETTLEMENT")
        assert parsed.normalized == (
            f"NEFT CR {VARIANT} {UTR} SETTLEMENT"
        )
        assert parsed.merchant_skeleton == "RZRPYSFTWR"
