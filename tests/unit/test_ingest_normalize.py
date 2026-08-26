"""Identifier, merchant and narration normalisation.

The tests that matter here are the ones about *recovery*: the PSP's mangled
order references and the bank's abbreviated merchant names are the two places
where a string that means the right thing does not compare equal to it, and
everything above T0 exists because of them.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from ledgerloop.generator.baseline import NON_BREAKING_HYPHEN
from ledgerloop.generator.vocab import MERCHANTS
from ledgerloop.ingest.normalize import (
    LEGAL_SUFFIXES,
    fold_text,
    is_order_ref_shaped,
    merchant_skeleton,
    normalize_identifier,
    normalize_merchant_name,
    normalize_narration,
    normalize_order_ref,
    normalize_utr,
)


class TestOrderReferenceRecovery:
    """The three corruptions of PLAN.md 5.1, and what normalisation can do."""

    def test_a_clean_reference_is_unchanged(self):
        assert normalize_order_ref("ORD-2026-004821") == "ORD-2026-004821"

    def test_the_space_separated_lowercase_form_is_recovered(self):
        assert normalize_order_ref("ord 2026 004821") == "ORD-2026-004821"

    def test_the_non_breaking_hyphen_form_is_recovered(self):
        mangled = f"ORD{NON_BREAKING_HYPHEN}2026{NON_BREAKING_HYPHEN}004821"
        assert mangled != "ORD-2026-004821"  # the corruption is real
        assert normalize_order_ref(mangled) == "ORD-2026-004821"

    def test_a_null_reference_stays_none(self):
        assert normalize_order_ref(None) is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n", "---", "///"])
    def test_nothing_recoverable_yields_none(self, blank):
        assert normalize_order_ref(blank) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "ORD--2026__004821",
            "  ord.2026.004821  ",
            "-ORD-2026-004821-",
            "ORD/2026/004821",
        ],
    )
    def test_every_separator_collapses_to_the_same_form(self, raw):
        assert normalize_order_ref(raw) == "ORD-2026-004821"

    def test_normalisation_does_not_judge_the_grammar(self):
        """A foreign-looking reference is canonicalised, not discarded.

        Deciding whether a reference names a real order is T0's job; dropping
        it here would delete the evidence an exception needs.
        """
        assert normalize_order_ref("inv 99 1") == "INV-99-1"

    @pytest.mark.parametrize(
        ("text", "shaped"),
        [
            ("ORD-2026-004821", True),
            ("ORD-2026-4821", False),
            ("INV-2026-004821", False),
            ("ORD-2026-004821-X", False),
            (None, False),
        ],
    )
    def test_the_grammar_check_is_separate_and_strict(self, text, shaped):
        assert is_order_ref_shaped(text) is shaped


#: The dash-like codepoints `normalize.py` folds. Written as codepoints for the
#: same reason the module itself does: a literal here is one linter autofix away
#: from becoming an ASCII hyphen, at which point the test proves nothing.
DASH_CODEPOINTS = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE58, 0xFE63, 0xFF0D)


class TestFolding:
    def test_folding_is_idempotent(self):
        once = fold_text(f"  Ord{chr(0x2011)}2026 {chr(0x2013)} 004821 ")
        assert fold_text(once) == once

    def test_every_dash_variant_becomes_ascii(self):
        for codepoint in DASH_CODEPOINTS:
            assert normalize_identifier(f"A{chr(codepoint)}B") == "A-B", hex(codepoint)

    def test_compatibility_forms_fold_to_ascii(self):
        fullwidth = "".join(chr(0xFF00 + ord(ch) - 0x20) for ch in "ORD-2026")
        assert normalize_identifier(fullwidth) == "ORD-2026"

    def test_accents_are_stripped_rather_than_dropped(self):
        """``NESTLE`` and ``NESTLÉ`` are the same counterparty.

        Removing the accented letter outright would lose a character the two
        forms share; decomposing and dropping the mark keeps it.
        """
        accented = "Nestl" + chr(0x00E9) + " India Limited"
        assert normalize_merchant_name(accented) == "NESTLE INDIA"
        assert merchant_skeleton("Nestl" + chr(0x00E9)) == merchant_skeleton("NESTLE")

    def test_a_script_with_no_ascii_form_is_not_transliterated(self):
        """A guess would be worse than an absence."""
        assert normalize_identifier("".join(chr(c) for c in (0x4E00, 0x4E8C, 0x4E09))) is None

    def test_nfkc_alone_would_not_have_been_enough(self):
        """U+2011 decomposes to U+2010, not to ASCII hyphen-minus."""
        import unicodedata

        assert unicodedata.normalize("NFKC", NON_BREAKING_HYPHEN) != "-"
        assert normalize_identifier(NON_BREAKING_HYPHEN.join(["A", "B"])) == "A-B"


class TestMerchantNames:
    def test_legal_suffixes_are_stripped_from_the_tail(self):
        assert normalize_merchant_name("Razorpay Software Private Limited") == (
            "RAZORPAY SOFTWARE"
        )
        assert normalize_merchant_name("RAZORPAY SOFTWARE PVT") == "RAZORPAY SOFTWARE"
        assert normalize_merchant_name("RAZORPAY SOFTWARE PRIVATE LTD") == "RAZORPAY SOFTWARE"

    def test_the_indian_p_l_abbreviation_is_a_legal_suffix(self):
        assert normalize_merchant_name("RZRPAY SFTWR P L") == "RZRPAY SFTWR"

    def test_a_suffix_word_inside_the_name_survives(self):
        """Tail-only stripping. ``URBAN COMPANY`` must not lose its ``COMPANY``."""
        assert normalize_merchant_name("URBAN COMPANY TECH LTD") == "URBAN COMPANY TECH"

    def test_a_name_made_entirely_of_suffixes_is_not_emptied(self):
        assert normalize_merchant_name("PVT LTD") == "PVT"

    def test_an_empty_name_normalises_to_empty(self):
        assert normalize_merchant_name("   ") == ""
        assert merchant_skeleton("!!!") == ""

    def test_the_suffix_table_is_upper_case(self):
        assert all(suffix == suffix.upper() for suffix in LEGAL_SUFFIXES)


class TestMerchantSkeleton:
    """The transformation ``vocab.py`` argues embeddings could not learn."""

    def test_the_worked_example_from_the_docstring(self):
        assert merchant_skeleton("Razorpay Software Private Limited") == "RZRPYSFTWR"
        assert merchant_skeleton("RZRPAY SFTWR P L") == "RZRPYSFTWR"

    def test_a_leading_vowel_survives(self):
        assert merchant_skeleton("INSTAMART") == "INSTMRT"
        assert merchant_skeleton("URBAN") == "URBN"

    def test_doubled_letters_collapse(self):
        assert merchant_skeleton("SWIGGY") == merchant_skeleton("SWGY")

    def test_word_splitting_differences_do_not_separate_two_forms(self):
        assert merchant_skeleton("NYKAA E RETAIL") == merchant_skeleton("NYKAA ERETAIL")
        assert merchant_skeleton("GROWW INVEST TECH") == merchant_skeleton("GROWW INVESTTECH")

    def test_most_of_the_vocabulary_collapses_onto_its_legal_name(self):
        """The measured claim, not an assumed one.

        Eight of the twelve merchants have *every* bank variant collapse onto
        the legal name's skeleton exactly. The four that do not differ only
        where a variant abbreviates a whole word (``TECH`` for
        ``TECHNOLOGIES``, ``SVCS`` for ``SERVICES``) -- a word-level
        substitution no character transformation can undo, and precisely what
        RapidFuzz is brought in for at T3.
        """
        exact = [
            merchant
            for merchant in MERCHANTS
            if all(
                merchant_skeleton(variant) == merchant_skeleton(merchant.legal_name)
                for variant in merchant.variants
            )
        ]
        assert len(exact) == 8

    def test_no_two_merchants_share_a_skeleton(self):
        """The property T3 actually depends on.

        A skeleton collision would make two merchants indistinguishable by
        name, and no amount of fuzzy scoring recovers from that. Across all
        forty-eight names in the vocabulary there are none.
        """
        owners: dict[str, set[str]] = {}
        for merchant in MERCHANTS:
            for name in (merchant.legal_name, *merchant.variants):
                owners.setdefault(merchant_skeleton(name), set()).add(merchant.merchant_id)
        assert all(len(ids) == 1 for ids in owners.values()), {
            skeleton: ids for skeleton, ids in owners.items() if len(ids) > 1
        }

    def test_every_bank_variant_is_nearest_its_own_merchant(self):
        """The measured prediction that T3 will work.

        For each of the thirty-six bank variants, the legal name whose skeleton
        it most resembles is its own. Scored with ``difflib`` rather than
        RapidFuzz because RapidFuzz is a step-6 dependency and this claim is
        about the *normalisation*, not about any particular scorer.
        """
        legal = {m.merchant_id: merchant_skeleton(m.legal_name) for m in MERCHANTS}
        for merchant in MERCHANTS:
            for variant in merchant.variants:
                skeleton = merchant_skeleton(variant)
                ranked = sorted(
                    (
                        (SequenceMatcher(None, skeleton, target).ratio(), merchant_id)
                        for merchant_id, target in legal.items()
                    ),
                    reverse=True,
                )
                assert ranked[0][1] == merchant.merchant_id, (variant, ranked[:2])
                assert ranked[0][0] >= 0.80, (variant, ranked[0])


class TestNarrationNormalisation:
    def test_separators_become_spaces(self):
        assert normalize_narration("NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT") == (
            "NEFT CR RAZORPAY SOFTWARE PVT UTR2026030412345 SETTLEMENT"
        )

    def test_slashes_and_hyphens_are_interchangeable(self):
        assert normalize_narration("A/B-C") == normalize_narration("A-B/C")

    def test_it_is_idempotent(self):
        once = normalize_narration("  IMPS CR//UTR123456//NYKAA E-RETAIL  ")
        assert normalize_narration(once) == once


class TestUtrNormalisation:
    def test_punctuation_and_case_are_removed(self):
        assert normalize_utr(" utr-2026-0304-12345 ") == "UTR2026030412345"

    def test_nothing_recoverable_yields_none(self):
        assert normalize_utr(None) is None
        assert normalize_utr("  ") is None
        assert normalize_utr("--") is None
