"""Property tests for ingest.

Two families, answering different questions.

**Text properties** ask whether normalisation is well-behaved on input nobody
anticipated -- arbitrary Unicode, empty strings, punctuation soup. The
comparison functions are idempotent and total: a normaliser that is not
idempotent silently makes ``f(a) == f(b)`` depend on how many times each side
has been through it. ``merchant_skeleton`` is the deliberate exception, and it
is tested for the properties it does have rather than the one it cannot.

**Corpus properties** ask whether ingest is faithful across the generator's
configuration space rather than on one committed fixture: several splits,
every difficulty, several seeds. The strongest of them is money -- what ingest
reads back must equal what the generator declared it wrote, to the paise.

Generation is the expensive part, so each ``(split, difficulty, seed)`` corpus
is built once and memoised for the whole module.
"""

from __future__ import annotations

import tempfile
from datetime import date
from functools import cache
from pathlib import Path
from string import ascii_uppercase, digits

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import IngestResult, ingest_dataset
from ledgerloop.ingest.dates import DateOrder, infer_date_order, parse_slash_date
from ledgerloop.ingest.narration import parse_narration
from ledgerloop.ingest.normalize import (
    fold_text,
    merchant_skeleton,
    normalize_identifier,
    normalize_merchant_name,
    normalize_narration,
    normalize_order_ref,
    normalize_utr,
)
from ledgerloop.models.enums import Difficulty, SplitName

_DASH_ALPHABET = "".join(chr(code) for code in (0x2010, 0x2011, 0x2013, 0x2212)) + "AB-"

# Weighted toward the characters that actually break normalisers: separators,
# dashes of every flavour, and case. Unbounded text would mostly test Unicode.
messy_text = st.one_of(
    st.text(max_size=40),
    st.text(alphabet="ORD-0123456789 _.abcdefghij", max_size=40),
    st.text(alphabet=_DASH_ALPHABET, max_size=20),
)


class TestNormalisationIsIdempotent:
    """``f(f(x)) == f(x)``. Otherwise equality depends on the call count."""

    @given(messy_text)
    def test_fold_text(self, raw: str):
        once = fold_text(raw)
        assert fold_text(once) == once

    @given(messy_text)
    def test_normalize_identifier(self, raw: str):
        once = normalize_identifier(raw)
        assert normalize_identifier(once) == once

    @given(messy_text)
    def test_normalize_order_ref(self, raw: str):
        once = normalize_order_ref(raw)
        assert normalize_order_ref(once) == once

    @given(messy_text)
    def test_normalize_narration(self, raw: str):
        once = normalize_narration(raw)
        assert normalize_narration(once) == once

    @given(messy_text)
    def test_normalize_merchant_name(self, raw: str):
        once = normalize_merchant_name(raw)
        assert normalize_merchant_name(once) == once

    @given(messy_text)
    def test_normalize_utr(self, raw: str):
        once = normalize_utr(raw)
        assert normalize_utr(once) == once


class TestTheSkeletonIsStableUnderNoise:
    """Idempotence is the wrong property here -- see ``merchant_skeleton``.

    What matters is that the skeleton ignores exactly the noise it was built to
    ignore, and that it is a function of the name rather than of its
    punctuation.
    """

    @given(messy_text)
    def test_it_is_deterministic(self, raw: str):
        assert merchant_skeleton(raw) == merchant_skeleton(raw)

    @given(messy_text)
    def test_separators_and_case_make_no_difference(self, raw: str):
        assert merchant_skeleton(raw) == merchant_skeleton(raw.lower().replace(" ", "-"))

    @given(messy_text, st.sampled_from(["PVT LTD", "PRIVATE LIMITED", "P L"]))
    def test_a_legal_form_makes_no_difference(self, raw: str, suffix: str):
        assume(normalize_merchant_name(raw))
        assert merchant_skeleton(raw) == merchant_skeleton(f"{raw} {suffix}")


class TestNormalisationIsTotal:
    """No input raises, and ``None`` means genuine absence rather than defeat."""

    @given(messy_text)
    def test_an_identifier_is_none_exactly_when_no_ascii_alnum_survives(self, raw: str):
        """A canonical identifier is ASCII, and the boundary is exact.

        NFKC has already folded everything that *has* an ASCII form, so what
        reaches this point without one -- Devanagari digits, CJK, Greek -- is
        genuinely not an identifier character.
        """
        folded = fold_text(raw)
        has_ascii_alnum = any(ch in ascii_uppercase + digits for ch in folded)
        result = normalize_identifier(raw)
        assert (result is not None) == has_ascii_alnum
        if result is not None:
            assert result == result.upper()
            assert not result.startswith("-")
            assert not result.endswith("-")

    @given(messy_text)
    def test_parse_narration_never_raises(self, raw: str):
        parsed = parse_narration(raw)
        assert parsed.raw == raw
        if parsed.merchant is not None:
            assert parsed.is_credit_shaped
            assert parsed.resolved_by_regex

    @given(messy_text)
    def test_a_skeleton_never_grows(self, raw: str):
        assert len(merchant_skeleton(raw)) <= len(normalize_merchant_name(raw))


class TestDateOrderRoundTrips:
    """Format, infer, parse. The dates that come back must be the ones that went in."""

    @given(
        st.lists(
            st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31)),
            min_size=1,
            max_size=30,
        )
    )
    def test_a_day_first_column(self, dates: list[date]):
        """Undecidable columns are still read correctly -- the default matches.

        ``proven`` tracks whether the column *witnessed* the order, and the
        assertion pins it to the only thing that can witness it: a day past the
        twelfth.
        """
        column = [value.strftime("%d/%m/%Y") for value in dates]
        evidence = infer_date_order(column)
        assert evidence.order is DateOrder.DAY_FIRST
        assert evidence.proven == any(value.day > 12 for value in dates)
        assert [parse_slash_date(text, evidence.order) for text in column] == dates

    @given(
        st.lists(
            st.builds(
                date,
                st.integers(min_value=2000, max_value=2099),
                st.integers(min_value=1, max_value=12),
                st.integers(min_value=13, max_value=28),
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_a_month_first_column(self, dates: list[date]):
        """Every day is past the twelfth, so month-first is always provable."""
        column = [value.strftime("%m/%d/%Y") for value in dates]
        evidence = infer_date_order(column)
        assert evidence.order is DateOrder.MONTH_FIRST
        assert evidence.proven
        assert [parse_slash_date(text, evidence.order) for text in column] == dates


@cache
def _corpus(split: SplitName, difficulty: Difficulty, seed: int) -> tuple[object, IngestResult]:
    """Generate one dataset and ingest it. Memoised -- generation is the slow half."""
    directory = Path(tempfile.mkdtemp(prefix="ll-ingest-"))
    dataset = generate_to_disk(
        GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
    )
    return dataset, ingest_dataset(directory, strict=True)


@pytest.mark.parametrize("split", [SplitName.DEV, SplitName.CALIBRATION])
@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", [7, 42])
class TestTheCorpusIngestsFaithfully:
    """Across split x difficulty x seed, not just on the committed fixture."""

    def test_every_generated_record_survives_ingest(self, split, difficulty, seed):
        dataset, result = _corpus(split, difficulty, seed)
        world = dataset.world
        assert result.problems == ()
        assert len(result.orders) == len(world.orders)
        assert len(result.payments) == len(world.payments)
        assert len(result.settlements) == len(world.settlements)
        assert len(result.bank_txns) == len(world.bank_txns)

    def test_money_read_back_equals_money_declared(self, split, difficulty, seed):
        """To the paise. The strongest thing ingest can claim about itself."""
        dataset, result = _corpus(split, difficulty, seed)
        world = dataset.world
        assert sum(s.net_minor for s in result.settlements) == world.declared_net_total_minor()
        assert sum(t.credit_minor for t in result.bank_txns) == sum(
            t.credit_minor for t in world.bank_txns
        )
        assert sum(t.debit_minor for t in result.bank_txns) == sum(
            t.debit_minor for t in world.bank_txns
        )
        assert sum(o.amount_minor for o in result.orders) == sum(
            o.amount_minor for o in world.orders
        )
        assert sum(p.amount_minor for p in result.payments) == sum(
            p.amount_minor for p in world.payments
        )

    def test_the_date_order_is_always_provable_on_a_real_split(self, split, difficulty, seed):
        """A statement spanning weeks always contains a day past the twelfth."""
        _, result = _corpus(split, difficulty, seed)
        assert result.date_order.proven
        assert result.date_order.order is DateOrder.DAY_FIRST

    def test_every_present_reference_is_recovered_into_a_real_order(
        self, split, difficulty, seed
    ):
        _, result = _corpus(split, difficulty, seed)
        known = {order.order_id for order in result.orders}
        for payment in result.payments:
            if payment.order_ref_raw is None:
                assert payment.order_ref_normalized is None
            else:
                assert payment.order_ref_normalized in known, payment.order_ref_raw

    def test_no_debit_row_ever_carries_a_counterparty(self, split, difficulty, seed):
        """Only incoming money is a settlement candidate."""
        _, result = _corpus(split, difficulty, seed)
        for txn in result.bank_txns:
            if not txn.is_credit:
                assert txn.extracted_merchant is None
                assert txn.extracted_utr is None

    def test_every_record_keeps_its_provenance(self, split, difficulty, seed):
        _, result = _corpus(split, difficulty, seed)
        assert all(record.raw is not None for record in result.normalized)


@settings(deadline=None, max_examples=8)
@given(seed=st.integers(min_value=0, max_value=500))
def test_ingest_loses_nothing_at_an_arbitrary_seed(seed: int):
    dataset, result = _corpus(SplitName.DEV, Difficulty.STANDARD, seed)
    world = dataset.world
    assert result.problems == ()
    assert result.record_count == (
        len(world.orders)
        + len(world.payments)
        + len(world.settlements)
        + len(world.bank_txns)
    )
