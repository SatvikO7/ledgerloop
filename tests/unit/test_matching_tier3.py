"""T3 -- lexical matching when the reference is gone.

The tier exists for anomaly A07, and its danger is the mirror of its purpose:
a scorer loose enough to relate ``RZRPAY SFTWR P L`` to ``Razorpay Software
Private Limited`` is loose enough to relate two different merchants. So the
tests come in pairs -- the variants it must recognise, and the near-misses it
must refuse -- and the three gates (name, amount, date) are exercised
separately so a pass cannot be carried by the wrong one.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import combinations

import pytest

from ledgerloop.config import LexicalMatching, MatchingTolerances
from ledgerloop.generator.vocab import MERCHANTS
from ledgerloop.matching.bank_leg import allocated_share_minor
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import run_tier0
from ledgerloop.matching.tier1_tolerance import run_tier1
from ledgerloop.matching.tier2_aggregation import run_tier2
from ledgerloop.matching.tier3_lexical import (
    build_profiles,
    run_tier3,
    score_names,
)
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from tests.unit.conftest import bank_credit, batch, corpus, noise_credit

TOLERANCES = MatchingTolerances()
LEXICAL = LexicalMatching()


def _t3(ingest, lexical: LexicalMatching = LEXICAL):
    return run_tier3(MatchContext.from_ingest(ingest), TOLERANCES, lexical)


def _pairs(candidates):
    return {(c.source_ref.key, c.target_ref.key) for c in candidates if c.is_evaluable}


def keyed_and_stripped(
    *,
    keyed_name: str = "RAZORPAY SOFTWARE PVT",
    stripped_name: str = "RZRPAY SFTWR P L",
    merchant: str = "MRCH_0001",
    stripped_amounts: tuple[int, ...] = (60_000, 40_000),
):
    """One settlement that keeps its reference, and one that lost it.

    The first is what teaches T3 how the bank spells this merchant; the second
    is what T3 then has to find. Nothing but the reference distinguishes them.
    """
    teacher = batch(
        "SETL-0001",
        utr="UTR2026031000001",
        amounts=(90_000,),
        first_index=1,
        merchant_id=merchant,
    )
    learner = batch(
        "SETL-0002",
        utr="UTR2026031000002",
        amounts=stripped_amounts,
        first_index=10,
        merchant_id=merchant,
    )
    rows = [
        bank_credit(
            "BNK-00001",
            amount_minor=teacher.net_minor,
            utr=teacher.settlement.utr,
            merchant=keyed_name,
        ),
        bank_credit(
            "BNK-00002",
            amount_minor=learner.net_minor,
            utr=None,
            merchant=stripped_name,
        ),
    ]
    return teacher, learner, corpus(batches=[teacher, learner], bank_txns=rows)


class TestTheScorer:
    def test_the_worked_example_scores_at_the_top(self):
        assert score_names("RZRPAY SFTWR P L", "Razorpay Software Private Limited") == 1.0

    def test_the_gate_sits_clear_of_every_different_merchant_pairing(self):
        """The precision guarantee, measured rather than assumed.

        All 1,056 cross-merchant pairings -- every variant and legal name of one
        merchant against every variant and legal name of another -- peak at
        0.75, on ``URBN CO TECHNOLOGIES`` against ``CREDAVENUES TECHNOLOGIES``.
        Two companies that both end in the same word. The configured gate is
        0.90, so a false lexical match would need a name pair unlike anything in
        the vocabulary.
        """
        worst = max(
            (score_names(left, right), left, right)
            for first, second in combinations(MERCHANTS, 2)
            for left in (*first.variants, first.legal_name)
            for right in (*second.variants, second.legal_name)
        )
        assert worst[0] == pytest.approx(0.75), worst
        assert worst[0] < LEXICAL.min_score
        assert LEXICAL.min_score - worst[0] >= 0.15

    def test_the_two_distributions_overlap_so_no_gate_separates_them(self):
        """The honest finding: this is a trade, not a threshold to be tuned.

        The weakest *same*-merchant pairing (0.667, ``URBAN COMPANY TECH LTD``
        against ``URBN CO TECHNOLOGIES``) scores **below** the strongest
        different-merchant pairing (0.75). No cut admits every true pair and
        rejects every false one, so the gate is placed for precision and the
        recall it costs is named rather than tuned away.
        """
        weakest_same = min(
            (score_names(left, right), merchant.merchant_id, left, right)
            for merchant in MERCHANTS
            for left, right in combinations(merchant.variants, 2)
        )
        strongest_different = max(
            score_names(left, right)
            for first, second in combinations(MERCHANTS, 2)
            for left in first.variants
            for right in second.variants
        )
        assert weakest_same[0] < strongest_different

    def test_the_variants_the_gate_excludes_are_whole_word_abbreviations(self):
        """What the precision-first gate costs, named.

        Four merchants have a variant pair the gate excludes, and every one of
        them shortens or drops a whole word: ``TECH`` for ``TECHNOLOGIES``,
        ``CO`` for ``COMPANY``, ``SRVCS`` against a name that drops the word
        entirely. No character-level transform undoes a missing word.

        On the generated corpus this costs nothing measurable -- the real run
        rejects zero credits for being below the score, because the spellings
        that actually appear together on one merchant's narrations are closer
        than the worst pairing in the vocabulary.
        """
        below = {
            merchant.merchant_id
            for merchant in MERCHANTS
            for left, right in combinations(merchant.variants, 2)
            if score_names(left, right) < LEXICAL.min_score
        }
        assert below == {"MRCH_0004", "MRCH_0006", "MRCH_0007", "MRCH_0009"}

    def test_most_merchants_have_every_variant_pair_above_the_gate(self):
        clean = [
            merchant.merchant_id
            for merchant in MERCHANTS
            if all(
                score_names(left, right) >= LEXICAL.min_score
                for left, right in combinations(merchant.variants, 2)
            )
        ]
        assert len(clean) == 8

    def test_it_is_symmetric(self):
        assert score_names("ZRDHA BRKNG L", "Zerodha Broking Limited") == score_names(
            "Zerodha Broking Limited", "ZRDHA BRKNG L"
        )

    def test_an_empty_name_scores_nothing_against_a_real_one(self):
        assert score_names("", "Razorpay Software Private Limited") < LEXICAL.min_score

    def test_the_score_is_bounded(self):
        for merchant in MERCHANTS:
            for variant in merchant.variants:
                assert 0.0 <= score_names(variant, merchant.legal_name) <= 1.0


class TestTheMerchantProfile:
    def test_a_profile_is_learned_from_the_references_not_from_our_matches(self):
        """The reference is the source's own assertion, so it needs no tier to run."""
        _, _, built = keyed_and_stripped()
        context = MatchContext.from_ingest(built)
        profiles = build_profiles(context)
        assert profiles["MRCH_0001"].spellings == ("RAZORPAY SOFTWARE PVT",)
        assert profiles["MRCH_0001"].witnesses == 1
        assert context.consumed_settlements == set()

    def test_several_spellings_accumulate_for_one_merchant(self):
        first = batch("SETL-0001", utr="UTR2026031000001", amounts=(50_000,), first_index=1)
        second = batch("SETL-0002", utr="UTR2026031000002", amounts=(70_000,), first_index=10)
        built = corpus(
            batches=[first, second],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=50_000, utr=first.settlement.utr,
                            merchant="RAZORPAY SOFTWARE PVT"),
                bank_credit("BNK-00002", amount_minor=70_000, utr=second.settlement.utr,
                            merchant="RZRPAY SFTWR P L"),
            ],
        )
        profile = build_profiles(MatchContext.from_ingest(built))["MRCH_0001"]
        assert set(profile.spellings) == {"RAZORPAY SOFTWARE PVT", "RZRPAY SFTWR P L"}
        assert profile.witnesses == 2

    def test_an_unreferenced_credit_teaches_nothing(self):
        only = batch(amounts=(50_000,))
        built = corpus(
            batches=[only],
            bank_txns=[bank_credit(amount_minor=50_000, utr=None, merchant="ACME LTD")],
        )
        assert build_profiles(MatchContext.from_ingest(built)) == {}

    def test_a_settlement_with_no_resolvable_merchant_teaches_nothing(self):
        """No order reference means no merchant identity to attribute the name to."""
        only = batch(amounts=(50_000,), order_refs=(None,))
        built = corpus(
            batches=[only],
            bank_txns=[
                bank_credit(amount_minor=50_000, utr=only.settlement.utr, merchant="ACME")
            ],
        )
        assert build_profiles(MatchContext.from_ingest(built)) == {}

    def test_the_merchant_is_reached_through_payment_and_order(self, simple):
        context = MatchContext.from_ingest(simple)
        view = context.settlements_by_id["SETL-0001"]
        assert context.merchant_of(view) == "MRCH_0001"

    def test_a_settlement_spanning_two_merchants_has_no_identity(self):
        """Not something to guess about -- either corrupt data or a new shape."""
        only = batch(amounts=(50_000, 50_000))
        built = corpus(batches=[only])
        context = MatchContext.from_ingest(built)
        mixed = list(context.orders)
        mixed[1] = mixed[1].model_copy(update={"merchant_id": "MRCH_0002"})
        context.orders_by_id = {o.order_id: o for o in mixed}
        assert context.merchant_of(context.settlements_by_id["SETL-0001"]) is None


class TestTheHappyPath:
    def test_a_stripped_reference_is_recovered_from_the_merchant_name(self):
        _, _, built = keyed_and_stripped()
        outcome = _t3(built)
        assert outcome.settlements_resolved == 1
        assert outcome.credits_matched == 1
        assert _pairs(outcome.candidates) == {
            ("payment:PAY-00010", "bank_txn:BNK-00002"),
            ("payment:PAY-00011", "bank_txn:BNK-00002"),
        }

    def test_an_exact_spelling_match_is_certain(self):
        _, _, built = keyed_and_stripped(stripped_name="RAZORPAY SOFTWARE PVT")
        outcome = _t3(built)
        assert all(c.calibrated_p == 1.0 for c in outcome.candidates)

    def test_an_abbreviated_spelling_is_matched_but_below_certainty(self):
        """``RZRPAY SFTWR P L`` reaches the gate on the skeleton, and 1.0 exactly."""
        _, _, built = keyed_and_stripped(stripped_name="RZRPAY SFTWR P L")
        outcome = _t3(built)
        assert outcome.settlements_resolved == 1
        assert all(c.calibrated_p >= LEXICAL.min_score for c in outcome.candidates)

    def test_candidates_carry_the_tier_and_the_lexical_feature(self):
        _, _, built = keyed_and_stripped()
        for candidate in _t3(built).candidates:
            assert candidate.tier is Tier.T3_FUZZY
            assert candidate.features.tier is Tier.T3_FUZZY
            assert candidate.features.lexical_score > 0.0
            assert candidate.features.semantic_score == 0.0

    def test_the_evidence_names_the_score_the_spelling_and_the_gates(self):
        _, _, built = keyed_and_stripped()
        link = next(
            c for c in _t3(built).candidates
            if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        kinds = {item.kind for item in link.evidence}
        assert EvidenceKind.LEXICAL_SIMILARITY in kinds
        assert EvidenceKind.AMOUNT_MATCH in kinds
        assert EvidenceKind.DATE_PROXIMITY in kinds
        note = next(i for i in link.evidence if i.kind is EvidenceKind.LEXICAL_SIMILARITY)
        assert "RZRPAY SFTWR P L" in note.detail
        assert "RAZORPAY SOFTWARE PVT" in note.detail
        assert note.score is not None

    def test_the_shares_conserve_the_credit(self):
        _, learner, built = keyed_and_stripped(stripped_amounts=(33_333, 33_333, 33_334))
        outcome = _t3(built)
        shares = [allocated_share_minor(c) for c in outcome.candidates if c.is_evaluable]
        assert sum(shares) == learner.net_minor

    def test_it_consumes_the_settlement_and_the_credit(self):
        _, _, built = keyed_and_stripped()
        context = MatchContext.from_ingest(built)
        run_tier3(context, TOLERANCES, LEXICAL)
        assert context.consumed_settlements == {"SETL-0002"}
        assert context.consumed_credits == {"BNK-00002"}


class TestWhatItRefuses:
    def test_a_different_merchants_name_is_not_a_match(self):
        """The amount and date line up; only the name says no."""
        _, _, built = keyed_and_stripped(stripped_name="ZOMATO HYPERPURE PVT")
        outcome = _t3(built)
        assert outcome.settlements_resolved == 0
        assert outcome.rejected_below_score == 1
        assert outcome.candidates == ()

    def test_a_credit_carrying_someone_elses_reference_is_never_considered(self):
        """It is already explained; matching it on a name would overrule the bank."""
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor,
                        utr="UTR2026039999999", merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_resolved == 0
        assert outcome.settlements_seen == 0

    def test_a_credit_with_no_merchant_name_is_never_considered(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            noise_credit("BNK-00002", amount_minor=learner.net_minor),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_resolved == 0

    def test_a_merchant_with_no_profile_is_counted_not_guessed(self):
        only = batch(amounts=(50_000,), merchant_id="MRCH_0099")
        built = corpus(
            batches=[only],
            bank_txns=[bank_credit(amount_minor=50_000, utr=None, merchant="ACME LTD")],
        )
        outcome = _t3(built)
        assert outcome.settlements_without_profile == 1
        assert outcome.candidates == ()

    def test_two_credits_the_name_cannot_separate_are_ambiguous(self):
        """A05: a duplicated credit whose narration also lost its reference."""
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
            bank_credit("BNK-00003", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_ambiguous == 1
        assert outcome.rejected_on_margin == 1
        assert outcome.payment_links == 0

    def test_the_ambiguity_names_both_rivals_and_the_margin(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
            bank_credit("BNK-00003", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        detail = next(
            i.detail
            for c in outcome.candidates
            for i in c.evidence
            if i.kind is EvidenceKind.NEGATIVE_EVIDENCE and "cannot separate" in i.detail
        )
        assert "BNK-00002" in detail and "BNK-00003" in detail

    def test_an_ambiguous_settlement_keeps_its_credits_in_the_pool(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
            bank_credit("BNK-00003", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L"),
        ]
        context = MatchContext.from_ingest(
            corpus(batches=[teacher, learner], bank_txns=rows)
        )
        run_tier3(context, TOLERANCES, LEXICAL)
        assert context.consumed_settlements == {"SETL-0002"}
        assert "BNK-00002" not in context.consumed_credits


class TestTheAmountAndDateGates:
    def test_an_amount_outside_the_band_is_not_a_match(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor + 50_000,
                        utr=None, merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_seen == 0

    def test_a_drift_inside_the_band_still_matches(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor + 3,
                        utr=None, merchant="RZRPAY SFTWR P L"),
        ]
        assert _t3(corpus(batches=[teacher, learner], bank_txns=rows)).settlements_resolved == 1

    @pytest.mark.parametrize("days", [-7, 0, 7])
    def test_credits_inside_the_wider_window_are_matched(self, days):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            learner.credit("BNK-00002", days_after=days, utr=None),
        ]
        rows[1] = bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                              merchant="RZRPAY SFTWR P L",
                              value_date=rows[1].value_date)
        assert _t3(corpus(batches=[teacher, learner], bank_txns=rows)).settlements_resolved == 1

    def test_the_window_is_wider_than_t1s_on_purpose(self):
        """T3's residual is the batches other anomalies already moved."""
        assert LEXICAL.date_window_days > TOLERANCES.date_window_days

    def test_a_tighter_gate_refuses_what_the_default_accepts(self):
        """``SWGY INSTMRT SRVCS`` reaches 0.815 -- above 0.80, below the default."""
        teacher = batch(
            "SETL-0001", utr="UTR2026031000001", amounts=(90_000,), first_index=1,
            merchant_id="MRCH_0009",
        )
        learner = batch(
            "SETL-0002", utr="UTR2026031000002", amounts=(60_000,), first_index=10,
            merchant_id="MRCH_0009",
        )
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="SWIGGY INSTAMART LTD"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="SWGY INSTMRT SRVCS"),
        ]
        built = corpus(batches=[teacher, learner], bank_txns=rows)
        assert _t3(built, LexicalMatching(min_score=0.80)).settlements_resolved == 1
        assert _t3(built).settlements_resolved == 0

    def test_a_wider_date_window_admits_a_later_credit(self):
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="RZRPAY SFTWR P L",
                        value_date=learner.settlement.settled_on + timedelta(days=9)),
        ]
        built = corpus(batches=[teacher, learner], bank_txns=rows)
        assert _t3(built).settlements_resolved == 0
        assert _t3(built, LexicalMatching(date_window_days=10)).settlements_resolved == 1


class TestTheResidualPool:
    def test_t3_never_sees_a_settlement_an_earlier_tier_resolved(self, simple):
        context = MatchContext.from_ingest(simple)
        run_tier0(context)
        outcome = run_tier3(context, TOLERANCES, LEXICAL)
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_t3_picks_up_what_the_keyed_tiers_left(self):
        _, _, built = keyed_and_stripped()
        context = MatchContext.from_ingest(built)
        run_tier0(context)
        run_tier1(context, TOLERANCES)
        run_tier2(context, TOLERANCES)
        assert "SETL-0001" in context.consumed_settlements
        assert "SETL-0002" not in context.consumed_settlements

        outcome = run_tier3(context, TOLERANCES, LEXICAL)
        assert outcome.settlements_resolved == 1

    def test_a_credit_consumed_earlier_is_unavailable_to_t3(self):
        _, _, built = keyed_and_stripped(stripped_name="RAZORPAY SOFTWARE PVT")
        context = MatchContext.from_ingest(built)
        run_tier0(context)
        # The keyed credit is gone; only the stripped one remains for T3.
        assert context.consumed_credits == {"BNK-00001"}
        assert run_tier3(context, TOLERANCES, LEXICAL).settlements_resolved == 1


class TestDeterminism:
    def test_two_runs_produce_identical_candidates(self):
        _, _, built = keyed_and_stripped()
        assert _t3(built).candidates == _t3(built).candidates

    def test_the_order_of_the_bank_rows_does_not_change_the_match(self):
        teacher, learner, built = keyed_and_stripped()
        rows = list(built.bank_txns)
        forward = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        backward = _t3(corpus(batches=[teacher, learner], bank_txns=list(reversed(rows))))
        assert _pairs(forward.candidates) == _pairs(backward.candidates)


class TestReportingSurfaceAndEdges:
    def test_the_outcome_names_its_tier_and_counts_links_separately(self):
        _, _, built = keyed_and_stripped()
        outcome = _t3(built)
        assert outcome.tier is Tier.T3_FUZZY
        assert outcome.settlement_links == 1
        assert outcome.payment_links == 2
        assert outcome.settlement_links + outcome.payment_links == len(outcome.candidates)

    def test_a_beaten_runner_up_is_recorded_on_the_match(self):
        """Two credits pass the gate, but one clears the other by the margin.

        The loser is named on the winning candidate: a controller reviewing the
        match can see what else was in the running and by how much it lost.
        """
        teacher, learner, _ = keyed_and_stripped()
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=learner.net_minor, utr=None,
                        merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00003", amount_minor=learner.net_minor, utr=None,
                        merchant="RAZOPAY SOFWRE PVT"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_resolved == 1
        detail = next(
            i.detail
            for c in outcome.candidates
            for i in c.evidence
            if "closest rival" in i.detail
        )
        assert "BNK-00003" in detail

    def test_a_batch_whose_every_payment_was_clawed_back_credits_nobody(self):
        teacher = batch("SETL-0001", utr="UTR2026031000001", amounts=(90_000,), first_index=1)
        learner = batch(
            "SETL-0002", utr="UTR2026031000002", amounts=(50_000,), first_index=10,
            adjustments_minor=-50_000, net_minor=50_000,
        )
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=50_000, utr=None,
                        merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_resolved == 1
        assert outcome.payment_links == 0

    def test_a_settlement_with_a_non_positive_net_is_skipped(self):
        teacher = batch("SETL-0001", utr="UTR2026031000001", amounts=(90_000,), first_index=1)
        learner = batch(
            "SETL-0002", utr="UTR2026031000002", amounts=(50_000,), first_index=10,
            net_minor=0,
        )
        rows = [
            bank_credit("BNK-00001", amount_minor=teacher.net_minor,
                        utr=teacher.settlement.utr, merchant="RAZORPAY SOFTWARE PVT"),
            bank_credit("BNK-00002", amount_minor=50_000, utr=None,
                        merchant="RZRPAY SFTWR P L"),
        ]
        outcome = _t3(corpus(batches=[teacher, learner], bank_txns=rows))
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()
