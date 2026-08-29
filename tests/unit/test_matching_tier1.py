"""T1 -- tolerance bands and date windows, and the boundaries of both.

Two things are being pinned here.

**The bands come from the configuration.** Every threshold T1 applies is a field
on :class:`~ledgerloop.config.MatchingTolerances`, so the tests drive it by
changing the config rather than by asserting a literal. A test that hard-coded
₹1 would still pass if the implementation hard-coded it too, and the point of
the config is that the value travels in ``RunConfig.config_hash`` alongside the
number it produced.

**T1 cannot override T0.** Not by convention -- structurally, because T0
consumes what it ruled on and T1 iterates the residual. The tests check the
pool, not just the output.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import MatchingTolerances
from ledgerloop.matching.bank_leg import resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import T0_RULE
from ledgerloop.matching.tier1_tolerance import rule_for, run_tier1
from ledgerloop.models.enums import EvidenceKind, Tier
from ledgerloop.money import tolerance_minor
from tests.unit.conftest import batch, corpus

DEFAULTS = MatchingTolerances()


def _t1(ingest, tolerances: MatchingTolerances = DEFAULTS):
    """T1 alone, on a fresh pool. Used to test the rule in isolation."""
    return run_tier1(MatchContext.from_ingest(ingest), tolerances)


def _ladder(ingest, tolerances: MatchingTolerances = DEFAULTS):
    """T0 then T1, sharing one pool. Used to test the interaction."""
    context = MatchContext.from_ingest(ingest)
    first = resolve_bank_leg(context, T0_RULE)
    second = run_tier1(context, tolerances)
    return context, first, second


class TestTheRuleComesFromConfiguration:
    def test_it_reads_every_field_from_the_tolerances(self):
        tolerances = MatchingTolerances(
            amount_floor_minor=250, amount_bps=25, date_window_days=7
        )
        rule = rule_for(tolerances)
        assert rule.tier is Tier.T1_TOLERANCE
        assert rule.amount_floor_minor == 250
        assert rule.amount_bps == 25
        assert rule.date_window_days == 7

    def test_the_band_is_the_money_module_band(self):
        """No second implementation of the tolerance arithmetic."""
        rule = rule_for(DEFAULTS)
        for amount in (100, 100_000, 55_555_555):
            assert rule.band_for(amount) == tolerance_minor(
                amount, floor_minor=DEFAULTS.amount_floor_minor, bps=DEFAULTS.amount_bps
            )

    def test_t0_is_the_same_rule_with_a_zero_width_band(self):
        assert T0_RULE.band_for(10_000_000) == 0
        assert T0_RULE.date_window_days is None


class TestAmountBoundaries:
    """The floor is ₹1 (100 paise) by default, so a small net is judged on it."""

    @pytest.mark.parametrize("delta", [-100, -99, -1, 1, 99, 100])
    def test_deltas_inside_the_floor_are_matched(self, delta):
        only = batch(amounts=(60_000, 40_000))
        assert only.net_minor == 100_000
        assert rule_for(DEFAULTS).band_for(only.net_minor) == 500  # 0.5% dominates here
        outcome = _t1(corpus(batches=[only], bank_txns=[only.credit(delta_minor=delta)]))
        assert outcome.resolved_settlements == 1

    def test_the_band_is_inclusive_at_its_edge(self):
        only = batch(amounts=(60_000, 40_000))
        band = rule_for(DEFAULTS).band_for(only.net_minor)
        outcome = _t1(corpus(batches=[only], bank_txns=[only.credit(delta_minor=band)]))
        assert outcome.resolved_settlements == 1

    def test_one_paise_beyond_the_band_is_declined(self):
        only = batch(amounts=(60_000, 40_000))
        band = rule_for(DEFAULTS).band_for(only.net_minor)
        outcome = _t1(corpus(batches=[only], bank_txns=[only.credit(delta_minor=band + 1)]))
        assert outcome.resolved_settlements == 0
        assert outcome.undecided_settlements == 1

    def test_the_floor_applies_when_the_proportional_part_is_smaller(self):
        """A ₹10 payout: 0.5% is 5 paise, so the ₹1 floor is what carries."""
        only = batch(amounts=(1_000,))
        assert rule_for(DEFAULTS).band_for(only.net_minor) == 100
        outcome = _t1(corpus(batches=[only], bank_txns=[only.credit(delta_minor=100)]))
        assert outcome.resolved_settlements == 1

    def test_the_proportional_part_applies_when_it_is_larger(self):
        """A ₹1,00,000 payout: 0.5% is ₹500, far above the floor."""
        only = batch(amounts=(10_000_000,))
        assert rule_for(DEFAULTS).band_for(only.net_minor) == 50_000
        outcome = _t1(corpus(batches=[only], bank_txns=[only.credit(delta_minor=50_000)]))
        assert outcome.resolved_settlements == 1

    def test_a_tighter_configured_band_declines_what_the_default_accepts(self):
        only = batch(amounts=(60_000, 40_000))
        credit = only.credit(delta_minor=400)
        assert _t1(corpus(batches=[only], bank_txns=[credit])).resolved_settlements == 1

        tight = MatchingTolerances(amount_floor_minor=1, amount_bps=1)
        assert (
            _t1(corpus(batches=[only], bank_txns=[credit]), tight).resolved_settlements == 0
        )

    def test_a_rounding_drift_of_three_paise_is_comfortably_inside(self):
        """A02, the class T1 exists for."""
        only = batch(amounts=(60_000, 40_000))
        for drift in (-3, -2, -1, 1, 2, 3):
            outcome = _t1(
                corpus(batches=[only], bank_txns=[only.credit(delta_minor=drift)])
            )
            assert outcome.resolved_settlements == 1, drift


class TestDateBoundaries:
    @pytest.mark.parametrize("days", [-3, -1, 0, 1, 2, 3])
    def test_credits_inside_the_window_are_matched(self, days):
        only = batch()
        outcome = _t1(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=days)])
        )
        assert outcome.resolved_settlements == 1

    @pytest.mark.parametrize("days", [-4, 4, 9])
    def test_credits_outside_the_window_are_declined(self, days):
        only = batch()
        outcome = _t1(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=days)])
        )
        assert outcome.resolved_settlements == 0
        assert outcome.undecided_settlements == 1

    def test_the_window_is_configurable(self):
        only = batch()
        far = corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=8)])
        assert _t1(far).resolved_settlements == 0
        wide = MatchingTolerances(date_window_days=9)
        assert _t1(far, wide).resolved_settlements == 1

    def test_a_zero_width_window_demands_the_settlement_date(self):
        only = batch()
        exact_day = MatchingTolerances(date_window_days=0)
        same = corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=0)])
        next_day = corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=1)])
        assert _t1(same, exact_day).resolved_settlements == 1
        assert _t1(next_day, exact_day).resolved_settlements == 0

    def test_the_date_evidence_is_recorded(self):
        only = batch()
        outcome = _t1(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=2)])
        )
        kinds = {i.kind for c in outcome.candidates for i in c.evidence}
        assert EvidenceKind.DATE_PROXIMITY in kinds

    def test_t1_records_the_signed_day_gap_as_a_feature(self):
        only = batch()
        outcome = _t1(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, days_after=-2)])
        )
        assert outcome.candidates[0].features.date_delta_days == -2


class TestT1CannotOverrideT0:
    def test_a_settlement_t0_resolved_is_never_revisited(self):
        only = batch()
        _, first, second = _ladder(
            corpus(batches=[only], bank_txns=[only.credit()])
        )
        assert first.resolved_settlements == 1
        assert second.candidates == ()
        assert second.resolved_settlements == 0

    def test_a_settlement_t0_contested_is_never_revisited(self):
        """A05 has been decided -- as contested. A looser rule may not re-open it."""
        only = batch()
        _, first, second = _ladder(
            corpus(
                batches=[only],
                # Same day: unorderable, so the duplicate-posting pass declines
                # and T0's contest -- the thing under test -- is what happens.
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
            )
        )
        assert first.contested_settlements == 1
        assert second.candidates == ()

    def test_a_credit_t0_claimed_cannot_be_reused_by_t1(self):
        first_batch = batch("SETL-0001", utr="UTR2026031000001", amounts=(100_000,), first_index=1)
        # A second settlement whose net is one paise off the same credit, and
        # which publishes the same key. T0 takes the exact one; T1 must not
        # then hand the consumed credit to the other.
        second_batch = batch(
            "SETL-0002", utr="UTR2026031000001", amounts=(100_001,), first_index=10
        )
        context, first, second = _ladder(
            corpus(
                batches=[first_batch, second_batch],
                bank_txns=[first_batch.credit("BNK-00001")],
            )
        )
        # Both settlements publish one key, so T0 declares a collision.
        assert first.key_collisions == 1
        assert context.consumed_settlements == {"SETL-0001", "SETL-0002"}
        assert second.candidates == ()

    def test_t1_picks_up_exactly_what_t0_left_undecided(self):
        clean = batch("SETL-0001", utr="UTR2026031000001", amounts=(100_000,), first_index=1)
        drifted = batch("SETL-0002", utr="UTR2026031000002", amounts=(100_000,), first_index=10)
        _, first, second = _ladder(
            corpus(
                batches=[clean, drifted],
                bank_txns=[clean.credit("BNK-00001"), drifted.credit("BNK-00002", delta_minor=2)],
            )
        )
        assert first.resolved_settlements == 1
        assert first.undecided_settlements == 1
        assert second.resolved_settlements == 1
        assert {c.tier for c in second.candidates} == {Tier.T1_TOLERANCE}


class TestWhatT1DeclinesOnPurpose:
    def test_a_split_payout_tranche_is_far_outside_the_band(self):
        """A09 is T2's problem. Neither tranche is within 0.5% of the whole net."""
        only = batch(amounts=(60_000, 40_000))
        first_tranche = only.credit("BNK-00001", delta_minor=-40_000)
        second_tranche = only.credit("BNK-00002", delta_minor=-60_000, days_after=2)
        outcome = _t1(corpus(batches=[only], bank_txns=[first_tranche, second_tranche]))
        assert outcome.resolved_settlements == 0
        assert outcome.undecided_settlements == 1

    def test_two_credits_both_inside_the_band_are_contested(self):
        """A02 on top of A05. T1 reaches them and still refuses to choose."""
        only = batch()
        outcome = _t1(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001", delta_minor=1),
                    only.credit("BNK-00002", delta_minor=1),
                ],
            )
        )
        assert outcome.contested_settlements == 1
        assert outcome.payment_links == 0

    def test_a_missing_key_is_still_a_missing_key(self):
        """T1 loosens the amount, not the requirement for a reference."""
        only = batch()
        outcome = _t1(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=2, utr=None)])
        )
        assert outcome.candidates == ()
