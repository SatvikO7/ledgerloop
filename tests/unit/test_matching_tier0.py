"""T0 -- exact keys, and the cases where an exact-looking key is not enough.

The bank leg is the tier's real work, and the tests that matter are the ones
where a naive exact join would say yes: a duplicated credit, a colliding key, a
debit carrying the right reference. Getting those wrong is how a matcher
produces confident false positives, which is the most expensive failure in the
system and the one baseline B0 makes on this corpus.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from ledgerloop.matching.bank_leg import allocated_share_minor, resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import T0_RULE, resolve_order_leg, run_tier0
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from tests.unit.conftest import bank_credit, batch, corpus, debit_row, noise_credit


def _t0(ingest):
    return resolve_bank_leg(MatchContext.from_ingest(ingest), T0_RULE)


def _pairs(candidates):
    return {(c.source_ref.key, c.target_ref.key) for c in candidates if c.is_evaluable}


class TestTheHappyPath:
    def test_an_exact_key_and_an_exact_amount_resolve(self, simple):
        outcome = _t0(simple)
        assert outcome.resolved_settlements == 1
        assert outcome.contested_settlements == 0
        assert outcome.settlement_links == 1
        assert outcome.payment_links == 2

    def test_every_payment_in_the_batch_is_credited(self, simple):
        outcome = _t0(simple)
        assert _pairs(outcome.candidates) == {
            ("payment:PAY-00001", "bank_txn:BNK-00001"),
            ("payment:PAY-00002", "bank_txn:BNK-00001"),
        }

    def test_the_credit_is_allocated_and_conserved_exactly(self):
        """No paise created by the matcher, none destroyed."""
        only = batch(amounts=(33_333, 33_333, 33_334))
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        shares = [
            allocated_share_minor(c) for c in outcome.candidates if c.is_evaluable
        ]
        assert sum(shares) == only.net_minor
        assert all(isinstance(share, int) for share in shares)

    def test_candidates_carry_the_tier_and_a_certain_probability(self, simple):
        for candidate in _t0(simple).candidates:
            assert candidate.tier is Tier.T0_EXACT
            assert candidate.features.tier is Tier.T0_EXACT
            assert candidate.calibrated_p == 1.0
            assert candidate.arithmetic_verified

    def test_the_evidence_names_the_key_the_amount_and_the_arithmetic(self, simple):
        settlement_link = next(
            c for c in _t0(simple).candidates
            if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        kinds = {item.kind for item in settlement_link.evidence}
        assert EvidenceKind.EXACT_KEY in kinds
        assert EvidenceKind.AMOUNT_MATCH in kinds
        assert EvidenceKind.ARITHMETIC_CHECK in kinds
        assert all(item.refs for item in settlement_link.evidence)

    def test_t0_applies_no_date_constraint(self):
        """An exact UTR and an exact net is enough. A04 and A12 arrive late but intact."""
        only = batch()
        late = only.credit(days_after=30)
        outcome = _t0(corpus(batches=[only], bank_txns=[late]))
        assert outcome.resolved_settlements == 1
        assert outcome.candidates[0].features.date_delta_days == 30


class TestMissingKeys:
    def test_a_settlement_with_no_utr_is_skipped(self):
        only = batch(utr=None)
        outcome = _t0(corpus(batches=[only], bank_txns=[bank_credit(utr=None)]))
        assert outcome.candidates == ()
        assert outcome.settlements_without_key == 1
        assert outcome.resolved_settlements == 0

    def test_a_credit_with_no_utr_cannot_be_reached(self):
        """A07: the reference was stripped from the narration. T3's problem."""
        only = batch()
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit(utr=None)]))
        assert outcome.candidates == ()
        assert outcome.undecided_settlements == 1

    def test_a_settlement_whose_utr_appears_nowhere_stays_in_the_pool(self):
        only = batch()
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=[noise_credit()]))
        outcome = resolve_bank_leg(context, T0_RULE)
        assert outcome.undecided_settlements == 1
        assert context.consumed_settlements == set()


class TestRowsThatMustNotMatch:
    def test_a_debit_carrying_the_utr_is_never_a_candidate(self):
        """Only incoming money is a payout. A sign error here reverses a payment."""
        only = batch()
        outcome = _t0(
            corpus(batches=[only], bank_txns=[debit_row(amount_minor=only.net_minor)])
        )
        assert outcome.candidates == ()

    def test_a_noise_credit_of_the_right_amount_is_not_matched(self):
        """Amount alone is not evidence. The key is what identifies the payout."""
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[noise_credit(amount_minor=only.net_minor)],
            )
        )
        assert outcome.candidates == ()


class TestAmountIsPartOfTheKey:
    @pytest.mark.parametrize("delta", [-1, 1, -100, 5_000])
    def test_any_amount_difference_declines_at_t0(self, delta):
        only = batch()
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit(delta_minor=delta)]))
        assert outcome.candidates == ()
        assert outcome.undecided_settlements == 1

    def test_declining_leaves_the_settlement_for_the_next_tier(self):
        """Undecided is not decided. This is how A02 reaches T1."""
        only = batch()
        context = MatchContext.from_ingest(
            corpus(batches=[only], bank_txns=[only.credit(delta_minor=3)])
        )
        resolve_bank_leg(context, T0_RULE)
        assert [v.settlement_id for v in context.open_settlements()] == ["SETL-0001"]


class TestContestedKeys:
    """A05 DUPLICATE_CREDIT, and the rule that an exact key is not enough."""

    def test_two_identical_credits_are_contested_not_matched(self):
        only = batch()
        first = only.credit("BNK-00001")
        duplicate = only.credit("BNK-00002", days_after=2)
        outcome = _t0(corpus(batches=[only], bank_txns=[first, duplicate]))

        assert outcome.resolved_settlements == 0
        assert outcome.contested_settlements == 1
        assert outcome.settlement_links == 2

    def test_a_contested_settlement_asserts_no_payment_links(self):
        """The closure follows belief in the settlement edge, and there is none."""
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
            )
        )
        assert outcome.payment_links == 0

    def test_contested_candidates_split_the_probability(self):
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
            )
        )
        assert [c.calibrated_p for c in outcome.candidates] == [0.5, 0.5]

    def test_the_contest_is_recorded_as_negative_evidence_naming_the_rivals(self):
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
            )
        )
        negatives = [
            item
            for c in outcome.candidates
            for item in c.evidence
            if item.kind is EvidenceKind.NEGATIVE_EVIDENCE
        ]
        assert negatives
        assert "BNK-00002" in negatives[0].detail
        assert "paid once" in negatives[0].detail

    def test_a_contested_settlement_leaves_the_pool_but_its_credits_do_not(self):
        """It has been decided -- as contested. Neither credit was claimed."""
        only = batch()
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
            )
        )
        resolve_bank_leg(context, T0_RULE)
        assert context.consumed_settlements == {"SETL-0001"}
        assert context.consumed_credits == set()

    def test_three_contenders_split_three_ways(self):
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001"),
                    only.credit("BNK-00002", days_after=2),
                    only.credit("BNK-00003", days_after=3),
                ],
            )
        )
        assert outcome.settlement_links == 3
        assert all(c.calibrated_p == pytest.approx(1 / 3) for c in outcome.candidates)


class TestKeyCollisions:
    """Two settlements publishing one UTR. The key identifies neither."""

    def test_a_shared_utr_contests_every_settlement_that_publishes_it(self):
        shared = "UTR2026031099999"
        first = batch("SETL-0001", utr=shared, amounts=(50_000,), first_index=1)
        second = batch("SETL-0002", utr=shared, amounts=(50_000,), first_index=10)
        outcome = _t0(
            corpus(
                batches=[first, second],
                bank_txns=[bank_credit(amount_minor=50_000, utr=shared)],
            )
        )
        assert outcome.key_collisions == 1
        assert outcome.contested_settlements == 2
        assert outcome.resolved_settlements == 0
        assert outcome.payment_links == 0

    def test_the_whole_colliding_group_leaves_the_pool_together(self):
        """Otherwise the survivor looks unambiguous on the next iteration."""
        shared = "UTR2026031099999"
        first = batch("SETL-0001", utr=shared, amounts=(50_000,), first_index=1)
        second = batch("SETL-0002", utr=shared, amounts=(50_000,), first_index=10)
        context = MatchContext.from_ingest(
            corpus(
                batches=[first, second],
                bank_txns=[bank_credit(amount_minor=50_000, utr=shared)],
            )
        )
        resolve_bank_leg(context, T0_RULE)
        assert context.consumed_settlements == {"SETL-0001", "SETL-0002"}

    def test_a_collision_is_explained_in_the_evidence(self):
        shared = "UTR2026031099999"
        first = batch("SETL-0001", utr=shared, amounts=(50_000,), first_index=1)
        second = batch("SETL-0002", utr=shared, amounts=(50_000,), first_index=10)
        outcome = _t0(
            corpus(
                batches=[first, second],
                bank_txns=[bank_credit(amount_minor=50_000, utr=shared)],
            )
        )
        details = [
            item.detail
            for c in outcome.candidates
            for item in c.evidence
            if item.kind is EvidenceKind.NEGATIVE_EVIDENCE
        ]
        assert details
        assert "2 settlements" in details[0]


class TestExclusivity:
    def test_a_credit_claimed_by_one_settlement_cannot_be_claimed_again(self):
        shared_amount = 100_000
        first = batch(
            "SETL-0001", utr="UTR2026031000001", amounts=(shared_amount,), first_index=1
        )
        second = batch(
            "SETL-0002", utr="UTR2026031000002", amounts=(shared_amount + 1,), first_index=10
        )
        # One credit whose narration carries both keys is impossible; instead
        # give each settlement its own credit and check nothing crosses over.
        outcome = _t0(
            corpus(
                batches=[first, second],
                bank_txns=[
                    first.credit("BNK-00001"),
                    second.credit("BNK-00002"),
                ],
            )
        )
        assert outcome.resolved_settlements == 2
        assert _pairs(outcome.candidates) == {
            ("payment:PAY-00001", "bank_txn:BNK-00001"),
            ("payment:PAY-00010", "bank_txn:BNK-00002"),
        }


class TestChargebackExclusion:
    """A08: a payment nested in the batch whose money never reached the bank."""

    def test_the_netted_off_payment_is_excluded_from_the_closure(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert _pairs(outcome.candidates) == {("payment:PAY-00001", "bank_txn:BNK-00001")}

    def test_the_remaining_shares_still_sum_to_the_credit(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        shares = [allocated_share_minor(c) for c in outcome.candidates if c.is_evaluable]
        assert sum(shares) == only.net_minor

    def test_the_exclusion_is_explained_on_every_surviving_link(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        link = next(c for c in outcome.candidates if c.is_evaluable)
        details = [item.detail for item in link.evidence]
        assert any("netted off" in d and "PAY-00002" in d for d in details)

    def test_a_clawback_matching_no_payment_excludes_nothing(self):
        """A06: the refund belongs to an earlier batch, so everything here arrived."""
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-7_777)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.payment_links == 2
        assert all(c.arithmetic_verified for c in outcome.candidates)

    def test_an_unattributable_clawback_is_matched_but_not_verified(self):
        """Two identical payments, one clawed back. Which one is not knowable."""
        only = batch(amounts=(40_000, 40_000), adjustments_minor=-40_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        payment_links = [c for c in outcome.candidates if c.is_evaluable]
        assert len(payment_links) == 2
        assert not any(c.arithmetic_verified for c in payment_links)

    def test_a_positive_adjustment_excludes_nothing(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=5_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.payment_links == 2


class TestTheArithmeticGate:
    def test_a_gross_that_disagrees_with_its_payments_is_not_verified(self):
        only = batch(amounts=(60_000, 40_000), gross_minor=999_999)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.resolved_settlements == 1
        assert not any(c.arithmetic_verified for c in outcome.candidates)

    def test_the_failure_is_stated_in_the_evidence(self):
        only = batch(amounts=(60_000, 40_000), gross_minor=999_999)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        settlement_link = next(
            c for c in outcome.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        detail = next(
            i.detail for i in settlement_link.evidence if i.kind is EvidenceKind.ARITHMETIC_CHECK
        )
        assert "does NOT equal" in detail

    def test_a03_fee_tax_mismatch_still_matches(self):
        """The settlement's internal arithmetic is evidence, never a gate.

        A03 breaks ``net == gross - fee - tax + adjustments`` on purpose, and
        the bank pays the declared net. Enforcing the identity would delete the
        anomaly the whole three-way structure exists to surface.
        """
        only = batch(amounts=(60_000, 40_000), fee_minor=2_000, net_minor=90_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.resolved_settlements == 1
        assert only.settlement.net_delta_minor != 0
        assert all(c.arithmetic_verified for c in outcome.candidates)


class TestTheOrderLeg:
    def test_a_recovered_reference_joins_its_order(self, simple):
        outcome = resolve_order_leg(MatchContext.from_ingest(simple))
        assert outcome.matched == 2
        assert {c.link_type for c in outcome.candidates} == {LinkType.ORDER_PAID_BY}

    def test_the_order_leg_is_never_scored(self, simple):
        """ARCHITECTURE.md §2 restricts the metrics to PAYMENT_CREDITED_AS."""
        outcome = resolve_order_leg(MatchContext.from_ingest(simple))
        assert not any(c.is_evaluable for c in outcome.candidates)

    def test_a_null_reference_is_counted_not_guessed(self):
        only = batch(order_refs=(None, "ORD-2026-000002"))
        outcome = resolve_order_leg(MatchContext.from_ingest(corpus(batches=[only])))
        assert outcome.matched == 1
        assert outcome.missing_reference == 1

    def test_a_reference_naming_no_known_order_is_declined(self):
        only = batch(order_refs=("ORD-2026-999999", "ORD-2026-000002"))
        outcome = resolve_order_leg(MatchContext.from_ingest(corpus(batches=[only])))
        assert outcome.matched == 1
        assert outcome.unknown_reference == 1

    def test_an_exact_key_that_disagrees_on_money_is_declined(self):
        """A reference that normalises onto the wrong order is not an exact match."""
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only])
        # Point the second payment at the first payment's order, which was
        # booked for a different amount.
        payments = list(built.payments)
        payments[1] = payments[1].model_copy(
            update={"order_ref_normalized": "ORD-2026-000001"}
        )
        wrong = replace(built, payments=tuple(payments))
        outcome = resolve_order_leg(MatchContext.from_ingest(wrong))
        assert outcome.matched == 1
        assert outcome.amount_disagreement == 1


class TestRunTier0:
    def test_it_runs_both_legs(self, simple):
        order_leg, bank_leg = run_tier0(MatchContext.from_ingest(simple))
        assert order_leg.matched == 2
        assert bank_leg.resolved_settlements == 1

    def test_the_order_leg_consumes_nothing(self):
        """It resolves a different pair of record types and cannot compete."""
        only = batch()
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=[]))
        resolve_order_leg(context)
        assert context.consumed_settlements == set()
        assert context.consumed_credits == set()


def test_a_settlement_with_no_payments_asserts_no_links():
    """A batch the parser stripped of every payment has nothing to credit."""
    only = batch(amounts=())
    empty = corpus(batches=[only], bank_txns=[bank_credit(amount_minor=0)])
    outcome = _t0(empty)
    assert outcome.payment_links == 0


def test_dates_far_apart_do_not_bother_t0():
    only = batch()
    early = only.credit(days_after=-40)
    outcome = _t0(corpus(batches=[only], bank_txns=[early]))
    assert outcome.resolved_settlements == 1
    assert outcome.candidates[0].features.date_delta_days == -40
    assert early.value_date == only.settlement.settled_on - timedelta(days=40)


class TestMutualUniqueness:
    """A match is safe only when it is unique from both sides.

    All three cases here were found by property tests rather than by
    inspection, and each is a composition of two anomalies. They are the
    reason an exact-looking key is not on its own sufficient.
    """

    def test_an_unkeyed_credit_of_the_same_amount_contests_the_keyed_one(self):
        """A05 + A07: the reference was stripped from the row that was the payout.

        The duplicate keeps the copied narration and is the only keyed row, so a
        one-sided uniqueness check matches the whole batch to the duplicate.
        """
        only = batch()
        payout = only.credit("BNK-00001", utr=None)  # A07 stripped its reference
        duplicate = only.credit("BNK-00002", days_after=1)
        outcome = _t0(corpus(batches=[only], bank_txns=[payout, duplicate]))
        assert outcome.resolved_settlements == 0
        assert outcome.contested_settlements == 1
        assert outcome.payment_links == 0

    def test_the_rival_is_named_in_the_evidence(self):
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001", utr=None), only.credit("BNK-00002")],
            )
        )
        details = [
            item.detail
            for c in outcome.candidates
            for item in c.evidence
            if item.kind is EvidenceKind.NEGATIVE_EVIDENCE
        ]
        assert any("BNK-00001" in d and "not uniquely identified" in d for d in details)

    def test_a_credit_keyed_to_another_settlement_is_not_a_rival(self):
        """It carries a reference and the reference is not ours. It is explained."""
        mine = batch("SETL-0001", utr="UTR2026031000001", amounts=(100_000,), first_index=1)
        theirs = batch("SETL-0002", utr="UTR2026031000002", amounts=(100_000,), first_index=10)
        outcome = _t0(
            corpus(
                batches=[mine, theirs],
                bank_txns=[mine.credit("BNK-00001"), theirs.credit("BNK-00002")],
            )
        )
        assert outcome.resolved_settlements == 2

    def test_an_unrelated_unkeyed_credit_of_a_different_amount_is_not_a_rival(self):
        only = batch()
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[only.credit(), noise_credit(amount_minor=only.net_minor + 1)],
            )
        )
        assert outcome.resolved_settlements == 1


class TestSplitPayoutIsLeftForT2:
    """A09. The key on two credits means the batch was paid in tranches."""

    def test_a_key_on_two_credits_is_never_matched_as_a_whole_batch(self):
        only = batch(amounts=(60_000, 40_000))
        first = only.credit("BNK-00001", delta_minor=-40_000)
        second = only.credit("BNK-00002", delta_minor=-60_000, days_after=2)
        outcome = _t0(corpus(batches=[only], bank_txns=[first, second]))
        assert outcome.split_suspected == 1
        assert outcome.payment_links == 0

    def test_a_suspected_split_stays_in_the_pool_for_an_aggregating_tier(self):
        """Undecided, not contested. Consuming it would take it away from T2."""
        only = batch(amounts=(60_000, 40_000))
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001", delta_minor=-40_000),
                    only.credit("BNK-00002", delta_minor=-60_000, days_after=2),
                ],
            )
        )
        outcome = resolve_bank_leg(context, T0_RULE)
        assert outcome.undecided_settlements == 1
        assert context.consumed_settlements == set()
        assert context.consumed_credits == set()

    def test_a_lopsided_split_inside_the_band_is_still_declined(self):
        """The tranche that went missing is smaller than the proportional band.

        A shortfall that exactly equals another unclaimed credit is not rounding
        drift -- it is the rest of the payout, sitting in the statement.
        """
        from ledgerloop.config import MatchingTolerances
        from ledgerloop.matching.tier1_tolerance import rule_for

        only = batch(amounts=(10_000_000,))
        small, large = 40_000, only.net_minor - 40_000
        assert small < rule_for(MatchingTolerances()).band_for(only.net_minor)

        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=small, utr=None),
                    bank_credit("BNK-00002", amount_minor=large, utr=only.settlement.utr),
                ],
            )
        )
        outcome = resolve_bank_leg(context, rule_for(MatchingTolerances()))
        assert outcome.split_suspected == 1
        assert outcome.payment_links == 0
        assert context.consumed_settlements == set()

    def test_an_overpayment_is_not_treated_as_a_split(self):
        only = batch(amounts=(100_000,))
        outcome = _t0(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001"),
                    bank_credit("BNK-00002", amount_minor=50_000, utr=None),
                ],
            )
        )
        assert outcome.resolved_settlements == 1

    def test_rounding_drift_is_never_mistaken_for_a_missing_tranche(self):
        """A02 drifts by paise; no credit in a statement is worth three paise."""
        from ledgerloop.config import MatchingTolerances
        from ledgerloop.matching.tier1_tolerance import rule_for

        only = batch(amounts=(60_000, 40_000))
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[only.credit(delta_minor=-3), noise_credit(amount_minor=777_000)],
            )
        )
        outcome = resolve_bank_leg(context, rule_for(MatchingTolerances()))
        assert outcome.resolved_settlements == 1
        assert outcome.split_suspected == 0


class TestEdgeCases:
    def test_reading_an_allocated_share_off_a_settlement_link_is_refused(self):
        """It carries an arithmetic check too -- for the batch gross, not a share.

        Returning that would hand the evaluator a number wrong by the whole fee,
        and plausible enough that nothing would notice.
        """
        outcome = _t0(corpus(batches=[batch()], bank_txns=[batch().credit()]))
        settlement_link = next(
            c for c in outcome.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        with pytest.raises(ValueError, match="only expanded payment links"):
            allocated_share_minor(settlement_link)

    def test_a_batch_whose_only_payment_was_clawed_back_credits_nobody(self):
        """The declared net survives the claw-back, but no payment's money did."""
        only = batch(amounts=(50_000,), adjustments_minor=-50_000, net_minor=50_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.resolved_settlements == 1
        assert outcome.payment_links == 0

    def test_a_batch_whose_payments_were_all_quarantined_credits_nobody(self):
        """Ingest dropped every payment, so the batch has nobody to credit.

        The settlement edge still resolves -- the money did arrive -- but the
        arithmetic cannot be verified, so nothing is auto-matched below it.
        """
        only = batch(amounts=(), net_minor=50_000)
        outcome = _t0(corpus(batches=[only], bank_txns=[only.credit()]))
        assert outcome.resolved_settlements == 1
        assert outcome.payment_links == 0
        assert not any(c.arithmetic_verified for c in outcome.candidates)


class TestContextIndexes:
    def test_a_payment_with_no_settlement_is_grouped_nowhere(self):
        """``settlement_id`` is optional on the contract; ingest always sets it."""
        only = batch()
        built = corpus(batches=[only], bank_txns=[only.credit()])
        orphaned = list(built.payments)
        orphaned[0] = orphaned[0].model_copy(update={"settlement_id": None})
        context = MatchContext.from_ingest(replace(built, payments=tuple(orphaned)))
        assert len(context.settlements_by_id["SETL-0001"].payments) == 1

    def test_the_indexes_cover_every_record(self, simple):
        context = MatchContext.from_ingest(simple)
        assert set(context.orders_by_id) == {"ORD-2026-000001", "ORD-2026-000002"}
        assert set(context.settlements_by_id) == {"SETL-0001"}
        assert context.credits_with_utr == 1
        assert context.settlements_with_utr == 1

    def test_only_credits_are_indexed_by_key(self):
        only = batch()
        built = corpus(batches=[only], bank_txns=[debit_row(utr=only.settlement.utr)])
        context = MatchContext.from_ingest(built)
        assert context.credits_by_utr == {}
        assert context.open_credits() == ()
