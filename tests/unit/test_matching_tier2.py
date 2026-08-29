"""T2 -- aggregation, on batches small enough to check by hand.

The tier's job is to partition a batch across the tranches it was paid in, and
its discipline is to refuse when the partition is not forced. So the tests come
in two halves: the splits it should solve exactly, and the shapes it must
decline -- several valid partitions, a batch that does not add up, a lone
tranche, a bucket too big to search exhaustively.

The interaction with the residual pool is tested through the pool itself rather
than through the output, because "T2 cannot override T0" is a property of what
it is allowed to see, not of what it happens to emit.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ledgerloop.config import MatchingTolerances
from ledgerloop.matching.bank_leg import allocated_share_minor, resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import T0_RULE, run_tier0
from ledgerloop.matching.tier1_tolerance import run_tier1
from ledgerloop.matching.tier2_aggregation import (
    credit_bucket,
    expected_credit_minor,
    payment_bucket,
    run_tier2,
)
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus, noise_credit

DEFAULTS = MatchingTolerances()


def _t2(ingest, tolerances: MatchingTolerances = DEFAULTS):
    """T2 alone, on a fresh pool -- the tier in isolation."""
    return run_tier2(MatchContext.from_ingest(ingest), tolerances)


def _ladder(ingest, tolerances: MatchingTolerances = DEFAULTS):
    """The whole ladder over one pool -- the tier in its place."""
    context = MatchContext.from_ingest(ingest)
    run_tier0(context)
    run_tier1(context, tolerances)
    return context, run_tier2(context, tolerances)


def _pairs(candidates):
    return {(c.source_ref.key, c.target_ref.key) for c in candidates if c.is_evaluable}


def split(only, first: tuple[int, ...], second: tuple[int, ...]):
    """Two tranches carrying the given payment positions, allocated like the truth."""
    grosses = [p.amount_minor for p in only.payments]
    left = sum(grosses[i] for i in first)
    right = sum(grosses[i] for i in second)
    amounts = allocate_minor(only.net_minor, [left, right])
    return (
        bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
        bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
    )


class TestASplitItSolves:
    @pytest.fixture
    def solved(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        tranches = split(only, (0,), (1, 2))
        return only, _t2(corpus(batches=[only], bank_txns=list(tranches)))

    def test_the_partition_is_found(self, solved):
        _, outcome = solved
        assert outcome.settlements_resolved == 1
        assert outcome.settlements_ambiguous == 0
        assert outcome.credits_matched == 2
        assert outcome.payments_matched == 3

    def test_every_payment_is_assigned_to_exactly_one_tranche(self, solved):
        _, outcome = solved
        assert _pairs(outcome.candidates) == {
            ("payment:PAY-00001", "bank_txn:BNK-00001"),
            ("payment:PAY-00002", "bank_txn:BNK-00002"),
            ("payment:PAY-00003", "bank_txn:BNK-00002"),
        }

    def test_the_shares_conserve_each_tranche_exactly(self, solved):
        only, outcome = solved
        by_credit: dict[str, int] = {}
        for candidate in outcome.candidates:
            if candidate.is_evaluable:
                key = candidate.target_ref.record_id
                by_credit[key] = by_credit.get(key, 0) + allocated_share_minor(candidate)
        credits = {c.txn_id: c.credit_minor for c in split(only, (0,), (1, 2))}
        assert by_credit == credits

    def test_the_tranches_conserve_the_whole_net(self, solved):
        only, outcome = solved
        total = sum(
            allocated_share_minor(c) for c in outcome.candidates if c.is_evaluable
        )
        assert total == only.net_minor

    def test_candidates_carry_the_tier_and_the_subset(self, solved):
        _, outcome = solved
        for candidate in outcome.candidates:
            assert candidate.tier is Tier.T2_AGGREGATION
            assert candidate.features.tier is Tier.T2_AGGREGATION
            assert candidate.subset_members
            assert candidate.features.subset_size == len(candidate.subset_members)

    def test_a_resolved_partition_is_certain_and_verified(self, solved):
        _, outcome = solved
        assert all(c.calibrated_p == 1.0 for c in outcome.candidates)
        assert all(c.arithmetic_verified for c in outcome.candidates)

    def test_the_evidence_names_the_subset_and_the_arithmetic(self, solved):
        _, outcome = solved
        settlement_link = next(
            c for c in outcome.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        kinds = {item.kind for item in settlement_link.evidence}
        assert EvidenceKind.SUBSET_SUM in kinds
        assert EvidenceKind.ARITHMETIC_CHECK in kinds
        notes = [
            i.detail
            for c in outcome.candidates
            for i in c.evidence
            if i.kind is EvidenceKind.SUBSET_SUM
        ]
        assert any("PAY-00001" in note for note in notes)
        assert any("PAY-00002, PAY-00003" in note for note in notes)
        assert all("residual" in note for note in notes)
        assert all(item.refs for item in settlement_link.evidence)

    def test_the_conservation_evidence_states_the_partition_is_complete(self, solved):
        _, outcome = solved
        details = [i.detail for c in outcome.candidates for i in c.evidence]
        assert any("covers all" in d and "exactly once" in d for d in details)

    def test_it_consumes_the_settlement_and_both_tranches(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        tranches = split(only, (0,), (1, 2))
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=list(tranches)))
        run_tier2(context, DEFAULTS)
        assert context.consumed_settlements == {"SETL-0001"}
        assert context.consumed_credits == {"BNK-00001", "BNK-00002"}

    def test_a_three_way_split_is_solved_too(self):
        only = batch(amounts=(70_000, 50_000, 30_000, 11_000), fee_minor=3_000)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(
            only.net_minor, [grosses[0], grosses[1], grosses[2] + grosses[3]]
        )
        rows = [
            bank_credit(f"BNK-0000{i + 1}", amount_minor=part, utr=only.settlement.utr)
            for i, part in enumerate(parts)
        ]
        outcome = _t2(corpus(batches=[only], bank_txns=rows))
        assert outcome.settlements_resolved == 1
        assert outcome.credits_matched == 3
        assert outcome.payments_matched == 4


class TestWhatItRefusesToSolve:
    def test_several_valid_partitions_are_ambiguous(self):
        """Two payments of equal gross: which tranche took which is a coin flip."""
        only = batch(amounts=(50_000, 50_000, 20_000))
        tranches = split(only, (0,), (1, 2))
        outcome = _t2(corpus(batches=[only], bank_txns=list(tranches)))
        assert outcome.settlements_ambiguous == 1
        assert outcome.settlements_resolved == 0
        assert outcome.payment_links == 0

    def test_the_ambiguity_names_the_competing_subsets(self):
        only = batch(amounts=(50_000, 50_000, 20_000))
        outcome = _t2(
            corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2))))
        )
        detail = next(
            i.detail
            for c in outcome.candidates
            for i in c.evidence
            if i.kind is EvidenceKind.NEGATIVE_EVIDENCE and "coin flip" in i.detail
        )
        assert "PAY-00001" in detail and "PAY-00002" in detail

    def test_an_ambiguous_settlement_gets_a_probability_below_tau_low(self):
        """0.5 routes to an exception through the configured policy, not a special case."""
        only = batch(amounts=(50_000, 50_000, 20_000))
        outcome = _t2(
            corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2))))
        )
        assert all(c.calibrated_p == 0.5 for c in outcome.candidates)
        assert not any(c.arithmetic_verified for c in outcome.candidates)

    def test_an_ambiguous_settlement_leaves_the_pool_but_its_credits_do_not(self):
        only = batch(amounts=(50_000, 50_000, 20_000))
        context = MatchContext.from_ingest(
            corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2))))
        )
        run_tier2(context, DEFAULTS)
        assert context.consumed_settlements == {"SETL-0001"}
        assert context.consumed_credits == set()

    def test_tranches_that_do_not_add_up_to_the_net_are_left_alone(self):
        only = batch(amounts=(60_000, 40_000))
        outcome = _t2(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=10_000, utr=only.settlement.utr),
                    bank_credit("BNK-00002", amount_minor=10_000, utr=only.settlement.utr),
                ],
            )
        )
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_a_lone_tranche_is_not_t2s_question(self):
        """One credit is a whole-batch match, which T0 and T1 own."""
        only = batch(amounts=(60_000, 40_000))
        outcome = _t2(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=60_000, utr=only.settlement.utr)
                ],
            )
        )
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_a_settlement_with_no_keyed_credit_is_counted_not_searched(self):
        """A07 stripped the reference. T3's problem, not T2's."""
        only = batch()
        outcome = _t2(corpus(batches=[only], bank_txns=[noise_credit()]))
        assert outcome.settlements_without_key == 1
        assert outcome.candidates == ()

    def test_an_unsolvable_partition_stays_in_the_pool(self):
        """The tranches sum to the net but no subset reaches either of them."""
        only = batch(amounts=(30_000, 30_000, 40_000))
        half = only.net_minor // 2
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=half, utr=only.settlement.utr),
                    bank_credit(
                        "BNK-00002",
                        amount_minor=only.net_minor - half,
                        utr=only.settlement.utr,
                    ),
                ],
            )
        )
        outcome = run_tier2(context, DEFAULTS)
        assert outcome.settlements_resolved == 0
        assert outcome.settlements_unsolved == 1
        assert context.consumed_settlements == set()


class TestToleranceBoundaries:
    def _drifted(self, drift: int, tolerances: MatchingTolerances = DEFAULTS):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        first, second = split(only, (0,), (1, 2))
        drifted = bank_credit(
            "BNK-00001", amount_minor=first.credit_minor + drift, utr=only.settlement.utr
        )
        sibling = bank_credit(
            "BNK-00002",
            amount_minor=second.credit_minor - drift,
            utr=only.settlement.utr,
        )
        return _t2(corpus(batches=[only], bank_txns=[drifted, sibling]), tolerances)

    @pytest.mark.parametrize("drift", [0, 1, -1, 300, -300])
    def test_drift_inside_epsilon_still_resolves(self, drift):
        assert self._drifted(drift).settlements_resolved == 1

    @pytest.mark.parametrize("drift", [301, -301])
    def test_drift_beyond_epsilon_does_not(self, drift):
        outcome = self._drifted(drift)
        assert outcome.settlements_resolved == 0

    def test_epsilon_comes_from_the_configuration(self):
        tight = MatchingTolerances(aggregation_epsilon_minor=1)
        assert self._drifted(200, tight).settlements_resolved == 0
        assert self._drifted(200).settlements_resolved == 1

    def test_the_band_is_reported_as_the_feature(self):
        outcome = self._drifted(0)
        assert all(
            c.features.tolerance_band_minor == DEFAULTS.aggregation_epsilon_minor
            for c in outcome.candidates
        )


class TestTheResidualPool:
    def test_t2_never_sees_a_settlement_t0_resolved(self):
        only = batch()
        context, outcome = _ladder(
            corpus(batches=[only], bank_txns=[only.credit()])
        )
        assert "SETL-0001" in context.consumed_settlements
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_t2_never_sees_a_settlement_t0_contested(self):
        """A05 was decided -- as contested. T2 may not re-open it."""
        only = batch()
        _, outcome = _ladder(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
            )
        )
        assert outcome.settlements_seen == 0

    def test_t2_picks_up_exactly_the_split_t0_and_t1_declined(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        tranches = split(only, (0,), (1, 2))
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=list(tranches)))
        first = resolve_bank_leg(context, T0_RULE)
        assert first.split_suspected == 1
        assert context.consumed_settlements == set()

        outcome = run_tier2(context, DEFAULTS)
        assert outcome.settlements_resolved == 1

    def test_a_credit_consumed_earlier_is_not_available_to_t2(self):
        """Exclusivity is two-sided and survives across tiers."""
        whole = batch("SETL-0001", utr="UTR2026031000001", amounts=(50_000,), first_index=1)
        context = MatchContext.from_ingest(
            corpus(batches=[whole], bank_txns=[whole.credit("BNK-00001")])
        )
        run_tier0(context)
        assert context.consumed_credits == {"BNK-00001"}
        assert credit_bucket(whole_view(context), context) == ()


def whole_view(context):
    return context.settlements_by_id["SETL-0001"]


class TestBucketing:
    def test_payments_are_bucketed_by_the_settlement_anchor(self, simple):
        context = MatchContext.from_ingest(simple)
        view = context.settlements_by_id["SETL-0001"]
        assert [p.payment_id for p in payment_bucket(view, context)] == [
            "PAY-00001",
            "PAY-00002",
        ]

    def test_a_charged_back_payment_is_left_out_of_the_bucket(self):
        """A08: its money never reached the bank, so it is in no tranche."""
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        context = MatchContext.from_ingest(corpus(batches=[only]))
        view = context.settlements_by_id["SETL-0001"]
        assert [p.payment_id for p in payment_bucket(view, context)] == ["PAY-00001"]

    def test_credits_are_bucketed_by_the_key_largest_first(self):
        only = batch(amounts=(60_000, 40_000))
        rows = [
            bank_credit("BNK-00001", amount_minor=40_000, utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=60_000, utr=only.settlement.utr),
            noise_credit("BNK-09001"),
        ]
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=rows))
        view = context.settlements_by_id["SETL-0001"]
        assert [c.txn_id for c in credit_bucket(view, context)] == ["BNK-00002", "BNK-00001"]

    def test_a_settlement_without_a_key_has_an_empty_credit_bucket(self):
        only = batch(utr=None)
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=[noise_credit()]))
        view = context.settlements_by_id["SETL-0001"]
        assert credit_bucket(view, context) == ()


class TestTheProportionalBridge:
    def test_a_full_subset_carries_the_whole_net(self):
        assert expected_credit_minor(1_000, 1_000, 900) == 900

    def test_an_empty_batch_carries_nothing(self):
        assert expected_credit_minor(0, 0, 900) == 0

    def test_shares_of_a_split_sum_to_the_net(self):
        gross, net = 1_000, 907
        for part in range(1, gross):
            first = expected_credit_minor(part, gross, net)
            second = expected_credit_minor(gross - part, gross, net)
            assert first + second in (net, net - 1, net + 1)

    def test_it_matches_the_allocation_the_truth_links_were_built_from(self):
        gross, net, part = 1_750_000, 1_713_362, 1_147_680
        assert expected_credit_minor(part, gross, net) == allocate_minor(
            net, [part, gross - part]
        )[0]


class TestLargeBuckets:
    def test_a_bucket_past_the_exhaustive_cap_falls_back_and_is_not_asserted(self):
        """Greedy can find a subset but never prove it alone, so nothing is matched."""
        amounts = tuple(10_000 + 137 * i for i in range(45))
        only = batch(amounts=amounts, fee_minor=1_000)
        grosses = [p.amount_minor for p in only.payments]
        first_half = sum(grosses[:20])
        parts = allocate_minor(only.net_minor, [first_half, sum(grosses[20:])])
        rows = [
            bank_credit("BNK-00001", amount_minor=parts[0], utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=parts[1], utr=only.settlement.utr),
        ]
        outcome = _t2(corpus(batches=[only], bank_txns=rows))
        assert outcome.greedy_fallbacks >= 1
        assert outcome.settlements_resolved == 0
        assert outcome.payment_links == 0

    def test_the_unproven_fallback_is_recorded_as_such(self):
        amounts = tuple(10_000 + 137 * i for i in range(45))
        only = batch(amounts=amounts, fee_minor=1_000)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [sum(grosses[:20]), sum(grosses[20:])])
        outcome = _t2(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=parts[0], utr=only.settlement.utr),
                    bank_credit("BNK-00002", amount_minor=parts[1], utr=only.settlement.utr),
                ],
            )
        )
        if outcome.candidates:
            assert all(c.calibrated_p < 1.0 for c in outcome.candidates)
            assert not any(c.arithmetic_verified for c in outcome.candidates)

    def test_a_twenty_payment_split_is_still_exhaustive_and_fast(self):
        """Amounts spread widely enough that only one subset can reach the tranche.

        A ``2^10``-per-half enumeration, which is instant -- the meet-in-the-
        middle split is what keeps a twenty-payment batch out of ``2^20``.
        """
        amounts = tuple(1_000 * 2**i for i in range(20))
        only = batch(amounts=amounts, fee_minor=2_000)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [sum(grosses[:7]), sum(grosses[7:])])
        outcome = _t2(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=parts[1], utr=only.settlement.utr),
                    bank_credit("BNK-00002", amount_minor=parts[0], utr=only.settlement.utr),
                ],
            )
        )
        assert outcome.settlements_resolved == 1
        assert outcome.greedy_fallbacks == 0
        assert outcome.timeouts == 0
        assert outcome.payments_matched == 20

    def test_a_batch_of_evenly_spaced_amounts_is_inherently_ambiguous(self):
        """Equal spacing makes many subsets share a sum, so nothing is forced.

        Twenty payments in arithmetic progression: a seven-payment subset and a
        six-payment one land inside the same tolerance. Refusing is the whole
        point -- the alternative is picking one and being right by luck.
        """
        amounts = tuple(10_000 + 311 * i for i in range(20))
        only = batch(amounts=amounts, fee_minor=2_000)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [sum(grosses[:7]), sum(grosses[7:])])
        outcome = _t2(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=parts[1], utr=only.settlement.utr),
                    bank_credit("BNK-00002", amount_minor=parts[0], utr=only.settlement.utr),
                ],
            )
        )
        assert outcome.settlements_ambiguous == 1
        assert outcome.settlements_resolved == 0
        assert outcome.payment_links == 0


class TestDeterminism:
    def test_two_runs_produce_identical_candidates(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        rows = list(split(only, (0,), (1, 2)))
        first = _t2(corpus(batches=[only], bank_txns=rows))
        second = _t2(corpus(batches=[only], bank_txns=rows))
        assert first.candidates == second.candidates

    def test_the_order_of_the_bank_rows_does_not_change_the_partition(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        rows = list(split(only, (0,), (1, 2)))
        forward = _t2(corpus(batches=[only], bank_txns=rows))
        backward = _t2(corpus(batches=[only], bank_txns=list(reversed(rows))))
        assert _pairs(forward.candidates) == _pairs(backward.candidates)

    def test_no_timeout_fires_on_an_ordinary_batch(self):
        """The deterministic bound is the item cap; the clock is only a safety net."""
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        outcome = _t2(corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2)))))
        assert outcome.timeouts == 0


class TestReportingSurface:
    def test_the_outcome_names_its_tier(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        outcome = _t2(corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2)))))
        assert outcome.tier is Tier.T2_AGGREGATION

    def test_settlement_and_payment_links_are_counted_separately(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        outcome = _t2(corpus(batches=[only], bank_txns=list(split(only, (0,), (1, 2)))))
        assert outcome.settlement_links == 2
        assert outcome.payment_links == 3
        assert outcome.settlement_links + outcome.payment_links == len(outcome.candidates)


class TestDegenerateBatches:
    def test_more_tranches_than_payments_cannot_be_partitioned(self):
        only = batch(amounts=(60_000, 40_000))
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [grosses[0], grosses[1] // 2, grosses[1] // 2])
        rows = [
            bank_credit(f"BNK-0000{i + 1}", amount_minor=part, utr=only.settlement.utr)
            for i, part in enumerate(parts)
        ]
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=rows))
        outcome = run_tier2(context, DEFAULTS)
        assert outcome.settlements_resolved == 0
        assert context.consumed_settlements == set()

    def test_a_batch_whose_every_payment_was_clawed_back_is_skipped(self):
        """Nothing is left to put in a tranche."""
        only = batch(amounts=(50_000,), adjustments_minor=-50_000, net_minor=50_000)
        rows = [
            bank_credit("BNK-00001", amount_minor=25_000, utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=25_000, utr=only.settlement.utr),
        ]
        outcome = _t2(corpus(batches=[only], bank_txns=rows))
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_a_time_bound_that_fires_leaves_the_settlement_in_the_pool(self):
        """A resource failure is not a data ambiguity, so nothing is concluded."""
        amounts = tuple(1_000 * 2**i for i in range(32))
        only = batch(amounts=amounts, fee_minor=2_000)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [sum(grosses[:9]), sum(grosses[9:])])
        rows = [
            bank_credit("BNK-00001", amount_minor=parts[1], utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=parts[0], utr=only.settlement.utr),
        ]
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=rows))
        outcome = run_tier2(
            context, MatchingTolerances(subset_solver_timeout_ms=1)
        )
        assert outcome.timeouts == 1
        assert outcome.settlements_unsolved == 1
        assert outcome.candidates == ()
        assert context.consumed_settlements == set()


class TestTheGreedyFallbackInPlace:
    def test_a_greedy_subset_is_proposed_but_never_asserted(self):
        """45 payments is past the exhaustive cap, and the largest twenty are
        exactly one tranche -- so greedy finds them, and still cannot prove
        they are the only set that would."""
        amounts = tuple(10_000 + 137 * i for i in range(45))
        only = batch(amounts=amounts, fee_minor=1_000)
        grosses = [p.amount_minor for p in only.payments]
        largest = sum(sorted(grosses)[-20:])
        parts = allocate_minor(only.net_minor, [largest, sum(grosses) - largest])
        rows = [
            bank_credit("BNK-00001", amount_minor=parts[0], utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=parts[1], utr=only.settlement.utr),
        ]
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=rows))
        outcome = run_tier2(context, DEFAULTS)
        assert outcome.greedy_fallbacks >= 1
        assert outcome.settlements_resolved == 0
        assert outcome.payment_links == 0
        if outcome.candidates:
            assert outcome.settlements_ambiguous == 1
            detail = next(
                item.detail
                for c in outcome.candidates
                for item in c.evidence
                if "found greedily" in item.detail
            )
            assert "cannot prove" in detail

    def test_a_tranche_left_with_no_payments_to_take_fails_the_partition(self):
        """Two payments, three tranches: the third has nothing left to explain."""
        only = batch(amounts=(60_000, 40_000))
        rows = [
            bank_credit("BNK-00001", amount_minor=60_000, utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=39_999, utr=only.settlement.utr),
            bank_credit("BNK-00003", amount_minor=1, utr=only.settlement.utr),
        ]
        context = MatchContext.from_ingest(corpus(batches=[only], bank_txns=rows))
        outcome = run_tier2(context, DEFAULTS)
        assert outcome.settlements_resolved == 0
        assert outcome.settlements_unsolved == 1
        assert context.consumed_settlements == set()


class TestTheAllocationDenominatorIsTheBucket:
    """A charged-back payment must leave the denominator as well as the subset.

    `payment_bucket` drops a payment a negative adjustment identifies as charged
    back -- its money never reached the bank, so it belongs to no tranche. Until
    Phase 2.5 the allocation that *sizes* the tranches still divided by the gross
    of **every** nested payment, so each tranche came out short by its share of
    money that was never paid, and a batch that was both split (A09) and charged
    back (A08) was refused for arithmetic that could not close.

    A batch with no claw-back is unaffected: the two denominators are equal, and
    `TestAgainstTheFixture` and the ablation pins elsewhere in this suite are
    what say so.
    """

    def _split_with_a_chargeback(self):
        """120,000 gross, one 20,000 payment charged back, 100,000 paid in two.

        The declared net is 100,000 (gross 120,000, adjustment -20,000) and the
        surviving payments are 60,000 and 40,000, whose bucket gross is exactly
        100,000 -- so the tranches are 60,000 and 40,000.

        The old denominator divided by the **batch** gross of 120,000 and
        predicted ``allocate(100000, [60000, 60000])[0]`` = 50,000 for the first
        tranche against an actual 60,000: out by 10,000, far outside epsilon, so
        the batch was refused.
        """
        only = batch(
            amounts=(60_000, 40_000, 20_000),
            adjustments_minor=-20_000,
            first_index=1,
        )
        first = bank_credit("BNK-00001", amount_minor=60_000, utr=only.settlement.utr)
        second = bank_credit(
            "BNK-00002",
            amount_minor=40_000,
            utr=only.settlement.utr,
            value_date=only.settlement.settled_on + timedelta(days=2),
        )
        return only, corpus(batches=[only], bank_txns=[first, second])

    def test_a_split_batch_with_a_chargeback_now_resolves(self):
        _only, sources = self._split_with_a_chargeback()
        outcome = _t2(sources)
        assert outcome.settlements_resolved == 1
        assert outcome.credits_matched == 2

    def test_the_charged_back_payment_is_in_no_tranche(self):
        """It is excluded from the subset *and* from the denominator, which are
        the same fact stated twice -- money that did not arrive cannot be part
        of a tranche and cannot size one either."""
        _only, sources = self._split_with_a_chargeback()
        outcome = _t2(sources)
        assigned = {
            c.source_ref.record_id for c in outcome.candidates if c.is_evaluable
        }
        assert assigned == {"PAY-00001", "PAY-00002"}
        assert "PAY-00003" not in assigned

    def test_every_tranche_it_asserts_is_arithmetic_verified(self):
        _only, sources = self._split_with_a_chargeback()
        outcome = _t2(sources)
        assert outcome.candidates
        assert all(c.arithmetic_verified for c in outcome.candidates)

    def test_a_batch_without_a_chargeback_is_unaffected(self):
        """The two denominators coincide, so nothing about the ordinary case
        changed. This is the control for the test above."""
        only = batch(amounts=(60_000, 40_000))
        first = bank_credit("BNK-00001", amount_minor=60_000, utr=only.settlement.utr)
        second = bank_credit(
            "BNK-00002",
            amount_minor=40_000,
            utr=only.settlement.utr,
            value_date=only.settlement.settled_on + timedelta(days=2),
        )
        outcome = _t2(corpus(batches=[only], bank_txns=[first, second]))
        assert outcome.settlements_resolved == 1
        assert _pairs(outcome.candidates) == {
            ("payment:PAY-00001", "bank_txn:BNK-00001"),
            ("payment:PAY-00002", "bank_txn:BNK-00002"),
        }

    def test_tranches_that_do_not_reach_the_declared_net_are_still_refused(self):
        """The fix corrects a denominator; it does not widen a tolerance."""
        only = batch(
            amounts=(60_000, 40_000, 20_000),
            adjustments_minor=-20_000,
            first_index=1,
        )
        first = bank_credit("BNK-00001", amount_minor=60_000, utr=only.settlement.utr)
        short = bank_credit(
            "BNK-00002",
            amount_minor=25_000,
            utr=only.settlement.utr,
            value_date=only.settlement.settled_on + timedelta(days=2),
        )
        outcome = _t2(corpus(batches=[only], bank_txns=[first, short]))
        assert outcome.settlements_resolved == 0
