"""Generator contract tests.

These verify the Phase 1 acceptance criteria: reproducibility, class coverage,
prevalence fidelity, and that ground truth genuinely describes the data rather
than being reverse-engineered from it.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import GeneratorConfig, prevalence_for
from ledgerloop.generator import generate
from ledgerloop.generator.baseline import build_clean_world
from ledgerloop.generator.scenarios import SCENARIOS
from ledgerloop.models.enums import (
    AnomalyClass,
    Difficulty,
    ExpectedStatus,
    LinkType,
    OrderStatus,
    SplitName,
)


def _dataset(**overrides):
    kwargs = {"split": SplitName.DEV, "seed": 42, "ensure_class_coverage": True}
    kwargs.update(overrides)
    return generate(GeneratorConfig(**kwargs))


class TestReproducibility:
    def test_same_seed_produces_identical_worlds(self):
        """PLAN.md Phase 1 acceptance: generate twice, get the same thing."""
        first = _dataset()
        second = _dataset()
        assert [o.order_id for o in first.world.orders] == [
            o.order_id for o in second.world.orders
        ]
        assert [o.amount_minor for o in first.world.orders] == [
            o.amount_minor for o in second.world.orders
        ]
        assert [t.narration for t in first.world.bank_txns] == [
            t.narration for t in second.world.bank_txns
        ]
        assert first.truth.links == second.truth.links
        assert first.truth.records == second.truth.records

    def test_different_seeds_produce_different_worlds(self):
        assert [t.narration for t in _dataset(seed=1).world.bank_txns] != [
            t.narration for t in _dataset(seed=2).world.bank_txns
        ]

    def test_difficulty_does_not_reshuffle_the_underlying_orders(self):
        """Two RNG streams: the dial changes what goes wrong, not what was sold.

        This is what makes the three difficulty columns comparable rather than
        three unrelated datasets.
        """
        easy = _dataset(difficulty=Difficulty.EASY, ensure_class_coverage=False)
        hard = _dataset(difficulty=Difficulty.HARD, ensure_class_coverage=False)
        assert [o.order_id for o in easy.world.orders] == [
            o.order_id for o in hard.world.orders
        ]
        assert [o.amount_minor for o in easy.world.orders] == [
            o.amount_minor for o in hard.world.orders
        ]


class TestStructure:
    def test_every_order_has_exactly_one_payment(self):
        world = _dataset().world
        assert len(world.payments) == len(world.orders)
        assert len({p.order_id for p in world.payments}) == len(world.orders)

    def test_every_payment_belongs_to_a_settlement(self):
        world = _dataset().world
        settlements = world.settlements_by_id()
        for payment in world.payments:
            assert payment.settlement_id in settlements

    def test_settlement_payment_ids_agree_with_payments(self):
        world = _dataset().world
        listed = sorted(pid for s in world.settlements for pid in s.payment_ids)
        assert listed == sorted(p.payment_id for p in world.payments)

    def test_batches_are_large_enough_to_be_an_aggregation_problem(self):
        """A batch of one is a 1:1 join. The N:1 problem is the whole point."""
        world = _dataset(split=SplitName.TEST, ensure_class_coverage=False).world
        sizes = [len(s.payment_ids) for s in world.settlements]
        assert min(sizes) >= 2
        assert sum(sizes) / len(sizes) >= 5

    def test_noise_rows_exist_and_belong_to_no_settlement(self):
        world = _dataset().world
        noise = [t for t in world.bank_txns if t.settlement_id is None]
        assert noise, "unrelated rows must exist so false positives are measurable"
        assert all(not t.covered_payment_ids for t in noise)

    def test_some_order_refs_are_corrupted(self):
        """T0 must not reach 100% even on clean money."""
        world = _dataset(split=SplitName.TEST, ensure_class_coverage=False).world
        raw = [p.order_ref_raw for p in world.payments]
        assert any(r is None for r in raw)
        assert any(
            r is not None and r != p.order_id
            for r, p in zip(raw, world.payments, strict=True)
        )


class TestClassCoverage:
    def test_every_scenario_has_an_implementation(self):
        assert set(SCENARIOS) == set(AnomalyClass)

    def test_fixture_sized_dataset_contains_every_class(self):
        """PLAN.md Phase 1 acceptance, for the committed fixture set."""
        world = _dataset().world
        produced = {effect.anomaly for effect in world.effects}
        expected = set(AnomalyClass) - {AnomalyClass.CLEAN}
        assert expected - produced == set()

    @pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.CALIBRATION, SplitName.TEST])
    def test_evaluated_splits_cover_every_class_they_can_expect_to(self, split):
        """Evaluated splits never use coverage seeding -- it would distort the
        prevalence they are measured against -- so coverage is only claimed
        where the sample size actually supports it.

        A 1%-prevalence class has a ~13% chance of not appearing at all in 200
        draws. Asserting full coverage on the calibration split would be
        asserting something untrue, so the bar is: every class whose expected
        count is at least three must be present.
        """
        dataset = _dataset(split=split, ensure_class_coverage=False)
        produced = {effect.anomaly for effect in dataset.world.effects}
        order_count = len(dataset.world.orders)

        for anomaly in set(AnomalyClass) - {AnomalyClass.CLEAN}:
            expected = dataset.config.prevalence[anomaly] * order_count
            if expected >= 3:
                assert anomaly in produced, f"{anomaly} expected ~{expected:.1f} times"

    def test_the_test_split_contains_every_class(self):
        """The split every published number comes from must exercise all eleven."""
        world = _dataset(split=SplitName.TEST, ensure_class_coverage=False).world
        produced = {effect.anomaly for effect in world.effects}
        assert set(AnomalyClass) - {AnomalyClass.CLEAN} - produced == set()

    def test_coverage_seeding_is_off_by_default(self):
        assert GeneratorConfig().ensure_class_coverage is False


class TestPrevalence:
    def test_realised_draws_match_the_configured_dial(self):
        """±2% per class, per the acceptance criterion.

        Measured on 2,000 draws so sampling error (~0.5% for a 5% class) is well
        inside the tolerance -- a 60-order dev set could not support this claim.
        """
        dataset = _dataset(order_count=2_000, ensure_class_coverage=False)
        realised = dataset.truth.realised_prevalence()
        configured = dataset.config.prevalence
        for anomaly in AnomalyClass:
            assert abs(realised[anomaly] - configured[anomaly]) <= 0.02, anomaly

    def test_draws_are_counted_once_per_order(self):
        dataset = _dataset()
        assert sum(dataset.truth.scenario_draws.values()) == len(dataset.world.orders)

    def test_seeded_effects_are_not_counted_as_draws(self):
        """Coverage seeding must not corrupt the prevalence report."""
        dataset = _dataset(ensure_class_coverage=True)
        assert sum(dataset.truth.scenario_draws.values()) == len(dataset.world.orders)

    @pytest.mark.parametrize("difficulty", list(Difficulty))
    def test_dial_weights_sum_to_one(self, difficulty):
        assert sum(prevalence_for(difficulty).values()) == pytest.approx(1.0)

    def test_harder_settings_produce_more_anomalies(self):
        counts = [
            len(
                _dataset(
                    split=SplitName.TEST, difficulty=d, ensure_class_coverage=False
                ).world.effects
            )
            for d in (Difficulty.EASY, Difficulty.STANDARD, Difficulty.HARD)
        ]
        assert counts[0] < counts[1] < counts[2]


class TestGroundTruth:
    def test_evaluation_pairs_are_payment_to_bank_only(self):
        truth = _dataset().truth
        credited = truth.links_by_type[LinkType.PAYMENT_CREDITED_AS]
        assert len(truth.evaluation_pairs) == len(credited)
        assert all(link.source_ref.record_type.value == "payment" for link in credited)
        assert all(link.target_ref.record_type.value == "bank_txn" for link in credited)

    def test_structural_edges_exist_but_are_not_evaluated(self):
        truth = _dataset().truth
        assert truth.links_by_type[LinkType.ORDER_PAID_BY]
        assert truth.links_by_type[LinkType.PAYMENT_SETTLED_IN]
        structural = {
            link.pair
            for link in truth.links
            if link.link_type is not LinkType.PAYMENT_CREDITED_AS
        }
        assert not (structural & truth.evaluation_pairs)

    def test_every_record_has_exactly_one_verdict(self):
        dataset = _dataset()
        world, truth = dataset.world, dataset.truth
        expected = (
            len(world.orders)
            + len(world.payments)
            + len(world.settlements)
            + len(world.bank_txns)
        )
        assert len(truth.records) == expected

    def test_orphan_credits_are_unmatchable_not_exceptions(self):
        """The honest ceiling: no data exists to resolve these."""
        truth = _dataset().truth
        orphans = [
            r
            for r in truth.records
            if r.anomaly_class is AnomalyClass.ORPHAN_BANK_CREDIT
        ]
        assert orphans
        assert all(r.expected_status is ExpectedStatus.UNMATCHABLE for r in orphans)

    def test_duplicate_credit_gets_no_link(self):
        """A system that links the duplicate has made a false positive."""
        dataset = _dataset()
        duplicates = [
            r.record_ref.record_id
            for r in dataset.truth.records
            if r.anomaly_class is AnomalyClass.DUPLICATE_CREDIT
        ]
        assert duplicates
        credited = {pair[1] for pair in dataset.truth.evaluation_pairs}
        for txn_id in duplicates:
            assert f"bank_txn:{txn_id}" not in credited

    def test_chargeback_payment_has_no_credit_link(self):
        """Its money never reached the bank, so no edge should exist."""
        dataset = _dataset()
        charged = [
            r.record_ref.record_id
            for r in dataset.truth.records
            if r.anomaly_class is AnomalyClass.CHARGEBACK_NETTED
        ]
        assert charged
        credited = {pair[0] for pair in dataset.truth.evaluation_pairs}
        for payment_id in charged:
            assert f"payment:{payment_id}" not in credited

    def test_split_payout_produces_two_credits_for_one_settlement(self):
        dataset = _dataset()
        # A09 labels the SECOND credit -- the row that would not otherwise
        # exist -- because a settlement-level label would collide with A03/A12.
        split_txn_ids = [
            r.record_ref.record_id
            for r in dataset.truth.records
            if r.anomaly_class is AnomalyClass.SPLIT_PAYOUT
        ]
        assert split_txn_ids
        txns = {t.txn_id: t for t in dataset.world.bank_txns}
        for txn_id in split_txn_ids:
            settlement_id = txns[txn_id].settlement_id
            assert settlement_id is not None
            assert len(dataset.world.credits_for_settlement(settlement_id)) == 2

    def test_refunded_orders_are_marked_refunded_in_the_ledger(self):
        dataset = _dataset()
        refunded = {
            r.record_ref.record_id
            for r in dataset.truth.records
            if r.anomaly_class is AnomalyClass.POST_SETTLEMENT_REFUND
        }
        assert refunded
        orders = dataset.world.orders_by_id()
        assert all(orders[oid].status is OrderStatus.REFUNDED for oid in refunded)

    def test_unmatchable_records_leave_the_denominator(self):
        truth = _dataset().truth
        assert truth.unmatchable_refs
        assert not (truth.unmatchable_refs & truth.reconcilable_refs)

    def test_impact_is_recorded_for_money_at_stake(self):
        truth = _dataset().truth
        chargebacks = [
            r for r in truth.records if r.anomaly_class is AnomalyClass.CHARGEBACK_NETTED
        ]
        assert all(r.impact_minor > 0 for r in chargebacks)

    def test_every_effect_carries_an_explanatory_note(self):
        """The note seeds the report's honesty section and explains dev failures."""
        for effect in _dataset().world.effects:
            assert effect.note.strip()


class TestSplits:
    @pytest.mark.parametrize(
        ("split", "expected"),
        [
            (SplitName.DEV, 60),
            (SplitName.TRAIN, 400),
            (SplitName.CALIBRATION, 200),
            (SplitName.TEST, 300),
        ],
    )
    def test_split_sizes(self, split, expected):
        assert len(_dataset(split=split, ensure_class_coverage=False).world.orders) == expected

    def test_dev_meets_the_fifty_record_challenge_floor(self):
        assert len(_dataset().world.orders) >= 50

    def test_splits_with_the_same_seed_are_not_prefixes_of_each_other(self):
        """Distinct splits must be genuinely distinct data, not nested samples --
        otherwise `test` would overlap `train` and every metric would leak."""
        train = _dataset(split=SplitName.TRAIN, ensure_class_coverage=False).world
        test = _dataset(split=SplitName.TEST, ensure_class_coverage=False).world
        overlap = {o.amount_minor for o in train.orders[:50]} & {
            o.amount_minor for o in test.orders[:50]
        }
        assert len(overlap) < 25


class TestCleanBaseline:
    def test_baseline_world_reconciles_exactly(self):
        """Before any anomaly, every credit equals its settlement's declared net."""
        import random

        world = build_clean_world(random.Random(42), 120)
        payments = world.payments_by_id()
        for settlement in world.settlements:
            credits = world.credits_for_settlement(settlement.settlement_id)
            assert len(credits) == 1
            assert credits[0].credit_minor == settlement.declared_net_minor(payments)

    def test_baseline_has_no_effects_and_no_mismatch(self):
        import random

        world = build_clean_world(random.Random(42), 120)
        assert world.effects == []
        assert all(s.net_mismatch_minor == 0 for s in world.settlements)
        assert all(s.adjustments_minor == 0 for s in world.settlements)
