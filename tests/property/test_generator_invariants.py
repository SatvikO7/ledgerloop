"""Property tests for the generator.

**The headline property is money conservation** (PLAN.md Phase 1 acceptance:
"total money conserved across all three sources modulo declared anomalies").

"Modulo declared anomalies" is the load-bearing phrase, and it is what makes
this checkable rather than rhetorical. Every scenario that deliberately moves
money declares a ``bank_delta_minor``. The invariant is then exact:

    sum(settlement-linked credits) - sum(declared nets) - sum(declared deltas) == 0

A non-zero residual means a scenario moved money without saying so, and every
metric computed on that dataset would be measuring the wrong world. This test
already caught one real bug: A06 claimed the ``amount`` aspect on the settlement
it originated from rather than the one whose credit it rewrites, which let a
claw-back silently overwrite an A02 rounding drift.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate
from ledgerloop.models.enums import (
    AnomalyClass,
    Difficulty,
    LinkType,
    RecordType,
    SplitName,
)
from ledgerloop.money import allocate_minor

# Generation is not free, so these run a modest number of examples with the
# deadline disabled. Breadth across seeds and difficulties matters more here
# than example count: the bugs this catches are combinations of scenarios
# landing on the same settlement, which needs varied seeds, not many draws.
GENERATOR_SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

seeds = st.integers(min_value=0, max_value=2**16)
difficulties = st.sampled_from(list(Difficulty))
order_counts = st.integers(min_value=40, max_value=260)


def _dataset(seed: int, difficulty: Difficulty, order_count: int):
    return generate(
        GeneratorConfig(
            split=SplitName.DEV,
            difficulty=difficulty,
            seed=seed,
            order_count=order_count,
            ensure_class_coverage=True,
        )
    )


class TestMoneyConservation:
    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_bank_credits_reconcile_to_declared_nets(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """THE property. Zero residual, whatever the seed or the dial."""
        assert _dataset(seed, difficulty, order_count).conservation_residual_minor == 0

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_gross_always_equals_the_sum_of_its_payments(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """No anomaly may invent or destroy a payment inside a batch."""
        world = _dataset(seed, difficulty, order_count).world
        payments = world.payments_by_id()
        for settlement in world.settlements:
            assert settlement.gross_minor(payments) == sum(
                payments[pid].amount_minor for pid in settlement.payment_ids
            )

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_link_amounts_sum_to_their_credit(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """Each credit is allocated exactly across the payments it carries.

        This is what makes a split payout expressible without losing a paise.
        """
        dataset = _dataset(seed, difficulty, order_count)
        by_txn: dict[str, int] = {}
        for link in dataset.truth.links:
            if link.link_type is LinkType.PAYMENT_CREDITED_AS:
                by_txn[link.target_ref.record_id] = (
                    by_txn.get(link.target_ref.record_id, 0) + link.amount_minor
                )
        for txn in dataset.world.bank_txns:
            if txn.covered_payment_ids:
                assert by_txn[txn.txn_id] == txn.credit_minor

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_split_payout_parts_sum_to_the_whole(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        dataset = _dataset(seed, difficulty, order_count)
        world = dataset.world
        payments = world.payments_by_id()

        # A split settlement may also carry an A02 drift, applied to the credit
        # *before* it was split. The parts therefore sum to the declared net
        # plus whatever delta was declared against those credits -- which is
        # exactly the "modulo declared anomalies" clause, checked locally.
        deltas = {
            effect.primary_ref.record_id: effect.bank_delta_minor
            for effect in world.effects
            if effect.primary_ref.record_type is RecordType.BANK_TXN
        }

        split_txn_ids = [
            record.record_ref.record_id
            for record in dataset.truth.records
            if record.anomaly_class is AnomalyClass.SPLIT_PAYOUT
        ]
        txns = {txn.txn_id: txn for txn in world.bank_txns}
        settlements = world.settlements_by_id()

        for txn_id in split_txn_ids:
            settlement_id = txns[txn_id].settlement_id
            assert settlement_id is not None
            credits = world.credits_for_settlement(settlement_id)
            assert len(credits) == 2
            declared_delta = sum(deltas.get(credit.txn_id, 0) for credit in credits)
            assert sum(c.credit_minor for c in credits) == (
                settlements[settlement_id].declared_net_minor(payments) + declared_delta
            )


class TestNoFloatsSurviveGeneration:
    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_every_generated_money_value_is_an_int(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """The money invariant has to survive the generator, not just the models."""
        dataset = _dataset(seed, difficulty, order_count)
        world = dataset.world
        for order in world.orders:
            assert type(order.amount_minor) is int
        for payment in world.payments:
            assert type(payment.amount_minor) is int
        for txn in world.bank_txns:
            assert type(txn.credit_minor) is int
            assert type(txn.debit_minor) is int
        for settlement in world.settlements:
            assert type(settlement.adjustments_minor) is int
            assert type(settlement.net_mismatch_minor) is int
        for link in dataset.truth.links:
            assert type(link.amount_minor) is int


class TestStructuralInvariants:
    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_no_credit_is_negative(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """A payout that claws back more than it pays is not a modelled anomaly."""
        world = _dataset(seed, difficulty, order_count).world
        for txn in world.bank_txns:
            assert txn.credit_minor >= 0
            assert txn.debit_minor >= 0

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_every_settlement_keeps_at_least_one_credited_payment(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """A chargeback must never strip a batch of everything it carries."""
        world = _dataset(seed, difficulty, order_count).world
        for settlement in world.settlements:
            covered = [
                pid
                for txn in world.credits_for_settlement(settlement.settlement_id)
                for pid in txn.covered_payment_ids
            ]
            assert covered, settlement.settlement_id

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_no_payment_is_credited_by_two_transactions(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """Partitioned splits, not proportional smearing: one payment, one credit."""
        world = _dataset(seed, difficulty, order_count).world
        seen: set[str] = set()
        for txn in world.bank_txns:
            for pid in txn.covered_payment_ids:
                assert pid not in seen, pid
                seen.add(pid)

    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_record_ids_are_unique(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        world = _dataset(seed, difficulty, order_count).world
        for items in (
            [o.order_id for o in world.orders],
            [p.payment_id for p in world.payments],
            [s.settlement_id for s in world.settlements],
            [t.txn_id for t in world.bank_txns],
        ):
            assert len(items) == len(set(items))


class TestDeterminism:
    @GENERATOR_SETTINGS
    @given(seeds, difficulties, order_counts)
    def test_generation_is_a_pure_function_of_its_config(
        self, seed: int, difficulty: Difficulty, order_count: int
    ):
        """Anyone can reproduce every number in the report."""
        first = _dataset(seed, difficulty, order_count)
        second = _dataset(seed, difficulty, order_count)
        assert first.truth.links == second.truth.links
        assert first.truth.records == second.truth.records
        assert first.truth.scenario_draws == second.truth.scenario_draws


class TestAllocationBacksTheSplit:
    @given(
        st.integers(min_value=1, max_value=10**10),
        st.lists(st.integers(min_value=1, max_value=10**8), min_size=2, max_size=2),
    )
    def test_two_way_allocation_conserves(self, total: int, weights: list[int]):
        """The primitive A09 relies on, exercised directly at the split's shape."""
        parts = allocate_minor(total, weights)
        assert sum(parts) == total
        assert len(parts) == 2
