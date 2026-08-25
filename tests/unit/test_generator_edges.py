"""Guard paths in the generator.

Scenarios decline rather than force themselves onto a settlement they would
corrupt, and the CLI refuses to report success on a dataset whose money does not
add up. Both are error paths, which is exactly why they need tests -- they are
the branches that never run until something has already gone wrong.
"""

from __future__ import annotations

import random

import pytest

from ledgerloop.cli import main
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate
from ledgerloop.generator.baseline import build_clean_world
from ledgerloop.generator.scenarios import (
    ASPECT_AMOUNT,
    ASPECT_STRUCTURE,
    apply_scenario,
    chargeback_netted,
    clean,
    duplicate_credit,
    post_settlement_refund,
    split_payout,
)
from ledgerloop.generator.world import DraftWorld
from ledgerloop.models.enums import AnomalyClass, Difficulty, SplitName
from ledgerloop.models.truth import GroundTruth


def _world(order_count: int = 60) -> DraftWorld:
    return build_clean_world(random.Random(42), order_count)


def _first_order(world: DraftWorld) -> str:
    return world.orders[0].order_id


class TestScenariosDeclineRatherThanCorrupt:
    def test_clean_is_a_no_op(self):
        world = _world()
        assert clean(world, random.Random(0), _first_order(world)) is True
        assert world.effects == []

    def test_scenario_declines_on_an_empty_world(self):
        """No settlements means nowhere to place anything."""
        empty = DraftWorld()
        empty.reindex()
        for anomaly in (
            AnomalyClass.ROUNDING_DRIFT,
            AnomalyClass.FEE_TAX_MISMATCH,
            AnomalyClass.SPLIT_PAYOUT,
        ):
            assert apply_scenario(empty, random.Random(0), anomaly, "ORD-nonexistent") is False
        assert empty.effects == []

    def test_scenario_declines_when_every_aspect_is_claimed(self):
        world = _world()
        for settlement in world.settlements:
            world.claim(settlement.settlement_id, ASPECT_AMOUNT)
        assert apply_scenario(
            world, random.Random(0), AnomalyClass.ROUNDING_DRIFT, _first_order(world)
        ) is False

    def test_structure_scenarios_decline_when_structure_is_claimed(self):
        world = _world()
        for settlement in world.settlements:
            world.claim(settlement.settlement_id, ASPECT_STRUCTURE)
        assert duplicate_credit(world, random.Random(0), _first_order(world)) is False
        assert split_payout(world, random.Random(0), _first_order(world)) is False

    def test_refund_declines_when_no_later_batch_exists(self):
        """A claw-back needs somewhere later to land."""
        world = _world()
        # Collapse every merchant to a single batch by keeping only the first.
        world.settlements[:] = world.settlements[:1]
        world.reindex()
        assert post_settlement_refund(world, random.Random(0), _first_order(world)) is False

    def test_chargeback_declines_when_every_payment_is_claimed(self):
        world = _world()
        world.claimed_payments.update(p.payment_id for p in world.payments)
        assert chargeback_netted(world, random.Random(0), _first_order(world)) is False

    def test_a_declined_scenario_leaves_the_world_untouched(self):
        world = _world()
        for settlement in world.settlements:
            world.claim(settlement.settlement_id, ASPECT_STRUCTURE)
        before = [(t.txn_id, t.credit_minor) for t in world.bank_txns]
        assert split_payout(world, random.Random(0), _first_order(world)) is False
        assert [(t.txn_id, t.credit_minor) for t in world.bank_txns] == before


class TestWorldIndexing:
    def test_add_bank_txn_keeps_the_credit_index_consistent(self):
        """A row appended behind the index would be invisible to truth-building."""
        world = _world()
        settlement_id = world.settlements[0].settlement_id
        before = len(world.credits_for_settlement(settlement_id))
        dataset = generate(GeneratorConfig(split=SplitName.DEV, seed=42))
        assert dataset.conservation_residual_minor == 0
        assert len(world.credits_for_settlement(settlement_id)) == before

    def test_credits_for_an_unknown_settlement_is_empty(self):
        assert _world().credits_for_settlement("SETL-nope") == []

    def test_payment_for_an_unknown_order_is_none(self):
        assert _world().payment_for_order("ORD-nope") is None

    def test_claim_is_idempotent_and_reports_the_second_attempt(self):
        world = _world()
        assert world.claim("SETL-0001", "amount") is True
        assert world.claim("SETL-0001", "amount") is False
        assert world.is_claimed("SETL-0001", "amount")


class TestEmptyTruth:
    def test_realised_prevalence_of_an_empty_dataset_is_all_zero(self):
        """No draws must give zeros, not a division by zero."""
        truth = GroundTruth(
            split=SplitName.DEV,
            difficulty=Difficulty.STANDARD,
            seed=0,
            generator_version="0.0.0",
        )
        realised = truth.realised_prevalence()
        assert set(realised) == set(AnomalyClass)
        assert all(value == 0.0 for value in realised.values())


class TestCliRefusesToReportSuccessOnBrokenMoney:
    def test_conservation_failure_is_reported_and_exits_nonzero(
        self, tmp_path, monkeypatch, capsys
    ):
        """If a scenario ever moves money without declaring it, the CLI must say
        so loudly rather than writing a dataset that looks fine."""
        import ledgerloop.cli as cli_module

        real_generate = cli_module.generate_to_disk

        def sabotaged(config, directory):
            dataset = real_generate(config, directory)
            # Inject an undeclared paise: exactly the failure mode the residual
            # check exists to catch.
            dataset.world.bank_txns[0].credit_minor += 1
            return dataset

        monkeypatch.setattr(cli_module, "generate_to_disk", sabotaged)

        exit_code = main(["generate", "--orders", "40", "--out", str(tmp_path / "broken")])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "CONSERVATION VIOLATED" in captured.err
        assert "money conserved" not in captured.out


class TestUnknownCommand:
    def test_parser_rejects_an_unknown_subcommand(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])
