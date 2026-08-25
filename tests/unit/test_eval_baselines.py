"""B0, the exact-join baseline.

Two jobs. First, prove the join works at all -- a baseline that is merely broken
proves nothing about the system that beats it. Second, pin the *specific* ways
it fails, because those failures are the specification for tiers T1-T5 and the
content of the "why not just SQL" answer.

The failure tests are written against hand-built two-settlement datasets rather
than against the generated corpus, so each one isolates a single anomaly class.
The corpus test then checks that the same failures show up at scale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.baselines import UTR_PATTERN, run_b0
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.generator.emitters import BANK_FILE, PSP_FILE
from ledgerloop.models.enums import AnomalyClass, SplitName

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


def _write_sources(directory: Path, settlements, bank_rows) -> Path:
    """Write only the two files B0 reads. No generator involved."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / PSP_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump({"settlements": settlements}, handle)
    with (directory / BANK_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["txn_id", "value_date", "narration", "credit_paise", "debit_paise", "balance_paise"]
        )
        writer.writerows(bank_rows)
    return directory


def _settlement(settlement_id: str, utr, payments) -> dict:
    return {
        "settlement_id": settlement_id,
        "utr": utr,
        "payments": [
            {"payment_id": payment_id, "amount_paise": amount}
            for payment_id, amount in payments
        ],
    }


def _credit(txn_id: str, narration: str, amount: int) -> list:
    return [txn_id, "18/03/2026", narration, amount, 0, 0]


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """B0 run once over the `test` split, shared by every corpus assertion."""
    directory = tmp_path_factory.mktemp("b0") / "test-split"
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    truth = load_ground_truth(directory)
    run = run_b0(directory)
    return run, truth, evaluate(run.predictions, truth, run_id="b0-test-42")


class TestUtrPattern:
    @pytest.mark.parametrize(
        "narration",
        [
            "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026031745389-SETTLEMENT",
            "IMPS CR/UTR2026032629136/ZOMATO HYPERPURE PVT/PAYOUT",
            "RTGS CR NYKAA ERETAIL P L UTR2026031892729 SETTLEMENT",
            "CR/NEFT/UTR2026031892729/NYKAA ERETAIL P L",
        ],
    )
    def test_extracts_the_utr_from_every_generated_narration_shape(self, narration):
        """B0 is given every advantage; a regex that missed a shape the
        generator emits would understate the baseline and flatter the system."""
        found = UTR_PATTERN.search(narration)
        assert found is not None
        assert found.group(0).startswith("UTR")

    @pytest.mark.parametrize(
        "narration",
        [
            "NEFT CR-RAZORPAY SOFTWARE PVT-SETTLEMENT",
            "CR/NEFT/NYKAA E RETAIL PVT LTD/BULK",
            "RENT PAYMENT COMMERCIAL PREMISES",
            "SALARY CREDIT PAYROLL BATCH",
        ],
    )
    def test_finds_nothing_where_there_is_nothing(self, narration):
        assert UTR_PATTERN.search(narration) is None


class TestTheJoinWorks:
    def test_a_clean_settlement_joins_to_all_its_payments(self, tmp_path):
        directory = _write_sources(
            tmp_path / "clean",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100), ("PAY-2", 200)])],
            [_credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 294)],
        )
        run = run_b0(directory)
        assert {link.pair for link in run.predictions} == {
            ("payment:PAY-1", "bank_txn:BNK-1"),
            ("payment:PAY-2", "bank_txn:BNK-1"),
        }
        assert run.credits_joined == 1
        assert run.settlements_joined == 1

    def test_debits_are_never_joined(self, tmp_path):
        """A payout batch is money arriving. Matching one to an outgoing payment
        is not an error an exact join would plausibly make, and letting B0 make
        it would pad its false-positive count with a strawman."""
        directory = _write_sources(
            tmp_path / "debit",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100)])],
            [["BNK-1", "18/03/2026", "NEFT DR-UTR2026031712345-X", 0, 100, 0]],
        )
        run = run_b0(directory)
        assert run.predictions == ()
        assert run.credits_seen == 0

    def test_the_asserted_amount_is_the_payment_gross(self, tmp_path):
        """The join has no model of fees and no way to allocate a credit across
        the payments it carries -- which is why B0 over-reports reconciled money
        even where its links are right."""
        directory = _write_sources(
            tmp_path / "gross",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 500_000)])],
            [_credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 489_000)],
        )
        run = run_b0(directory)
        assert run.predictions[0].amount_minor == 500_000


class TestTheJoinFails:
    def test_a07_missing_reference_makes_the_settlement_unreachable(self, tmp_path):
        """No UTR in the narration means nothing to join on. Every payment in
        that settlement becomes a false negative."""
        directory = _write_sources(
            tmp_path / "a07",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100), ("PAY-2", 200)])],
            [_credit("BNK-1", "CR/NEFT/ACME CORP PVT LTD/BULK", 294)],
        )
        run = run_b0(directory)
        assert run.predictions == ()
        assert run.credits_with_utr == 0
        assert run.settlements_unjoined == 1

    def test_a05_duplicate_credit_credits_the_whole_batch_twice(self, tmp_path):
        """The join has no notion of "already credited"."""
        directory = _write_sources(
            tmp_path / "a05",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100), ("PAY-2", 200)])],
            [
                _credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 294),
                _credit("BNK-2", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 294),
            ],
        )
        run = run_b0(directory)
        assert len(run.predictions) == 4
        assert {link.target_ref.record_id for link in run.predictions} == {"BNK-1", "BNK-2"}

    def test_a09_split_payout_emits_the_full_cross_product(self, tmp_path):
        """The truth partitions the payments between the two credits; the join
        asserts every payment against both."""
        directory = _write_sources(
            tmp_path / "a09",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100), ("PAY-2", 200)])],
            [
                _credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 100),
                _credit("BNK-2", "IMPS CR/UTR2026031712345/ACME/PAYOUT", 194),
            ],
        )
        run = run_b0(directory)
        assert len(run.predictions) == 4

    def test_a10_orphan_credit_with_an_unknown_utr_joins_to_nothing(self, tmp_path):
        directory = _write_sources(
            tmp_path / "a10",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100)])],
            [_credit("BNK-9", "NEFT CR-ACME-UTR2026039999999-SETTLEMENT", 500)],
        )
        run = run_b0(directory)
        assert run.predictions == ()
        assert run.credits_with_utr == 1
        assert run.credits_joined == 0

    def test_a_settlement_with_no_utr_is_never_joined(self, tmp_path):
        directory = _write_sources(
            tmp_path / "noutr",
            [_settlement("SETL-1", None, [("PAY-1", 100)])],
            [_credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 100)],
        )
        run = run_b0(directory)
        assert run.predictions == ()
        assert run.settlements_unjoined == 1

    def test_a_utr_collision_returns_both_settlements(self, tmp_path):
        """Nothing enforces UTR uniqueness. Silently keeping one would hide the
        collision instead of letting the metrics show it."""
        directory = _write_sources(
            tmp_path / "collide",
            [
                _settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100)]),
                _settlement("SETL-2", "UTR2026031712345", [("PAY-2", 200)]),
            ],
            [_credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 294)],
        )
        run = run_b0(directory)
        assert len(run.predictions) == 2
        assert run.settlements_joined == 2


class TestOnTheGeneratedCorpus:
    def test_it_produces_a_non_trivial_number(self, scored):
        """A baseline that matched nothing, or everything, would be useless as a
        comparison point."""
        _, _, metrics = scored
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.true_positives > 0
        assert 0.0 < metrics.auto_match_precision < 1.0
        assert 0.0 < metrics.match_rate < 1.0

    def test_it_falls_far_short_of_the_precision_target(self, scored):
        """PLAN.md §9.1 targets >= 0.99. If an exact join reached that, the rest
        of this project would not need to exist."""
        _, _, metrics = scored
        assert metrics.auto_match_precision < 0.9

    def test_its_false_positives_carry_a_real_rupee_cost(self, scored):
        _, _, metrics = scored
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.false_positive_cost_minor > 0

    def test_missing_reference_recall_is_zero(self, scored):
        """The headline failure: no UTR, no join, no possibility of a match."""
        _, _, metrics = scored
        recalls = metrics.recall_by_anomaly_class
        assert recalls[AnomalyClass.MISSING_REFERENCE] == 0.0

    def test_classes_the_utr_join_does_not_touch_are_fully_recovered(self, scored):
        """Rounding drift and timing shifts do not disturb the reference, so an
        exact join still finds them. The baseline is not uniformly bad, and
        saying so is what makes the comparison honest."""
        _, _, metrics = scored
        recalls = metrics.recall_by_anomaly_class
        assert recalls[AnomalyClass.ROUNDING_DRIFT] == 1.0
        assert recalls[AnomalyClass.TIMING_SHIFT] == 1.0

    def test_it_is_deterministic(self, scored, tmp_path):
        run, _, _ = scored
        directory = tmp_path / "again"
        generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
        assert {link.pair for link in run_b0(directory).predictions} == {
            link.pair for link in run.predictions
        }

    def test_reconciled_and_outstanding_conserve_the_link_money(self, scored):
        _, truth, metrics = scored
        total = sum(
            link.amount_minor
            for link in truth.links
            if link.pair in truth.evaluation_pairs
        )
        assert metrics.reconciled_minor + metrics.outstanding_minor == total


class TestOnTheCommittedFixture:
    def test_the_fixture_exercises_the_duplicate_and_split_paths(self):
        """The fixture forces one effect per class, so B0's cross-product
        failures are guaranteed to appear -- which is what makes it the right
        dataset for pinning them."""
        run = run_b0(FIXTURE)
        truth = load_ground_truth(FIXTURE)
        metrics = evaluate(run.predictions, truth, run_id="b0-dev-42")
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.false_positives > 0
        assert metrics.link_metrics.false_negatives > 0
