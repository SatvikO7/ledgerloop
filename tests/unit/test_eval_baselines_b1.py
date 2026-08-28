"""B1 -- exact join plus fuzzy recovery plus nearest amount.

Two jobs, the same two B0's tests have. Prove the fuzzy stages *work*, because
a baseline that is merely broken proves nothing about the system that beats it;
then pin the specific ways they fail, because those failures are the argument
for T2's uniqueness counting and T3's margin gate.

B1's stages are exercised through the module defaults wherever possible. The
parameters exist so a test can isolate one stage, never so a test can find the
setting that makes the baseline look worse.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.baselines import (
    B1_AMOUNT_BPS,
    B1_DATE_WINDOW_DAYS,
    B1_FUZZY_THRESHOLD,
    run_b0,
    run_b1,
)
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.generator.emitters import BANK_FILE, PSP_FILE
from ledgerloop.models.enums import SplitName


def _write_sources(directory: Path, settlements, bank_rows) -> Path:
    """Write only the two files the baselines read. No generator involved."""
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


def _settlement(settlement_id, utr, payments, *, net=None, settled_on="2026-03-18"):
    gross = sum(amount for _, amount in payments)
    return {
        "settlement_id": settlement_id,
        "utr": utr,
        "net_paise": gross if net is None else net,
        "gross_paise": gross,
        "settled_on": settled_on,
        "payments": [
            {"payment_id": payment_id, "amount_paise": amount}
            for payment_id, amount in payments
        ],
    }


def _credit(txn_id, narration, amount, *, value_date="18/03/2026"):
    return [txn_id, value_date, narration, amount, 0, 0]


def _pairs(run):
    return {link.pair for link in run.predictions}


class TestB1StartsFromB0:
    def test_the_exact_join_is_the_same_join(self, tmp_path):
        """B1 layers on B0 rather than reimplementing it, so any difference
        between the two rows is the fuzzy stages and nothing else."""
        directory = _write_sources(
            tmp_path / "clean",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 100), ("PAY-2", 200)])],
            [_credit("BNK-1", "NEFT CR-ACME-UTR2026031712345-SETTLEMENT", 300)],
        )
        assert _pairs(run_b1(directory)) == _pairs(run_b0(directory))

    def test_b1_never_asserts_fewer_links_than_b0(self, tmp_path):
        """Its stages only run over what the join left open."""
        directory = tmp_path / "corpus"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=7), directory)
        assert _pairs(run_b0(directory)) <= _pairs(run_b1(directory))


class TestTheFuzzyReferenceStage:
    def test_a_transposed_reference_still_joins(self, tmp_path):
        """The stage exists for a damaged reference and recovers one."""
        directory = _write_sources(
            tmp_path / "damaged",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 500)])],
            # One digit dropped: the exact join misses, the fuzzy stage does not.
            [_credit("BNK-1", "NEFT CR-ACME-UTR202603171234-SETTLEMENT", 500)],
        )
        assert _pairs(run_b0(directory)) == set()
        assert _pairs(run_b1(directory)) == {("payment:PAY-1", "bank_txn:BNK-1")}

    def test_an_unrelated_narration_is_not_recovered(self, tmp_path):
        """A stage that matched anything would make B1 a strawman."""
        directory = _write_sources(
            tmp_path / "unrelated",
            [_settlement("SETL-1", "UTR2026031712345", [("PAY-1", 500)])],
            [_credit("BNK-1", "SALARY CREDIT PAYROLL BATCH", 999_999)],
        )
        assert _pairs(run_b1(directory)) == set()

    def test_the_threshold_is_a_documented_constant_not_a_tuned_one(self):
        """Pinned so a later change is a deliberate edit with a diff, rather
        than a quiet improvement to a number the report compares against."""
        assert B1_FUZZY_THRESHOLD == 85.0
        assert B1_AMOUNT_BPS == 100
        assert B1_DATE_WINDOW_DAYS == 5


class TestTheNearestAmountStage:
    def test_a_referenceless_credit_matches_the_nearest_net(self, tmp_path):
        directory = _write_sources(
            tmp_path / "nearest",
            [_settlement("SETL-1", None, [("PAY-1", 100_000)], net=99_000)],
            [_credit("BNK-1", "CR/NEFT/ACME RETAIL/BULK", 99_000)],
        )
        assert _pairs(run_b1(directory)) == {("payment:PAY-1", "bank_txn:BNK-1")}

    def test_it_takes_the_argmax_with_no_ambiguity_check(self, tmp_path):
        """The absence of the check is the measurement.

        Two settlements sit the same distance from one credit. T2 would count
        both subsets and refuse; B1 sorts and takes the first, so exactly one of
        these two links is asserted and it is a coin flip which.
        """
        directory = _write_sources(
            tmp_path / "ambiguous",
            [
                _settlement("SETL-1", None, [("PAY-1", 100_000)], net=100_000),
                _settlement("SETL-2", None, [("PAY-2", 100_000)], net=100_000),
            ],
            [_credit("BNK-1", "CR/NEFT/UNKNOWN/BULK", 100_000)],
        )
        asserted = _pairs(run_b1(directory))
        assert len(asserted) == 1
        assert asserted < {("payment:PAY-1", "bank_txn:BNK-1"), ("payment:PAY-2", "bank_txn:BNK-1")}

    def test_a_credit_outside_the_date_window_is_left_alone(self, tmp_path):
        directory = _write_sources(
            tmp_path / "late",
            [_settlement("SETL-1", None, [("PAY-1", 100_000)], settled_on="2026-03-01")],
            [_credit("BNK-1", "CR/NEFT/ACME/BULK", 100_000, value_date="18/03/2026")],
        )
        assert _pairs(run_b1(directory)) == set()

    def test_a_credit_outside_the_amount_band_is_left_alone(self, tmp_path):
        directory = _write_sources(
            tmp_path / "wide",
            [_settlement("SETL-1", None, [("PAY-1", 100_000)], net=100_000)],
            [_credit("BNK-1", "CR/NEFT/ACME/BULK", 150_000)],
        )
        assert _pairs(run_b1(directory)) == set()

    def test_one_settlement_is_claimed_by_one_credit(self, tmp_path):
        """Greedy, best-fit-first. The charitable reading of what a script does,
        and it keeps B1's false-positive count from being inflated by an error a
        careless implementation would make but a typical one would not."""
        directory = _write_sources(
            tmp_path / "greedy",
            [_settlement("SETL-1", None, [("PAY-1", 100_000)], net=100_000)],
            [
                _credit("BNK-1", "CR/NEFT/ACME/BULK", 100_000),
                _credit("BNK-2", "CR/NEFT/ACME/BULK", 100_050),
            ],
        )
        asserted = _pairs(run_b1(directory))
        assert asserted == {("payment:PAY-1", "bank_txn:BNK-1")}


class TestB1IsDeterministic:
    def test_two_runs_over_one_corpus_assert_the_same_links(self, tmp_path):
        directory = tmp_path / "corpus"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=11), directory)
        first, second = run_b1(directory), run_b1(directory)
        assert first.predictions == second.predictions


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """B1 run once over the `test` split, shared by every corpus assertion.

    Module-scoped and a plain function: a class-scoped fixture defined as an
    instance method is deprecated in pytest 8 and warns.
    """
    directory = tmp_path_factory.mktemp("b1") / "test-split"
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    truth = load_ground_truth(directory)
    run = run_b1(directory)
    return run, truth, evaluate(run.predictions, truth, run_id="b1-test-42")


class TestB1OnTheCorpus:
    def test_it_reaches_further_than_the_exact_join(self, scored):
        """The point of the row: fuzzy matching buys recall."""
        _, _, metrics = scored
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.recall > 0.80

    def test_and_is_wrong_more_often_in_absolute_terms(self, scored):
        """And the point of the column beside it: reach costs false positives,
        priced in rupees rather than as a ratio."""
        _, _, metrics = scored
        links = metrics.link_metrics
        assert links is not None
        assert links.false_positives > 100
        assert links.false_positive_cost_minor > 30_00_000_00

    def test_its_precision_is_far_below_the_target(self, scored):
        """PLAN.md §9.1 asks for >= 0.99. B1 is the typical submission and is
        nowhere near it -- which is the comparison, not a criticism of B1."""
        _, _, metrics = scored
        assert metrics.auto_match_precision < 0.75
