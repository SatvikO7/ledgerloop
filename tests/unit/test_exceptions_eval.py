"""Scoring the queue: recall, the rectangular confusion matrix, the floor.

The definition being pinned here is the one PLAN.md §9.1 names but never states:
**exception recall over the records ground truth calls exceptions**, with the
unmatchable floor on its own line. Both halves matter --

* counting unmatchable records inside recall would let a system inflate the
  headline by describing items nobody could resolve;
* dropping them entirely would hide the one number that distinguishes a real
  ceiling from a model failure.

The confusion matrix is rectangular by construction (`ARCHITECTURE.md` §6, 5),
and its rows may overlap, so the tests assert both rather than checking that it
sums to the queue size.
"""

from __future__ import annotations

import pytest

from ledgerloop.eval.metrics import (
    covered_refs,
    exception_confusion,
    exception_coverage,
    exception_impact_minor,
    exceptions_by_class,
)
from ledgerloop.models.enums import (
    AnomalyClass,
    Difficulty,
    ExceptionClass,
    ExpectedStatus,
    Severity,
    SplitName,
)
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.models.truth import GroundTruth, GroundTruthRecord


def record(ref, status: ExpectedStatus, anomaly: AnomalyClass = AnomalyClass.CLEAN):
    return GroundTruthRecord(record_ref=ref, expected_status=status, anomaly_class=anomaly)


def truth_with(*records) -> GroundTruth:
    return GroundTruth(
        split=SplitName.TEST,
        difficulty=Difficulty.STANDARD,
        seed=1,
        generator_version="0.2.0",
        links=(),
        records=tuple(records),
    )


def exception(
    *refs,
    exception_class: ExceptionClass = ExceptionClass.UNKNOWN_RESIDUAL,
    impact: int = 100,
) -> ReconException:
    return ReconException(
        exception_id=f"exception:{refs[0].key}",
        exception_class=exception_class,
        severity=Severity.LOW,
        impact_minor=impact,
        involved_refs=tuple(refs),
        root_cause="x.",
        suggested_action="y",
        classification_confidence=0.8,
    )


class TestCoverage:
    def test_a_covered_expected_record_counts(self):
        truth = truth_with(
            record(settlement_ref("SETL-1"), ExpectedStatus.EXCEPTION),
        )
        coverage = exception_coverage([exception(settlement_ref("SETL-1"))], truth)
        assert coverage.recall == 1.0
        assert coverage.missed == frozenset()

    def test_a_record_named_anywhere_in_the_chain_is_covered(self):
        """A chargeback naming payment, settlement and order told the controller all three."""
        truth = truth_with(record(payment_ref("PAY-1"), ExpectedStatus.EXCEPTION))
        item = exception(settlement_ref("SETL-1"), payment_ref("PAY-1"))
        assert exception_coverage([item], truth).recall == 1.0

    def test_an_uncovered_expected_record_is_named(self):
        truth = truth_with(
            record(settlement_ref("SETL-1"), ExpectedStatus.EXCEPTION),
            record(settlement_ref("SETL-2"), ExpectedStatus.EXCEPTION),
        )
        coverage = exception_coverage([exception(settlement_ref("SETL-1"))], truth)
        assert coverage.recall == pytest.approx(0.5)
        assert coverage.missed == frozenset({"settlement:SETL-2"})

    def test_unmatchable_records_are_reported_apart_from_the_recall(self):
        truth = truth_with(
            record(settlement_ref("SETL-1"), ExpectedStatus.EXCEPTION),
            record(bank_ref("BNK-1"), ExpectedStatus.UNMATCHABLE),
        )
        coverage = exception_coverage(
            [exception(settlement_ref("SETL-1")), exception(bank_ref("BNK-1"))], truth
        )
        assert coverage.recall == 1.0
        assert len(coverage.expected) == 1
        assert coverage.unmatchable_recall == 1.0
        assert len(coverage.unmatchable) == 1

    def test_describing_only_unmatchable_items_earns_no_recall(self):
        """The inflation this split exists to prevent."""
        truth = truth_with(
            record(settlement_ref("SETL-1"), ExpectedStatus.EXCEPTION),
            *[record(bank_ref(f"BNK-{i}"), ExpectedStatus.UNMATCHABLE) for i in range(50)],
        )
        coverage = exception_coverage(
            [exception(bank_ref(f"BNK-{i}")) for i in range(50)], truth
        )
        assert coverage.recall == 0.0
        assert coverage.unmatchable_recall == 1.0

    def test_matched_records_are_in_neither_denominator(self):
        truth = truth_with(record(settlement_ref("SETL-1"), ExpectedStatus.MATCHED))
        coverage = exception_coverage([], truth)
        assert coverage.expected == frozenset()
        assert coverage.unmatchable == frozenset()
        assert coverage.recall == 0.0

    def test_out_of_scope_records_leave_both_denominators_and_are_counted(self):
        truth = truth_with(
            record(bank_ref("BNK-DEBIT"), ExpectedStatus.UNMATCHABLE),
            record(bank_ref("BNK-1"), ExpectedStatus.UNMATCHABLE),
        )
        coverage = exception_coverage(
            [exception(bank_ref("BNK-1"))],
            truth,
            out_of_scope=frozenset({"bank_txn:BNK-DEBIT"}),
        )
        assert coverage.out_of_scope == 1
        assert coverage.unmatchable == frozenset({"bank_txn:BNK-1"})
        assert coverage.unmatchable_recall == 1.0

    def test_an_empty_denominator_reads_zero_not_perfect(self):
        assert exception_coverage([], truth_with()).recall == 0.0

    def test_covered_refs_collects_the_whole_chain(self):
        item = exception(settlement_ref("SETL-1"), payment_ref("PAY-1"), bank_ref("BNK-1"))
        assert covered_refs([item]) == frozenset(
            {"settlement:SETL-1", "payment:PAY-1", "bank_txn:BNK-1"}
        )


class TestTheConfusionMatrix:
    def test_it_maps_true_anomaly_to_predicted_class(self):
        truth = truth_with(
            record(
                settlement_ref("SETL-1"),
                ExpectedStatus.EXCEPTION,
                AnomalyClass.FEE_TAX_MISMATCH,
            )
        )
        matrix = exception_confusion(
            [
                exception(
                    settlement_ref("SETL-1"),
                    exception_class=ExceptionClass.FEE_TAX_MISMATCH,
                )
            ],
            truth,
        )
        assert matrix == {"A03_FEE_TAX_MISMATCH": {"E_FEE_TAX_MISMATCH": 1}}

    def test_it_is_rectangular_not_an_identity(self):
        """The two vocabularies answer different questions -- 11 against 13."""
        truth = truth_with(
            record(
                bank_ref("BNK-1"),
                ExpectedStatus.UNMATCHABLE,
                AnomalyClass.ORPHAN_BANK_CREDIT,
            )
        )
        matrix = exception_confusion(
            [exception(bank_ref("BNK-1"), exception_class=ExceptionClass.UNMATCHABLE)],
            truth,
        )
        assert matrix == {"A10_ORPHAN_BANK_CREDIT": {"E_UNMATCHABLE": 1}}

    def test_one_exception_is_attributed_to_every_anomaly_it_touches(self):
        truth = truth_with(
            record(
                settlement_ref("SETL-1"),
                ExpectedStatus.EXCEPTION,
                AnomalyClass.CHARGEBACK_NETTED,
            ),
            record(
                payment_ref("PAY-1"),
                ExpectedStatus.EXCEPTION,
                AnomalyClass.POST_SETTLEMENT_REFUND,
            ),
        )
        matrix = exception_confusion(
            [exception(settlement_ref("SETL-1"), payment_ref("PAY-1"))], truth
        )
        assert len(matrix) == 2
        assert sum(sum(row.values()) for row in matrix.values()) == 2

    def test_an_exception_on_records_truth_calls_clean_lands_in_the_clean_row(self):
        truth = truth_with(record(bank_ref("BNK-1"), ExpectedStatus.UNMATCHABLE))
        matrix = exception_confusion([exception(bank_ref("BNK-1"))], truth)
        assert "A01_CLEAN" in matrix

    def test_rows_come_out_in_anomaly_order(self):
        truth = truth_with(
            record(bank_ref("BNK-1"), ExpectedStatus.EXCEPTION, AnomalyClass.SPLIT_PAYOUT),
            record(
                bank_ref("BNK-2"), ExpectedStatus.EXCEPTION, AnomalyClass.ROUNDING_DRIFT
            ),
        )
        matrix = exception_confusion(
            [exception(bank_ref("BNK-1")), exception(bank_ref("BNK-2"))], truth
        )
        assert list(matrix) == ["A02_ROUNDING_DRIFT", "A09_SPLIT_PAYOUT"]

    def test_an_empty_queue_produces_an_empty_matrix(self):
        assert exception_confusion([], truth_with()) == {}


class TestTheSummaries:
    def test_counts_come_out_in_taxonomy_order(self):
        items = [
            exception(bank_ref("BNK-1"), exception_class=ExceptionClass.UNMATCHABLE),
            exception(bank_ref("BNK-2"), exception_class=ExceptionClass.ROUNDING_DRIFT),
            exception(bank_ref("BNK-3"), exception_class=ExceptionClass.ROUNDING_DRIFT),
        ]
        counts = exceptions_by_class(items)
        assert list(counts) == [
            ExceptionClass.ROUNDING_DRIFT,
            ExceptionClass.UNMATCHABLE,
        ]
        assert counts[ExceptionClass.ROUNDING_DRIFT] == 2

    def test_the_total_impact_goes_through_the_money_gate(self):
        items = [
            exception(bank_ref("BNK-1"), impact=100),
            exception(bank_ref("BNK-2"), impact=250),
        ]
        assert exception_impact_minor(items) == 350

    def test_an_empty_queue_is_worth_nothing(self):
        assert exception_impact_minor([]) == 0
        assert exceptions_by_class([]) == {}
