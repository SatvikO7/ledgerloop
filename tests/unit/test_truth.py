"""Ground-truth contract tests.

The point of these is narrow and important: prove the link-level schema can
represent the two anomaly classes the plan's original flat-row schema could
not (A05 DUPLICATE_CREDIT and A09 SPLIT_PAYOUT), and pin down the evaluation
surface the metrics will be computed against.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledgerloop.models import (
    AnomalyClass,
    Difficulty,
    ExpectedStatus,
    GroundTruth,
    GroundTruthLink,
    GroundTruthRecord,
    LinkType,
    SplitName,
    bank_ref,
    order_ref,
    payment_ref,
    settlement_ref,
)
from ledgerloop.money import allocate_minor


def _truth(**overrides) -> GroundTruth:
    kwargs = {
        "split": SplitName.DEV,
        "difficulty": Difficulty.STANDARD,
        "seed": 42,
        "generator_version": "0.1.0",
    }
    kwargs.update(overrides)
    return GroundTruth(**kwargs)


def _credited(payment: str, bank: str, amount: int, anomaly=AnomalyClass.CLEAN):
    return GroundTruthLink(
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref(payment),
        target_ref=bank_ref(bank),
        amount_minor=amount,
        anomaly_class=anomaly,
    )


class TestEvaluationSurface:
    def test_only_payment_credited_as_links_are_evaluated(self):
        """Structural edges are largely given by the sources; counting them
        would inflate every score with edges the system never worked for."""
        truth = _truth(
            links=(
                _credited("PAY-1", "BNK-1", 499_900),
                GroundTruthLink(
                    link_type=LinkType.ORDER_PAID_BY,
                    source_ref=order_ref("ORD-1"),
                    target_ref=payment_ref("PAY-1"),
                    amount_minor=499_900,
                ),
                GroundTruthLink(
                    link_type=LinkType.PAYMENT_SETTLED_IN,
                    source_ref=payment_ref("PAY-1"),
                    target_ref=settlement_ref("SETL-1"),
                    amount_minor=499_900,
                ),
            )
        )
        assert truth.evaluation_pairs == {("payment:PAY-1", "bank_txn:BNK-1")}

    def test_links_are_grouped_by_type(self):
        truth = _truth(links=(_credited("PAY-1", "BNK-1", 100),))
        assert len(truth.links_by_type[LinkType.PAYMENT_CREDITED_AS]) == 1
        assert LinkType.ORDER_PAID_BY not in truth.links_by_type


class TestSplitPayoutIsRepresentable:
    """A09: one settlement arrives as TWO bank credits.

    The plan's flat `bank_txn_id` column had nowhere to put the second.
    """

    def test_one_payment_may_credit_two_bank_transactions(self):
        parts = allocate_minor(3_680_323, [1, 1])
        truth = _truth(
            links=(
                _credited("PAY-1", "BNK-1", parts[0], AnomalyClass.SPLIT_PAYOUT),
                _credited("PAY-1", "BNK-2", parts[1], AnomalyClass.SPLIT_PAYOUT),
            )
        )
        assert truth.evaluation_pairs == {
            ("payment:PAY-1", "bank_txn:BNK-1"),
            ("payment:PAY-1", "bank_txn:BNK-2"),
        }

    def test_split_parts_sum_to_the_settlement_net(self):
        """Money conservation across the split."""
        net = 3_680_323
        parts = allocate_minor(net, [1, 1])
        truth = _truth(
            links=(
                _credited("PAY-1", "BNK-1", parts[0], AnomalyClass.SPLIT_PAYOUT),
                _credited("PAY-1", "BNK-2", parts[1], AnomalyClass.SPLIT_PAYOUT),
            )
        )
        assert sum(link.amount_minor for link in truth.links) == net


class TestDuplicateCreditIsRepresentable:
    """A05: the same UTR credited twice. The duplicate must match nothing."""

    def test_duplicate_credit_has_no_truth_link(self):
        truth = _truth(
            links=(_credited("PAY-1", "BNK-1", 499_900),),
            records=(
                GroundTruthRecord(
                    record_ref=bank_ref("BNK-1"),
                    expected_status=ExpectedStatus.MATCHED,
                    anomaly_class=AnomalyClass.CLEAN,
                ),
                GroundTruthRecord(
                    record_ref=bank_ref("BNK-2"),
                    expected_status=ExpectedStatus.EXCEPTION,
                    anomaly_class=AnomalyClass.DUPLICATE_CREDIT,
                    impact_minor=499_900,
                    note="same UTR as BNK-1, credited twice",
                ),
            ),
        )
        credited_banks = {pair[1] for pair in truth.evaluation_pairs}
        assert "bank_txn:BNK-2" not in credited_banks
        assert truth.verdict_by_ref["bank_txn:BNK-2"].anomaly_class is (
            AnomalyClass.DUPLICATE_CREDIT
        )


class TestUnmatchableFloor:
    def test_unmatchable_records_are_excluded_from_the_denominator(self):
        """Charging the system for irreconcilable items is no more honest
        than excusing it from real failures."""
        truth = _truth(
            records=(
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-1"),
                    expected_status=ExpectedStatus.MATCHED,
                    anomaly_class=AnomalyClass.CLEAN,
                ),
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-2"),
                    expected_status=ExpectedStatus.EXCEPTION,
                    anomaly_class=AnomalyClass.CHARGEBACK_NETTED,
                    impact_minor=431_200,
                ),
                GroundTruthRecord(
                    record_ref=bank_ref("BNK-9"),
                    expected_status=ExpectedStatus.UNMATCHABLE,
                    anomaly_class=AnomalyClass.ORPHAN_BANK_CREDIT,
                    impact_minor=250_000,
                ),
            )
        )
        assert truth.reconcilable_refs == {"payment:PAY-1", "payment:PAY-2"}
        assert truth.unmatchable_refs == {"bank_txn:BNK-9"}

    def test_exceptions_remain_reconcilable(self):
        """An EXCEPTION is a resolvable item the system failed to match --
        it stays in the denominator. Only UNMATCHABLE leaves."""
        truth = _truth(
            records=(
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-2"),
                    expected_status=ExpectedStatus.EXCEPTION,
                    anomaly_class=AnomalyClass.CHARGEBACK_NETTED,
                ),
            )
        )
        assert "payment:PAY-2" in truth.reconcilable_refs


class TestImpactAndPrevalence:
    def test_impact_totals(self):
        truth = _truth(
            records=(
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-1"),
                    expected_status=ExpectedStatus.EXCEPTION,
                    anomaly_class=AnomalyClass.CHARGEBACK_NETTED,
                    impact_minor=431_200,
                ),
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-2"),
                    expected_status=ExpectedStatus.MATCHED,
                    anomaly_class=AnomalyClass.CLEAN,
                ),
            )
        )
        assert truth.impact_total_minor() == 431_200
        assert truth.impact_total_minor(["payment:PAY-2"]) == 0

    def test_anomaly_counts_cover_every_class(self):
        """Zero-count classes must appear, so a class that never generated is
        visible in the prevalence check rather than absent from the table."""
        truth = _truth(
            records=(
                GroundTruthRecord(
                    record_ref=payment_ref("PAY-1"),
                    expected_status=ExpectedStatus.MATCHED,
                    anomaly_class=AnomalyClass.CLEAN,
                ),
            )
        )
        counts = truth.anomaly_counts()
        assert len(counts) == len(AnomalyClass)
        assert counts[AnomalyClass.CLEAN] == 1
        assert counts[AnomalyClass.SPLIT_PAYOUT] == 0


class TestTruthIntegrity:
    def test_duplicate_verdicts_are_rejected(self):
        with pytest.raises(ValidationError, match="duplicate ground-truth verdict"):
            _truth(
                records=(
                    GroundTruthRecord(
                        record_ref=payment_ref("PAY-1"),
                        expected_status=ExpectedStatus.MATCHED,
                        anomaly_class=AnomalyClass.CLEAN,
                    ),
                    GroundTruthRecord(
                        record_ref=payment_ref("PAY-1"),
                        expected_status=ExpectedStatus.EXCEPTION,
                        anomaly_class=AnomalyClass.TIMING_SHIFT,
                    ),
                )
            )

    def test_generator_version_is_required(self):
        """A metric is only comparable to one from the same generator version."""
        with pytest.raises(ValidationError):
            GroundTruth(split=SplitName.DEV, difficulty=Difficulty.STANDARD, seed=1)

    def test_links_reject_float_amounts(self):
        with pytest.raises(ValidationError, match="float is forbidden"):
            GroundTruthLink(
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref("PAY-1"),
                target_ref=bank_ref("BNK-1"),
                amount_minor=4999.0,
            )
