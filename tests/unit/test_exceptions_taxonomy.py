"""The classification cascade, severity, and what counts as a residual item.

The rules are tested one at a time on corpora small enough to check by hand,
because the cascade's *order* is the argument (`taxonomy.py`'s docstring) and an
order can only be verified by constructing an item that two rules would both
accept and asserting which one wins.

The taxonomy is deliberately **not** the anomaly taxonomy, so nothing here
imports `AnomalyClass`. A test that asserted "A03 produces E_FEE_TAX_MISMATCH"
would be asserting the identity `ARCHITECTURE.md` §6, 5 says does not exist;
the mapping is measured by the confusion matrix, in the evaluator, afterwards.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ledgerloop.config import MatchingTolerances, RunConfig, SeverityThresholds
from ledgerloop.exceptions.taxonomy import (
    AGENT_RESOLVABLE_CLASSES,
    CreditItem,
    PaymentItem,
    SettlementItem,
    classify_credit,
    classify_payment,
    classify_settlement,
    residual_items,
    severity_for,
)
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.enums import ExceptionClass, Severity
from tests.unit.conftest import SETTLED_ON, bank_credit, batch, corpus, debit_row, noise_credit

CONFIG = RunConfig(run_id="tax-test")
WINDOW = MatchingTolerances().date_window_days


def context_for(ingest) -> MatchContext:
    return MatchContext.from_ingest(ingest)


def items(ingest, **kwargs):
    defaults = {
        "matched_settlements": frozenset(),
        "matched_credits": frozenset(),
        "matched_payments": frozenset(),
    }
    return residual_items(context_for(ingest), **{**defaults, **kwargs})


def settlement_item(ingest, **kwargs) -> SettlementItem:
    settlements, _, _ = items(ingest, **kwargs)
    return settlements[0]


def credit_item(ingest, txn_id: str = "BNK-00001", **kwargs) -> CreditItem:
    _, credits, _ = items(ingest, **kwargs)
    return next(item for item in credits if item.key == txn_id)


class TestWhatBecomesAnItem:
    def test_an_uncredited_payout_is_an_item(self, simple):
        settlements, _, _ = items(simple)
        assert [item.key for item in settlements] == ["SETL-0001"]
        assert not settlements[0].credited

    def test_a_credited_payout_that_closes_is_not_an_item(self, simple):
        settlements, _, _ = items(
            simple,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001", "PAY-00002"}),
        )
        assert settlements == ()

    def test_a_credited_payout_whose_identity_does_not_close_is_still_an_item(self):
        """The bank agrees with the net; the PSP's own arithmetic does not."""
        only = batch(net_minor=99_000)  # gross 100_000, no fee, so net is 1,000 off
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        settlements, _, _ = items(
            ingest,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001", "PAY-00002"}),
        )
        assert len(settlements) == 1
        assert settlements[0].credited

    def test_an_unclaimed_credit_is_an_item(self, simple):
        _, credits, _ = items(simple)
        assert [item.key for item in credits] == ["BNK-00001"]

    def test_a_claimed_credit_is_not(self, simple):
        _, credits, _ = items(simple, matched_credits=frozenset({"BNK-00001"}))
        assert credits == ()

    def test_a_debit_never_becomes_an_item(self):
        """Outgoing money is not a payout this system reconciles."""
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[only.credit(), debit_row("BNK-09002", utr=only.settlement.utr)],
        )
        _, credits, _ = items(ingest)
        assert all(item.key != "BNK-09002" for item in credits)

    def test_a_payment_left_out_of_a_credited_batch_is_an_item(self):
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        _, _, payments = items(
            ingest,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001"}),
        )
        assert [item.key for item in payments] == ["PAY-00002"]

    def test_payments_of_an_uncredited_batch_are_not_separate_items(self, simple):
        """The settlement item already names them; two rows for one problem is noise."""
        _, _, payments = items(simple)
        assert payments == ()


class TestTheSettlementCascade:
    def test_a_broken_identity_wins_over_everything_else(self):
        """The document contradicts itself; no weaker explanation is needed."""
        only = batch(net_minor=99_000, adjustments_minor=-40_000)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        item = settlement_item(ingest)
        assert classify_settlement(item, date_window_days=WINDOW) is (
            ExceptionClass.FEE_TAX_MISMATCH
        )

    def test_an_adjustment_matching_one_payment_is_a_chargeback(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.CHARGEBACK_NETTED

    def test_an_adjustment_matching_no_payment_is_a_refund_from_elsewhere(self):
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-777)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.POST_SETTLEMENT_REFUND

    def test_two_keyed_credits_that_do_not_add_up_is_an_incomplete_split(self):
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=10_000, utr=only.settlement.utr),
                bank_credit("BNK-00002", amount_minor=10_000, utr=only.settlement.utr),
            ],
        )
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.SPLIT_PAYOUT_INCOMPLETE

    def test_a_keyed_credit_short_of_the_net_is_a_refund(self):
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[
                bank_credit(
                    "BNK-00001",
                    amount_minor=only.net_minor - 5_000,
                    utr=only.settlement.utr,
                )
            ],
        )
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.POST_SETTLEMENT_REFUND

    def test_a_keyed_credit_outside_the_window_is_a_timing_shift(self):
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[only.credit(days_after=20)])
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.TIMING_SHIFT

    def test_an_unreferenced_credit_of_the_right_size_is_a_missing_reference(self):
        only = batch(utr=None)
        ingest = corpus(
            batches=[only],
            bank_txns=[bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None)],
        )
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.MISSING_REFERENCE

    def test_a_right_sized_credit_far_outside_the_window_is_a_timing_shift(self):
        only = batch(utr=None)
        ingest = corpus(
            batches=[only],
            bank_txns=[
                bank_credit(
                    "BNK-00001",
                    amount_minor=only.net_minor,
                    utr=None,
                    value_date=SETTLED_ON + timedelta(days=30),
                )
            ],
        )
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.TIMING_SHIFT

    def test_nothing_arriving_at_all_is_a_late_arrival(self):
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[noise_credit(amount_minor=7)])
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW
        ) is ExceptionClass.LATE_ARRIVAL

    def test_a_declared_ambiguity_overrides_the_whole_cascade(self):
        """T2 already concluded two subsets fit. This module does not re-litigate it."""
        only = batch(net_minor=99_000)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        assert classify_settlement(
            settlement_item(ingest), date_window_days=WINDOW, ambiguous=True
        ) is ExceptionClass.AMBIGUOUS_AGGREGATION

    def test_a_credited_payout_is_only_ever_a_fee_tax_mismatch(self):
        """None of the missing-credit rules can apply; the money arrived."""
        only = batch(net_minor=99_000, adjustments_minor=-40_000)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        item = settlement_item(
            ingest,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001", "PAY-00002"}),
        )
        assert item.credited
        assert classify_settlement(item, date_window_days=WINDOW) is (
            ExceptionClass.FEE_TAX_MISMATCH
        )


class TestTheCreditCascade:
    def test_a_twin_carrying_a_known_reference_is_a_duplicate(self):
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
        )
        assert classify_credit(credit_item(ingest)) is ExceptionClass.DUPLICATE_CREDIT

    def test_a_lone_credit_on_a_known_reference_is_an_unknown_residual(self):
        """The pair exists in the sources; something else stopped the match."""
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        assert classify_credit(credit_item(ingest)) is ExceptionClass.UNKNOWN_RESIDUAL

    def test_a_reference_naming_nothing_is_an_orphan(self):
        ingest = corpus(bank_txns=[bank_credit("BNK-00001", utr="UTR2026999999999")])
        assert classify_credit(credit_item(ingest)) is ExceptionClass.ORPHAN_BANK_CREDIT

    def test_an_unreferenced_credit_with_a_known_merchant_lost_its_reference(self):
        ingest = corpus(bank_txns=[bank_credit("BNK-00001", utr=None)])
        item = credit_item(ingest, merchant_profiles=frozenset({"RAZORPAY SOFTWARE PVT"}))
        assert classify_credit(item) is ExceptionClass.MISSING_REFERENCE

    def test_a_credit_with_neither_reference_nor_merchant_is_the_floor(self):
        """Nothing in the three sources can relate it. That is not a failure to try."""
        ingest = corpus(bank_txns=[noise_credit()])
        assert classify_credit(credit_item(ingest, "BNK-09001")) is (
            ExceptionClass.UNMATCHABLE
        )

    def test_an_unknown_merchant_with_no_reference_is_an_orphan(self):
        ingest = corpus(bank_txns=[bank_credit("BNK-00001", utr=None)])
        item = credit_item(ingest, merchant_profiles=frozenset({"SOMEONE ELSE"}))
        assert classify_credit(item) is ExceptionClass.ORPHAN_BANK_CREDIT

    def test_a_matched_twin_is_recorded_so_the_prose_can_be_honest(self):
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
        )
        item = credit_item(ingest, "BNK-00002", matched_credits=frozenset({"BNK-00001"}))
        assert [txn.txn_id for txn in item.matched_twins] == ["BNK-00001"]


class TestThePaymentRule:
    def _item(self, clawed_back: bool) -> PaymentItem:
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        _, _, payments = items(
            ingest,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001"}),
        )
        item = payments[0]
        assert item.clawed_back is clawed_back
        return item

    def test_an_adjustment_naming_the_payment_is_a_chargeback(self):
        assert classify_payment(self._item(True)) is ExceptionClass.CHARGEBACK_NETTED

    def test_any_other_uncredited_payment_is_an_unknown_residual(self):
        only = batch(amounts=(60_000, 40_000))
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        _, _, payments = items(
            ingest,
            matched_settlements=frozenset({"SETL-0001"}),
            matched_payments=frozenset({"PAY-00001"}),
        )
        assert classify_payment(payments[0]) is ExceptionClass.UNKNOWN_RESIDUAL


class TestSeverity:
    THRESHOLDS = SeverityThresholds()

    @pytest.mark.parametrize(
        ("impact", "expected"),
        [
            (10_000_000, Severity.CRITICAL),
            (9_999_999, Severity.HIGH),
            (1_000_000, Severity.HIGH),
            (999_999, Severity.MEDIUM),
            (100_000, Severity.MEDIUM),
            (99_999, Severity.LOW),
            (0, Severity.LOW),
        ],
    )
    def test_money_sets_the_band(self, impact, expected):
        assert severity_for(impact, age_days=0, thresholds=self.THRESHOLDS) is expected

    def test_age_escalates_by_one_step(self):
        assert severity_for(0, age_days=14, thresholds=self.THRESHOLDS) is Severity.MEDIUM

    def test_age_never_escalates_past_critical(self):
        assert severity_for(
            10**9, age_days=365, thresholds=self.THRESHOLDS
        ) is Severity.CRITICAL

    def test_age_never_lowers_a_band(self):
        fresh = severity_for(5_000_000, age_days=0, thresholds=self.THRESHOLDS)
        old = severity_for(5_000_000, age_days=99, thresholds=self.THRESHOLDS)
        assert fresh is Severity.HIGH
        assert old is Severity.CRITICAL

    def test_the_bands_must_descend(self):
        with pytest.raises(ValueError, match="strictly descend"):
            SeverityThresholds(critical_minor=1, high_minor=2, medium_minor=3)


def test_only_three_classes_are_ever_agent_resolvable():
    """PLAN.md §8.3 names three. Anything else is proposal-only."""
    assert {
        ExceptionClass.ROUNDING_DRIFT,
        ExceptionClass.TIMING_SHIFT,
        ExceptionClass.DUPLICATE_CREDIT,
    } == AGENT_RESOLVABLE_CLASSES
    assert ExceptionClass.UNMATCHABLE not in AGENT_RESOLVABLE_CLASSES
