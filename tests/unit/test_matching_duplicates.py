"""The duplicate-posting pass: one payout the statement shows more than once.

Three things need to be true of this pass, and each has a section below.

1. **It only ever fires on a re-posting.** Every guard is tested by making it
   the single thing that differs, so a passing test names the reason.
2. **The row it holds out is still reported.** A pass that raised the match
   rate by making money disappear from the exception queue would be the exact
   failure this project exists to argue against.
3. **It is right on real corpora.** :class:`TestOnGeneratedCorpora` is the one
   that matters: it regenerates several splits, seeds and difficulties and
   checks the *earliest* posting of every group it forms against link-level
   ground truth. That check is what makes "the first posting is the payout" a
   measurement rather than an assumption.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ledgerloop.config import Difficulty, GeneratorConfig, RunConfig, SplitName
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.matching import run_matching
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.duplicates import detect_duplicate_postings
from ledgerloop.models.enums import LinkType
from tests.unit.conftest import bank_credit, batch, corpus, debit_row


def _detect(*txns, window_days: int = 7):
    return detect_duplicate_postings(txns, window_days=window_days)


class TestWhatItCalls_A_Reposting:
    def test_two_identical_credits_a_day_apart_are_one_payout(self):
        first = bank_credit("BNK-00001")
        second = bank_credit("BNK-00002", value_date=first.value_date + timedelta(days=1))
        found = _detect(first, second)

        assert len(found.groups) == 1
        group = found.groups[0]
        assert group.original.txn_id == "BNK-00001"
        assert [txn.txn_id for txn in group.repostings] == ["BNK-00002"]
        assert found.reposted_ids == {"BNK-00002"}

    def test_the_earliest_row_is_the_payout_whatever_order_it_is_read_in(self):
        """Source order must not decide it. The value date does."""
        early = bank_credit("BNK-00009")
        late = bank_credit("BNK-00002", value_date=early.value_date + timedelta(days=2))
        assert _detect(late, early).groups[0].original.txn_id == "BNK-00009"

    def test_a_payout_posted_three_times_yields_one_group_of_two_repostings(self):
        first = bank_credit("BNK-00001")
        second = bank_credit("BNK-00002", value_date=first.value_date + timedelta(days=1))
        third = bank_credit("BNK-00003", value_date=first.value_date + timedelta(days=2))
        group = _detect(first, second, third).groups[0]

        assert group.original.txn_id == "BNK-00001"
        assert [txn.txn_id for txn in group.repostings] == ["BNK-00002", "BNK-00003"]
        assert group.duplicated_minor == 2 * group.amount_minor
        assert group.span_days == 2

    def test_it_works_without_a_reference_because_a_stripped_narration_is_the_hard_case(
        self,
    ):
        """A07 composes with A05: the duplicate can lose its UTR too. What is
        left -- the same amount to the paise and the same merchant text -- is
        still the same instruction, and that is what the pass compares."""
        first = bank_credit("BNK-00001", utr=None)
        second = bank_credit(
            "BNK-00002", utr=None, value_date=first.value_date + timedelta(days=1)
        )
        assert _detect(first, second).reposted_ids == {"BNK-00002"}


class TestEveryGuardDeclines:
    """Each test changes exactly one thing and expects the pass to say nothing."""

    def test_a_single_credit_is_not_a_group(self):
        assert _detect(bank_credit("BNK-00001")).groups == ()

    def test_amounts_that_differ_by_one_paisa_are_two_events(self):
        """Not a tolerance. A copy that differs is not a copy -- three paise
        short is T1's band, and a split tranche is T2's arithmetic."""
        first = bank_credit("BNK-00001", amount_minor=100_000)
        second = bank_credit(
            "BNK-00002",
            amount_minor=100_001,
            value_date=first.value_date + timedelta(days=1),
        )
        assert _detect(first, second).groups == ()

    def test_a_different_narration_is_a_different_instruction(self):
        first = bank_credit("BNK-00001", merchant="RAZORPAY SOFTWARE PVT")
        second = bank_credit(
            "BNK-00002",
            merchant="NYKAA E RETAIL PVT LTD",
            value_date=first.value_date + timedelta(days=1),
        )
        assert _detect(first, second).groups == ()

    def test_two_postings_on_one_day_have_no_earliest_so_nothing_is_concluded(self):
        """The whole conclusion is an ordering. Without one the pair falls
        through to the tier ladder's existing contested path."""
        first = bank_credit("BNK-00001")
        second = bank_credit("BNK-00002", value_date=first.value_date)
        assert _detect(first, second).groups == ()

    def test_rows_further_apart_than_the_window_are_a_recurring_payout(self):
        first = bank_credit("BNK-00001")
        second = bank_credit("BNK-00002", value_date=first.value_date + timedelta(days=30))
        assert _detect(first, second).groups == ()
        assert _detect(first, second, window_days=30).reposted_ids == {"BNK-00002"}

    def test_the_window_only_ever_declines_more(self):
        """Narrowing it can shrink the set and never grow it, which is why the
        knob cannot be turned to buy recall."""
        first = bank_credit("BNK-00001")
        second = bank_credit("BNK-00002", value_date=first.value_date + timedelta(days=3))
        wide = _detect(first, second, window_days=7).reposted_ids
        narrow = _detect(first, second, window_days=1).reposted_ids
        assert narrow <= wide
        assert narrow == frozenset()

    def test_debits_are_never_considered(self):
        """Outgoing money is not a payout, however identical two rows are."""
        assert _detect(debit_row("BNK-00001"), debit_row("BNK-00002")).groups == ()

    def test_a_negative_window_is_a_configuration_error_not_a_zero(self):
        with pytest.raises(ValueError, match="window_days"):
            _detect(bank_credit(), window_days=-1)


class TestWhatTheLadderSees:
    def test_the_settlement_resolves_instead_of_being_contested(self):
        only = batch()
        run = run_matching(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001"),
                    only.credit("BNK-00002", days_after=2),
                ],
            ),
            RunConfig(run_id="dedup-on"),
        )
        assert run.settlements_resolved == 1
        assert run.settlements_contested == 0
        assert len(run.predictions) == len(only.payments)

    def test_switching_the_pass_off_restores_the_contest_exactly(self):
        only = batch()
        sources = corpus(
            batches=[only],
            bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002", days_after=2)],
        )
        run = run_matching(
            sources,
            RunConfig(run_id="dedup-off", duplicates={"enabled": False}),
        )
        assert run.settlements_resolved == 0
        assert run.settlements_contested == 1
        assert run.predictions == ()

    def test_the_reposting_is_never_claimed_by_the_settlement_it_copies(self):
        """It leaves the matchable pool; it is not matched to anything."""
        only = batch()
        run = run_matching(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001"),
                    only.credit("BNK-00002", days_after=2),
                ],
            ),
            RunConfig(run_id="dedup-target"),
        )
        targets = {
            decision.target_ref.record_id
            for decision in run.decisions
            if decision.link_type is not LinkType.ORDER_PAID_BY
        }
        assert targets == {"BNK-00001"}

    def test_the_reposting_stays_an_unclaimed_credit_the_queue_can_reach(self):
        only = batch()
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[
                    only.credit("BNK-00001"),
                    only.credit("BNK-00002", days_after=2),
                ],
            )
        )
        assert context.duplicates.reposted_ids == {"BNK-00002"}
        assert "BNK-00002" not in context.consumed_credits
        assert "BNK-00002" in {txn.txn_id for txn in context.bank_txns}
        assert "BNK-00002" not in {txn.txn_id for txn in context.open_credits()}


class TestOnGeneratedCorpora:
    """The measurement, not the illustration."""

    @pytest.mark.parametrize(
        ("split", "difficulty", "seed"),
        [
            (SplitName.DEV, Difficulty.STANDARD, 42),
            (SplitName.TEST, Difficulty.STANDARD, 42),
            (SplitName.TEST, Difficulty.STANDARD, 45),
            (SplitName.TEST, Difficulty.EASY, 42),
            (SplitName.TEST, Difficulty.HARD, 44),
            (SplitName.TRAIN, Difficulty.STANDARD, 46),
        ],
    )
    def test_the_earliest_posting_is_the_one_ground_truth_links(
        self, tmp_path, split, difficulty, seed
    ):
        """The rule's whole claim, checked against link-level truth.

        For every group the pass forms: the earliest row carries the payments'
        ``PAYMENT_CREDITED_AS`` links, and **no** later row carries any. Ground
        truth is read here and nowhere the system can see it -- this is a test,
        not a tier.
        """
        directory = tmp_path / f"{split.value}-{difficulty.value}-{seed}"
        generate_to_disk(
            GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
        )
        truth = load_ground_truth(directory)
        run = run_system(directory, measure_calibration_quality=False)
        linked = {target for _source, target in truth.evaluation_pairs}

        groups = run.matched.context.duplicates.groups
        assert groups, "this corpus is expected to contain at least one duplicate"
        for group in groups:
            for reposting in group.repostings:
                assert f"bank_txn:{reposting.txn_id}" not in linked
            assert f"bank_txn:{group.original.txn_id}" in linked

    def test_it_costs_no_precision_on_the_headline_corpus(self, tmp_path):
        directory = tmp_path / "test-standard-42"
        generate_to_disk(
            GeneratorConfig(split=SplitName.TEST, seed=42), directory
        )
        run = run_system(directory, measure_calibration_quality=False)
        links = run.metrics.link_metrics
        assert links is not None
        assert links.false_positives == 0
        assert links.false_positive_cost_minor == 0
