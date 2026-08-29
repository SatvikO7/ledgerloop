"""Claw-backs taken out of somebody else's batch (A06 POST_SETTLEMENT_REFUND).

The gap this closes was invisible until Phase 2.3 removed the accident hiding
it. An order refunded *after* its payout has left is clawed back from a later
batch, and nothing else in the queue reaches it:

* the later batch reconciles to the paise -- its credit equals its declared net
  with the adjustment included -- so it raises nothing;
* the earlier batch was paid in full, so it raises nothing;
* the refunded order sits in no unresolved link, so no payment item covers it.

It used to be reported only when some unrelated anomaly happened to leave its
batch contested, and the contested settlement's evidence chain named every order
in it. The duplicate-posting pass matched those batches, the accident stopped,
and 9 of 119 expected exceptions across five seeds went silent. This module
covers the rule that reports them on their own merits instead.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ledgerloop.config import GeneratorConfig, RunConfig, SplitName
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.exceptions import classify_exceptions
from ledgerloop.exceptions.taxonomy import clawback_items
from ledgerloop.generator import generate_to_disk
from ledgerloop.matching import run_matching
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.enums import AnomalyClass, ExceptionClass, ExpectedStatus, OrderStatus
from tests.unit.conftest import batch, corpus


def _refunded(sources, order_id: str):
    """Mark one ledger order REFUNDED, the way the generator's A06 does."""
    orders = tuple(
        order.model_copy(update={"status": OrderStatus.REFUNDED})
        if order.order_id == order_id
        else order
        for order in sources.orders
    )
    return replace(sources, orders=orders)


@pytest.fixture
def clawed_back():
    """Two batches. The second absorbs a refund belonging to the first.

    ``SETL-0001`` paid two orders of 60,000 and 40,000 and settled first.
    ``SETL-0002`` settles later and carries an adjustment of -40,000, which
    matches **none** of its own payments -- so the money it accounts for was
    paid out somewhere else, and the ledger says which order it belongs to.
    """
    first = batch("SETL-0001", utr="UTR2026031000001", amounts=(60_000, 40_000))
    second = batch(
        "SETL-0002",
        utr="UTR2026031000002",
        amounts=(90_000, 70_000),
        adjustments_minor=-40_000,
        first_index=10,
    )
    sources = corpus(
        batches=[first, second],
        bank_txns=[first.credit("BNK-00001"), second.credit("BNK-00002")],
    )
    return _refunded(sources, first.orders[1].order_id), first, second


class TestTheAttribution:
    def test_a_negative_adjustment_matching_no_nested_payment_is_traced(
        self, clawed_back
    ):
        sources, first, _second = clawed_back
        items = clawback_items(
            MatchContext.from_ingest(sources), matched_settlements=frozenset()
        )
        assert len(items) == 1
        item = items[0]
        assert item.view.settlement_id == "SETL-0002"
        assert item.amount_minor == 40_000
        assert item.attributed is not None
        assert item.attributed.order_id == first.orders[1].order_id

    def test_the_source_batch_is_named_only_when_the_ladder_credited_it(
        self, clawed_back
    ):
        """The chain must never assert a payout the run did not make."""
        sources, _first, _second = clawed_back
        context = MatchContext.from_ingest(sources)
        without = clawback_items(context, matched_settlements=frozenset())
        with_it = clawback_items(context, matched_settlements=frozenset({"SETL-0001"}))
        assert without[0].source_settlements == ()
        assert [view.settlement_id for view in with_it[0].source_settlements] == [
            "SETL-0001"
        ]

    def test_an_adjustment_matching_a_payment_of_its_own_batch_is_not_this(self):
        """That is A08 CHARGEBACK_NETTED, and `classify_settlement` owns it."""
        only = batch("SETL-0001", amounts=(60_000, 40_000), adjustments_minor=-40_000)
        sources = _refunded(corpus(batches=[only]), only.orders[1].order_id)
        assert clawback_items(
            MatchContext.from_ingest(sources), matched_settlements=frozenset()
        ) == ()

    def test_no_refunded_order_of_that_amount_means_no_conclusion(self, clawed_back):
        """The adjustment alone is not enough. The ledger has to agree."""
        sources, _first, _second = clawed_back
        pristine = replace(
            sources,
            orders=tuple(
                order.model_copy(update={"status": OrderStatus.CAPTURED})
                for order in sources.orders
            ),
        )
        assert clawback_items(
            MatchContext.from_ingest(pristine), matched_settlements=frozenset()
        ) == ()

    def test_a_positive_adjustment_is_not_a_claw_back(self, clawed_back):
        sources, _first, _second = clawed_back
        flipped = replace(
            sources,
            settlements=tuple(
                settlement.model_copy(update={"adjustments_minor": 40_000})
                if settlement.settlement_id == "SETL-0002"
                else settlement
                for settlement in sources.settlements
            ),
        )
        assert clawback_items(
            MatchContext.from_ingest(flipped), matched_settlements=frozenset()
        ) == ()


class TestTheQueueRow:
    def test_it_is_classified_as_a_post_settlement_refund_against_the_order(
        self, clawed_back
    ):
        sources, first, _second = clawed_back
        context = MatchContext.from_ingest(sources)
        run = run_matching(sources, RunConfig(run_id="clawback"))
        queue = classify_exceptions(
            context, run.decisions, run.candidates, RunConfig(run_id="clawback")
        )
        rows = [
            item
            for item in queue.exceptions
            if item.exception_class is ExceptionClass.POST_SETTLEMENT_REFUND
        ]
        assert len(rows) == 1
        assert rows[0].impact_minor == 40_000
        assert first.orders[1].order_id in {ref.record_id for ref in rows[0].involved_refs}
        assert queue.clawbacks_seen == 1

    def test_the_impact_is_the_refund_and_not_the_payout(self, clawed_back):
        """The rest of the batch arrived; sorting the queue by a figure that
        included it would put the wrong item at the top."""
        sources, _first, _second = clawed_back
        run = run_matching(sources, RunConfig(run_id="clawback"))
        queue = classify_exceptions(
            MatchContext.from_ingest(sources),
            run.decisions,
            run.candidates,
            RunConfig(run_id="clawback"),
        )
        row = next(
            item
            for item in queue.exceptions
            if item.exception_class is ExceptionClass.POST_SETTLEMENT_REFUND
        )
        assert row.impact_minor == 40_000
        assert row.impact_minor != 160_000  # SETL-0002's declared net

    def test_the_evidence_names_both_documents_that_have_to_agree(self, clawed_back):
        sources, first, _second = clawed_back
        run = run_matching(sources, RunConfig(run_id="clawback"))
        queue = classify_exceptions(
            MatchContext.from_ingest(sources),
            run.decisions,
            run.candidates,
            RunConfig(run_id="clawback"),
        )
        row = next(
            item
            for item in queue.exceptions
            if item.exception_class is ExceptionClass.POST_SETTLEMENT_REFUND
        )
        blob = " ".join(item.detail for item in row.evidence)
        assert "SETL-0002" in blob
        assert "REFUNDED" in blob
        assert first.orders[1].order_id in blob


class TestOnAGeneratedCorpus:
    @pytest.fixture(scope="class")
    @staticmethod
    def scored(tmp_path_factory):
        directory = tmp_path_factory.mktemp("a06") / "test-standard-42"
        generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
        return run_system(directory, measure_calibration_quality=False), load_ground_truth(
            directory
        )

    def test_every_a06_order_the_truth_expects_is_now_in_the_queue(self, scored):
        run, truth = scored
        expected = {
            key
            for key, verdict in truth.verdict_by_ref.items()
            if verdict.anomaly_class is AnomalyClass.POST_SETTLEMENT_REFUND
            and verdict.expected_status is ExpectedStatus.EXCEPTION
        }
        assert expected, "seed 42 is expected to contain A06 records"
        covered = {ref.key for item in run.exceptions for ref in item.involved_refs}
        assert expected <= covered

    def test_exception_recall_is_one_on_this_corpus(self, scored):
        run, _truth = scored
        assert run.coverage.recall == 1.0
        assert run.coverage.missed == frozenset()
