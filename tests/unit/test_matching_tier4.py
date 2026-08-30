"""T4 -- the graph repository and the four constraint rules.

The rules fire zero times on the generated corpus, and that is the honest
outcome rather than a gap in the tests: every earlier tier matches at
*settlement* granularity, so the partial assignments path closure and sibling
completion exist to finish never arise. Both facts are asserted -- that the
rules work when the situation is constructed, and that the situation does not
occur in the corpus.

Exclusivity is the rule that does real work either way. It produces no matches,
only refusals, and a refusal is what stops the other two overfilling a credit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ledgerloop.matching.pipeline as pipeline
from ledgerloop.config import DecisionThresholds, GraphInference, MatchingTolerances
from ledgerloop.graph.interface import GraphRepo
from ledgerloop.graph.memory_repo import MemoryGraphRepo
from ledgerloop.matching.bank_leg import allocated_share_minor
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import run_tier0
from ledgerloop.matching.tier4_graph import build_graph, detect_rings, run_tier4
from ledgerloop.models.candidates import FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, OrderStatus, RecordType, Tier
from ledgerloop.models.refs import bank_ref, order_ref, payment_ref, settlement_ref
from tests.unit.conftest import batch, corpus, make_order

GRAPH = GraphInference()
THRESHOLDS = DecisionThresholds()
TOLERANCES = MatchingTolerances()


class TestTheRepository:
    def test_it_satisfies_the_protocol(self):
        assert isinstance(MemoryGraphRepo(), GraphRepo)

    def test_nodes_and_edges_round_trip(self):
        repo = MemoryGraphRepo()
        repo.add_edge(
            payment_ref("PAY-1"), settlement_ref("SETL-1"), LinkType.PAYMENT_SETTLED_IN
        )
        assert repo.has_edge(
            payment_ref("PAY-1"), settlement_ref("SETL-1"), LinkType.PAYMENT_SETTLED_IN
        )
        assert repo.neighbours(payment_ref("PAY-1")) == (settlement_ref("SETL-1"),)
        assert len(repo) == 2

    def test_both_endpoints_are_created_on_demand(self):
        """A rule that had to remember to add nodes first would forget."""
        repo = MemoryGraphRepo()
        repo.add_edge(order_ref("ORD-1"), payment_ref("PAY-1"), LinkType.ORDER_PAID_BY)
        assert set(repo.nodes()) == {order_ref("ORD-1"), payment_ref("PAY-1")}

    def test_repeating_an_edge_is_a_no_op(self):
        repo = MemoryGraphRepo()
        for _ in range(3):
            repo.add_edge(
                payment_ref("PAY-1"), settlement_ref("SETL-1"), LinkType.PAYMENT_SETTLED_IN
            )
        assert len(repo.edges_of_type(LinkType.PAYMENT_SETTLED_IN)) == 1

    def test_neighbours_can_be_filtered_by_link_type(self):
        repo = MemoryGraphRepo()
        repo.add_edge(payment_ref("PAY-1"), settlement_ref("S"), LinkType.PAYMENT_SETTLED_IN)
        repo.add_edge(payment_ref("PAY-1"), bank_ref("BNK-1"), LinkType.PAYMENT_CREDITED_AS)
        assert repo.neighbours(payment_ref("PAY-1"), LinkType.PAYMENT_CREDITED_AS) == (
            bank_ref("BNK-1"),
        )

    def test_traversal_order_follows_insertion(self):
        """Non-deterministic order would make seeded runs irreproducible."""
        repo = MemoryGraphRepo()
        for index in range(5):
            repo.add_edge(
                payment_ref("PAY-1"),
                bank_ref(f"BNK-{index}"),
                LinkType.PAYMENT_CREDITED_AS,
            )
        assert [ref.record_id for ref in repo.neighbours(payment_ref("PAY-1"))] == [
            f"BNK-{i}" for i in range(5)
        ]

    def test_path_exists_walks_the_whole_chain(self):
        repo = MemoryGraphRepo()
        repo.add_edge(order_ref("O"), payment_ref("P"), LinkType.ORDER_PAID_BY)
        repo.add_edge(payment_ref("P"), settlement_ref("S"), LinkType.PAYMENT_SETTLED_IN)
        repo.add_edge(settlement_ref("S"), bank_ref("B"), LinkType.SETTLEMENT_CREDITED_AS)
        assert repo.path_exists(order_ref("O"), bank_ref("B"))
        assert repo.path_exists(order_ref("O"), order_ref("O"))
        assert not repo.path_exists(bank_ref("B"), order_ref("O"))

    def test_path_exists_is_false_across_disconnected_components(self):
        repo = MemoryGraphRepo()
        repo.add_edge(order_ref("O"), payment_ref("P"), LinkType.ORDER_PAID_BY)
        repo.add_node(bank_ref("B"))
        assert not repo.path_exists(order_ref("O"), bank_ref("B"))

    def test_a_cycle_does_not_hang_the_traversal(self):
        repo = MemoryGraphRepo()
        repo.add_edge(payment_ref("A"), payment_ref("B"), LinkType.ORDER_PAID_BY)
        repo.add_edge(payment_ref("B"), payment_ref("A"), LinkType.ORDER_PAID_BY)
        assert not repo.path_exists(payment_ref("A"), bank_ref("Z"))

    def test_siblings_are_read_off_the_settled_in_edges(self):
        repo = MemoryGraphRepo()
        for name in ("PAY-1", "PAY-2"):
            repo.add_edge(
                payment_ref(name), settlement_ref("SETL-1"), LinkType.PAYMENT_SETTLED_IN
            )
        repo.add_edge(payment_ref("PAY-9"), settlement_ref("SETL-2"), LinkType.PAYMENT_SETTLED_IN)
        assert repo.siblings_in_settlement(settlement_ref("SETL-1")) == (
            payment_ref("PAY-1"),
            payment_ref("PAY-2"),
        )

    def test_consumed_credits_are_flagged_and_listed(self):
        repo = MemoryGraphRepo()
        repo.add_node(bank_ref("BNK-1"))
        repo.add_node(bank_ref("BNK-2"))
        repo.mark_consumed(bank_ref("BNK-1"))
        assert tuple(repo.consumed_credits()) == (bank_ref("BNK-1"),)

    def test_consumption_can_be_lifted(self):
        repo = MemoryGraphRepo()
        repo.mark_consumed(bank_ref("BNK-1"))
        repo.mark_consumed(bank_ref("BNK-1"), consumed=False)
        assert tuple(repo.consumed_credits()) == ()

    def test_nodes_can_be_filtered_by_record_type(self):
        repo = MemoryGraphRepo()
        repo.add_node(order_ref("O"))
        repo.add_node(bank_ref("B"))
        assert repo.nodes(RecordType.BANK_TXN) == (bank_ref("B"),)

    def test_attributes_are_kept_and_merged(self):
        repo = MemoryGraphRepo()
        repo.add_node(order_ref("O"), merchant_id="MRCH_0001")
        repo.add_node(order_ref("O"), flagged=True)
        assert repo.attributes(order_ref("O")) == {"merchant_id": "MRCH_0001", "flagged": True}

    def test_clear_drops_everything(self):
        repo = MemoryGraphRepo()
        repo.add_edge(order_ref("O"), payment_ref("P"), LinkType.ORDER_PAID_BY)
        repo.clear()
        assert len(repo) == 0
        assert repo.edges_of_type(LinkType.ORDER_PAID_BY) == ()


class TestBuildingTheGraph:
    def test_the_asserted_edges_come_from_the_sources(self, simple):
        context = MatchContext.from_ingest(simple)
        repo = build_graph(context, ())
        assert len(repo.edges_of_type(LinkType.PAYMENT_SETTLED_IN)) == 2
        assert len(repo.edges_of_type(LinkType.ORDER_PAID_BY)) == 2
        assert repo.edges_of_type(LinkType.SETTLEMENT_CREDITED_AS) == ()

    def test_established_candidates_become_inferred_edges(self, simple):
        context = MatchContext.from_ingest(simple)
        _, bank = run_tier0(context)
        repo = build_graph(context, bank.candidates)
        assert len(repo.edges_of_type(LinkType.SETTLEMENT_CREDITED_AS)) == 1
        assert len(repo.edges_of_type(LinkType.PAYMENT_CREDITED_AS)) == 2

    def test_a_payment_quoting_an_unknown_order_gets_no_order_edge(self):
        only = batch(order_refs=("ORD-2026-999999", "ORD-2026-000002"))
        repo = build_graph(MatchContext.from_ingest(corpus(batches=[only])), ())
        assert len(repo.edges_of_type(LinkType.ORDER_PAID_BY)) == 1

    def test_only_credits_become_bank_nodes(self):
        only = batch()
        built = corpus(batches=[only], bank_txns=[only.credit()])
        repo = build_graph(MatchContext.from_ingest(built), ())
        assert repo.nodes(RecordType.BANK_TXN) == (bank_ref("BNK-00001"),)


class TestExclusivityPruning:
    def test_a_fully_absorbed_credit_is_marked_consumed(self, simple):
        context = MatchContext.from_ingest(simple)
        _, bank = run_tier0(context)
        outcome = run_tier4(context, bank.candidates, GRAPH, THRESHOLDS)
        assert outcome.credits_fully_absorbed == 1

    def test_an_untouched_credit_is_not_marked(self):
        only = batch()
        built = corpus(batches=[only], bank_txns=[only.credit()])
        outcome = run_tier4(MatchContext.from_ingest(built), (), GRAPH, THRESHOLDS)
        assert outcome.credits_fully_absorbed == 0

    def test_a_premise_headed_for_review_is_not_a_premise(self, simple):
        """An inference built on doubt inherits it without inheriting the caveat."""
        context = MatchContext.from_ingest(simple)
        _, bank = run_tier0(context)
        doubtful = tuple(
            c.model_copy(update={"calibrated_p": 0.5}) for c in bank.candidates
        )
        outcome = run_tier4(context, doubtful, GRAPH, THRESHOLDS)
        assert outcome.credits_fully_absorbed == 0
        assert outcome.candidates == ()

    def test_an_unverified_premise_is_not_a_premise(self, simple):
        context = MatchContext.from_ingest(simple)
        _, bank = run_tier0(context)
        unverified = tuple(
            c.model_copy(update={"arithmetic_verified": False}) for c in bank.candidates
        )
        assert run_tier4(context, unverified, GRAPH, THRESHOLDS).credits_fully_absorbed == 0


class TestPathClosure:
    """``P -> S`` known and ``S -> C`` known implies ``P -> C``. A deduction."""

    def _partial(self):
        """A settlement credited as a known credit, with its payments unexpanded."""
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        settlement_only = tuple(
            c for c in bank.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        return context, settlement_only

    def test_it_completes_the_payments_of_a_credited_settlement(self):
        context, premises = self._partial()
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert outcome.path_closures == 1
        assert outcome.payment_links == 2

    def test_the_deduction_is_certain(self):
        context, premises = self._partial()
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert all(c.calibrated_p == 1.0 for c in outcome.candidates)
        assert all(c.tier is Tier.T4_GRAPH for c in outcome.candidates)

    def test_the_inferred_shares_conserve_the_credit(self):
        context, premises = self._partial()
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert sum(allocated_share_minor(c) for c in outcome.candidates) == 100_000

    def test_the_evidence_names_the_rule_and_the_chain(self):
        context, premises = self._partial()
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        note = next(
            i for c in outcome.candidates for i in c.evidence
            if i.kind is EvidenceKind.GRAPH_RULE
        )
        assert note.detail.startswith("path closure")
        assert "SETL-0001" in note.detail
        assert note.score == 1.0

    def test_a_charged_back_payment_is_never_completed(self):
        """A08's money never reached the bank. Inferring a link would invent it."""
        only = batch(amounts=(60_000, 40_000), adjustments_minor=-40_000)
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        premises = tuple(
            c for c in bank.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert {c.source_ref.record_id for c in outcome.candidates} == {"PAY-00001"}

    def test_exclusivity_blocks_a_closure_onto_a_full_credit(self):
        context = MatchContext.from_ingest(
            corpus(batches=[batch()], bank_txns=[batch().credit()])
        )
        _, bank = run_tier0(context)
        outcome = run_tier4(context, bank.candidates, GRAPH, THRESHOLDS)
        assert outcome.candidates == ()
        assert outcome.path_closures == 0


class TestSiblingCompletion:
    """Most of a batch points at one credit, so the rest are constrained. Induction."""

    def _majority(self, linked: int, total: int):
        only = batch(amounts=tuple(10_000 for _ in range(total)))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        payments = [c for c in bank.candidates if c.is_evaluable][:linked]
        # No settlement edge: only the payment links, so path closure cannot fire
        # and sibling completion is the rule under test.
        return context, tuple(payments)

    def test_a_majority_constrains_the_remainder(self):
        context, premises = self._majority(linked=4, total=5)
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert outcome.sibling_completions == 1
        assert outcome.payment_links == 1

    def test_the_confidence_is_the_support_not_certainty(self):
        """An induction that claimed certainty would be a false-positive generator."""
        context, premises = self._majority(linked=4, total=5)
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert all(c.calibrated_p == pytest.approx(0.8) for c in outcome.candidates)
        assert all(c.features.graph_support == pytest.approx(0.8) for c in outcome.candidates)

    def test_below_the_threshold_nothing_is_inferred(self):
        context, premises = self._majority(linked=3, total=5)
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert outcome.sibling_completions == 0
        assert outcome.candidates == ()

    def test_the_threshold_comes_from_the_configuration(self):
        context, premises = self._majority(linked=3, total=5)
        relaxed = GraphInference(sibling_completion_threshold=0.6)
        assert run_tier4(context, premises, relaxed, THRESHOLDS).sibling_completions == 1

    def test_the_evidence_names_the_rule_and_the_majority(self):
        context, premises = self._majority(linked=4, total=5)
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        note = next(
            i for c in outcome.candidates for i in c.evidence
            if i.kind is EvidenceKind.GRAPH_RULE
        )
        assert note.detail.startswith("sibling completion")
        assert "4 of 5" in note.detail



class TestRingDetection:
    def test_a_customer_refunding_across_merchants_is_flagged(self):
        orders = [
            make_order("ORD-2026-000001", merchant_id="MRCH_0001",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=1),
            make_order("ORD-2026-000002", merchant_id="MRCH_0002",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=2),
            make_order("ORD-2026-000003", merchant_id="MRCH_0003",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=3),
        ]
        context = MatchContext.from_ingest(corpus(extra_orders=orders))
        (finding,) = detect_rings(context, GRAPH)
        assert finding.customer_ref == "CUST_99999"
        assert finding.events == 3
        assert finding.merchants == ("MRCH_0001", "MRCH_0002", "MRCH_0003")

    def test_one_merchant_is_a_difficult_customer_not_a_ring(self):
        orders = [
            make_order(f"ORD-2026-00000{i}", merchant_id="MRCH_0001",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=i)
            for i in range(1, 5)
        ]
        context = MatchContext.from_ingest(corpus(extra_orders=orders))
        assert detect_rings(context, GRAPH) == ()

    def test_too_few_events_is_not_a_ring(self):
        orders = [
            make_order("ORD-2026-000001", merchant_id="MRCH_0001",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=1),
            make_order("ORD-2026-000002", merchant_id="MRCH_0002",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=2),
        ]
        context = MatchContext.from_ingest(corpus(extra_orders=orders))
        assert detect_rings(context, GRAPH) == ()

    def test_captured_orders_are_not_refund_events(self, simple):
        assert detect_rings(MatchContext.from_ingest(simple), GRAPH) == ()

    def test_the_thresholds_come_from_the_configuration(self):
        orders = [
            make_order("ORD-2026-000001", merchant_id="MRCH_0001",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=1),
            make_order("ORD-2026-000002", merchant_id="MRCH_0002",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=2),
        ]
        context = MatchContext.from_ingest(corpus(extra_orders=orders))
        assert detect_rings(context, GraphInference(ring_min_events=2)) != ()

    def test_a_ring_never_produces_a_match(self, simple):
        """PLAN.md 6.4: a bonus signal in the exception report, not a decision."""
        orders = [
            make_order("ORD-2026-00900" + str(i), merchant_id=f"MRCH_000{i}",
                       customer_ref="CUST_99999", status=OrderStatus.REFUNDED, line=900 + i)
            for i in range(1, 4)
        ]
        built = corpus(extra_orders=orders)
        outcome = run_tier4(MatchContext.from_ingest(built), (), GRAPH, THRESHOLDS)
        assert outcome.rings
        assert outcome.candidates == ()


class TestDeterminism:
    def test_two_runs_produce_identical_candidates(self):
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context_a = MatchContext.from_ingest(built)
        _, bank_a = run_tier0(context_a)
        premises = tuple(
            c for c in bank_a.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        first = run_tier4(context_a, premises, GRAPH, THRESHOLDS)
        second = run_tier4(context_a, premises, GRAPH, THRESHOLDS)
        assert first.candidates == second.candidates


class TestReportingSurfaceAndExclusivityBlocks:
    def test_the_outcome_names_its_tier_and_counts_payment_links(self):
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        premises = tuple(
            c for c in bank.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert outcome.tier is Tier.T4_GRAPH
        assert outcome.payment_links == 2

    def test_a_settlement_with_nothing_left_to_complete_is_skipped(self, simple):
        context = MatchContext.from_ingest(simple)
        _, bank = run_tier0(context)
        outcome = run_tier4(context, bank.candidates, GRAPH, THRESHOLDS)
        assert outcome.candidates == ()
        assert outcome.path_closures == 0

    def test_a_settlement_whose_only_payment_was_clawed_back_is_skipped(self):
        only = batch(amounts=(50_000,), adjustments_minor=-50_000, net_minor=50_000)
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        outcome = run_tier4(context, bank.candidates, GRAPH, THRESHOLDS)
        assert outcome.candidates == ()

    def test_exclusivity_blocks_a_closure_onto_a_credit_another_batch_filled(self):
        """Two settlements claiming one credit. The second cannot have it."""
        first = batch("SETL-0001", utr="UTR2026031000001", amounts=(100_000,), first_index=1)
        second = batch("SETL-0002", utr="UTR2026031000002", amounts=(70_000,), first_index=10)
        built = corpus(
            batches=[first, second], bank_txns=[first.credit("BNK-00001")]
        )
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        intruder = MatchCandidate(
            candidate_id="manual|SETTLEMENT_CREDITED_AS|settlement:SETL-0002|bank_txn:BNK-00001",
            link_type=LinkType.SETTLEMENT_CREDITED_AS,
            source_ref=settlement_ref("SETL-0002"),
            target_ref=bank_ref("BNK-00001"),
            tier=Tier.T3_FUZZY,
            features=FeatureVector(tier=Tier.T3_FUZZY),
            calibrated_p=1.0,
            arithmetic_verified=True,
        )
        outcome = run_tier4(
            context, (*bank.candidates, intruder), GRAPH, THRESHOLDS
        )
        assert outcome.inferences_blocked == 1
        assert outcome.candidates == ()

    def test_exclusivity_blocks_a_sibling_completion_onto_a_full_credit(self):
        only = batch(amounts=(25_000, 25_000, 25_000, 25_000, 0))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        # The four paying siblings absorb the whole credit between them; the
        # fifth carries nothing, so the majority has no capacity left to give.
        premises = tuple(c for c in bank.candidates if c.is_evaluable)[:4]
        outcome = run_tier4(context, premises, GRAPH, THRESHOLDS)
        assert outcome.inferences_blocked == 1
        assert outcome.candidates == ()


class TestWhyItContributesZero:
    """The structural reason, asserted rather than observed.

    ``ARCHITECTURE.md`` decision 31 reports that T4 fires zero times and gives
    the reason: every earlier tier matches at *settlement* granularity, so it
    establishes the settlement-to-credit edge and expands the whole batch in one
    go. Path closure and sibling completion both need a settlement that is
    **partly** assigned, and no such state is ever produced.

    Phase 2.7 measured that claim instead of restating it. Across all 29 corpora
    on disk, and again on a 5,000-order corpus, every settlement the pipeline
    hands T4 is either fully linked or not linked at all -- 1228 fully linked
    against 242 untouched at 300 orders, 762 against 142 at 5,000, and **zero
    partial in either**. The tests below pin the invariant rather than the
    counts, so they stay true on any corpus while still failing the moment a
    tier starts leaving a batch half-assigned.

    That failure would be *good news*: it is precisely the situation T4 was
    built for, and the tier would begin contributing on its own.
    """

    @staticmethod
    def _fixture() -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"

    def _premise_state(self) -> list[tuple[int, int, int]]:
        """(payments covered, payments linked, settlement-credit premises) per settlement."""
        from ledgerloop.eval.harness import run_system
        from ledgerloop.matching.tier4_graph import _covered_payments

        captured: list[tuple[object, tuple[MatchCandidate, ...], object]] = []
        real = pipeline.run_tier4

        def spy(context, established, graph_config, thresholds):  # type: ignore[no-untyped-def]
            captured.append((context, established, thresholds))
            return real(context, established, graph_config, thresholds)

        pipeline.run_tier4 = spy  # type: ignore[assignment]
        try:
            run_system(self._fixture(), measure_calibration_quality=False)
        finally:
            pipeline.run_tier4 = real  # type: ignore[assignment]

        rows: list[tuple[int, int, int]] = []
        for context, established, thresholds in captured:
            premises = [
                candidate
                for candidate in established
                if candidate.calibrated_p is not None
                and candidate.calibrated_p >= thresholds.tau_high
                and candidate.arithmetic_verified
            ]
            linked = {
                candidate.source_ref.record_id
                for candidate in premises
                if candidate.link_type is LinkType.PAYMENT_CREDITED_AS
            }
            credited: dict[str, int] = {}
            for candidate in premises:
                if candidate.link_type is LinkType.SETTLEMENT_CREDITED_AS:
                    key = candidate.source_ref.record_id
                    credited[key] = credited.get(key, 0) + 1
            for view in context.settlements:
                covered = _covered_payments(view)
                if not covered:
                    continue
                have = sum(1 for p in covered if p.payment_id in linked)
                rows.append((len(covered), have, credited.get(view.settlement_id, 0)))
        return rows

    def test_the_pipeline_never_hands_it_a_partly_assigned_settlement(self):
        """The invariant that makes T4's zero structural rather than accidental."""
        rows = self._premise_state()
        assert rows, "the fixture produced no settlements to inspect"
        partial = [row for row in rows if 0 < row[1] < row[0]]
        assert partial == [], (
            "a settlement was handed to T4 partly assigned, which is the state "
            "path closure and sibling completion exist for -- T4 should now be "
            "contributing, and decision 31 needs rewriting"
        )

    def test_a_settlement_credit_edge_never_arrives_without_its_payments(self):
        """Path closure's premise, and why it has none.

        ``S -> C`` established with payments outstanding is exactly what the
        rule deduces from. Every tier that asserts the settlement edge asserts
        the payment edges in the same breath, so the premise never stands alone.
        """
        rows = self._premise_state()
        orphans = [row for row in rows if row[2] > 0 and row[1] < row[0]]
        assert orphans == []

    def test_it_still_runs_and_reports_on_the_corpus(self):
        """Zero contribution is not zero work. Exclusivity does real work here.

        A tier that never executed and a tier that executed and found nothing
        are different findings, and the run record has to be able to tell them
        apart -- the dashboard draws them differently for the same reason.
        """
        from ledgerloop.eval.harness import run_system

        run = run_system(self._fixture(), measure_calibration_quality=False)
        graph = run.matched.graph
        assert graph.nodes > 0
        assert graph.edges > 0
        assert graph.candidates == ()
        assert graph.path_closures == 0
        assert graph.sibling_completions == 0
        assert graph.credits_fully_absorbed > 0

    def test_the_same_code_fires_the_moment_the_state_is_partial(self):
        """The other half of the claim: unexercised, not broken.

        The corpus never produces a partial assignment, so this constructs one
        from the same tier's own output and shows the rule completing it. If
        this ever failed while the invariant above still held, T4 would be dead
        code rather than an unexercised rung.
        """
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        whole = run_tier4(context, bank.candidates, GRAPH, THRESHOLDS)
        assert whole.candidates == ()

        settlement_only = tuple(
            c for c in bank.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        partial = run_tier4(context, settlement_only, GRAPH, THRESHOLDS)
        assert partial.path_closures == 1
        assert partial.payment_links == 2

    def test_what_it_infers_agrees_with_the_tier_that_would_have_done_it(self):
        """No false positive, checked against T0 rather than against truth.

        T0 resolves this batch on the reference alone. Stripping its payment
        edges and letting T4 deduce them back must reproduce the same links and
        the same rupee shares -- so the deduction is checked against another
        tier's answer, and no ground truth is read.
        """
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context_a = MatchContext.from_ingest(built)
        _, bank = run_tier0(context_a)
        by_t0 = {
            (c.source_ref.key, c.target_ref.key): allocated_share_minor(c)
            for c in bank.candidates
            if c.is_evaluable
        }

        context_b = MatchContext.from_ingest(built)
        _, bank_b = run_tier0(context_b)
        settlement_only = tuple(
            c for c in bank_b.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        inferred = run_tier4(context_b, settlement_only, GRAPH, THRESHOLDS)
        by_t4 = {
            (c.source_ref.key, c.target_ref.key): allocated_share_minor(c)
            for c in inferred.candidates
            if c.is_evaluable
        }
        assert by_t4 == by_t0

    def test_it_carries_its_own_provenance(self):
        """An inferred link has to say it was inferred, and from what."""
        only = batch(amounts=(60_000, 40_000))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        context = MatchContext.from_ingest(built)
        _, bank = run_tier0(context)
        settlement_only = tuple(
            c for c in bank.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )
        outcome = run_tier4(context, settlement_only, GRAPH, THRESHOLDS)
        assert outcome.candidates
        for candidate in outcome.candidates:
            assert candidate.tier is Tier.T4_GRAPH
            details = " ".join(item.detail for item in candidate.evidence)
            assert "path closure" in details
            assert any(
                item.kind is EvidenceKind.ARITHMETIC_CHECK for item in candidate.evidence
            )
