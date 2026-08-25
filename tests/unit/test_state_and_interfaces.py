"""Run state and the deferred-infrastructure interfaces."""

from __future__ import annotations

from datetime import UTC, datetime

from ledgerloop.config import RunConfig
from ledgerloop.graph.interface import GraphRepo
from ledgerloop.models import (
    DecisionOutcome,
    LinkType,
    MatchDecision,
    Tier,
    bank_ref,
    payment_ref,
)
from ledgerloop.state import ReconState
from ledgerloop.vector.interface import VectorRepo

NOW = datetime(2026, 3, 4, tzinfo=UTC)


def _state() -> ReconState:
    return ReconState(run_id="RUN-1", config=RunConfig(run_id="RUN-1"))


def _decision(decision_id: str, supersedes: str | None = None) -> MatchDecision:
    return MatchDecision(
        decision_id=decision_id,
        candidate_id="CAND-1",
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref("PAY-1"),
        target_ref=bank_ref("BNK-1"),
        tier=Tier.T0_EXACT,
        outcome=DecisionOutcome.AUTO_MATCHED,
        calibrated_p=0.99,
        arithmetic_verified=True,
        decided_at=NOW,
        reason="test",
        supersedes=supersedes,
    )


class TestAuditSequencing:
    def test_sequence_is_monotonic(self):
        """Replay orders by sequence, not timestamp -- events share milliseconds."""
        state = _state()
        assert [state.next_audit_sequence() for _ in range(4)] == [0, 1, 2, 3]

    def test_sequences_are_independent_per_state(self):
        assert _state().next_audit_sequence() == _state().next_audit_sequence() == 0


class TestSupersedingDecisions:
    def test_log_is_append_only_and_open_view_filters(self):
        """The tier ladder loops; a late resolution can overturn an earlier call."""
        state = _state()
        original = _decision("DEC-1")
        revision = _decision("DEC-2", supersedes="DEC-1")
        state.decisions.extend([original, revision])

        assert len(state.decisions) == 2, "history must not be edited"
        assert [d.decision_id for d in state.open_decisions] == ["DEC-2"]

    def test_no_supersession_means_everything_is_open(self):
        state = _state()
        state.decisions.extend([_decision("DEC-1"), _decision("DEC-2")])
        assert len(state.open_decisions) == 2


class TestGroundTruthIsolation:
    def test_ground_truth_defaults_to_absent(self):
        """The matcher must never see truth; it is attached for the evaluator only."""
        assert _state().ground_truth is None


class TestDeferredInterfaces:
    def test_graph_repo_is_a_runtime_checkable_protocol(self):
        """Neo4j is cut; NetworkX will satisfy this same Protocol."""

        class Stub:
            def add_node(self, ref, /, **attributes): ...
            def add_edge(self, source, target, link_type, /, **attributes): ...
            def neighbours(self, ref, link_type=None): return []
            def has_edge(self, source, target, link_type): return False
            def edges_of_type(self, link_type): return []
            def path_exists(self, source, target): return False
            def siblings_in_settlement(self, settlement): return []
            def consumed_credits(self): return []
            def clear(self): ...

        assert isinstance(Stub(), GraphRepo)

    def test_incomplete_graph_implementation_fails_the_check(self):
        class Partial:
            def add_node(self, ref, /, **attributes): ...

        assert not isinstance(Partial(), GraphRepo)

    def test_vector_repo_is_a_runtime_checkable_protocol(self):
        """ChromaDB is cut; the contract stands for a later ablation."""

        class Stub:
            def index(self, key, text, /, **metadata): ...
            def query(self, text, *, top_k=5): return []
            def clear(self): ...

        assert isinstance(Stub(), VectorRepo)

    def test_no_implementation_ships_in_the_mvp(self):
        """Step 0 fixes the contracts; the repositories land with T4."""
        import ledgerloop.graph as graph_pkg
        import ledgerloop.vector as vector_pkg

        assert not hasattr(graph_pkg, "Neo4jGraphRepo")
        assert not hasattr(vector_pkg, "ChromaVectorRepo")
