"""The LangGraph assembly: topology, transitions, state, and equivalence.

The graph's only job is to move data between functions that were already
tested. So the tests that matter are not about reconciliation -- they are about
whether the wiring can *change* a result, and whether the cycle and the
conditional branch are real.

The load-bearing one is :class:`TestTheGraphAndTheChainAgree`. If the graph ever
produces a different number from ``run_system``, every metric in
``EVALUATION.md`` becomes a claim about which path produced it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.agent.graph import (
    GRAPH_EDGES,
    LangGraphUnavailable,
    build_recon_graph,
    langgraph_available,
)
from ledgerloop.agent.nodes import (
    NODE_SEQUENCE,
    build_entity_graph,
    ingest_sources,
    normalize_records,
    should_loop,
    tier_ladder,
)
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.state import RunResources, initial_state
from ledgerloop.config import GeneratorConfig, LLMConfig
from ledgerloop.eval.harness import prepare_run, run_system
from ledgerloop.generator import generate_to_disk
from ledgerloop.llm.client import LLMClient
from ledgerloop.models.audit import AuditEventType
from ledgerloop.models.enums import SplitName

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("graph") / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def pair(corpus):
    """The same corpus through both paths. Module-scoped and a plain function:
    a class-scoped fixture defined as an instance method warns in pytest 8."""
    graph = run_graph(corpus, measure_calibration_quality=False, store=None)
    direct = run_system(corpus, measure_calibration_quality=False)
    return graph.require(), direct


@pytest.fixture(scope="module")
def executed(corpus):
    """One graph run, shared. ``store=None`` so no fixture pollutes reports/."""
    return run_graph(corpus, measure_calibration_quality=False, store=None)


def _resources(corpus, **kwargs):
    return RunResources(setup=prepare_run(corpus, **kwargs))


class TestTopology:
    def test_the_documented_edges_include_the_cycle_and_its_exit(self):
        """PLAN.md §4.3 names the loop as LangGraph's justification. If the two
        edges out of `tier_ladder` are not both there, the framework is
        decoration."""
        pairs = {(source, target) for source, target, _ in GRAPH_EDGES}
        assert ("tier_ladder", "tier_ladder") in pairs
        assert ("tier_ladder", "llm_adjudicate") in pairs

    def test_the_conditional_branch_out_of_build_entity_graph_exists(self):
        conditions = {
            (source, target): condition
            for source, target, condition in GRAPH_EDGES
            if source == "build_entity_graph"
        }
        assert len(conditions) == 2
        assert all(condition is not None for condition in conditions.values())

    def test_every_documented_node_is_in_the_declared_sequence(self):
        """`NODE_SEQUENCE` is data the UI renders and the tests assert on, so it
        must not drift from the edges."""
        edge_nodes = {source for source, _, _ in GRAPH_EDGES} | {
            target for _, target, _ in GRAPH_EDGES
        }
        edge_nodes -= {"__start__", "__end__"}
        assert edge_nodes == set(NODE_SEQUENCE)

    def test_the_compiled_graph_has_the_same_nodes(self, corpus):
        graph = build_recon_graph(_resources(corpus))
        compiled = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert compiled == set(NODE_SEQUENCE)


class TestTransitions:
    def test_the_happy_path_visits_every_node(self, executed):
        assert executed.ok
        assert set(executed.node_log) == set(NODE_SEQUENCE)

    def test_it_visits_them_in_the_documented_order(self, executed):
        """Repeats of `tier_ladder` collapse; everything else runs once, in the
        order PLAN.md §4.1 draws."""
        seen: list[str] = []
        for node in executed.node_log:
            if not seen or seen[-1] != node:
                seen.append(node)
        assert tuple(seen) == NODE_SEQUENCE

    def test_the_loop_fires_more_than_once_on_this_corpus(self, executed):
        """A cycle that only ever ran once would be an edge nobody exercises."""
        assert executed.residual_iterations >= 2
        assert executed.passes == executed.residual_iterations

    def test_the_loop_stops_when_a_pass_adds_nothing(self, corpus):
        """The exit condition, driven directly rather than through the graph."""
        resources = _resources(corpus)
        state = initial_state("t")
        state.update(ingest_sources(state, resources))
        state.update(normalize_records(state, resources))
        state.update(build_entity_graph(state, resources))

        assert should_loop(state) == "tier_ladder"
        for _ in range(10):
            state.update(tier_ladder(state, resources))
            if should_loop(state) != "tier_ladder":
                break
        assert should_loop(state) == "llm_adjudicate"
        assert state["ladder"].last_pass_added == 0

    def test_the_edge_reads_the_ladder_predicate_rather_than_its_own_copy(
        self, corpus
    ):
        """`should_loop` must delegate to `should_run_residual_pass`, or the
        graph's cycle and the CLI's `while` could disagree about pass counts."""
        from ledgerloop.matching.pipeline import should_run_residual_pass

        resources = _resources(corpus)
        state = initial_state("t")
        state.update(ingest_sources(state, resources))
        state.update(normalize_records(state, resources))
        state.update(build_entity_graph(state, resources))
        for _ in range(4):
            expected = (
                "tier_ladder"
                if should_run_residual_pass(state["ladder"])
                else "llm_adjudicate"
            )
            assert should_loop(state) == expected
            if expected == "llm_adjudicate":
                break
            state.update(tier_ladder(state, resources))

    def test_a_ladder_with_no_residual_tier_skips_the_loop_entirely(self, corpus):
        """T0/T1 only: there is nothing for T2-T4 to unlock, so running a pass
        would report `passes = 1` for a ladder that has no residual stage."""
        result = run_graph(
            corpus, enabled_tiers=(0, 1), measure_calibration_quality=False, store=None
        )
        assert result.ok
        assert "tier_ladder" not in result.node_log
        assert result.passes == 0


class TestStatePropagation:
    def test_each_node_hands_its_product_to_the_next(self, executed):
        run = executed.require()
        assert run.ingest is not None
        assert run.matched.context is not None
        assert run.metrics.link_metrics is not None

    def test_the_audit_log_is_append_only_and_totally_ordered(self, executed):
        sequences = [event.sequence for event in executed.audit.events]
        assert sequences == sorted(sequences)
        assert sequences == list(range(len(sequences)))

    def test_the_log_opens_with_a_run_started_and_closes_with_a_run_completed(
        self, executed
    ):
        assert executed.audit.events[0].event_type is AuditEventType.RUN_STARTED
        assert executed.audit.events[-1].event_type is AuditEventType.RUN_COMPLETED

    def test_every_node_records_entry_and_completion(self, executed):
        entered = {
            event.node
            for event in executed.audit.of_type(AuditEventType.NODE_ENTERED)
        }
        assert entered == set(NODE_SEQUENCE)

    def test_the_loop_records_one_entry_per_iteration(self, executed):
        passes = [
            event
            for event in executed.audit.of_type(AuditEventType.NODE_COMPLETED)
            if event.node == "tier_ladder"
        ]
        assert len(passes) == executed.residual_iterations
        assert [event.payload["pass_number"] for event in passes] == list(
            range(1, executed.residual_iterations + 1)
        )

    def test_decisions_reach_the_log_with_their_tier_and_outcome(self, executed):
        decisions = executed.audit.of_type(AuditEventType.DECISION_MADE)
        assert decisions
        for event in decisions:
            assert event.decision_id is not None
            assert "tier" in event.payload
            assert "outcome" in event.payload

    def test_exceptions_reach_the_log_with_their_class_and_money(self, executed):
        raised = executed.audit.of_type(AuditEventType.EXCEPTION_RAISED)
        assert len(raised) == len(executed.require().exceptions)
        for event in raised:
            assert event.payload["exception_class"]
            assert isinstance(event.payload["impact_minor"], int)


class TestTheGraphAndTheChainAgree:
    """The load-bearing test of Step 11.

    If these ever diverge, every metric in `EVALUATION.md` becomes a claim about
    which code path produced it. They are the same functions in the same order,
    ending in the same `assemble_system_run`; this asserts the wiring did not
    quietly change that.
    """

    def test_the_summary_rows_are_identical(self, pair):
        graph, direct = pair
        assert graph.summary() == direct.summary()

    def test_the_predictions_are_identical(self, pair):
        graph, direct = pair
        assert graph.matched.predictions == direct.matched.predictions

    def test_the_decisions_agree_on_outcome_and_tier(self, pair):
        graph, direct = pair
        assert [
            (d.candidate_id, d.outcome, d.tier) for d in graph.matched.decisions
        ] == [(d.candidate_id, d.outcome, d.tier) for d in direct.matched.decisions]

    def test_the_tier_table_is_identical(self, pair):
        graph, direct = pair
        assert [
            (r.tier, r.candidates_proposed, r.auto_matched, r.marginal_auto_matched)
            for r in graph.metrics.tier_contributions
        ] == [
            (r.tier, r.candidates_proposed, r.auto_matched, r.marginal_auto_matched)
            for r in direct.metrics.tier_contributions
        ]

    def test_the_exception_queue_is_identical(self, pair):
        graph, direct = pair
        assert [
            (e.exception_class, e.severity, e.impact_minor) for e in graph.exceptions
        ] == [
            (e.exception_class, e.severity, e.impact_minor) for e in direct.exceptions
        ]

    def test_the_residual_pass_count_agrees(self, pair):
        graph, direct = pair
        assert graph.matched.passes == direct.matched.passes

    @pytest.mark.parametrize("tiers", [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)])
    def test_they_agree_for_every_ablation_ladder(self, corpus, tiers):
        """The ablation drives `enabled_tiers`, so the graph has to honour it
        the same way -- including the ladders where the loop never runs."""
        graph = run_graph(
            corpus, enabled_tiers=tiers, measure_calibration_quality=False, store=None
        )
        direct = run_system(
            corpus, enabled_tiers=tiers, measure_calibration_quality=False
        )
        assert graph.require().summary() == direct.summary()


class TestNoLlm:
    def test_no_llm_takes_the_same_path_with_one_branch_not_taken(self, corpus):
        """`--no-llm` must stay one code path with a branch, not a second
        implementation. The node still runs; it just does nothing."""
        disabled = LLMClient(config=LLMConfig(enabled=False), provider=None)
        result = run_graph(
            corpus, client=disabled, measure_calibration_quality=False, store=None
        )
        assert result.ok
        assert "normalize_records" in result.node_log
        assert "llm_adjudicate" in result.node_log
        assert result.require().llm_available is False
        assert result.require().cost.llm_calls == 0

    def test_it_produces_the_same_numbers_as_no_client_at_all(self, corpus):
        """A machine with no key and an explicit `--no-llm` reach the same
        place, and the report says which reason applied."""
        disabled = LLMClient(config=LLMConfig(enabled=False), provider=None)
        with_flag = run_graph(
            corpus, client=disabled, measure_calibration_quality=False, store=None
        )
        without_client = run_graph(
            corpus, measure_calibration_quality=False, store=None
        )
        assert with_flag.require().summary() == without_client.require().summary()

    def test_a_key_less_client_never_reaches_t5(self, corpus):
        """A config listing T5 on a machine with no key ran T0-T4, and the tier
        table must not credit a tier that never ran."""
        wanting = LLMClient(config=LLMConfig(enabled=True), provider=None)
        result = run_graph(
            wanting and corpus,
            client=wanting,
            measure_calibration_quality=False,
            store=None,
        )
        run = result.require()
        assert run.config.enabled_tiers == (0, 1, 2, 3, 4)
        assert run.matched.name == "T0-T4"


class TestNodesHoldNoLogic:
    def test_the_node_module_does_no_arithmetic_on_money(self):
        """The property that keeps the graph from becoming a second
        implementation. A node that computed an amount would be reconciliation
        logic living outside `matching`, where nothing tests it as such.
        """
        source = Path(
            __import__("ledgerloop.agent.nodes", fromlist=["__file__"]).__file__
        ).read_text(encoding="utf-8")
        for banned in ("_minor +", "_minor -", "_minor *", "allocate_minor", "sum_minor"):
            assert banned not in source

    def test_it_compares_no_threshold(self):
        source = Path(
            __import__("ledgerloop.agent.nodes", fromlist=["__file__"]).__file__
        ).read_text(encoding="utf-8")
        for banned in ("tau_high", "tau_low", "calibrated_p >", "calibrated_p <"):
            assert banned not in source

    def test_matching_still_does_not_import_llm(self):
        """ARCHITECTURE.md §6, decision 43, re-checked after Step 11 gave the
        ladder four new public entry points."""
        import subprocess
        import sys

        probe = (
            "import sys; import ledgerloop.matching.pipeline; "
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith('ledgerloop.llm'))))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == ""


class TestTheExtraIsOptional:
    def test_the_core_imports_without_langgraph(self):
        """`ledgerloop eval` and every metric in EVALUATION.md must not need the
        extra. Checked in a subprocess with the import blocked."""
        import subprocess
        import sys

        probe = (
            "import builtins, sys\n"
            "real = builtins.__import__\n"
            "def blocked(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'langgraph':\n"
            "        raise ImportError('blocked for the test')\n"
            "    return real(name, *args, **kwargs)\n"
            "builtins.__import__ = blocked\n"
            "import ledgerloop.cli, ledgerloop.eval.harness, ledgerloop.eval.report\n"
            "from ledgerloop.agent.graph import langgraph_available\n"
            "assert langgraph_available() is False\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_the_error_names_the_install_command(self, corpus, monkeypatch):
        """An `ImportError` from four frames down is not an error message."""
        import ledgerloop.agent.graph as graph_module

        def refuse(*args, **kwargs):
            raise ImportError("no langgraph")

        monkeypatch.setattr(graph_module, "build_recon_graph", refuse)
        with pytest.raises(ImportError):
            graph_module.build_recon_graph(_resources(corpus))

    def test_the_hint_mentions_the_extra_and_that_eval_does_not_need_it(self):
        from ledgerloop.agent.graph import _INSTALL_HINT

        assert "[graph]" in _INSTALL_HINT
        assert "EVALUATION.md" in _INSTALL_HINT
        assert issubclass(LangGraphUnavailable, RuntimeError)
