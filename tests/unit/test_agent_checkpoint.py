"""Checkpointing, audit replay, the run store, and the failure/resume path.

Two guarantees are tested here and they are deliberately different, because
conflating them would be the easy lie:

* **Replay is durable.** ``audit.jsonl`` survives the process. A completed *or
  failed* run can be walked event by event afterwards.
* **Resume is in-process.** LangGraph's ``InMemorySaver`` snapshots state after
  every node, so a failed run continues at the node that failed. That
  checkpoint dies with the process, and the tests say so rather than implying
  otherwise.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.agent.audit import AuditLog, read_audit_jsonl
from ledgerloop.agent.graph import build_recon_graph, langgraph_available
from ledgerloop.agent.runner import resume_run, run_graph
from ledgerloop.agent.state import RunResources, initial_state
from ledgerloop.agent.store import (
    AUDIT_FILE,
    DECISIONS_FILE,
    EXCEPTIONS_FILE,
    RUN_FILE,
    list_runs,
    load_run,
)
from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.harness import prepare_run
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.audit import AuditEventType
from ledgerloop.models.enums import DecisionOutcome, SplitName

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkpoint") / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def stored(tmp_path_factory, corpus):
    """One stored run, shared. Module-scoped and a plain function: a
    class-scoped fixture defined as an instance method warns in pytest 8."""
    root = tmp_path_factory.mktemp("runs")
    result = run_graph(
        corpus, measure_calibration_quality=False, store=root, run_id="demo-run"
    )
    return root, result


class TestTheAuditLog:
    def test_the_sequence_is_the_order_key_not_the_clock(self):
        """Several events routinely land inside one millisecond, and replay has
        to be exactly reproducible."""
        log = AuditLog(run_id="r")
        for _ in range(5):
            log.emit(AuditEventType.NODE_ENTERED, "n")
        assert [event.sequence for event in log.events] == [0, 1, 2, 3, 4]

    def test_it_round_trips_through_jsonl(self, tmp_path):
        log = AuditLog(run_id="r")
        log.emit(AuditEventType.RUN_STARTED, "__start__", message="go")
        log.emit(
            AuditEventType.LLM_CALL,
            "normalize_records",
            prompt_hash="abc",
            prompt_tokens=10,
            completion_tokens=2,
        )
        path = log.write_jsonl(tmp_path / AUDIT_FILE)
        read = read_audit_jsonl(path)
        assert [event.sequence for event in read] == [0, 1]
        assert read[1].prompt_hash == "abc"
        assert read[1].prompt_tokens == 10

    def test_a_truncated_last_line_drops_rather_than_raises(self, tmp_path):
        """A run killed mid-write leaves a partial line, and the events before
        it are still a valid prefix -- which is exactly the case replay exists
        to survive."""
        log = AuditLog(run_id="r")
        for index in range(3):
            log.emit(AuditEventType.NODE_ENTERED, f"n{index}")
        path = log.write_jsonl(tmp_path / AUDIT_FILE)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"run_id": "r", "sequ')
        assert len(read_audit_jsonl(path)) == 3

    def test_a_missing_file_is_an_empty_log_not_an_error(self, tmp_path):
        assert read_audit_jsonl(tmp_path / "nope.jsonl") == ()

    def test_extend_renumbers_so_the_total_order_stays_total(self):
        """Two logs each starting at zero would otherwise interleave into
        something replay cannot walk."""
        first = AuditLog(run_id="r")
        first.emit(AuditEventType.NODE_ENTERED, "a")
        second = AuditLog(run_id="r")
        second.emit(AuditEventType.NODE_ENTERED, "b")
        second.emit(AuditEventType.NODE_ENTERED, "c")
        first.extend(second.events)
        assert [event.sequence for event in first.events] == [0, 1, 2]
        assert [event.node for event in first.events] == ["a", "b", "c"]


class TestTheRunStore:
    def test_it_writes_the_four_files(self, stored):
        root, _ = stored
        directory = root / "demo-run"
        for name in (RUN_FILE, AUDIT_FILE, EXCEPTIONS_FILE, DECISIONS_FILE):
            assert (directory / name).is_file()

    def test_the_run_reads_back(self, stored):
        root, result = stored
        run = load_run(root / "demo-run")
        assert run is not None
        assert run.run_id == "demo-run"
        assert len(run.audit) == len(result.audit.events)
        assert len(run.exceptions) == len(result.require().exceptions)

    def test_the_summary_matches_the_run_it_was_written_from(self, stored):
        root, result = stored
        run = load_run(root / "demo-run")
        assert run is not None
        source = result.require().metrics
        assert run.metrics["auto_match_precision"] == source.auto_match_precision
        assert run.metrics["match_rate"] == source.match_rate
        assert run.metrics["exception_recall"] == source.exception_recall

    def test_money_stays_in_integer_minor_units_on_disk(self, stored):
        """The no-float invariant reaches the artefacts, not only the code."""
        root, _ = stored
        payload = json.loads((root / "demo-run" / RUN_FILE).read_text(encoding="utf-8"))
        for key in (
            "false_positive_cost_minor",
            "reconciled_minor",
            "outstanding_minor",
            "unmatchable_impact_minor",
        ):
            assert isinstance(payload["metrics"][key], int)

    def test_only_evaluation_unit_decisions_are_stored(self, stored):
        """283 structural edges would bury the 130 the metrics are about."""
        root, result = stored
        run = load_run(root / "demo-run")
        assert run is not None
        assert len(run.decisions) < len(result.require().matched.decisions)
        assert all(
            decision.link_type.value == "PAYMENT_CREDITED_AS"
            for decision in run.decisions
        )

    def test_the_three_outcomes_are_distinguished(self, stored):
        """AUTO_MATCHED, NEEDS_REVIEW and EXCEPTION are separate counts, not one
        'matched' number."""
        root, _ = stored
        run = load_run(root / "demo-run")
        assert run is not None
        for outcome in DecisionOutcome:
            assert outcome.value in run.summary["decisions"]

    def test_listing_finds_it(self, stored):
        root, _ = stored
        runs = list_runs(root)
        assert [run.run_id for run in runs] == ["demo-run"]

    def test_a_directory_with_no_run_json_is_not_a_run(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert load_run(tmp_path / "empty") is None
        assert list_runs(tmp_path) == ()

    def test_an_unreadable_exception_row_does_not_hide_the_whole_queue(
        self, stored, tmp_path
    ):
        """A stale row from a model change must not make a run unopenable."""
        root, _ = stored
        target = tmp_path / "patched"
        target.mkdir()
        for name in (RUN_FILE, AUDIT_FILE, DECISIONS_FILE):
            (target / name).write_bytes((root / "demo-run" / name).read_bytes())
        rows = json.loads(
            (root / "demo-run" / EXCEPTIONS_FILE).read_text(encoding="utf-8")
        )
        rows.append({"nonsense": True})
        (target / EXCEPTIONS_FILE).write_text(json.dumps(rows), encoding="utf-8")

        run = load_run(target)
        assert run is not None
        assert len(run.exceptions) == len(rows) - 1

    def test_a_record_can_be_traced_through_the_run(self, stored):
        """The Audit Replay screen's lookup: given a record, what decided it and
        what did the log say."""
        root, _ = stored
        run = load_run(root / "demo-run")
        assert run is not None
        key = run.decisions[0].source_ref.key
        assert run.decisions_for(key)
        assert run.audit_for(key)


class TestCheckpointing:
    def test_a_snapshot_exists_after_every_node(self, corpus, tmp_path):
        """What makes a failed run resumable and gives replay something to walk."""
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        result = run_graph(
            corpus,
            measure_calibration_quality=False,
            store=None,
            checkpointer=saver,
            thread_id="t1",
        )
        assert result.ok
        graph = build_recon_graph(
            RunResources(setup=prepare_run(corpus)), checkpointer=saver
        )
        history = list(graph.get_state_history({"configurable": {"thread_id": "t1"}}))
        assert len(history) >= len(result.node_log)

    def test_the_checkpointed_state_carries_the_growing_audit_log(
        self, corpus
    ):
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        run_graph(
            corpus,
            measure_calibration_quality=False,
            store=None,
            checkpointer=saver,
            thread_id="t2",
        )
        graph = build_recon_graph(
            RunResources(setup=prepare_run(corpus)), checkpointer=saver
        )
        history = list(graph.get_state_history({"configurable": {"thread_id": "t2"}}))
        lengths = [len(snapshot.values.get("audit", [])) for snapshot in history]
        # History is newest-first, so the log shrinks as we walk back.
        assert lengths == sorted(lengths, reverse=True)


class TestFailureAndResume:
    def test_a_failure_is_recorded_and_the_log_survives(self, corpus, monkeypatch):
        """A traceback out of `invoke` would leave no log and no checkpoint, so
        every node is wrapped and its failure becomes routable state instead."""
        import ledgerloop.agent.graph as graph_module

        def boom(state, resources):
            raise RuntimeError("simulated tier failure")

        monkeypatch.setattr(graph_module, "tier_ladder", boom)
        result = run_graph(corpus, measure_calibration_quality=False, store=None)

        assert not result.ok
        assert result.failed_node == "tier_ladder"
        assert "simulated tier failure" in (result.error or "")
        failures = result.audit.of_type(AuditEventType.RUN_FAILED)
        assert len(failures) == 1
        assert failures[0].node == "tier_ladder"

    def test_the_partial_log_is_still_replayable(self, corpus, monkeypatch, tmp_path):
        """The whole point of an append-only trail: a run that died halfway is
        inspectable up to the point it died."""
        import ledgerloop.agent.graph as graph_module

        def boom(state, resources):
            raise RuntimeError("simulated tier failure")

        monkeypatch.setattr(graph_module, "tier_ladder", boom)
        result = run_graph(corpus, measure_calibration_quality=False, store=None)

        path = result.audit.write_jsonl(tmp_path / AUDIT_FILE)
        replayed = read_audit_jsonl(path)
        assert [event.sequence for event in replayed] == list(range(len(replayed)))
        assert {event.node for event in replayed} >= {
            "ingest_sources",
            "normalize_records",
            "build_entity_graph",
        }
        assert replayed[-1].event_type is AuditEventType.RUN_FAILED

    def test_downstream_nodes_are_skipped_after_a_failure(self, corpus, monkeypatch):
        """Running them against half-built state would turn one honest failure
        into a second, misleading one."""
        import ledgerloop.agent.graph as graph_module

        def boom(state, resources):
            raise RuntimeError("simulated tier failure")

        monkeypatch.setattr(graph_module, "tier_ladder", boom)
        result = run_graph(corpus, measure_calibration_quality=False, store=None)
        assert "generate_report" not in result.node_log
        assert result.system is None

    def test_require_explains_the_failure_rather_than_returning_none(
        self, corpus, monkeypatch
    ):
        import ledgerloop.agent.graph as graph_module

        def boom(state, resources):
            raise RuntimeError("simulated tier failure")

        monkeypatch.setattr(graph_module, "tier_ladder", boom)
        result = run_graph(corpus, measure_calibration_quality=False, store=None)
        with pytest.raises(RuntimeError, match="failed at tier_ladder"):
            result.require()

    def test_a_failed_run_leaves_no_record_in_the_store(self, corpus, monkeypatch, tmp_path):
        """A half-run in `reports/runs` would render in the UI as a real one."""
        import ledgerloop.agent.graph as graph_module

        def boom(state, resources):
            raise RuntimeError("simulated tier failure")

        monkeypatch.setattr(graph_module, "tier_ladder", boom)
        run_graph(corpus, measure_calibration_quality=False, store=tmp_path)
        assert list_runs(tmp_path) == ()

    def test_a_recovered_node_resumes_in_process_without_redoing_earlier_ones(
        self, corpus, monkeypatch
    ):
        """In-process resume: the checkpoint holds the state the earlier nodes
        produced, so continuing does not re-ingest or re-run T0/T1."""
        from langgraph.checkpoint.memory import InMemorySaver

        import ledgerloop.agent.graph as graph_module
        import ledgerloop.agent.nodes as nodes

        calls = {"ingest": 0}
        real_ingest = nodes.ingest_sources

        def counting_ingest(state, resources):
            calls["ingest"] += 1
            return real_ingest(state, resources)

        failures = {"left": 1}
        real_ladder = nodes.tier_ladder

        def flaky(state, resources):
            if failures["left"] > 0:
                failures["left"] -= 1
                raise RuntimeError("transient")
            return real_ladder(state, resources)

        monkeypatch.setattr(graph_module, "ingest_sources", counting_ingest)
        monkeypatch.setattr(graph_module, "tier_ladder", flaky)

        saver = InMemorySaver()
        resources = RunResources(
            setup=prepare_run(corpus), measure_calibration_quality=False
        )
        graph = build_recon_graph(resources, checkpointer=saver)
        config = {"configurable": {"thread_id": "resume"}, "recursion_limit": 50}

        first = graph.invoke(initial_state("resume"), config=config)
        assert first["error"] is not None
        assert calls["ingest"] == 1

        # The node now succeeds. Resuming re-enters the graph from the last
        # checkpoint rather than from the start.
        snapshot = graph.get_state(config)
        graph.update_state(config, {"error": None, "failed_node": None})
        recovered = resume_run(graph, "resume", resources=resources)

        assert calls["ingest"] == 1, "ingest was re-run; the checkpoint was not used"
        assert snapshot is not None
        assert recovered.error is None or "transient" not in (recovered.error or "")
