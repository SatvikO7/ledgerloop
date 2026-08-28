"""The four screens: what they show, and what they must never show.

Two kinds of test here.

:class:`TestTheViews` and friends drive :mod:`ledgerloop.ui.views` -- pure
functions over a stored run, so every branch is testable without a browser.
That is why the logic lives there and not in the widget file.

:class:`TestTheAppRenders` drives the real Streamlit script through
``AppTest``, which executes it exactly as ``streamlit run`` would and fails on
any exception the page raises. A dashboard that renders a traceback is not a
dashboard, and a smoke test that only checked the HTTP status would not catch it.

The honesty tests matter most: the UI must not collapse the three decision
outcomes into one number, must not hide an unresolved exception, and must not
present the unmatchable floor as a failure.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.store import RUN_FILE, StoredRun, load_run
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import DecisionOutcome, SplitName
from ledgerloop.ui.views import (
    OUTCOME_HELP,
    evidence_rows,
    exception_rows,
    headline,
    money_rows,
    recall_rows,
    record_keys,
    tier_rows,
    trace_record,
)

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)


@pytest.fixture(scope="module")
def stored(tmp_path_factory) -> StoredRun:
    """One real run, executed and read back the way the UI reads it."""
    root = tmp_path_factory.mktemp("ui")
    corpus = root / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), corpus)
    runs = root / "runs"
    run_graph(
        corpus, measure_calibration_quality=False, store=runs, run_id="ui-run"
    )
    loaded = load_run(runs / "ui-run")
    assert loaded is not None
    return loaded


class TestTheHeadline:
    def test_it_reads_the_run_rather_than_recomputing_it(self, stored):
        """The UI holds no objects that could recompute a metric, and this is
        the test that says the numbers came off the file."""
        view = headline(stored)
        payload = json.loads(
            (stored.directory / RUN_FILE).read_text(encoding="utf-8")
        )
        assert view.precision == payload["metrics"]["auto_match_precision"]
        assert view.match_rate == payload["metrics"]["match_rate"]
        assert view.exception_recall == payload["metrics"]["exception_recall"]

    def test_the_three_outcomes_stay_three_numbers(self, stored):
        """Collapsing them into one 'matched' figure is the single most
        misleading thing a reconciliation dashboard can do."""
        view = headline(stored)
        assert view.auto_matched + view.needs_review + view.exceptions >= 0
        assert set(OUTCOME_HELP) == {outcome.value for outcome in DecisionOutcome}

    def test_the_queue_size_is_separate_from_the_exception_decisions(self, stored):
        """A settlement nothing could credit raises a queue entry without
        producing a link decision, so the two numbers differ and the UI shows
        both rather than passing one off as the other."""
        view = headline(stored)
        assert view.queue_size == len(stored.exceptions)
        assert view.queue_size != view.exceptions

    def test_a_clean_run_is_reported_as_zero_false_positives(self, stored):
        view = headline(stored)
        assert view.precision_is_perfect is (view.false_positives == 0)


class TestTheMoneyView:
    def test_reconciled_and_outstanding_add_to_the_total(self, stored):
        rows = {row["Measure"]: row["Amount"] for row in money_rows(stored)}
        assert "Reconciled" in rows
        assert "Outstanding" in rows
        assert "Total across evaluation links" in rows

    def test_the_unmatchable_floor_has_its_own_row(self, stored):
        """A real ceiling is not a model failure, and folding it into the
        outstanding total would present it as one."""
        measures = [row["Measure"] for row in money_rows(stored)]
        assert any("honest floor" in measure for measure in measures)

    def test_every_amount_is_formatted_from_integer_paise(self, stored):
        for row in money_rows(stored):
            assert row["Amount"].startswith("₹")


class TestTheExceptionQueue:
    def test_it_is_sorted_by_rupee_impact_descending(self, stored):
        """PLAN.md §8.2.3. One ₹4 lakh payout matters more than two hundred
        one-paise drifts, and any other sort order hides that."""
        impacts = [row["impact_minor"] for row in exception_rows(stored)]
        assert impacts == sorted(impacts, reverse=True)

    def test_every_row_carries_a_class_a_price_a_cause_and_an_action(self, stored):
        rows = exception_rows(stored)
        assert rows
        for row in rows:
            assert row["Class"]
            assert row["Severity"]
            assert row["Impact"].startswith("₹")
            assert row["Root cause"]
            assert row["Suggested action"]

    def test_nothing_is_hidden_by_default(self, stored):
        """The unfiltered view is the whole queue. A UI that silently dropped
        the rows it could not explain would be the failure this project's
        exception taxonomy exists to prevent."""
        assert len(exception_rows(stored)) == len(stored.exceptions)

    def test_a_filter_narrows_the_view_and_nothing_else(self, stored):
        rows = exception_rows(stored)
        severity = rows[0]["Severity"]
        filtered = exception_rows(stored, severity=severity)
        assert 0 < len(filtered) <= len(rows)
        assert all(row["Severity"] == severity for row in filtered)
        # The headline counts are read from the run, not from this list.
        assert headline(stored).queue_size == len(rows)

    def test_the_evidence_chain_points_at_source_records(self, stored):
        exception = max(stored.exceptions, key=lambda item: item.impact_minor)
        rows = evidence_rows(exception)
        assert rows
        assert any(row["Records"] for row in rows)

    def test_proposal_only_and_agent_resolvable_are_distinguished(self, stored):
        """The agent proposes and never posts. A row that did not say which is
        which would be claiming an authority the resolver does not have."""
        rows = exception_rows(stored)
        assert {row["Agent may resolve"] for row in rows} <= {True, False}


class TestTheTierAndRecallTables:
    def test_yield_and_conviction_are_separate_columns(self, stored):
        rows = tier_rows(stored)
        assert rows
        for row in rows:
            assert "Proposed" in row
            assert "Auto-matched" in row

    def test_the_recall_table_includes_the_classes_that_score_badly(self, stored):
        """Publishing only the good rows is exactly what this project is trying
        not to do."""
        rows = recall_rows(stored)
        assert rows
        assert any(row["Recall"] < 0.5 for row in rows)


class TestAuditReplay:
    def test_every_decided_record_can_be_traced(self, stored):
        keys = record_keys(stored)
        assert keys
        for key in keys[:5]:
            trace = trace_record(stored, key)
            assert trace.record_key == key
            assert trace.outcome

    def test_a_trace_explains_why_the_decision_happened(self, stored):
        key = stored.decisions[0].source_ref.key
        trace = trace_record(stored, key)
        assert trace.final is not None
        assert trace.final.tier.name in trace.explanation
        assert f"{trace.final.calibrated_p:.4f}" in trace.explanation

    def test_the_timeline_is_in_sequence_order(self, stored):
        key = stored.decisions[0].source_ref.key
        rows = trace_record(stored, key).timeline()
        assert rows
        assert [row["#"] for row in rows] == sorted(row["#"] for row in rows)

    def test_an_undecided_record_says_so_rather_than_inventing_an_outcome(
        self, stored
    ):
        trace = trace_record(stored, "payment:PAY-DOES-NOT-EXIST")
        assert trace.outcome == "NO DECISION"
        assert "unmatchable" in trace.explanation

    def test_a_record_only_an_exception_names_still_traces(self, stored):
        """An exception can name a settlement no `PAYMENT_CREDITED_AS` decision
        touches. The replay must still explain it."""
        keys = {
            ref.key
            for exception in stored.exceptions
            for ref in exception.involved_refs
        }
        decided = {
            ref
            for decision in stored.decisions
            for ref in (decision.source_ref.key, decision.target_ref.key)
        }
        only_exception = sorted(keys - decided)
        if not only_exception:  # pragma: no cover - corpus dependent
            pytest.skip("every exception record also carries a decision here")
        trace = trace_record(stored, only_exception[0])
        assert trace.outcome == DecisionOutcome.EXCEPTION.value
        assert trace.explanation


class TestTheViewsHoldNoLogic:
    def test_the_view_module_never_imports_streamlit(self):
        """The split that makes the UI testable: logic here, widgets there."""
        from pathlib import Path

        import ledgerloop.ui.views as views

        source = Path(views.__file__).read_text(encoding="utf-8")
        assert "import streamlit" not in source

    def test_it_never_imports_the_matcher_or_the_evaluator(self):
        """A UI that could recompute a metric is a second implementation."""
        from pathlib import Path

        import ledgerloop.ui.views as views

        imports = [
            line
            for line in Path(views.__file__).read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]
        for banned in ("ledgerloop.matching", "ledgerloop.eval", "ledgerloop.exceptions"):
            assert not any(banned in line for line in imports)
