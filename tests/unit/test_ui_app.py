"""The Streamlit script, executed the way ``streamlit run`` executes it.

``AppTest`` runs ``app.py`` top to bottom in-process and collects the widgets
and exceptions it produced. That is what makes these real UI tests rather than
a smoke check: a page that renders a traceback still returns HTTP 200, and only
running the script catches it.

The app reads runs from ``reports/runs``, so every test here points it at a
temporary directory. A UI test that wrote into the project's own run store
would leave fixtures a demo would then display.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import SplitName

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)

#: Absolute, because ``AppTest.from_file`` resolves a relative path against
#: the *calling test file* rather than the working directory.
APP = str(
    Path(__file__).resolve().parents[2] / "src" / "ledgerloop" / "ui" / "app.py"
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A dataset, and one completed run in an isolated store."""
    root = tmp_path_factory.mktemp("app")
    corpus = root / "data" / "dev-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), corpus)
    runs = root / "runs"
    run_graph(corpus, measure_calibration_quality=False, store=runs, run_id="app-run")
    return corpus, runs


def _app(monkeypatch, runs, data_root):
    """The script, pointed at a temporary store rather than the project's.

    Through the environment, because ``AppTest`` re-executes the script in a
    fresh namespace: a patched module constant would be overwritten on the
    first line of the run.
    """
    monkeypatch.setenv("LEDGERLOOP_RUNS_DIR", str(runs))
    monkeypatch.setenv("LEDGERLOOP_DATA_DIR", str(data_root))
    monkeypatch.setenv("LEDGERLOOP_CALIBRATION", str(runs / "no-such-bundle.json"))
    return AppTest.from_file(APP, default_timeout=120)


class TestTheScriptRuns:
    def test_it_renders_without_raising(self, workspace, monkeypatch):
        """A page that renders a traceback still returns HTTP 200. Only running
        the script catches that."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert not test.exception
        assert test.title[0].value == "LedgerLoop"

    def test_it_shows_the_four_tabs(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [tab.label for tab in test.tabs]
        assert labels == ["Run", "Results", "Exceptions", "Audit replay"]

    def test_it_lists_the_completed_run(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert any("app-run" in str(option) for option in test.radio[0].options)

    def test_an_empty_store_says_so_rather_than_rendering_blank_screens(
        self, workspace, monkeypatch, tmp_path
    ):
        """A dashboard with no data must say it has none. Rendering empty
        tables would look like a run that found nothing."""
        corpus, _ = workspace
        test = _app(monkeypatch, tmp_path / "empty", corpus.parent).run()
        assert not test.exception
        messages = [item.value for item in test.info]
        assert any("No runs yet" in message for message in messages)
        assert any("No run selected" in message for message in messages)


class TestTheResultsScreen:
    def test_it_shows_the_headline_three(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [metric.label for metric in test.metric]
        for expected in (
            "Auto-match precision",
            "Match rate",
            "Link recall",
            "Exception recall",
        ):
            assert expected in labels

    def test_it_shows_every_decision_outcome_separately(self, workspace, monkeypatch):
        """AUTO_MATCHED, NEEDS_REVIEW, EXCEPTION and REJECTED are four figures.
        A single 'matched' number would be the most misleading thing here."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [metric.label for metric in test.metric]
        for outcome in ("AUTO_MATCHED", "NEEDS_REVIEW", "EXCEPTION", "REJECTED"):
            assert outcome in labels

    def test_a_clean_run_reports_zero_false_positives_rather_than_silence(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert any("zero** false positives" in item.value for item in test.success)

    def test_it_states_that_no_llm_ran(self, workspace, monkeypatch):
        """The demo runs deterministically by default and the UI must say so
        rather than leaving a reader to assume a model was involved."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert any("No LLM ran" in item.value for item in test.info)

    def test_it_names_the_unmatchable_floor(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "unmatchable by construction" in captions
        assert "not a model" in captions


class TestTheExceptionScreen:
    def test_it_renders_the_queue_and_says_how_it_is_sorted(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "sorted by rupee impact descending" in captions

    def test_it_offers_a_severity_and_a_class_filter(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [box.label for box in test.selectbox]
        assert "Severity" in labels
        assert "Class" in labels

    def test_it_shows_an_evidence_chain(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "Root cause." in markdown
        assert "Suggested action." in markdown

    def test_it_says_when_the_agent_may_not_resolve_an_item(
        self, workspace, monkeypatch
    ):
        """The agent proposes and never posts. A row that did not say which is
        which would claim an authority the resolver does not have."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        text = " ".join(
            [item.value for item in test.warning] + [item.value for item in test.info]
        )
        assert "never posts" in text or "needs a person" in text


class TestTheAuditScreen:
    def test_it_offers_a_record_picker(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [box.label for box in test.selectbox]
        assert "Record" in labels

    def test_it_explains_the_final_outcome(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        headers = " ".join(item.value for item in test.subheader)
        assert "Final outcome" in headers

    def test_it_says_the_replay_is_read_not_recomputed(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "nothing here is re-derived" in captions


class TestTheRunScreen:
    def test_it_offers_the_datasets_on_disk(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [box.label for box in test.selectbox]
        assert "Dataset" in labels

    def test_it_can_start_a_reconciliation_end_to_end(
        self, workspace, monkeypatch, tmp_path
    ):
        """The one action in the UI that runs anything. It calls the same
        `run_graph` the CLI does, and the run it produces must be readable by
        the same store the other screens read."""
        corpus, _ = workspace
        fresh = tmp_path / "fresh-runs"
        test = _app(monkeypatch, fresh, corpus.parent).run()
        assert not test.exception

        test.button(key="reconcile").click().run()
        assert not test.exception

        from ledgerloop.agent.store import list_runs

        produced = list_runs(fresh)
        assert len(produced) == 1
        assert produced[0].metrics["auto_match_precision"] == 1.0

    def test_it_can_generate_a_dataset(self, workspace, monkeypatch, tmp_path):
        _, runs = workspace
        data_root = tmp_path / "fresh-data"
        data_root.mkdir()
        test = _app(monkeypatch, runs, data_root).run()
        assert not test.exception
        test.button(key="generate").click().run()
        assert not test.exception
        assert (data_root / "dev-standard-42" / "manifest.json").is_file()
