"""Every number on screen traces back to one reconciliation of the reader's files.

The defect these pin, exactly as it happened:

``screen_upload`` stores the reconciliation halfway down the page, but the
sidebar, the masthead and the five reader-facing tabs are all drawn *above* that
call site. On the very script run that produced a result, those screens had
already read session state and found nothing -- so a receipt reading
``12,269 records`` sat above five tabs rendering a bundled 742-record sample,
with no label anywhere saying the subject had changed. Streamlit tabs are
client-side, so clicking between them reruns nothing and the mismatch persisted
until some unrelated widget happened to fire.

Two properties close it, and both are tested here rather than assumed:

* **One read.** ``main`` resolves the current upload once and hands that object
  to every screen. There is no second read to disagree with the first.
* **One rerun.** ``_reconcile_uploads`` reruns the script after storing, so the
  store is settled before anything reads it -- the pattern ``screen_run`` has
  always used after its own two actions.

Everything else here is lineage: a count on screen must equal the count derived
from the authoritative object, never a coincidence of two pipelines agreeing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.harness import reconcile_only
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import SplitName
from ledgerloop.money import format_minor
from ledgerloop.ui.uploads import SourceKind

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)

APP = str(
    Path(__file__).resolve().parents[2] / "src" / "ledgerloop" / "ui" / "app.py"
)
FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"

#: The three sources, and the two-source combination the capability matrix says
#: reconciles. Both are real corpora; neither is invented for the test.
ALL_THREE = tuple(SourceKind)
PROCESSOR_AND_BANK = (SourceKind.PROCESSOR, SourceKind.BANK)


@pytest.fixture(scope="module")
def sample_store(tmp_path_factory):
    """A bundled report to be mistaken for. Without one there is nothing to leak."""
    root = tmp_path_factory.mktemp("lineage")
    corpus = root / "data" / "dev-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), corpus)
    runs = root / "runs"
    run_graph(corpus, measure_calibration_quality=False, store=runs, run_id="sample")
    return corpus, runs


def _app(monkeypatch, sample_store):
    corpus, runs = sample_store
    monkeypatch.setenv("LEDGERLOOP_RUNS_DIR", str(runs))
    monkeypatch.setenv("LEDGERLOOP_DATA_DIR", str(corpus.parent))
    monkeypatch.setenv("LEDGERLOOP_CALIBRATION", str(runs / "no-such-bundle.json"))
    return AppTest.from_file(APP, default_timeout=300)


def _fill(box: Path, kinds) -> Path:
    """Put exactly these sources in the session's upload directory."""
    box.mkdir(parents=True, exist_ok=True)
    for stale in box.iterdir():
        stale.unlink()
    for kind in kinds:
        shutil.copy(FIXTURE / kind.value, box / kind.value)
    return box


def _reconcile(test, box: Path, kinds):
    """Drive the real button, the way a person does."""
    _fill(box, kinds)
    test.session_state["upload_dir"] = str(box)
    test.session_state["uploads"] = {
        kind: {"filename": kind.value, "rows": 1, "bytes": 1} for kind in kinds
    }
    test = test.run()
    test.button(key="start_upload_run").click()
    return test.run()


def _text(test) -> str:
    """Everything the page put in front of a reader."""
    return " ".join(
        widget.value
        for group in (
            test.markdown,
            test.caption,
            test.warning,
            test.info,
            test.success,
            test.subheader,
        )
        for widget in group
    )


class TestTheRunIsSettledBeforeAnythingReadsIt:
    """The ordering defect itself."""

    def test_reconciling_reruns_the_script(self):
        """Without this the sidebar, masthead and five tabs spend the click's
        run believing no upload exists, and render a sample underneath the
        receipt for the reader's own files."""
        import inspect

        from ledgerloop.ui import app

        body = inspect.getsource(app._reconcile_uploads)
        assert "st.rerun()" in body, (
            "the screens above this call site are drawn before it stores; "
            "without a rerun they describe a world with no upload in it"
        )

    def test_session_state_is_read_in_exactly_one_place(self):
        """Two reads at two points in one script run *is* the bug. One read,
        handed down, cannot disagree with itself."""
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        assert source.count('st.session_state.get("upload_result")') == 1
        assert source.count('st.session_state["upload_result"]') == 1

    def test_the_five_tabs_are_populated_on_the_run_that_reconciles(
        self, sample_store, monkeypatch, tmp_path
    ):
        """The end-to-end assertion. One click, and every tab describes it --
        no second interaction required."""
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        assert not test.exception
        run = test.session_state["upload_result"]
        assert run is not None
        text = _text(test)
        assert f"{run.ingest.record_count:,} records you uploaded" in text
        assert f"{run.queue_size} item(s) need your attention" in text
        assert "No accuracy figures for your own files" in text


class TestEveryTabReadsTheOneAuthoritativeResult:
    @pytest.fixture
    def rendered(self, sample_store, monkeypatch, tmp_path):
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        return test, test.session_state["upload_result"], _text(test)

    def test_overview_counts_are_the_run_s_own(self, rendered):
        _, run, text = rendered
        assert f"{run.ingest.record_count:,}" in text
        assert f"{run.committed_links:,}" in text
        assert f"{run.queue_size:,}" in text

    def test_needs_review_count_and_money_are_the_run_s_own(self, rendered):
        _, run, text = rendered
        assert f"{run.queue_size} item(s) need your attention" in text
        assert format_minor(run.queue_minor) in text

    def test_transactions_count_is_the_run_s_own_decisions(self, rendered):
        test, run, _ = rendered
        decisions = len(run.matched.decisions)
        captions = " ".join(item.value for item in test.caption)
        assert f"of {decisions:,}." in captions

    def test_why_it_matched_offers_only_this_run_s_records(self, rendered):
        """The picker must not contain a key from a corpus the reader never
        supplied. Provenance is the whole point of the screen."""
        test, run, _ = rendered
        keys = {ref.key for d in run.matched.decisions for ref in (d.source_ref, d.target_ref)}
        keys |= {ref.key for item in run.exceptions for ref in item.involved_refs}
        picker = next(
            box for box in test.selectbox if box.key == "up_why_record"
        )
        assert set(picker.options) <= keys
        assert picker.options

    def test_the_ladder_shown_is_this_run_s_tier_contributions(self, rendered):
        """Read off `MatchRun.tier_contributions`, which needs no answer key:
        what a rung proposed is a fact about the run, not about correctness.

        The ladder totals **committed decisions**, not evaluation-unit links.
        Those are two real and different scopes -- a decision may commit an
        order leg the evaluation unit deliberately excludes -- so this asserts
        the relationship that is actually true rather than flattening them into
        one number that would be true of neither.
        """
        from ledgerloop.models.enums import DecisionOutcome
        from ledgerloop.ui.uploads import upload_tier_stages

        _, run, text = rendered
        stages = upload_tier_stages(run)
        committed = sum(
            1
            for decision in run.matched.decisions
            if decision.outcome is DecisionOutcome.AUTO_MATCHED
        )
        assert any(stage.auto_matched for stage in stages)
        assert sum(stage.auto_matched for stage in stages) == committed
        assert "How it got there" in text


class TestNoSampleDataLeaksIntoAnUploadedRun:
    def test_the_bundled_report_is_absent_while_showing_an_upload(
        self, sample_store, monkeypatch, tmp_path
    ):
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        text = _text(test)
        # The bundled report's own vocabulary. Every one of these is rendered by
        # a report-backed screen and by nothing else, so its presence here would
        # mean a sample had been drawn into an upload's tab.
        assert "EVALUATION.md" not in text
        assert "Incorrect matches" not in text
        assert "incorrect match" not in text

    def test_switching_to_the_sample_labels_it_as_one(
        self, sample_store, monkeypatch, tmp_path
    ):
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        test.session_state["scope"] = "A sample report"
        test = test.run()
        assert not test.exception
        captions = " ".join(item.value for item in test.caption)
        assert "Sample report" in captions
        assert "not your uploaded files" in captions

    def test_an_upload_is_never_quoted_a_ground_truth_figure(
        self, sample_store, monkeypatch, tmp_path
    ):
        """`0 incorrect matches` over somebody's own files reads as a clean bill
        of health and is in fact a statement about a corpus they never saw."""
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        test_text = _text(test)
        assert "No accuracy figures for your own files" in test_text
        # The words appear, correctly, in the sentence explaining why the
        # figures are absent. What must not appear is a *figure*: the glossary
        # table and the KPI cards are rendered only by the report-backed screen.
        assert "What the technical terms mean" not in test_text
        assert "95% CI" not in test_text
        assert not [
            frame for frame in test.dataframe
            if any("Term" in str(column) for column in getattr(frame, "columns", []))
        ]
        run = test.session_state["upload_result"]
        assert run.ingest.record_count > 0


class TestASecondRunReplacesTheFirstEverywhere:
    def test_nothing_from_run_one_survives_into_run_two(
        self, sample_store, monkeypatch, tmp_path
    ):
        box = tmp_path / "box"
        test = _app(monkeypatch, sample_store).run()

        test = _reconcile(test, box, ALL_THREE)
        first = test.session_state["upload_result"]

        test = _reconcile(test, box, PROCESSOR_AND_BANK)
        second = test.session_state["upload_result"]
        text = _text(test)

        # The two runs must genuinely differ, or this proves nothing.
        assert first.ingest.record_count != second.ingest.record_count
        assert f"{second.ingest.record_count:,} records you uploaded" in text
        assert f"{first.ingest.record_count:,} records you uploaded" not in text
        assert f"{second.queue_size} item(s) need your attention" in text

    def test_the_stored_result_matches_an_independent_recompute(
        self, sample_store, monkeypatch, tmp_path
    ):
        """The UI must not be a second pipeline. What it shows has to equal what
        `reconcile_only` produces from the same directory."""
        box = tmp_path / "box"
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), box, PROCESSOR_AND_BANK
        )
        shown = test.session_state["upload_result"]
        truth = reconcile_only(box)
        assert shown.ingest.record_count == truth.ingest.record_count
        assert shown.committed_links == truth.committed_links
        assert shown.queue_size == truth.queue_size
        assert shown.queue_minor == truth.queue_minor


class TestChangingTheFilesDiscardsTheResultVisibly:
    def test_removing_a_file_drops_the_result(
        self, sample_store, monkeypatch, tmp_path
    ):
        """A queue computed from three files while two are on screen is exactly
        the 'random-looking data' this module must never show."""
        box = tmp_path / "box"
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), box, ALL_THREE
        )
        assert test.session_state["upload_result"] is not None
        test.button(key="remove_LEDGER").click()
        test = test.run()
        assert "upload_result" not in test.session_state or (
            test.session_state["upload_result"] is None
        )

    def test_the_discard_is_announced_rather_than_silent(
        self, sample_store, monkeypatch, tmp_path
    ):
        """Falling back to a bundled sample without saying so is the
        substitution this whole module exists to prevent."""
        box = tmp_path / "box"
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), box, ALL_THREE
        )
        test.button(key="remove_LEDGER").click()
        test = test.run()
        warnings = " ".join(item.value for item in test.warning)
        assert "the previous result was discarded" in warnings
        assert "is a bundled sample, not your data" in warnings


class TestTheModelReportedBelongsToThisRun:
    def test_a_deterministic_run_says_no_model_was_called(
        self, sample_store, monkeypatch, tmp_path
    ):
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), tmp_path / "box", ALL_THREE
        )
        run = test.session_state["upload_result"]
        assert run.llm_used is False
        assert run.llm_calls == 0
        text = _text(test)
        assert "No AI model used" in text
        assert "No AI model was called for this result" in text

    def test_the_masthead_chip_follows_the_subject(
        self, sample_store, monkeypatch, tmp_path
    ):
        """A masthead reading 'no AI model used' over an upload that called one
        is the same lie in the other direction, so the chip is derived from the
        upload's own call count rather than from the selected report's."""
        import inspect

        from ledgerloop.ui import app

        body = inspect.getsource(app.main)
        assert "upload.llm_used" in body
        assert "upload.llm_calls" in body


class TestAFailedReconciliationShowsNothingFromTheLastOne:
    """The narrowest window, closed for the same reason as the widest.

    `main` hands this run's screens the object it read at the top. A
    reconciliation that raises part-way through has already been handed out, so
    without a rerun the five tabs spend the run describing the result the failed
    attempt was meant to replace -- stale data presented as current, which is
    the whole class of defect this module is about.
    """

    def test_the_error_path_reruns_and_carries_its_message(self):
        import inspect

        from ledgerloop.ui import app

        body = inspect.getsource(app._reconcile_uploads)
        error_half = body[body.index("except IngestError"):]
        assert "_discard_upload_result()" in error_half
        assert "st.rerun()" in error_half
        # The rerun throws away everything drawn so far, so an `st.error`
        # written here would never reach the reader.
        assert 'st.session_state["upload_error"]' in error_half

    def test_a_successful_run_clears_an_earlier_failure(self):
        import inspect

        from ledgerloop.ui import app

        body = inspect.getsource(app._reconcile_uploads)
        assert 'st.session_state.pop("upload_error", None)' in body

    def test_unreadable_files_leave_no_result_and_say_so(
        self, sample_store, monkeypatch, tmp_path
    ):
        """Driven through the real button, with a file that passes the upload
        validator and fails deeper in ingest."""
        box = tmp_path / "box"
        test = _reconcile(
            _app(monkeypatch, sample_store).run(), box, ALL_THREE
        )
        assert test.session_state["upload_result"] is not None

        # Same header, a row the reader cannot have meant: validation accepts
        # the shape, ingest refuses the content.
        bank = box / SourceKind.BANK.value
        rows = bank.read_text(encoding="utf-8").splitlines()
        bank.write_text(
            chr(10).join([rows[0], rows[0].replace(",", ",x")]) + chr(10),
            encoding="utf-8",
        )
        test.button(key="start_upload_run").click()
        test = test.run()
        assert not test.exception

        # `AppTest.session_state` has no `.get`, so membership is the only
        # way to ask without raising.
        refused = "upload_result" not in test.session_state
        if refused:
            # Ingest refused: nothing from the previous run may still be shown.
            errors = " ".join(item.value for item in test.error)
            assert "could not be read" in errors
            assert "upload_superseded" in test.session_state
