"""A widget may change the view. It may never change the subject.

THE DEFECT THESE PIN
--------------------
``st.file_uploader`` hands its file back on **every** rerun, not only the one it
arrived on. ``_upload_card`` treated every one of those as a fresh arrival and
called ``_accept``, which ends in ``_discard_upload_result()`` -- so the reader's
own reconciliation was deleted from session state on the very run that had just
finished drawing it. The run itself looked right, because ``main`` had already
read the result at the top. The *next* interaction did not: moving **Needs
review -> Show** from *All* to *Critical* reran the script, found no upload, and
quietly redrew all five tabs from a bundled sample report.

That is the worst shape a bug can take here. Nothing errored, nothing was
blank, and the numbers were internally consistent -- they were just about
somebody else's corpus. A person changing a severity filter has no reason to
re-read a record count.

WHY THESE TESTS DRIVE THE WIDGETS
---------------------------------
``test_ui_run_lineage`` seeds ``session_state["uploads"]`` directly, which is
why it stayed green throughout: seeding the dict never touches the uploader, so
the accept path it exercised was not the accept path that runs in a browser.
Every test below goes through ``file_uploader.set_value`` and then through the
real button, and the assertion is always made **after a further rerun** -- the
rerun is the whole event under test.

The second half is the same invariant for bundled reports, whose picker was
unkeyed: Streamlit derives an unkeyed widget's identity from its own arguments,
so the current report was a function of what happened to be on disk rather than
of what the reader chose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.store import list_runs
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import SplitName
from ledgerloop.ui.plain import (
    attention_items,
    attention_items_from,
    report_labels,
    snapshot,
)
from ledgerloop.ui.uploads import SourceKind

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)

APP = str(Path(__file__).resolve().parents[2] / "src" / "ledgerloop" / "ui" / "app.py")

#: The committed 60-order fixture set. Real corpus, real ingest, real ladder --
#: nothing here invents a report object, because a fabricated one would not have
#: gone through the accept path that broke.
FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"

ALL_THREE = tuple(SourceKind)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Two bundled reports with different splits, so their labels differ.

    A `dev` run is labelled *Demo report* and a `test` run *Test report* by
    :func:`report_labels`. Two of them are needed because leaking requires
    somewhere to leak from, and because the picker's own persistence cannot be
    tested against a list of one.
    """
    root = tmp_path_factory.mktemp("isolation")
    data = root / "data"
    runs = root / "runs"
    demo = data / "dev-standard-42"
    other = data / "test-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), demo)
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), other)
    run_graph(demo, measure_calibration_quality=False, store=runs, run_id="demo-run")
    run_graph(other, measure_calibration_quality=False, store=runs, run_id="test-run")
    return data, runs


@pytest.fixture(scope="module")
def reports(store):
    """The two stored runs, keyed by id, as the app reads them."""
    _, runs = store
    return {run.run_id: run for run in list_runs(runs)}


def _app(monkeypatch, store):
    data, runs = store
    monkeypatch.setenv("LEDGERLOOP_RUNS_DIR", str(runs))
    monkeypatch.setenv("LEDGERLOOP_DATA_DIR", str(data))
    monkeypatch.setenv("LEDGERLOOP_CALIBRATION", str(runs / "no-such-bundle.json"))
    return AppTest.from_file(APP, default_timeout=300)


def _upload(test, kinds=ALL_THREE):
    """Drop real files into the real uploaders, the way a person does."""
    for kind in kinds:
        mime = "application/json" if kind is SourceKind.PROCESSOR else "text/csv"
        test.file_uploader(key=f"upload_{kind.name}").set_value(
            (kind.value, (FIXTURE / kind.value).read_bytes(), mime)
        )
    return test.run()


def _reconcile(test):
    test.button(key="start_upload_run").click()
    return test.run()


def _uploaded_app(monkeypatch, store):
    """An app holding one finished reconciliation of the reader's own files."""
    return _reconcile(_upload(_app(monkeypatch, store).run()))


def _text(test) -> str:
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


def _subject(test) -> str:
    """Which report the five tabs actually drew, read off the page itself.

    Not off session state. The screens announce their subject in words, and it
    is that announcement -- the thing a reader would have to notice -- that
    these tests hold to.
    """
    text = _text(test)
    own = "Your files —" in text
    sample = "Sample report —" in text
    assert own != sample, f"the tabs named {own=} {sample=}; exactly one is right"
    return "upload" if own else "sample"


def _held(test):
    """The one authoritative upload, or ``None``."""
    if "upload_result" not in test.session_state:
        return None
    return test.session_state["upload_result"]


# --------------------------------------------------------------------------
# The root cause, on its own
# --------------------------------------------------------------------------
class TestTheUploadSurvivesTheRunThatProducedIt:
    def test_a_bare_rerun_does_not_delete_the_result(self, store, monkeypatch):
        """The whole bug in three lines: reconcile, rerun, and the reader's
        result is gone before they have touched anything."""
        test = _uploaded_app(monkeypatch, store)
        assert _held(test) is not None
        test = test.run()
        assert _held(test) is not None, (
            "the uploader handed its file back on the rerun and the accept path "
            "discarded the reconciliation of those same bytes"
        )

    def test_a_bare_rerun_does_not_claim_the_files_changed(self, store, monkeypatch):
        """`upload_superseded` is how the tabs *say* a result went away. Setting
        it when nothing changed puts a false warning over correct data."""
        test = _uploaded_app(monkeypatch, store).run()
        assert "upload_superseded" not in test.session_state

    def test_the_same_bytes_are_not_a_new_file(self, store, monkeypatch):
        """The fix, stated as the property it rests on: re-offering identical
        bytes describes the identical file set, so the result computed from
        them is still true."""
        from ledgerloop.ui import app

        test = _uploaded_app(monkeypatch, store)
        held = test.session_state["uploads"][SourceKind.BANK]
        payload = (FIXTURE / SourceKind.BANK.value).read_bytes()
        import hashlib

        assert held["digest"] == hashlib.sha256(payload).hexdigest()
        assert "_discard_upload_result()" in __import__("inspect").getsource(
            app._accept
        )


# --------------------------------------------------------------------------
# TEST A-E: a filter, a search and a tab may not change the data source
# --------------------------------------------------------------------------
class TestFilteringAnUploadFiltersOnlyTheView:
    def test_the_severity_filter_keeps_the_upload(self, store, monkeypatch):
        """TEST A. The reported scenario, exactly: Needs review -> Show -> a
        severity, and the subject must not move."""
        test = _uploaded_app(monkeypatch, store)
        before = _held(test)
        assert before is not None
        assert _subject(test) == "upload"

        picker = test.selectbox(key="up_attn_sev")
        severities = [option for option in picker.options if option != "All"]
        assert severities, "the fixture must queue something to filter"
        for severity in severities:
            test.selectbox(key="up_attn_sev").set_value(severity)
            test = test.run()
            assert _held(test) is before, f"the subject moved on {severity!r}"
            assert _subject(test) == "upload"
        test.selectbox(key="up_attn_sev").set_value("All")
        test = test.run()
        assert _held(test) is before
        assert _subject(test) == "upload"

    def test_the_filtered_items_come_from_the_upload(self, store, monkeypatch):
        """Filtering narrows this run's queue. It does not fetch another one."""
        test = _uploaded_app(monkeypatch, store)
        run = _held(test)
        assert run is not None
        # Through the shaping function the screen itself uses, so this asserts
        # provenance rather than re-deriving a second wording of it.
        severities = {item.severity for item in attention_items_from(run.exceptions)}
        options = set(test.selectbox(key="up_attn_sev").options)
        assert options == {"All", *severities}

    def test_the_status_filter_keeps_the_upload(self, store, monkeypatch):
        """TEST B."""
        test = _uploaded_app(monkeypatch, store)
        before = _held(test)
        for status in test.selectbox(key="up_tx_status").options:
            test.selectbox(key="up_tx_status").set_value(status)
            test = test.run()
            assert _held(test) is before
            assert _subject(test) == "upload"

    def test_every_tab_is_drawn_from_the_one_result(self, store, monkeypatch):
        """TEST C, stated the way Streamlit actually works.

        Tab clicks are client-side and rerun nothing, so "switching tabs" cannot
        be driven here -- and that is not a gap, because all five tab bodies are
        executed on every script run. The guarantee worth holding is therefore
        the stronger one: on one run, all five said the same subject.
        """
        test = _uploaded_app(monkeypatch, store).run()
        captions = [item.value for item in test.caption]
        own = [line for line in captions if "Your files —" in line]
        assert len(own) == 5, f"expected five tabs to name the upload, got {own}"
        assert not [line for line in captions if "Sample report —" in line]

    def test_searching_keeps_the_upload(self, store, monkeypatch):
        """TEST D."""
        test = _uploaded_app(monkeypatch, store)
        before = _held(test)
        run = _held(test)
        assert run is not None
        needle = run.matched.decisions[0].source_ref.key
        test.text_input(key="up_tx_search").set_value(needle)
        test = test.run()
        assert _held(test) is before
        assert _subject(test) == "upload"
        test.text_input(key="up_tx_search").set_value("")
        test = test.run()
        assert _held(test) is before
        assert _subject(test) == "upload"

    def test_the_whole_sequence_never_shows_a_sample(self, store, monkeypatch):
        """TEST E. The acceptance criterion, walked end to end."""
        test = _uploaded_app(monkeypatch, store)
        before = _held(test)
        assert before is not None
        record = test.selectbox(key="up_why_record").options[-1]
        steps = [
            ("selectbox", "up_attn_sev", "Critical"),
            ("selectbox", "up_attn_sev", "All"),
            ("selectbox", "up_tx_status", test.selectbox(key="up_tx_status").options[-1]),
            ("text_input", "up_tx_search", "PAY"),
            ("text_input", "up_tx_search", ""),
            ("selectbox", "up_why_record", record),
        ]
        for widget, key, value in steps:
            options = getattr(test, widget)(key=key)
            if widget == "selectbox" and value not in options.options:
                continue
            options.set_value(value)
            test = test.run()
            assert not test.exception, f"{key}={value!r} raised"
            assert _held(test) is before, f"{key}={value!r} changed the subject"
            assert _subject(test) == "upload", f"{key}={value!r} fell back to a sample"


class TestASampleStaysASample:
    def test_choosing_a_sample_and_filtering_keeps_the_sample(
        self, store, monkeypatch
    ):
        """TEST F. The mirror image, and it matters just as much: a reader who
        deliberately switched to a bundled report must not be jumped back to
        their own files by a dropdown either."""
        test = _uploaded_app(monkeypatch, store)
        test.sidebar.radio(key="scope").set_value("A sample report")
        test = test.run()
        assert _subject(test) == "sample"
        chosen = test.session_state["report_id"]
        for severity in test.selectbox(key="report_attn_sev").options:
            test.selectbox(key="report_attn_sev").set_value(severity)
            test = test.run()
            assert _subject(test) == "sample"
            assert test.session_state["report_id"] == chosen
        # And the upload is still there, untouched, to switch back to.
        assert _held(test) is not None


# --------------------------------------------------------------------------
# TEST G-H: the bundled report picker is state, not a widget default
# --------------------------------------------------------------------------
class TestTheChosenReportSurvivesEveryWidget:
    def test_the_picker_is_keyed_by_run_id(self, store, monkeypatch, reports):
        """Identity, not position. A label is renumbered when a run is added;
        a run id is not."""
        test = _app(monkeypatch, store).run()
        assert set(test.sidebar.radio(key="report_id").options) == set(
            report_labels(list(reports.values())).values()
        )
        assert test.session_state["report_id"] in reports

    def test_the_severity_filter_never_moves_the_report(
        self, store, monkeypatch, reports
    ):
        """TEST H, on the screen the report was made against: All -> Critical ->
        High -> Medium -> All, and the run id is the same at every step."""
        test = _app(monkeypatch, store).run()
        test.sidebar.radio(key="report_id").set_value("test-run")
        test = test.run()
        assert test.session_state["report_id"] == "test-run"
        wanted = ["Critical", "High", "Medium", "All"]
        available = test.selectbox(key="report_attn_sev").options
        for severity in [name for name in wanted if name in available]:
            test.selectbox(key="report_attn_sev").set_value(severity)
            test = test.run()
            assert test.session_state["report_id"] == "test-run"
            assert "Run id `test-run`" in _text(test)

    def test_two_reports_do_not_leak_into_each_other(
        self, store, monkeypatch, reports
    ):
        """TEST G. Pick A, filter it, pick B, filter it, and go back to A."""
        test = _app(monkeypatch, store).run()
        seen = {}
        for run_id in ("test-run", "demo-run", "test-run"):
            test.sidebar.radio(key="report_id").set_value(run_id)
            test = test.run()
            assert test.session_state["report_id"] == run_id
            test.selectbox(key="report_attn_sev").set_value(
                test.selectbox(key="report_attn_sev").options[-1]
            )
            test = test.run()
            assert test.session_state["report_id"] == run_id
            # Every screen's headline count is this report's own, read the same
            # way the report writer read it.
            view = snapshot(reports[run_id])
            checked = view.checked if view.checked is not None else view.records
            text = _text(test)
            assert f"{checked:,} transactions checked" in text
            assert f"{len(attention_items(reports[run_id])):,} item(s) need" in text
            seen[run_id] = checked
        assert seen["test-run"] != seen["demo-run"], (
            "the two fixtures must differ or this test proves nothing"
        )

    def test_searching_and_the_record_picker_never_move_the_report(
        self, store, monkeypatch
    ):
        """The remaining reader-facing controls, held to the same rule."""
        test = _app(monkeypatch, store).run()
        test.sidebar.radio(key="report_id").set_value("test-run")
        test = test.run()
        test.text_input(key="report_tx_search").set_value("PAY")
        test = test.run()
        assert test.session_state["report_id"] == "test-run"
        test.selectbox(key="report_why_record").set_value(
            test.selectbox(key="report_why_record").options[-1]
        )
        test = test.run()
        assert test.session_state["report_id"] == "test-run"
        test.selectbox(key="audit_record").set_value(
            test.selectbox(key="audit_record").options[-1]
        )
        test = test.run()
        assert test.session_state["report_id"] == "test-run"
        assert not test.exception

    def test_a_report_that_vanished_falls_to_a_report_and_never_to_nothing(
        self, store, monkeypatch
    ):
        """The one implicit change of subject in the file, and its bounds.

        A stale id cannot stay selected -- the run is not on disk. Dropping it
        must land on another *report*, not on an empty page and not on some
        privileged sample.
        """
        test = _app(monkeypatch, store)
        test.session_state["report_id"] = "a-run-that-was-deleted"
        test = test.run()
        assert not test.exception
        assert test.session_state["report_id"] in {"test-run", "demo-run"}


class TestNoWidgetReRunsTheReconciliation:
    def test_reconciliation_happens_in_one_place_only(self):
        """A filter that recomputes is a filter that can compute something
        different. The pipeline is entered from two call sites and both are
        behind a button."""
        source = Path(APP).read_text(encoding="utf-8")
        assert source.count("reconcile_only(") == 1
        assert source.count("run_graph(") == 1

    def test_the_filter_run_does_not_touch_the_ingest_layer(
        self, store, monkeypatch
    ):
        """Stated as an identity check rather than a timing one: the object the
        page filters is the object the button produced, not an equal copy that
        some later rerun happened to rebuild."""
        test = _uploaded_app(monkeypatch, store)
        first = _held(test)
        assert first is not None
        test.selectbox(key="up_attn_sev").set_value("All")
        test = test.run()
        assert _held(test) is first
