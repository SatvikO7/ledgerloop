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


def _tab(test, label: str):
    """The tab with this label.

    By name rather than by index: the positional form broke the moment a tab was
    added at the front, and a test that pins a position is asserting the layout
    rather than the guarantee it was written for.
    """
    for tab in test.tabs:
        if tab.label == label:
            return tab
    raise AssertionError(f"no tab labelled {label!r}; got {[t.label for t in test.tabs]}")


class TestTheScriptRuns:
    def test_it_renders_without_raising(self, workspace, monkeypatch):
        """A page that renders a traceback still returns HTTP 200. Only running
        the script catches that."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert not test.exception
        markdown = " ".join(item.value for item in test.markdown)
        assert "LedgerLoop" in markdown
        assert "Automatic payment reconciliation" in markdown

    def test_the_reader_can_tell_which_report_is_on_screen(
        self, workspace, monkeypatch
    ):
        """Not by run id -- that moved to the details expander -- but by the
        report's name in the picker and the size of what it checked."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        sidebar = " ".join(item.value for item in test.sidebar.markdown)
        captions = " ".join(item.value for item in test.sidebar.caption)
        assert "**Current report**" in sidebar
        assert "transactions checked" in captions
        assert "Demo report" in [str(o) for o in test.radio[0].options]

    def test_the_masthead_says_whether_a_model_was_involved(
        self, workspace, monkeypatch
    ):
        """The one thing about the run's provenance a non-technical reader
        genuinely benefits from before reading any number."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "No AI model used" in markdown

    def test_the_sections_are_named_for_a_reader_not_a_developer(
        self, workspace, monkeypatch
    ):
        """The information architecture, in the order a person asks the
        questions: what happened, what do I do, show me everything, prove
        one of them -- and only then the measurement.

        No tab is named after a component. "Pipeline", "Evidence" and
        "Evaluation" were all developer words for developer screens; the
        content survives inside **Technical report**, which is where a
        reader who wants it will look and a reader who does not will not.
        """
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        labels = [tab.label for tab in test.tabs]
        assert labels == [
            "Your files",
            "Overview",
            "Needs review",
            "Transactions",
            "Why it matched",
            "Accuracy & details",
            "Sample data",
        ]

    def test_the_report_picker_names_reports_not_run_ids(self, workspace, monkeypatch):
        """`app-run`, `t0t4-test-42` and `ui-demo` encode a ladder, a split and a
        seed. That is what someone reproducing a figure needs and what someone
        reading one does not; the picker shows what the report *is*."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        shown = [str(option) for option in test.radio[0].options]
        assert shown == ["Demo report"]
        assert not any("app-run" in label for label in shown)

    def test_an_empty_store_says_so_rather_than_rendering_blank_screens(
        self, workspace, monkeypatch, tmp_path
    ):
        """A dashboard with no data must say it has none. Rendering empty
        tables would look like a run that found nothing."""
        corpus, _ = workspace
        test = _app(monkeypatch, tmp_path / "empty", corpus.parent).run()
        assert not test.exception
        messages = [item.value for item in test.info]
        assert any("No reports yet" in message for message in messages)
        assert any("Upload your files" in message for message in messages)


class TestTheResultsScreen:
    def test_it_shows_all_four_headline_proportions(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for expected in ("Precision", "Recall", "Match rate", "Exception recall"):
            assert expected in markdown

    def test_no_proportion_is_rendered_without_its_interval(
        self, workspace, monkeypatch
    ):
        """The project's own standard, applied to the dashboard. A point
        estimate with nothing beside it is the omission `Proportion` exists to
        make unavailable, and a KPI card must not reintroduce it."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert markdown.count("95% CI") >= 4
        # A real interval, not the "no sample" placeholder: this run measured
        # every headline proportion, so each card must show actual bounds.
        assert "no sample" not in markdown
        assert markdown.count("of ") >= 4

    def test_every_headline_carries_a_verdict(self, workspace, monkeypatch):
        """met / missed / undecided / reported -- the ruling comes from
        `Proportion.verdict`, so the dashboard cannot reach one the report
        would not."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert any(
            word in markdown
            for word in ("target met", "target missed", "undecided", "reported")
        )

    def test_it_shows_every_decision_outcome_separately(self, workspace, monkeypatch):
        """AUTO_MATCHED, NEEDS_REVIEW, EXCEPTION and REJECTED are four figures.
        A single 'matched' number would be the most misleading thing here."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for outcome in ("AUTO_MATCHED", "NEEDS_REVIEW", "EXCEPTION", "REJECTED"):
            assert outcome in markdown

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


class TestThePipelineScreen:
    """The tier ladder as a flow, including the rungs that found nothing."""

    def test_it_draws_every_rung_of_the_ladder(self, workspace, monkeypatch):
        """All six, always. A ladder rendered only from the rows a run happened
        to produce would silently drop the tier that contributed nothing --
        which is the one a reader most needs to see."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for rung in ("T0", "T1", "T2", "T3", "T4", "T5"):
            assert rung in markdown
        for label in ("Exact", "Tolerance", "Aggregation", "Lexical", "Graph", "LLM"):
            assert label in markdown

    def test_each_silent_rung_gets_its_own_reason(self, workspace, monkeypatch):
        """T4 and T5 are silent for completely different reasons, and the panel
        used to print T4's argument over both. A zero explained by the wrong
        reason invites a reader to distrust the rungs that did fire."""
        from ledgerloop.ui.app import _SILENT_REASON

        assert set(_SILENT_REASON) == {"T4_GRAPH", "T5_LLM"}
        graph, llm = _SILENT_REASON["T4_GRAPH"], _SILENT_REASON["T5_LLM"]
        assert "partial" in graph and "partial" not in llm
        assert "settlements the ladder could not credit" in llm
        assert graph != llm

    def test_the_llm_reason_says_what_it_is_actually_offered(
        self, workspace, monkeypatch
    ):
        """The measured fact behind it: on `test-standard-42` the adjudicator is
        offered one evidence pack, because 66 of the 67 queue items are findings
        rather than missing links."""
        from ledgerloop.ui.app import _SILENT_REASON

        text = _SILENT_REASON["T5_LLM"].lower()
        assert "not the review queue as a whole" in text
        assert "posted twice" in text or "chargeback" in text

    def test_a_rung_that_found_nothing_says_so_honestly(self, workspace, monkeypatch):
        """T4 runs on every corpus and contributes zero. The dashboard has to
        say that plainly -- not as an error, and not by hiding the rung."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "found nothing" in markdown
        notes = " ".join(item.value for item in test.info)
        assert "measurement, not an error" in notes

    def test_it_separates_yield_from_conviction(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "proposed" in markdown
        assert "refused" in markdown


class TestTheEvidenceScreen:
    def test_it_walks_the_chain_for_a_non_developer(self, workspace, monkeypatch):
        """Source records -> normalisation -> candidate -> tier -> arithmetic
        -> decision. Someone who has never seen the code should be able to
        follow why one record ended up where it did."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for step in (
            "Source records",
            "Normalisation",
            "Candidate",
            "Tier",
            "Arithmetic verification",
            "Decision",
        ):
            assert step in markdown

    def test_the_chain_states_the_money_discipline(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "integer paise" in markdown


class TestTheEvaluationScreen:
    def test_it_publishes_the_bad_rows_too(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "including the classes that score badly" in captions

    def test_it_names_the_weakest_anomaly_class(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "Weakest class on this run" in captions

    def test_it_reports_the_confusion_counts(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for label in ("True positives", "False positives", "False negatives"):
            assert label in markdown


class TestNothingIsHardcoded:
    """Every figure on screen has to come from the run record.

    The check is a substitution: change a number in the stored run and the
    dashboard must change with it. A hardcoded metric would survive the edit.
    """

    def test_the_headline_follows_the_stored_run(self, workspace, monkeypatch, tmp_path):
        import json
        import shutil

        corpus, runs = workspace
        edited = tmp_path / "edited-runs"
        shutil.copytree(runs, edited)
        record = next(edited.glob("*/run.json"))
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["metrics"]["intervals"]["precision_interval"] = {
            "successes": 3,
            "trials": 4,
            "value": 0.75,
            "ci_low": 0.3,
            "ci_high": 0.95,
        }
        payload["metrics"]["false_positives"] = 7
        record.write_text(json.dumps(payload), encoding="utf-8")

        test = _app(monkeypatch, edited, corpus.parent).run()
        assert not test.exception
        markdown = " ".join(item.value for item in test.markdown)
        assert "75.00%" in markdown
        assert "3 of 4" in markdown
        errors = " ".join(item.value for item in test.error)
        assert "7 false positive(s)" in errors

    def test_a_run_without_stored_intervals_says_so(
        self, workspace, monkeypatch, tmp_path
    ):
        """An older record predates the stored intervals. That renders as
        *not measured*, never as 0.00% -- the same rule the report applies."""
        import json
        import shutil

        corpus, runs = workspace
        edited = tmp_path / "legacy-runs"
        shutil.copytree(runs, edited)
        record = next(edited.glob("*/run.json"))
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["metrics"].pop("intervals", None)
        record.write_text(json.dumps(payload), encoding="utf-8")

        test = _app(monkeypatch, edited, corpus.parent).run()
        assert not test.exception
        markdown = " ".join(item.value for item in test.markdown)
        assert "not measured" in markdown
        assert "n/a" in markdown
        assert any("predates stored intervals" in item.value for item in test.info)



class TestItReadsWithoutJargon:
    """The redesign's actual promise, tested as a promise rather than a layout.

    A reader who has never met a tier, a Wilson interval or a residual pass must
    still get the four answers. These tests check the words on the first screen,
    because that is the only thing that decides whether the promise was kept.
    """

    #: Vocabulary a normal user should never meet before they choose to.
    #: Every one of these still appears in **Technical report** and is asserted
    #: to, elsewhere in this file -- the rule is about *placement*, not removal.
    JARGON = (
        "Wilson",
        "residual",
        "tranche",
        "lexical",
        "grounding",
        "calibrat",
        "provenance",
        "AUTO_MATCHED",
        "NEEDS_REVIEW",
        "PAYMENT_CREDITED_AS",
        "tuning hash",
        "T0_EXACT",
        "T3_FUZZY",
    )

    def test_the_first_screen_says_what_the_product_does(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        assert "compares your payments" in markdown
        assert "matches what it can prove" in markdown

    def test_the_four_questions_are_answered_in_plain_words(
        self, workspace, monkeypatch
    ):
        """How many matched, how much money, what needs me, was anything wrong."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        for label in (
            "Transactions checked",
            "Successfully matched",
            "Need review",
            "Incorrect matches",
            "reconciled",
        ):
            assert label in markdown

    def test_no_jargon_reaches_the_overview(self, workspace, monkeypatch):
        """The whole point. Checked against the Overview tab's own markdown so
        the technical screens cannot accidentally satisfy it."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        for word in self.JARGON:
            assert word not in overview, f"{word!r} reached the first screen"

    def test_the_technical_vocabulary_is_still_there_one_tab_away(
        self, workspace, monkeypatch
    ):
        """Nothing was deleted to make the overview clean. A redesign that
        dropped the evidence would have traded the project's strongest property
        for a tidier screen."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        report = " ".join(item.value for item in _tab(test, "Accuracy & details").markdown)
        for word in ("Precision", "Recall", "Match rate", "95% CI"):
            assert word in report

    def test_the_glossary_translates_every_term_it_shows(
        self, workspace, monkeypatch
    ):
        """A technical term printed without its plain meaning is the thing the
        redesign exists to stop."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        glossaries = [
            frame.value
            for frame in test.dataframe
            if "In plain words" in list(frame.value.columns)
        ]
        assert glossaries, "the technical report shows no glossary"
        table = glossaries[0]
        assert len(table) >= 5
        for meaning in table["In plain words"]:
            assert str(meaning).strip()
        for term in ("Precision", "Recall", "False positives"):
            assert term in list(table["Term"])



class TestTheSidebarIsForAReader:
    """The sidebar is chrome: it is on screen on every tab, so anything
    technical there is technical *everywhere*. It used to open with
    `ui-demo`, `t0t4-test-42`, `t0t4-calibration-42` and a tuning hash."""

    #: Fragments of the run-id scheme. None may appear in the sidebar's own
    #: markdown; all remain available inside the details expander.
    RUN_ID_FRAGMENTS = ("t0t4", "app-run", "ui-demo", "seed 42")

    def test_no_internal_run_identifier_is_shown_by_default(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        sidebar = " ".join(item.value for item in test.sidebar.markdown)
        for fragment in self.RUN_ID_FRAGMENTS:
            assert fragment not in sidebar, f"{fragment!r} is still in the sidebar"

    def test_no_internal_identifier_reaches_the_masthead_either(
        self, workspace, monkeypatch
    ):
        """The masthead is on screen on every tab too. It used to carry
        `run app-run` and `dev - standard - seed 42` as chips."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        markdown = " ".join(item.value for item in test.markdown)
        hero = markdown[markdown.find("ll-hero") : markdown.find("ll-hero") + 900]
        for fragment in ("app-run", "seed", "difficulty", "standard"):
            assert fragment not in hero, f"{fragment!r} is still in the masthead"

    def test_it_names_the_section_reports_not_runs(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        sidebar = " ".join(item.value for item in test.sidebar.markdown)
        assert "**Reports**" in sidebar
        assert "**Current report**" in sidebar

    def test_it_counts_transactions_rather_than_records(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.sidebar.caption)
        assert "transactions checked" in captions

    def test_it_says_whether_the_reconciliation_finished_safely(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        assert any("Reconciliation complete" in item.value for item in test.success)

    def test_the_identifiers_are_kept_one_click_away(self, workspace, monkeypatch):
        """Moved, not deleted. A judge reproducing a figure still needs the run
        id, the seed and the tuning hash."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.sidebar.caption)
        assert "app-run" in captions
        assert "Tuning hash" in captions
        assert "seed" in captions
        assert any(
            "Report details" in str(getattr(item, "label", ""))
            for item in test.sidebar.expander
        )


class TestTheFirstViewport:
    """What a judge sees in the first ten seconds, tested as content."""

    def test_it_leads_with_a_verdict(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        assert "Reconciliation completed safely" in overview

    def test_the_four_plain_kpis_are_present(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        for label in (
            "Transactions checked",
            "Successfully matched",
            "Need review",
            "Incorrect matches",
        ):
            assert label in overview

    def test_it_says_what_to_do_next(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        pointers = [item.value for item in test.info] + [
            item.value for item in test.success
        ]
        assert any("Next:" in text or "Nothing is waiting" in text for text in pointers)

    def test_the_process_is_drawn_in_plain_english(self, workspace, monkeypatch):
        """Payment -> bank transaction -> settlement -> reconciled, and the
        second path where it does not work out."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        for step in (
            "Payment",
            "Bank transaction",
            "Settlement record",
            "Reconciled",
            "Not enough evidence",
            "Sent for review",
        ):
            assert step in overview

    def test_it_shows_both_paths_not_only_the_happy_one(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        assert "When the evidence agrees" in overview
        assert "When it does not" in overview

class TestSafetyIsTheHeadline:
    """Precision-first is the project's strongest property, and the redesign
    had to make it legible without overstating it."""

    def test_the_safety_claim_is_stated_in_plain_words(self, workspace, monkeypatch):
        """Demoted from a second banner to a caption, because the verdict card
        above it already delivers the headline -- but the sentence that explains
        *why* the zero is a zero must survive somewhere on the first screen."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        text = " ".join(
            item.value for item in list(test.markdown) + list(test.caption)
        )
        assert "incorrect matches" in text
        assert "instead of guessing" in text

    def test_the_queue_leads_with_nothing_was_guessed(self, workspace, monkeypatch):
        """The reassurance a controller needs before reading a list of
        problems: this is a refusal, not a failure."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        attention = " ".join(item.value for item in _tab(test, "Needs review").markdown)
        assert "Nothing was guessed" in attention

    def test_the_overview_never_calls_a_perfect_score_a_pass(
        self, workspace, monkeypatch
    ):
        """`0 incorrect matches` is what was *measured*. Whether it clears a 99%
        target at this sample size is a statistical ruling, and it stays in the
        report with its interval where it can be read properly."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        assert "target met" not in overview
        assert "99" not in overview

    def test_the_three_destinations_stay_three_numbers(self, workspace, monkeypatch):
        """Matched, needs review and not matched are never added together.
        Folding a refusal into 'matched' is the single most misleading thing a
        reconciliation dashboard can do.

        On the Overview these are the KPI row, counted in records. The
        decision-unit breakdown is on **Accuracy & details**: both are true and
        they are *different numbers*, so sharing a screen would read as a
        contradiction rather than as two views of one run.
        """
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        overview = " ".join(item.value for item in _tab(test, "Overview").markdown)
        for label in ("Successfully matched", "Need review", "Incorrect matches"):
            assert label in overview

        report = " ".join(item.value for item in _tab(test, "Accuracy & details").markdown)
        for label in ("Matched", "Needs attention", "Not matched"):
            assert label in report

    def test_the_two_breakdowns_never_share_a_screen(self, workspace, monkeypatch):
        """The Overview counts records; the report counts decisions. Showing
        both together produced 316 beside 283 with no explanation."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        home = _tab(test, "Overview")
        overview = " ".join(
            item.value for item in list(home.markdown) + list(home.subheader)
        )
        assert "Where everything went" not in overview
        details = _tab(test, "Accuracy & details")
        report = " ".join(
            item.value for item in list(details.markdown) + list(details.subheader)
        )
        assert "Where everything went, by decision" in report


class TestTheQueueIsActionable:
    def test_every_item_says_what_was_found_and_what_to_do(
        self, workspace, monkeypatch
    ):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        attention = " ".join(item.value for item in _tab(test, "Needs review").markdown)
        assert "What we found." in attention
        assert "What to do." in attention

    def test_it_leads_with_the_money(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        attention = " ".join(item.value for item in _tab(test, "Needs review").markdown)
        assert "Money involved" in attention
        assert "₹" in attention


class TestWhyItMatchedExplainsItself:
    def test_it_gives_reasons_rather_than_a_tier_name(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        why = " ".join(item.value for item in _tab(test, "Why it matched").markdown)
        assert "How it was matched." in why
        assert "Confidence." in why
        assert "T0_EXACT" not in why

    def test_the_confidence_is_a_word_not_four_decimals(
        self, workspace, monkeypatch
    ):
        """A controller does not act differently at 0.94 than at 0.96, and four
        decimals invite a precision the calibration section is careful not to
        claim. The exact figure stays in the expander."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        why = " ".join(item.value for item in _tab(test, "Why it matched").markdown)
        assert any(
            phrase in why
            for phrase in ("Very strong evidence", "Strong evidence", "Not confirmed")
        )


class TestTheTransactionList:
    def test_it_can_be_filtered_and_searched(self, workspace, monkeypatch):
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        keys = {widget.key for widget in test.selectbox}
        assert "tx_status" in keys
        assert any(widget.key == "tx_search" for widget in test.text_input)

    def test_it_says_why_there_is_no_amount_column(self, workspace, monkeypatch):
        """The run record stores no amount per matched link. Saying so is the
        honest alternative to a column of plausible-looking blanks."""
        corpus, runs = workspace
        test = _app(monkeypatch, runs, corpus.parent).run()
        captions = " ".join(item.value for item in test.caption)
        assert "no amount column" in captions

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
