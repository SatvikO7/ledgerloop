"""The plain-language layer, tested as a translation rather than a rewrite.

``ui/plain.py`` exists to say the same thing ``ui/views.py`` says, in words a
finance person uses. So the tests that matter are the ones that would catch it
*drifting*: a number that stopped agreeing with the run record, a refusal
quietly folded into a match, or a claim made stronger than what was measured.
"""

from __future__ import annotations

import json
import shutil

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.runner import run_graph
from ledgerloop.agent.store import list_runs
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import SplitName, Tier
from ledgerloop.ui import plain
from ledgerloop.ui.views import headline

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    root = tmp_path_factory.mktemp("plain")
    corpus = root / "dev-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), corpus)
    runs = root / "runs"
    run_graph(corpus, measure_calibration_quality=False, store=runs, run_id="plain-run")
    return list_runs(runs)[0]


class TestItTranslatesRatherThanRecomputes:
    def test_every_headline_number_agrees_with_the_run(self, stored):
        """The whole risk of a second vocabulary is that it becomes a second set
        of numbers. Each one is checked against ``views.headline``."""
        view = headline(stored)
        snap = plain.snapshot(stored)
        assert snap.matched == view.auto_matched
        assert snap.needs_attention == view.queue_size
        assert snap.not_matched == view.false_negatives
        assert snap.incorrect == view.false_positives
        assert snap.reconciled == view.reconciled
        assert snap.outstanding == view.outstanding
        assert snap.referred == view.needs_review

    def test_it_reports_what_was_committed_and_what_was_wrong_separately(self, stored):
        """`matched` counts committed links and `incorrect` counts the wrong
        ones among them. Showing a pre-cleaned number would hide the claim."""
        view = headline(stored)
        snap = plain.snapshot(stored)
        assert snap.matched == view.auto_matched
        assert snap.matched >= snap.incorrect

    def test_the_module_never_imports_streamlit(self):
        """The same rule ``views.py`` is held to: shaping must be testable
        without a browser, and a widget call here would make it not be."""
        source = (
            pytest.importorskip("pathlib").Path(plain.__file__).read_text(encoding="utf-8")
        )
        assert "import streamlit" not in source

    def test_it_never_imports_the_matcher_or_the_evaluator(self):
        """A translation layer that reached into the pipeline could disagree
        with the run record it is supposed to be reading."""
        source = (
            pytest.importorskip("pathlib").Path(plain.__file__).read_text(encoding="utf-8")
        )
        assert "ledgerloop.matching" not in source
        assert "ledgerloop.eval" not in source


class TestTheThreeDestinationsStayThree:
    def test_matched_review_and_unmatched_are_never_added_together(self, stored):
        labels = [bucket.label for bucket in plain.buckets(stored)]
        assert labels == ["Matched", "Needs attention", "Not matched"]

    def test_each_bucket_says_what_it_means(self, stored):
        for bucket in plain.buckets(stored):
            assert bucket.note.strip()
            assert bucket.tone in {"good", "warn", "muted"}

    def test_the_unmatched_bucket_says_nothing_was_guessed(self, stored):
        note = next(b.note for b in plain.buckets(stored) if b.label == "Not matched")
        assert "guessed" in note


class TestTheSafetyClaimIsNotOverstated:
    def test_a_clean_run_says_zero_incorrect_matches(self, stored):
        title, body = plain.safety_note(stored)
        assert plain.snapshot(stored).incorrect == 0
        assert title == "0 incorrect matches"
        assert "instead of guessing" in body

    def test_it_never_claims_a_target_was_met(self, stored):
        """Whether a perfect score clears a 99% target at this sample size is a
        statistical ruling. It belongs with its interval, in the report."""
        title, body = plain.safety_note(stored)
        text = f"{title} {body}".lower()
        for word in ("target", "99%", "certified", "guaranteed", "perfect"):
            assert word not in text

    def test_a_dirty_run_is_priced_rather_than_congratulated(self, stored, tmp_path):
        """The banner must be able to deliver bad news. A safety indicator that
        only ever says 'all clear' is not one.

        Written into a **copy** of the run: mutating the module-scoped fixture
        would make every test after this one depend on where it ran in the file.
        """
        copied = tmp_path / "runs" / stored.directory.name
        copied.mkdir(parents=True)
        for source in stored.directory.iterdir():
            shutil.copy2(source, copied / source.name)
        poisoned = json.loads((copied / "run.json").read_text(encoding="utf-8"))
        poisoned["metrics"]["false_positives"] = 3
        poisoned["metrics"]["false_positive_cost_minor"] = 123_400
        (copied / "run.json").write_text(json.dumps(poisoned), encoding="utf-8")

        reloaded = list_runs(copied.parent)[0]
        title, body = plain.safety_note(reloaded)
        assert title == "3 incorrect matches"
        assert "₹1,234.00" in body
        assert not plain.snapshot(reloaded).is_clean


class TestTheQueueIsActionable:
    def test_it_is_sorted_by_money_descending(self, stored):
        items = plain.attention_items(stored)
        amounts = [item.amount_minor for item in items]
        assert amounts == sorted(amounts, reverse=True)

    def test_every_item_carries_a_cause_and_an_action(self, stored):
        for item in plain.attention_items(stored):
            assert item.found.strip()
            assert item.action.strip()
            assert item.amount.startswith("₹")

    def test_the_wording_is_the_classifiers_own(self, stored):
        """Passed through untouched. Paraphrasing here would be this module
        inventing a claim the run did not make."""
        items = plain.attention_items(stored)
        by_id = {item.exception_id: item for item in stored.exceptions}
        for item in items:
            assert item.found == by_id[item.exception_id].root_cause
            assert item.action == by_id[item.exception_id].suggested_action


class TestConfidenceIsAWord:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (1.0, "Very strong evidence"),
            (0.95, "Strong evidence"),
            (0.75, "Reasonable evidence"),
            (0.2, "Weak evidence"),
        ],
    )
    def test_a_probability_becomes_a_phrase(self, probability, expected):
        assert plain.confidence_word(probability, verified=True) == expected

    def test_an_unverified_link_is_never_called_strong(self):
        """Whatever the model scored it, the arithmetic did not close."""
        assert plain.confidence_word(1.0, verified=False) == "Not confirmed"


class TestTheTransactionTable:
    def test_it_shows_only_what_the_run_actually_stores(self, stored):
        """No amount and no merchant column: the run record holds neither per
        decision, and a column of plausible blanks would be worse than none."""
        rows = plain.transaction_rows(stored)
        assert rows
        for row in rows:
            assert "Amount" not in row
            assert "Merchant" not in row
            assert row["Status"]
            assert row["How it was matched"]

    def test_a_filter_narrows_the_view_and_nothing_else(self, stored):
        everything = plain.transaction_rows(stored)
        matched = plain.transaction_rows(stored, status="Matched")
        assert len(matched) <= len(everything)
        assert all(row["Status"] == "Matched" for row in matched)

    def test_search_matches_an_identifier_a_person_would_paste(self, stored):
        rows = plain.transaction_rows(stored)
        needle = str(rows[0]["Payment"])
        found = plain.transaction_search(rows, needle)
        assert found
        assert all(needle in str(row["Payment"]) for row in found)

    def test_an_empty_search_changes_nothing(self, stored):
        rows = plain.transaction_rows(stored)
        assert plain.transaction_search(rows, "   ") == rows

    def test_every_tier_has_a_plain_name(self):
        """A tier without one would surface as `T3_FUZZY` on a user's screen."""
        for tier in Tier:
            stage = plain.stage_of(tier)
            assert stage.plain.strip()
            assert stage.because.strip()
            assert "T" + str(tier.value) not in stage.plain


class TestTheMatchStory:
    def test_a_committed_link_is_explained_without_jargon(self, stored):
        rows = plain.transaction_rows(stored, status="Matched")
        story = plain.match_story(stored, str(rows[0]["record_key"]))
        assert story.matched
        assert story.headline == "Match confirmed"
        assert story.reasons
        for reason in story.reasons:
            assert "tier" not in reason.lower()
            assert "calibrated" not in reason.lower()

    def test_the_jargon_is_present_in_the_technical_pairs(self, stored):
        """Moved, not deleted. The evidence has to stay checkable."""
        rows = plain.transaction_rows(stored, status="Matched")
        story = plain.match_story(stored, str(rows[0]["record_key"]))
        labels = dict(story.technical)
        assert "Tier" in labels
        assert "Calibrated probability" in labels
        assert "Arithmetic verified" in labels

    def test_the_partner_is_the_other_end_of_the_link(self, stored):
        """A record can be either side of a decision. Printing the target
        unconditionally rendered "BNK-00002 to BNK-00002" whenever a bank row
        was the one selected."""
        rows = plain.transaction_rows(stored, status="Matched")
        row = rows[0]
        forward = plain.match_story(stored, str(row["record_key"]))
        assert forward.partner == str(row["Bank transaction"])
        assert forward.partner not in str(row["record_key"])

        backward = plain.match_story(stored, f"bank_txn:{row['Bank transaction']}")
        assert backward.partner != str(row["Bank transaction"])

    def test_an_unknown_record_says_so_rather_than_inventing_an_outcome(self, stored):
        story = plain.match_story(stored, "payment:NOT-A-RECORD")
        assert not story.matched
        assert story.headline == "Not matched"
        assert "nothing was guessed" in " ".join(story.reasons).lower()

    def test_a_record_only_an_exception_names_still_explains_itself(self, stored):
        keys = {
            ref.key
            for item in stored.exceptions
            for ref in item.involved_refs
            if not stored.decisions_for(ref.key)
        }
        if not keys:
            pytest.skip("this corpus decided every record an exception names")
        story = plain.match_story(stored, sorted(keys)[0])
        assert not story.matched
        assert story.reasons[0].strip()


class TestTheGlossary:
    def test_every_term_has_a_plain_meaning_and_this_run_s_value(self, stored):
        entries = plain.glossary(headline(stored))
        assert len(entries) == len(plain.GLOSSARY)
        for entry in entries:
            assert entry.plain.strip()
            assert entry.value.strip()

    def test_it_covers_the_terms_a_reader_will_actually_meet(self, stored):
        terms = {entry.term for entry in plain.glossary(headline(stored))}
        for expected in ("Precision", "Recall", "False positives", "Wilson interval"):
            assert expected in terms


class TestTheReportLabels:
    def test_a_split_becomes_a_human_name(self, stored):
        labels = plain.report_labels([stored])
        assert labels[stored.run_id] == "Demo report"

    def test_no_label_leaks_a_run_id(self, stored):
        for label in plain.report_labels([stored]).values():
            assert stored.run_id not in label

    def test_repeats_are_numbered_so_the_picker_never_shows_one_name_twice(
        self, stored, tmp_path
    ):
        """Two runs of the same split would otherwise share a label, and the
        picker would show the same words twice."""
        second = tmp_path / "runs" / "another-demo"
        second.mkdir(parents=True)
        for source in stored.directory.iterdir():
            shutil.copy2(source, second / source.name)
        # The run id lives in run.json, not in the directory name, so a plain
        # copy would collide with the original rather than sit beside it.
        summary = json.loads((second / "run.json").read_text(encoding="utf-8"))
        summary["run_id"] = "another-demo"
        (second / "run.json").write_text(json.dumps(summary), encoding="utf-8")
        other = list_runs(second.parent)[0]
        assert other.run_id != stored.run_id

        labels = plain.report_labels([stored, other])
        assert len(labels) == 2
        assert sorted(labels.values()) == ["Demo report", "Demo report (2)"]

    def test_an_unknown_split_still_gets_a_readable_name(self, stored):
        """A split this table has never seen must not fall back to the run id."""
        assert "" not in plain.REPORT_NAMES
        assert plain.REPORT_NAMES["test"] == "Test report"


class TestTheStatusVerdict:
    def test_a_clean_run_reads_as_completed_safely(self, stored):
        status = plain.status_of(stored)
        assert status.title == "Reconciliation completed safely"
        assert status.tone == "good"

    def test_a_wrong_match_outranks_a_queue(self, stored, tmp_path):
        """A queue is work; a wrong match is a mistake already in the books.
        The verdict must lead with the second one."""
        copied = tmp_path / "runs" / stored.directory.name
        copied.mkdir(parents=True)
        for source in stored.directory.iterdir():
            shutil.copy2(source, copied / source.name)
        poisoned = json.loads((copied / "run.json").read_text(encoding="utf-8"))
        poisoned["metrics"]["false_positives"] = 2
        (copied / "run.json").write_text(json.dumps(poisoned), encoding="utf-8")

        status = plain.status_of(list_runs(copied.parent)[0])
        assert status.tone == "bad"
        assert "wrong" in status.title


class TestTheSameUnitTriple:
    def test_the_counts_add_up_when_the_run_stored_them(self, stored):
        """checked = matched + needs review, or the screen does not show them
        as a breakdown at all."""
        snap = plain.snapshot(stored)
        if snap.counts_add_up:
            assert snap.resolved is not None and snap.unresolved is not None
            assert snap.resolved + snap.unresolved == snap.checked
        else:
            assert snap.checked is None

    def test_they_come_from_the_run_s_own_match_rate_denominator(self, stored):
        interval = stored.metrics.get("intervals", {}).get("match_rate_interval")
        snap = plain.snapshot(stored)
        if interval:
            assert snap.checked == interval["trials"]
            assert snap.resolved == interval["successes"]

    def test_records_is_never_confused_with_checked(self, stored):
        """`records` counts everything read, including rows nothing could ever
        resolve. Quoting it beside `resolved` would invent a failure rate."""
        snap = plain.snapshot(stored)
        if snap.checked is not None:
            assert snap.checked <= snap.records


class TestTheScreenNeverContradictsItself:
    def test_the_verdict_quotes_the_same_figure_as_the_need_review_card(self, stored):
        """29 in the card and 67 in the sentence beside it read as a
        contradiction, even though both are true of different units."""
        snap = plain.snapshot(stored)
        status = plain.status_of(stored)
        if snap.unresolved is not None and snap.unresolved:
            assert f"{snap.unresolved:,}" in status.body
        elif snap.needs_attention:
            assert f"{snap.needs_attention:,}" in status.body

    def test_the_verdict_counts_transactions_not_queue_items(self, stored):
        snap = plain.snapshot(stored)
        if snap.unresolved is not None:
            assert "transaction(s)" in plain.status_of(stored).body


class TestTheJourney:
    def test_both_paths_are_drawn(self, stored):
        settled, stuck = plain.journey(stored)
        assert [step.label for step in settled] == [
            "Payment",
            "Bank transaction",
            "Settlement record",
            "Reconciled",
        ]
        assert stuck[-1].label == "Sent for review"

    def test_the_unsettled_path_says_nothing_was_guessed(self, stored):
        _, stuck = plain.journey(stored)
        assert any("guessed" in step.note for step in stuck)

    def test_every_step_is_plain_english(self, stored):
        settled, stuck = plain.journey(stored)
        for step in settled + stuck:
            for word in ("tier", "T0", "lexical", "residual", "calibrat"):
                assert word not in step.label
                assert word not in step.note


class TestTheAssistantActivity:
    """What the model did, read off the run and never inferred.

    `0 calls` had three different meanings and the dashboard showed none of
    them: no model configured, a model configured whose provider refused, and a
    model that answered and had every answer thrown out. Only the third is the
    system working, and it was invisible.
    """

    def test_a_deterministic_run_reports_no_model(self, stored):
        activity = plain.assistant_activity(stored)
        assert activity.available is False
        assert activity.used is False

    def test_used_follows_calls_not_availability(self, stored, tmp_path):
        """The claim the gate exists to prevent: a configured model that never
        answered must not read as "the model ran"."""
        copied = tmp_path / "runs" / stored.directory.name
        copied.mkdir(parents=True)
        for source in stored.directory.iterdir():
            shutil.copy2(source, copied / source.name)
        summary = json.loads((copied / "run.json").read_text(encoding="utf-8"))
        summary["llm"] = {"available": True, "calls": 0}
        (copied / "run.json").write_text(json.dumps(summary), encoding="utf-8")

        activity = plain.assistant_activity(list_runs(copied.parent)[0])
        assert activity.available is True
        assert activity.used is False

    def test_it_reads_every_gate_counter(self, stored, tmp_path):
        copied = tmp_path / "runs" / stored.directory.name
        copied.mkdir(parents=True)
        for source in stored.directory.iterdir():
            shutil.copy2(source, copied / source.name)
        summary = json.loads((copied / "run.json").read_text(encoding="utf-8"))
        summary["llm"] = {
            "available": True,
            "calls": 9,
            "total_tokens": 13097,
            "equivalent_paid_cost_inr": 8.81,
            "accepted": 0,
            "rejected_ungrounded": 9,
            "rejected_unverified": 0,
            "prose_rewritten": 20,
        }
        (copied / "run.json").write_text(json.dumps(summary), encoding="utf-8")

        activity = plain.assistant_activity(list_runs(copied.parent)[0])
        assert activity.used is True
        assert activity.calls == 9
        assert activity.refused == 9
        assert activity.prose_rewritten == 20
        assert activity.did_anything_visible is True

    def test_refused_sums_both_gates(self):
        activity = plain.AssistantActivity(
            available=True, calls=5, tokens=0, cost_inr=0.0, cache_hits=0,
            accepted=1, refused_ungrounded=3, demoted_unverified=2,
            prose_rewritten=0,
        )
        assert activity.refused == 5
