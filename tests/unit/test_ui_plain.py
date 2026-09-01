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
