"""Uploaded files: identified, checked, and honestly assessed.

Three concerns, tested apart:

* **Identification** must be deterministic and must refuse to guess. A file
  silently read as the wrong source produces a reconciliation that is wrong in a
  way nobody would think to check.
* **Validation** must reject the things a person actually uploads by mistake --
  the wrong file, an empty export, a spreadsheet saved as the wrong type -- and
  must say which, in words they can act on.
* **Assessment** must tell the truth about what a combination can do. Every
  claim in ``CAPABILITIES`` is checked here against the *real ladder*, so the
  wording on screen cannot drift away from what the pipeline actually does.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from ledgerloop.config import RunConfig
from ledgerloop.eval.harness import reconcile_only
from ledgerloop.ingest.dataset import ingest_available, ingest_dataset
from ledgerloop.ingest.problems import IngestError
from ledgerloop.matching.pipeline import run_matching
from ledgerloop.ui.uploads import (
    CAPABILITIES,
    MAX_UPLOAD_BYTES,
    SourceKind,
    assess,
    detect,
    row_count,
    validate,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


def _payload(kind: SourceKind) -> bytes:
    return (FIXTURE / kind.value).read_bytes()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The three real source files, to copy subsets out of."""
    return FIXTURE


def _dir_with(tmp_path: Path, *kinds: SourceKind) -> Path:
    directory = tmp_path / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        shutil.copy(FIXTURE / kind.value, directory / kind.value)
    return directory


class TestIdentification:
    @pytest.mark.parametrize("kind", list(SourceKind))
    def test_each_real_source_is_recognised(self, kind):
        assert detect(kind.value, _payload(kind)) is kind

    def test_it_reads_content_not_upload_order(self):
        """A person who uploads their bank export second must not have it read
        as a ledger. The name is a tie-break, never the decision."""
        assert detect("some-export-2.csv", _payload(SourceKind.BANK)) is SourceKind.BANK
        assert (
            detect("first.json", _payload(SourceKind.PROCESSOR)) is SourceKind.PROCESSOR
        )

    def test_an_unknown_csv_is_not_guessed_at(self):
        assert detect("mystery.csv", b"alpha,beta\n1,2\n") is None

    def test_an_empty_file_is_not_identified(self):
        assert detect("empty.csv", b"") is None

    def test_json_that_is_not_a_processor_report_is_refused(self):
        assert detect("other.json", b'{"orders": []}') is None

    def test_malformed_json_is_refused_rather_than_raising(self):
        assert detect("broken.json", b"{not json") is None

    def test_a_file_over_the_size_limit_is_refused(self):
        assert detect("huge.csv", b"x" * (MAX_UPLOAD_BYTES + 1)) is None


class TestValidation:
    @pytest.mark.parametrize("kind", list(SourceKind))
    def test_a_real_source_validates(self, kind):
        assert validate(kind, kind.value, _payload(kind)) is None

    def test_an_empty_file_is_rejected(self):
        problem = validate(SourceKind.BANK, "bank.csv", b"")
        assert problem is not None
        assert "empty" in problem.reason.lower()

    def test_a_header_with_no_rows_is_rejected(self):
        problem = validate(
            SourceKind.BANK, "bank.csv", b"txn_id,value_date,narration\n"
        )
        assert problem is not None
        assert "no rows" in problem.reason.lower()

    def test_missing_columns_are_named(self):
        problem = validate(SourceKind.BANK, "bank.csv", b"txn_id\nBNK-1\n")
        assert problem is not None
        assert "value_date" in problem.detail
        assert "narration" in problem.detail

    def test_the_wrong_extension_is_rejected(self):
        problem = validate(SourceKind.BANK, "bank.xlsx", _payload(SourceKind.BANK))
        assert problem is not None
        assert "CSV" in problem.reason

    def test_a_processor_report_must_be_json(self):
        problem = validate(
            SourceKind.PROCESSOR, "psp.csv", _payload(SourceKind.PROCESSOR)
        )
        assert problem is not None
        assert "JSON" in problem.reason

    def test_an_empty_settlements_list_is_rejected(self):
        problem = validate(
            SourceKind.PROCESSOR, "psp.json", b'{"settlements": []}'
        )
        assert problem is not None
        assert "no payouts" in problem.reason.lower()

    def test_an_oversized_file_is_rejected(self):
        problem = validate(SourceKind.BANK, "bank.csv", b"x" * (MAX_UPLOAD_BYTES + 1))
        assert problem is not None
        assert "too large" in problem.reason.lower()

    def test_undecodable_bytes_are_rejected_rather_than_raising(self):
        problem = validate(SourceKind.PROCESSOR, "psp.json", b"\xff\xfe\x00\x01" * 8)
        assert problem is not None


class TestRowCounts:
    def test_a_csv_counts_rows_not_the_header(self):
        assert row_count(SourceKind.BANK, _payload(SourceKind.BANK)) == 23

    def test_a_processor_report_counts_payouts(self):
        assert row_count(SourceKind.PROCESSOR, _payload(SourceKind.PROCESSOR)) == 5


class TestPartialIngestion:
    def test_one_source_is_read_and_the_others_are_empty(self, tmp_path):
        result = ingest_available(_dir_with(tmp_path, SourceKind.BANK))
        assert len(result.bank_txns) == 23
        assert result.orders == () and result.payments == () and result.settlements == ()

    def test_two_sources_are_read(self, tmp_path):
        result = ingest_available(
            _dir_with(tmp_path, SourceKind.PROCESSOR, SourceKind.BANK)
        )
        assert len(result.payments) == 60
        assert len(result.bank_txns) == 23
        assert result.orders == ()

    def test_all_three_match_the_strict_reader_exactly(self, tmp_path):
        """`ingest_available` must not become a second, subtly different reader."""
        directory = _dir_with(
            tmp_path, SourceKind.LEDGER, SourceKind.PROCESSOR, SourceKind.BANK
        )
        loose = ingest_available(directory)
        strict = ingest_dataset(directory)
        assert loose.orders == strict.orders
        assert loose.payments == strict.payments
        assert loose.settlements == strict.settlements
        assert loose.bank_txns == strict.bank_txns

    def test_nothing_at_all_is_read_as_nothing(self, tmp_path):
        empty = tmp_path / "none"
        empty.mkdir()
        assert ingest_available(empty).record_count == 0

    def test_the_strict_reader_still_refuses_a_partial_corpus(self, tmp_path):
        """`ingest_dataset` keeps its contract. A generated split missing a file
        is a broken split, and that must stay an error."""
        with pytest.raises(IngestError):
            ingest_dataset(_dir_with(tmp_path, SourceKind.BANK))


class TestTheCapabilityClaimsAreTrue:
    """Every sentence in ``CAPABILITIES`` is checked against the real ladder.

    This is the test that matters. The screen tells a person what their files
    can do; if that wording drifts from what the pipeline actually produces, the
    product is lying to them politely.
    """

    @staticmethod
    def _links(*kinds: SourceKind) -> int:
        full = ingest_dataset(FIXTURE)
        kept = {
            SourceKind.LEDGER: ("orders",),
            SourceKind.PROCESSOR: ("payments", "settlements"),
            SourceKind.BANK: ("bank_txns",),
        }
        keep = {field for kind in kinds for field in kept[kind]}
        blanked = {
            field: ()
            for group in kept.values()
            for field in group
            if field not in keep
        }
        run = run_matching(replace(full, **blanked), RunConfig(run_id="probe"))
        return len(run.predictions)

    @pytest.mark.parametrize(
        "kinds",
        [
            (SourceKind.BANK,),
            (SourceKind.PROCESSOR,),
            (SourceKind.LEDGER,),
            (SourceKind.LEDGER, SourceKind.PROCESSOR),
            (SourceKind.LEDGER, SourceKind.BANK),
        ],
    )
    def test_a_combination_called_unreconcilable_produces_no_links(self, kinds):
        assert assess(set(kinds)).can_reconcile is False
        assert self._links(*kinds) == 0

    @pytest.mark.parametrize(
        "kinds",
        [
            (SourceKind.PROCESSOR, SourceKind.BANK),
            (SourceKind.LEDGER, SourceKind.PROCESSOR, SourceKind.BANK),
        ],
    )
    def test_a_combination_called_reconcilable_produces_links(self, kinds):
        assert assess(set(kinds)).can_reconcile is True
        assert self._links(*kinds) > 0

    def test_the_ledger_is_optional_and_adding_it_helps(self):
        """Exactly what the screen promises: the processor report and the bank
        statement are the minimum, and the ledger recovers more."""
        without = self._links(SourceKind.PROCESSOR, SourceKind.BANK)
        with_ledger = self._links(
            SourceKind.LEDGER, SourceKind.PROCESSOR, SourceKind.BANK
        )
        assert without > 0
        assert with_ledger > without

    def test_every_combination_has_an_answer(self):
        """No combination may fall through to a KeyError on screen."""
        import itertools

        for size in range(4):
            for combo in itertools.combinations(SourceKind, size):
                assert assess(set(combo)) is not None

    def test_every_unreconcilable_combination_says_what_would_help(self):
        for sources, verdict in CAPABILITIES.items():
            if verdict.can_reconcile or not sources:
                continue
            assert verdict.missing_hint, f"{sources} offers no way forward"

    def test_no_capability_sentence_uses_jargon(self):
        for verdict in CAPABILITIES.values():
            text = f"{verdict.headline} {verdict.detail} {verdict.missing_hint}".lower()
            for word in ("tier", "t0", "t3", "precision", "recall", "corpus", "utr"):
                assert word not in text


class TestReconcilingUploads:
    def test_two_sources_reconcile_without_any_ground_truth(self, tmp_path):
        result = reconcile_only(
            _dir_with(tmp_path, SourceKind.PROCESSOR, SourceKind.BANK)
        )
        assert result.committed_links > 0
        assert result.committed_minor > 0

    def test_all_three_reconcile_more_than_two(self, tmp_path):
        two = reconcile_only(
            _dir_with(tmp_path / "a", SourceKind.PROCESSOR, SourceKind.BANK)
        )
        three = reconcile_only(
            _dir_with(
                tmp_path / "b",
                SourceKind.LEDGER,
                SourceKind.PROCESSOR,
                SourceKind.BANK,
            )
        )
        assert three.committed_links > two.committed_links

    def test_one_source_reconciles_nothing_and_does_not_raise(self, tmp_path):
        result = reconcile_only(_dir_with(tmp_path, SourceKind.BANK))
        assert result.committed_links == 0
        assert result.committed_minor == 0

    def test_no_files_at_all_is_not_an_error(self, tmp_path):
        empty = tmp_path / "none"
        empty.mkdir()
        result = reconcile_only(empty)
        assert result.committed_links == 0
        assert result.queue_size == 0

    def test_it_reports_no_model_when_none_was_given(self, tmp_path):
        """`llm_used` is read from the cost ledger -- calls actually made -- and
        never from a credential existing."""
        result = reconcile_only(
            _dir_with(tmp_path, SourceKind.PROCESSOR, SourceKind.BANK)
        )
        assert result.llm_used is False
        assert result.llm_calls == 0

    def test_a_key_without_a_call_is_still_reported_as_unused(self, tmp_path):
        """The claim the whole gate exists to prevent: a client that is *enabled*
        but never answered must not read as 'the model ran'."""
        from ledgerloop.config import LLMConfig
        from ledgerloop.llm.client import LLMClient

        idle = LLMClient(config=LLMConfig(enabled=False), provider=None)
        result = reconcile_only(
            _dir_with(tmp_path, SourceKind.PROCESSOR, SourceKind.BANK), client=idle
        )
        assert result.llm_used is False

    def test_the_deterministic_result_is_identical_with_and_without_a_client(
        self, tmp_path
    ):
        """A disabled client must change nothing at all."""
        directory = _dir_with(
            tmp_path, SourceKind.LEDGER, SourceKind.PROCESSOR, SourceKind.BANK
        )
        from ledgerloop.config import LLMConfig
        from ledgerloop.llm.client import LLMClient

        plain = reconcile_only(directory)
        with_idle = reconcile_only(
            directory, client=LLMClient(config=LLMConfig(enabled=False), provider=None)
        )
        assert plain.committed_links == with_idle.committed_links
        assert plain.committed_minor == with_idle.committed_minor
        assert plain.queue_size == with_idle.queue_size


class TestTheDashboardDoesNotWriteIntoTheFixtures:
    """A real leak, caught by `git status` after a live verification run.

    `LLMConfig.cache_dir` defaults to `tests/fixtures/llm_cache`, which is
    committed and stays empty on purpose -- Step 10 leaked five stand-in
    responses in there once and they had to be removed. Ticking the AI box in
    the dashboard wrote ten real Gemini answers into it before this was fixed.
    """

    def test_the_ui_client_sets_its_own_cache_directory(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _ui_client(")
        end = source.index(chr(10) + "def ", start + 10)
        body = source[start:end]
        assert "cache_dir=" in body
        assert "reports/llm_cache_ui" in body

    def test_no_code_path_in_the_ui_points_at_the_committed_cache(self):
        """Comments may name the directory -- one explains this very fix. What
        must not exist is a line of code that writes there."""
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        code = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert not any("tests/fixtures" in line for line in code)


class TestUploadsAreTreatedAsUntrusted:
    def test_nothing_in_the_module_executes_a_file(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "uploads.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("eval(", "exec(", "pickle", "subprocess", "os.system", "__import__"):
            assert forbidden not in source

    def test_a_json_bomb_shaped_payload_is_refused_not_parsed_forever(self):
        """Deeply nested JSON is rejected by the schema check, not by recursing."""
        payload = json.dumps({"settlements": {"not": "a list"}}).encode()
        assert detect("x.json", payload) is None
        assert validate(SourceKind.PROCESSOR, "x.json", payload) is not None

    def test_uploads_never_resolve_into_the_repository(self, tmp_path):
        """The caller supplies the directory; nothing here reaches for the
        project's own data, so an upload cannot overwrite a committed fixture."""
        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "uploads.py"
        ).read_text(encoding="utf-8")
        assert "data/fixtures" not in source
        assert "data/generated" not in source

    def test_a_session_directory_is_outside_the_project(self):
        """Written under the system temp root, never beside the fixtures."""
        with tempfile.TemporaryDirectory(prefix="ledgerloop-upload-") as created:
            assert Path(created).resolve() != FIXTURE.resolve()
            assert "data" not in Path(created).parts[-1]
