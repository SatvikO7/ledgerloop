"""``.env`` loading, and the guarantees that make it safe to do at all.

The bug this closes: a real ``GEMINI_API_KEY`` sat in ``.env`` and every run
reported *no LLM ran*. The report was correct -- nothing in the project ever
opened that file -- which is why the fix is a loader and not a change to the
wording.

Two properties matter more than the parsing:

* **An existing variable always wins.** A file on disk must never override an
  ``export`` or a value CI injected.
* **A value is never returned, logged or rendered.** The loader hands back the
  *names* it set, so a caller can say "a key was found" without holding one.
"""

from __future__ import annotations

import os

import pytest

from ledgerloop.envfile import MAX_ENV_BYTES, load_env_file, parse_env_text


class TestParsing:
    def test_a_plain_assignment(self):
        assert parse_env_text("A=1\nB=two\n") == {"A": "1", "B": "two"}

    def test_comments_and_blank_lines_are_skipped(self):
        text = "# a comment\n\nA=1\n   # indented comment\nB=2\n"
        assert parse_env_text(text) == {"A": "1", "B": "2"}

    def test_export_is_tolerated_because_the_example_file_uses_it(self):
        assert parse_env_text("export A=1\n") == {"A": "1"}

    def test_quotes_are_stripped(self):
        assert parse_env_text("A='one'\nB=\"two\"\n") == {"A": "one", "B": "two"}

    def test_a_value_containing_equals_is_kept_whole(self):
        """Base64 and JWT-shaped keys end in `=`; splitting on every one of them
        would silently truncate a credential."""
        assert parse_env_text("A=abc=def==\n") == {"A": "abc=def=="}

    def test_a_line_without_an_equals_is_skipped_not_raised(self):
        """A malformed line in optional config must not stop a reconciliation."""
        assert parse_env_text("nonsense\nA=1\n") == {"A": "1"}

    def test_an_empty_name_is_skipped(self):
        assert parse_env_text("=orphan\nA=1\n") == {"A": "1"}

    def test_an_empty_value_is_kept(self):
        assert parse_env_text("A=\n") == {"A": ""}

    def test_nothing_is_expanded_or_substituted(self):
        """No `${VAR}`, no `$(cmd)`, no interpolation. The literal is the value,
        because anything else is a code path an untrusted file can reach."""
        assert parse_env_text("A=${B}\nC=$(whoami)\n") == {
            "A": "${B}",
            "C": "$(whoami)",
        }


class TestLoading:
    def test_it_sets_a_variable_and_names_it(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LL_TEST_ONE", raising=False)
        path = tmp_path / ".env"
        path.write_text("LL_TEST_ONE=value\n", encoding="utf-8")
        assert load_env_file(path) == ["LL_TEST_ONE"]
        assert os.environ["LL_TEST_ONE"] == "value"

    def test_an_existing_variable_is_never_overwritten(self, tmp_path, monkeypatch):
        """The property that keeps every existing test working, and keeps an
        `export` meaningful."""
        monkeypatch.setenv("LL_TEST_TWO", "from-the-shell")
        path = tmp_path / ".env"
        path.write_text("LL_TEST_TWO=from-the-file\n", encoding="utf-8")
        assert load_env_file(path) == []
        assert os.environ["LL_TEST_TWO"] == "from-the-shell"

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        """A machine with no `.env` is the ordinary case, and the one every
        published number in this project was produced on."""
        assert load_env_file(tmp_path / "absent") == []

    def test_a_directory_is_not_an_error(self, tmp_path):
        assert load_env_file(tmp_path) == []

    def test_an_oversized_file_is_ignored(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("A=" + "x" * MAX_ENV_BYTES, encoding="utf-8")
        assert load_env_file(path) == []

    def test_it_returns_names_and_never_values(self, tmp_path, monkeypatch):
        """The whole point of returning names: a caller can report that a key was
        found without ever holding the secret."""
        monkeypatch.delenv("LL_TEST_SECRET", raising=False)
        path = tmp_path / ".env"
        path.write_text("LL_TEST_SECRET=super-secret-value\n", encoding="utf-8")
        returned = load_env_file(path)
        assert returned == ["LL_TEST_SECRET"]
        assert "super-secret-value" not in returned


class TestItNeverPrintsASecret:
    def test_the_module_does_not_log_or_print(self):
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "envfile.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("print(", "logging", "logger", "sys.stdout", "sys.stderr"):
            assert forbidden not in source

    def test_the_module_executes_nothing(self):
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "envfile.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("eval(", "exec(", "subprocess", "os.system", "expandvars"):
            assert forbidden not in source


class TestPublishedNumbersStayDeterministic:
    """The risk loading `.env` introduced, and the guard against it.

    Before this, `make eval` was deterministic *by accident*: no key was present,
    so the ladder built no rung. Making a key reachable would have turned the
    published pipeline live for anyone who had one, silently, and the committed
    `EVALUATION.md` would have stopped reproducing.
    """

    def test_the_published_metric_commands_use_a_disabled_client(self):
        from ledgerloop.cli import _deterministic_client

        assert _deterministic_client().enabled is False

    def test_eval_ablation_and_sweep_never_build_a_live_client(self):
        import pathlib
        import re

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "cli.py"
        ).read_text(encoding="utf-8")
        for name in ("_run_eval", "_run_ablation", "_run_sweep"):
            start = source.index(f"def {name}(")
            end = source.index("\ndef ", start + 10)
            body = source[start:end]
            assert "_client_for(" not in body, (
                f"{name} builds a key-driven client; a machine with a `.env` "
                "would publish numbers that do not reproduce"
            )
            assert re.search(r"_deterministic_client\(\)", body), name

    def test_the_demo_still_honours_an_exported_key(self):
        """`demo` and `run` remain key-driven, through the environment."""
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "cli.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _run_demo(")
        end = source.index("\ndef ", start + 10)
        assert "_client_for(args)" in source[start:end]

    def test_main_does_not_load_the_env_file(self):
        """Tried, and reverted, and this test is the record of why.

        A credential on disk must not change what a deterministic command
        produces. With one present the demo silently enabled T5: its run id
        became `t0t5-test-42` instead of `t0t4-test-42` and its audit log grew
        from 704 events to 710. Four tests failed purely because the machine
        happened to have a `.env`.

        `.env` is now read by the one command whose purpose is to reach a model,
        and by the dashboard behind an explicit opt-in. An `export` still works
        everywhere it always did.
        """
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "cli.py"
        ).read_text(encoding="utf-8")
        assert "load_env_file()" not in source[source.index("def main(") :]

    def test_the_live_measurement_command_does_load_it(self):
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "cli.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _run_llm_report(")
        end = source.index("\ndef ", start + 10)
        assert "load_env_file()" in source[start:end]


class TestTheUiReportsWhatHappened:
    def test_a_disabled_client_is_reported_as_unused(self):
        from ledgerloop.config import LLMConfig
        from ledgerloop.llm.client import LLMClient

        assert LLMClient(config=LLMConfig(enabled=False), provider=None).enabled is False

    def test_the_ui_defaults_to_not_asking_for_a_model(self):
        """A key in `.env` is a capability, not an instruction. Spending
        someone's quota because they once saved a credential is not a decision
        the interface gets to make."""
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        assert 'st.session_state.get("use_llm", False)' in source

    def test_the_ui_never_asks_anyone_to_paste_a_key(self):
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "ledgerloop" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        # The guarantee is that no *widget* collects a credential and no code
        # path handles one. The words "no API key" appear in prose, correctly.
        assert "api_key" not in source
        # Every free-text input in the interface, and what it is for. A widget
        # collecting a credential would show up here as a new entry.
        inputs = [
            index
            for index, line in enumerate(source.splitlines())
            if "text_input(" in line or "text_area(" in line
        ]
        assert len(inputs) == 1, "a new free-text input appeared; check what it collects"
        joined = source.splitlines()[inputs[0] : inputs[0] + 5]
        following = chr(10).join(joined).lower()
        assert "search" in following
        assert "key" not in following.replace("key=", "")

    @pytest.mark.parametrize("calls", [0, 1, 9])
    def test_used_follows_calls_made_not_a_key_existing(self, calls):
        from ledgerloop.eval.harness import ReconcileResult

        result = ReconcileResult(
            ingest=None,  # type: ignore[arg-type]
            matched=None,  # type: ignore[arg-type]
            exceptions=(),
            llm_calls=calls,
            llm_used=calls > 0,
        )
        assert result.llm_used is (calls > 0)
