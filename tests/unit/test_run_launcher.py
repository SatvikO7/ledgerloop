"""``run.py`` -- the one command a reviewer runs, and the checks that guard it.

**Nothing here starts Streamlit.** Launching it blocks, and a test that hung
waiting for a browser would be worse than no test -- the same rule
``test_cli_demo.py`` follows with ``--no-ui``. Both children are replaced at the
module's single seam, :func:`run._run`, and what is asserted is the *commands it
would have run* and *whether it would have run them at all*.

The launcher is a launcher: it holds no reconciliation logic, no metric and no
money arithmetic. One test greps it to keep that true, because a wrapper that
started computing would be a second implementation living where nothing measures
it against ``EVALUATION.md``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

#: Resolved from this file rather than from the working directory, so the suite
#: passes from anywhere -- the convention the fixture-loading tests already use.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "run.py"


def _load() -> ModuleType:
    """Import ``run.py`` by path.

    It lives at the repository root rather than in ``src``, so it is not on the
    import path the way ``ledgerloop`` is, and a plain ``import run`` would
    depend on how pytest happened to seed ``sys.path``.
    """
    spec = importlib.util.spec_from_file_location("ledgerloop_run_launcher", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run() -> ModuleType:
    return _load()


class TestTheCanonicalCommands:
    """The launcher must call what the project already has, not something near it."""

    def test_the_demo_command_is_the_documented_one(self, run):
        assert run.demo_command() == [
            sys.executable,
            "-m",
            "ledgerloop.cli",
            "demo",
            "--no-ui",
            "--no-llm",
        ]

    def test_the_default_is_deterministic_whatever_is_configured(self, run):
        """`--no-llm` is not tidiness. A saved credential reaches the CLI the
        moment `--llm` is passed, so the quiet path has to *say* it is offline
        rather than rely on nobody having a key."""
        assert "--no-llm" in run.demo_command()
        assert "--llm" not in run.demo_command()

    def test_asking_for_the_model_swaps_the_flag(self, run):
        command = run.demo_command(use_llm=True)
        assert "--llm" in command
        assert "--no-llm" not in command

    def test_the_two_flags_are_never_sent_together(self, run):
        """They ask for opposite things, and the CLI rejects the pair. The
        launcher must not be the thing that produces it."""
        for wanted in (True, False):
            command = run.demo_command(use_llm=wanted)
            assert ("--llm" in command) != ("--no-llm" in command)

    def test_the_flag_is_offered_to_the_person_running_it(self, run, capsys):
        with pytest.raises(SystemExit):
            run.main(["--help"])
        assert "--llm" in " ".join(capsys.readouterr().out.split())

    def test_the_demo_runs_headless_so_a_failure_can_be_caught(self, run):
        """``--no-ui`` is what separates 'the run failed' from 'the screen is blank'.

        Without it the demo would launch Streamlit itself and its exit code would
        arrive far too late to be worth checking.
        """
        assert "--no-ui" in run.demo_command()

    def test_the_streamlit_command_is_the_documented_one(self, run):
        assert run.streamlit_command() == [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "src" / "ledgerloop" / "ui" / "app.py"),
        ]

    def test_both_commands_use_this_interpreter(self, run):
        """Not a bare ``python`` or ``streamlit``.

        A console script may be absent from ``PATH`` inside a venv on Windows,
        and a bare name could resolve to a different environment's Streamlit than
        the one holding ``ledgerloop``.
        """
        assert run.demo_command()[0] == sys.executable
        assert run.streamlit_command()[0] == sys.executable

    def test_a_moved_port_is_passed_through_and_the_default_is_not(self, run):
        assert "--server.port" not in run.streamlit_command(port=run.DEFAULT_PORT)
        moved = run.streamlit_command(port=run.DEFAULT_PORT + 1)
        assert moved[-2:] == ["--server.port", str(run.DEFAULT_PORT + 1)]

    def test_no_browser_asks_streamlit_for_headless(self, run):
        assert run.streamlit_command(headless=True)[-2:] == ["--server.headless", "true"]
        assert "--server.headless" not in run.streamlit_command(headless=False)


class TestPathsArePortable:
    def test_every_path_is_absolute_and_derived_from_this_file(self, run):
        assert run.ROOT == REPO_ROOT
        assert run.UI_APP.is_absolute()
        assert run.UI_APP.is_file()

    def test_the_app_path_is_built_with_pathlib_not_string_joining(self, run):
        source = RUN_PY.read_text(encoding="utf-8")
        assert 'src/ledgerloop/ui/app.py"' not in source
        assert 'ROOT / "src" / "ledgerloop" / "ui" / "app.py"' in source

    def test_children_run_from_the_repository_root(self, run, monkeypatch):
        """``RUNS_ROOT`` is a *relative* path, so the demo and the dashboard only
        agree about where the run record is when both start from the root."""
        seen: dict[str, object] = {}

        def fake_call(command, cwd=None, env=None):
            seen["cwd"] = cwd
            return 0

        monkeypatch.setattr(run.subprocess, "call", fake_call)
        run._run(["anything"])
        assert seen["cwd"] == str(REPO_ROOT)


class TestItRefusesToOpenABrokenDashboard:
    def test_a_failed_demo_stops_before_streamlit(self, run, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(command, *, env=None):
            calls.append(command)
            return 3

        monkeypatch.setattr(run, "_run", fake_run)
        code = run.main([])
        assert code != 0
        assert calls == [run.demo_command()], "Streamlit must not have been reached"

    def test_the_failure_names_the_command_that_failed(self, run, monkeypatch, capsys):
        monkeypatch.setattr(run, "_run", lambda command, *, env=None: 3)
        run.main([])
        err = capsys.readouterr().err
        assert "ledgerloop.cli demo --no-ui" in err
        assert "3" in err

    def test_a_successful_demo_reaches_streamlit(self, run, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(command, *, env=None):
            calls.append(command)
            return 0

        monkeypatch.setattr(run, "_run", fake_run)
        monkeypatch.setattr(run, "first_free_port", lambda: run.DEFAULT_PORT)
        assert run.main([]) == 0
        assert calls == [run.demo_command(), run.streamlit_command()]


class TestItChecksTheEnvironmentBeforeSpendingTime:
    def test_a_missing_dependency_names_it_and_the_install_command(
        self, run, monkeypatch, capsys
    ):
        monkeypatch.setattr(run, "missing_requirements", lambda names: ["streamlit"])
        monkeypatch.setattr(
            run, "_run", lambda command, *, env=None: pytest.fail("must not run anything")
        )
        assert run.main([]) != 0
        err = capsys.readouterr().err
        assert "streamlit" in err
        assert 'pip install -e ".[demo]"' in err

    def test_it_does_not_offer_to_install_anything(self, run):
        """A launcher that pip-installed behind a reviewer's back would be
        modifying the environment it was asked to demonstrate."""
        source = RUN_PY.read_text(encoding="utf-8")
        assert "Nothing was installed for you" in source
        for forbidden in ("pip install", "-m pip", "check_call", "ensurepip"):
            if forbidden == "pip install":
                continue  # appears only inside the printed hint
            assert forbidden not in source

    def test_a_broken_checkout_is_reported_rather_than_run(
        self, run, monkeypatch, capsys
    ):
        monkeypatch.setattr(run, "missing_markers", lambda: ["src/ledgerloop/cli.py"])
        monkeypatch.setattr(
            run, "_run", lambda command, *, env=None: pytest.fail("must not run anything")
        )
        assert run.main([]) != 0
        assert "does not look like a LedgerLoop checkout" in capsys.readouterr().err

    def test_this_checkout_passes_its_own_checks(self, run):
        assert run.missing_markers() == []
        assert run.missing_requirements(("ledgerloop",)) == []


class TestTheChildEnvironment:
    def test_an_installed_package_needs_no_path_help(self, run, monkeypatch):
        monkeypatch.setattr(run, "missing_requirements", lambda names: [])
        monkeypatch.setenv("PYTHONPATH", "")
        assert run.child_environment().get("PYTHONPATH", "") == ""

    def test_an_uninstalled_source_tree_is_put_on_the_path(self, run, monkeypatch):
        monkeypatch.setattr(run, "missing_requirements", lambda names: ["ledgerloop"])
        monkeypatch.delenv("PYTHONPATH", raising=False)
        assert run.child_environment()["PYTHONPATH"] == str(REPO_ROOT / "src")

    def test_an_existing_pythonpath_is_prepended_to_not_replaced(self, run, monkeypatch):
        monkeypatch.setattr(run, "missing_requirements", lambda names: ["ledgerloop"])
        monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
        value = run.child_environment()["PYTHONPATH"]
        assert value.startswith(str(REPO_ROOT / "src"))
        assert value.endswith("/somewhere/else")
        assert os.pathsep in value

    def test_no_credential_is_read_or_written(self, run):
        """The demo path is deterministic and needs no key; the launcher must not
        become the place one gets handled."""
        source = RUN_PY.read_text(encoding="utf-8")
        for forbidden in ("API_KEY", "GEMINI", "GROQ", "OPENROUTER", "load_dotenv"):
            assert forbidden not in source
        # Prose may name the file; code may not open it. Forbidding the string
        # literal draws that line exactly.
        assert '".env"' not in source
        assert "'.env'" not in source


class TestPortHandling:
    def test_a_free_port_is_used_as_is(self, run, monkeypatch):
        monkeypatch.setattr(run, "port_in_use", lambda port, host="127.0.0.1": False)
        assert run.first_free_port() == run.DEFAULT_PORT

    def test_an_occupied_port_moves_up_rather_than_stacking_a_second_server(
        self, run, monkeypatch
    ):
        busy = {run.DEFAULT_PORT, run.DEFAULT_PORT + 1}
        monkeypatch.setattr(
            run, "port_in_use", lambda port, host="127.0.0.1": port in busy
        )
        assert run.first_free_port() == run.DEFAULT_PORT + 2

    def test_an_entirely_busy_range_falls_back_to_the_default(self, run, monkeypatch):
        monkeypatch.setattr(run, "port_in_use", lambda port, host="127.0.0.1": True)
        assert run.first_free_port() == run.DEFAULT_PORT

    def test_nothing_is_listening_on_a_port_nothing_is_listening_on(self, run):
        assert run.port_in_use(1) is False


class TestTheInterface:
    def test_help_works_and_exits_zero(self, run, capsys):
        with pytest.raises(SystemExit) as exit_info:
            run.main(["--help"])
        assert exit_info.value.code == 0
        # argparse hard-wraps its help, so compare on normalised whitespace.
        out = " ".join(capsys.readouterr().out.split())
        assert "no API key is needed" in out

    def test_help_points_at_the_commands_it_wraps(self, run, capsys):
        with pytest.raises(SystemExit):
            run.main(["--help"])
        out = " ".join(capsys.readouterr().out.split())
        assert "ledgerloop.cli demo" in out
        assert "streamlit run" in out
        assert "DEMO.md" in out

    def test_an_unknown_option_is_rejected(self, run):
        with pytest.raises(SystemExit) as exit_info:
            run.main(["--make-the-numbers-better"])
        assert exit_info.value.code != 0


class TestItStaysALauncher:
    def test_it_holds_no_reconciliation_logic(self, run):
        """The same rule ``agent/nodes.py`` is held to: a wrapper that computed a
        number would be a second implementation living where nothing checks it
        against ``EVALUATION.md``."""
        source = RUN_PY.read_text(encoding="utf-8")
        for forbidden in ("precision", "recall", "_minor", "tolerance", "settlement"):
            assert forbidden not in source.lower(), f"{forbidden!r} does not belong here"

    def test_it_imports_nothing_from_ledgerloop(self, run):
        """Importing the package would make the launcher's own start-up depend on
        the thing it is meant to be checking for."""
        source = RUN_PY.read_text(encoding="utf-8")
        assert "import ledgerloop" not in source
        assert "from ledgerloop" not in source
