#!/usr/bin/env python3
"""``python run.py`` -- the whole LedgerLoop demo, for someone who has never seen it.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a **launcher**, not a pipeline. It runs two commands the project already
has, in the order ``DEMO.md`` documents:

1. ``python -m ledgerloop.cli demo --no-ui`` -- generate the corpora, fit the
   calibration bundle, reconcile through the LangGraph pipeline, write the run
   record. Deterministic, offline, no API key.
2. ``python -m streamlit run src/ledgerloop/ui/app.py`` -- the dashboard, reading
   the run record stage 1 just wrote.

It contains **no reconciliation logic, no metric, and no money arithmetic**, and
it must stay that way. Everything it does is check the environment, print what is
happening, and call the two canonical commands. Both are still available directly
and are what a developer should use; this file exists so a reviewer does not have
to know either of them.

WHY ``--no-ui`` AND THEN STREAMLIT SEPARATELY
---------------------------------------------
``ledgerloop demo`` on its own already ends by launching Streamlit, so the
obvious launcher would be a one-line wrapper around it. It is split here for one
reason: **a failed reconciliation must not open a dashboard.** Running the demo
to completion first means its exit code can be checked before anything is shown,
and a broken run reports the command that failed instead of rendering a screen
full of nothing.

CWD IS LOAD-BEARING
-------------------
``agent/store.RUNS_ROOT`` is ``Path("reports/runs")`` -- relative. The demo writes
there and the dashboard reads from there, so **both children run with the
repository root as their working directory**, whatever directory the launcher was
invoked from.

NO CREDENTIALS
--------------
This file never reads, writes or prints ``.env``, a key, or any credential. The
demo path is deterministic and needs none; the optional live LLM is configured
exactly as ``DEMO.md`` describes, and the launcher is not part of it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

#: The repository root, derived from this file rather than from the caller's
#: working directory, so ``python /somewhere/else/run.py`` behaves identically.
ROOT = Path(__file__).resolve().parent

#: The dashboard's entry point. Located by path, never imported -- importing it
#: would make Streamlit a hard dependency of the launcher, which is the same
#: reason ``cli.py`` locates it by path.
UI_APP = ROOT / "src" / "ledgerloop" / "ui" / "app.py"

#: Files whose absence means this is not a usable checkout.
CHECKOUT_MARKERS = (
    Path("pyproject.toml"),
    Path("src") / "ledgerloop" / "cli.py",
    Path("src") / "ledgerloop" / "ui" / "app.py",
)

#: The install command ``DEMO.md`` gives. Printed verbatim on a missing
#: dependency; nothing here ever installs anything itself.
INSTALL_HINT = 'uv pip install -e ".[demo]"      (or: pip install -e ".[demo]")'

#: What each stage needs, as import names. ``langgraph`` is in the demo list
#: because the demo reconciles through the graph, not the plain chain.
DEMO_REQUIREMENTS = ("pydantic", "rapidfuzz", "langgraph")
UI_REQUIREMENTS = ("streamlit",)

#: Streamlit's default. If it is already serving, the launcher moves up rather
#: than starting a second instance on top of a confusing one.
DEFAULT_PORT = 8501
PORT_SEARCH_LIMIT = 20


def demo_command() -> list[str]:
    """The canonical deterministic demo, as ``DEMO.md`` documents it.

    Two arguments, and both are about what the launcher promises rather than
    about the corpus. Every other default -- the seed, the split, where the run
    record goes -- belongs to the demo command and is not the launcher's to have
    an opinion about.

    ``--no-ui`` so a failed reconciliation cannot reach a dashboard.

    ``--no-llm`` so the launcher is **deterministic and offline, always**. This
    file has claimed that since it was written, and until `.env` was loaded it
    was true by accident: no credential reached the process, so no ladder was
    built. Now that a saved key does reach it, the promise has to be stated. A
    configured model is a *capability*, not an instruction, and a reviewer
    running this for the first time should get the fast repeatable path rather
    than a network round trip they did not ask for. The model is one tick-box
    away inside the dashboard, and `ledgerloop demo` without this flag still
    honours a key.
    """
    return [sys.executable, "-m", "ledgerloop.cli", "demo", "--no-ui", "--no-llm"]


def streamlit_command(*, port: int = DEFAULT_PORT, headless: bool = False) -> list[str]:
    """The canonical dashboard command, on **this** interpreter.

    ``sys.executable -m streamlit`` rather than a bare ``streamlit``: the console
    script may not be on ``PATH`` inside a virtual environment on Windows, and
    resolving it by name could reach a different environment's Streamlit than the
    one holding ``ledgerloop``.
    """
    command = [sys.executable, "-m", "streamlit", "run", str(UI_APP)]
    if port != DEFAULT_PORT:
        command += ["--server.port", str(port)]
    if headless:
        command += ["--server.headless", "true"]
    return command


def missing_markers() -> list[str]:
    """Checkout files that are not where they should be."""
    return [str(marker) for marker in CHECKOUT_MARKERS if not (ROOT / marker).is_file()]


def missing_requirements(names: tuple[str, ...]) -> list[str]:
    """Which of ``names`` cannot be imported.

    ``find_spec`` rather than ``import``: it answers the question without paying
    for Streamlit's import, and without executing third-party module code.
    """
    absent: list[str] = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive
            found = False
        if not found:
            absent.append(name)
    return absent


def child_environment() -> dict[str, str]:
    """The environment the two children run in.

    Identical to this process's, with one addition: if ``ledgerloop`` is not
    importable but the source tree is present, ``src`` is put on ``PYTHONPATH``
    so a reviewer who cloned and installed only the third-party dependencies
    still gets a working demo. That is a *path*, not an install -- nothing is
    downloaded and nothing on disk is modified.
    """
    env = dict(os.environ)
    if not missing_requirements(("ledgerloop",)):
        return env
    source = ROOT / "src"
    if not (source / "ledgerloop" / "__init__.py").is_file():
        return env
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else str(source)
    return env


def port_in_use(port: int, *, host: str = "127.0.0.1") -> bool:
    """Whether something is already listening there.

    A connect attempt rather than a bind attempt: binding to test a port can
    succeed on a socket another process is about to take, and on some platforms
    ``SO_REUSEADDR`` makes a successful bind say nothing at all.
    """
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def first_free_port(start: int = DEFAULT_PORT, *, limit: int = PORT_SEARCH_LIMIT) -> int:
    """``start``, or the next port nothing is serving on.

    Returns ``start`` if the whole range is busy -- at that point the machine has
    a problem the launcher should report by letting Streamlit fail loudly, not by
    guessing further.
    """
    for candidate in range(start, start + limit):
        if not port_in_use(candidate):
            return candidate
    return start


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    """Run a child attached to this terminal, from the repository root.

    Attached on purpose: the demo's progress and Streamlit's log go straight to
    the user's console, and Ctrl-C reaches Streamlit rather than being swallowed
    by a wrapper. This is the single seam the tests replace.
    """
    return subprocess.call(command, cwd=str(ROOT), env=env)


def _fail(message: str, *lines: str) -> int:
    sys.stdout.flush()
    print(f"\n  ERROR: {message}", file=sys.stderr)
    for line in lines:
        print(f"         {line}", file=sys.stderr)
    sys.stderr.flush()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Run the LedgerLoop demo and open the dashboard. Deterministic and "
            "offline: no API key is needed, and none is read."
        ),
        epilog=(
            "This is a launcher around two existing commands -- "
            "`python -m ledgerloop.cli demo` and `python -m streamlit run "
            "src/ledgerloop/ui/app.py`. Both still work on their own; see DEMO.md."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the dashboard without opening a browser tab, and print the "
        "URL instead. Useful over SSH or in a recording.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print("LedgerLoop")
    print("  1. reconcile   the deterministic pipeline, end to end (no API key)")
    print("  2. inspect     the dashboard, reading the run it just wrote")
    print()

    absent_files = missing_markers()
    if absent_files:
        return _fail(
            f"this does not look like a LedgerLoop checkout: {ROOT}",
            "missing: " + ", ".join(absent_files),
            "Run this from the repository root, or re-clone.",
        )

    missing_demo = missing_requirements(DEMO_REQUIREMENTS)
    missing_ui = missing_requirements(UI_REQUIREMENTS)
    if missing_demo or missing_ui:
        # Both stages are checked before either runs: finding out that the
        # dashboard cannot start *after* a reconciliation has finished would
        # waste the reviewer's time on a failure that was knowable up front.
        return _fail(
            "missing dependencies: " + ", ".join(missing_demo + missing_ui),
            f"Install them with:  {INSTALL_HINT}",
            "",
            "Nothing was installed for you, on purpose.",
            "For the numbers alone, without the dashboard, the deterministic",
            "pipeline runs on its own:",
            "    python -m ledgerloop.cli demo --no-ui",
        )

    env = child_environment()
    if env.get("PYTHONPATH", "").startswith(str(ROOT / "src")):
        print(f"  note: `ledgerloop` is not installed; running from {ROOT / 'src'}.")
        print(f"        The documented setup is:  {INSTALL_HINT}")
        print()

    command = demo_command()
    print("[1/2] reconciling")
    print(f"      {' '.join(command)}")
    print()
    # Flushed before every child starts. Python block-buffers stdout when it is
    # a pipe rather than a terminal, so without this the launcher's own banner
    # arrives *after* the output of the command it is announcing.
    sys.stdout.flush()
    code = _run(command, env=env)
    if code != 0:
        return _fail(
            f"the demo exited with status {code}; the dashboard was not started",
            "The command that failed was:",
            "    " + " ".join(command),
            "",
            "A dashboard opened on a failed run would show nothing useful,",
            "which is why this stops here.",
        )

    port = first_free_port()
    print()
    print("[2/2] the dashboard")
    print("      the stage the demo just skipped, opened here so a failed run")
    print("      can never reach a screen")
    if port != DEFAULT_PORT:
        print(
            f"      something is already serving on port {DEFAULT_PORT}; "
            f"using {port} instead"
        )
    ui = streamlit_command(port=port, headless=args.no_browser)
    print(f"      {' '.join(ui)}")
    print(f"      http://localhost:{port}    Ctrl-C to stop.")
    print()
    sys.stdout.flush()
    try:
        return _run(ui, env=env)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0
    except FileNotFoundError:  # pragma: no cover - guarded by the check above
        return _fail(
            "could not start Streamlit on this interpreter",
            f"Install it with:  {INSTALL_HINT}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
