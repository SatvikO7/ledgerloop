"""Read ``.env`` into the process environment, because nothing else was.

THE BUG THIS FIXES
------------------
``.env`` held a real ``GEMINI_API_KEY`` and every run still reported *no LLM
ran*. The report was **correct** -- that is the important part, and it is why
this module exists instead of a change to the wording.

The path is short and it is entirely mechanical:

* :func:`~ledgerloop.llm.providers.build_ladder` reads ``os.environ``;
* ``os.environ`` is populated by the shell, not by a file on disk;
* nothing in the project ever opened ``.env`` -- ``python-dotenv`` is not a
  dependency and never was;
* so the ladder found no credential, built no rung, and the client reported
  itself disabled. Truthfully.

``DEMO.md`` documented ``export GEMINI_API_KEY=...`` beside "copy
``.env.example`` to ``.env``", which is two instructions where a reader
reasonably follows one. A file named ``.env`` is expected to be loaded; this
loads it.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* **It never overwrites a variable that is already set.** An ``export`` in the
  shell, or a value injected by CI, is a deliberate act by whoever started the
  process; a file on disk must not silently win against it. This also keeps
  every existing test that sets an environment variable working unchanged.
* **It never returns, logs or renders a value.** :func:`load_env_file` returns
  the *names* it set and nothing else, so a caller that wants to report "a key
  was found" can do so without ever holding the secret.
* **It adds no dependency.** ``python-dotenv`` handles interpolation, multi-line
  values and export syntax that this project does not use. Thirty lines of
  parsing here is the same argument ``ARCHITECTURE.md`` decision 42 makes about
  the HTTP transport.
* **It is not clever.** No shell expansion, no ``${VAR}`` substitution, no
  command substitution. A ``.env`` is a list of names and literal values; a
  loader that evaluated anything would be a code path an untrusted file could
  reach.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DEFAULT_ENV_FILE", "load_env_file", "parse_env_text"]

#: Where the loader looks, relative to the working directory. The same file
#: ``.gitignore`` excludes and ``.env.example`` is a template for.
DEFAULT_ENV_FILE = Path(".env")

#: Longest file the loader will read, in bytes. A ``.env`` is a handful of
#: lines; anything larger is a mistake or a file that is not a ``.env``, and
#: reading it into memory is the only harm available here.
MAX_ENV_BYTES = 64 * 1024


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines. No expansion, no substitution, no execution.

    Blank lines and ``#`` comments are skipped. A leading ``export`` is
    tolerated because ``.env.example`` shows that form. Surrounding single or
    double quotes are stripped from the value, which is the one piece of shell
    syntax a real ``.env`` reliably contains.

    A line with no ``=`` is skipped rather than raised on: a malformed line in
    an optional config file should not stop a reconciliation.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def load_env_file(path: Path | None = None) -> list[str]:
    """Load ``path`` into ``os.environ`` and return the names that were **set**.

    Names already present in the environment are left alone and are *not*
    returned: the caller asked what this file contributed, and the answer for an
    already-set variable is "nothing".

    A missing or unreadable file returns an empty list. This is optional
    configuration, and a machine without one is the ordinary case -- every
    published number in this project was produced on one.
    """
    target = DEFAULT_ENV_FILE if path is None else path
    try:
        if not target.is_file() or target.stat().st_size > MAX_ENV_BYTES:
            return []
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []

    applied: list[str] = []
    for name, value in parse_env_text(text).items():
        if name in os.environ:
            continue
        os.environ[name] = value
        applied.append(name)
    return applied
