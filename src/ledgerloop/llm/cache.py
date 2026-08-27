"""Content-hash response cache, on disk, committed as fixtures.

PLAN.md 7.3: "responses content-hashed and cached to disk so reruns and CI are
free and deterministic". Three properties follow from that sentence and each is
enforced here rather than assumed:

**The key is the whole request.** Provider, model, temperature, prompt version
and the rendered prompt all go into the hash. A cache keyed on the prompt alone
would serve a Llama answer to a Gemini run and nobody would notice; a cache
keyed without the prompt *version* would serve yesterday's answer to today's
question.

**A hit is provable.** ``CostLedger.cache_hit_rate`` reaching 1.0 on a second
identical run is the evidence that a demo consumed zero live API calls, and
:class:`~ledgerloop.models.audit.AuditEvent` records the ``prompt_hash`` beside
every call. The hash written into the audit log is the filename on disk, so the
claim is checkable by ``ls``.

**A corrupt entry is a miss, not a crash.** Cache files are ordinary JSON in a
directory people can edit, and a half-written file after an interrupted run is
a real thing. Unreadable content is treated as absent -- the worst case is one
extra API call, and the alternative is a run that cannot start.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CacheKey", "ResponseCache"]


@dataclass(frozen=True)
class CacheKey:
    """Everything that could change an answer, hashed into one filename."""

    provider: str
    model: str
    temperature: float
    prompt_version: str
    prompt: str

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "prompt_version": self.prompt_version,
                "prompt": self.prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def filename(self) -> str:
        return f"{self.digest}.json"


@dataclass
class ResponseCache:
    """A directory of hashed responses, with hit and miss counters.

    The counters are the point of the class as much as the storage is: a cost
    ledger that reported calls without reporting hits could not distinguish a
    cheap run from a lucky one.
    """

    directory: Path
    hits: int = field(default=0)
    misses: int = field(default=0)
    writes: int = field(default=0)

    def path_for(self, key: CacheKey) -> Path:
        return self.directory / key.filename

    def get(self, key: CacheKey) -> str | None:
        """The cached completion for this request, or ``None``.

        Unreadable content counts as a miss. See the module docstring: one extra
        API call is a better failure than a run that will not start.
        """
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            completion = payload["completion"]
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None
        if not isinstance(completion, str):
            self.misses += 1
            return None
        self.hits += 1
        return completion

    def put(self, key: CacheKey, completion: str) -> Path:
        """Store a completion, with the request beside it.

        The prompt is written into the file rather than only hashed into its
        name. A cache directory whose entries cannot be read back by a human is
        a directory nobody can audit, and these are committed as fixtures
        precisely so a reviewer can read what the model was asked.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        payload = {
            "provider": key.provider,
            "model": key.model,
            "temperature": key.temperature,
            "prompt_version": key.prompt_version,
            "prompt": key.prompt,
            "completion": completion,
        }
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.writes += 1
        return path

    @property
    def attempts(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """A second identical run must reach 1.0 -- zero live API calls."""
        return self.hits / self.attempts if self.attempts else 0.0
