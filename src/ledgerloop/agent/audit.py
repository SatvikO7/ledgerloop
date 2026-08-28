"""The append-only audit log, and the JSONL file it persists to.

PLAN.md D7: *every decision — tier fired, score, evidence, LLM prompt hash,
tokens, latency — is persisted, and the UI can step through any single
reconciliation decision like a debugger.*
:class:`~ledgerloop.models.audit.AuditEvent` has been the contract for that
since Step 0 and had **no producer**. This module is the producer.

WHY THE ORDER KEY IS A COUNTER, NOT A CLOCK
-------------------------------------------
:attr:`AuditEvent.sequence` is a monotonic integer handed out by
:meth:`~ledgerloop.state.ReconState.next_audit_sequence`. Several events
routinely land inside one millisecond, and replay has to be exactly
reproducible, so the wall clock is recorded but never ordered on.

WHY JSONL
---------
One event per line, appended, never rewritten. That is what makes the log
*append-only* in the filesystem and not merely by convention: a replay reads
lines in order and stops wherever the file stops, so a run killed mid-node
leaves a log that still replays up to the point it died. A single JSON document
would have to be rewritten on every event and would be unreadable if truncated.

No new dependency: the format is `AuditEvent.to_jsonl()`, which Step 0 defined.

WHAT IS AND IS NOT RECORDED
---------------------------
Recorded: node entry and exit, per-node timings, candidate and decision counts,
exceptions raised, LLM calls with their prompt hash and token counts, and the
residual loop's own iteration count. **Not** recorded: the source records
themselves. The log points at them by :class:`~ledgerloop.models.refs.RecordRef`
and the replay reads them from the dataset, so the log stays small and cannot
drift from the data it describes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ledgerloop.models.audit import AuditEvent, AuditEventType
from ledgerloop.models.refs import RecordRef

__all__ = ["AUDIT_FILE", "AuditLog", "read_audit_jsonl"]

#: The log's filename inside a run directory.
AUDIT_FILE = "audit.jsonl"


@dataclass
class AuditLog:
    """An in-memory append-only log for one run.

    The sequence counter lives here rather than on
    :class:`~ledgerloop.state.ReconState` because the graph emits events before
    a ``ReconState`` exists -- ingest and normalisation both happen before the
    ladder builds one. The two counters would otherwise have to be reconciled,
    and a replay cursor that could disagree with itself is worse than no cursor.
    """

    run_id: str
    events: list[AuditEvent] = field(default_factory=list)
    _sequence: int = 0

    def emit(
        self,
        event_type: AuditEventType,
        node: str,
        *,
        message: str = "",
        refs: Sequence[RecordRef] = (),
        candidate_id: str | None = None,
        decision_id: str | None = None,
        exception_id: str | None = None,
        payload: dict[str, object] | None = None,
        prompt_hash: str | None = None,
        prompt_version: str | None = None,
        provider: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> AuditEvent:
        """Append one event and return it. The only way to write to the log."""
        event = AuditEvent(
            run_id=self.run_id,
            sequence=self._sequence,
            event_type=event_type,
            node=node,
            timestamp=datetime.now(),
            message=message,
            refs=tuple(refs),
            candidate_id=candidate_id,
            decision_id=decision_id,
            exception_id=exception_id,
            payload=payload or {},
            prompt_hash=prompt_hash,
            prompt_version=prompt_version,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        self._sequence += 1
        self.events.append(event)
        return event

    def extend(self, events: Iterable[AuditEvent]) -> None:
        """Absorb events from elsewhere, renumbering them into this log's order.

        Used when a resumed run continues a log it did not start. Renumbering
        rather than trusting the incoming sequence is what keeps the total order
        total: two logs each starting at zero would otherwise interleave into
        something replay cannot walk.
        """
        for event in events:
            self.events.append(event.model_copy(update={"sequence": self._sequence}))
            self._sequence += 1

    @property
    def next_sequence(self) -> int:
        return self._sequence

    def by_node(self) -> dict[str, int]:
        """Event count per node. The Audit Replay screen's summary row."""
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.node] = counts.get(event.node, 0) + 1
        return counts

    def of_type(self, *types: AuditEventType) -> tuple[AuditEvent, ...]:
        wanted = set(types)
        return tuple(event for event in self.events if event.event_type in wanted)

    def write_jsonl(self, path: Path) -> Path:
        """Write the whole log, one event per line, with ``\\n`` endings.

        Rewrites rather than appends, because the in-memory log is the whole
        truth for a completed run. Resumption reads the file back in through
        :func:`read_audit_jsonl` and :meth:`extend`, so nothing is lost.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for event in self.events:
                handle.write(event.to_jsonl())
                handle.write("\n")
        return path


def read_audit_jsonl(path: Path) -> tuple[AuditEvent, ...]:
    """Read a log back, in sequence order.

    **A truncated last line is dropped, not raised on.** A run killed mid-write
    leaves a partial line, and the events before it are still a valid prefix of
    the run -- which is exactly the case audit replay exists to survive. The
    alternative is a log that becomes unreadable at the moment it matters most.
    """
    if not path.is_file():
        return ()
    events: list[AuditEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(AuditEvent.model_validate_json(stripped))
        except ValueError:
            continue
    return tuple(sorted(events, key=lambda event: event.sequence))
