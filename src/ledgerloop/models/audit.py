"""The audit trail -- append-only, replayable.

PLAN.md D7: every decision records the tier that fired, the score, the
evidence, the prompt hash, tokens and latency, so the UI can step through a run
like a debugger. That only works if events are (a) immutable, (b) totally
ordered, and (c) self-contained enough to render without re-running anything.

:attr:`AuditEvent.sequence` gives the total order. Wall-clock timestamps are
recorded too, but they are not the ordering key -- two events inside the same
millisecond are common, and replay must be exactly reproducible.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.refs import RecordRef

__all__ = ["AuditEvent", "AuditEventType"]


class AuditEventType(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    NODE_ENTERED = "NODE_ENTERED"
    NODE_COMPLETED = "NODE_COMPLETED"
    RECORD_INGESTED = "RECORD_INGESTED"
    CANDIDATE_PROPOSED = "CANDIDATE_PROPOSED"
    ARITHMETIC_VERIFIED = "ARITHMETIC_VERIFIED"
    ARITHMETIC_FAILED = "ARITHMETIC_FAILED"
    DECISION_MADE = "DECISION_MADE"
    DECISION_SUPERSEDED = "DECISION_SUPERSEDED"
    EXCEPTION_RAISED = "EXCEPTION_RAISED"
    AUTO_RESOLUTION_APPLIED = "AUTO_RESOLUTION_APPLIED"
    AUTO_RESOLUTION_REFUSED = "AUTO_RESOLUTION_REFUSED"
    LLM_CALL = "LLM_CALL"
    LLM_CACHE_HIT = "LLM_CACHE_HIT"
    LLM_VALIDATION_FAILED = "LLM_VALIDATION_FAILED"
    PROVIDER_FAILOVER = "PROVIDER_FAILOVER"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


class AuditEvent(FrozenLedgerModel):
    """One immutable entry in the run log."""

    run_id: str
    sequence: int = Field(ge=0, description="Total order within the run. The replay cursor.")
    event_type: AuditEventType
    node: str = Field(description="Pipeline node that emitted this, e.g. 'tier2_aggregation'.")
    timestamp: datetime
    message: str = ""

    refs: tuple[RecordRef, ...] = ()
    candidate_id: str | None = None
    decision_id: str | None = None
    exception_id: str | None = None

    payload: dict[str, object] = Field(
        default_factory=dict,
        description="Event-specific detail: features, scores, subset members, "
        "tolerance bands. Rendered verbatim by the Audit Replay screen.",
    )

    # --- LLM provenance (PLAN.md §7.3) ---
    prompt_hash: str | None = Field(
        default=None,
        description="Content hash of the rendered prompt. Also the response-cache key, "
        "so a replay can prove a run consumed zero live API calls.",
    )
    prompt_version: str | None = Field(
        default=None, description="Version of the prompt template under agent/prompts/."
    )
    provider: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)

    def to_jsonl(self) -> str:
        """One line of the JSONL log. The on-disk audit format."""
        return self.model_dump_json(exclude_none=True)
