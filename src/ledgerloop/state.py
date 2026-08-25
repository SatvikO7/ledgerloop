"""The shared run state (PLAN.md §4.2).

Every pipeline node takes state and returns state. No hidden globals, which is
what makes the whole pipeline a testable near-pure function and what lets the
tier ladder loop without nodes reaching into each other.

This is a plain Pydantic model with no LangGraph import. The state machine is
assembled late (after the deterministic system is complete and measured), and
wrapping already-working node functions is a day's work -- whereas building
inside a framework while the data model is still settling taxes every step.
Keeping the framework out of the contract layer preserves that option.

LAYERING
--------
This lives at ``ledgerloop.state``, not ``ledgerloop.models.state``, because it
is not a pure contract -- it holds a :class:`RunConfig`. The dependency order is
``models`` (pure contracts, no internal deps beyond each other) <- ``config``
<- ``state``. Putting it inside ``models`` made that cycle real: importing
``models.base`` runs ``models/__init__``, which would import state, which
imports config, which imports ``models.base`` again.
"""

from __future__ import annotations

from pydantic import Field

from ledgerloop.config import RunConfig
from ledgerloop.models.audit import AuditEvent
from ledgerloop.models.base import LedgerModel
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import SourceName
from ledgerloop.models.metrics import CostLedger, RunMetrics
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.records import CanonicalRecord, RawRecord
from ledgerloop.models.truth import GroundTruth

__all__ = ["ReconState"]


class ReconState(LedgerModel):
    """Accumulating state for one reconciliation run."""

    run_id: str
    config: RunConfig

    raw: dict[SourceName, list[RawRecord]] = Field(default_factory=dict)
    normalized: list[CanonicalRecord] = Field(default_factory=list)

    candidates: list[MatchCandidate] = Field(default_factory=list)
    decisions: list[MatchDecision] = Field(default_factory=list)
    exceptions: list[ReconException] = Field(default_factory=list)

    metrics: RunMetrics | None = None
    cost: CostLedger = Field(default_factory=CostLedger)
    audit: list[AuditEvent] = Field(default_factory=list)

    ground_truth: GroundTruth | None = Field(
        default=None,
        description="Present only for evaluation and for building training and "
        "calibration sets. The matcher must never read this -- it is attached to "
        "state for the evaluator's convenience, and leaking it into a tier would "
        "make every metric meaningless. Step 1 adds an assertion to that effect.",
    )

    _audit_sequence: int = 0

    def next_audit_sequence(self) -> int:
        """Monotonic counter backing :attr:`AuditEvent.sequence`.

        Replay orders by this rather than by timestamp: several events routinely
        share a millisecond, and replay has to be exactly reproducible.
        """
        value = self._audit_sequence
        self._audit_sequence = value + 1
        return value

    @property
    def open_decisions(self) -> list[MatchDecision]:
        """Decisions not superseded by a later revision.

        The log is append-only, so a revision adds a record rather than editing
        one. This is the current view over that history.
        """
        superseded = {d.supersedes for d in self.decisions if d.supersedes is not None}
        return [d for d in self.decisions if d.decision_id not in superseded]
