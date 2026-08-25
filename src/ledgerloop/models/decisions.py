"""Decisions -- immutable, append-only.

PLAN.md §3.2: "Decisions are never mutated; a revision writes a new record.
Replay is just reading the log in order." :attr:`MatchDecision.supersedes`
is what makes that work -- the tier ladder loops, and a late resolution can
overturn an earlier call, so the log must record both the original and the
revision rather than editing history.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.enums import DecisionOutcome, LinkType, Tier
from ledgerloop.models.refs import RecordRef

__all__ = ["MatchDecision"]


class MatchDecision(FrozenLedgerModel):
    """The policy's ruling on one candidate."""

    decision_id: str
    candidate_id: str
    link_type: LinkType
    source_ref: RecordRef
    target_ref: RecordRef
    tier: Tier
    outcome: DecisionOutcome
    calibrated_p: float = Field(ge=0.0, le=1.0)
    arithmetic_verified: bool
    decided_at: datetime
    reason: str = Field(
        description="Which rule fired, e.g. 'p=0.981 >= tau_high=0.95' or "
        "'demoted: verify_arithmetic failed'."
    )
    supersedes: str | None = Field(
        default=None,
        description="decision_id this revises. The superseded record stays in the log.",
    )

    @model_validator(mode="after")
    def _auto_match_requires_verified_arithmetic(self) -> MatchDecision:
        """An unverified link can never be auto-matched.

        This encodes PLAN.md §7.4 as a type-level guarantee rather than a
        convention the policy code has to remember. The most expensive failure
        mode in the system is a confident wrong auto-match, so the invariant is
        enforced where it cannot be bypassed.
        """
        if self.outcome is DecisionOutcome.AUTO_MATCHED and not self.arithmetic_verified:
            raise ValueError(
                "AUTO_MATCHED requires arithmetic_verified=True; "
                "unverified links must be demoted to NEEDS_REVIEW"
            )
        return self

    @property
    def pair(self) -> tuple[str, str]:
        return (self.source_ref.key, self.target_ref.key)

    @property
    def is_positive_prediction(self) -> bool:
        """Whether the system asserted this link exists.

        Only ``AUTO_MATCHED`` counts. ``NEEDS_REVIEW`` is explicitly *not* a
        prediction -- it is a referral to a human, and counting it as a match
        would be the precision-inflating trap the plan warns about.
        """
        return self.outcome is DecisionOutcome.AUTO_MATCHED
