"""What the agent proposes to do about an exception, and the leash it wore.

A contract rather than a behaviour, and it lives in ``models`` for the same
reason :class:`~ledgerloop.models.recon_exception.ReconException` does: the
report renders these, the audit trail records them, and the resolver produces
them. Three consumers means the shape has to be settled in one place that none
of them owns.

It also breaks a real import cycle. ``eval.report`` renders resolutions;
``exceptions.resolver`` builds them and needs the tier ladder to reason about
money; the ladder's package imports ``eval.metrics``. With the type declared
here, the report depends on a contract instead of on the package that fills it
in, and the cycle does not exist.

**Nothing here is ever posted anywhere.** PLAN.md 1.3: the agent proposes
journal adjustments and never writes to a real system. ``applied`` means "this
passed its bound and the agent stands behind it", not "this happened".
"""

from __future__ import annotations

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import ExceptionClass
from ledgerloop.models.refs import RecordRef

__all__ = ["AutoResolution"]


class AutoResolution(FrozenLedgerModel):
    """One proposal the agent is willing to make, and the rule that allowed it."""

    exception_id: str
    exception_class: ExceptionClass
    rule: str = Field(description="Which PLAN.md 8.3 rule fired.")
    action: str = Field(description="What the agent proposes, in one sentence.")
    amount_minor: MinorUnits = Field(
        default=0, description="Money the proposal moves. Zero for a re-window."
    )
    refs: tuple[RecordRef, ...] = ()
    applied: bool = Field(
        default=True,
        description="False when a bound refused it. Refusals are emitted, never "
        "dropped -- a leash nobody can see is not a leash.",
    )
    bound: str = Field(default="", description="The bound this was checked against.")
    refusal: str | None = Field(
        default=None, description="Why the bound refused, when it did."
    )
