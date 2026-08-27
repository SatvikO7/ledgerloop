"""Bounded auto-resolution -- what the agent may do by itself, and its leash.

PLAN.md 8.3 names three classes and a hard bound for each. Two properties of
this module matter more than the rules themselves:

**Nothing is posted anywhere.** PLAN.md 1.3: the agent *proposes* journal
adjustments and never writes to a real system. A resolution here is a record
saying "this is what I would do and this is the rule that let me", carrying the
bound it was checked against. The evaluated predictions are untouched -- a
resolver that could add a link would be a sixth matching tier wearing a
different name, and every precision figure in the project would become a claim
about it.

**The bounds are checked in code, printed in the report, and refuse loudly.**
A proposal that exceeds one is emitted as ``refused`` with the bound named, not
dropped. A leash nobody can see is not a leash.

WHY THE RUN BUDGET IS ORDER-DEPENDENT, AND WHY THAT IS FINE
------------------------------------------------------------
``rounding_per_run_minor`` caps the total across a run, so which items fit
depends on the order they are considered in. The order is the queue's own --
descending money, then id -- which is total and reproducible, so two runs over
the same data resolve exactly the same set. Spending the budget on the largest
eligible drifts first is also the right policy: a controller would rather see
the ₹4.80 adjustment made than four ₹0.03 ones.

NO LLM
------
Nothing here consults a model. An LLM cannot decide that an amount is within a
bound, because that is arithmetic, and it cannot decide that a class is
resolvable, because that is the taxonomy. Step 9 may rewrite the *prose* on the
exception a resolution attaches to; it may not create, approve or size one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ledgerloop.config import AutoResolutionBounds
from ledgerloop.exceptions.taxonomy import AGENT_RESOLVABLE_CLASSES
from ledgerloop.models.enums import ExceptionClass
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.resolution import AutoResolution
from ledgerloop.money import format_minor

__all__ = [
    "AutoResolution",
    "ResolutionOutcome",
    "resolve_bounded",
]


@dataclass(frozen=True)
class ResolutionOutcome:
    """Everything the bounded rules concluded, applied and refused alike."""

    resolutions: tuple[AutoResolution, ...] = ()
    rounding_spent_minor: int = 0

    @property
    def applied(self) -> tuple[AutoResolution, ...]:
        return tuple(item for item in self.resolutions if item.applied)

    @property
    def refused(self) -> tuple[AutoResolution, ...]:
        return tuple(item for item in self.resolutions if not item.applied)

    @property
    def resolved_ids(self) -> frozenset[str]:
        return frozenset(item.exception_id for item in self.applied)

    @property
    def applied_amount_minor(self) -> int:
        return sum(item.amount_minor for item in self.applied)


def _rounding(
    exception: ReconException, bounds: AutoResolutionBounds, spent: int
) -> AutoResolution:
    """Post a rounding adjustment: ≤ ₹5 per record, ≤ ₹500 per run."""
    amount = exception.impact_minor
    if amount > bounds.rounding_per_record_minor:
        return AutoResolution(
            exception_id=exception.exception_id,
            exception_class=exception.exception_class,
            rule="rounding adjustment",
            action=f"propose a rounding adjustment of {format_minor(amount)}",
            amount_minor=amount,
            refs=exception.involved_refs[:1],
            applied=False,
            bound=f"≤ {format_minor(bounds.rounding_per_record_minor)} per record",
            refusal=(
                f"{format_minor(amount)} exceeds the per-record bound of "
                f"{format_minor(bounds.rounding_per_record_minor)}"
            ),
        )
    if spent + amount > bounds.rounding_per_run_minor:
        return AutoResolution(
            exception_id=exception.exception_id,
            exception_class=exception.exception_class,
            rule="rounding adjustment",
            action=f"propose a rounding adjustment of {format_minor(amount)}",
            amount_minor=amount,
            refs=exception.involved_refs[:1],
            applied=False,
            bound=f"≤ {format_minor(bounds.rounding_per_run_minor)} per run",
            refusal=(
                f"the run has already committed {format_minor(spent)} of its "
                f"{format_minor(bounds.rounding_per_run_minor)} rounding budget"
            ),
        )
    return AutoResolution(
        exception_id=exception.exception_id,
        exception_class=exception.exception_class,
        rule="rounding adjustment",
        action=(
            f"post a rounding adjustment of {format_minor(amount)} against "
            f"{exception.involved_refs[0].record_id}"
        ),
        amount_minor=amount,
        refs=exception.involved_refs[:1],
        applied=True,
        bound=(
            f"≤ {format_minor(bounds.rounding_per_record_minor)} per record, "
            f"≤ {format_minor(bounds.rounding_per_run_minor)} per run"
        ),
    )


def _timing(exception: ReconException, bounds: AutoResolutionBounds) -> AutoResolution:
    """Re-window and re-match: ≤ 5 days.

    The day gap is read off the evidence chain the classifier already built,
    rather than recomputed here. One derivation of a number, in the module that
    owns it -- two would eventually disagree, and the disagreement would be
    between the queue a controller reads and the leash it was checked against.
    """
    gap = _day_gap(exception)
    if gap is None:
        return AutoResolution(
            exception_id=exception.exception_id,
            exception_class=exception.exception_class,
            rule="re-window and re-match",
            action="propose re-matching across a wider date window",
            refs=exception.involved_refs[:1],
            applied=False,
            bound=f"≤ {bounds.timing_shift_max_days} days",
            refusal="no candidate credit carries a measurable day gap",
        )
    if gap > bounds.timing_shift_max_days:
        return AutoResolution(
            exception_id=exception.exception_id,
            exception_class=exception.exception_class,
            rule="re-window and re-match",
            action=f"propose re-matching across a {gap}-day window",
            refs=exception.involved_refs[:1],
            applied=False,
            bound=f"≤ {bounds.timing_shift_max_days} days",
            refusal=(
                f"the gap of {gap} day(s) exceeds the bound of "
                f"{bounds.timing_shift_max_days}"
            ),
        )
    return AutoResolution(
        exception_id=exception.exception_id,
        exception_class=exception.exception_class,
        rule="re-window and re-match",
        action=(
            f"re-window {exception.involved_refs[0].record_id} by {gap} day(s) and "
            "re-match, for review before anything is posted"
        ),
        refs=exception.involved_refs[:1],
        applied=True,
        bound=f"≤ {bounds.timing_shift_max_days} days",
    )


def _duplicate(exception: ReconException) -> AutoResolution:
    """Flag the second credit and link it to the first. Never deletes anything."""
    refs = exception.involved_refs
    other = refs[1].record_id if len(refs) > 1 else "the earlier credit"
    return AutoResolution(
        exception_id=exception.exception_id,
        exception_class=exception.exception_class,
        rule="flag and link",
        action=(
            f"flag {refs[0].record_id} as a duplicate of {other} and link the two; "
            "neither row is deleted or amended"
        ),
        amount_minor=exception.impact_minor,
        refs=refs[:2],
        applied=True,
        bound="never deletes anything",
    )


def _day_gap(exception: ReconException) -> int | None:
    """The day gap the classifier recorded, parsed back off its own evidence."""
    for item in exception.evidence:
        marker = " day(s) from "
        if marker in item.detail:
            head = item.detail.split(marker, 1)[0]
            token = head.rsplit(" ", 1)[-1]
            try:
                return abs(int(token))
            except ValueError:  # pragma: no cover - the format is ours
                continue
    return None


def resolve_bounded(
    exceptions: Sequence[ReconException], bounds: AutoResolutionBounds
) -> ResolutionOutcome:
    """Apply the three bounded rules over a queue, in queue order.

    ``bounds.enabled=False`` makes every exception proposal-only -- the whole
    resolver is switched off rather than quietly narrowed, so a run can be shown
    with and without it and the difference attributed.

    ``UNMATCHABLE`` is never resolvable, and that is enforced twice: it is not
    in :data:`AGENT_RESOLVABLE_CLASSES`, and
    :class:`~ledgerloop.models.recon_exception.ReconException` refuses to be
    constructed claiming otherwise.
    """
    if not bounds.enabled:
        return ResolutionOutcome()

    resolutions: list[AutoResolution] = []
    spent = 0
    for exception in exceptions:
        if exception.exception_class not in AGENT_RESOLVABLE_CLASSES:
            continue
        if exception.exception_class is ExceptionClass.ROUNDING_DRIFT:
            resolution = _rounding(exception, bounds, spent)
            if resolution.applied:
                spent += resolution.amount_minor
        elif exception.exception_class is ExceptionClass.TIMING_SHIFT:
            resolution = _timing(exception, bounds)
        else:
            resolution = _duplicate(exception)
        resolutions.append(resolution)

    return ResolutionOutcome(
        resolutions=tuple(resolutions), rounding_spent_minor=spent
    )


def mark_resolvable(
    exceptions: Sequence[ReconException], outcome: ResolutionOutcome
) -> tuple[ReconException, ...]:
    """Stamp ``resolvable_by_agent`` on the exceptions a bounded rule accepted.

    A copy rather than a mutation: :class:`ReconException` is frozen because the
    queue is an artefact of a run, and re-deriving the flag at render time would
    let a report disagree with the resolution log it is printed beside.
    """
    resolved = outcome.resolved_ids
    return tuple(
        exception.model_copy(update={"resolvable_by_agent": True})
        if exception.exception_id in resolved
        else exception
        for exception in exceptions
    )


__all__ += ["mark_resolvable"]
