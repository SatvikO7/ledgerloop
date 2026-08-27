"""The exception object -- the actual deliverable of a reconciliation run.

Named :class:`ReconException`, not ``Exception``. PLAN.md §8.1 called it
``Exception``, which shadows the Python builtin inside a module that also
raises errors; the resulting bugs are subtle and permanent.

Note this is a **data record, not a raised error**. It does not inherit from
``BaseException`` and is never raised. It describes money the system could not
account for, which is a business finding rather than a program fault.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.candidates import Evidence
from ledgerloop.models.enums import ExceptionClass, ProseSource, Severity
from ledgerloop.models.refs import RecordRef

__all__ = ["Hypothesis", "ReconException"]


class Hypothesis(FrozenLedgerModel):
    """One competing explanation, preserved rather than collapsed.

    PLAN.md §8.2.4: when two subsets both satisfy a bank credit within
    tolerance, showing both with their probabilities is the honest answer.
    Silently picking the higher-scoring one is the dishonesty the track brief
    warns against, and it is invisible in aggregate metrics -- which is exactly
    why it has to be structural.
    """

    summary: str
    probability: float = Field(ge=0.0, le=1.0)
    implied_refs: tuple[RecordRef, ...] = ()
    evidence: tuple[Evidence, ...] = ()


class ReconException(FrozenLedgerModel):
    """An item the system could not resolve, with everything a human needs.

    The bar from PLAN.md §8.2.2: a bare "unmatched" count is not a deliverable.
    Every instance carries a class, a root cause, an evidence chain, a rupee
    impact and a suggested action.
    """

    exception_id: str
    exception_class: ExceptionClass
    severity: Severity
    impact_minor: MinorUnits = Field(
        ge=0,
        description="Money at stake, as a magnitude. The queue's sort key -- a "
        "controller cares about one ₹4L exception, not two hundred one-paise "
        "drifts. Non-negative because a negative sort key would put the worst "
        "item last: a settlement can declare a negative net (claw-backs "
        "exceeding its payments) and the amount at risk is still what is at "
        "risk. Found by a property test, not by inspection.",
    )
    involved_refs: tuple[RecordRef, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()

    root_cause: str = Field(description="Plain English, grounded in the evidence chain.")
    root_cause_source: ProseSource = ProseSource.TEMPLATE
    suggested_action: str = Field(
        description="What a human should do next, e.g. 'Request chargeback detail "
        "for SETL-0091'."
    )
    suggested_action_source: ProseSource = ProseSource.TEMPLATE

    classification_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated confidence in the CLASS assignment, not in any match.",
    )
    resolvable_by_agent: bool = Field(
        default=False,
        description="Whether a bounded auto-resolution rule (PLAN.md §8.3) applies.",
    )
    hypotheses: tuple[Hypothesis, ...] = Field(
        default=(),
        description="Populated when more than one explanation fits. Ordered by "
        "descending probability.",
    )

    @model_validator(mode="after")
    def _ambiguity_is_structural(self) -> ReconException:
        if self.exception_class is ExceptionClass.AMBIGUOUS_AGGREGATION and (
            len(self.hypotheses) < 2
        ):
            raise ValueError(
                "AMBIGUOUS_AGGREGATION must carry at least two competing hypotheses; "
                "an ambiguity with one explanation is not an ambiguity"
            )
        if self.hypotheses:
            probabilities = [h.probability for h in self.hypotheses]
            if probabilities != sorted(probabilities, reverse=True):
                raise ValueError("hypotheses must be ordered by descending probability")
        return self

    @model_validator(mode="after")
    def _unmatchable_is_never_agent_resolvable(self) -> ReconException:
        """The honest floor cannot be auto-resolved.

        ``UNMATCHABLE`` means the data to resolve it does not exist in the three
        sources. An agent claiming it can resolve one is asserting information
        it does not have.
        """
        if self.exception_class is ExceptionClass.UNMATCHABLE and self.resolvable_by_agent:
            raise ValueError("UNMATCHABLE exceptions are irreconcilable by construction")
        return self
