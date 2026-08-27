"""What the model is allowed to say, as types.

PLAN.md 7.4: "All LLM output through Pydantic strict schemas; validation
failure -> one retry with the error appended -> then fall through to exception.
Never a crash, never a silent default."

Every model here sets ``extra="forbid"`` through
:class:`~ledgerloop.models.base.FrozenLedgerModel`. That is not tidiness: a
hallucinated field silently ignored is the difference between "the model
returned an unexpected key" and "the model returned a key we thought we were
reading". The first is caught here; the second is a bug that survives to
production.

WHAT THESE SCHEMAS DELIBERATELY CANNOT EXPRESS
----------------------------------------------
The shape of a contract is a statement about authority, so the omissions matter
more than the fields:

* **No amount fields anywhere.** The model never states a rupee figure. Money
  is re-derived from the sources by
  :func:`~ledgerloop.matching.verify.verify_arithmetic`, and a proposal that
  does not reconcile is refused whatever the model said about it.
* **No exception class, severity or impact.** Step 8's classifier owns those
  and is deterministic. The model may rewrite the *prose* on an exception that
  already exists, and :attr:`~ledgerloop.models.recon_exception.ReconException.
  root_cause_source` records that it did.
* **No decision.** ``ResidualHypothesis`` carries a *proposal* and a
  self-reported confidence, and that confidence is a **feature** -- it enters
  the blender's design matrix beside every other signal and is calibrated with
  them, because raw LLM confidence is systematically overconfident (PLAN.md
  7.4).
* **No free-form references.** Every id the model returns is checked against
  the pack it was given (:mod:`ledgerloop.llm.gates`). A reference to a record
  that was not in the prompt is a hallucination, not a discovery.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel

__all__ = [
    "AdjudicationBatch",
    "ExceptionExplanation",
    "ExplanationBatch",
    "NarrationBatch",
    "NarrationExtraction",
    "ProposedLink",
    "ResidualHypothesis",
]


class NarrationExtraction(FrozenLedgerModel):
    """One narration the regex layer could not resolve, read by the model.

    ``item_id`` is the index the prompt used, not a record id: the narration is
    free text and the model never sees which bank row it belongs to, so it
    cannot invent a link even by accident.
    """

    item_id: str = Field(description="Echo of the id the prompt supplied.")
    utr: str | None = Field(
        default=None,
        description="Reference the model believes is in the text. Checked against "
        "the narration itself before it is used -- a UTR that is not in the "
        "string is a hallucination, however plausible it looks.",
    )
    merchant: str | None = Field(
        default=None,
        description="Counterparty name the model believes is in the text. Checked "
        "the same way.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NarrationBatch(FrozenLedgerModel):
    """The whole response to one ``parse_narration`` call."""

    extractions: tuple[NarrationExtraction, ...] = ()


class ProposedLink(FrozenLedgerModel):
    """A link the model thinks exists. Never a link the system asserts.

    Endpoint ids only. No amount, no probability of its own, and no claim about
    what else the link implies -- everything downstream is re-derived from the
    sources by deterministic code.
    """

    payment_id: str
    bank_txn_id: str
    settlement_id: str | None = None
    payment_ids: tuple[str, ...] = Field(
        default=(),
        description="For an aggregation proposal: every payment the model thinks "
        "travelled in this credit. Verified by re-deriving the credit from them.",
    )


class ResidualHypothesis(FrozenLedgerModel):
    """The model's account of one residual item (PLAN.md 7.3, call site 2)."""

    item_id: str
    hypothesis: str = Field(
        max_length=600, description="One paragraph, grounded in the evidence pack."
    )
    proposed_link: ProposedLink | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=1200)
    evidence_refs: tuple[str, ...] = Field(
        default=(),
        description="Record keys the model is reasoning from. Every one is checked "
        "against the pack it was given.",
    )

    @model_validator(mode="after")
    def _a_link_needs_two_ends(self) -> ResidualHypothesis:
        """A proposal naming one end is not a proposal, it is a guess with a gap.

        Caught here rather than downstream because the verification gate would
        otherwise have to decide what a half-specified link means, and the
        honest answer is that it does not mean anything.
        """
        link = self.proposed_link
        if link is not None and not (link.payment_id and link.bank_txn_id):
            raise ValueError("a proposed link must name both endpoints")
        return self


class AdjudicationBatch(FrozenLedgerModel):
    """The whole response to one ``adjudicate_residual`` call."""

    hypotheses: tuple[ResidualHypothesis, ...] = ()


class ExceptionExplanation(FrozenLedgerModel):
    """Prose for one exception the classifier already built and priced."""

    exception_id: str
    root_cause: str = Field(min_length=1, max_length=800)
    suggested_action: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def _prose_is_not_a_restatement_of_the_id(self) -> ExceptionExplanation:
        if self.root_cause.strip() == self.suggested_action.strip():
            raise ValueError(
                "root cause and suggested action must say different things; a "
                "diagnosis repeated as an instruction is neither"
            )
        return self


class ExplanationBatch(FrozenLedgerModel):
    """The whole response to one ``explain_exception`` call."""

    explanations: tuple[ExceptionExplanation, ...] = ()
