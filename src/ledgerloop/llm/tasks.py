"""The three call sites, each with its deterministic fallback beside it.

PLAN.md 7.2 allows exactly three, and each has the same shape:

    deterministic result -> is there a residue? -> ask -> validate -> gate ->
    accept or keep the deterministic result

The last arrow is the important one. Every function here returns something
useful when the model is disabled, unreachable, over budget, malformed or
caught inventing a reference. ``--no-llm`` is not a special path through this
module; it is the same path with the first branch taken, which is why the whole
system stays runnable without a key.

WHAT EACH SITE MAY AND MAY NOT DO
---------------------------------
+---------------------+--------------------------------+------------------------+
| site                | may                            | may never              |
+=====================+================================+========================+
| ``parse_narration`` | read a UTR / merchant out of   | invent one; touch any  |
|                     | free text the regex missed     | amount or link         |
+---------------------+--------------------------------+------------------------+
| ``adjudicate_``     | propose a link and explain     | decide it; do the      |
| ``residual``        | its reasoning                  | arithmetic; cite       |
|                     |                                | records it was not     |
|                     |                                | given                  |
+---------------------+--------------------------------+------------------------+
| ``explain_``        | rewrite the prose on an        | change the class, the  |
| ``exception``       | exception that already exists  | severity, the money,   |
|                     |                                | or name a record the   |
|                     |                                | exception does not     |
|                     |                                | involve                |
+---------------------+--------------------------------+------------------------+

Every acceptance is counted and every refusal is counted, because "the model
helped" is only a claim worth making if the report can say how often it did and
how often it was overruled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ledgerloop.config import DecisionThresholds, RunConfig
from ledgerloop.llm.client import (
    LLMClient,
    LLMError,
    batched,
)
from ledgerloop.llm.contracts import (
    AdjudicationBatch,
    ExplanationBatch,
    NarrationBatch,
    ResidualHypothesis,
)
from ledgerloop.llm.gates import (
    grounded_in_text,
    grounded_refs,
    prose_names_only_known_records,
)
from ledgerloop.llm.prompts import (
    ADJUDICATION_VERSION,
    EXPLANATION_VERSION,
    NARRATION_VERSION,
    EvidencePack,
    render_adjudication,
    render_explanation,
    render_narration,
)
from ledgerloop.matching.bank_leg import candidate_id
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.verify import ArithmeticCheck, verify_arithmetic
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, ProseSource, Tier
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.money import delta_ratio, format_minor

__all__ = [
    "AdjudicationOutcome",
    "ExplanationOutcome",
    "NarrationOutcome",
    "NarrationRepair",
    "adjudicate_residual",
    "evidence_pack_for",
    "explain_exceptions",
    "parse_narrations",
    "unmeasured_probability",
]


@dataclass(frozen=True)
class NarrationRepair:
    """One narration the model resolved and the gates accepted."""

    item_id: str
    utr: str | None
    merchant: str | None
    confidence: float


@dataclass
class LLMTaskCounters:
    """How a call site went. Every field appears in the report."""

    attempted: int = 0
    accepted: int = 0
    rejected_ungrounded: int = 0
    rejected_unverified: int = 0
    calls_refused: int = 0
    failures: tuple[str, ...] = ()

    def failed(self, reason: str) -> None:
        self.calls_refused += 1
        self.failures = (*self.failures, reason)


@dataclass
class NarrationOutcome(LLMTaskCounters):
    repairs: tuple[NarrationRepair, ...] = ()

    @property
    def by_id(self) -> dict[str, NarrationRepair]:
        return {repair.item_id: repair for repair in self.repairs}


@dataclass
class AdjudicationOutcome(LLMTaskCounters):
    candidates: tuple[MatchCandidate, ...] = ()
    hypotheses: tuple[ResidualHypothesis, ...] = ()
    demoted: int = 0
    """Proposals whose arithmetic did not close.

    Demoted rather than dropped -- PLAN.md 7.4. "The model suggested this and
    the arithmetic disagrees" is information a controller wants, and the
    candidate carries the failure as evidence so the decision policy can route
    it to review.
    """


@dataclass
class ExplanationOutcome(LLMTaskCounters):
    exceptions: tuple[ReconException, ...] = ()
    rewritten: int = 0


def parse_narrations(
    client: LLMClient,
    items: Sequence[tuple[str, str]],
    *,
    batch_size: int = 20,
) -> NarrationOutcome:
    """Call site 1. Read the narrations the regex layer could not.

    ``items`` is ``(item_id, narration)`` for **regex misses only**. Feeding it
    everything would work and would cost twenty times as much for nothing --
    the deterministic parser resolves most narrations, and PLAN.md 7.3's budget
    depends on that being true.
    """
    outcome = NarrationOutcome()
    if not items or not client.enabled:
        return outcome

    repairs: list[NarrationRepair] = []
    sources = dict(items)
    for batch in batched(list(items), batch_size):
        outcome.attempted += len(batch)
        try:
            parsed, _ = client.complete_json(
                render_narration(list(batch)),
                NarrationBatch,
                prompt_version=NARRATION_VERSION,
            )
        except LLMError as exc:
            outcome.failed(str(exc))
            continue

        for extraction in parsed.extractions:
            source = sources.get(extraction.item_id)
            if source is None:
                outcome.rejected_ungrounded += 1
                continue
            utr_gate = grounded_in_text(extraction.utr, source)
            merchant_gate = grounded_in_text(extraction.merchant, source)
            if not utr_gate or not merchant_gate:
                # A UTR is a join key. An invented one would create a match out
                # of nothing, which is the single most expensive failure this
                # system can make -- so a partial hallucination discards the
                # whole extraction rather than keeping the half that passed.
                outcome.rejected_ungrounded += 1
                continue
            if extraction.utr is None and extraction.merchant is None:
                continue  # an honest "I could not find one"
            repairs.append(
                NarrationRepair(
                    item_id=extraction.item_id,
                    utr=extraction.utr,
                    merchant=extraction.merchant,
                    confidence=extraction.confidence,
                )
            )
            outcome.accepted += 1

    outcome.repairs = tuple(repairs)
    return outcome


def evidence_pack_for(
    item_id: str,
    *,
    summary: str,
    refs: Sequence[str],
    facts: Sequence[str] = (),
    candidates: Sequence[str] = (),
) -> EvidencePack:
    """Assemble one pack. Sorted refs, so the prompt -- and the cache key -- is stable."""
    return EvidencePack(
        item_id=item_id,
        summary=summary,
        refs=tuple(sorted(set(refs))),
        facts=tuple(facts),
        candidates=tuple(candidates),
    )


def unmeasured_probability(thresholds: DecisionThresholds) -> float:
    """The probability an LLM proposal carries when nothing has measured it.

    The middle of the review band, and the reasoning is worth stating because
    it is the crux of the whole tier:

    * The model's own confidence must **never** become the probability. It is
      systematically overconfident (PLAN.md 7.4), so a proposal asserting 0.99
      would auto-match itself -- which is precisely "the LLM decides a match by
      itself", the one thing PLAN.md 7.1 forbids.
    * A calibrated probability would be legitimate, and that is what happens
      when a fitted bundle covers T5: the blender scores it and the isotonic
      prices it against measured outcomes like any other tier's.
    * Absent that, the honest value is one that says *unmeasured*. The middle of
      the review band is below any ``tau_high`` and above any ``tau_low``, so an
      arithmetic-verified proposal reaches a human and never a ledger.

    A degenerate band (``tau_low == tau_high``, which a fitted threshold can
    produce) collapses to ``tau_low`` -- routing to an exception rather than
    inventing head-room that the configuration does not have.

    So does a band too narrow for a float to sit inside. With
    ``tau_low = 0.0`` and ``tau_high = 5e-324`` there is no representable value
    strictly between them, and the midpoint rounds down onto ``tau_low``. That
    is not a case any fitted threshold produces, and it is handled rather than
    excluded because the alternative is a function whose stated guarantee --
    *strictly below any tau_high* -- is true only for the inputs someone thought
    to test. A property test found it; the fix is one comparison, and it
    collapses in the same safe direction the degenerate case does.
    """
    if thresholds.tau_low >= thresholds.tau_high:
        return thresholds.tau_low
    midpoint = (thresholds.tau_low + thresholds.tau_high) / 2.0
    if not thresholds.tau_low < midpoint < thresholds.tau_high:
        return thresholds.tau_low
    return midpoint


def _adjudication_candidate(
    hypothesis: ResidualHypothesis,
    check: ArithmeticCheck,
    *,
    tolerance_band_minor: int,
    provisional_p: float,
) -> MatchCandidate:
    """A T5 candidate carrying the model's confidence as a **feature**.

    Note where the confidence lands: ``FeatureVector.llm_confidence``, not
    ``calibrated_p``. Raw self-reported confidence is systematically
    overconfident (PLAN.md 7.4), so it is one input to the blender beside the
    amount delta and the date gap, and the calibrator prices it against measured
    outcomes.

    ``calibrated_p`` is the *unmeasured* value from
    :func:`unmeasured_probability`, which a fitted bundle overwrites and an
    unfitted one leaves alone -- so a T5 proposal on a corpus whose calibrator
    never saw T5 goes to review by construction rather than by policy.
    """
    credit = check.credit
    assert credit is not None  # a check that reached here has a credit
    payment = check.payments[0]
    residual = check.residual_minor
    return MatchCandidate(
        candidate_id=candidate_id(
            Tier.T5_LLM,
            LinkType.PAYMENT_CREDITED_AS,
            payment_ref(payment.payment_id).key,
            bank_ref(credit.txn_id).key,
        ),
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref(payment.payment_id),
        target_ref=bank_ref(credit.txn_id),
        tier=Tier.T5_LLM,
        features=FeatureVector(
            tier=Tier.T5_LLM,
            amount_delta_minor=residual,
            tolerance_band_minor=tolerance_band_minor,
            amount_delta_ratio=delta_ratio(residual, credit.credit_minor),
            llm_confidence=hypothesis.confidence,
        ),
        evidence=(
            Evidence(
                kind=EvidenceKind.LLM_HYPOTHESIS,
                detail=hypothesis.hypothesis,
                refs=(payment_ref(payment.payment_id), bank_ref(credit.txn_id)),
                score=hypothesis.confidence,
            ),
            Evidence(
                kind=(
                    EvidenceKind.ARITHMETIC_CHECK
                    if check.verified
                    else EvidenceKind.NEGATIVE_EVIDENCE
                ),
                detail=check.reason,
                refs=(
                    (settlement_ref(check.settlement_id),)
                    if check.settlement_id
                    else ()
                ),
                amount_minor=check.credit_minor,
            ),
        ),
        calibrated_p=provisional_p,
        arithmetic_verified=check.verified,
    )


def adjudicate_residual(
    client: LLMClient,
    context: MatchContext,
    packs: Sequence[EvidencePack],
    config: RunConfig,
    *,
    batch_size: int | None = None,
) -> AdjudicationOutcome:
    """Call site 2. A hypothesis per residual item, and a verified link at most.

    Three gates stand between the model and a candidate, in this order:

    1. **The schema.** A malformed answer is retried once and then abandoned.
    2. **The evidence gate.** Every id cited must have been in that item's pack.
       A hypothesis citing anything else is discarded whole -- partial trust in
       an answer that invented a record is not a defensible position.
    3. **The arithmetic gate.** ``verify_arithmetic`` re-derives the money from
       the sources. A proposal that fails is **demoted, not dropped**: it
       becomes a candidate with ``arithmetic_verified=False``, which the policy
       routes to review and which the blender is forbidden to re-score.

    Nothing here decides anything. A verified proposal becomes a T5
    *candidate*; the decision policy rules on it exactly as it rules on T2's.
    """
    outcome = AdjudicationOutcome()
    if not packs or not client.enabled:
        return outcome

    size = batch_size or config.llm.adjudication_batch_size
    accepted: list[MatchCandidate] = []
    kept: list[ResidualHypothesis] = []
    by_id = {pack.item_id: pack for pack in packs}

    for batch in batched(list(packs), size):
        outcome.attempted += len(batch)
        try:
            parsed, _ = client.complete_json(
                render_adjudication(list(batch)),
                AdjudicationBatch,
                prompt_version=ADJUDICATION_VERSION,
            )
        except LLMError as exc:
            outcome.failed(str(exc))
            continue

        for hypothesis in parsed.hypotheses:
            pack = by_id.get(hypothesis.item_id)
            if pack is None:
                outcome.rejected_ungrounded += 1
                continue
            gate = grounded_refs(hypothesis.evidence_refs, pack.refs)
            if not gate:
                outcome.rejected_ungrounded += 1
                continue
            kept.append(hypothesis)
            link = hypothesis.proposed_link
            if link is None:
                continue

            payment_ids = link.payment_ids or (link.payment_id,)
            check = verify_arithmetic(
                context,
                payment_ids=tuple(payment_ids),
                bank_txn_id=link.bank_txn_id,
                epsilon_minor=config.tolerances.aggregation_epsilon_minor,
                settlement_id=link.settlement_id,
            )
            if check.credit is None or not check.payments:
                # The proposal named records that do not exist. There is nothing
                # to demote -- a candidate needs two real endpoints.
                outcome.rejected_unverified += 1
                continue
            candidate = _adjudication_candidate(
                hypothesis,
                check,
                tolerance_band_minor=config.tolerances.aggregation_epsilon_minor,
                provisional_p=unmeasured_probability(config.thresholds),
            )
            accepted.append(candidate)
            if check.verified:
                outcome.accepted += 1
            else:
                outcome.demoted += 1
                outcome.rejected_unverified += 1

    outcome.candidates = tuple(accepted)
    outcome.hypotheses = tuple(kept)
    return outcome


def explain_exceptions(
    client: LLMClient,
    exceptions: Sequence[ReconException],
    *,
    batch_size: int = 8,
) -> ExplanationOutcome:
    """Call site 3. Better prose on exceptions that are already complete.

    Batched **by class**, as PLAN.md 7.3 specifies: one call per exception
    cluster rather than one per exception. Items of a class share their shape,
    so the model writes better prose seeing several together and the call count
    stays inside the budget.

    An exception whose rewrite is rejected keeps its template prose and its
    ``ProseSource.TEMPLATE`` marker. The queue never has a hole in it, and the
    report can state exactly how many rows the model actually improved.
    """
    outcome = ExplanationOutcome(exceptions=tuple(exceptions))
    if not exceptions or not client.enabled:
        return outcome

    by_class: dict[str, list[ReconException]] = {}
    for exception in exceptions:
        by_class.setdefault(exception.exception_class.value, []).append(exception)

    rewritten: dict[str, ReconException] = {}
    for _, group in sorted(by_class.items()):
        for batch in batched(group, batch_size):
            outcome.attempted += len(batch)
            items = [
                (
                    item.exception_id,
                    f"{item.exception_class.value}, {item.severity.value}, "
                    f"{format_minor(item.impact_minor)} at stake, subject "
                    f"{item.involved_refs[0].record_id}",
                    [evidence.detail for evidence in item.evidence[:4]],
                )
                for item in batch
            ]
            try:
                parsed, _ = client.complete_json(
                    render_explanation(items),
                    ExplanationBatch,
                    prompt_version=EXPLANATION_VERSION,
                )
            except LLMError as exc:
                outcome.failed(str(exc))
                continue

            known = {item.exception_id: item for item in batch}
            for explanation in parsed.explanations:
                original = known.get(explanation.exception_id)
                if original is None:
                    outcome.rejected_ungrounded += 1
                    continue
                cause_gate = prose_names_only_known_records(
                    explanation.root_cause, original.involved_refs
                )
                action_gate = prose_names_only_known_records(
                    explanation.suggested_action, original.involved_refs
                )
                if not cause_gate or not action_gate:
                    outcome.rejected_ungrounded += 1
                    continue
                rewritten[original.exception_id] = original.model_copy(
                    update={
                        "root_cause": explanation.root_cause,
                        "root_cause_source": ProseSource.LLM,
                        "suggested_action": explanation.suggested_action,
                        "suggested_action_source": ProseSource.LLM,
                    }
                )
                outcome.accepted += 1

    outcome.rewritten = len(rewritten)
    outcome.exceptions = tuple(
        rewritten.get(exception.exception_id, exception) for exception in exceptions
    )
    return outcome
