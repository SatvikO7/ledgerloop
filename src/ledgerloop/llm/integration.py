"""Where the three call sites attach to a run, and what they may change.

Composition, deliberately outside :mod:`ledgerloop.matching` and
:mod:`ledgerloop.exceptions`. Neither of those packages imports this one, so
neither can develop a dependency on a model being present -- which is what keeps
``--no-llm`` a first-class path rather than a degraded one.

THE THREE ATTACHMENT POINTS
---------------------------
``repair_narrations``
    **Before matching.** A narration the regex layer could not read is re-read,
    and an accepted repair is written onto the bank row exactly as the regex
    layer would have written it. Everything downstream then treats it as an
    ordinary parsed field -- there is no "LLM-derived" branch in the matcher,
    because a reference is a reference once it has been checked against the
    text it came from.

``adjudicator_for``
    **Inside the ladder, as T5.** Returns a callable the pipeline invokes after
    T2/T3/T4 have settled. Its candidates go through the same decision policy,
    the same arithmetic invariant on ``MatchDecision``, and -- when a bundle is
    fitted -- the same blender. T5 proposes; nothing about it decides.

``explain_queue``
    **After classification.** Rewrites prose on exceptions that already have a
    class, a severity and a rupee figure. Those three are never sent back for
    revision.

WHAT AN ACCEPTED REPAIR CANNOT DO
---------------------------------
A repaired narration can only add a reference or a merchant name to a row that
had none. It cannot overwrite one the regex layer read, cannot alter an amount
or a date, and cannot mark a debit as a credit. The first of those is the one
worth stating: the deterministic parser is more reliable than the model on the
narrations it *can* read, so a model contradicting it is a model to be ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ledgerloop.config import RunConfig
from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.prompts import EvidencePack
from ledgerloop.llm.tasks import (
    AdjudicationOutcome,
    ExplanationOutcome,
    NarrationOutcome,
    adjudicate_residual,
    evidence_pack_for,
    explain_exceptions,
    parse_narrations,
)
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.pipeline import ResidualAdjudicator
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.money import format_minor

__all__ = [
    "LLMRunSummary",
    "adjudicator_for",
    "explain_queue",
    "repair_narrations",
    "residual_packs",
]


@dataclass(frozen=True)
class LLMRunSummary:
    """What the model contributed to one run, and what it was refused.

    Both halves are reported. "The model helped" is only a claim worth making
    if the same report says how often it was overruled.
    """

    narration: NarrationOutcome
    adjudication: AdjudicationOutcome
    explanation: ExplanationOutcome

    @property
    def accepted(self) -> int:
        return (
            self.narration.accepted
            + self.adjudication.accepted
            + self.explanation.accepted
        )

    @property
    def rejected_ungrounded(self) -> int:
        return (
            self.narration.rejected_ungrounded
            + self.adjudication.rejected_ungrounded
            + self.explanation.rejected_ungrounded
        )

    @property
    def rejected_unverified(self) -> int:
        return self.adjudication.rejected_unverified

    @property
    def calls_refused(self) -> int:
        return (
            self.narration.calls_refused
            + self.adjudication.calls_refused
            + self.explanation.calls_refused
        )


def repair_narrations(
    client: LLMClient, ingest: IngestResult, *, batch_size: int = 20
) -> tuple[IngestResult, NarrationOutcome]:
    """Re-read the narrations the regex layer missed, and apply what survives.

    Only credits with **neither** a reference nor a merchant are sent: a row the
    deterministic parser resolved is not a row a model can improve, and sending
    it would spend budget to be told what is already known.
    """
    unresolved = [
        (txn.txn_id, txn.narration_raw)
        for txn in ingest.bank_txns
        if txn.is_credit
        and txn.extracted_utr is None
        and txn.extracted_merchant is None
    ]
    outcome = parse_narrations(client, unresolved, batch_size=batch_size)
    if not outcome.repairs:
        return ingest, outcome

    repairs = outcome.by_id
    rows = []
    for txn in ingest.bank_txns:
        repair = repairs.get(txn.txn_id)
        if repair is None or not txn.is_credit:
            rows.append(txn)
            continue
        rows.append(
            txn.model_copy(
                update={
                    # Only ever fills a gap. A reference the regex layer read is
                    # never overwritten -- see the module docstring.
                    "extracted_utr": txn.extracted_utr or repair.utr,
                    "extracted_merchant": txn.extracted_merchant or repair.merchant,
                }
            )
        )
    return replace(ingest, bank_txns=tuple(rows)), outcome


def residual_packs(
    context: MatchContext,
    established: Sequence[MatchCandidate],
    *,
    limit: int = 20,
) -> tuple[EvidencePack, ...]:
    """Compact packs for the settlements the ladder could not credit.

    Bounded by ``limit`` and taken largest-payout-first, because the budget is
    finite and the money is the ranking every other queue in this system uses.

    Each pack carries only the records of *that* item. Every reference the model
    returns is checked against this list, so a narrow pack is a tight gate as
    well as a cheap prompt.
    """
    matched = {
        candidate.source_ref.record_id
        for candidate in established
        if candidate.arithmetic_verified
    }
    open_settlements = [
        view
        for view in context.settlements
        if view.settlement_id not in matched and view.payments
    ]
    open_settlements.sort(key=lambda view: (-view.net_minor, view.settlement_id))

    claimed = {
        candidate.target_ref.record_id
        for candidate in established
        if candidate.arithmetic_verified
    }
    unclaimed = [txn for txn in context.credits if txn.txn_id not in claimed]

    packs: list[EvidencePack] = []
    for view in open_settlements[:limit]:
        refs = [
            settlement_ref(view.settlement_id).key,
            *(payment_ref(p.payment_id).key for p in view.payments),
        ]
        near = [
            txn
            for txn in unclaimed
            if abs(txn.credit_minor - view.net_minor) <= max(view.net_minor // 20, 500)
        ][:4]
        refs.extend(bank_ref(txn.txn_id).key for txn in near)
        packs.append(
            evidence_pack_for(
                view.settlement_id,
                summary=(
                    f"payout of {format_minor(view.net_minor)} declared on "
                    f"{view.settlement.settled_on.isoformat()} with "
                    f"{len(view.payments)} payment(s), not credited"
                ),
                refs=refs,
                facts=(
                    f"reference published: {view.utr or 'none'}",
                    f"declared gross {format_minor(view.settlement.gross_minor)}, "
                    f"fee {format_minor(view.settlement.fee_minor)}, "
                    f"adjustments {format_minor(view.settlement.adjustments_minor)}",
                ),
                candidates=tuple(
                    f"{txn.txn_id} credits {format_minor(txn.credit_minor)} on "
                    f"{txn.value_date.isoformat()}, narration "
                    f"{txn.narration_raw!r}"
                    for txn in near
                ),
            )
        )
    return tuple(packs)


def adjudicator_for(
    client: LLMClient, config: RunConfig, *, limit: int = 20
) -> ResidualAdjudicator | None:
    """The T5 callable the pipeline invokes, or ``None`` when there is no model.

    ``None`` rather than a no-op: the pipeline reports a tier row only for tiers
    that ran, and a T5 row of zeros on a ``--no-llm`` run would be a false
    measurement of an unbuilt contribution -- the same rule Step 6 applied to
    T4 and Step 2 applied to the pending baselines.
    """
    if not client.enabled:
        return None

    def adjudicate(
        context: MatchContext, established: Sequence[MatchCandidate]
    ) -> tuple[Sequence[MatchCandidate], object]:
        packs = residual_packs(context, established, limit=limit)
        outcome = adjudicate_residual(client, context, packs, config)
        return outcome.candidates, outcome

    return adjudicate


def explain_queue(
    client: LLMClient, exceptions: Sequence[ReconException], *, batch_size: int = 8
) -> tuple[tuple[ReconException, ...], ExplanationOutcome]:
    """Better prose on a finished queue. The class and the money are untouched."""
    outcome = explain_exceptions(client, exceptions, batch_size=batch_size)
    return outcome.exceptions, outcome
