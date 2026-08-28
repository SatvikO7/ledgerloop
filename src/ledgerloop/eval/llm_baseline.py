"""B2 -- the LLM-only baseline. The "why not just an LLM" answer.

PLAN.md §9.2's third row: *dump all records, ask it to reconcile*. No tiers, no
subset-sum solver, no calibration, no ``verify_arithmetic``. Whatever the model
returns **is** the answer, and the answer is scored against exactly the same
ground truth as the deterministic ladder.

WHY THE ABSENCE OF THE GATES IS THE POINT
-----------------------------------------
Every safeguard Step 9 built exists somewhere in this file's negative space:

* a returned link is asserted, not proposed -- there is no decision policy;
* the money is never re-derived -- there is no arithmetic gate;
* a record id the model invented is asserted anyway -- there is no grounding
  gate, only a counter recording how often it happened;
* the model's self-reported confidence is ignored rather than calibrated,
  because a baseline with no calibration set has nothing to calibrate against.

Removing any one of those from the production system would be a defect. Here
they are absent *by construction*, which is what makes the comparison an
argument for them rather than an assertion about them.

THIS FILE CANNOT REACH THE PRODUCTION SYSTEM
--------------------------------------------
:mod:`ledgerloop.matching` does not import :mod:`ledgerloop.eval`, so nothing
here can enter the ladder, the decision policy or the calibrator. B2's
predictions go to :func:`~ledgerloop.eval.metrics.evaluate` and to the report,
and nowhere else. There is a test asserting the import direction and a test
asserting that a B2 run leaves the production run's predictions unchanged.

WHY DEV AND NOT TEST
--------------------
PLAN.md §9.2 says so, and the reason is quota: B2 sends the corpus rather than
its residual, so the token cost scales with the whole dataset. Sixty orders is
enough to demonstrate that an unverified model is less precise than a verified
one; three hundred would spend a meaningful slice of a day's free tier to
demonstrate it again. The reduced scope is printed in ``EVALUATION.md`` beside
the row rather than left for a reader to infer.

WHAT IS REPORTED WHEN NO MODEL IS REACHABLE
--------------------------------------------
``ran=False`` and nothing else. Not a precision of zero, not an empty link set
scored as a perfect abstention -- the same rule the rest of the report follows:
a zero for something that did not run is a false measurement.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import Field

from ledgerloop.eval.artifacts import LLMBaselineArtifact
from ledgerloop.eval.metrics import PredictedLink
from ledgerloop.eval.truth_io import DatasetManifest
from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.llm.client import LLMClient, LLMError, batched
from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.records import CanonicalBankTxn, CanonicalPayment
from ledgerloop.models.refs import bank_ref, payment_ref
from ledgerloop.money import format_minor

__all__ = [
    "B2_DESCRIPTION",
    "B2_NAME",
    "B2_PROMPT_VERSION",
    "DEFAULT_PAYMENTS_PER_CALL",
    "LLMBaselineArtifact",
    "LLMOnlyBatch",
    "LLMOnlyLink",
    "render_reconciliation",
    "run_b2",
]

B2_NAME = "B2"
B2_DESCRIPTION = "LLM-only -- the whole corpus in the prompt, the model's answer asserted"
B2_PROMPT_VERSION = "reconcile-all/1.0.0"

#: Payments per call. The bank statement goes into **every** prompt, because a
#: payment can be credited by any row; only the payment side is chunked.
#:
#: Twelve rather than sixty: a single prompt carrying every payment would be
#: cheaper per token and would also be the least favourable way to ask, and a
#: baseline built to lose proves nothing. Smaller batches give the model a
#: tractable question, and the extra prompt tokens they cost are exactly the
#: cost the comparison is measuring.
DEFAULT_PAYMENTS_PER_CALL = 12


class LLMOnlyLink(FrozenLedgerModel):
    """One link the model asserts. No amount -- and nothing re-derives one.

    ``ProposedLink`` in :mod:`ledgerloop.llm.contracts` is the production
    equivalent and is deliberately *not* reused: that one is a proposal headed
    for ``verify_arithmetic``, and giving it a second life as an assertion would
    blur the line this baseline exists to draw.
    """

    payment_id: str
    bank_txn_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMOnlyBatch(FrozenLedgerModel):
    """The whole response to one B2 call."""

    links: tuple[LLMOnlyLink, ...] = ()


def _payment_line(payment: CanonicalPayment) -> str:
    return (
        f"{payment.payment_id}: {format_minor(payment.amount_minor)} captured "
        f"{payment.captured_at.date().isoformat()}, "
        f"settlement {payment.settlement_id or 'none'}, "
        f"order {payment.order_ref_raw or 'none'}"
    )


def _credit_line(txn: CanonicalBankTxn) -> str:
    return (
        f"{txn.txn_id}: {format_minor(txn.credit_minor)} on "
        f"{txn.value_date.isoformat()}, narration {txn.narration_raw!r}"
    )


def render_reconciliation(
    payments: Sequence[CanonicalPayment], credits: Sequence[CanonicalBankTxn]
) -> str:
    """The B2 prompt: a slice of the payments, the whole bank statement, no rules.

    Deliberately *not* the production prompt. The three in
    :mod:`ledgerloop.llm.prompts` each state a boundary -- do not calculate,
    return null rather than guess, cite only what you were given -- because
    those boundaries are what the production gates enforce. B2 has no gates, so
    stating boundaries it will not enforce would be dishonest in the other
    direction: it would make the baseline look disciplined while its output was
    asserted unchecked.

    What it does say is what a reconciliation prompt says: here are the payments,
    here is the statement, tell me which credited which.
    """
    lines = [
        "You are reconciling a payment processor's settlement report against a "
        "bank statement. Decide which bank credit paid each payment.",
        "",
        "Payments:",
        *(_payment_line(payment) for payment in payments),
        "",
        "Bank credits:",
        *(_credit_line(txn) for txn in credits),
        "",
        "One bank credit usually carries many payments at once: several payments "
        "are batched into one settlement and paid out as a single transfer, net "
        "of fees. Return every (payment, bank credit) pair you believe is real, "
        "and omit a payment you cannot place.",
        "",
        "Reply with JSON only, matching the schema exactly.",
        "",
        'Schema: {"links": [{"payment_id": str, "bank_txn_id": str, '
        '"confidence": float}]}',
    ]
    return "\n".join(lines)


def run_b2(
    client: LLMClient,
    ingest: IngestResult,
    manifest: DatasetManifest,
    *,
    payments_per_call: int = DEFAULT_PAYMENTS_PER_CALL,
) -> tuple[tuple[PredictedLink, ...], LLMBaselineArtifact]:
    """Ask the model to reconcile the corpus, and assert whatever it says.

    Returns the predictions and a **partial** artefact: the counters and the
    cost ledger are filled in here, and the scored fields are attached by the
    caller once :func:`~ledgerloop.eval.metrics.evaluate` has run. Splitting it
    that way keeps this module free of any dependency on ground truth -- a
    baseline that could see the labels is not a baseline.

    ``ran=False`` with a reason when the client is disabled. No fabricated
    zeros: see the module docstring.
    """
    started_ns = time.perf_counter_ns()
    payments = [
        payment for payment in ingest.payments if payment.settlement_id is not None
    ]
    credits = [txn for txn in ingest.bank_txns if txn.is_credit]

    if not client.enabled:
        return (), LLMBaselineArtifact(
            ran=False,
            reason="no model was reachable (--no-llm, or no API key configured)",
            payments_offered=len(payments),
            credits_offered=len(credits),
            split=manifest.split.value,
            difficulty=manifest.difficulty.value,
            seed=manifest.seed,
            generator_version=manifest.generator_version,
        )

    known_payments = {payment.payment_id for payment in ingest.payments}
    known_credits = {txn.txn_id for txn in ingest.bank_txns}

    seen: set[tuple[str, str]] = set()
    predictions: list[PredictedLink] = []
    gross_by_payment = {
        payment.payment_id: payment.amount_minor for payment in ingest.payments
    }
    attempted = failed = returned = duplicated = 0
    unknown_payments = unknown_credits = 0

    for batch in batched(payments, payments_per_call):
        attempted += 1
        try:
            parsed, _ = client.complete_json(
                render_reconciliation(batch, credits),
                LLMOnlyBatch,
                prompt_version=B2_PROMPT_VERSION,
            )
        except LLMError:
            # A failed batch is a failed batch. B2 has no deterministic answer
            # to fall back to -- that is the whole difference between it and the
            # production system, and papering over it with a fallback would
            # quietly turn the baseline into the thing it is being compared to.
            failed += 1
            continue

        for link in parsed.links:
            returned += 1
            pair = (link.payment_id, link.bank_txn_id)
            if pair in seen:
                duplicated += 1
                continue
            seen.add(pair)
            if link.payment_id not in known_payments:
                unknown_payments += 1
            if link.bank_txn_id not in known_credits:
                unknown_credits += 1
            try:
                source = payment_ref(link.payment_id)
                target = bank_ref(link.bank_txn_id)
            except ValueError:
                # An id that is not even a well-formed record key -- empty, or
                # carrying the ':' separator. It cannot be scored as a link at
                # all, so it is counted above and dropped here rather than
                # crashing the baseline it is evidence against.
                continue
            predictions.append(
                PredictedLink(
                    source_ref=source,
                    target_ref=target,
                    # The model states no amount and nothing re-derives one, so
                    # the assertion is priced at the payment's declared gross --
                    # the same figure B0 asserts. An invented payment has no
                    # gross, so its false-positive cost is zero and the row is
                    # counted separately instead of being priced at a guess.
                    amount_minor=gross_by_payment.get(link.payment_id, 0),
                )
            )

    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
    return tuple(predictions), LLMBaselineArtifact(
        ran=True,
        payments_offered=len(payments),
        credits_offered=len(credits),
        calls_attempted=attempted,
        calls_failed=failed,
        links_returned=returned,
        links_asserted=len(predictions),
        links_duplicated=duplicated,
        unknown_payment_ids=unknown_payments,
        unknown_bank_txn_ids=unknown_credits,
        cost=client.ledger(),
        wall_clock_ms=int(elapsed_ms),
        split=manifest.split.value,
        difficulty=manifest.difficulty.value,
        seed=manifest.seed,
        generator_version=manifest.generator_version,
    )
