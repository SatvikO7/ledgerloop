"""Root causes and suggested actions, written deterministically.

PLAN.md 8.1 says ``root_cause`` is "LLM-written, evidence-grounded". This module
is the half that runs **without any model at all**, and it exists first for the
reason ARCHITECTURE.md 6 decision 15 gives for the narration parser: building
the deterministic half and measuring it is what makes the LLM's contribution
attributable later. ``ProseSource.TEMPLATE`` versus ``ProseSource.LLM`` on every
string is the routing signal, and the cost ledger reports the ratio.

WHAT A TEMPLATE IS ALLOWED TO SAY
---------------------------------
Only what the evidence already establishes. Each template interpolates figures
the classifier computed from the sources -- a settlement id, a rupee shortfall,
a day gap -- and never characterises intent ("the PSP appears to have..."). A
sentence that goes beyond the evidence chain is unverifiable, and an
unverifiable root cause is worse than a terse one, because a controller cannot
tell which parts to check.

The actions are written as instructions to a person, naming the record to look
at and the document to ask for. "Investigate" is not an action.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledgerloop.models.enums import ExceptionClass
from ledgerloop.money import format_minor

__all__ = ["PROSE_VERSION", "Prose", "prose_for"]

#: Bumped when a template's wording changes. Reported beside the exception so a
#: queue rendered by two versions of this file is not silently compared.
PROSE_VERSION = "1.0.0"


@dataclass(frozen=True)
class Prose:
    """A root cause and the action it implies."""

    root_cause: str
    suggested_action: str


def prose_for(
    exception_class: ExceptionClass,
    *,
    subject: str,
    impact_minor: int,
    detail: str = "",
    counterpart: str | None = None,
    day_gap: int | None = None,
) -> Prose:
    """The template pair for one exception.

    ``subject`` is the record the queue row is about; ``counterpart`` is the
    other end where there is one. Both are ids rather than descriptions, so a
    controller can search for them.
    """
    amount = format_minor(impact_minor)
    tail = f" {detail}" if detail else ""

    if exception_class is ExceptionClass.ROUNDING_DRIFT:
        return Prose(
            root_cause=(
                f"{subject} differs from the credited amount by {amount}, within the "
                f"per-record rounding tolerance.{tail}"
            ),
            suggested_action=(
                f"Post a rounding adjustment of {amount} against {subject}, or accept "
                "the difference if your policy writes off sub-rupee drift."
            ),
        )

    if exception_class is ExceptionClass.FEE_TAX_MISMATCH:
        return Prose(
            root_cause=(
                f"{subject} does not close on its own terms: the declared net differs "
                f"from gross minus fee minus tax plus adjustments by {amount}.{tail}"
            ),
            suggested_action=(
                f"Ask the PSP for the fee and tax breakdown behind {subject}; the "
                "discrepancy is in their report, not in the bank statement."
            ),
        )

    if exception_class is ExceptionClass.TIMING_SHIFT:
        gap = f" {day_gap:+d} day(s) from the settlement date" if day_gap is not None else ""
        return Prose(
            root_cause=(
                f"{subject} is worth {amount} and the only credit that fits it landed"
                f"{gap}, outside the matching window.{tail}"
            ),
            suggested_action=(
                f"Confirm the value date on {counterpart or 'the credit'} and re-match "
                f"{subject} across the wider window if the bank confirms the delay."
            ),
        )

    if exception_class is ExceptionClass.DUPLICATE_CREDIT:
        # Two ways to establish a duplicate, and they cite different evidence.
        # The keyed case has a shared reference to point at; the composed
        # A05+A07 case has none -- the narration lost it -- and the finding rests
        # on the amount, the narration and the ordering instead. Saying "under a
        # reference" about a row that carries no reference would send a
        # controller looking for something that is not there.
        return Prose(
            root_cause=(
                f"{subject} credits {amount} under a reference that also appears on "
                f"{counterpart or 'another credit'}. The same payout is in the "
                f"statement twice.{tail}"
                if day_gap is None
                else f"{subject} credits {amount} — the same amount under the same "
                f"narration as {counterpart or 'an earlier credit'}, "
                f"{day_gap} day(s) later. The same payout is in the statement "
                f"twice.{tail}"
            ),
            suggested_action=(
                f"Flag {subject} as a duplicate of {counterpart or 'the earlier credit'} "
                "and raise it with the bank. Do not delete either row."
            ),
        )

    if exception_class is ExceptionClass.POST_SETTLEMENT_REFUND:
        return Prose(
            root_cause=(
                f"{subject} declares a negative adjustment of {amount} that matches no "
                f"payment of its own; a refund raised after settlement was netted off "
                f"this payout.{tail}"
            ),
            suggested_action=(
                f"Obtain the refund reference behind the adjustment on {subject} and "
                "book it against the batch that originally carried the payment."
            ),
        )

    if exception_class is ExceptionClass.MISSING_REFERENCE:
        return Prose(
            root_cause=(
                f"{subject} is worth {amount} and no bank narration carries its "
                f"reference; the identifier did not survive into the statement.{tail}"
            ),
            suggested_action=(
                f"Ask the bank for the full remittance advice for {counterpart or subject}, "
                "which carries the reference the narration dropped."
            ),
        )

    if exception_class is ExceptionClass.CHARGEBACK_NETTED:
        return Prose(
            root_cause=(
                f"{subject} declares a negative adjustment of {amount} equal to exactly "
                f"one of its own payments{f' ({counterpart})' if counterpart else ''}; "
                f"that payment was charged back and its money never reached the bank.{tail}"
            ),
            suggested_action=(
                f"Request the chargeback detail for {counterpart or subject} and book the "
                "reversal against the original order."
            ),
        )

    if exception_class is ExceptionClass.SPLIT_PAYOUT_INCOMPLETE:
        # Two subjects reach this class and they need different sentences. A
        # settlement "was paid in tranches"; a bank credit *is* one of them, and
        # telling a controller that a credit was paid in tranches would be
        # nonsense. `day_gap` is the discriminator the classifier sets, exactly
        # as it does for DUPLICATE_CREDIT above.
        if day_gap is None:
            return Prose(
                root_cause=(
                    f"{subject} was paid in tranches and the credits carrying its "
                    f"reference do not account for the whole {amount} payout.{tail}"
                ),
                suggested_action=(
                    f"Ask the PSP for the tranche schedule behind {subject}; at least "
                    "one tranche is missing from the statement or carries a different "
                    "reference."
                ),
            )
        return Prose(
            root_cause=(
                f"{subject} credits {amount} under a reference that also appears on "
                f"{counterpart or 'another credit'} for a different amount. The payout "
                f"was split across them and the ladder could not establish which "
                f"payments travelled in which tranche.{tail}"
            ),
            suggested_action=(
                f"Ask the PSP which payments {subject} carries; the tranche split is "
                "not derivable from the statement alone."
            ),
        )

    if exception_class is ExceptionClass.ORPHAN_BANK_CREDIT:
        return Prose(
            root_cause=(
                f"{subject} credits {amount} under a reference that names no settlement "
                f"in this period.{tail}"
            ),
            suggested_action=(
                f"Trace {subject} with the bank; it is either income from outside this "
                "ledger or a payout declared in a period this run does not cover."
            ),
        )

    if exception_class is ExceptionClass.LATE_ARRIVAL:
        return Prose(
            root_cause=(
                f"{subject} declares a payout of {amount} and no credit in this statement "
                f"carries its reference or its amount.{tail}"
            ),
            suggested_action=(
                f"Check the next statement period for {subject}; if the money has still "
                "not arrived, raise the payout with the PSP."
            ),
        )

    if exception_class is ExceptionClass.AMBIGUOUS_AGGREGATION:
        return Prose(
            root_cause=(
                f"More than one set of payments in {subject} composes the {amount} that "
                f"arrived, and the arithmetic cannot separate them.{tail}"
            ),
            suggested_action=(
                f"Ask the PSP which payments travelled in each tranche of {subject}. The "
                "competing explanations are listed with this exception."
            ),
        )

    if exception_class is ExceptionClass.UNMATCHABLE:
        return Prose(
            root_cause=(
                f"{subject} credits {amount} with no reference and no merchant this "
                f"corpus has seen; nothing in the three sources can relate it.{tail}"
            ),
            suggested_action=(
                f"No action is available from these sources. Resolving {subject} needs "
                "data outside the ledger, the PSP report and the statement."
            ),
        )

    if exception_class is ExceptionClass.UNKNOWN_RESIDUAL:
        return Prose(
            root_cause=(
                f"{subject} is worth {amount} and reached the end of the ladder without "
                f"a rule explaining it. This is a gap in the system, not a named "
                f"anomaly.{tail}"
            ),
            suggested_action=(
                f"Escalate {subject} for manual reconciliation and record what the "
                "resolution turns out to be -- an unknown residual is a missing rule."
            ),
        )

    # No fallback. ``ExceptionClass`` is a closed vocabulary and every member is
    # handled above, so a default branch would be unreachable code that looks
    # like safety -- and would silently absorb a new class instead of failing
    # the moment one is added without prose. mypy proves the exhaustiveness.
    raise AssertionError(f"no template for {exception_class!r}")
