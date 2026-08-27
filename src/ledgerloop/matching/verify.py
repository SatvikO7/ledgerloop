"""``verify_arithmetic`` -- the hard gate every proposed link passes.

PLAN.md 7.1, the hard rule: *"The LLM never decides a match by itself, and
never does arithmetic."* This module is the second half of that sentence made
executable. It re-derives, from the source documents alone, whether a proposed
link's money closes -- and it does not care where the proposal came from.

That indifference is the design. The function takes ids and returns a verdict;
it has no parameter for a model's confidence, its reasoning, or the fact that a
model was involved at all. A gate that could be told "this one is from a very
confident model" would eventually be told exactly that.

WHAT IS ACTUALLY CHECKED
------------------------
The same arithmetic T2 commits a partition on, run in the same direction:

1. Every named record exists, and the payments really are nested in the named
   settlement by the PSP's own report. A proposal spanning two batches is
   refused before any arithmetic happens -- it is not a hard sum, it is an
   incoherent one.
2. The payments' **gross** is inverted into the credit's **net** through
   ``allocate_minor``, the same conserving split the generator built the truth
   links from and the same one T0-T3 allocate with.
3. The re-derived net matches the credit within the configured aggregation
   epsilon.
4. The allocation back across the payments conserves exactly.

A proposal that passes has been verified against the sources, not believed. A
proposal that fails is **demoted, never dropped**: PLAN.md 7.4 says a failed
arithmetic check demotes to ``NEEDS_REVIEW`` with the failure logged as
evidence, because "the model suggested this and the arithmetic disagrees" is
information a controller wants, and silently discarding it would hide both the
suggestion and the disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier2_aggregation import expected_credit_minor
from ledgerloop.models.records import CanonicalBankTxn, CanonicalPayment
from ledgerloop.money import allocate_minor, format_minor, sum_minor, within_tolerance

__all__ = ["ArithmeticCheck", "verify_arithmetic"]


@dataclass(frozen=True)
class ArithmeticCheck:
    """The verdict, and enough of the working to put in an evidence chain."""

    verified: bool
    reason: str
    settlement_id: str | None = None
    credit_minor: int = 0
    expected_minor: int = 0
    gross_minor: int = 0
    payments: tuple[CanonicalPayment, ...] = ()
    credit: CanonicalBankTxn | None = None

    @property
    def residual_minor(self) -> int:
        """What the proposal leaves unexplained. Zero when it closes exactly."""
        return self.credit_minor - self.expected_minor

    def __bool__(self) -> bool:
        return self.verified


def verify_arithmetic(
    context: MatchContext,
    *,
    payment_ids: tuple[str, ...],
    bank_txn_id: str,
    epsilon_minor: int,
    settlement_id: str | None = None,
) -> ArithmeticCheck:
    """Whether these payments' money can be this credit, by the sources alone.

    ``settlement_id`` is optional and is *derived* when absent rather than
    trusted when present: the anchor is the PSP's own nesting, so a proposal
    that names a settlement its payments do not belong to is refused on that
    ground alone.
    """
    credit = next(
        (txn for txn in context.bank_txns if txn.txn_id == bank_txn_id), None
    )
    if credit is None:
        return ArithmeticCheck(False, f"{bank_txn_id} is not a row in the statement")
    if not credit.is_credit:
        return ArithmeticCheck(
            False,
            f"{bank_txn_id} is an outgoing row; money leaving the account cannot "
            "settle a payout",
        )
    if not payment_ids:
        return ArithmeticCheck(False, "no payments were named")
    if len(set(payment_ids)) != len(payment_ids):
        return ArithmeticCheck(
            False, "the same payment was named twice; a payment travels once"
        )

    by_id = {
        payment.payment_id: payment
        for view in context.settlements
        for payment in view.payments
    }
    missing = [name for name in payment_ids if name not in by_id]
    if missing:
        return ArithmeticCheck(
            False, f"no such payment(s) in the PSP report: {', '.join(sorted(missing))}"
        )

    payments = tuple(by_id[name] for name in payment_ids)
    anchors = {payment.settlement_id for payment in payments}
    if len(anchors) != 1 or None in anchors:
        return ArithmeticCheck(
            False,
            "the payments named are not all nested in one settlement, so no single "
            "payout can carry them",
        )
    anchor = anchors.pop()
    assert anchor is not None  # the guard above rules None out
    if settlement_id is not None and settlement_id != anchor:
        return ArithmeticCheck(
            False,
            f"the payments named sit in {anchor}, not in {settlement_id}",
            settlement_id=anchor,
        )

    view = context.settlements_by_id.get(anchor)
    if view is None:
        return ArithmeticCheck(False, f"{anchor} is not a settlement in this run")

    gross = sum_minor(
        (payment.amount_minor for payment in payments), field=f"{anchor}.verify"
    )
    expected = expected_credit_minor(gross, view.payment_gross_minor, view.net_minor)
    if not within_tolerance(expected, credit.credit_minor, epsilon_minor):
        return ArithmeticCheck(
            False,
            f"{len(payments)} payment(s) carrying gross {format_minor(gross)} allocate "
            f"to {format_minor(expected)} of {anchor}'s net, which is not "
            f"{format_minor(credit.credit_minor)} within "
            f"{format_minor(epsilon_minor)}",
            settlement_id=anchor,
            credit_minor=credit.credit_minor,
            expected_minor=expected,
            gross_minor=gross,
            payments=payments,
            credit=credit,
        )

    shares = allocate_minor(
        credit.credit_minor, [payment.amount_minor for payment in payments]
    )
    if sum_minor(shares, field=f"{bank_txn_id}.verify") != credit.credit_minor:
        return ArithmeticCheck(  # pragma: no cover - allocate_minor conserves by contract
            False,
            "the allocation back across the payments does not conserve",
            settlement_id=anchor,
            credit_minor=credit.credit_minor,
            expected_minor=expected,
            gross_minor=gross,
            payments=payments,
            credit=credit,
        )

    return ArithmeticCheck(
        True,
        f"{len(payments)} payment(s) of {anchor} carrying gross {format_minor(gross)} "
        f"allocate to {format_minor(expected)}, matching {bank_txn_id}'s "
        f"{format_minor(credit.credit_minor)} within {format_minor(epsilon_minor)}",
        settlement_id=anchor,
        credit_minor=credit.credit_minor,
        expected_minor=expected,
        gross_minor=gross,
        payments=payments,
        credit=credit,
    )
