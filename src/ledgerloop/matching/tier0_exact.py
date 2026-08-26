"""T0 -- exact keys. The cheapest and most certain tier.

Two exact joins, on the two ends of the chain the sources give a key for:

**The order leg** -- ``payment.order_ref_normalized`` against ``order.order_id``,
plus an exact amount check. This is the join Step 3's reference recovery
existed for: roughly one PSP reference in five is written as ``null``,
``ord 2026 004821``, or with a U+2011 non-breaking hyphen, and the two
recoverable forms only join at all because ingest canonicalised them. The edge
is ``ORDER_PAID_BY`` and it is deliberately **not** part of the headline
metrics -- ``ARCHITECTURE.md`` §2 restricts those to ``PAYMENT_CREDITED_AS`` --
so it can neither inflate nor deflate the score. It is matched because the
audit trail needs the chain to be complete and because a payment quoting an
order that does not exist is a finding in its own right.

**The bank leg** -- ``settlement.utr`` against ``bank_txn.extracted_utr``, plus
an exact net. Delegated to :mod:`ledgerloop.matching.bank_leg`, where T1 shares
the same resolver with a wider band.

WHY THE ORDER LEG CHECKS THE AMOUNT
------------------------------------
The reference alone would join. Requiring ``payment.amount_minor ==
order.amount_minor`` costs nothing on well-formed data -- the PSP captures what
the ledger booked -- and turns a mangled reference that happens to normalise
onto the wrong order from a silent wrong link into a declined one. An exact key
that disagrees on money is not an exact match.

WHY THERE IS NO ``PAYMENT_SETTLED_IN`` JOIN
--------------------------------------------
That edge is the PSP file's own nesting: a payment object lives *inside* a
settlement object, and ingest records it as ``CanonicalPayment.settlement_id``.
There is no key to join and nothing to earn. Manufacturing a match candidate
for it would be inventing work, and -- because candidate counts are reported --
inventing work that flatters the yield table.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledgerloop.matching.bank_leg import BankLegOutcome, BankLegRule, candidate_id, resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.refs import order_ref, payment_ref
from ledgerloop.money import delta_ratio, format_minor

__all__ = ["T0_RULE", "OrderLegOutcome", "resolve_order_leg", "run_tier0"]

#: T0's admission test: the exact key, an exact amount, and no date constraint.
#:
#: A zero floor and zero bps make :func:`~ledgerloop.money.tolerance_minor`
#: return a zero-width band, so ``within_tolerance`` degenerates to equality.
#: T0 is not a special case of the resolver -- it is the resolver at its limit.
T0_RULE = BankLegRule(
    tier=Tier.T0_EXACT,
    amount_floor_minor=0,
    amount_bps=0,
    date_window_days=None,
)


@dataclass(frozen=True)
class OrderLegOutcome:
    """What the order-leg join produced, with the misses broken out.

    The three failure counts are the point of returning a structure rather than
    a list. They are the measured cost of the PSP's reference corruption, and
    they separate "the source published nothing" from "the source published
    something wrong" -- two findings a controller acts on differently.
    """

    candidates: tuple[MatchCandidate, ...]
    matched: int = 0
    missing_reference: int = 0
    unknown_reference: int = 0
    amount_disagreement: int = 0


def resolve_order_leg(context: MatchContext) -> OrderLegOutcome:
    """Join every payment to the order it names. Exact reference, exact amount."""
    candidates: list[MatchCandidate] = []
    matched = missing = unknown = disagreement = 0

    for view in context.settlements:
        for payment in view.payments:
            reference = payment.order_ref_normalized
            if reference is None:
                missing += 1
                continue

            order = context.orders_by_id.get(reference)
            if order is None:
                unknown += 1
                continue

            delta = payment.amount_minor - order.amount_minor
            if delta != 0:
                disagreement += 1
                continue

            candidates.append(
                MatchCandidate(
                    candidate_id=candidate_id(
                        Tier.T0_EXACT,
                        LinkType.ORDER_PAID_BY,
                        order_ref(order.order_id).key,
                        payment_ref(payment.payment_id).key,
                    ),
                    link_type=LinkType.ORDER_PAID_BY,
                    source_ref=order_ref(order.order_id),
                    target_ref=payment_ref(payment.payment_id),
                    tier=Tier.T0_EXACT,
                    features=FeatureVector(
                        tier=Tier.T0_EXACT,
                        amount_delta_minor=delta,
                        tolerance_band_minor=0,
                        amount_delta_ratio=delta_ratio(delta, order.amount_minor),
                        date_delta_days=(payment.captured_at.date() - order.booked_at.date()).days,
                    ),
                    evidence=(
                        Evidence(
                            kind=EvidenceKind.EXACT_KEY,
                            detail=(
                                f"{payment.payment_id} quotes order reference "
                                f"{payment.order_ref_raw!r}, which normalises to "
                                f"{reference} and names a booked order"
                            ),
                            refs=(order_ref(order.order_id), payment_ref(payment.payment_id)),
                        ),
                        Evidence(
                            kind=EvidenceKind.AMOUNT_MATCH,
                            detail=(
                                f"captured {format_minor(payment.amount_minor)} equals the "
                                f"amount {order.order_id} was booked for"
                            ),
                            refs=(order_ref(order.order_id), payment_ref(payment.payment_id)),
                            amount_minor=payment.amount_minor,
                        ),
                    ),
                    calibrated_p=1.0,
                    arithmetic_verified=True,
                )
            )
            matched += 1

    return OrderLegOutcome(
        candidates=tuple(candidates),
        matched=matched,
        missing_reference=missing,
        unknown_reference=unknown,
        amount_disagreement=disagreement,
    )


def run_tier0(context: MatchContext) -> tuple[OrderLegOutcome, BankLegOutcome]:
    """Run both of T0's joins, consuming the pool as it goes.

    The order leg runs first and consumes nothing: it resolves a different pair
    of record types and cannot compete with the bank leg for either end.
    """
    return resolve_order_leg(context), resolve_bank_leg(context, T0_RULE)
