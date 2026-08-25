"""Phase 1 -- build a world that reconciles perfectly.

Everything here is deterministic given the seeded RNG. Iteration is over lists
in construction order and every random draw comes from the one ``Random``
instance, so two runs with the same seed produce byte-identical output.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from ledgerloop.generator.vocab import MERCHANTS, NARRATION_WITH_UTR, NOISE_NARRATIONS
from ledgerloop.generator.world import (
    DraftBankTxn,
    DraftOrder,
    DraftPayment,
    DraftSettlement,
    DraftWorld,
)
from ledgerloop.models.enums import OrderStatus

__all__ = ["build_clean_world", "make_utr"]

#: All dates hang off this epoch, so a dataset's calendar is a function of its
#: seed and size alone -- never of the day it was generated.
EPOCH = date(2026, 3, 1)

#: Order values, in paise. ₹300 to ₹48,000 -- a realistic e-commerce spread.
MIN_ORDER_MINOR = 30_000
MAX_ORDER_MINOR = 4_800_000

#: Payments per settlement batch. The lower bound matters: a batch of one is a
#: 1:1 join, and the N:1 aggregation problem is the point of the project.
MIN_BATCH = 6
MAX_BATCH = 16

#: Noise bank rows per 10 orders -- rent, salary, GST. These must match nothing.
NOISE_PER_TEN_ORDERS = 2

#: Orders each merchant needs before another merchant is brought in.
#:
#: Merchant count scales with dataset size rather than being fixed at twelve.
#: Spreading 60 dev orders across all twelve merchants would give each one a
#: single five-payment batch, which breaks the corpus in two ways: batches stop
#: being large enough for the N:1 aggregation problem to be interesting, and
#: A06 (a refund clawed back from a *later* batch) could never fire at all,
#: because no merchant would have a later batch.
ORDERS_PER_MERCHANT = 20

#: U+2011 NON-BREAKING HYPHEN. Renders identically to the ASCII hyphen and
#: never compares equal to it -- the reference corruption PLAN.md §5.1 asks for.
#: Built with chr() rather than written literally so that an editor, a
#: copy-paste, or a linter autofix cannot silently normalise it back to ASCII
#: and quietly delete the anomaly.
NON_BREAKING_HYPHEN = chr(0x2011)


def make_utr(rng: random.Random, settled_on: date) -> str:
    """A UTR shaped like the real thing: ``UTR`` + date + a 5-digit tail."""
    return f"UTR{settled_on.strftime('%Y%m%d')}{rng.randint(10_000, 99_999)}"


def _corrupt_order_ref(rng: random.Random, order_id: str) -> str | None:
    """Mangle the PSP's copy of the order reference, as PLAN.md §5.1 requires.

    Roughly a fifth of payments get a reference that will not join exactly. This
    is baseline noise, not one of the eleven anomaly classes: it is what makes
    T0 fall short of 100% even on completely clean money, and what gives the
    normaliser something to recover.
    """
    roll = rng.random()
    if roll < 0.80:
        return order_id
    if roll < 0.87:
        return None
    if roll < 0.94:
        return order_id.replace("-", " ").lower()
    return order_id.replace("-", NON_BREAKING_HYPHEN)


def build_clean_world(rng: random.Random, order_count: int) -> DraftWorld:
    """Phase 1: a world where every rupee reconciles.

    Structure: orders are dealt to merchants, each merchant's orders are cut into
    settlement batches, and each batch lands as exactly one bank credit for its
    net. Anomalies are applied afterwards, on top of this.
    """
    world = DraftWorld()
    merchant_count = max(2, min(len(MERCHANTS), order_count // ORDERS_PER_MERCHANT))

    # --- orders ---
    for index in range(order_count):
        merchant = MERCHANTS[index % merchant_count]
        booked_at = datetime.combine(
            EPOCH + timedelta(days=rng.randint(0, 27)),
            datetime.min.time(),
        ) + timedelta(
            hours=rng.randint(6, 22),
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )
        world.orders.append(
            DraftOrder(
                order_id=f"ORD-2026-{index + 1:06d}",
                merchant_id=merchant.merchant_id,
                customer_ref=f"CUST_{rng.randint(10_000, 19_999)}",
                amount_minor=rng.randrange(MIN_ORDER_MINOR, MAX_ORDER_MINOR, 100),
                booked_at=booked_at,
                status=OrderStatus.CAPTURED,
            )
        )

    # --- batch each merchant's orders into settlements ---
    orders_by_merchant: dict[str, list[DraftOrder]] = {}
    for order in world.orders:
        orders_by_merchant.setdefault(order.merchant_id, []).append(order)

    payment_seq = 0
    settlement_seq = 0
    bank_seq = 0

    # Sorted for determinism: dict insertion order already follows `orders`, but
    # relying on that would make the generator fragile to an unrelated reorder.
    for merchant_id in sorted(orders_by_merchant):
        merchant_orders = sorted(orders_by_merchant[merchant_id], key=lambda o: o.booked_at)
        batches: list[list[DraftOrder]] = []
        cursor = 0
        while cursor < len(merchant_orders):
            batch = merchant_orders[cursor : cursor + rng.randint(MIN_BATCH, MAX_BATCH)]
            cursor += len(batch)
            batches.append(batch)

        # Fold a short trailing batch into its predecessor. A remainder batch of
        # one payment is a 1:1 join, and the N:1 aggregation problem is the
        # reason this project exists -- leaving those in would quietly hand the
        # matcher free wins that inflate every tier's yield.
        if len(batches) > 1 and len(batches[-1]) < MIN_BATCH:
            batches[-2].extend(batches.pop())

        for batch in batches:
            settlement_seq += 1

            # Settle the day after the batch's last capture; credit lands T+1.
            settled_on = max(order.booked_at.date() for order in batch) + timedelta(days=1)
            settlement = DraftSettlement(
                settlement_id=f"SETL-{settlement_seq:04d}",
                merchant_id=merchant_id,
                utr=make_utr(rng, settled_on),
                settled_on=settled_on,
            )

            for order in batch:
                payment_seq += 1
                payment = DraftPayment(
                    payment_id=f"PAY-{payment_seq:05d}",
                    order_id=order.order_id,
                    settlement_id=settlement.settlement_id,
                    amount_minor=order.amount_minor,
                    captured_at=order.booked_at + timedelta(seconds=rng.randint(1, 90)),
                    order_ref_raw=_corrupt_order_ref(rng, order.order_id),
                )
                world.payments.append(payment)
                settlement.payment_ids.append(payment.payment_id)

            world.settlements.append(settlement)

    # Identity mappings are complete now; scenarios only mutate values.
    world.reindex()

    # --- one bank credit per settlement, for its declared net ---
    payments = world.payments_by_id()
    for settlement in world.settlements:
        bank_seq += 1
        merchant = next(m for m in MERCHANTS if m.merchant_id == settlement.merchant_id)
        template = NARRATION_WITH_UTR[rng.randrange(len(NARRATION_WITH_UTR))]
        world.add_bank_txn(
            DraftBankTxn(
                txn_id=f"BNK-{bank_seq:05d}",
                value_date=settlement.settled_on + timedelta(days=1),
                narration=template.format(
                    variant=merchant.variants[rng.randrange(len(merchant.variants))],
                    utr=settlement.utr,
                ),
                credit_minor=settlement.declared_net_minor(payments),
                settlement_id=settlement.settlement_id,
                covered_payment_ids=list(settlement.payment_ids),
            )
        )

    # --- unrelated rows that must match nothing ---
    noise_count = max(2, (order_count * NOISE_PER_TEN_ORDERS) // 10)
    for _ in range(noise_count):
        bank_seq += 1
        is_credit = rng.random() < 0.35
        amount = rng.randrange(50_000, 9_000_000, 100)
        world.add_bank_txn(
            DraftBankTxn(
                txn_id=f"BNK-{bank_seq:05d}",
                value_date=EPOCH + timedelta(days=rng.randint(0, 30)),
                narration=NOISE_NARRATIONS[rng.randrange(len(NOISE_NARRATIONS))],
                credit_minor=amount if is_credit else 0,
                debit_minor=0 if is_credit else amount,
                settlement_id=None,
            )
        )

    return world
