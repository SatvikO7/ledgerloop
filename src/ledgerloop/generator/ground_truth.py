"""Assemble :class:`GroundTruth` from a finished draft world.

Truth comes from two places, and neither is the emitted files:

* **Links** are read off ``DraftBankTxn.covered_payment_ids`` -- the generator's
  own statement of whose money each credit carries.
* **Verdicts** come from the :class:`ScenarioEffect` list -- the generator's own
  record of what it did.

Every record not named by an effect is ``CLEAN`` / ``MATCHED`` with zero impact,
because by construction the baseline world reconciled before phase 2 touched it.
"""

from __future__ import annotations

from ledgerloop.generator.world import DraftWorld
from ledgerloop.models.enums import AnomalyClass, Difficulty, ExpectedStatus, LinkType, SplitName
from ledgerloop.models.refs import RecordRef, bank_ref, order_ref, payment_ref, settlement_ref
from ledgerloop.models.truth import GroundTruth, GroundTruthLink, GroundTruthRecord
from ledgerloop.money import allocate_minor

__all__ = ["build_ground_truth"]


def _links(world: DraftWorld) -> list[GroundTruthLink]:
    """Every edge that should be discovered."""
    links: list[GroundTruthLink] = []
    payments = world.payments_by_id()

    # Structural edges: asserted by the sources, not earned by the matcher.
    # Recorded for lineage and graph inference, excluded from the metrics.
    for payment in world.payments:
        links.append(
            GroundTruthLink(
                link_type=LinkType.ORDER_PAID_BY,
                source_ref=order_ref(payment.order_id),
                target_ref=payment_ref(payment.payment_id),
                amount_minor=payment.amount_minor,
            )
        )
        links.append(
            GroundTruthLink(
                link_type=LinkType.PAYMENT_SETTLED_IN,
                source_ref=payment_ref(payment.payment_id),
                target_ref=settlement_ref(payment.settlement_id),
                amount_minor=payment.amount_minor,
            )
        )

    for txn in world.bank_txns:
        if txn.settlement_id is None:
            continue

        # A duplicate credit (A05) names a settlement but carries no payments.
        # It must produce no SETTLEMENT_CREDITED_AS edge either -- linking it is
        # exactly the false positive this class exists to catch.
        if not txn.covered_payment_ids:
            continue

        links.append(
            GroundTruthLink(
                link_type=LinkType.SETTLEMENT_CREDITED_AS,
                source_ref=settlement_ref(txn.settlement_id),
                target_ref=bank_ref(txn.txn_id),
                amount_minor=txn.credit_minor,
            )
        )

        # THE EVALUATION UNIT. The credit is allocated across the payments it
        # carries, so the link amounts sum to exactly the credit -- no paise
        # created, none destroyed, however the batch was split.
        shares = allocate_minor(
            txn.credit_minor,
            [payments[pid].amount_minor for pid in txn.covered_payment_ids],
        )
        for payment_id, share in zip(txn.covered_payment_ids, shares, strict=True):
            links.append(
                GroundTruthLink(
                    link_type=LinkType.PAYMENT_CREDITED_AS,
                    source_ref=payment_ref(payment_id),
                    target_ref=bank_ref(txn.txn_id),
                    amount_minor=share,
                )
            )

    return links


def _records(world: DraftWorld) -> list[GroundTruthRecord]:
    """One verdict per record, defaulting to CLEAN/MATCHED."""
    verdicts: dict[str, GroundTruthRecord] = {}

    def default(ref: RecordRef, status: ExpectedStatus = ExpectedStatus.MATCHED) -> None:
        verdicts[ref.key] = GroundTruthRecord(
            record_ref=ref,
            expected_status=status,
            anomaly_class=AnomalyClass.CLEAN,
            impact_minor=0,
        )

    for order in world.orders:
        default(order_ref(order.order_id))
    for payment in world.payments:
        default(payment_ref(payment.payment_id))
    for settlement in world.settlements:
        default(settlement_ref(settlement.settlement_id))
    for txn in world.bank_txns:
        # Noise rows and debits belong to nothing and are irreconcilable by
        # construction. They are true negatives, so they leave the match-rate
        # denominator alongside the orphan credits.
        status = (
            ExpectedStatus.MATCHED
            if txn.settlement_id is not None
            else ExpectedStatus.UNMATCHABLE
        )
        default(bank_ref(txn.txn_id), status)

    # Effects overwrite the defaults. Later effects on the same record win, which
    # is correct: a scenario that declined never appended one.
    for effect in world.effects:
        verdicts[effect.primary_ref.key] = GroundTruthRecord(
            record_ref=effect.primary_ref,
            expected_status=effect.expected_status,
            anomaly_class=effect.anomaly,
            impact_minor=effect.impact_minor,
            note=effect.note,
        )

    return [verdicts[key] for key in sorted(verdicts)]


def build_ground_truth(
    world: DraftWorld,
    *,
    split: SplitName,
    difficulty: Difficulty,
    seed: int,
    generator_version: str,
) -> GroundTruth:
    return GroundTruth(
        split=split,
        difficulty=difficulty,
        seed=seed,
        generator_version=generator_version,
        links=tuple(_links(world)),
        records=tuple(_records(world)),
        scenario_draws=dict(sorted(world.draws.items(), key=lambda item: item[0].value)),
    )
