"""Write the three heterogeneous sources, plus ground truth and a manifest.

**Byte-identical output is a hard requirement** (PLAN.md Phase 1 acceptance).
Everything here is therefore explicit: ``newline="\\n"`` so the writer never
emits CRLF on Windows, fixed column order, ``sort_keys=True`` on JSON, and no
timestamp or path anywhere in the payload. Two runs at the same seed produce the
same bytes on any machine.

The three sources are deliberately *not* uniform. The ledger is clean CSV, the
PSP report is nested JSON, and the bank statement is CSV with free-text
narration and ``DD/MM/YYYY`` dates -- the ambiguous format, on purpose.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ledgerloop.generator.world import DraftWorld
from ledgerloop.models.truth import GroundTruth

__all__ = [
    "BANK_FILE",
    "GROUND_TRUTH_LINKS_FILE",
    "GROUND_TRUTH_RECORDS_FILE",
    "LEDGER_FILE",
    "MANIFEST_FILE",
    "PSP_FILE",
    "write_dataset",
]

LEDGER_FILE = "ledger_orders.csv"
PSP_FILE = "psp_settlements.json"
BANK_FILE = "bank_statement.csv"
GROUND_TRUTH_LINKS_FILE = "ground_truth_links.csv"
GROUND_TRUTH_RECORDS_FILE = "ground_truth_records.csv"
MANIFEST_FILE = "manifest.json"


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _emit_ledger(path: Path, world: DraftWorld) -> None:
    """Source A -- clean, structured, our own system of record."""
    _write_csv(
        path,
        (
            "order_id",
            "merchant_id",
            "customer_ref",
            "amount_gross_paise",
            "currency",
            "booked_at",
            "status",
        ),
        [
            (
                order.order_id,
                order.merchant_id,
                order.customer_ref,
                order.amount_minor,
                "INR",
                order.booked_at.isoformat(),
                order.status.value,
            )
            for order in world.orders
        ],
    )


def _emit_psp(path: Path, world: DraftWorld) -> None:
    """Source B -- nested JSON, one object per payout batch. Fees live here."""
    payments = world.payments_by_id()
    batches: list[dict[str, Any]] = []
    for settlement in world.settlements:
        batches.append(
            {
                "settlement_id": settlement.settlement_id,
                "merchant_id": settlement.merchant_id,
                "utr": settlement.utr,
                "settled_on": settlement.settled_on.isoformat(),
                "gross_paise": settlement.gross_minor(payments),
                "fee_paise": settlement.fee_minor(payments),
                "tax_paise": settlement.tax_minor(payments),
                "adjustments_paise": settlement.adjustments_minor,
                "net_paise": settlement.declared_net_minor(payments),
                "payments": [
                    {
                        "payment_id": payment_id,
                        # Deliberately mangled for some payments; null for others.
                        "order_ref": payments[payment_id].order_ref_raw,
                        "amount_paise": payments[payment_id].amount_minor,
                        "captured_at": payments[payment_id].captured_at.isoformat(),
                    }
                    for payment_id in settlement.payment_ids
                ],
            }
        )
    _write_json(path, {"settlements": batches})


def _emit_bank(path: Path, world: DraftWorld) -> None:
    """Source C -- the messy one. Free text, and DD/MM/YYYY on purpose.

    Rows are ordered by value date then id, the way a statement actually
    arrives, so ingest cannot rely on generation order. The running balance is
    computed over that order.
    """
    ordered = sorted(world.bank_txns, key=lambda txn: (txn.value_date, txn.txn_id))
    balance = 19_844_210
    rows: list[Sequence[object]] = []
    for txn in ordered:
        balance += txn.credit_minor - txn.debit_minor
        rows.append(
            (
                txn.txn_id,
                txn.value_date.strftime("%d/%m/%Y"),
                txn.narration,
                txn.credit_minor,
                txn.debit_minor,
                balance,
            )
        )
    _write_csv(
        path,
        ("txn_id", "value_date", "narration", "credit_paise", "debit_paise", "balance_paise"),
        rows,
    )


def _emit_ground_truth(directory: Path, truth: GroundTruth) -> None:
    _write_csv(
        directory / GROUND_TRUTH_LINKS_FILE,
        ("link_type", "source_ref", "target_ref", "amount_paise", "anomaly_class"),
        [
            (
                link.link_type.value,
                link.source_ref.key,
                link.target_ref.key,
                link.amount_minor,
                link.anomaly_class.value,
            )
            for link in truth.links
        ],
    )
    _write_csv(
        directory / GROUND_TRUTH_RECORDS_FILE,
        ("record_ref", "expected_status", "anomaly_class", "impact_paise", "note"),
        [
            (
                record.record_ref.key,
                record.expected_status.value,
                record.anomaly_class.value,
                record.impact_minor,
                record.note or "",
            )
            for record in truth.records
        ],
    )


def write_dataset(directory: Path, world: DraftWorld, truth: GroundTruth) -> dict[str, int]:
    """Write all five files plus the manifest. Returns the counts it recorded."""
    directory.mkdir(parents=True, exist_ok=True)

    _emit_ledger(directory / LEDGER_FILE, world)
    _emit_psp(directory / PSP_FILE, world)
    _emit_bank(directory / BANK_FILE, world)
    _emit_ground_truth(directory, truth)

    counts = {
        "orders": len(world.orders),
        "payments": len(world.payments),
        "settlements": len(world.settlements),
        "bank_txns": len(world.bank_txns),
        "truth_links": len(truth.links),
        "evaluation_pairs": len(truth.evaluation_pairs),
        "reconcilable_records": len(truth.reconcilable_refs),
        "unmatchable_records": len(truth.unmatchable_refs),
        "effects_applied": len(world.effects),
    }

    _write_json(
        directory / MANIFEST_FILE,
        {
            "split": truth.split.value,
            "difficulty": truth.difficulty.value,
            "seed": truth.seed,
            "generator_version": truth.generator_version,
            "counts": counts,
            "scenario_draws": {
                anomaly.value: count for anomaly, count in truth.scenario_draws.items()
            },
            "money": {
                "settled_credit_total_paise": world.settled_credit_total_minor(),
                "declared_net_total_paise": world.declared_net_total_minor(),
                "declared_bank_delta_paise": world.declared_bank_delta_minor(),
            },
        },
    )
    return counts
