"""Source A -- the internal ledger. Clean CSV in, :class:`CanonicalOrder` out.

The easy one, and it is first for that reason: it settles the shape every other
parser follows -- validate the header against the declared schema, walk rows,
build a :class:`~ledgerloop.models.records.RawRecord` for provenance, convert
fields through :mod:`ledgerloop.ingest.fields`, quarantine what will not
convert, and return what did.

This is our own system of record, so it is the one source whose identifiers are
trusted: a duplicate ``order_id`` here is a real defect rather than expected
mess, and the first occurrence wins with the rest quarantined. Silently keeping
the last would let a later row overwrite an amount that other records already
reconciled against.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import ValidationError

from ledgerloop.ingest.fields import (
    RESTKEY,
    FieldError,
    read_currency,
    read_enum,
    read_money,
    read_timestamp,
    reject_ragged_row,
    require_text,
)
from ledgerloop.ingest.problems import IngestError, IngestProblemCode, ProblemLog
from ledgerloop.ingest.schemas import LEDGER_SCHEMA
from ledgerloop.models.enums import Currency, OrderStatus, SourceName
from ledgerloop.models.records import CanonicalOrder, RawRecord

__all__ = ["parse_ledger_csv", "parse_ledger_rows", "read_ledger_rows"]


def read_ledger_rows(path: Path) -> list[dict[str, str]]:
    """Read the CSV, validating its header against the declared schema.

    A missing required column raises rather than quarantining: every row would
    be misread, and a run that continued would produce a confidently wrong
    answer instead of a loud one.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, restkey=RESTKEY)
        header = reader.fieldnames or []
        mismatch = LEDGER_SCHEMA.describe_mismatch(header)
        if mismatch is not None:
            raise IngestError(mismatch)
        return list(reader)


def parse_ledger_rows(
    rows: list[dict[str, str]], log: ProblemLog
) -> tuple[CanonicalOrder, ...]:
    """Normalise ledger rows into canonical orders."""
    orders: list[CanonicalOrder] = []
    seen: set[str] = set()

    for line, row in enumerate(rows):
        payload: dict[str, object] = dict(row)
        record_id = str(row.get("order_id") or "").strip() or None
        try:
            reject_ragged_row(row)
            order_id = require_text(row, "order_id")
            if order_id in seen:
                raise FieldError(
                    "order_id",
                    IngestProblemCode.DUPLICATE_ID,
                    f"{order_id} already appeared in this file; keeping the first",
                )
            order = CanonicalOrder(
                raw=RawRecord(source=SourceName.LEDGER, source_line=line, payload=payload),
                order_id=order_id,
                merchant_id=require_text(row, "merchant_id"),
                customer_ref=require_text(row, "customer_ref"),
                amount_minor=read_money(row, "amount_gross_paise"),
                currency=read_currency(row, "currency", default=Currency.INR),
                booked_at=read_timestamp(row, "booked_at"),
                status=read_enum(row, "status", OrderStatus),
            )
        except FieldError as exc:
            log.record(
                source=SourceName.LEDGER,
                source_line=line,
                code=exc.code,
                detail=exc.detail,
                field=exc.field,
                record_id=record_id,
                payload=payload,
            )
            continue
        except ValidationError as exc:
            log.record(
                source=SourceName.LEDGER,
                source_line=line,
                code=IngestProblemCode.CONTRACT_VIOLATION,
                detail=str(exc),
                record_id=record_id,
                payload=payload,
            )
            continue

        seen.add(order.order_id)
        orders.append(order)

    return tuple(orders)


def parse_ledger_csv(path: Path, log: ProblemLog) -> tuple[CanonicalOrder, ...]:
    """Read and normalise ``ledger_orders.csv``."""
    return parse_ledger_rows(read_ledger_rows(path), log)
