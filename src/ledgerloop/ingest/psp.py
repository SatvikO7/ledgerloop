"""Source B -- the PSP settlement report. Nested JSON, two entity types out.

One file yields both :class:`CanonicalSettlement` and :class:`CanonicalPayment`,
because the N:1 aggregation problem T2 solves is precisely about which payments
compose a settlement. Flattening the batch into its payments would delete the
structure; keeping only the batch would delete the members.

TWO THINGS THIS PARSER DELIBERATELY DOES NOT DO
------------------------------------------------
**It does not validate the settlement arithmetic.** ``net = gross - fee - tax +
adjustments`` is exactly what anomaly A03 ``FEE_TAX_MISMATCH`` breaks on
purpose. A parser enforcing it would make 4% of the corpus unparseable and
delete the anomaly the system exists to detect. The check lives on
:attr:`CanonicalSettlement.net_delta_minor` and its result is *evidence*.

**It does not decide whether an order reference is real.** It normalises what
the PSP wrote and stores both forms. Roughly a fifth of references are
corrupted (PLAN.md 5.1) -- ``null``, ``"ord 2026 004821"``, or
``ORD-2026-004821`` carrying U+2011 NON-BREAKING HYPHEN. Two of the three are
recoverable by :func:`~ledgerloop.ingest.normalize.normalize_order_ref` and one
is not, and keeping ``order_ref_raw`` alongside ``order_ref_normalized`` is
what lets an exception explain *why* an exact join missed rather than merely
that it did.

PROVENANCE THROUGH A NESTED DOCUMENT
------------------------------------
``RawRecord.source_line`` is a position within a record *stream*, not a line in
a text file -- JSON has no lines. Settlements are numbered in document order,
and payments are numbered in a separate flat sequence across the whole file, so
the pair (record type, source_line) addresses exactly one object. Each payload
is the source object verbatim, which is what the audit trail has to be able to
show a controller.

A settlement and its payments succeed or fail together. Payments are built
first but only published once their parent settlement has been constructed --
otherwise a batch that failed contract validation would leave its payments in
the corpus with a ``settlement_id`` pointing at nothing, and T2 would anchor on
a settlement that does not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ledgerloop.ingest.fields import (
    FieldError,
    optional_money,
    optional_text,
    read_iso_date,
    read_money,
    read_timestamp,
    require_text,
)
from ledgerloop.ingest.normalize import normalize_order_ref, normalize_utr
from ledgerloop.ingest.problems import IngestError, IngestProblemCode, ProblemLog
from ledgerloop.ingest.schemas import PSP_PAYMENT_SCHEMA, PSP_SETTLEMENT_SCHEMA
from ledgerloop.models.enums import SourceName
from ledgerloop.models.records import CanonicalPayment, CanonicalSettlement, RawRecord

__all__ = ["PspRecords", "parse_psp_json", "parse_psp_payload", "read_psp_payload"]

_SOURCE = SourceName.PSP


@dataclass(frozen=True)
class PspRecords:
    """The two entity streams one PSP file produces."""

    settlements: tuple[CanonicalSettlement, ...]
    payments: tuple[CanonicalPayment, ...]


def read_psp_payload(path: Path) -> list[dict[str, Any]]:
    """Read the JSON document and return its settlement objects.

    Structural failures raise: a document that is not an object, has no
    ``settlements`` key, or whose ``settlements`` is not a list of objects
    cannot be read at all, and quarantining every record one at a time would
    report a hundred problems where there is one.
    """
    with path.open("r", encoding="utf-8") as handle:
        document: object = json.load(handle)

    if not isinstance(document, dict):
        raise IngestError(f"{path}: expected a JSON object at the top level")
    batches = document.get("settlements")
    if not isinstance(batches, list):
        raise IngestError(f"{path}: expected a 'settlements' array at the top level")
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise IngestError(
                f"{path}: settlements[{index}] is a {type(batch).__name__}, not an object"
            )
    return list(batches)


def _parse_payments(
    entries: list[Any],
    *,
    settlement_id: str,
    first_line: int,
    log: ProblemLog,
) -> list[CanonicalPayment]:
    """Normalise one batch's payments. Bad payments are quarantined individually."""
    payments: list[CanonicalPayment] = []

    for offset, entry in enumerate(entries):
        line = first_line + offset
        if not isinstance(entry, dict):
            log.record(
                source=_SOURCE,
                source_line=line,
                code=IngestProblemCode.MALFORMED_STRUCTURE,
                detail=f"payment is a {type(entry).__name__}, not an object",
                field="payments",
                record_id=settlement_id,
            )
            continue

        payload: dict[str, object] = dict(entry)
        record_id = str(entry.get("payment_id") or "").strip() or None
        try:
            mismatch = PSP_PAYMENT_SCHEMA.describe_mismatch(entry.keys())
            if mismatch is not None:
                raise FieldError("payment", IngestProblemCode.MISSING_FIELD, mismatch)
            order_ref_raw = optional_text(entry, "order_ref")
            payments.append(
                CanonicalPayment(
                    raw=RawRecord(source=_SOURCE, source_line=line, payload=payload),
                    payment_id=require_text(entry, "payment_id"),
                    settlement_id=settlement_id,
                    order_ref_raw=order_ref_raw,
                    order_ref_normalized=normalize_order_ref(order_ref_raw),
                    amount_minor=read_money(entry, "amount_paise"),
                    captured_at=read_timestamp(entry, "captured_at"),
                )
            )
        except FieldError as exc:
            log.record(
                source=_SOURCE,
                source_line=line,
                code=exc.code,
                detail=exc.detail,
                field=exc.field,
                record_id=record_id,
                payload=payload,
            )
        except ValidationError as exc:
            log.record(
                source=_SOURCE,
                source_line=line,
                code=IngestProblemCode.CONTRACT_VIOLATION,
                detail=str(exc),
                record_id=record_id,
                payload=payload,
            )

    return payments


def parse_psp_payload(batches: list[dict[str, Any]], log: ProblemLog) -> PspRecords:
    """Normalise PSP settlement objects into settlements and payments."""
    settlements: list[CanonicalSettlement] = []
    payments: list[CanonicalPayment] = []
    seen_settlements: set[str] = set()
    seen_payments: set[str] = set()
    next_payment_line = 0

    for line, batch in enumerate(batches):
        payload: dict[str, object] = dict(batch)
        record_id = str(batch.get("settlement_id") or "").strip() or None
        raw_payments = batch.get("payments") or []

        try:
            mismatch = PSP_SETTLEMENT_SCHEMA.describe_mismatch(batch.keys())
            if mismatch is not None:
                raise FieldError("settlement", IngestProblemCode.MISSING_FIELD, mismatch)
            if not isinstance(raw_payments, list):
                raise FieldError(
                    "payments",
                    IngestProblemCode.MALFORMED_STRUCTURE,
                    f"'payments' is a {type(raw_payments).__name__}, not an array",
                )
            settlement_id = require_text(batch, "settlement_id")
            if settlement_id in seen_settlements:
                raise FieldError(
                    "settlement_id",
                    IngestProblemCode.DUPLICATE_ID,
                    f"{settlement_id} already appeared in this file; keeping the first",
                )
            gross = read_money(batch, "gross_paise")
            fee = read_money(batch, "fee_paise")
            tax = optional_money(batch, "tax_paise")
            adjustments = optional_money(batch, "adjustments_paise")
            net = read_money(batch, "net_paise")
            settled_on = read_iso_date(batch, "settled_on")
        except FieldError as exc:
            # The batch is unreadable, so its payments are never numbered: the
            # payment stream stays a dense sequence over the records that exist.
            log.record(
                source=_SOURCE,
                source_line=line,
                code=exc.code,
                detail=exc.detail,
                field=exc.field,
                record_id=record_id,
                payload=payload,
            )
            continue

        batch_payments = _parse_payments(
            list(raw_payments),
            settlement_id=settlement_id,
            first_line=next_payment_line,
            log=log,
        )
        next_payment_line += len(raw_payments)

        kept: list[CanonicalPayment] = []
        for payment in batch_payments:
            if payment.payment_id in seen_payments:
                log.record(
                    source=_SOURCE,
                    source_line=payment.raw.source_line if payment.raw else line,
                    code=IngestProblemCode.DUPLICATE_ID,
                    detail=(
                        f"{payment.payment_id} already appeared in this file; "
                        "keeping the first"
                    ),
                    field="payment_id",
                    record_id=payment.payment_id,
                )
                continue
            kept.append(payment)

        try:
            settlement = CanonicalSettlement(
                raw=RawRecord(source=_SOURCE, source_line=line, payload=payload),
                settlement_id=settlement_id,
                utr=normalize_utr(optional_text(batch, "utr")),
                settled_on=settled_on,
                gross_minor=gross,
                fee_minor=fee,
                tax_minor=tax,
                adjustments_minor=adjustments,
                net_minor=net,
                payment_ids=tuple(payment.payment_id for payment in kept),
            )
        except ValidationError as exc:
            log.record(
                source=_SOURCE,
                source_line=line,
                code=IngestProblemCode.CONTRACT_VIOLATION,
                detail=str(exc),
                record_id=record_id,
                payload=payload,
            )
            continue

        seen_settlements.add(settlement_id)
        settlements.append(settlement)
        payments.extend(kept)
        seen_payments.update(payment.payment_id for payment in kept)

    return PspRecords(settlements=tuple(settlements), payments=tuple(payments))


def parse_psp_json(path: Path, log: ProblemLog) -> PspRecords:
    """Read and normalise ``psp_settlements.json``."""
    return parse_psp_payload(read_psp_payload(path), log)
