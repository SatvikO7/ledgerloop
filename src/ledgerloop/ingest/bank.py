"""Source C -- the bank statement. The messy one.

Two things make this parser different from the other two, and both are the
reason the source exists at all (PLAN.md 5.1):

**The dates are ambiguous.** ``06/03/2026`` is 6 March or 3 June depending on a
convention the file never states. The column is therefore read twice: once to
infer the component order from the column as a whole, once to parse every row
under that single decision. See :mod:`ledgerloop.ingest.dates` for the
inference and its three outcomes.

**There is no reference field.** Everything joinable is buried in free text,
and :func:`~ledgerloop.ingest.narration.parse_narration` digs it out
deterministically. ``extracted_utr`` and ``extracted_merchant`` stay ``None``
under anomaly A07 ``MISSING_REFERENCE`` and on the unrelated noise rows, which
is correct in both cases and for different reasons: A07 has had its reference
stripped, and a rent debit never had one.

Credits and debits stay separate non-negative columns, mirroring how statements
are actually published. A row asserting both is contradictory -- a single line
cannot be money arriving and money leaving -- and is quarantined rather than
netted, because netting would silently turn a data error into a plausible
smaller number.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ledgerloop.ingest.dates import DateOrder, DateOrderEvidence, infer_date_order
from ledgerloop.ingest.fields import (
    RESTKEY,
    FieldError,
    read_money,
    read_slash_date,
    reject_ragged_row,
    require_non_negative,
    require_text,
)
from ledgerloop.ingest.narration import NarrationParse, parse_narration
from ledgerloop.ingest.problems import IngestError, IngestProblemCode, ProblemLog
from ledgerloop.ingest.schemas import BANK_SCHEMA, value_of
from ledgerloop.models.enums import SourceName
from ledgerloop.models.records import CanonicalBankTxn, RawRecord

__all__ = ["BankRecords", "parse_bank_csv", "parse_bank_rows", "read_bank_rows"]

_SOURCE = SourceName.BANK
_DATE_FIELD = "value_date"


@dataclass(frozen=True)
class BankRecords:
    """Bank transactions, plus the parse decisions taken to produce them.

    The date-order evidence and the narration parses travel with the records
    because both are *claims*. A convention inferred from data and a merchant
    name recovered by subtraction are each defensible only if the basis can be
    shown, and neither fits on the canonical record.
    """

    transactions: tuple[CanonicalBankTxn, ...]
    date_order: DateOrderEvidence
    narrations: dict[str, NarrationParse]


def read_bank_rows(path: Path) -> list[dict[str, str]]:
    """Read the CSV, validating its header against the declared schema."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, restkey=RESTKEY)
        header = reader.fieldnames or []
        mismatch = BANK_SCHEMA.describe_mismatch(header)
        if mismatch is not None:
            raise IngestError(mismatch)
        return list(reader)


def parse_bank_rows(
    rows: list[dict[str, str]],
    log: ProblemLog,
    *,
    default_date_order: DateOrder = DateOrder.DAY_FIRST,
) -> BankRecords:
    """Normalise bank rows into canonical transactions.

    Pass one infers the date convention across the whole column; pass two
    parses every row under it. A column that witnesses both conventions raises
    :class:`~ledgerloop.ingest.problems.IngestError` -- it is a file-level
    contradiction, and picking a winner would silently misdate half the rows.
    """
    try:
        evidence = infer_date_order(
            (row.get(_DATE_FIELD) or "" for row in rows), default=default_date_order
        )
    except ValueError as exc:
        raise IngestError(f"{BANK_SCHEMA.filename}: {exc}") from exc

    # An unproven date order is *evidence*, not a malformed record: nothing was
    # rejected and nothing needs fixing by hand. It travels on
    # `BankRecords.date_order` and is surfaced by the report, rather than being
    # logged as a problem that a strict run would then refuse to continue past.
    transactions: list[CanonicalBankTxn] = []
    narrations: dict[str, NarrationParse] = {}
    seen: set[str] = set()

    for line, row in enumerate(rows):
        payload: dict[str, object] = dict(row)
        record_id = str(row.get("txn_id") or "").strip() or None
        try:
            reject_ragged_row(row)
            txn_id = require_text(row, "txn_id")
            if txn_id in seen:
                raise FieldError(
                    "txn_id",
                    IngestProblemCode.DUPLICATE_ID,
                    f"{txn_id} already appeared in this file; keeping the first",
                )
            credit = require_non_negative(read_money(row, "credit_paise"), "credit_paise")
            debit = require_non_negative(read_money(row, "debit_paise"), "debit_paise")
            if credit and debit:
                raise FieldError(
                    "credit_paise",
                    IngestProblemCode.CONTRADICTORY_SIGNS,
                    f"row asserts both a credit ({credit}) and a debit ({debit}); "
                    "one line cannot be money arriving and money leaving",
                )
            narration_raw = require_text(row, "narration")
            parsed = parse_narration(narration_raw)
            has_balance = value_of(row, "balance_paise") is not None
            txn = CanonicalBankTxn(
                raw=RawRecord(source=_SOURCE, source_line=line, payload=payload),
                txn_id=txn_id,
                value_date=read_slash_date(row, _DATE_FIELD, evidence.order),
                narration_raw=narration_raw,
                narration_normalized=parsed.normalized,
                extracted_utr=parsed.utr,
                extracted_merchant=parsed.merchant,
                credit_minor=credit,
                debit_minor=debit,
                balance_minor=read_money(row, "balance_paise") if has_balance else None,
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
            continue
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

        seen.add(txn_id)
        transactions.append(txn)
        narrations[txn_id] = parsed

    return BankRecords(
        transactions=tuple(transactions), date_order=evidence, narrations=narrations
    )


def parse_bank_csv(
    path: Path, log: ProblemLog, *, default_date_order: DateOrder = DateOrder.DAY_FIRST
) -> BankRecords:
    """Read and normalise ``bank_statement.csv``."""
    return parse_bank_rows(read_bank_rows(path), log, default_date_order=default_date_order)
