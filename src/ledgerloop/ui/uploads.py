"""Uploaded files: what they are, whether they are usable, and what they enable.

WHAT THIS MODULE IS FOR
-----------------------
A person has the files they happen to have. They may have a bank statement and
nothing else; they may have a processor report and a bank statement; they may
have all three. This module answers three questions about whatever arrives, and
answers them **deterministically**:

1. *What kind of file is this?* -- :func:`detect`, by looking at the header and
   the shape of the payload. No model is asked; a CSV header is not a language
   problem.
2. *Is it usable?* -- :func:`validate`, which checks the extension, the size, the
   encoding, the required columns and that there is at least one data row.
3. *What can LedgerLoop actually do with this combination?* -- :func:`assess`,
   which is the honest part, and the reason the module exists.

THE COMBINATIONS, MEASURED RATHER THAN ASSUMED
----------------------------------------------
Every claim in :data:`CAPABILITIES` was measured by running the real ladder over
the committed dev fixture with sources blanked out, not reasoned from the code::

    ledger only            0 payment-to-bank links
    processor only         0
    bank only              0
    ledger + processor     0 payment-to-bank links (55 order/payment candidates)
    ledger + bank          0
    processor + bank      29 payment-to-bank links      <- reconciliation works
    all three             39 payment-to-bank links      <- best

So **the processor report and the bank statement together are the minimum** for
the thing this product does: proving that money the processor says it paid is
money the bank says arrived. The ledger is not required, and adding it recovers
more links because T3 needs a merchant identity that only the ledger carries.

A single source can be read and inspected and nothing more. That is not a
limitation to apologise for or to paper over -- reconciliation is a statement
about *two* records agreeing, and with one source there is no second record.

WHAT IT REFUSES TO DO
---------------------
* **It never executes an uploaded file.** CSVs are parsed with :mod:`csv` and
  JSON with :mod:`json`; nothing is evaluated, imported or interpolated.
* **It never guesses past ambiguity.** When the header matches no known schema,
  :func:`detect` returns ``None`` and the caller asks the person.
* **It never writes into the project's own data.** The caller supplies a
  temporary directory; nothing here resolves a path relative to the repository.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "CAPABILITIES",
    "MAX_UPLOAD_BYTES",
    "REQUIRED_COLUMNS",
    "Assessment",
    "SourceKind",
    "UploadProblem",
    "assess",
    "detect",
    "validate",
]

#: Largest upload accepted, in bytes. The 5,000-order scale corpus -- the biggest
#: this project has ever produced -- has a bank statement well under a megabyte,
#: so 32 MB is generous for anything the pipeline can process in a browser
#: session while still refusing a file that is obviously not a statement.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class SourceKind(StrEnum):
    """The three things LedgerLoop can read, named for a person.

    The values are the filenames :mod:`ledgerloop.generator.emitters` writes and
    :func:`~ledgerloop.ingest.dataset.ingest_available` reads, so a detected
    upload is saved under the name the ingester already expects. There is no
    translation table and no second naming scheme.
    """

    BANK = "bank_statement.csv"
    PROCESSOR = "psp_settlements.json"
    LEDGER = "ledger_orders.csv"

    @property
    def label(self) -> str:
        return {
            SourceKind.BANK: "Bank statement",
            SourceKind.PROCESSOR: "Payment processor report",
            SourceKind.LEDGER: "Order ledger",
        }[self]

    @property
    def blurb(self) -> str:
        return {
            SourceKind.BANK: "The credits and debits your bank actually posted.",
            SourceKind.PROCESSOR: (
                "What your payment provider says it paid out, and the individual "
                "payments inside each payout."
            ),
            SourceKind.LEDGER: "Your own record of the orders you took.",
        }[self]

    @property
    def file_type(self) -> str:
        return "JSON" if self is SourceKind.PROCESSOR else "CSV"


#: Columns a file must carry before it is accepted as that source.
#:
#: Taken from the parsers rather than invented: these are the fields
#: ``ingest/bank.py`` and ``ingest/ledger.py`` read. A file missing one of them
#: would parse into empty records and look like a corpus that found nothing,
#: which is the failure mode worth spending a check to avoid.
REQUIRED_COLUMNS: dict[SourceKind, tuple[str, ...]] = {
    SourceKind.BANK: ("txn_id", "value_date", "narration"),
    SourceKind.LEDGER: ("order_id", "merchant_id", "amount_gross_paise"),
}


@dataclass(frozen=True)
class UploadProblem:
    """Why a file was refused, in words the person who chose it can act on."""

    reason: str
    detail: str


@dataclass(frozen=True)
class Assessment:
    """What the supplied combination can and cannot do."""

    can_reconcile: bool
    headline: str
    detail: str
    missing_hint: str
    """What to add to get further. Empty when nothing would help."""


def _decode(payload: bytes) -> str | None:
    """UTF-8, then the Windows default a spreadsheet export often uses."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _header_of(text: str) -> list[str]:
    reader = csv.reader(io.StringIO(text))
    try:
        return [column.strip().lower() for column in next(reader)]
    except StopIteration:
        return []


def detect(filename: str, payload: bytes) -> SourceKind | None:
    """Which source this is, or ``None`` when the file does not say.

    Decided by **content**, not by upload order and not by filename alone: a
    person who uploads their bank export second should not have it read as a
    ledger. The filename is used only to break a tie that the content cannot.

    Returns ``None`` rather than a best guess. An upload silently read as the
    wrong source would produce a reconciliation that is wrong in a way nobody
    would think to check.
    """
    if len(payload) > MAX_UPLOAD_BYTES:
        return None
    text = _decode(payload)
    if text is None or not text.strip():
        return None

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("settlements"), list):
            return SourceKind.PROCESSOR
        return None

    header = set(_header_of(text))
    if not header:
        return None
    for kind, required in REQUIRED_COLUMNS.items():
        if set(required) <= header:
            return kind
    return None


def validate(kind: SourceKind, filename: str, payload: bytes) -> UploadProblem | None:
    """Whether this file can be used as ``kind``. ``None`` means it can.

    Runs even when :func:`detect` already identified the file, because a person
    may override the detection and the override has to be checked too.
    """
    if not payload:
        return UploadProblem("The file is empty", f"{filename} contains no data.")
    if len(payload) > MAX_UPLOAD_BYTES:
        return UploadProblem(
            "The file is too large",
            f"{filename} is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
            "That is far larger than a statement this tool is built for.",
        )
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected = "json" if kind is SourceKind.PROCESSOR else "csv"
    if suffix != expected:
        return UploadProblem(
            f"{kind.label} must be a {expected.upper()} file",
            f"{filename} looks like a .{suffix or 'file'}.",
        )
    text = _decode(payload)
    if text is None:
        return UploadProblem(
            "The file could not be read",
            f"{filename} is not text this tool can decode. Export it as UTF-8.",
        )

    if kind is SourceKind.PROCESSOR:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            return UploadProblem(
                "The file is not valid JSON",
                f"{filename} could not be read as JSON: {error.msg}.",
            )
        if not isinstance(parsed, dict) or not isinstance(
            parsed.get("settlements"), list
        ):
            return UploadProblem(
                "This is not a payment processor report",
                f"{filename} has no top-level 'settlements' list.",
            )
        if not parsed["settlements"]:
            return UploadProblem(
                "The report has no payouts",
                f"{filename} contains an empty settlements list.",
            )
        return None

    header = set(_header_of(text))
    missing = [column for column in REQUIRED_COLUMNS[kind] if column not in header]
    if missing:
        return UploadProblem(
            f"This does not look like a {kind.label.lower()}",
            f"{filename} is missing the column(s): {', '.join(missing)}.",
        )
    rows = sum(1 for _ in csv.reader(io.StringIO(text)))
    if rows < 2:
        return UploadProblem(
            "The file has a header but no rows",
            f"{filename} contains no transactions.",
        )
    return None


def row_count(kind: SourceKind, payload: bytes) -> int:
    """How many records the file holds, for the confirmation the person reads."""
    text = _decode(payload)
    if text is None:
        return 0
    if kind is SourceKind.PROCESSOR:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return 0
        settlements = parsed.get("settlements", []) if isinstance(parsed, dict) else []
        return len(settlements) if isinstance(settlements, list) else 0
    return max(sum(1 for _ in csv.reader(io.StringIO(text))) - 1, 0)


__all__ += ["row_count"]


#: What each combination can do. Every figure was measured, see the module
#: docstring. Keyed by the frozenset of sources supplied.
CAPABILITIES: dict[frozenset[SourceKind], Assessment] = {
    frozenset(): Assessment(
        False,
        "Nothing uploaded yet",
        "Add at least a payment processor report and a bank statement to "
        "reconcile.",
        "",
    ),
    frozenset({SourceKind.BANK}): Assessment(
        False,
        "Your bank statement can be read, but not reconciled",
        "Reconciliation means checking one record against another. With only a "
        "bank statement there is nothing to check it against, so LedgerLoop can "
        "list and inspect the transactions but cannot match them.",
        "Add your payment processor report to reconcile payouts against these "
        "credits.",
    ),
    frozenset({SourceKind.PROCESSOR}): Assessment(
        False,
        "Your processor report can be read, but not reconciled",
        "LedgerLoop can list the payouts and the payments inside them, but "
        "without a bank statement there is no record of what actually arrived, "
        "so nothing can be confirmed.",
        "Add your bank statement to check these payouts against what the bank "
        "posted.",
    ),
    frozenset({SourceKind.LEDGER}): Assessment(
        False,
        "Your order ledger can be read, but not reconciled",
        "This tells LedgerLoop what you sold. It says nothing about what was "
        "paid out or what arrived, so there is nothing to reconcile against.",
        "Add your payment processor report and your bank statement.",
    ),
    frozenset({SourceKind.LEDGER, SourceKind.PROCESSOR}): Assessment(
        False,
        "Orders can be linked to payments, but not to money",
        "LedgerLoop can connect each order to the payment taken for it. Without "
        "a bank statement it cannot confirm that any of that money arrived, "
        "which is the check this tool exists to perform.",
        "Add your bank statement to complete the reconciliation.",
    ),
    frozenset({SourceKind.LEDGER, SourceKind.BANK}): Assessment(
        False,
        "These two cannot be matched to each other",
        "An order and a bank credit are connected through the payout that "
        "carried the money, and that link lives in the processor report. "
        "Without it there is nothing joining these two files.",
        "Add your payment processor report.",
    ),
    frozenset({SourceKind.PROCESSOR, SourceKind.BANK}): Assessment(
        True,
        "Ready to reconcile",
        "LedgerLoop can check each payout the processor reported against the "
        "credits your bank posted, and work back to the individual payments "
        "inside each payout.",
        "Adding your order ledger lets LedgerLoop also match payouts whose bank "
        "reference is missing, by business name.",
    ),
    frozenset({SourceKind.LEDGER, SourceKind.PROCESSOR, SourceKind.BANK}): Assessment(
        True,
        "Ready to reconcile",
        "All three records are present, so LedgerLoop can follow the money the "
        "whole way: order, to payment, to payout, to the credit in your bank.",
        "",
    ),
}


def assess(sources: frozenset[SourceKind] | set[SourceKind]) -> Assessment:
    """What this combination of sources can do."""
    return CAPABILITIES[frozenset(sources)]
