"""Field-level readers that fail with a diagnosis instead of a stack trace.

Every parser needs the same six or seven conversions -- text, money, date,
timestamp, enum, currency -- and every one of them needs to say *which field*
and *why* when it fails, because that message is what a controller uses to fix
the file. Doing that inline would bury the three parsers in error handling.

So each reader here raises :class:`FieldError`, which carries the field name
and an :class:`~ledgerloop.ingest.problems.IngestProblemCode`. A parser wraps
one record in a single ``try`` and turns any :class:`FieldError` into one
quarantined :class:`~ledgerloop.ingest.problems.IngestProblem`.

**Money never goes through ``int()`` here.** Every amount is routed through
:func:`~ledgerloop.money.parse_minor_units`, the same gate the rest of the
system uses. That is what makes the no-float invariant hold at the one boundary
where it could actually be breached -- text becoming money. ``int()`` would
accept ``"1e3"`` and silently agree with a float-shaped input.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import TypeVar

from ledgerloop.ingest.dates import DateOrder, parse_iso_date, parse_slash_date, parse_timestamp
from ledgerloop.ingest.problems import IngestProblemCode
from ledgerloop.ingest.schemas import value_of
from ledgerloop.models.enums import Currency
from ledgerloop.money import MoneyError, parse_minor_units

__all__ = [
    "RESTKEY",
    "FieldError",
    "optional_money",
    "optional_text",
    "read_currency",
    "read_enum",
    "read_iso_date",
    "read_money",
    "read_slash_date",
    "read_timestamp",
    "reject_ragged_row",
    "require_non_negative",
    "require_text",
]

#: Where ``csv.DictReader`` parks values beyond the header.
#:
#: Named rather than left at the default ``None``, for one concrete reason: a
#: ``None`` key would make the row's provenance payload fail
#: ``dict[str, object]`` validation, turning a diagnosable data defect into an
#: opaque contract violation. With a name, the extra values survive into the
#: audit trail and :func:`reject_ragged_row` can say what is actually wrong.
RESTKEY = "__unparsed__"

_EnumT = TypeVar("_EnumT", bound=StrEnum)


class FieldError(Exception):
    """One unreadable field, already diagnosed."""

    def __init__(self, field: str, code: IngestProblemCode, detail: str) -> None:
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.code = code
        self.detail = detail


def reject_ragged_row(record: Mapping[str, object]) -> None:
    """Refuse a CSV row carrying more values than the header declares.

    Usually broken quoting -- an unescaped comma inside a narration. The named
    fields may look plausible, but which value belongs to which column is no
    longer knowable, so the row is quarantined rather than half-believed.
    """
    extra = record.get(RESTKEY)
    if extra is not None:
        raise FieldError(
            RESTKEY,
            IngestProblemCode.MALFORMED_STRUCTURE,
            f"row carries {len(extra) if isinstance(extra, list) else 1} value(s) beyond "
            "the header; column alignment is not recoverable",
        )


def require_text(record: Mapping[str, object], field: str) -> str:
    """A non-empty string. Absent, blank and whitespace-only are all failures."""
    value = value_of(record, field)
    if value is None:
        raise FieldError(
            field, IngestProblemCode.EMPTY_IDENTIFIER, "required value is absent or blank"
        )
    return str(value).strip()


def optional_text(record: Mapping[str, object], field: str) -> str | None:
    """A string, or ``None`` when the field is absent, null or blank."""
    value = value_of(record, field)
    return None if value is None else str(value).strip()


def read_money(record: Mapping[str, object], field: str) -> int:
    """A minor-unit amount, through the money gate."""
    value = value_of(record, field)
    if value is None:
        raise FieldError(field, IngestProblemCode.MISSING_FIELD, "required amount is absent")
    if isinstance(value, float):
        raise FieldError(
            field,
            IngestProblemCode.MALFORMED_MONEY,
            f"{value!r} arrived as a float; minor units must be integers",
        )
    if not isinstance(value, str | int):
        raise FieldError(
            field,
            IngestProblemCode.MALFORMED_MONEY,
            f"{type(value).__name__} is not a minor-unit amount",
        )
    try:
        return parse_minor_units(value, field=field)
    except MoneyError as exc:
        raise FieldError(field, IngestProblemCode.MALFORMED_MONEY, str(exc)) from exc


def optional_money(record: Mapping[str, object], field: str, *, default: int = 0) -> int:
    """A minor-unit amount, or ``default`` when the field is absent or null."""
    if value_of(record, field) is None:
        return default
    return read_money(record, field)


def require_non_negative(value: int, field: str) -> int:
    """Reject a negative amount in a field whose sign convention forbids it.

    Bank credit and debit columns are separate non-negative fields, mirroring
    how statements are actually published. A negative credit would flip an
    incoming payment into an outgoing one, which is the sign error most likely
    to survive review -- so it is rejected rather than normalised.
    """
    if value < 0:
        raise FieldError(
            field,
            IngestProblemCode.NEGATIVE_AMOUNT,
            f"{value} is negative in a field that must not be",
        )
    return value


def read_enum(record: Mapping[str, object], field: str, enum: type[_EnumT]) -> _EnumT:
    """A closed-vocabulary value, or a failure naming the whole vocabulary."""
    text = require_text(record, field)
    try:
        return enum(text)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum)
        raise FieldError(
            field,
            IngestProblemCode.UNKNOWN_ENUM_VALUE,
            f"{text!r} is not one of: {allowed}",
        ) from exc


def read_currency(record: Mapping[str, object], field: str, *, default: Currency) -> Currency:
    """A currency the money path can actually handle.

    ``Currency.USD`` parses and is then **rejected**, because A11
    FX/multicurrency is cut from the MVP and ``Currency.USD.supported`` is
    ``False``. Rejecting it loudly is the point: mis-scaling dollars as paise
    would produce a reconciliation that balances and is wrong by a factor of
    the exchange rate. The cut is testable rather than merely absent.
    """
    if value_of(record, field) is None:
        return default
    currency = read_enum(record, field, Currency)
    if not currency.supported:
        raise FieldError(
            field,
            IngestProblemCode.UNSUPPORTED_CURRENCY,
            f"{currency.value} is out of scope for the MVP (A11 FX is cut); "
            "the money path is INR-only and would mis-scale it as paise",
        )
    return currency


def read_timestamp(record: Mapping[str, object], field: str) -> datetime:
    """An ISO-8601 timestamp, normalised to naive IST."""
    text = require_text(record, field)
    try:
        return parse_timestamp(text)
    except ValueError as exc:
        raise FieldError(
            field, IngestProblemCode.MALFORMED_DATE, f"{text!r} is not an ISO-8601 timestamp"
        ) from exc


def read_iso_date(record: Mapping[str, object], field: str) -> date:
    """An unambiguous ISO-8601 date."""
    text = require_text(record, field)
    try:
        return parse_iso_date(text)
    except ValueError as exc:
        raise FieldError(
            field, IngestProblemCode.MALFORMED_DATE, f"{text!r} is not an ISO-8601 date"
        ) from exc


def read_slash_date(record: Mapping[str, object], field: str, order: DateOrder) -> date:
    """A ``NN/NN/YYYY`` date, under an order already decided for the column."""
    text = require_text(record, field)
    try:
        return parse_slash_date(text, order)
    except ValueError as exc:
        raise FieldError(
            field,
            IngestProblemCode.MALFORMED_DATE,
            f"{text!r} is not a valid date read as {order.value}",
        ) from exc
