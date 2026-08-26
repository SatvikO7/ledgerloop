"""Date parsing, and the DD/MM ambiguity the bank statement is built around.

``06/03/2026`` is 6 March under the Indian convention and 3 June under the
American one. Nothing in the row says which. Getting it wrong shifts every
affected transaction by up to eleven months, which silently destroys T1 (a
three-day date window) and T2 (settlement-anchored bucketing) while leaving T0
untouched -- so the failure would show up as an unexplained recall collapse in
the residual tiers rather than as a parse error.

HOW THE AMBIGUITY IS RESOLVED
-----------------------------
Per row, it cannot be. Per *file*, it usually can: a statement covering more
than a fortnight almost certainly contains at least one day past the 12th, and
``14/03/2026`` can only be day-first. So the column is read twice --

1. **Scan** every value. A first component above 12 witnesses day-first; a
   second component above 12 witnesses month-first.
2. **Decide** from the witnesses, and record how the decision was reached.
3. **Parse** every row under that one decision.

Three outcomes, all of them explicit:

* **Proven.** Witnesses for exactly one order. That order is used and
  :attr:`DateOrderEvidence.proven` is ``True``.
* **Contradictory.** Witnesses for both. The file cannot be read under any
  single convention, so ingest raises rather than picking a winner -- half the
  rows would be silently wrong either way.
* **Genuinely ambiguous.** No witnesses at all, e.g. a short statement that
  happens to stay inside the first twelve days of a month. The configured
  default is applied and ``proven`` is ``False``, so a report can say the
  dates rest on a convention rather than on evidence.

The generator emits ``%d/%m/%Y``, so the fixture proves day-first on its own
data -- but the inference is what makes that a *measured* fact rather than a
hardcoded assumption about the file.

TIMEZONES
---------
:func:`parse_timestamp` converts any offset-aware timestamp to IST and drops
the tzinfo. PLAN.md 5.1 shows the PSP writing ``+05:30`` while the generator
writes naive local time, and mixing the two raises ``TypeError`` at the first
comparison inside T1. The corpus is one Indian entity on one clock; making
that explicit here is cheaper than defending every later comparison.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from typing import Final

__all__ = [
    "IST",
    "DateOrder",
    "DateOrderEvidence",
    "infer_date_order",
    "parse_iso_date",
    "parse_slash_date",
    "parse_timestamp",
]

#: Indian Standard Time. The single clock the corpus is expressed in.
IST: Final[timezone] = timezone(timedelta(hours=5, minutes=30), name="IST")

_SLASH_DATE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$")

_MONTHS_PER_YEAR: Final[int] = 12


class DateOrder(StrEnum):
    """Which component of ``NN/NN/YYYY`` is the day."""

    DAY_FIRST = "DAY_FIRST"
    MONTH_FIRST = "MONTH_FIRST"


@dataclass(frozen=True)
class DateOrderEvidence:
    """What the column-wide scan concluded, and on what basis.

    Carried in the ingest result and printed in the report. A date convention
    inferred from data is a claim, and a claim that cannot be audited is worth
    very little -- so the witness counts travel with the answer.
    """

    order: DateOrder
    proven: bool
    day_first_witnesses: int
    month_first_witnesses: int
    ambiguous_values: int
    unparsable_values: int
    total_values: int

    @property
    def basis(self) -> str:
        """One line a report can print verbatim."""
        if self.proven:
            witnesses = (
                self.day_first_witnesses
                if self.order is DateOrder.DAY_FIRST
                else self.month_first_witnesses
            )
            return (
                f"{self.order.value} proven by {witnesses} of {self.total_values} values "
                f"whose leading component exceeds {_MONTHS_PER_YEAR}"
            )
        return (
            f"{self.order.value} assumed by convention: all {self.total_values} values "
            f"are readable both ways"
        )


def infer_date_order(
    values: Iterable[str], *, default: DateOrder = DateOrder.DAY_FIRST
) -> DateOrderEvidence:
    """Decide day-first vs month-first for a whole column of ``NN/NN/YYYY`` values.

    ``default`` applies only when the column carries no witness either way. It
    is day-first because every source in this project is Indian, and because
    that is what PLAN.md 5.1 specifies the bank statement uses.

    Raises :class:`ValueError` when the column witnesses *both* orders: no
    single convention can read the file, and choosing one would silently
    corrupt the rows supporting the other.
    """
    day_first = 0
    month_first = 0
    ambiguous = 0
    unparsable = 0
    total = 0

    for raw in values:
        total += 1
        match = _SLASH_DATE.match(raw)
        if match is None:
            unparsable += 1
            continue
        first, second = int(match.group(1)), int(match.group(2))
        first_is_day_only = first > _MONTHS_PER_YEAR
        second_is_day_only = second > _MONTHS_PER_YEAR
        if first_is_day_only and second_is_day_only:
            raise ValueError(
                f"{raw!r} has both components above {_MONTHS_PER_YEAR}; it is not a date"
            )
        if first_is_day_only:
            day_first += 1
        elif second_is_day_only:
            month_first += 1
        else:
            ambiguous += 1

    if day_first and month_first:
        raise ValueError(
            f"date column witnesses both orders ({day_first} day-first, "
            f"{month_first} month-first); no single convention reads this file"
        )

    if day_first:
        order, proven = DateOrder.DAY_FIRST, True
    elif month_first:
        order, proven = DateOrder.MONTH_FIRST, True
    else:
        order, proven = default, False

    return DateOrderEvidence(
        order=order,
        proven=proven,
        day_first_witnesses=day_first,
        month_first_witnesses=month_first,
        ambiguous_values=ambiguous,
        unparsable_values=unparsable,
        total_values=total,
    )


def parse_slash_date(raw: str, order: DateOrder) -> date:
    """Parse ``NN/NN/YYYY`` under an already-decided component order.

    Takes the order rather than inferring it, because inference is a property
    of the column and this function only ever sees one row. Raises
    :class:`ValueError` on anything that is not a well-formed date under that
    order -- ``31/02/2026`` included, since a bad date is a data error and
    rolling it forward to 3 March would be a fabrication.
    """
    match = _SLASH_DATE.match(raw)
    if match is None:
        raise ValueError(f"{raw!r} is not a DD/MM/YYYY or MM/DD/YYYY date")
    first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    day, month = (first, second) if order is DateOrder.DAY_FIRST else (second, first)
    return date(year, month, day)


def parse_iso_date(raw: str) -> date:
    """Parse an unambiguous ISO-8601 date (``2026-03-06``)."""
    return date.fromisoformat(raw.strip())


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp, returning naive IST.

    Offset-aware input is converted to IST and stripped of its tzinfo; naive
    input is taken as already being IST. The result is therefore always
    comparable to every other timestamp in the run -- see the module docstring.
    """
    parsed = datetime.fromisoformat(raw.strip())
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(IST).replace(tzinfo=None)
