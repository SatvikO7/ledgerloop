"""What ingest does when a record is wrong.

A reconciliation run that crashes on one bad row is useless in operations: the
one file that arrives malformed is the one the controller most needs to see.
So ingest **quarantines** rather than raises. Every rejected record produces an
:class:`IngestProblem` carrying the source, the position, the offending field
and the raw payload, and the run continues with the records that parsed.

Two failure levels, deliberately different:

* **Row level** -> quarantine. One bad amount does not invalidate the file.
* **File level** -> :class:`IngestError`. A missing required column means every
  row would be misread, and continuing would produce a confidently wrong
  answer rather than a loud one. Structural failures raise.

``strict=True`` promotes row-level problems to :class:`IngestError` too. That
is what the test suite and CI use, so a fixture that starts quarantining rows
fails the build instead of quietly shrinking the corpus.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel
from ledgerloop.models.enums import SourceName

__all__ = ["IngestError", "IngestProblem", "IngestProblemCode", "ProblemLog"]


class IngestProblemCode(StrEnum):
    """Why a record or a file was rejected.

    Kept in the ingest package rather than ``models/enums.py``: these are
    parser diagnostics, not part of the reconciliation vocabulary the matcher
    and evaluator share. Nothing downstream branches on them -- they are read
    by humans and counted in the report.
    """

    MISSING_COLUMN = "MISSING_COLUMN"
    MISSING_FIELD = "MISSING_FIELD"
    EMPTY_IDENTIFIER = "EMPTY_IDENTIFIER"
    DUPLICATE_ID = "DUPLICATE_ID"
    MALFORMED_MONEY = "MALFORMED_MONEY"
    MALFORMED_DATE = "MALFORMED_DATE"
    MALFORMED_STRUCTURE = "MALFORMED_STRUCTURE"
    UNKNOWN_ENUM_VALUE = "UNKNOWN_ENUM_VALUE"
    UNSUPPORTED_CURRENCY = "UNSUPPORTED_CURRENCY"
    NEGATIVE_AMOUNT = "NEGATIVE_AMOUNT"
    CONTRADICTORY_SIGNS = "CONTRADICTORY_SIGNS"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


class IngestError(ValueError):
    """A failure ingest refuses to continue past.

    Raised for structural problems (a missing column, a JSON document of the
    wrong shape, a date column that is internally contradictory), and for any
    problem at all when ``strict=True``.
    """


class IngestProblem(FrozenLedgerModel):
    """One quarantined record, with everything needed to fix it by hand."""

    source: SourceName
    source_line: int
    code: IngestProblemCode
    detail: str
    field: str | None = None
    record_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)

    def __str__(self) -> str:
        where = f"{self.source.value}[{self.source_line}]"
        if self.record_id:
            where += f" {self.record_id}"
        if self.field:
            where += f".{self.field}"
        return f"{where}: {self.code.value} -- {self.detail}"


def _displayable(payload: dict[str, object] | None) -> dict[str, object]:
    """Coerce a payload's keys to strings so recording a problem cannot fail.

    The payload is diagnostic display data, not a contract: it exists so a
    controller can see the record that was rejected. When the *payload itself*
    is what is malformed -- a source dict with a non-string key -- validating
    it strictly would make the error handler raise while handling an error,
    and the run would die on the one row it was built to survive.
    """
    if not payload:
        return {}
    return {str(key): value for key, value in payload.items()}


class ProblemLog:
    """Collects quarantined records, or raises immediately in strict mode.

    Mutable by design and deliberately not a Pydantic model: it is a working
    accumulator threaded through the parsers, and the frozen
    :class:`IngestProblem` records it holds are the contract.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._problems: list[IngestProblem] = []
        self.strict = strict

    def record(
        self,
        *,
        source: SourceName,
        source_line: int,
        code: IngestProblemCode,
        detail: str,
        field: str | None = None,
        record_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> IngestProblem:
        problem = IngestProblem(
            source=source,
            source_line=source_line,
            code=code,
            detail=detail,
            field=field,
            record_id=record_id,
            payload=_displayable(payload),
        )
        if self.strict:
            raise IngestError(str(problem))
        self._problems.append(problem)
        return problem

    @property
    def problems(self) -> tuple[IngestProblem, ...]:
        return tuple(self._problems)

    def __len__(self) -> int:
        return len(self._problems)

    def __bool__(self) -> bool:
        return bool(self._problems)
