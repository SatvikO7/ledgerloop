"""Hallucination gates: everything the model says is checked against what it was told.

A schema proves the *shape* of an answer. These functions prove its
*provenance*: that every reference the model returned was in the prompt, that
every string it claims to have read is in the text it was given, and that no id
appears in the prose which does not appear in the evidence chain.

That distinction is the whole safety argument for letting a model touch this
system at all. A model that returns ``{"utr": "UTR2026030412345"}`` for a
narration containing no such reference has produced a schema-valid, plausible,
completely invented fact -- and a UTR is a *join key*, so accepting one would
create a match out of nothing. The schema cannot catch it. This can.

THREE GATES, THREE FAILURE MODES
--------------------------------
* :func:`grounded_in_text` -- a value the model claims to have extracted must
  actually occur in the source string. Used by ``parse_narration``.
* :func:`grounded_refs` -- every record key the model reasons from must have
  been in the pack it was sent. Used by ``adjudicate_residual``.
* :func:`prose_names_only_known_records` -- prose may not introduce record ids
  the exception does not already involve. Used by ``explain_exception``, where
  the output is read by a human who will act on it.

Every rejection is counted, never silently swallowed. A run whose model
hallucinated half its answers and whose report said nothing would be worse than
a run with no model at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ledgerloop.ingest.normalize import fold_text
from ledgerloop.models.refs import RecordRef

__all__ = [
    "RECORD_ID_PATTERN",
    "GateResult",
    "grounded_in_text",
    "grounded_refs",
    "prose_names_only_known_records",
]

#: Record identifiers as the generator writes them: ``ORD-2026-000123``,
#: ``PAY-00042``, ``SETL-0091``, ``BNK-00007``, ``UTR2026030412345``.
#:
#: Used to find ids a model *invented* inside free prose, so the pattern has to
#: match the shape rather than a known list -- an id that does not exist is
#: exactly what is being looked for.
RECORD_ID_PATTERN = re.compile(r"\b(?:ORD-[\d-]+|PAY-\d+|SETL-\d+|BNK-\d+|UTR\d+)\b")


@dataclass(frozen=True)
class GateResult:
    """Whether the claim is grounded, and what was wrong if it is not."""

    ok: bool
    reason: str = ""
    offending: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


def grounded_in_text(value: str | None, source: str) -> GateResult:
    """Whether a value the model claims to have read is really in the text.

    Compared after :func:`~ledgerloop.ingest.normalize.fold_text`, so casing and
    separator noise do not cause a false rejection -- the model reading
    ``razorpay software pvt`` out of ``RAZORPAY SOFTWARE PVT`` has read it
    correctly. Anything beyond that is an invention.

    ``None`` is grounded: "I could not find one" is an answer the deterministic
    layer already knows how to handle, and it is the honest one to encourage.
    """
    if value is None:
        return GateResult(True)
    if not value.strip():
        return GateResult(False, "the model returned an empty string, not an absence")
    folded_source = fold_text(source)
    folded_value = fold_text(value)
    if folded_value and folded_value in folded_source:
        return GateResult(True)
    return GateResult(
        False,
        f"{value!r} does not occur in the narration it was extracted from",
        (value,),
    )


def grounded_refs(
    claimed: Sequence[str], supplied: Iterable[str]
) -> GateResult:
    """Whether every reference the model cited was in the pack it was sent.

    A citation to a record the prompt never mentioned is not a discovery -- the
    model has no access to the data -- so it is either a hallucination or a
    coincidence, and neither may reach an evidence chain a controller trusts.
    """
    known = set(supplied)
    unknown = tuple(sorted({ref for ref in claimed if ref not in known}))
    if not unknown:
        return GateResult(True)
    return GateResult(
        False,
        f"cited {len(unknown)} record(s) that were not in the evidence pack: "
        + ", ".join(unknown),
        unknown,
    )


def prose_names_only_known_records(
    prose: str, involved: Sequence[RecordRef]
) -> GateResult:
    """Whether free text mentions only records the exception already involves.

    The output of ``explain_exception`` is read by a person who will act on it,
    so an invented settlement id in a root cause sends them looking for a record
    that does not exist. Ids the exception genuinely involves are fine and in
    fact wanted -- the template prose names them too.
    """
    allowed = {ref.record_id for ref in involved}
    found = set(RECORD_ID_PATTERN.findall(prose))
    unknown = tuple(sorted(found - allowed))
    if not unknown:
        return GateResult(True)
    return GateResult(
        False,
        "prose names record(s) the exception does not involve: " + ", ".join(unknown),
        unknown,
    )
