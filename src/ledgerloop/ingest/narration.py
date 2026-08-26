"""Deterministic narration parsing: pull a UTR and a merchant out of free text.

The bank statement has no reference field. Everything a matcher can use is
buried in one string that a bank wrote for a human:

``NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT``

PLAN.md 7.3 specifies **regex first, LLM only on regex miss**. This module is
the regex half, and it is complete on its own: nothing here calls a model, and
the LLM fallback added at step 9 will consume :class:`NarrationParse` rather
than replace it. That ordering is the whole cost argument -- a deterministic
parser that resolves most narrations for nothing leaves the model paying only
for the residue.

WHAT IT EXTRACTS
----------------
``extracted_utr``
    ``UTR`` followed by digits. Absent under anomaly A07 ``MISSING_REFERENCE``,
    which is the point: the field is ``None`` there and T3 has to earn the
    match on the name instead.
``extracted_merchant``
    The counterparty name, recovered by *subtraction* rather than by matching a
    template. Narration formats vary per bank and per rail, so a template list
    would be a maintenance sink and would silently return nothing the first
    time a real statement used a shape nobody anticipated. Instead every token
    that is demonstrably **not** a name is removed -- rail, direction,
    reference, purpose -- and whatever survives is the name.

THE TWO GATES THAT KEEP NOISE ROWS UNMATCHED
--------------------------------------------
``RENT PAYMENT COMMERCIAL PREMISES`` and the nine other noise narrations of
PLAN.md 5.1 must match nothing. Subtraction alone would hand back
``RENT PAYMENT COMMERCIAL PREMISES`` as a merchant name, and while no fuzzy
score would ever accept it, a parser that invents a counterparty for a rent
debit is wrong before the matcher ever sees it. So:

1. **Rail gate.** A merchant is extracted only from a narration naming a
   transfer rail (``NEFT``, ``IMPS``, ``RTGS``, ...) or carrying a UTR. Money
   arriving from a merchant always names the rail it arrived on; an
   electricity bill does not.
2. **Generic-counterparty gate.** ``NEFT CR-DIRECT TRANSFER-370162-INWARD`` --
   the A10 ``ORPHAN_BANK_CREDIT`` shape -- passes the rail gate but names no
   merchant. ``DIRECT``, ``FUND``, ``SELF`` and their kin are descriptors, not
   counterparties, so a segment made only of them yields ``None``.

Both gates are properties of the narration, not of this corpus: neither
consults the merchant vocabulary, so neither can be accused of having been
fitted to the fixture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ledgerloop.ingest.normalize import (
    fold_text,
    merchant_skeleton,
    normalize_merchant_name,
    normalize_narration,
)

__all__ = [
    "CANONICAL_UTR_DIGITS",
    "UTR_PATTERN",
    "NarrationParse",
    "parse_narration",
]

#: ``UTR`` plus its digits, read off the *normalised* narration, where the
#: bank's separators have already become spaces.
#:
#: One optional space is allowed after the prefix because some statements write
#: ``UTR 2026030412345``. The lower bound of 6 digits accepts the truncated
#: references PLAN.md 5.1 mentions; :attr:`NarrationParse.utr_is_truncated`
#: flags them rather than dropping them, so a short reference reaches T3 as a
#: weak signal instead of vanishing as no signal at all.
UTR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bUTR\s?(\d{6,20})\b")

#: Digits in a well-formed UTR: ``YYYYMMDD`` plus a five-digit tail.
CANONICAL_UTR_DIGITS: Final[int] = 13

#: Transfer rails. Their presence is what makes a narration credit-shaped.
_RAILS: Final[frozenset[str]] = frozenset(
    {"NEFT", "IMPS", "RTGS", "UPI", "ACH", "NACH", "ECS", "SWIFT", "FT"}
)

#: Direction markers. Never part of a name.
_DIRECTIONS: Final[frozenset[str]] = frozenset({"CR", "DR"})

#: Purpose and disposition words, stripped from either end of a segment.
_PURPOSE: Final[frozenset[str]] = frozenset(
    {
        "SETTLEMENT",
        "SETTLEMENTS",
        "PAYOUT",
        "PAYOUTS",
        "MERCHANT",
        "BULK",
        "BATCH",
        "INWARD",
        "OUTWARD",
        "TRANSFER",
        "REMITTANCE",
        "PAYMENT",
        "CREDIT",
        "DEBIT",
        "TXN",
        "REF",
    }
)

#: Placeholders a bank writes where a counterparty name would go.
_GENERIC_COUNTERPARTY: Final[frozenset[str]] = frozenset(
    {"DIRECT", "SELF", "FUND", "FUNDS", "ACCOUNT", "AC", "MISC", "SUNDRY", "OTHERS", "THIRD"}
)

#: Segment separators. Banks use these interchangeably with spaces.
_SEGMENT_SPLIT: Final[re.Pattern[str]] = re.compile(r"[-/|,]+")

#: Precomputed unions, so the hot loop does not rebuild a frozenset per segment.
_LEADING_NOISE: Final[frozenset[str]] = _RAILS | _DIRECTIONS | _PURPOSE
_TRAILING_NOISE: Final[frozenset[str]] = _PURPOSE | _DIRECTIONS
_NOT_A_NAME: Final[frozenset[str]] = _GENERIC_COUNTERPARTY | _PURPOSE

_DIGITS_ONLY: Final[re.Pattern[str]] = re.compile(r"^\d+$")


@dataclass(frozen=True)
class NarrationParse:
    """Everything the regex layer recovered from one narration.

    Richer than the three fields :class:`~ledgerloop.models.CanonicalBankTxn`
    stores, deliberately. The record keeps what the matcher joins on; this
    keeps what the *audit trail* and the step-9 LLM fallback need in order to
    explain or improve a miss -- which tokens were discarded, whether the rail
    gate fired, whether a reference was present but short.
    """

    raw: str
    normalized: str
    utr: str | None = None
    utr_is_truncated: bool = False
    merchant: str | None = None
    merchant_normalized: str | None = None
    merchant_skeleton: str | None = None
    rail: str | None = None
    reference_tokens: tuple[str, ...] = ()
    discarded_segments: tuple[str, ...] = ()

    @property
    def is_credit_shaped(self) -> bool:
        """Whether this narration looks like an inbound transfer at all."""
        return self.rail is not None or self.utr is not None

    @property
    def resolved_by_regex(self) -> bool:
        """Whether the deterministic layer recovered anything joinable.

        Step 9 batches the narrations where this is ``False`` to the LLM, and
        the cost ledger reports the ratio. Nothing else routes on it.
        """
        return self.utr is not None or self.merchant is not None


def _strip_edges(words: list[str]) -> list[str]:
    """Remove rail, direction and purpose words from both ends of a segment.

    Edges only. A purpose word in the middle of a segment is far more likely to
    be part of the name (``BOAT LIFESTYLE RETAIL``) than a stray marker, and
    stripping it everywhere would quietly shorten real merchant names.
    """
    head = 0
    tail = len(words)
    while head < tail and words[head] in _LEADING_NOISE:
        head += 1
    while tail > head and words[tail - 1] in _TRAILING_NOISE:
        tail -= 1
    return words[head:tail]


def _is_reference(word: str) -> bool:
    """Digit runs and UTR tokens: references, never names."""
    return bool(_DIGITS_ONLY.match(word)) or word.startswith("UTR")


def _could_be_a_name(words: list[str]) -> bool:
    """Whether what survived subtraction could name a counterparty.

    Two ways it could not: nothing survived, or everything surviving is a
    placeholder a bank writes *instead* of a name -- the ``DIRECT TRANSFER`` of
    the A10 orphan-credit shape.

    There is no third check for "contains no letters", because a surviving word
    always contains one: :func:`~ledgerloop.ingest.normalize.normalize_narration`
    restricts the alphabet to ``[0-9A-Z ]``, and :func:`_is_reference` has
    already removed every all-digit token.
    """
    if not words:
        return False
    return not all(word in _NOT_A_NAME for word in words)


def _pick_name(
    candidates: list[tuple[int, list[str]]], discarded: list[str]
) -> str | None:
    """Choose the merchant name from the segments that survived subtraction.

    **Adjacent candidates are one name, not two.** The separator a bank uses
    between fields is also a character that appears inside names:
    ``NEFT CR-NYKAA E-RETAIL PRIVATE-UTR...-SETTLEMENT`` splits into
    ``NYKAA E`` and ``RETAIL PRIVATE``, two fragments of one merchant. Runs of
    consecutive surviving segments are therefore rejoined -- nothing separates
    them but the very character that split the name.

    Between runs, the longest wins. A merchant name is the widest free-text
    field in a narration, and the failure modes are asymmetric: an over-long
    name costs a few fuzzy points at T3, while a truncated one loses the token
    that distinguished the merchant in the first place.
    """
    if not candidates:
        return None

    runs: list[list[str]] = []
    previous_index: int | None = None
    for index, words in candidates:
        if previous_index is not None and index == previous_index + 1:
            runs[-1].extend(words)
        else:
            runs.append(list(words))
        previous_index = index

    best = max(runs, key=len)
    discarded.extend(" ".join(run) for run in runs if run is not best)
    return " ".join(best)


def parse_narration(raw: str) -> NarrationParse:
    """Extract UTR and merchant from one bank narration. Pure and deterministic.

    Never raises: an unreadable narration yields a parse with everything
    ``None``, because a bank statement row that cannot be understood is an
    exception for a human, not a crash for the pipeline.
    """
    normalized = normalize_narration(raw)
    if not normalized:
        return NarrationParse(raw=raw, normalized=normalized)

    utr_match = UTR_PATTERN.search(normalized)
    utr: str | None = None
    truncated = False
    if utr_match is not None:
        digits = utr_match.group(1)
        utr = f"UTR{digits}"
        truncated = len(digits) < CANONICAL_UTR_DIGITS

    words = normalized.split()
    rail = next((word for word in words if word in _RAILS), None)

    # Segment on the *folded* text, so the source separators still delimit
    # fields -- `normalize_narration` has already flattened them into spaces,
    # which is exactly the information segmentation needs.
    segments = [segment.strip() for segment in _SEGMENT_SPLIT.split(fold_text(raw))]

    reference_tokens: list[str] = []
    discarded: list[str] = []
    candidates: list[tuple[int, list[str]]] = []

    for index, segment in enumerate(segments):
        segment_words = [word for word in normalize_narration(segment).split() if word]
        if not segment_words:
            continue
        reference_tokens.extend(word for word in segment_words if _DIGITS_ONLY.match(word))

        kept = [word for word in _strip_edges(segment_words) if not _is_reference(word)]
        if _could_be_a_name(kept):
            candidates.append((index, kept))
        else:
            discarded.append(" ".join(segment_words))

    merchant = _pick_name(candidates, discarded)

    # The rail gate. Applied after extraction rather than before, so the
    # discarded-segment trail is populated either way and an operator can see
    # what the gate refused rather than only that it refused.
    if merchant is not None and rail is None and utr is None:
        discarded.append(merchant)
        merchant = None

    return NarrationParse(
        raw=raw,
        normalized=normalized,
        utr=utr,
        utr_is_truncated=truncated,
        merchant=merchant,
        merchant_normalized=normalize_merchant_name(merchant) if merchant else None,
        merchant_skeleton=merchant_skeleton(merchant) if merchant else None,
        rail=rail,
        reference_tokens=tuple(reference_tokens),
        discarded_segments=tuple(discarded),
    )
