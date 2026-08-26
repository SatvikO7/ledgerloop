"""Text normalisation: identifiers, merchant names, narration.

This module exists because two strings that *mean* the same thing routinely do
not *compare* equal, and every tier above T0 is an attempt to recover from
that. Normalising once, at ingest, is what lets T0 stay an exact join instead
of accumulating special cases.

Three families of transformation, each with a different consumer:

* **Identifiers** (:func:`normalize_order_ref`) -- feed T0's exact join. The
  PSP mangles roughly a fifth of its order references (PLAN.md 5.1): ``null``,
  ``"ord 2026 004821"``, and ``ORD-2026-004821`` written with U+2011
  NON-BREAKING HYPHEN, which renders identically to ASCII ``-`` and never
  compares equal to it. Two of those three are recoverable; this recovers them.
* **Merchant names** (:func:`normalize_merchant_name`, :func:`merchant_skeleton`)
  -- feed T3. A bank statement writes ``RZRPAY SFTWR P L`` where the PSP writes
  ``Razorpay Software Private Limited``.
* **Narration** (:func:`normalize_narration`) -- feeds both T3's token matching
  and the regex parser in :mod:`ledgerloop.ingest.narration`.

WHY A CONSONANT SKELETON
------------------------
``generator/vocab.py`` argues the case for cutting embeddings: ``RZRPAY SFTWR
P L`` does not sit near ``Razorpay Software Private Limited`` in
sentence-embedding space, because MiniLM has no reason to relate a consonant
skeleton to its expansion. :func:`merchant_skeleton` relates them directly --
it *is* the transformation the bank clerk applied. Dropping interior vowels and
collapsing doubled letters maps both forms onto ``RZRPYSFTWR``, exactly.

Every function here is pure, deterministic, and nowhere near the money path.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

__all__ = [
    "LEGAL_SUFFIXES",
    "ORDER_REF_PATTERN",
    "fold_text",
    "is_order_ref_shaped",
    "merchant_skeleton",
    "normalize_identifier",
    "normalize_merchant_name",
    "normalize_narration",
    "normalize_order_ref",
    "normalize_utr",
]

#: Every dash-like codepoint a source might use where ASCII ``-`` was meant.
#:
#: NFKC alone is not enough: U+2011 decomposes to U+2010 HYPHEN, not to
#: U+002D HYPHEN-MINUS, so a pure NFKC pass leaves the PSP's corrupted order
#: references still unequal to the ledger's. Written as codepoints for the same
#: reason ``generator/baseline.py`` uses ``chr()`` -- so an editor, a paste or a
#: linter autofix cannot silently normalise this table's own contents back to
#: ASCII and quietly delete the thing it defends against.
_DASHES: Final[tuple[int, ...]] = (
    0x2010,  # HYPHEN
    0x2011,  # NON-BREAKING HYPHEN -- the generator's reference corruption
    0x2012,  # FIGURE DASH
    0x2013,  # EN DASH
    0x2014,  # EM DASH
    0x2015,  # HORIZONTAL BAR
    0x2212,  # MINUS SIGN
    0xFE58,  # SMALL EM DASH
    0xFE63,  # SMALL HYPHEN-MINUS
    0xFF0D,  # FULLWIDTH HYPHEN-MINUS
)
_DASH_TABLE: Final[dict[int, str]] = dict.fromkeys(_DASHES, "-")

#: The order-reference grammar the ledger publishes: ``ORD-2026-004821``.
ORDER_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^ORD-\d{4}-\d{6}$")

#: Tokens stripped from the *tail* of a merchant name, iteratively.
#:
#: Tail-only and iterative, because position carries meaning: ``URBAN COMPANY
#: TECH LTD`` must lose ``LTD`` and keep ``COMPANY``. ``P`` and ``L`` are here
#: because ``P L`` is how an Indian bank statement abbreviates "Private
#: Limited" -- ``RZRPAY SFTWR P L``.
LEGAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "PRIVATE",
        "PVT",
        "PVTLTD",
        "LIMITED",
        "LTD",
        "LLP",
        "LLC",
        "INC",
        "PLC",
        "CORP",
        "CORPORATION",
        "CO",
        "P",
        "L",
    }
)

_VOWELS: Final[frozenset[str]] = frozenset("AEIOU")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9A-Z]+")
_NON_ALNUM_KEEP_SPACE = re.compile(r"[^0-9A-Z ]+")


def fold_text(raw: str) -> str:
    """Make a string comparable: NFKC, dashes, accents, case, whitespace.

    The single entry point for the whole module. Four passes, in an order that
    matters:

    1. **NFKC**, which folds compatibility forms -- fullwidth Latin letters
       (U+FF21 and up) become ASCII, superscript digits become digits.
    2. **Dash folding**, because NFKC does *not* reach ASCII here: U+2011
       decomposes to U+2010 HYPHEN, never to U+002D HYPHEN-MINUS.
    3. **Accent stripping** via NFD plus removal of combining marks, so
       ``NESTLÉ`` and ``NESTLE`` are the same counterparty. Dropping the
       accented letter outright would lose a character the two forms share.
    4. **Case and whitespace**.

    Scripts with no ASCII fold at all -- Devanagari digits, CJK, Greek -- are
    left as they are here and dropped by the callers that require ASCII. That
    is deliberate: the identifiers and merchant names this project reconciles
    are ASCII by construction, and inventing a transliteration would be a
    guess rather than a normalisation.
    """
    folded = unicodedata.normalize("NFKC", raw).translate(_DASH_TABLE)
    decomposed = unicodedata.normalize("NFD", folded)
    unaccented = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return _WHITESPACE.sub(" ", unaccented).strip().upper()


def normalize_identifier(raw: str | None) -> str | None:
    """Canonicalise an identifier: ``"ord 2026 004821"`` to ``"ORD-2026-004821"``.

    Every run of non-alphanumeric characters becomes a single ``-``, so spaces,
    underscores, dots, doubled hyphens and the non-breaking hyphen all collapse
    onto the same separator. Returns ``None`` when there is nothing to recover:
    a null, a blank, or a string with no alphanumeric content at all.
    """
    if raw is None:
        return None
    folded = fold_text(raw)
    if not folded:
        return None
    collapsed = _NON_ALNUM.sub("-", folded).strip("-")
    return collapsed or None


def normalize_order_ref(raw: str | None) -> str | None:
    """Recover the ledger's order reference from the PSP's copy of it.

    Deliberately **shape-agnostic**: it returns the canonical string form of
    whatever it was given and does not check the result against the ledger's
    grammar. Deciding whether a reference names a real order is T0's job, and a
    normaliser that silently dropped anything it did not recognise would delete
    exactly the evidence an exception needs in order to explain itself. Use
    :func:`is_order_ref_shaped` where the grammar matters.
    """
    return normalize_identifier(raw)


def is_order_ref_shaped(text: str | None) -> bool:
    """Whether a normalised reference matches the published order grammar."""
    return text is not None and ORDER_REF_PATTERN.match(text) is not None


def normalize_utr(raw: str | None) -> str | None:
    """Canonicalise a UTR: upper-case, alphanumerics only."""
    if raw is None:
        return None
    stripped = _NON_ALNUM.sub("", fold_text(raw))
    return stripped or None


def normalize_narration(raw: str) -> str:
    """Flatten free-text narration into space-separated upper-case tokens.

    ``NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT`` becomes
    ``NEFT CR RAZORPAY SOFTWARE PVT UTR2026030412345 SETTLEMENT``. Punctuation
    carries no information here -- banks use ``-``, ``/`` and spaces
    interchangeably as field separators -- so it becomes whitespace, and token
    comparison works without every consumer having to re-learn that.
    """
    folded = fold_text(raw)
    spaced = _NON_ALNUM_KEEP_SPACE.sub(" ", folded)
    return _WHITESPACE.sub(" ", spaced).strip()


def _strip_legal_suffixes(words: list[str]) -> list[str]:
    """Drop trailing legal-form tokens, never emptying the name."""
    trimmed = list(words)
    while len(trimmed) > 1 and trimmed[-1] in LEGAL_SUFFIXES:
        trimmed.pop()
    return trimmed


def normalize_merchant_name(raw: str) -> str:
    """Strip punctuation and trailing legal form from a merchant name.

    ``"Razorpay Software Private Limited"``, ``"RAZORPAY SOFTWARE PVT"`` and
    ``"RAZORPAY SOFTWARE PRIVATE LTD"`` all normalise to ``"RAZORPAY SOFTWARE"``.
    """
    words = normalize_narration(raw).split()
    if not words:
        return ""
    return " ".join(_strip_legal_suffixes(words))


def _word_skeleton(word: str) -> str:
    """First character, then the consonants of the rest.

    The leading character survives even when it is a vowel, because bank
    abbreviations keep it: ``INSTAMART`` to ``INSTMRT``, ``URBAN`` to ``URBN``.
    Dropping it would make the two forms disagree on their first letter, which
    is the character a fuzzy matcher weighs most heavily.
    """
    return word[0] + "".join(ch for ch in word[1:] if ch not in _VOWELS)


def _collapse_runs(text: str) -> str:
    """``SWGGY`` to ``SWGY``. Doubled letters are the other thing abbreviations drop."""
    out: list[str] = []
    for ch in text:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def merchant_skeleton(raw: str) -> str:
    """The consonant skeleton a bank statement abbreviates a name down to.

    ``"Razorpay Software Private Limited"`` and ``"RZRPAY SFTWR P L"`` both
    yield ``"RZRPYSFTWR"``. Spaces are dropped so that word-splitting
    differences (``NYKAA E RETAIL`` vs ``NYKAA ERETAIL``) do not separate the
    two forms, and runs are collapsed *after* joining so ``GROWW INVEST TECH``
    and ``GROWW INVESTTECH`` agree.

    **Not idempotent, and it cannot be.** The first letter of every word
    survives even when it is a vowel, so the transform depends on word
    boundaries that joining then destroys: ``NYKAA E RETAIL`` skeletonises to
    ``NYKERTL``, and skeletonising *that* would give ``NYKRTL``. This is
    defined on names, and applying it to its own output is a caller error --
    compare two skeletons, never a skeleton against a re-skeletonised one.
    """
    words = normalize_merchant_name(raw).split()
    if not words:
        return ""
    return _collapse_runs("".join(_word_skeleton(word) for word in words))
