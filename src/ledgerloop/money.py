"""Integer-minor-unit money primitives.

THE INVARIANT
-------------
No ``float`` ever touches the money path. Every monetary quantity in
LedgerLoop is a Python ``int`` counting *minor units* (paise for INR).
Parsing from decimal text goes through :class:`decimal.Decimal`, which is
exact for decimal strings; ``float`` is rejected at every entry point.

This module draws the boundary explicitly:

* **Money space** -- ``int`` minor units. Addition, subtraction, tolerance
  comparison and allocation all stay in ``int``.
* **Feature space** -- ``float`` ratios and scores. :func:`delta_ratio` is
  the *only* sanctioned crossing, and its result may never be written back
  into a money field. It exists to feed the blender's feature vector.

Rejecting ``bool`` matters: ``bool`` is a subclass of ``int`` in Python, so
``isinstance(True, int)`` is ``True`` and ``True + 499900`` silently yields
``499901``. Every guard checks ``bool`` first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Final

__all__ = [
    "MINOR_UNITS_PER_MAJOR",
    "RUPEE",
    "MoneyError",
    "allocate_minor",
    "assert_minor",
    "delta_ratio",
    "format_minor",
    "parse_major_to_minor",
    "parse_minor_units",
    "sum_minor",
    "tolerance_minor",
    "within_tolerance",
]

#: Minor units in one major unit. INR only in the MVP (A11 FX is cut), so this
#: is a module constant rather than a per-currency table. When multicurrency
#: returns, this becomes a lookup keyed by :class:`~ledgerloop.models.enums.Currency`.
MINOR_UNITS_PER_MAJOR: Final[int] = 100

#: One rupee, in paise. Used for tolerance floors so call sites read as money.
RUPEE: Final[int] = 100

#: Basis points in one whole. ``0.5%`` is ``50`` bps.
_BPS_PER_WHOLE: Final[int] = 10_000


class MoneyError(ValueError):
    """Raised when a value would violate the integer-minor-unit invariant."""


def assert_minor(value: object, *, field: str = "amount") -> int:
    """Return ``value`` as an ``int`` minor-unit amount, or raise.

    This is the single gate every monetary value passes through. It rejects
    ``bool`` (an ``int`` subclass that would corrupt arithmetic silently),
    ``float`` (the invariant this module exists to protect), and ``Decimal``
    (which must be converted deliberately via :func:`parse_major_to_minor`,
    never coerced implicitly).
    """
    if isinstance(value, bool):
        raise MoneyError(f"{field}: bool is not a money value (got {value!r})")
    if isinstance(value, float):
        raise MoneyError(
            f"{field}: float is forbidden in the money path (got {value!r}); "
            "parse decimal text with parse_major_to_minor() instead"
        )
    if isinstance(value, Decimal):
        raise MoneyError(
            f"{field}: Decimal must be converted explicitly (got {value!r}); "
            "use parse_major_to_minor()"
        )
    if not isinstance(value, int):
        raise MoneyError(f"{field}: expected int minor units, got {type(value).__name__}")
    return value


def parse_minor_units(text: str | int, *, field: str = "amount") -> int:
    """Parse a value that is *already* in minor units.

    Source A and Source B publish integer paise directly (``amount_gross_paise``,
    ``net_paise``). Accepts an ``int`` or a string of digits with optional sign,
    underscores and surrounding whitespace. Anything with a decimal point is
    rejected -- that input belongs in :func:`parse_major_to_minor`.
    """
    if isinstance(text, bool):
        raise MoneyError(f"{field}: bool is not a money value (got {text!r})")
    if isinstance(text, int):
        return text

    cleaned = text.strip().replace("_", "").replace(",", "")
    if not cleaned:
        raise MoneyError(f"{field}: empty string is not a money value")
    if "." in cleaned:
        raise MoneyError(
            f"{field}: {text!r} carries a decimal point but was read as minor units; "
            "use parse_major_to_minor() for major-unit text"
        )
    try:
        return int(cleaned, 10)
    except ValueError as exc:
        raise MoneyError(f"{field}: {text!r} is not an integer minor-unit amount") from exc


def parse_major_to_minor(text: str | int, *, field: str = "amount") -> int:
    """Parse major-unit decimal text (``"36803.23"``) into minor units.

    Exact via :class:`~decimal.Decimal`; ``float`` never appears. Input carrying
    more precision than one minor unit is an error rather than a silent round,
    because a source file quoting ``"36803.234"`` means the file is wrong or the
    scale assumption is wrong, and both deserve a loud failure during ingest.
    """
    if isinstance(text, bool):
        raise MoneyError(f"{field}: bool is not a money value (got {text!r})")
    if isinstance(text, float):
        raise MoneyError(f"{field}: float input defeats exact parsing (got {text!r})")

    raw = str(text).strip().replace("_", "").replace(",", "")
    if not raw:
        raise MoneyError(f"{field}: empty string is not a money value")
    try:
        amount = Decimal(raw)
    except InvalidOperation as exc:
        raise MoneyError(f"{field}: {text!r} is not a decimal amount") from exc
    if not amount.is_finite():
        raise MoneyError(f"{field}: {text!r} is not a finite amount")

    scaled = amount * MINOR_UNITS_PER_MAJOR
    if scaled != scaled.to_integral_value():
        raise MoneyError(
            f"{field}: {text!r} carries sub-minor-unit precision; "
            "refusing to round silently in the money path"
        )
    return int(scaled)


def _group_indian(major: int) -> str:
    """Indian digit grouping: ``1234567`` -> ``"12,34,567"``.

    The last three digits group together, then pairs. Used rather than Western
    grouping because every rupee figure in this project is read by an Indian
    finance audience, where ``₹12,34,567`` parses instantly and ``₹1,234,567``
    does not.
    """
    digits = str(major)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    # `head` holds 1 or 2 digits here: the guard above proved it started
    # non-empty, and the loop only exits while at least one digit remains.
    parts.insert(0, head)
    return ",".join([*parts, tail])


def format_minor(amount_minor: int, *, symbol: str = "₹") -> str:
    """Render minor units for humans (reports, the exception queue, logs).

    Presentation only -- the returned string never re-enters the money path.
    """
    value = assert_minor(amount_minor, field="format_minor")
    sign = "-" if value < 0 else ""
    major, minor = divmod(abs(value), MINOR_UNITS_PER_MAJOR)
    return f"{sign}{symbol}{_group_indian(major)}.{minor:02d}"


def sum_minor(amounts: Iterable[int], *, field: str = "amounts") -> int:
    """Sum minor-unit amounts, guarding every element.

    ``sum()`` over a list containing one ``float`` would quietly produce a
    ``float`` total and breach the invariant deep inside the subset-sum solver,
    where it would be hardest to trace. This guards up front instead.
    """
    total = 0
    for index, amount in enumerate(amounts):
        total += assert_minor(amount, field=f"{field}[{index}]")
    return total


def tolerance_minor(amount_minor: int, *, floor_minor: int, bps: int) -> int:
    """Tolerance band for an amount: ``max(floor, amount * bps)``, in minor units.

    PLAN.md §6.1 specifies T1 tolerance as ``±max(₹1, 0.5%)`` -- here
    ``floor_minor=RUPEE, bps=50``. The proportional part uses ceiling division
    so the band never rounds *down* to something tighter than configured; a
    tolerance that is silently stricter than advertised would show up as
    unexplained recall loss.
    """
    amount = assert_minor(amount_minor, field="tolerance_minor.amount")
    floor = assert_minor(floor_minor, field="tolerance_minor.floor_minor")
    if floor < 0:
        raise MoneyError("tolerance_minor: floor_minor must be non-negative")
    if bps < 0:
        raise MoneyError("tolerance_minor: bps must be non-negative")
    proportional = -(-abs(amount) * bps // _BPS_PER_WHOLE)  # ceiling division
    return max(floor, proportional)


def within_tolerance(left_minor: int, right_minor: int, tolerance_band_minor: int) -> bool:
    """True when two amounts agree within an inclusive tolerance band."""
    left = assert_minor(left_minor, field="within_tolerance.left")
    right = assert_minor(right_minor, field="within_tolerance.right")
    band = assert_minor(tolerance_band_minor, field="within_tolerance.band")
    if band < 0:
        raise MoneyError("within_tolerance: band must be non-negative")
    return abs(left - right) <= band


def delta_ratio(delta_minor: int, base_minor: int) -> float:
    """Relative size of a discrepancy, as a **feature-space** float.

    This is the one sanctioned money->float crossing. The result feeds
    ``FeatureVector.amount_delta_ratio`` and must never be assigned back into a
    money field. A zero base yields ``0.0`` when the delta is also zero and
    ``inf`` otherwise, so a discrepancy against nothing is never mistaken for
    a perfect match.
    """
    delta = assert_minor(delta_minor, field="delta_ratio.delta")
    base = assert_minor(base_minor, field="delta_ratio.base")
    if base == 0:
        return 0.0 if delta == 0 else float("inf")
    return abs(delta) / abs(base)


def allocate_minor(total_minor: int, weights: Sequence[int]) -> list[int]:
    """Split ``total_minor`` across ``weights`` with exact conservation.

    Largest-remainder allocation: the returned parts always sum to exactly
    ``total_minor``, with no paise created or destroyed. Needed by the A09
    ``SPLIT_PAYOUT`` generator (one settlement arriving as two bank credits)
    and by any fee apportionment across a settlement's payments.

    Ties in the remainder are broken by original position, keeping the result a
    pure function of its inputs -- allocation must be reproducible across runs
    for seeded regeneration to stay byte-identical.
    """
    total = assert_minor(total_minor, field="allocate_minor.total")
    if not weights:
        raise MoneyError("allocate_minor: weights must be non-empty")
    guarded = [assert_minor(w, field=f"allocate_minor.weights[{i}]") for i, w in enumerate(weights)]
    if any(w < 0 for w in guarded):
        raise MoneyError("allocate_minor: weights must be non-negative")

    weight_total = sum(guarded)
    if weight_total == 0:
        raise MoneyError("allocate_minor: weights must not sum to zero")

    # Work on magnitude, reapply sign at the end, so negative totals (an
    # adjustments line, a chargeback) allocate symmetrically.
    sign = -1 if total < 0 else 1
    magnitude = abs(total)

    scaled = [magnitude * w for w in guarded]
    parts = [s // weight_total for s in scaled]
    remainders = [s % weight_total for s in scaled]

    shortfall = magnitude - sum(parts)
    order = sorted(range(len(guarded)), key=lambda i: (-remainders[i], i))
    for i in order[:shortfall]:
        parts[i] += 1

    return [sign * p for p in parts]
