"""Duplicate postings: one payout that appears in the statement more than once.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
Phase 2 measured where recall was actually going. On ``test`` seed 42 the loss is
**not** spread across payments: 17 of 35 bank credits are fully resolved, 18 are
not resolved at all, and none is partially resolved. Resolution is decided at the
settlement end of the chain, and 118 of the 164 missing links -- 72% -- belong to
ten settlements whose payout was posted to the statement **twice**.

The ladder's behaviour on those was correct and is unchanged: two credits that
qualify equally under a tier's own rule cannot be told apart *by that rule*, and
:mod:`ledgerloop.matching.bank_leg` refuses rather than flipping a coin. What was
missing is that the two rows are not two candidate payouts at all. They are one
payout the bank posted twice, and the question "which of these is the payout?"
has an answer that needs no similarity score: the **first** one is, and every
later one is a re-posting to be reversed.

So this is not a loosened threshold and not a tie-break heuristic. It is a
statement-hygiene pass -- the thing a controller does before reconciling anything
-- and it runs *before* the ladder so that every tier sees a de-duplicated
statement rather than each tier learning the same lesson separately.

THE INVARIANT IT RESTS ON
-------------------------
A UTR is a *Unique* Transaction Reference: one transfer, one reference. Two
credits carrying the same reference for the same amount are the same transfer
seen twice, not two transfers. Where the reference has been stripped (anomaly
A07 composes with A05 -- three of the ten settlements above are exactly that),
the same conclusion is reached from what survived: the same amount **to the
paise**, the same normalised narration, and a value date a day or two later is a
re-posting. Two genuinely distinct payouts to one merchant agreeing to the paise
on adjacent days is not a case this refuses to consider -- it is a case the
guards below decline.

THE GUARDS ARE THE PRECISION ARGUMENT
-------------------------------------
Every one of them only ever *declines* to call a group duplicated:

* **Exact amount equality.** Not a tolerance. A re-posting is a copy; a copy that
  differs by three paise is a different event and belongs to T1's band, not here.
* **Identical normalised narration.** Same reference when there is one, same
  counterparty text when there is not. A different narration is a different
  instruction.
* **Distinct value dates, with a unique earliest.** Two postings on the same day
  give no ordering, and an ordering is the whole conclusion. Ties are refused and
  fall through to the ladder's existing contested path.
* **A bounded window.** A re-posting follows its original promptly. Rows a
  fortnight apart carrying the same amount are a recurring payout, not a
  duplicate, and the window is what separates them.
* **Credits only.** A debit is money leaving; it is not a payout at all.

MEASURED, NOT ASSUMED
---------------------
Across the twenty-nine corpora on disk at Phase 2 (dev, train, calibration and
the fifteen ``test`` corpora of the sweep) the rule fired on **148 groups and was
right on 148 of them**: in every group the earliest posting is the credit ground
truth links the payments to, and no later posting carries a link. That check is a
test, not a note -- see ``tests/unit/test_matching_duplicates.py``.

WHAT HAPPENS TO THE DUPLICATE
-----------------------------
It leaves the *matchable* pool and nothing else. It is not consumed, not matched
and not deleted: it stays an unclaimed credit, so the exception classifier still
reaches it and still raises ``E_DUPLICATE_CREDIT`` against it with the full
amount as impact. The queue is where a duplicate belongs -- somebody has to
reverse it -- and a pass that made the money disappear from the report in order
to raise the match rate would be the exact failure this project is built against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from ledgerloop.models.records import CanonicalBankTxn

__all__ = [
    "DuplicateGroup",
    "DuplicatePostings",
    "detect_duplicate_postings",
]


@dataclass(frozen=True)
class DuplicateGroup:
    """One payout and the later re-postings of it.

    ``original`` is the earliest row; ``repostings`` is everything after it, in
    date then id order. Both are kept -- the report names the pair, and the
    exception's evidence chain has to point at the row that *was* matched.
    """

    original: CanonicalBankTxn
    repostings: tuple[CanonicalBankTxn, ...]

    @property
    def amount_minor(self) -> int:
        """The amount every member carries. Equal by construction."""
        return self.original.credit_minor

    @property
    def duplicated_minor(self) -> int:
        """Money the statement shows that the payout does not explain."""
        return self.amount_minor * len(self.repostings)

    @property
    def span_days(self) -> int:
        """Days between the payout and its last re-posting."""
        return (self.repostings[-1].value_date - self.original.value_date).days

    @property
    def narration(self) -> str:
        return _narration_key(self.original)


@dataclass(frozen=True)
class DuplicatePostings:
    """The pass's whole result: the groups, and the rows to keep out of the pool."""

    groups: tuple[DuplicateGroup, ...] = ()

    @property
    def reposted_ids(self) -> frozenset[str]:
        """Transaction ids the tier ladder must not offer as counterparts."""
        return frozenset(
            txn.txn_id for group in self.groups for txn in group.repostings
        )

    @property
    def original_ids(self) -> frozenset[str]:
        """Transaction ids that survived the pass as the genuine payout."""
        return frozenset(group.original.txn_id for group in self.groups)

    @property
    def duplicated_minor(self) -> int:
        return sum(group.duplicated_minor for group in self.groups)

    def group_for(self, txn_id: str) -> DuplicateGroup | None:
        """The group ``txn_id`` belongs to, whether as payout or as re-posting."""
        for group in self.groups:
            if txn_id == group.original.txn_id:
                return group
            if any(txn.txn_id == txn_id for txn in group.repostings):
                return group
        return None

    def __bool__(self) -> bool:
        return bool(self.groups)


def _narration_key(txn: CanonicalBankTxn) -> str:
    """What "the same instruction" means, in one place.

    The normalised narration when the parser produced one -- separators folded,
    case flattened -- and the raw string otherwise, so a row the normaliser could
    not read is compared against another unreadable row on exactly what the file
    said and not on nothing.
    """
    return (txn.narration_normalized or txn.narration_raw).strip()


def detect_duplicate_postings(
    bank_txns: Iterable[CanonicalBankTxn],
    *,
    window_days: int = 7,
) -> DuplicatePostings:
    """Find the credits that are re-postings of an earlier identical credit.

    ``window_days`` bounds how long after the payout a re-posting may arrive. It
    is a *decline* control: widening it can only add groups, and the default of
    seven days is a settlement week -- long enough for a bank to repost across a
    weekend, short enough that a monthly payout of an identical amount is never
    caught by it.

    Raises ``ValueError`` on a negative window rather than silently treating it
    as zero: a window is a policy, and an invalid one is a configuration error.
    """
    if window_days < 0:
        raise ValueError(f"window_days must not be negative, got {window_days}")

    buckets: dict[tuple[int, str], list[CanonicalBankTxn]] = {}
    for txn in bank_txns:
        if not txn.is_credit:
            continue
        buckets.setdefault((txn.credit_minor, _narration_key(txn)), []).append(txn)

    groups: list[DuplicateGroup] = []
    for members in buckets.values():
        group = _group_from(members, window_days=window_days)
        if group is not None:
            groups.append(group)

    # Source order over the payout row, so the pass is a pure function of the
    # statement and two runs produce identical artefacts.
    groups.sort(key=lambda group: group.original.txn_id)
    return DuplicatePostings(groups=tuple(groups))


def _group_from(
    members: Sequence[CanonicalBankTxn], *, window_days: int
) -> DuplicateGroup | None:
    """Apply the guards to one amount-and-narration bucket.

    Returns ``None`` -- meaning "leave these to the ladder" -- unless every guard
    passes. The order is cheapest first, and each one is a reason a reader can
    name rather than a threshold.
    """
    if len(members) < 2:
        return None

    dates: list[date] = [txn.value_date for txn in members]
    if len(set(dates)) != len(dates):
        # No unique ordering. Two postings of the same amount on the same day
        # could be a duplicate or a genuine second payout, and this pass exists
        # to answer only the question it can answer.
        return None

    ordered = sorted(members, key=lambda txn: (txn.value_date, txn.txn_id))
    original, repostings = ordered[0], tuple(ordered[1:])
    if (repostings[-1].value_date - original.value_date).days > window_days:
        return None
    return DuplicateGroup(original=original, repostings=repostings)
