"""The subset-sum search. Pure integers, no domain knowledge.

PLAN.md 6.2 calls T2 the algorithmic core, and this module is its engine. It
knows nothing about settlements, credits or money -- it takes a list of
integer amounts and a target *window*, and answers a question that is not "find
a subset" but **"how many subsets fit?"**.

WHY IT COUNTS RATHER THAN FINDS
--------------------------------
A solver that returns the first subset it stumbles on makes the decision for
the caller, and makes it badly: where two different sets of payments both add
up to a credit, picking one is a coin flip dressed as arithmetic. PLAN.md 6.2
is explicit that this is the case to refuse. So the search stops as soon as it
has found ``want`` solutions -- two, normally -- and the caller reads the count,
not the answer.

That also bounds the work. Proving "at least two" is cheap; enumerating all of
them is not, and nothing needs all of them.

THE TWO STRATEGIES, AND WHY EXHAUSTIVENESS IS REPORTED
-------------------------------------------------------
* **Meet in the middle** -- split the items in half, enumerate each half's
  ``2^(n/2)`` sums, and match one against the other. Exhaustive: when it says
  one solution, there is exactly one.
* **Greedy accumulation** -- for buckets past the configured cap, where
  ``2^(n/2)`` stops being affordable. It can find a subset; it can never prove
  that subset is alone, and it can miss one that exists.

:attr:`SubsetSearch.exhaustive` is the difference, and it is carried out of the
module rather than resolved inside it, because "I found one" and "I found the
only one" license completely different decisions. T2 auto-matches the second and
routes the first for review.

DETERMINISM
-----------
Items are never reordered in place: the halves are split by position, subsets
enumerate in mask order, and the second half is searched in ``(sum, mask)``
order. Two runs over the same input return the same solutions in the same
order, which is what lets a reproducibility test compare decision logs directly.

The wall-clock cap is the one thing here that could make a run vary between
machines. It is a **safety net, not a bound**: the deterministic bound is
``max_exact_items``, and a run where :attr:`SubsetSearch.timed_out` is ever true
is a run whose reproducibility cannot be claimed. The tier counts those, and a
test asserts the count is zero on the corpus.
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from ledgerloop.money import assert_minor

__all__ = [
    "SubsetSearch",
    "SubsetSolution",
    "find_subsets",
    "greedy_subset",
    "meet_in_the_middle",
]

#: How often the wall-clock cap is consulted, in enumeration steps. Checking a
#: clock is far more expensive than the work between checks, so it is amortised.
_CLOCK_EVERY: Final[int] = 4096

#: Enumerating one half costs ``2^(n/2)`` entries, and past some width the
#: enumeration itself is the cost rather than the pairing. 16 allows a
#: thirty-two item bucket exhaustively at 65,536 entries a half, which is
#: instant; beyond that the greedy strategy takes over **even when the
#: configured ``max_subset_size`` is higher**. The config value is the policy
#: cap; this is the implementation's own limit, and whichever binds first wins.
#: Real batches in this corpus run to about twenty payments, so neither binds.
_MAX_HALF_WIDTH: Final[int] = 16


@dataclass(frozen=True)
class SubsetSolution:
    """One subset that lands inside the target window.

    ``indices`` are positions in the caller's original sequence, ascending, so
    the caller can map them back without having to know how the search
    reordered anything -- it did not.
    """

    indices: tuple[int, ...]
    total_minor: int

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class SubsetSearch:
    """What the search found, and how much it is entitled to claim.

    ``exhaustive`` is the field that matters. It is ``True`` only when the whole
    space was explored, so ``len(solutions) == 1 and exhaustive`` is the *only*
    combination that means "this subset is the answer". Everything else means
    "at most a hypothesis".
    """

    solutions: tuple[SubsetSolution, ...]
    exhaustive: bool
    method: str
    examined: int = 0
    timed_out: bool = False

    @property
    def found(self) -> int:
        return len(self.solutions)

    @property
    def is_unique(self) -> bool:
        """Exactly one subset fits, and the search is entitled to say so."""
        return self.exhaustive and len(self.solutions) == 1

    @property
    def is_ambiguous(self) -> bool:
        """Several subsets fit. PLAN.md 6.2: do not match, report both."""
        return len(self.solutions) > 1


#: A caller-supplied check applied to each candidate before it counts as a
#: solution. T2 uses it to re-derive the credit a subset implies and compare it
#: exactly, so a subset that merely lands in a widened search window but does not
#: actually reconcile is never counted -- and never inflates the ambiguity count.
Accept = Callable[[tuple[int, ...], int], bool]


def _enumerate_half(
    amounts: Sequence[int], offset: int, count: int
) -> list[tuple[int, int]]:
    """Every subset sum of one half, as ``(total, mask)`` pairs in mask order."""
    sums: list[tuple[int, int]] = [(0, 0)]
    for step in range(count):
        amount = amounts[offset + step]
        bit = 1 << step
        sums.extend([(total + amount, mask | bit) for total, mask in sums])
    return sums


def _indices_from(mask: int, offset: int, out: list[int]) -> None:
    step = 0
    while mask:
        if mask & 1:
            out.append(offset + step)
        mask >>= 1
        step += 1


def meet_in_the_middle(
    amounts: Sequence[int],
    low: int,
    high: int,
    *,
    want: int = 2,
    accept: Accept | None = None,
    deadline_ns: int | None = None,
) -> SubsetSearch:
    """Exhaustive search for subsets summing into ``[low, high]``.

    Splits the items in half by position, enumerates both halves, and pairs
    them. ``2^(n/2)`` instead of ``2^n``: a twenty-payment batch costs about two
    thousand entries rather than a million.

    The empty subset is never a solution. A credit explained by no payments is
    not an explanation.
    """
    count = len(amounts)
    split = count // 2
    left = _enumerate_half(amounts, 0, split)
    right = _enumerate_half(amounts, split, count - split)
    right.sort()
    right_sums = [total for total, _ in right]

    solutions: list[SubsetSolution] = []
    examined = 0
    steps = 0
    next_check = _CLOCK_EVERY

    for left_total, left_mask in left:
        steps += 1
        if deadline_ns is not None and steps >= next_check:
            next_check = steps + _CLOCK_EVERY
            if time.perf_counter_ns() > deadline_ns:
                return SubsetSearch(
                    solutions=tuple(solutions),
                    exhaustive=False,
                    method="meet_in_the_middle",
                    examined=examined,
                    timed_out=True,
                )
        window_low = low - left_total
        window_high = high - left_total
        start = bisect_left(right_sums, window_low)
        stop = bisect_right(right_sums, window_high)
        steps += stop - start
        for position in range(start, stop):
            right_total, right_mask = right[position]
            examined += 1
            if left_mask == 0 and right_mask == 0:
                continue
            indices: list[int] = []
            _indices_from(left_mask, 0, indices)
            _indices_from(right_mask, split, indices)
            candidate = tuple(indices)
            total = left_total + right_total
            if accept is not None and not accept(candidate, total):
                continue
            solutions.append(SubsetSolution(indices=candidate, total_minor=total))
            if len(solutions) >= want:
                return SubsetSearch(
                    solutions=tuple(solutions),
                    exhaustive=False,  # stopped early; more may exist
                    method="meet_in_the_middle",
                    examined=examined,
                )

    return SubsetSearch(
        solutions=tuple(solutions),
        exhaustive=True,
        method="meet_in_the_middle",
        examined=examined,
    )


def greedy_subset(
    amounts: Sequence[int],
    low: int,
    high: int,
    *,
    accept: Accept | None = None,
) -> SubsetSearch:
    """Largest-first accumulation. The fallback past the exhaustive cap.

    Returns at most one subset and **never** claims exhaustiveness, so T2 can
    record what it found without being entitled to auto-match it.

    Deterministic: items are ordered by ``(-amount, index)``, so equal amounts
    break ties by original position rather than by sort stability.

    NO LOCAL-SWAP REPAIR, AND WHY NOT
    ---------------------------------
    The obvious next move when the accumulation lands short is to swap one
    chosen item for one skipped item. Under descending accumulation that can
    never work, so it is not attempted rather than written and left dead.

    Suppose the run ends with total ``T < low`` and chosen set ``C``. Every
    skipped ``x`` was skipped because the running total ``R`` at the time
    satisfied ``R + x > high``. For swapping ``c`` out and ``x`` in to land in
    the window:

    * If ``x > c`` then ``x`` was considered *before* ``c``, so ``R <= T - c``,
      so ``x > high - R >= high - T + c``, so ``T - c + x > high``. It
      overshoots.
    * If ``x < c`` then ``T - c + x < T < low``. It undershoots.

    Either way no single swap lands, and a multi-item repair is just the
    exhaustive search this strategy exists to avoid. Missing a reachable target
    is therefore a real limitation of the fallback -- and the reason a greedy
    result is only ever a hypothesis.
    """
    order = sorted(range(len(amounts)), key=lambda i: (-amounts[i], i))
    chosen: list[int] = []
    total = 0
    for index in order:
        if total + amounts[index] <= high:
            chosen.append(index)
            total += amounts[index]
        if total >= low:
            break

    found = tuple(sorted(chosen))
    if found and low <= total <= high and (accept is None or accept(found, total)):
        return SubsetSearch(
            solutions=(SubsetSolution(indices=found, total_minor=total),),
            exhaustive=False,
            method="greedy",
            examined=len(amounts),
        )
    return SubsetSearch(
        solutions=(), exhaustive=False, method="greedy", examined=len(amounts)
    )


def find_subsets(
    amounts: Sequence[int],
    low: int,
    high: int,
    *,
    want: int = 2,
    max_exact_items: int = 40,
    timeout_ms: int = 200,
    accept: Accept | None = None,
) -> SubsetSearch:
    """Find up to ``want`` subsets summing into ``[low, high]``.

    Picks the strategy from the bucket size: exhaustive while that is
    affordable, greedy past ``max_exact_items``. Every amount is routed through
    :func:`~ledgerloop.money.assert_minor`, so a float reaching the solver
    raises here rather than producing a subset that looks right.
    """
    guarded = [
        assert_minor(amount, field=f"subset_sum.amounts[{index}]")
        for index, amount in enumerate(amounts)
    ]
    assert_minor(low, field="subset_sum.low")
    assert_minor(high, field="subset_sum.high")
    if low > high:
        raise ValueError(f"empty target window: [{low}, {high}]")
    if not guarded:
        return SubsetSearch(solutions=(), exhaustive=True, method="meet_in_the_middle")

    too_many = len(guarded) > max_exact_items
    too_wide = (len(guarded) - len(guarded) // 2) > _MAX_HALF_WIDTH
    if too_many or too_wide:
        return greedy_subset(guarded, low, high, accept=accept)

    deadline = time.perf_counter_ns() + timeout_ms * 1_000_000 if timeout_ms > 0 else None
    return meet_in_the_middle(
        guarded, low, high, want=want, accept=accept, deadline_ns=deadline
    )
