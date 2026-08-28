"""One run's scalar metrics, and the mean ± std of several of them.

PLAN.md §9.4: *"5 seeds x 3 difficulties, report mean ± std. A single run's
number is noise."* Steps 4-9 each reported a single seed, which was the right
thing to do while the question was "does this tier work at all"; Step 10 is
where the numbers become claims, and a claim needs a spread.

WHY A FLAT SUMMARY RATHER THAN A LIST OF ``RunMetrics``
-------------------------------------------------------
:class:`~ledgerloop.models.metrics.RunMetrics` carries per-class dictionaries,
a confusion matrix and an exception queue. Averaging those across seeds is
either meaningless (what is the mean of two confusion matrices over different
corpora?) or misleading (a class absent from one seed is not a zero in it).
:class:`RunSummary` is deliberately the subset that *does* average: counts and
rates over a fixed denominator, one row per run.

The per-class tables stay single-seed and say so. That is the honest split:
aggregate what aggregates, and report the rest at the seed it was measured on.

WHY NO WALL CLOCK
-----------------
There is no timing field here, and its absence is deliberate. Wall clock and
throughput are the only figures in this project that differ between two runs
over identical data, and an artefact carrying one could never be compared byte
for byte -- which is the check that says a rerun *reproduced* a result rather
than merely resembling it. The timings are still reported, in the labelled
``#### Measured timings`` block of ``EVALUATION.md`` that exists to hold exactly
the numbers a diff should ignore.

WHY SAMPLE STANDARD DEVIATION
-----------------------------
``ddof=1``. Five seeds are a sample of the generator's distribution, not the
population of every corpus it can produce, and the population formula would
report a spread narrower than the evidence supports. With one observation the
standard deviation is **undefined** and is reported as such rather than as
0.0 -- a single run has no spread, and printing zero would claim it has none.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import Difficulty, SplitName
from ledgerloop.models.metrics import CostLedger, RunMetrics

__all__ = ["Aggregate", "RunSummary", "aggregate", "summarise"]


class RunSummary(FrozenLedgerModel):
    """The scalars one run contributes to an aggregated table.

    ``candidates_proposed`` is **candidate yield** and ``auto_matched`` is
    conviction; both are reported because a configuration that proposes a
    hundred links and commits forty has not performed like one that proposes
    forty and commits forty (``matching/pipeline.py``). A baseline has no
    proposal stage distinct from its output, so for B0 and B1 the two are equal
    by construction and the report says so rather than implying a policy that
    is not there.
    """

    label: str = Field(description="Row label, e.g. `T0-T2` or `B1`.")
    split: SplitName
    difficulty: Difficulty
    seed: int = Field(ge=0)
    config_hash: str = Field(
        default="",
        description="The run's full configuration hash, corpus identity included. "
        "Identifies the run for the audit trail.",
    )
    tuning_hash: str = Field(
        default="",
        description="The tunables alone -- no split, seed, difficulty or ladder. "
        "One value across the rows of a table is what witnesses that nothing but "
        "the corpus (or the ladder) varied; two rows differing here are not the "
        "same configuration, whatever their labels say.",
    )

    candidates_proposed: int = Field(default=0, ge=0)
    auto_matched: int = Field(default=0, ge=0)
    needs_review: int = Field(default=0, ge=0)

    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)

    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_ci_low: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_ci_high: float = Field(default=1.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)
    match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    false_positive_cost_minor: MinorUnits = Field(default=0)

    exception_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    exceptions_raised: int = Field(default=0, ge=0)
    exceptions_expected: int = Field(default=0, ge=0)
    unmatchable_count: int = Field(default=0, ge=0)

    llm_calls: int = Field(default=0, ge=0)
    llm_cache_hits: int = Field(default=0, ge=0)
    llm_tokens: int = Field(default=0, ge=0)
    actual_cost_inr: float = Field(default=0.0, ge=0.0)
    equivalent_paid_cost_inr: float = Field(default=0.0, ge=0.0)

    record_count: int = Field(default=0, ge=0)
    llm_available: bool = Field(
        default=False,
        description="Whether a model was reachable for this row. False makes a "
        "zero in the LLM columns a statement about the environment rather than "
        "about the tier -- the two are not the same finding.",
    )


class Aggregate(FrozenLedgerModel):
    """One metric across several runs: mean, sample std, and the range.

    ``std`` is ``None`` for a single observation. A spread of zero and an
    undefined spread are different claims, and the report renders the second as
    ``n/a`` rather than as agreement between one run and itself.
    """

    metric: str
    count: int = Field(ge=0)
    mean: float = 0.0
    std: float | None = None
    minimum: float = 0.0
    maximum: float = 0.0

    def rendered(self, *, digits: int = 4) -> str:
        """``0.9931 ± 0.0042``, or ``0.9931`` when the spread is undefined."""
        if self.count == 0:
            return "n/a"
        if self.std is None:
            return f"{self.mean:.{digits}f}"
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"


def summarise(
    label: str,
    metrics: RunMetrics,
    *,
    split: SplitName,
    difficulty: Difficulty,
    seed: int,
    config_hash: str = "",
    tuning_hash: str = "",
    candidates_proposed: int = 0,
    auto_matched: int = 0,
    needs_review: int = 0,
    exceptions_raised: int = 0,
    exceptions_expected: int = 0,
    cost: CostLedger | None = None,
    llm_available: bool = False,
) -> RunSummary:
    """Flatten a scored run into the row an aggregated table averages.

    Everything derivable from ``metrics`` is read from it rather than passed in
    again, so a summary cannot disagree with the report section beside it.
    """
    links = metrics.link_metrics
    ledger = cost or metrics.cost
    return RunSummary(
        label=label,
        split=split,
        difficulty=difficulty,
        seed=seed,
        config_hash=config_hash,
        tuning_hash=tuning_hash,
        candidates_proposed=candidates_proposed,
        auto_matched=auto_matched,
        needs_review=needs_review,
        true_positives=links.true_positives if links else 0,
        false_positives=links.false_positives if links else 0,
        false_negatives=links.false_negatives if links else 0,
        precision=links.precision if links else 0.0,
        precision_ci_low=links.precision_ci_low if links else 0.0,
        precision_ci_high=links.precision_ci_high if links else 1.0,
        recall=links.recall if links else 0.0,
        f1=links.f1 if links else 0.0,
        match_rate=metrics.match_rate,
        false_positive_cost_minor=links.false_positive_cost_minor if links else 0,
        exception_recall=metrics.exception_recall,
        exceptions_raised=exceptions_raised,
        exceptions_expected=exceptions_expected,
        unmatchable_count=metrics.unmatchable_count,
        llm_calls=ledger.llm_calls,
        llm_cache_hits=ledger.cache_hits,
        llm_tokens=ledger.total_tokens,
        actual_cost_inr=ledger.actual_cost_inr,
        equivalent_paid_cost_inr=ledger.equivalent_paid_cost_inr,
        record_count=metrics.record_count,
        llm_available=llm_available,
    )


def aggregate(metric: str, values: Sequence[float]) -> Aggregate:
    """Mean, sample standard deviation and range over one metric's observations.

    ``ddof=1`` -- see the module docstring. Written out rather than reached for
    through ``statistics.stdev`` only because that function *raises* on a single
    observation, and a table cell is not the place to discover that: here one
    observation returns ``std=None`` and the renderer prints the mean alone.
    """
    count = len(values)
    if count == 0:
        return Aggregate(metric=metric, count=0)
    mean = sum(values) / count
    if count == 1:
        return Aggregate(
            metric=metric,
            count=1,
            mean=mean,
            std=None,
            minimum=values[0],
            maximum=values[0],
        )
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return Aggregate(
        metric=metric,
        count=count,
        mean=mean,
        std=sqrt(variance),
        minimum=min(values),
        maximum=max(values),
    )
