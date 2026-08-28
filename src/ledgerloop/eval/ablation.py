"""The six-row ablation: what each tier adds, and what it costs.

PLAN.md §9.3 asks for one row per prefix of the ladder -- ``T0``, ``T0-T1``,
... ``T0-T5`` -- with match rate, auto precision, LLM calls and cost. This
module produces exactly those rows, over several seeds, and nothing else.

WHY THE ROWS ARE RE-RUN AND NOT SUBTRACTED
------------------------------------------
It is tempting to take one full run and read the ablation off
``TierContribution``: T2's row already records how many links T2 auto-matched.
That number answers a different question. Every tier consumes from a shared
pool, so switching T1 off does not simply remove T1's matches -- it leaves T1's
settlements *undecided*, and T2 then sees them. The marginal contribution of a
tier is what the ladder does **without** it, which can only be measured by
running the ladder without it.

The cost of being right here is six runs per seed instead of one, and on a
300-order corpus that is a few seconds. The cost of being wrong is an ablation
table that describes an arithmetic identity rather than a system.

WHAT IS HELD FIXED
------------------
Everything except ``enabled_tiers``: the same tolerances, the same lexical
gates, the same fitted bundle, the same threshold, the same corpus. Each row
records the ``config_hash`` it actually ran under, and there is a test asserting
that two rows differ in nothing but their ladder.

WHY STANDARD DIFFICULTY AND FIVE SEEDS
--------------------------------------
Six rows x five seeds x three difficulties is ninety runs to answer a question
about tiers, which difficulty does not change the shape of. The ablation is
fixed at **standard**, and the difficulty dial is swept separately over the
headline configuration alone (:mod:`ledgerloop.eval.sweep`). Multiplying the
two would be more numbers, not more evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ledgerloop.eval.artifacts import AblationArtifact, AblationRow
from ledgerloop.eval.harness import SystemRun, run_system
from ledgerloop.eval.summary import Aggregate, aggregate
from ledgerloop.llm.client import LLMClient
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.matching.pipeline import ladder_name

__all__ = [
    "ABLATION_LADDERS",
    "AblationArtifact",
    "AblationRow",
    "run_ablation",
]

#: The six ladders PLAN.md §9.3 tabulates: every prefix of T0-T5.
#:
#: Prefixes rather than leave-one-out. A leave-one-out table answers "what does
#: the system lose without T3", which on a strictly residual ladder is a
#: question about T4's ability to pick up T3's slack; the prefix table answers
#: "what has the ladder bought by the time it reaches T3", which is what a
#: reader deciding whether to build T3 actually wants.
ABLATION_LADDERS: tuple[tuple[int, ...], ...] = (
    (0,),
    (0, 1),
    (0, 1, 2),
    (0, 1, 2, 3),
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3, 4, 5),
)


def _row(
    tiers: tuple[int, ...], runs: Sequence[SystemRun]
) -> AblationRow:
    summaries = tuple(run.summary() for run in runs)

    def over(field: str) -> Aggregate:
        return aggregate(field, [float(getattr(row, field)) for row in summaries])

    return AblationRow(
        label=ladder_name(tiers),
        tiers=tiers,
        seeds=tuple(row.seed for row in summaries),
        tuning_hashes=tuple(dict.fromkeys(row.tuning_hash for row in summaries)),
        runs=summaries,
        precision=over("precision"),
        recall=over("recall"),
        match_rate=over("match_rate"),
        f1=over("f1"),
        exception_recall=over("exception_recall"),
        candidate_yield=over("candidates_proposed"),
        auto_matched=over("auto_matched"),
        false_positives=over("false_positives"),
        false_positive_cost_minor=over("false_positive_cost_minor"),
        llm_calls=over("llm_calls"),
        llm_tokens=over("llm_tokens"),
        equivalent_paid_cost_inr=over("equivalent_paid_cost_inr"),
        llm_available=any(row.llm_available for row in summaries),
    )


def run_ablation(
    directories: Sequence[Path],
    *,
    bundle: CalibrationBundle | None = None,
    client_factory: object = None,
    ladders: Sequence[tuple[int, ...]] = ABLATION_LADDERS,
) -> AblationArtifact:
    """Run every ladder over every corpus and aggregate across seeds.

    ``client_factory`` is a zero-argument callable returning a **fresh**
    :class:`~ledgerloop.llm.client.LLMClient`, or ``None`` for the deterministic
    path. Fresh per row and per seed, because the cost ledger is per-client: one
    shared client would report the whole table's spend on every row of it. The
    response cache is shared through :class:`~ledgerloop.llm.cache.CacheKey`, so
    a repeated prompt still costs nothing.

    Corpora are run in the order given and rows in ladder order, so the artefact
    is byte-identical between two runs over the same inputs.
    """
    if not directories:
        raise ValueError("the ablation needs at least one dataset directory")

    rows: list[AblationRow] = []
    manifests: list[str] = []
    seeds: list[int] = []
    difficulties: set[str] = set()
    splits: set[str] = set()

    for tiers in ladders:
        runs: list[SystemRun] = []
        for directory in directories:
            client = _new_client(client_factory)
            run = run_system(
                directory,
                bundle=bundle,
                client=client,
                enabled_tiers=tiers,
                measure_calibration_quality=False,
            )
            runs.append(run)
            manifests.append(run.manifest.generator_version)
            splits.add(run.manifest.split.value)
            difficulties.add(run.manifest.difficulty.value)
            if run.manifest.seed not in seeds:
                seeds.append(run.manifest.seed)
        rows.append(_row(tuple(tiers), runs))

    if len(splits) > 1 or len(difficulties) > 1:
        raise ValueError(
            "an ablation table compares ladders, so every corpus in it must be "
            f"one split at one difficulty; got splits {sorted(splits)} and "
            f"difficulties {sorted(difficulties)}"
        )
    if len(set(manifests)) > 1:
        raise ValueError(
            "corpora from different generator versions are not comparable: "
            + ", ".join(sorted(set(manifests)))
        )

    hashes = {value for row in rows for value in row.tuning_hashes}
    if len(hashes) > 1:
        raise ValueError(
            "the ablation rows did not share a tuning configuration "
            f"({', '.join(sorted(hashes))}); a table whose rows differ in more "
            "than their ladder is not an ablation"
        )

    return AblationArtifact(
        tuning_hash=next(iter(hashes)) if hashes else "",
        split=splits.pop(),
        difficulty=difficulties.pop(),
        seeds=tuple(seeds),
        generator_version=manifests[0],
        calibrated=bundle is not None,
        rows=tuple(rows),
    )


def _new_client(factory: object) -> LLMClient | None:
    """Build a fresh client, or ``None``. Narrowed here so the caller stays plain."""
    if factory is None:
        return None
    if not callable(factory):  # pragma: no cover - a programming error, not input
        raise TypeError("client_factory must be callable or None")
    client = factory()
    if client is not None and not isinstance(client, LLMClient):  # pragma: no cover
        raise TypeError("client_factory must return an LLMClient or None")
    return client
