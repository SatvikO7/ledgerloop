"""Multi-seed and difficulty sweeps over the headline configuration.

PLAN.md §9.4: *"Seeded, deterministic reruns. 5 seeds x 3 difficulties, report
mean ± std. A single run's number is noise."* This module runs the **headline**
configuration -- the full ladder, the fitted bundle, nothing switched off --
across seeds and across the difficulty dial, and reports both.

TWO SWEEPS, KEPT APART
----------------------
* **Multi-seed.** Five seeds of ``test`` at standard difficulty. This is the
  spread on the headline number, and it is the one that belongs beside every
  claim the project makes.
* **Difficulty.** The same configuration at easy / standard / hard. This is not
  a spread, it is a *response curve*: it says how precision, recall, match rate
  and the exception queue move as more of the corpus is broken.

They are separate tables because they answer separate questions, and because
merging them would put a mean across difficulties in a cell -- a number with no
referent, since the difficulties are not samples from one distribution.

THE THRESHOLD IS NOT REFITTED PER DIFFICULTY
--------------------------------------------
One bundle, fitted once on ``train`` and ``calibration`` at **standard**
difficulty, is applied to every row. Fitting a separate bundle per difficulty
would be defensible and is not done, for one reason and one honest consequence:

* The reason: a deployed system has one threshold. Refitting per difficulty
  measures the calibrator's ceiling rather than the system's behaviour, and the
  ceiling is not what a controller experiences.
* The consequence, stated rather than smoothed: on ``hard`` the bundle is
  operating off-distribution, and because ``tau_high`` was fitted for precision
  the effect is conservative -- fewer auto-matches, not wrong ones. The
  difficulty table shows that as a falling match rate, and the report says why.

Under no circumstances is the threshold selected against a difficulty's *test*
result. That would be tuning on test, which is the one thing §9.4 exists to
forbid.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ledgerloop.eval.artifacts import SweepArtifact, SweepGroup
from ledgerloop.eval.harness import SystemRun, run_system
from ledgerloop.eval.summary import aggregate
from ledgerloop.llm.client import LLMClient
from ledgerloop.matching.calibration import CalibrationBundle

__all__ = ["SWEPT_METRICS", "SweepArtifact", "SweepGroup", "run_sweep"]

#: The metrics a sweep group aggregates. Named as data so the report and the
#: runner cannot disagree about which columns exist.
SWEPT_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "match_rate",
    "exception_recall",
    "candidates_proposed",
    "auto_matched",
    "false_positives",
    "false_positive_cost_minor",
    "unmatchable_count",
    "llm_calls",
    "llm_tokens",
    "equivalent_paid_cost_inr",
)


def _group(runs: Sequence[SystemRun]) -> SweepGroup:
    summaries = tuple(run.summary() for run in runs)
    return SweepGroup(
        difficulty=summaries[0].difficulty.value,
        split=summaries[0].split.value,
        seeds=tuple(row.seed for row in summaries),
        runs=summaries,
        aggregates={
            metric: aggregate(
                metric, [float(getattr(row, metric)) for row in summaries]
            )
            for metric in SWEPT_METRICS
        },
    )


def run_sweep(
    directories: Sequence[Path],
    *,
    bundle: CalibrationBundle | None = None,
    client_factory: object = None,
    headline_difficulty: str = "standard",
) -> SweepArtifact:
    """Run the headline configuration over every corpus, grouped by difficulty.

    Every directory is run with the **full** ladder and nothing switched off, so
    the standard-difficulty group is literally the headline number repeated
    across seeds -- not an approximation of it. Corpora are grouped by the
    difficulty their own manifest declares rather than by anything the caller
    asserts, so a directory named wrongly cannot land in the wrong row.

    ``client_factory`` returns a fresh client per run, for the reason
    :func:`~ledgerloop.eval.ablation.run_ablation` documents: the cost ledger is
    per-client, and a shared one would report every run's spend on all of them.
    """
    if not directories:
        raise ValueError("a sweep needs at least one dataset directory")

    by_difficulty: dict[str, list[SystemRun]] = {}
    versions: set[str] = set()
    splits: set[str] = set()
    for directory in directories:
        client = _new_client(client_factory)
        run = run_system(
            directory,
            bundle=bundle,
            client=client,
            measure_calibration_quality=False,
        )
        by_difficulty.setdefault(run.manifest.difficulty.value, []).append(run)
        versions.add(run.manifest.generator_version)
        splits.add(run.manifest.split.value)

    if len(versions) > 1:
        raise ValueError(
            "corpora from different generator versions are not comparable: "
            + ", ".join(sorted(versions))
        )
    if len(splits) > 1:
        raise ValueError(
            "a sweep reports one split's behaviour; got " + ", ".join(sorted(splits))
        )

    # Difficulty order is the dial's order, not dictionary order, so the response
    # curve reads left to right as "more of the corpus is broken".
    order = {"easy": 0, "standard": 1, "hard": 2}
    groups = tuple(
        _group(runs)
        for _, runs in sorted(
            by_difficulty.items(), key=lambda item: order.get(item[0], 99)
        )
    )
    return SweepArtifact(
        split=splits.pop(),
        generator_version=versions.pop(),
        calibrated=bundle is not None,
        headline_difficulty=headline_difficulty,
        groups=groups,
    )


def _new_client(factory: object) -> LLMClient | None:
    if factory is None:
        return None
    if not callable(factory):  # pragma: no cover - a programming error, not input
        raise TypeError("client_factory must be callable or None")
    client = factory()
    if client is not None and not isinstance(client, LLMClient):  # pragma: no cover
        raise TypeError("client_factory must return an LLMClient or None")
    return client
