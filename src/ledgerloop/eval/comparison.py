"""Before and after, over the same corpora, with one thing changed.

WHY THIS IS AN ARTEFACT AND NOT A PARAGRAPH
--------------------------------------------
Phase 2.3 changed the reconciliation system. A change to a reconciliation system
that is described rather than measured is a claim, and this project's whole
argument is that a claim without a denominator is not a result. So the change is
run **both ways** over the identical set of corpora, with the identical bundle,
and the two arms are written to one artefact the report renders as a table.

The arms differ in exactly one field -- ``RunConfig.duplicates.enabled`` -- and
the artefact records each arm's ``tuning_hash`` so that "one thing changed" is a
check a reader can perform rather than a sentence they have to trust. Two arms
whose hashes differ by more than the switch would be two experiments, and
:func:`run_comparison` refuses to build an artefact whose arms disagree on
anything else about their own corpora.

WHAT IS AND IS NOT AGGREGATED
------------------------------
The same rule the sweep follows (``eval/summary.py``): counts and rates over a
fixed denominator aggregate across seeds; per-class tables and confusion
matrices do not, and stay single-seed with their seed named. Difficulties are
grouped and never averaged together -- easy, standard and hard are not samples
from one distribution, and a mean across them would have no referent.

THE DELTA COLUMN IS A DIFFERENCE OF MEANS, NOT A TEST
------------------------------------------------------
:meth:`ComparisonRow.delta` subtracts the arms' means and nothing more. Five
seeds is enough for a spread and not enough for a significance claim, so the
report prints the two means with their standard deviations and the difference,
and leaves it there. Manufacturing a p-value from five paired runs of a
deterministic system would dress an arithmetic identity as an inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ledgerloop.config import DuplicateDetection
from ledgerloop.eval.artifacts import ComparisonArm, ComparisonArtifact, ComparisonRow
from ledgerloop.eval.harness import SystemRun, run_system
from ledgerloop.eval.summary import aggregate
from ledgerloop.eval.sweep import SWEPT_METRICS
from ledgerloop.matching.calibration import CalibrationBundle

__all__ = ["run_comparison"]

def _arm(label: str, runs: Sequence[SystemRun]) -> ComparisonArm:
    summaries = tuple(run.summary() for run in runs)
    hashes = tuple(dict.fromkeys(row.tuning_hash for row in summaries))
    if len(hashes) > 1:
        raise ValueError(
            f"arm `{label}` ran more than one configuration: " + ", ".join(hashes)
        )
    return ComparisonArm(
        label=label,
        difficulty=summaries[0].difficulty.value,
        seeds=tuple(row.seed for row in summaries),
        tuning_hash=hashes[0],
        runs=summaries,
        aggregates={
            metric: aggregate(metric, [float(getattr(row, metric)) for row in summaries])
            for metric in SWEPT_METRICS
        },
    )


def run_comparison(
    directories: Sequence[Path],
    *,
    bundle: CalibrationBundle | None = None,
    before: DuplicateDetection | None = None,
    after: DuplicateDetection | None = None,
    change: str = (
        "the duplicate-posting pass over the bank statement, before the tier ladder"
    ),
    before_label: str = "without the pass",
    after_label: str = "with the pass",
) -> ComparisonArtifact:
    """Run every corpus twice -- once per arm -- and group the pair by difficulty.

    The same directories, the same bundle, the same full ladder, in both arms.
    Only ``duplicates`` differs, and both arms' ``tuning_hash`` values are
    recorded so a reader can see that.

    Corpora are grouped by the difficulty their own manifest declares, never by
    anything the caller asserts -- a directory named wrongly cannot land in the
    wrong row.
    """
    if not directories:
        raise ValueError("a comparison needs at least one dataset directory")

    before_config = before if before is not None else DuplicateDetection(enabled=False)
    after_config = after if after is not None else DuplicateDetection()

    grouped: dict[str, dict[str, list[SystemRun]]] = {}
    versions: set[str] = set()
    splits: set[str] = set()
    for directory in directories:
        pair = {
            "before": run_system(
                directory,
                bundle=bundle,
                duplicates=before_config,
                measure_calibration_quality=False,
            ),
            "after": run_system(
                directory,
                bundle=bundle,
                duplicates=after_config,
                measure_calibration_quality=False,
            ),
        }
        difficulty = pair["after"].manifest.difficulty.value
        slot = grouped.setdefault(difficulty, {"before": [], "after": []})
        slot["before"].append(pair["before"])
        slot["after"].append(pair["after"])
        versions.add(pair["after"].manifest.generator_version)
        splits.add(pair["after"].manifest.split.value)

    if len(versions) > 1:
        raise ValueError(
            "corpora from different generator versions are not comparable: "
            + ", ".join(sorted(versions))
        )
    if len(splits) > 1:
        raise ValueError(
            "a comparison reports one split's behaviour; got " + ", ".join(sorted(splits))
        )

    order = {"easy": 0, "standard": 1, "hard": 2}
    rows = tuple(
        ComparisonRow(
            difficulty=difficulty,
            before=_arm(before_label, arms["before"]),
            after=_arm(after_label, arms["after"]),
        )
        for difficulty, arms in sorted(
            grouped.items(), key=lambda item: order.get(item[0], 99)
        )
    )
    return ComparisonArtifact(
        change=change,
        before_label=before_label,
        after_label=after_label,
        split=splits.pop(),
        generator_version=versions.pop(),
        calibrated=bundle is not None,
        rows=rows,
    )
