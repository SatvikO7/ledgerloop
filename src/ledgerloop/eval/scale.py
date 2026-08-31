"""The throughput run, and the precision claim at a size that can break it.

PLAN.md left a 5,000-order ``scale`` split as a stretch item: *generate it,
record throughput*. Running it answered a question nobody had asked.

WHAT THE RUN IS ACTUALLY FOR
----------------------------
Throughput was the stated goal and is the smaller half. Every published
precision figure in this project comes from a corpus of 60 to 400 orders, and
two of the tier ladder's uniqueness arguments are quietly size-dependent: a
merchant's payouts are lakhs apart on a base of crores, so two settlements of
the *same* merchant fall inside each other's tolerance band as soon as that
merchant has enough of them, and their bank narrations are the same string
because it is the same merchant. Below about a thousand orders that never
happens. At five thousand it happens repeatedly, and the first run of this
module produced **twenty-two wrong links at ``p = 1.0``** -- the failure mode
the whole architecture is built to avoid.

So this module reports precision, recall, match rate and the false-positive
count at each size, and the throughput beside them. The guards that closed
those twenty-two are in :mod:`ledgerloop.matching.tier3_lexical`; this is the
measurement that has to keep passing for them to stay closed.

A CURVE, NOT A POINT
--------------------
Sizes are run as a series because one number cannot distinguish "fast" from
"fast so far". Reconciliation is full of quadratic temptations -- every pass
that asks "is there another row like this one" is a join -- and the growth
between points is the only thing that says whether the design holds. A single
figure at 5,000 would hide a curve bending upward.

DETERMINISTIC AND MEASURED ARE NOT MIXED
----------------------------------------
Quality reproduces exactly; timings do not reproduce at all. They travel in
separate fields of :class:`~ledgerloop.eval.artifacts.ScalePoint` and the
artefact records the machine, so a throughput number can never be quoted as a
property of the system. This is the same rule ``eval/report.py`` applies when it
confines wall clock to one labelled block.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from pathlib import Path

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.artifacts import ScaleArtifact, ScalePoint
from ledgerloop.eval.harness import run_system
from ledgerloop.generator.generate import generate_to_disk
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.models.enums import Difficulty, SplitName

__all__ = [
    "DEFAULT_SCALE_SEEDS",
    "DEFAULT_SCALE_SIZES",
    "describe_machine",
    "run_scale",
]

#: The sizes the benchmark walks by default.
#:
#: Roughly geometric, ending at the ``scale`` split's declared 5,000. The small
#: end is deliberately the size of ``test``: it is the corpus every published
#: number comes from, so it anchors the curve to a figure that is already known
#: and makes a regression at the small end impossible to miss.
DEFAULT_SCALE_SIZES: tuple[int, ...] = (300, 1_000, 2_500, 5_000)

#: The seeds every size is run at.
#:
#: More than one, because a single-seed curve cannot tell a trend from a draw --
#: and because it hid a precision failure. Phase 2.9 ran five seeds and found
#: recall at 300 orders spanning **0.7407 to 0.9796**: the published 0.9628 was
#: seed 42 near the top of its own spread, and most of the apparent "drop to
#: 5,000 orders" was that spread rather than scale. The same five seeds found 17
#: false positives at 5,000 on seed 45, on a curve that had been reporting
#: *precision held at every size* for months.
#:
#: The same seeds PLAN.md 9.4 uses elsewhere, so the scale curve is sampled the
#: way every other published figure in this project already is.
DEFAULT_SCALE_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def describe_machine() -> str:
    """A one-line identity for whatever produced the timings."""
    return f"{platform.system()} {platform.machine()} · Python {platform.python_version()}"


def run_scale(
    data_dir: Path,
    *,
    sizes: Sequence[int] = DEFAULT_SCALE_SIZES,
    bundle: CalibrationBundle | None = None,
    seeds: Sequence[int] = DEFAULT_SCALE_SEEDS,
    difficulty: Difficulty = Difficulty.STANDARD,
    regenerate: bool = False,
) -> ScaleArtifact:
    """Generate, run and measure one corpus per size, smallest first.

    Corpora are written under ``data_dir`` and reused when they are already
    there, so a re-run measures the same data rather than a fresh draw -- the
    quality columns would otherwise move for a reason that has nothing to do
    with the change being tested.

    ``measure_calibration_quality`` is off. It harvests contenders over the
    whole corpus to build the reliability diagram, costs about as much again as
    the run, and answers a question about the calibrator rather than about
    throughput -- so leaving it on would put a diagnostic in the timing.
    """
    if not sizes:
        raise ValueError("a scale run needs at least one size")
    if not seeds:
        raise ValueError("a scale run needs at least one seed")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"corpus sizes must be positive; got {tuple(sizes)}")

    points: list[ScalePoint] = []
    versions: set[str] = set()
    hashes: set[str] = set()
    for size in sorted(sizes):
        for seed in sorted(seeds):
            directory = data_dir / f"scale-{difficulty.value}-{seed}-n{size}"
            config = GeneratorConfig(
                split=SplitName.SCALE, difficulty=difficulty, seed=seed, order_count=size
            )
            generate_ms = 0
            if regenerate or not directory.exists():
                started = time.perf_counter()
                generate_to_disk(config, directory)
                generate_ms = int((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            run = run_system(directory, bundle=bundle, measure_calibration_quality=False)
            elapsed = time.perf_counter() - started

            versions.add(run.manifest.generator_version)
            hashes.add(run.config.tuning_hash)
            metrics = run.metrics
            links = metrics.link_metrics
            if links is None:  # pragma: no cover - run_system always scores
                raise ValueError(
                    f"the run over {directory} produced no link metrics; a scale point "
                    "without them would publish a precision of zero for a corpus that "
                    "was never scored"
                )
            points.append(
                ScalePoint(
                    orders=len(run.ingest.orders),
                    seed=seed,
                    records=metrics.record_count,
                    settlements=len(run.ingest.settlements),
                    bank_rows=len(run.ingest.bank_txns),
                    true_positives=links.true_positives,
                    false_positives=links.false_positives,
                    false_negatives=links.false_negatives,
                    precision=links.precision,
                    recall=links.recall,
                    match_rate=metrics.match_rate,
                    exception_recall=metrics.exception_recall,
                    false_positive_cost_minor=links.false_positive_cost_minor,
                    wall_clock_ms=int(elapsed * 1000),
                    generate_ms=generate_ms,
                    records_per_second=(
                        metrics.record_count / elapsed if elapsed > 0 else 0.0
                    ),
                )
            )

    if len(hashes) > 1:
        raise ValueError(
            "a scale curve reports one configuration; got tuning hashes "
            + ", ".join(sorted(hashes))
        )

    return ScaleArtifact(
        split=SplitName.SCALE.value,
        difficulty=difficulty.value,
        seed=sorted(seeds)[0],
        generator_version=sorted(versions)[0],
        tuning_hash=sorted(hashes)[0],
        calibrated=bundle is not None,
        machine=describe_machine(),
        points=tuple(points),
    )
