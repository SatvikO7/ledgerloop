"""Fitting a calibration bundle from datasets on disk.

The composition layer for Step 7, sitting where :mod:`ledgerloop.cli` sits:
:mod:`ledgerloop.matching` knows how to fit a model from feature rows and
:mod:`ledgerloop.eval` knows how to read a dataset's truth off disk, and this
module is the twenty lines that put the two together. It lives outside both so
that neither package has to import the other.

WHY SEVERAL CORPORA PER HALF
----------------------------
One ``train`` split at one seed contains 400 orders, of which the residual
tiers reach a few dozen settlements. That is a handful of decision points, and
several of the anomaly classes the blender should learn from appear once or not
at all. The generator is seeded and costs a second per corpus, so the fit takes
**several seeds of the same split** rather than one, and records every one of
them in the bundle's provenance.

This is not a way of manufacturing agreement. Each seed is an independent
corpus from the same generator, so the fit sees more of the distribution it will
be applied to, and the calibration half likewise -- which is what makes its
achieved-precision interval mean anything at this sample size.

The rule the provenance validator enforces is the one that matters: no corpus
may appear in both halves, and neither half may be ``test``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ledgerloop.config import RunConfig
from ledgerloop.eval.truth_io import DatasetManifest, load_ground_truth, load_manifest
from ledgerloop.ingest import ingest_dataset
from ledgerloop.matching.blender import DEFAULT_L2
from ledgerloop.matching.calibration import (
    DEFAULT_BIN_COUNT,
    CalibrationBundle,
    CalibrationProvenance,
    fit_bundle,
)
from ledgerloop.matching.harvest import DEFAULT_TOP_K, HarvestResult, harvest
from ledgerloop.models.candidates import FeatureVector
from ledgerloop.models.enums import SplitName

__all__ = ["CorpusHarvest", "FittingError", "HarvestSet", "fit_from_corpora", "harvest_corpora"]


class FittingError(ValueError):
    """Raised when the corpora given cannot produce an honest fit."""


@dataclass(frozen=True)
class CorpusHarvest:
    """One dataset directory, harvested and labelled."""

    directory: Path
    manifest: DatasetManifest
    result: HarvestResult

    @property
    def label(self) -> str:
        return f"{self.manifest.split.value}-{self.manifest.seed}"


@dataclass(frozen=True)
class HarvestSet:
    """One half of the fit: several corpora of the same split, concatenated."""

    corpora: tuple[CorpusHarvest, ...]

    @property
    def split(self) -> SplitName:
        return self.corpora[0].manifest.split

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(corpus.manifest.seed for corpus in self.corpora)

    @property
    def generator_version(self) -> str:
        return self.corpora[0].manifest.generator_version

    @property
    def features(self) -> tuple[FeatureVector, ...]:
        return tuple(row for corpus in self.corpora for row in corpus.result.features)

    @property
    def labels(self) -> tuple[bool, ...]:
        return tuple(row for corpus in self.corpora for row in corpus.result.labels)

    @property
    def diagnostic_features(self) -> tuple[FeatureVector, ...]:
        return tuple(
            row for corpus in self.corpora for row in corpus.result.diagnostic_features
        )

    @property
    def diagnostic_labels(self) -> tuple[bool, ...]:
        return tuple(
            row for corpus in self.corpora for row in corpus.result.diagnostic_labels
        )

    @property
    def positives(self) -> int:
        return sum(1 for label in self.labels if label)

    @property
    def rows(self) -> int:
        return len(self.labels)

    def by_tier(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for corpus in self.corpora:
            for tier, count in corpus.result.by_tier().items():
                counts[tier] = counts.get(tier, 0) + count
        return counts

    def decision_points(self) -> tuple[int, int]:
        """``(considered, resolved)`` across every corpus in this half."""
        return (
            sum(corpus.result.decision_points for corpus in self.corpora),
            sum(corpus.result.resolved_points for corpus in self.corpora),
        )


def harvest_corpora(
    directories: Sequence[Path],
    *,
    config: RunConfig,
    top_k: int = DEFAULT_TOP_K,
) -> HarvestSet:
    """Ingest, label and harvest each directory, in the order given.

    Order is preserved because the fit is order-dependent and must be
    reproducible: the same directories in the same order always produce the same
    coefficients.
    """
    if not directories:
        raise FittingError("at least one dataset directory is required")

    corpora: list[CorpusHarvest] = []
    for directory in directories:
        if not directory.is_dir():
            raise FittingError(f"no such dataset directory: {directory}")
        manifest = load_manifest(directory)
        truth = load_ground_truth(directory)
        ingested = ingest_dataset(directory, strict=False)
        corpora.append(
            CorpusHarvest(
                directory=directory,
                manifest=manifest,
                result=harvest(ingested, truth, config, top_k=top_k),
            )
        )

    splits = {corpus.manifest.split for corpus in corpora}
    if len(splits) > 1:
        named = ", ".join(sorted(split.value for split in splits))
        raise FittingError(f"one half of the fit must be one split, got {named}")
    versions = {corpus.manifest.generator_version for corpus in corpora}
    if len(versions) > 1:
        raise FittingError(
            "corpora from different generator versions are not comparable: "
            + ", ".join(sorted(versions))
        )
    return HarvestSet(corpora=tuple(corpora))


def fit_from_corpora(
    train: HarvestSet,
    calibration: HarvestSet,
    *,
    target_precision: float,
    top_k: int = DEFAULT_TOP_K,
    l2: float = DEFAULT_L2,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> CalibrationBundle:
    """Fit the bundle from two already-harvested halves.

    The generator version has to agree across both halves. A probability fitted
    against generator ``0.2.0`` is not a probability about ``0.3.0`` data, and
    the bundle records the version precisely so a later run can refuse to use a
    stale one rather than quietly applying it.
    """
    if train.generator_version != calibration.generator_version:
        raise FittingError(
            "the two halves come from different generator versions "
            f"({train.generator_version} and {calibration.generator_version}); "
            "a calibrated probability is only meaningful against the corpus it "
            "was fitted on"
        )
    provenance = CalibrationProvenance(
        train_split=train.split,
        train_seeds=train.seeds,
        calibration_split=calibration.split,
        calibration_seeds=calibration.seeds,
        generator_version=train.generator_version,
        top_k=top_k,
        train_rows=train.rows,
        train_positives=train.positives,
        calibration_rows=calibration.rows,
        calibration_positives=calibration.positives,
        train_rows_by_tier=train.by_tier(),
        calibration_rows_by_tier=calibration.by_tier(),
    )
    return fit_bundle(
        train.features,
        train.labels,
        calibration.features,
        calibration.labels,
        provenance=provenance,
        target_precision=target_precision,
        l2=l2,
        bin_count=bin_count,
    )
