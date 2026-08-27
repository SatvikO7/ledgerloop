"""Assembling a fit from datasets on disk.

:mod:`ledgerloop.fitting` is twenty lines of composition, and every one of its
branches is a refusal. That is the point of testing it: the module exists to
say *no* to corpora that would produce a fit nobody should trust -- a half that
mixes splits, two halves from different generator versions, a directory that is
not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.cli import main
from ledgerloop.config import RunConfig
from ledgerloop.fitting import (
    FittingError,
    HarvestSet,
    fit_from_corpora,
    harvest_corpora,
)
from ledgerloop.models.enums import SplitName

CONFIG = RunConfig(run_id="fit-test")


@pytest.fixture(scope="module")
def corpora(tmp_path_factory) -> dict[str, list[Path]]:
    """Two train corpora and one calibration corpus, generated small."""
    root = tmp_path_factory.mktemp("corpora")
    made: dict[str, list[Path]] = {"train": [], "calibration": []}
    for split, seeds in (("train", (21, 22)), ("calibration", (23,))):
        for seed in seeds:
            out = root / f"{split}-{seed}"
            assert (
                main(
                    ["generate", "--split", split, "--seed", str(seed),
                     "--orders", "120", "--out", str(out)]
                )
                == 0
            )
            made[split].append(out)
    return made


class TestHarvestingAHalf:
    def test_it_keeps_the_corpora_in_the_order_given(self, corpora):
        half = harvest_corpora(corpora["train"], config=CONFIG)
        assert half.seeds == (21, 22)
        assert half.split is SplitName.TRAIN

    def test_the_rows_are_the_concatenation_of_its_corpora(self, corpora):
        half = harvest_corpora(corpora["train"], config=CONFIG)
        assert half.rows == sum(
            len(corpus.result.fit_rows) for corpus in half.corpora
        )
        assert len(half.features) == half.rows == len(half.labels)

    def test_it_reports_the_diagnostic_population_separately(self, corpora):
        half = harvest_corpora(corpora["train"], config=CONFIG)
        assert len(half.diagnostic_features) == len(half.diagnostic_labels)
        assert len(half.diagnostic_labels) >= half.rows

    def test_it_counts_decision_points_across_every_corpus(self, corpora):
        half = harvest_corpora(corpora["train"], config=CONFIG)
        considered, resolved = half.decision_points()
        assert considered >= resolved >= 0
        assert sum(half.by_tier().values()) == half.rows

    def test_it_names_each_corpus_for_the_report(self, corpora):
        half = harvest_corpora(corpora["train"], config=CONFIG)
        assert [corpus.label for corpus in half.corpora] == ["train-21", "train-22"]

    def test_an_empty_half_is_refused(self):
        with pytest.raises(FittingError, match="at least one"):
            harvest_corpora([], config=CONFIG)

    def test_a_missing_directory_is_named(self, tmp_path):
        with pytest.raises(FittingError, match="no such dataset directory"):
            harvest_corpora([tmp_path / "absent"], config=CONFIG)

    def test_a_half_that_mixes_splits_is_refused(self, corpora):
        with pytest.raises(FittingError, match="one split"):
            harvest_corpora(
                [*corpora["train"], *corpora["calibration"]], config=CONFIG
            )


class TestFittingFromHalves:
    def test_the_bundle_records_both_halves(self, corpora):
        train = harvest_corpora(corpora["train"], config=CONFIG)
        calibration = harvest_corpora(corpora["calibration"], config=CONFIG)
        bundle = fit_from_corpora(train, calibration, target_precision=0.99)
        assert bundle.provenance.train_seeds == (21, 22)
        assert bundle.provenance.calibration_seeds == (23,)
        assert bundle.provenance.train_rows == train.rows

    def test_halves_from_different_generator_versions_are_refused(self, corpora):
        """A probability fitted on one corpus is not a probability about another."""
        train = harvest_corpora(corpora["train"], config=CONFIG)
        calibration = harvest_corpora(corpora["calibration"], config=CONFIG)
        stale = HarvestSet(
            corpora=tuple(
                corpus.__class__(
                    directory=corpus.directory,
                    manifest=corpus.manifest.__class__(
                        **{**corpus.manifest.__dict__, "generator_version": "0.9.9"}
                    ),
                    result=corpus.result,
                )
                for corpus in calibration.corpora
            )
        )
        with pytest.raises(FittingError, match="generator versions"):
            fit_from_corpora(train, stale, target_precision=0.99)
