"""The multi-seed and difficulty sweeps.

The sweep's claim is narrower than the ablation's and easier to get wrong: that
the spread it reports is **corpus variance**, not configuration drift. That
needs the configuration held fixed across every row and the rows grouped by
what the corpora actually are rather than by what the caller says they are.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.sweep import SWEPT_METRICS, SweepArtifact, run_sweep
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import Difficulty, SplitName


def _corpus(root, difficulty, seed):
    directory = root / f"{difficulty.value}-{seed}"
    generate_to_disk(
        GeneratorConfig(split=SplitName.TEST, difficulty=difficulty, seed=seed, order_count=60),
        directory,
    )
    return directory


@pytest.fixture(scope="module")
def three_seeds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep-seeds")
    return [_corpus(root, Difficulty.STANDARD, seed) for seed in (42, 43, 44)]


@pytest.fixture(scope="module")
def three_difficulties(tmp_path_factory):
    """Three difficulties over five seeds each -- fifteen 60-order corpora.

    Five seeds rather than one because the response curve this fixture exists to
    check is a claim about the *dial*, and one 60-order corpus does not measure
    the dial. It has three or four settlements, so a single unresolvable split
    payout moves its recall by twenty points and the easy column can land below
    the hard one on noise alone. The published sweep runs five seeds at 300
    orders for exactly this reason (PLAN.md 9.4); this is the same discipline at
    a size a unit test can afford.
    """
    root = tmp_path_factory.mktemp("sweep-difficulty")
    return [
        _corpus(root, difficulty, seed)
        for difficulty in (Difficulty.EASY, Difficulty.STANDARD, Difficulty.HARD)
        for seed in (42, 43, 44, 45, 46)
    ]


class TestMultiSeed:
    def test_every_seed_becomes_a_row_of_one_group(self, three_seeds):
        artifact = run_sweep(three_seeds)
        assert len(artifact.groups) == 1
        group = artifact.groups[0]
        assert group.seeds == (42, 43, 44)
        assert len(group.runs) == 3

    def test_the_spread_is_corpus_variance_not_configuration_drift(self, three_seeds):
        """One tuning hash across the seeds is what witnesses that. Comparing
        `config_hash` would witness nothing: the seed is part of it."""
        artifact = run_sweep(three_seeds)
        assert len(artifact.groups[0].config_hashes) == 1

    def test_every_swept_metric_gets_an_aggregate(self, three_seeds):
        artifact = run_sweep(three_seeds)
        group = artifact.groups[0]
        for metric in SWEPT_METRICS:
            assert group.of(metric).count == 3

    def test_a_metric_that_was_not_swept_reports_no_evidence(self, three_seeds):
        """An absent aggregate is `count=0`, which renders as `n/a`. Never a
        zero: no measurement is not a measurement of zero."""
        artifact = run_sweep(three_seeds)
        missing = artifact.groups[0].of("not_a_metric")
        assert missing.count == 0
        assert missing.rendered() == "n/a"

    def test_the_headline_group_is_the_standard_one(self, three_seeds):
        artifact = run_sweep(three_seeds)
        headline = artifact.headline
        assert headline is not None
        assert headline.difficulty == "standard"

    def test_the_headline_is_absent_rather_than_wrong_when_it_was_not_run(
        self, tmp_path
    ):
        directory = _corpus(tmp_path, Difficulty.HARD, 42)
        artifact = run_sweep([directory], headline_difficulty="standard")
        assert artifact.headline is None

    def test_precision_never_falls_below_the_target_on_any_seed(self, three_seeds):
        """The precision-first claim across corpora rather than on one. A seed
        that asserted nothing reports the degenerate 0.0, so the check is on
        false positives, which is unambiguous either way."""
        artifact = run_sweep(three_seeds)
        for run in artifact.groups[0].runs:
            assert run.false_positives == 0
            if run.auto_matched > 0:
                assert run.precision == 1.0


class TestDifficulty:
    def test_groups_come_from_the_manifests_not_from_the_caller(
        self, three_difficulties
    ):
        """A directory named wrongly cannot land in the wrong row: the group is
        the difficulty the dataset's own manifest declares."""
        artifact = run_sweep(three_difficulties)
        assert [group.difficulty for group in artifact.groups] == [
            "easy",
            "standard",
            "hard",
        ]

    def test_the_dial_reads_left_to_right_as_more_goes_wrong(self, three_difficulties):
        """Difficulty order, not dictionary order, so the response curve is
        readable as a curve -- and, over five seeds, monotone."""
        artifact = run_sweep(three_difficulties)
        recalls = [group.of("recall").mean for group in artifact.groups]
        assert recalls == sorted(recalls, reverse=True)

    def test_one_bundle_is_applied_to_every_difficulty(self, three_difficulties):
        """A deployed system has one threshold. Refitting per difficulty would
        measure the calibrator's ceiling rather than the system's behaviour."""
        artifact = run_sweep(three_difficulties)
        hashes = {value for group in artifact.groups for value in group.config_hashes}
        assert len(hashes) == 1

    def test_precision_holds_as_difficulty_rises(self, three_difficulties):
        """The conservative consequence of a precision-fitted threshold applied
        off-distribution: fewer auto-matches, not wrong ones."""
        artifact = run_sweep(three_difficulties)
        for group in artifact.groups:
            assert group.of("false_positives").mean == 0.0


class TestRefusals:
    def test_two_splits_in_one_sweep_are_refused(self, tmp_path):
        """A sweep reports one split's behaviour; mixing two would put means
        from different populations in one cell."""
        dev = tmp_path / "dev"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=1), dev)
        test = _corpus(tmp_path, Difficulty.STANDARD, 1)
        with pytest.raises(ValueError, match="one split"):
            run_sweep([dev, test])

    def test_an_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="at least one dataset directory"):
            run_sweep([])


class TestReproducibility:
    def test_two_sweeps_over_the_same_corpora_agree_byte_for_byte(self, three_seeds):
        first = run_sweep(three_seeds)
        second = run_sweep(three_seeds)
        assert first.model_dump_json() == second.model_dump_json()

    def test_the_artefact_round_trips_through_disk(self, three_seeds, tmp_path):
        artifact = run_sweep(three_seeds)
        path = tmp_path / "sweep.json"
        artifact.save(path)
        assert SweepArtifact.load(path).model_dump_json() == artifact.model_dump_json()

    def test_the_order_of_the_directories_does_not_change_the_aggregates(
        self, three_seeds
    ):
        """The mean of a set is not an ordering, and a sweep whose numbers moved
        with the argument order would not be reproducible from a command line."""
        forward = run_sweep(three_seeds)
        backward = run_sweep(list(reversed(three_seeds)))
        assert forward.groups[0].of("recall").mean == backward.groups[0].of("recall").mean
        assert forward.groups[0].of("precision").mean == backward.groups[0].of(
            "precision"
        ).mean
