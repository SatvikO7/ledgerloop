"""The before/after study: two arms, the same corpora, one thing changed.

The properties that make the artefact evidence rather than a table:

* both arms see the **same** corpora, and the grouping comes from each corpus's
  own manifest rather than from anything the caller asserts;
* each arm ran **one** configuration, and its ``tuning_hash`` is recorded so
  "nothing else changed" is checkable;
* the arms differ, and differ in a way the hash can see -- an artefact whose
  arms hashed identically would be reporting one experiment twice.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import Difficulty, DuplicateDetection, GeneratorConfig, SplitName
from ledgerloop.eval.comparison import run_comparison
from ledgerloop.generator import generate_to_disk


def _corpus(root, difficulty: Difficulty, seed: int):
    directory = root / f"{difficulty.value}-{seed}"
    generate_to_disk(
        GeneratorConfig(
            split=SplitName.TEST, difficulty=difficulty, seed=seed, order_count=120
        ),
        directory,
    )
    return directory


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison")
    return [
        _corpus(root, difficulty, seed)
        for difficulty in (Difficulty.EASY, Difficulty.STANDARD, Difficulty.HARD)
        for seed in (42, 43, 44)
    ]


@pytest.fixture(scope="module")
def artifact(corpora):
    return run_comparison(corpora)


class TestTheShapeOfTheStudy:
    def test_one_row_per_difficulty_in_the_dials_order(self, artifact):
        assert [row.difficulty for row in artifact.rows] == ["easy", "standard", "hard"]

    def test_both_arms_saw_the_same_seeds(self, artifact):
        for row in artifact.rows:
            assert row.before.seeds == row.after.seeds == (42, 43, 44)

    def test_the_group_comes_from_the_manifest_not_the_caller(self, tmp_path):
        """A directory named wrongly cannot land in the wrong row."""
        directory = tmp_path / "definitely-not-hard"
        generate_to_disk(
            GeneratorConfig(
                split=SplitName.TEST,
                difficulty=Difficulty.HARD,
                seed=42,
                order_count=60,
            ),
            directory,
        )
        assert run_comparison([directory]).rows[0].difficulty == "hard"

    def test_each_arm_ran_exactly_one_configuration(self, artifact):
        for row in artifact.rows:
            assert row.before.tuning_hash
            assert row.after.tuning_hash

    def test_the_arms_differ_and_the_hash_can_see_it(self, artifact):
        """An artefact whose arms hashed identically would be one experiment
        reported twice."""
        for row in artifact.rows:
            assert row.before.tuning_hash != row.after.tuning_hash

    def test_the_headline_row_is_standard_difficulty(self, artifact):
        headline = artifact.headline
        assert headline is not None
        assert headline.difficulty == "standard"

    def test_an_empty_directory_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one dataset"):
            run_comparison([])


class TestWhatTheStudyFound:
    def test_recall_rose_at_every_difficulty(self, artifact):
        for row in artifact.rows:
            assert row.delta("recall") > 0.0, row.difficulty

    def test_match_rate_rose_at_every_difficulty(self, artifact):
        for row in artifact.rows:
            assert row.delta("match_rate") > 0.0, row.difficulty

    def test_precision_did_not_move_and_no_seed_produced_a_false_positive(
        self, artifact
    ):
        """The only result that makes the recall column mean anything. Read off
        the count rather than the ratio -- a ratio can round, a count cannot."""
        assert artifact.precision_held_everywhere
        for row in artifact.rows:
            assert all(run.false_positives == 0 for run in row.after.runs)
            assert all(run.false_positives == 0 for run in row.before.runs)
            assert row.delta("false_positive_cost_minor") == 0.0

    def test_the_before_arm_is_the_pre_phase_2_system(self, corpora):
        """Passing the switch explicitly must agree with the default, or the
        artefact's `before` column would not be the system it names."""
        explicit = run_comparison(
            corpora[:1], before=DuplicateDetection(enabled=False)
        )
        default = run_comparison(corpora[:1])
        assert (
            explicit.rows[0].before.tuning_hash == default.rows[0].before.tuning_hash
        )
        assert explicit.rows[0].before.of("recall").mean == pytest.approx(
            default.rows[0].before.of("recall").mean
        )

    def test_two_runs_of_the_comparison_agree(self, corpora):
        """Deterministic, like every other artefact here -- otherwise a delta
        could not be attributed to the change rather than to the run."""
        first = run_comparison(corpora[:3])
        second = run_comparison(corpora[:3])
        assert first.model_dump_json() == second.model_dump_json()
