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

from pathlib import Path

import pytest

from ledgerloop.config import (
    Difficulty,
    GeneratorConfig,
    SplitCompletion,
    SplitName,
)
from ledgerloop.eval.comparison import run_comparison
from ledgerloop.eval.harness import run_system
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
    """The default study: split completion, the most recent change."""
    return run_comparison(corpora)


@pytest.fixture(scope="module")
def duplicates_artifact(corpora):
    """The earlier change, still reachable -- and still measurable.

    An older arm is what says a previous gain has not been quietly undone by
    everything built on top of it.
    """
    return run_comparison(corpora, switch="duplicates")


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
    def test_recall_never_falls_and_rises_where_the_shape_exists(self, artifact):
        """Split completion fires only where a split payout **lost its
        reference**, which not every corpus contains -- these fixtures are 120
        orders, and the `easy` one happens to hold none. So the assertion that
        means something is two-sided: the change may never cost a link anywhere,
        and must earn some where the shape it exists for occurs.

        Asserting a gain at *every* difficulty would be asserting a property of
        the fixture rather than of the pass, and would fail the day a seed
        stopped producing the anomaly.
        """
        deltas = {row.difficulty: row.delta("recall") for row in artifact.rows}
        assert all(delta >= 0.0 for delta in deltas.values()), deltas
        assert any(delta > 0.0 for delta in deltas.values()), deltas
        assert deltas["hard"] > 0.0, deltas

    def test_match_rate_moves_with_recall(self, artifact):
        for row in artifact.rows:
            assert row.delta("match_rate") >= 0.0, row.difficulty
        assert any(row.delta("match_rate") > 0.0 for row in artifact.rows)

    def test_the_earlier_change_still_earns_its_place(self, duplicates_artifact):
        """The duplicate-posting pass, re-measured on top of everything since.

        Its arms move as later changes land, but its **contribution** should
        not: a gain that evaporated once something else was built would mean the
        two overlap, and the ablation would be double-counting.
        """
        for row in duplicates_artifact.rows:
            assert row.delta("recall") > 0.0, row.difficulty
        assert duplicates_artifact.precision_held_everywhere

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

    def test_each_switch_moves_the_field_it_names_and_no_other(self, corpora):
        """The arms must differ in exactly the switch, or the study is measuring
        something the artefact does not name."""
        split = run_comparison(corpora[:1], switch="split-completion")
        duplicates = run_comparison(corpora[:1], switch="duplicates")
        assert split.rows[0].before.tuning_hash != split.rows[0].after.tuning_hash
        assert (
            duplicates.rows[0].before.tuning_hash
            != duplicates.rows[0].after.tuning_hash
        )
        # The two studies share their `after` arm -- both end at the shipped
        # configuration -- and differ in which arm was rolled back.
        assert (
            split.rows[0].after.tuning_hash == duplicates.rows[0].after.tuning_hash
        )

    def test_the_before_arm_is_the_system_it_names(self, corpora):
        """The artefact's `before` column has to *be* the configuration it
        claims, not merely be labelled it."""
        study = run_comparison(corpora[:1], switch="split-completion")
        direct = run_system(
            corpora[0],
            split_completion=SplitCompletion(enabled=False),
            measure_calibration_quality=False,
        )
        assert study.rows[0].before.tuning_hash == direct.config.tuning_hash
        assert study.rows[0].before.of("recall").mean == pytest.approx(
            direct.metrics.link_metrics.recall  # type: ignore[union-attr]
        )

    def test_an_unknown_switch_is_refused(self):
        with pytest.raises(ValueError, match="unknown switch"):
            run_comparison([Path(".")], switch="nonesuch")

    def test_two_runs_of_the_comparison_agree(self, corpora):
        """Deterministic, like every other artefact here -- otherwise a delta
        could not be attributed to the change rather than to the run."""
        first = run_comparison(corpora[:3])
        second = run_comparison(corpora[:3])
        assert first.model_dump_json() == second.model_dump_json()
