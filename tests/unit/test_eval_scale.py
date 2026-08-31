"""The size curve: what it measures, and what it refuses to conflate.

Sizes here are tiny. The point of these tests is the module's contract -- one
point per size, ascending, quality separated from timing, corpora reused -- and
not the 5,000-order figure, which takes seconds and belongs in a benchmark run
rather than in a suite that has to stay fast.
"""

from __future__ import annotations

import pytest

from ledgerloop.eval.artifacts import ScaleArtifact
from ledgerloop.eval.scale import (
    DEFAULT_SCALE_SEEDS,
    DEFAULT_SCALE_SIZES,
    describe_machine,
    run_scale,
)
from ledgerloop.models.enums import Difficulty

SIZES = (20, 40)


@pytest.fixture(scope="module")
def curve(tmp_path_factory):
    """One curve, built once. Generating corpora is the slow half."""
    directory = tmp_path_factory.mktemp("scale")
    return run_scale(directory, sizes=SIZES, seeds=(42,)), directory


class TestTheCurve:
    def test_one_point_per_size(self, curve):
        artifact, _ = curve
        assert len(artifact.points) == len(SIZES)

    def test_the_points_ascend_however_the_sizes_were_given(self, tmp_path):
        """Smallest first, so a run that degrades reads top to bottom."""
        artifact = run_scale(tmp_path, sizes=(40, 20), seeds=(42,))
        assert [p.orders for p in artifact.points] == [20, 40]

    def test_the_largest_point_is_the_headline(self, curve):
        artifact, _ = curve
        assert artifact.largest is not None
        assert artifact.largest.orders == max(SIZES)

    def test_records_grow_with_orders(self, curve):
        artifact, _ = curve
        counts = [p.records for p in artifact.points]
        assert counts == sorted(counts)
        assert counts[0] > 0

    def test_it_reports_the_scale_split_and_one_configuration(self, curve):
        artifact, _ = curve
        assert artifact.split == "scale"
        assert artifact.difficulty == Difficulty.STANDARD.value
        assert artifact.tuning_hash
        assert artifact.generator_version


class TestDeterministicAndMeasuredAreSeparate:
    def test_quality_reproduces_exactly_on_the_same_corpora(self, curve):
        """Everything but the stopwatch is a property of the data."""
        _, directory = curve
        again = run_scale(directory, sizes=SIZES, seeds=(42,))
        first = run_scale(directory, sizes=SIZES, seeds=(42,))
        for left, right in zip(again.points, first.points, strict=True):
            assert left.precision == right.precision
            assert left.recall == right.recall
            assert left.match_rate == right.match_rate
            assert (left.true_positives, left.false_positives, left.false_negatives) == (
                right.true_positives, right.false_positives, right.false_negatives
            )

    def test_an_existing_corpus_is_reused_rather_than_redrawn(self, curve):
        """A second run must measure the same data, or the columns move for
        a reason that has nothing to do with the system."""
        _, directory = curve
        again = run_scale(directory, sizes=SIZES, seeds=(42,))
        assert all(point.generate_ms == 0 for point in again.points)

    def test_the_machine_is_recorded_with_the_timings(self, curve):
        artifact, _ = curve
        assert artifact.machine == describe_machine()
        assert artifact.machine.strip()


class TestWhatItRefuses:
    def test_a_curve_needs_at_least_one_size(self, tmp_path):
        with pytest.raises(ValueError, match="at least one size"):
            run_scale(tmp_path, sizes=())

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_corpus_cannot_have_no_orders(self, tmp_path, bad):
        with pytest.raises(ValueError, match="positive"):
            run_scale(tmp_path, sizes=(bad,))


class TestTheArtifact:
    def test_it_round_trips_through_disk(self, curve, tmp_path):
        artifact, _ = curve
        path = tmp_path / "scale.json"
        artifact.save(path)
        assert ScaleArtifact.load(path) == artifact

    def test_the_default_sizes_start_at_the_size_of_test(self):
        """The curve is anchored to the corpus every published number uses."""
        assert DEFAULT_SCALE_SIZES[0] == 300
        assert list(DEFAULT_SCALE_SIZES) == sorted(DEFAULT_SCALE_SIZES)
        assert DEFAULT_SCALE_SIZES[-1] == 5_000


class TestTheCurveSamplesMoreThanOneSeed:
    """A single-seed curve cannot tell a trend from a draw, and it hid a bug.

    Phase 2.9 ran the four sizes across five seeds. Recall at 300 orders spans
    **0.7407 to 0.9796** -- the published 0.9628 was seed 42 near the top of its
    own spread, so most of the advertised "drop to 5,000 orders" was that spread
    rather than scale. The same five seeds turned up **17 false positives at
    5,000 orders on seed 45**, on a curve that had been printing *precision held
    at every size* because it only ever ran one.
    """

    def test_every_size_is_run_at_every_seed(self, tmp_path):
        artifact = run_scale(tmp_path, sizes=(20, 40), seeds=(42, 43))
        assert len(artifact.points) == 4
        assert {(p.orders, p.seed) for p in artifact.points} == {
            (20, 42), (20, 43), (40, 42), (40, 43)
        }

    def test_a_point_records_the_seed_it_measured(self, tmp_path):
        """Without it a false positive cannot be reproduced: the reader needs
        the corpus, and the corpus is (size, seed)."""
        artifact = run_scale(tmp_path, sizes=(20,), seeds=(43,))
        assert [p.seed for p in artifact.points] == [43]

    def test_the_default_seeds_are_the_ones_the_rest_of_the_project_uses(self):
        assert DEFAULT_SCALE_SEEDS == (42, 43, 44, 45, 46)
        assert len(DEFAULT_SCALE_SEEDS) > 1, (
            "one seed cannot distinguish a trend from a draw; that is how a "
            "precision failure went unseen"
        )

    def test_a_curve_needs_at_least_one_seed(self, tmp_path):
        with pytest.raises(ValueError, match="at least one seed"):
            run_scale(tmp_path, sizes=(20,), seeds=())

    def test_points_stay_ordered_by_size_then_seed(self, tmp_path):
        """Deterministic output: the artefact must not depend on set order."""
        artifact = run_scale(tmp_path, sizes=(40, 20), seeds=(43, 42))
        assert [(p.orders, p.seed) for p in artifact.points] == [
            (20, 42), (20, 43), (40, 42), (40, 43)
        ]
