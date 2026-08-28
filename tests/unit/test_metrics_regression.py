"""Pinned metrics. What Steps 4-9 measured, asserted so Step 10 cannot move it.

Every step from 4 onward ended with a number, and each of those numbers was
argued for in `ARCHITECTURE.md` and in the private step notes. Step 10 rewrote
the run path -- the ladder now reads `enabled_tiers`, the CLI now runs through a
shared harness -- so the risk it carries is that a refactor quietly changed a
result while every behavioural test kept passing.

These are the guard. They are **exact**, on a fixed corpus at a fixed seed, and
a change to any of them is a real change to the system that has to be argued
for rather than absorbed. That is the point: a regression test whose bound is
loose enough never to fail is not protecting anything.

WHY THE `test` SPLIT AT SEED 42
-------------------------------
Because that is the corpus every published number in the project was measured
on. Pinning a different one would protect a result nobody quoted.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.baselines import run_b0
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import AnomalyClass, SplitName

#: The ladder as it stands: T0-T4 uncalibrated, on `test` seed 42, generator
#: 0.2.0. Uncalibrated because the fitted bundle is an artefact this test would
#: otherwise have to rebuild, and Step 7 measured precision and recall as
#: **unchanged** by the fit -- so the uncalibrated numbers are the same ones.
EXPECTED = {
    "true_positives": 130,
    "false_positives": 0,
    "false_negatives": 164,
    "false_positive_cost_minor": 0,
}

#: B0's floor, from the Step 2 notes. The "why not just SQL" answer in numbers.
EXPECTED_B0 = {
    "true_positives": 195,
    "false_positives": 135,
    "false_negatives": 99,
    "false_positive_cost_minor": 34_98_306_00,
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("regression") / "test-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def system(corpus):
    return run_system(corpus, measure_calibration_quality=False)


class TestTheLadderStillMeasuresWhatItMeasured:
    def test_the_link_counts_are_exactly_what_step_9_reported(self, system):
        links = system.metrics.link_metrics
        assert links is not None
        actual = {
            "true_positives": links.true_positives,
            "false_positives": links.false_positives,
            "false_negatives": links.false_negatives,
            "false_positive_cost_minor": links.false_positive_cost_minor,
        }
        assert actual == EXPECTED

    def test_precision_is_still_exactly_one(self, system):
        """The project's central claim. 130 correct against zero wrong."""
        assert system.metrics.auto_match_precision == 1.0

    def test_recall_is_still_0_4422(self, system):
        links = system.metrics.link_metrics
        assert links is not None
        assert links.recall == pytest.approx(0.4422, abs=5e-5)

    def test_match_rate_is_still_0_4261(self, system):
        assert system.metrics.match_rate == pytest.approx(0.4261, abs=5e-5)

    def test_exception_recall_is_still_0_9333(self, system):
        """Step 8's headline, after the third item kind took it from 0.4667."""
        assert system.metrics.exception_recall == pytest.approx(0.9333, abs=5e-5)

    def test_the_unmatchable_floor_is_reported_and_excluded(self, system):
        """35 records the sources cannot reconcile, covered by the queue in full
        and kept out of the headline recall."""
        assert system.coverage.unmatchable_recall == 1.0
        assert len(system.coverage.unmatchable) == 35

    def test_the_outgoing_rows_stay_outside_the_unit(self, system):
        assert system.coverage.out_of_scope == 34


class TestThePerTierResultsAreUnmoved:
    """Each of these was the headline of the step that built the tier."""

    @pytest.mark.parametrize(
        ("tiers", "recall", "label"),
        [
            ((0, 1), 0.2109, "Step 4: the first defensible number"),
            ((0, 1, 2), 0.3401, "Step 5: T2 took A09 from 49 misses to 11"),
            ((0, 1, 2, 3), 0.4422, "Step 6: T3 took A07 recall from 0.00 to 0.35"),
            ((0, 1, 2, 3, 4), 0.4422, "Step 6: T4's rules fire zero times here"),
        ],
    )
    def test_the_ladder_prefix_still_recalls_what_it_did(
        self, corpus, tiers, recall, label
    ):
        run = run_system(
            corpus, enabled_tiers=tiers, measure_calibration_quality=False
        )
        links = run.metrics.link_metrics
        assert links is not None, label
        assert links.recall == pytest.approx(recall, abs=5e-5), label
        assert links.false_positives == 0, label

    def test_t4_still_infers_nothing_on_this_corpus(self, system):
        """Reported as zero rather than engineered around (ARCHITECTURE.md §6,
        decision 31). Loosening a rule until it fired would trade precision for
        the appearance of contribution."""
        assert system.matched.graph.candidates == ()

    def test_t3_still_learns_its_merchant_master_from_the_statement(self, system):
        """12 profiles from 16 spellings, learned before any tier has run."""
        assert len(system.matched.merchant_spellings) == 16


class TestThePerClassRecallIsUnmoved:
    @pytest.mark.parametrize(
        ("anomaly", "recall"),
        [
            (AnomalyClass.CLEAN, 0.555556),
            (AnomalyClass.ROUNDING_DRIFT, 0.266667),
            (AnomalyClass.TIMING_SHIFT, 0.645833),
            (AnomalyClass.MISSING_REFERENCE, 0.352941),
            (AnomalyClass.SPLIT_PAYOUT, 0.342857),
        ],
    )
    def test_every_class_recalls_what_it_did(self, system, anomaly, recall):
        """Including the ones that score badly -- publishing only the good rows
        is exactly what this project is trying not to do (PLAN.md §9.1).

        A07 and A09 are the two T3 and T2 were built for, so they are the rows a
        change to either tier moves first. A02 at 0.27 and A01 at 0.56 are the
        uncomfortable ones and are pinned for the same reason.
        """
        measured = system.metrics.recall_by_anomaly_class[anomaly]
        assert measured == pytest.approx(recall, abs=5e-6)

    def test_no_class_is_quietly_dropped_from_the_table(self, system):
        """Five classes have evaluation links on this corpus and all five are
        reported. A table that shrank would be hiding a row, not improving."""
        assert set(system.metrics.recall_by_anomaly_class) == {
            AnomalyClass.CLEAN,
            AnomalyClass.ROUNDING_DRIFT,
            AnomalyClass.TIMING_SHIFT,
            AnomalyClass.MISSING_REFERENCE,
            AnomalyClass.SPLIT_PAYOUT,
        }

    def test_a05_is_absent_from_the_table_and_that_is_correct(self, system):
        """The duplicate bank row has no truth link by construction, so it
        cannot be recalled. Its damage shows up in precision, as the false
        positives B0 makes and the ladder does not."""
        assert AnomalyClass.DUPLICATE_CREDIT not in system.metrics.recall_by_anomaly_class


class TestTheBaselineFloorIsUnmoved:
    def test_b0_still_scores_exactly_what_step_2_reported(self, corpus):
        truth = load_ground_truth(corpus)
        run = run_b0(corpus)
        links = evaluate(run.predictions, truth, run_id="b0").link_metrics
        assert links is not None
        actual = {
            "true_positives": links.true_positives,
            "false_positives": links.false_positives,
            "false_negatives": links.false_negatives,
            "false_positive_cost_minor": links.false_positive_cost_minor,
        }
        assert actual == EXPECTED_B0

    def test_and_the_system_still_beats_it_where_it_matters(self, corpus, system):
        """B0 finds more links and is wrong 135 times doing it. That trade,
        priced in rupees, is the whole thesis."""
        truth = load_ground_truth(corpus)
        baseline = evaluate(run_b0(corpus).predictions, truth, run_id="b0")
        assert baseline.auto_match_precision < system.metrics.auto_match_precision
        assert baseline.link_metrics is not None
        assert baseline.link_metrics.false_positive_cost_minor > 0
        assert system.metrics.link_metrics is not None
        assert system.metrics.link_metrics.false_positive_cost_minor == 0


class TestTheCorpusItselfIsUnmoved:
    def test_the_generator_version_and_shape_are_pinned(self, corpus):
        """Every number above is conditioned on this corpus. A generator change
        that altered it would make them incomparable rather than wrong, and the
        version is what says which."""
        truth = load_ground_truth(corpus)
        assert truth.generator_version == "0.2.0"
        assert len(truth.evaluation_pairs) == 294
        assert len(truth.records) == 742
