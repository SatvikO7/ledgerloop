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

PHASE 2 KEPT EVERY PIN, AND ADDED A SECOND SET
-----------------------------------------------
Phase 2.3 changed one thing about the run: a duplicate-posting pass now runs
over the statement before the ladder does. The historical numbers below are
therefore pinned on ``duplicates=DuplicateDetection(enabled=False)``, and they
**still hold exactly** -- 130/0/164, recall 0.4422, match rate 0.4261, and every
ladder prefix and per-class figure to six places. That is the strongest
statement available about the change: it is additive, it is switchable, and
turning it off reproduces Steps 4-9 to the digit rather than approximately.

:class:`TestPhase2Defaults` pins what the shipped configuration measures now.
Both sets are exact. One exception recall moved in *both* arms, and its own test
says why.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import DuplicateDetection, GeneratorConfig
from ledgerloop.eval.baselines import run_b0
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import AnomalyClass, SplitName

#: The pre-Phase-2 pass switch. Every historical pin is measured with the
#: duplicate-posting pass off, which is what makes "Phase 2.3 changed nothing
#: else" a check rather than a claim.
PRE_PHASE_2 = DuplicateDetection(enabled=False)

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
    """The ladder as Steps 4-9 ran it: Phase 2.3's pass switched off."""
    return run_system(
        corpus, duplicates=PRE_PHASE_2, measure_calibration_quality=False
    )


@pytest.fixture(scope="module")
def shipped(corpus):
    """The ladder as it ships after Phase 2, with every default in force."""
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

    def test_exception_recall_moved_from_0_9333_to_1_0000(self, system, shipped):
        """The one pin Phase 2 moved, and it moved in **both** arms.

        Step 8 reported 0.9333 -- 28 of 30 -- and the two it missed were orders
        refunded after their own payout had already left, whose claw-back was
        netted off a *later* batch (A06). Nothing in the queue reached them:
        the later batch reconciles to the paise, the earlier one was paid in
        full, and the order appears in no unresolved link. They used to be
        covered only by accident, when some unrelated anomaly happened to leave
        their batch contested and its evidence chain named them.

        Phase 2.3 added :func:`~ledgerloop.exceptions.taxonomy.clawback_items`,
        which attributes such an adjustment to the refunded order the ledger
        itself marks REFUNDED. It is independent of the duplicate-posting pass
        -- which is why both arms read 1.0000 here -- and it is why 30 of 30 is
        now covered.
        """
        assert system.metrics.exception_recall == 1.0
        assert shipped.metrics.exception_recall == 1.0
        assert len(system.coverage.expected) == 30

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
            corpus,
            enabled_tiers=tiers,
            duplicates=PRE_PHASE_2,
            measure_calibration_quality=False,
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


class TestPhase2Defaults:
    """What the shipped configuration measures, pinned as exactly as the rest.

    Every number here is on the same corpus as the historical pins above, so the
    two classes read as one before-and-after table. Precision and the false
    positive cost are the ones that must not move, and they do not.
    """

    def test_the_link_counts_after_the_duplicate_posting_pass(self, shipped):
        links = shipped.metrics.link_metrics
        assert links is not None
        actual = {
            "true_positives": links.true_positives,
            "false_positives": links.false_positives,
            "false_negatives": links.false_negatives,
            "false_positive_cost_minor": links.false_positive_cost_minor,
        }
        assert actual == {
            "true_positives": 248,
            "false_positives": 0,
            "false_negatives": 46,
            "false_positive_cost_minor": 0,
        }

    def test_precision_is_still_exactly_one(self, shipped):
        """The whole point. 118 more links asserted and still nothing wrong."""
        assert shipped.metrics.auto_match_precision == 1.0

    def test_recall_is_0_8435(self, shipped):
        links = shipped.metrics.link_metrics
        assert links is not None
        assert links.recall == pytest.approx(0.8435, abs=5e-5)

    def test_match_rate_is_0_7971(self, shipped):
        assert shipped.metrics.match_rate == pytest.approx(0.7971, abs=5e-5)

    @pytest.mark.parametrize(
        ("tiers", "recall"),
        [
            ((0, 1), 0.496599),
            ((0, 1, 2), 0.625850),
            ((0, 1, 2, 3), 0.843537),
            ((0, 1, 2, 3, 4), 0.843537),
        ],
    )
    def test_every_ladder_prefix_gains_and_none_loses_precision(
        self, corpus, tiers, recall
    ):
        """The pass lifts every rung, T4 still contributes nothing, and no rung
        buys its recall with a false positive."""
        run = run_system(corpus, enabled_tiers=tiers, measure_calibration_quality=False)
        links = run.metrics.link_metrics
        assert links is not None
        assert links.recall == pytest.approx(recall, abs=5e-6)
        assert links.false_positives == 0

    @pytest.mark.parametrize(
        ("anomaly", "recall"),
        [
            (AnomalyClass.CLEAN, 0.975309),
            (AnomalyClass.ROUNDING_DRIFT, 1.0),
            (AnomalyClass.TIMING_SHIFT, 1.0),
            (AnomalyClass.MISSING_REFERENCE, 0.752941),
            (AnomalyClass.SPLIT_PAYOUT, 0.342857),
        ],
    )
    def test_per_class_recall_including_the_row_that_did_not_move(
        self, shipped, anomaly, recall
    ):
        """A09 SPLIT_PAYOUT is pinned at exactly its old 0.342857, and that is
        the honest headline of Phase 2.3: the pass fixed the duplicate-posting
        loss and did nothing at all for split payouts, which are the whole of
        the 46 links still missing. Reported as unmoved rather than averaged
        away.
        """
        assert shipped.metrics.recall_by_anomaly_class[anomaly] == pytest.approx(
            recall, abs=5e-6
        )

    def test_the_duplicate_postings_it_found_are_still_in_the_queue(self, shipped):
        """Ten groups on this corpus, and every re-posting is a bank row the
        queue still reports -- the pass moves money out of the *matchable* pool
        and never out of the report.
        """
        duplicates = shipped.matched.context.duplicates
        assert len(duplicates.groups) == 10
        covered = {
            ref.key
            for exception in shipped.exceptions
            for ref in exception.involved_refs
        }
        assert all(
            f"bank_txn:{txn_id}" in covered for txn_id in duplicates.reposted_ids
        )
