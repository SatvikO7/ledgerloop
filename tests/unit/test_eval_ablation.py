"""The ablation: tier isolation, configuration isolation, and reproducibility.

The table's whole claim is that a difference between two rows is the tier that
was switched off. Three things have to hold for that, and each is tested:

1. A disabled tier really does not run -- no candidates, no row in the tier
   table, no contribution to the counters.
2. Everything except the ladder is held fixed, witnessed by one `tuning_hash`
   across every row.
3. The rows are produced by re-running the ladder, not by subtracting counters
   off one full run -- so switching T1 off leaves its settlements for T2, which
   is the marginal contribution the reader is being shown.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import GeneratorConfig, RunConfig
from ledgerloop.eval.ablation import ABLATION_LADDERS, AblationArtifact, run_ablation
from ledgerloop.eval.harness import run_system
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import ingest_dataset
from ledgerloop.matching import run_matching
from ledgerloop.matching.pipeline import ladder_name
from ledgerloop.models.enums import SplitName, Tier


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """One `dev` corpus, generated once. Small enough to run thirty times."""
    directory = tmp_path_factory.mktemp("ablation") / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def two_corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation-seeds")
    directories = []
    for seed in (42, 43):
        directory = root / f"dev-{seed}"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=seed), directory)
        directories.append(directory)
    return directories


class TestTierGating:
    """`RunConfig.enabled_tiers` has carried its description since Step 0. This
    is the step where the ladder reads it, so this is where it gets tested."""

    @pytest.mark.parametrize("tiers", ABLATION_LADDERS)
    def test_only_the_enabled_tiers_appear_in_the_tier_table(self, corpus, tiers):
        """A zero row for a tier that did not run is a false measurement --
        the same rule that keeps T5 out of a `--no-llm` run's table."""
        ingested = ingest_dataset(corpus, strict=False)
        config = RunConfig(run_id="t", enabled_tiers=tiers)
        run = run_matching(ingested, config)
        # T5 never runs without an adjudicator, so it is absent from every row.
        expected = {
            tier for tier in Tier if tier.value in set(tiers) and tier is not Tier.T5_LLM
        }
        assert {row.tier for row in run.tier_contributions} == expected

    def test_a_disabled_tier_proposes_nothing(self, corpus):
        ingested = ingest_dataset(corpus, strict=False)
        without = run_matching(ingested, RunConfig(run_id="t", enabled_tiers=(0, 1)))
        assert without.aggregation.candidates == ()
        assert without.lexical.candidates == ()
        assert without.graph.candidates == ()

    def test_the_run_is_named_for_the_ladder_that_ran(self, corpus):
        ingested = ingest_dataset(corpus, strict=False)
        run = run_matching(ingested, RunConfig(run_id="t", enabled_tiers=(0, 1, 2)))
        assert run.name == "T0-T2"

    def test_a_config_listing_t5_with_no_model_is_named_t0_t4(self, corpus):
        """The name describes what ran, not what was permitted. Labelling a
        model-less run `T0-T5` would credit the ladder with a tier it never
        invoked."""
        ingested = ingest_dataset(corpus, strict=False)
        run = run_matching(ingested, RunConfig(run_id="t", enabled_tiers=(0, 1, 2, 3, 4, 5)))
        assert run.name == "T0-T4"
        assert Tier.T5_LLM not in {row.tier for row in run.tier_contributions}

    def test_t0_alone_still_produces_a_scored_run(self, corpus):
        ingested = ingest_dataset(corpus, strict=False)
        run = run_matching(ingested, RunConfig(run_id="t", enabled_tiers=(0,)))
        assert len(run.predictions) > 0
        assert run.passes == 0

    def test_disabling_t0_leaves_its_work_for_t1(self, corpus):
        """The pool is shared, which is why the ablation re-runs rather than
        subtracts: a tier that does not run does not remove its settlements, it
        leaves them undecided for the next one."""
        ingested = ingest_dataset(corpus, strict=False)
        without_t0 = run_matching(ingested, RunConfig(run_id="t", enabled_tiers=(1,)))
        assert without_t0.bank_legs[0].candidates == ()
        assert len(without_t0.bank_legs[1].candidates) > 0


class TestLadderName:
    @pytest.mark.parametrize(
        ("tiers", "expected"),
        [
            ((0,), "T0"),
            ((0, 1), "T0-T1"),
            ((0, 1, 2, 3, 4, 5), "T0-T5"),
            ((0, 2, 4), "T0+T2+T4"),
        ],
    )
    def test_a_gap_is_listed_rather_than_ranged(self, tiers, expected):
        """A row labelled `T0-T4` that skipped T2 would be a mislabelled
        measurement rather than a terse one."""
        assert ladder_name(tiers) == expected


class TestConfigurationIsolation:
    def test_every_row_shares_one_tuning_hash(self, two_corpora):
        artifact = run_ablation(two_corpora)
        hashes = {value for row in artifact.rows for value in row.tuning_hashes}
        assert len(hashes) == 1
        assert artifact.tuning_hash == next(iter(hashes))

    def test_the_tuning_hash_ignores_the_ladder_and_the_corpus(self):
        """Otherwise it would restate the row label and the seed column instead
        of witnessing that nothing else moved."""
        base = RunConfig(run_id="a", enabled_tiers=(0, 1), seed=42, split=SplitName.TEST)
        other = RunConfig(
            run_id="b", enabled_tiers=(0, 1, 2, 3), seed=99, split=SplitName.DEV
        )
        assert base.tuning_hash == other.tuning_hash
        assert base.config_hash != other.config_hash

    def test_the_tuning_hash_does_notice_a_threshold(self):
        base = RunConfig(run_id="a")
        moved = RunConfig(
            run_id="a",
            thresholds=base.thresholds.model_copy(update={"tau_high": 0.999}),
        )
        assert base.tuning_hash != moved.tuning_hash

    def test_it_notices_a_tolerance_too(self):
        base = RunConfig(run_id="a")
        moved = RunConfig(
            run_id="a",
            tolerances=base.tolerances.model_copy(update={"amount_bps": 500}),
        )
        assert base.tuning_hash != moved.tuning_hash

    def test_rows_that_disagree_on_tuning_are_refused(self, two_corpora, monkeypatch):
        """A table whose rows differ in more than their ladder is not an
        ablation, so `run_ablation` refuses to build one rather than publishing
        a comparison that is not one."""
        real = run_system
        seen = {"n": 0}

        def drifting(directory, **kwargs):
            seen["n"] += 1
            if seen["n"] > 2:
                kwargs = dict(kwargs)
            run = real(directory, **kwargs)
            if seen["n"] > 2:
                moved = run.config.model_copy(
                    update={
                        "thresholds": run.config.thresholds.model_copy(
                            update={"tau_low": 0.11}
                        )
                    }
                )
                object.__setattr__(run, "config", moved)
            return run

        monkeypatch.setattr("ledgerloop.eval.ablation.run_system", drifting)
        with pytest.raises(ValueError, match="did not share a tuning configuration"):
            run_ablation(two_corpora)


class TestTheTableItself:
    def test_it_has_one_row_per_planned_ladder(self, two_corpora):
        artifact = run_ablation(two_corpora)
        assert [row.tiers for row in artifact.rows] == [
            tuple(tiers) for tiers in ABLATION_LADDERS
        ]

    def test_recall_never_falls_as_the_ladder_grows(self, two_corpora):
        """Each tier sees only what the previous ones left, so a longer prefix
        can only add. A row that went backwards would mean a tier was consuming
        records another one would have matched correctly."""
        artifact = run_ablation(two_corpora)
        recalls = [row.recall.mean for row in artifact.rows]
        assert recalls == sorted(recalls)

    def test_no_row_ever_asserts_a_wrong_link(self, two_corpora):
        """The precision-first claim, checked at every rung rather than only at
        the top: no tier buys its recall with a wrong auto-match.

        Stated as false positives rather than as precision because a row that
        asserted *nothing* reports precision 0.0 -- the documented degenerate
        convention (`eval/metrics.py`), and a real case: T0 alone finds no clean
        order reference at all on one of these 60-order corpora. A row with no
        predictions has not achieved perfect precision, and the metric refuses
        to say it has.
        """
        artifact = run_ablation(two_corpora)
        for row in artifact.rows:
            assert row.false_positives.mean == 0.0
            assert row.false_positive_cost_minor.mean == 0.0

    def test_precision_is_one_wherever_a_row_asserted_anything(self, two_corpora):
        artifact = run_ablation(two_corpora)
        for row in artifact.rows:
            for run in row.runs:
                if run.auto_matched > 0:
                    assert run.precision == 1.0

    def test_the_marginal_column_is_a_difference_of_means(self, two_corpora):
        artifact = run_ablation(two_corpora)
        assert artifact.marginal(0, "recall") == artifact.rows[0].recall.mean
        for index in range(1, len(artifact.rows)):
            expected = (
                artifact.rows[index].recall.mean - artifact.rows[index - 1].recall.mean
            )
            assert artifact.marginal(index, "recall") == pytest.approx(expected)

    def test_a_mixed_difficulty_table_is_refused(self, tmp_path):
        """An ablation compares ladders, so every corpus in it must be one split
        at one difficulty; otherwise a row's mean mixes two populations."""
        from ledgerloop.models.enums import Difficulty

        directories = []
        for difficulty in (Difficulty.EASY, Difficulty.HARD):
            directory = tmp_path / difficulty.value
            generate_to_disk(
                GeneratorConfig(split=SplitName.DEV, difficulty=difficulty, seed=3),
                directory,
            )
            directories.append(directory)
        with pytest.raises(ValueError, match="one split at one difficulty"):
            run_ablation(directories)

    def test_an_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="at least one dataset directory"):
            run_ablation([])


class TestReproducibility:
    def test_two_ablations_over_one_corpus_agree_exactly(self, corpus):
        """Fixed seeds, deterministic generation, deterministic evaluation. Two
        runs of the same command produce the same artefact, **byte for byte**.

        Byte-identity is available here only because `RunSummary` carries no
        wall clock: timing is the one figure that legitimately differs between
        two runs over identical data, and it lives in the report's labelled
        timings block instead.
        """
        first = run_ablation([corpus])
        second = run_ablation([corpus])
        assert first.model_dump_json() == second.model_dump_json()

    def test_the_artefact_round_trips_through_disk(self, corpus, tmp_path):
        artifact = run_ablation([corpus])
        path = tmp_path / "ablation.json"
        artifact.save(path)
        assert AblationArtifact.load(path).model_dump_json() == artifact.model_dump_json()

    def test_the_headline_row_matches_a_plain_run_of_the_pipeline(self, corpus):
        """The ablation's last row must be the system, not a re-implementation
        of it: same harness, same config, same numbers."""
        artifact = run_ablation([corpus])
        full = run_system(corpus, measure_calibration_quality=False)
        assert artifact.rows[-1].recall.mean == pytest.approx(full.summary().recall)
        assert artifact.rows[-1].precision.mean == pytest.approx(full.summary().precision)
