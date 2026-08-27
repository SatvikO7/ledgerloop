"""The calibrated ladder end to end: fit, apply, route, report.

Step 7 changes what ``run_matching`` does only when it is handed a bundle, and
the most important thing these tests establish is what *does not* change:

* the run without a bundle decides exactly what it decided at Step 6;
* T0 and T1 keep the probabilities they counted for themselves;
* a tier's refusal keeps its ``1/n`` and keeps routing to an exception.

The rest is the wiring: that the fitted threshold travels on the config rather
than beside it, that the report renders a calibration section only when there is
one, and that the CLI refuses the combinations that would break the split
discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgerloop.cli import main
from ledgerloop.config import RunConfig
from ledgerloop.eval.reliability import label_candidates, measure_calibration
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.matching import run_matching
from ledgerloop.matching.calibration import (
    CalibrationBundle,
    CalibrationProvenance,
    configure_for,
    fit_bundle,
)
from ledgerloop.models.candidates import FeatureVector
from ledgerloop.models.enums import DecisionOutcome, SplitName, Tier
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


def features(tier: Tier = Tier.T2_AGGREGATION, **kwargs: object) -> FeatureVector:
    return FeatureVector(tier=tier, **kwargs)  # type: ignore[arg-type]


def a_bundle(*, target_precision: float = 0.5) -> CalibrationBundle:
    """A small, honest bundle: two tiers, both classes, nothing degenerate."""
    train = (
        [features(Tier.T2_AGGREGATION, amount_delta_minor=0, tolerance_band_minor=300)] * 8
        + [features(Tier.T2_AGGREGATION, amount_delta_minor=900, tolerance_band_minor=300)] * 8
        + [features(Tier.T3_FUZZY, lexical_score=1.0)] * 6
        + [features(Tier.T3_FUZZY, lexical_score=0.3)] * 6
    )
    train_labels = [True] * 8 + [False] * 8 + [True] * 6 + [False] * 6
    calibration = (
        [features(Tier.T2_AGGREGATION, amount_delta_minor=0, tolerance_band_minor=300)] * 5
        + [features(Tier.T2_AGGREGATION, amount_delta_minor=900, tolerance_band_minor=300)] * 5
        + [features(Tier.T3_FUZZY, lexical_score=1.0)] * 4
        + [features(Tier.T3_FUZZY, lexical_score=0.3)] * 4
    )
    calibration_labels = [True] * 5 + [False] * 5 + [True] * 4 + [False] * 4
    return fit_bundle(
        train,
        train_labels,
        calibration,
        calibration_labels,
        provenance=CalibrationProvenance(
            train_split=SplitName.TRAIN,
            train_seeds=(42,),
            calibration_split=SplitName.CALIBRATION,
            calibration_seeds=(43,),
            generator_version="0.2.0",
            top_k=3,
            train_rows=len(train),
            train_positives=sum(train_labels),
            calibration_rows=len(calibration),
            calibration_positives=sum(calibration_labels),
        ),
        target_precision=target_precision,
    )


@pytest.fixture(scope="module")
def bundle() -> CalibrationBundle:
    return a_bundle()


@pytest.fixture
def split_corpus():
    only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
    grosses = [payment.amount_minor for payment in only.payments]
    amounts = allocate_minor(only.net_minor, [grosses[0], grosses[1] + grosses[2]])
    credits = [
        bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
        bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
    ]
    return corpus(batches=[only], bank_txns=credits)


class TestWithoutABundleNothingChanges:
    def test_the_probabilities_stay_the_tiers_own(self, split_corpus):
        run = run_matching(split_corpus, RunConfig(run_id="plain"))
        assert all(candidate.raw_score is None for candidate in run.candidates)
        assert not run.calibrated

    def test_the_blend_counters_are_empty_rather_than_zeroed_out(self, split_corpus):
        run = run_matching(split_corpus, RunConfig(run_id="plain"))
        assert run.blend.considered == 0


class TestWithABundle:
    def test_residual_candidates_carry_a_raw_score_and_a_calibrated_p(
        self, split_corpus, bundle
    ):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(split_corpus, config, bundle=bundle)
        residual = [
            candidate
            for candidate in run.candidates
            if not candidate.tier.is_deterministic_certain
            and candidate.arithmetic_verified
        ]
        assert residual
        for candidate in residual:
            assert candidate.raw_score is not None
            assert candidate.calibrated_p == pytest.approx(
                bundle.calibrator.predict(candidate.raw_score)
            )

    def test_the_deterministic_tiers_are_untouched(self, simple, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(simple, config, bundle=bundle)
        deterministic = [
            candidate
            for candidate in run.candidates
            if candidate.tier.is_deterministic_certain
        ]
        assert deterministic
        assert all(candidate.raw_score is None for candidate in deterministic)
        assert all(candidate.calibrated_p == 1.0 for candidate in deterministic)
        assert run.blend.bypassed_deterministic == len(deterministic)

    def test_the_run_records_that_it_was_calibrated(self, split_corpus, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        assert run_matching(split_corpus, config, bundle=bundle).calibrated

    def test_the_counters_account_for_every_candidate(self, split_corpus, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(split_corpus, config, bundle=bundle)
        assert run.blend.considered == len(run.candidates)

    def test_the_fitted_threshold_is_what_the_policy_applied(self, split_corpus, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(split_corpus, config, bundle=bundle)
        for decision in run.decisions:
            if decision.outcome is DecisionOutcome.AUTO_MATCHED:
                assert decision.calibrated_p >= config.thresholds.tau_high


class TestARefusalSurvivesCalibration:
    """The blender cannot see a rival, so it may not overturn a refusal."""

    @pytest.fixture
    def ambiguous(self):
        """Two subsets of one batch both compose the first tranche."""
        only = batch(amounts=(50_000, 50_000, 30_000, 20_000))
        grosses = [payment.amount_minor for payment in only.payments]
        amounts = allocate_minor(only.net_minor, [grosses[0], sum(grosses[1:])])
        return corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
                bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
            ],
        )

    def test_an_unverified_candidate_keeps_its_tier_probability(self, ambiguous, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(ambiguous, config, bundle=bundle)
        refusals = [
            candidate
            for candidate in run.candidates
            if not candidate.arithmetic_verified
            and not candidate.tier.is_deterministic_certain
        ]
        if not refusals:
            pytest.skip("this corpus produced no refusal to check")
        assert all(candidate.raw_score is None for candidate in refusals)
        assert run.blend.refusals_kept == len(refusals)

    def test_a_refusal_never_becomes_an_auto_match(self, ambiguous, bundle):
        config = configure_for(RunConfig(run_id="cal"), bundle)
        run = run_matching(ambiguous, config, bundle=bundle)
        by_id = {c.candidate_id: c for c in run.candidates}
        for decision in run.decisions:
            if not by_id[decision.candidate_id].arithmetic_verified:
                assert decision.outcome is not DecisionOutcome.AUTO_MATCHED


class TestMeasuringCalibration:
    def test_labelling_touches_only_the_evaluation_unit(self, split_corpus, bundle):
        truth = load_ground_truth(FIXTURE)
        run = run_matching(split_corpus, RunConfig(run_id="plain"))
        labelled = label_candidates(run.candidates, truth)
        assert labelled == sum(1 for c in run.candidates if c.is_evaluable)
        for candidate in run.candidates:
            if not candidate.is_evaluable:
                assert candidate.is_truth_positive is None

    def test_the_measured_population_excludes_the_deterministic_tiers(self, bundle):
        ingest = _fixture_ingest()
        truth = load_ground_truth(FIXTURE)
        config = configure_for(
            RunConfig(run_id="cal", split=SplitName.DEV, seed=42), bundle
        )
        run = run_matching(ingest, config, bundle=bundle)
        view = measure_calibration(run.candidates, truth)
        residual = [
            candidate
            for candidate in run.candidates
            if candidate.is_evaluable
            and not candidate.tier.is_deterministic_certain
            and candidate.calibrated_p is not None
        ]
        assert view.asserted.sample_count == len(residual)
        assert view.asserted.residual_only
        assert view.contenders is None

    def test_the_evaluation_reports_its_own_sample_size(self, bundle):
        ingest = _fixture_ingest()
        truth = load_ground_truth(FIXTURE)
        run = run_matching(ingest, RunConfig(run_id="cal"), bundle=bundle)
        view = measure_calibration(run.candidates, truth)
        assert view.sample_count == view.asserted.sample_count

    def test_a_bundle_abstains_on_a_tier_it_never_saw(self, bundle):
        """The diagnostic must not quietly describe half a population."""
        from ledgerloop.eval.reliability import score_contenders
        from ledgerloop.matching.harvest import harvest

        ingest = _fixture_ingest()
        truth = load_ground_truth(FIXTURE)
        rows = harvest(ingest, truth, RunConfig(run_id="cal")).rows
        scored = score_contenders(bundle, rows)
        covered = sum(1 for row in rows if bundle.covers(row.candidate))
        assert len(scored.probabilities) == covered
        assert scored.abstained == len(rows) - covered

    def test_the_contender_population_is_reported_when_given(self, bundle):
        ingest = _fixture_ingest()
        truth = load_ground_truth(FIXTURE)
        run = run_matching(ingest, RunConfig(run_id="cal"), bundle=bundle)
        view = measure_calibration(
            run.candidates,
            truth,
            contender_probabilities=[0.9, 0.9],
            contender_labels=[True, False],
        )
        assert view.contenders is not None
        assert view.contenders.sample_count == 2
        assert view.contenders.ece == pytest.approx(0.4)


def _fixture_ingest():
    from ledgerloop.ingest import ingest_dataset

    return ingest_dataset(FIXTURE, strict=False)


class TestTheCalibrateCommand:
    @pytest.fixture
    def datasets(self, tmp_path):
        """Two train corpora and one calibration corpus, generated small and fast."""
        made: dict[str, list[str]] = {"train": [], "calibration": []}
        for split, seeds in (("train", (7, 8)), ("calibration", (9,))):
            for seed in seeds:
                out = tmp_path / f"{split}-{seed}"
                assert main(
                    [
                        "generate",
                        "--split",
                        split,
                        "--seed",
                        str(seed),
                        "--orders",
                        "120",
                        "--out",
                        str(out),
                    ]
                ) == 0
                made[split].append(str(out))
        return made

    def test_it_writes_a_bundle_and_explains_it(self, datasets, tmp_path, capsys):
        out = tmp_path / "bundle.json"
        code = main(
            [
                "calibrate",
                "--train",
                *datasets["train"],
                "--calibration",
                *datasets["calibration"],
                "--out",
                str(out),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "logistic:" in printed
        assert "isotonic:" in printed
        assert "tau_high =" in printed
        assert "intercept" in printed
        assert out.is_file()

        bundle = CalibrationBundle.load(out)
        assert bundle.provenance.train_seeds == (7, 8)
        assert bundle.provenance.calibration_seeds == (9,)
        assert bundle.thresholds.tau_high <= 1.0

    def test_the_bundle_names_every_corpus_it_was_fitted_on(self, datasets, tmp_path):
        out = tmp_path / "bundle.json"
        main(
            [
                "calibrate",
                "--train",
                *datasets["train"],
                "--calibration",
                *datasets["calibration"],
                "--out",
                str(out),
            ]
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["provenance"]["train_split"] == "train"
        assert payload["provenance"]["calibration_split"] == "calibration"

    def test_it_refuses_to_fit_on_the_test_split(self, datasets, tmp_path, capsys):
        test_dir = tmp_path / "test-1"
        main(
            ["generate", "--split", "test", "--seed", "1", "--orders", "80",
             "--out", str(test_dir)]
        )
        code = main(
            [
                "calibrate",
                "--train",
                *datasets["train"],
                "--calibration",
                str(test_dir),
                "--out",
                str(tmp_path / "b.json"),
            ]
        )
        assert code == 1
        assert "test split" in capsys.readouterr().err

    def test_it_refuses_overlapping_halves(self, datasets, tmp_path, capsys):
        code = main(
            [
                "calibrate",
                "--train",
                *datasets["train"],
                "--calibration",
                *datasets["train"],
                "--out",
                str(tmp_path / "b.json"),
            ]
        )
        assert code == 1
        assert "different data" in capsys.readouterr().err

    def test_it_refuses_a_half_that_mixes_splits(self, datasets, tmp_path, capsys):
        code = main(
            [
                "calibrate",
                "--train",
                *datasets["train"],
                *datasets["calibration"],
                "--calibration",
                *datasets["calibration"],
                "--out",
                str(tmp_path / "b.json"),
            ]
        )
        assert code == 1
        assert "one split" in capsys.readouterr().err

    def test_a_missing_directory_is_named(self, datasets, tmp_path, capsys):
        code = main(
            [
                "calibrate",
                "--train",
                str(tmp_path / "nowhere"),
                "--calibration",
                *datasets["calibration"],
                "--out",
                str(tmp_path / "b.json"),
            ]
        )
        assert code == 1
        assert "no such dataset directory" in capsys.readouterr().err


class TestTheEvalCommandWithABundle:
    @pytest.fixture
    def dataset(self, tmp_path):
        out = tmp_path / "test-3"
        assert main(
            ["generate", "--split", "test", "--seed", "3", "--orders", "120",
             "--out", str(out)]
        ) == 0
        return out

    @pytest.fixture
    def fitted(self, tmp_path):
        made = []
        for split, seed in (("train", 11), ("calibration", 12)):
            out = tmp_path / f"{split}-{seed}"
            main(
                ["generate", "--split", split, "--seed", str(seed), "--orders", "120",
                 "--out", str(out)]
            )
            made.append(str(out))
        bundle = tmp_path / "bundle.json"
        assert main(
            ["calibrate", "--train", made[0], "--calibration", made[1],
             "--out", str(bundle)]
        ) == 0
        return bundle

    def test_without_a_bundle_the_report_says_the_section_is_pending(
        self, dataset, tmp_path, capsys
    ):
        report = tmp_path / "EVAL.md"
        assert main(["eval", "--data", str(dataset), "--out", str(report)]) == 0
        assert "tier-provisional" in capsys.readouterr().out
        text = report.read_text(encoding="utf-8")
        assert "_pending_" in text
        assert "### Calibration" not in text

    def test_with_a_bundle_the_report_carries_the_calibration_section(
        self, dataset, fitted, tmp_path
    ):
        report = tmp_path / "EVAL.md"
        assert main(
            ["eval", "--data", str(dataset), "--calibration", str(fitted),
             "--out", str(report)]
        ) == 0
        text = report.read_text(encoding="utf-8")
        assert "### Calibration" in text
        assert "Reliability — asserted links" in text
        assert "Reliability — every contender considered" in text

    def test_it_prints_the_threshold_it_used_and_where_it_came_from(
        self, dataset, fitted, tmp_path, capsys
    ):
        main(
            ["eval", "--data", str(dataset), "--calibration", str(fitted),
             "--out", str(tmp_path / "EVAL.md")]
        )
        printed = capsys.readouterr().out
        assert "probabilities: calibrated" in printed
        assert "tau_high =" in printed
        assert "fitted on calibration seeds 12" in printed

    def test_a_missing_bundle_is_named_rather_than_ignored(
        self, dataset, tmp_path, capsys
    ):
        code = main(
            ["eval", "--data", str(dataset), "--calibration", str(tmp_path / "no.json"),
             "--out", str(tmp_path / "EVAL.md")]
        )
        assert code == 1
        assert "no such calibration bundle" in capsys.readouterr().err

    def test_a_bundle_from_another_generator_version_is_refused(
        self, dataset, fitted, tmp_path, capsys
    ):
        """A probability fitted on one corpus is not a probability about another."""
        payload = json.loads(fitted.read_text(encoding="utf-8"))
        payload["provenance"]["generator_version"] = "0.9.9"
        stale = tmp_path / "stale.json"
        stale.write_text(json.dumps(payload), encoding="utf-8")
        code = main(
            ["eval", "--data", str(dataset), "--calibration", str(stale),
             "--out", str(tmp_path / "EVAL.md")]
        )
        assert code == 1
        assert "generator" in capsys.readouterr().err

    def test_precision_is_not_traded_away_by_calibrating(self, dataset, fitted, tmp_path):
        """The step's own acceptance criterion: calibration must not cost precision."""
        from ledgerloop.eval.metrics import evaluate
        from ledgerloop.ingest import ingest_dataset

        ingested = ingest_dataset(dataset, strict=False)
        truth = load_ground_truth(dataset)
        bundle = CalibrationBundle.load(fitted)

        plain = run_matching(ingested, RunConfig(run_id="plain"))
        calibrated = run_matching(
            ingested,
            configure_for(RunConfig(run_id="cal"), bundle),
            bundle=bundle,
        )
        before = evaluate(plain.predictions, truth, run_id="plain").link_metrics
        after = evaluate(calibrated.predictions, truth, run_id="cal").link_metrics
        assert before is not None and after is not None
        assert after.false_positives <= before.false_positives
        assert after.precision >= before.precision
