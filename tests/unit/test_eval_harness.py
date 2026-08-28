"""The shared run harness: leakage, split isolation, and reproducibility.

Step 10 runs the pipeline thirty-odd times per evaluation, and every one of
those runs goes through :func:`~ledgerloop.eval.harness.run_system`. That makes
this module the single place where ground truth could leak into a decision, so
it is the place the leakage tests live.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgerloop.config import GeneratorConfig, LLMConfig, RunConfig
from ledgerloop.eval.harness import (
    DEFAULT_TIERS,
    DETERMINISTIC_TIERS,
    StaleCalibrationError,
    load_bundle_for,
    run_system,
)
from ledgerloop.eval.truth_io import load_manifest
from ledgerloop.exceptions import classify_exceptions
from ledgerloop.generator import generate_to_disk
from ledgerloop.llm.client import LLMClient
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.models.enums import SplitName


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("harness") / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), directory)
    return directory


class TestNoLeakage:
    def test_the_exception_classifier_never_sees_a_ground_truth_label(self, corpus):
        """`measure_calibration` writes `is_truth_positive` onto the candidate
        objects in place, and those same objects are handed to the classifier.

        The ordering in `run_system` puts the classifier first for exactly that
        reason. This test checks the stronger property the ordering protects:
        the classifier's output does not depend on the label at all, so even a
        future reordering could not change a queue.
        """
        run = run_system(corpus, measure_calibration_quality=False)
        assert run.matched.context is not None
        honest = classify_exceptions(
            run.matched.context,
            run.matched.decisions,
            run.matched.candidates,
            run.config,
            merchant_profiles=run.matched.merchant_spellings,
        )
        # Every label flipped to the wrong answer. A classifier consulting the
        # field would now produce a different queue.
        for candidate in run.matched.candidates:
            if candidate.is_evaluable:
                candidate.is_truth_positive = not bool(candidate.is_truth_positive)
        poisoned = classify_exceptions(
            run.matched.context,
            run.matched.decisions,
            run.matched.candidates,
            run.config,
            merchant_profiles=run.matched.merchant_spellings,
        )
        assert [item.exception_class for item in honest.exceptions] == [
            item.exception_class for item in poisoned.exceptions
        ]
        assert [item.impact_minor for item in honest.exceptions] == [
            item.impact_minor for item in poisoned.exceptions
        ]

    def test_the_matcher_does_not_import_the_evaluator(self):
        """`matching` scores nothing and must not be able to. If it could import
        `eval`, a tier could reach ground truth through `truth_io` and every
        number in this project would become a claim about itself."""
        import ledgerloop.matching.pipeline as pipeline

        source = Path(pipeline.__file__).read_text(encoding="utf-8")
        assert "from ledgerloop.eval.metrics import PredictedLink" in source
        # The contract type only. Nothing that can read a label off disk.
        assert "truth_io" not in source
        assert "load_ground_truth" not in source

    def test_the_llm_baseline_cannot_reach_the_ladder(self):
        """B2's predictions go to `evaluate` and to the report, and nowhere
        else. `matching` never imports `eval`, so the import direction is what
        guarantees it rather than a convention."""
        import ledgerloop.matching.pipeline as pipeline
        import ledgerloop.matching.policy as policy

        for module in (pipeline, policy):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "llm_baseline" not in source

    def test_matching_never_reaches_llm_even_transitively(self):
        """ARCHITECTURE.md §6, decision 43: `matching` must not depend on `llm`,
        because the moment it does, `--no-llm` stops being one code path with a
        branch and becomes a second implementation nobody measures.

        Transitively, not just directly. Step 10 broke this by accident and the
        interpreter reported it four files from the cause: `report.py` imported
        the ablation runner, which imports the harness, which imports `llm` --
        and `matching.pipeline` imports `eval.metrics` for one contract type,
        which initialises the whole `eval` package. The artefact models were
        split into `eval/artifacts.py` to break it, and this is the test that
        keeps it broken.
        """
        import subprocess
        import sys

        probe = (
            "import sys; import ledgerloop.matching.pipeline; "
            "leaked = sorted(m for m in sys.modules if m.startswith('ledgerloop.llm')); "
            "print(','.join(leaked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == ""

    def test_the_report_imports_the_artefacts_not_the_runners(self):
        """A document needs the shape of a result, never the machinery that
        produced one -- and here that distinction is what keeps the layering
        acyclic rather than merely tidy."""
        import ledgerloop.eval.report as report

        source = Path(report.__file__).read_text(encoding="utf-8")
        assert "from ledgerloop.eval.artifacts import" in source
        for runner in ("eval.ablation", "eval.sweep", "eval.llm_baseline", "eval.harness"):
            assert f"from ledgerloop.{runner} import" not in source

    def test_the_artefact_models_carry_no_heavy_dependency(self):
        """`eval/artifacts.py` is imported by the renderer, so it must stay on
        the light side of the layering: models, and nothing that runs."""
        import ledgerloop.eval.artifacts as artifacts

        imports = [
            line
            for line in Path(artifacts.__file__).read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]
        for heavy in ("ledgerloop.llm", "ledgerloop.matching", "ledgerloop.ingest"):
            assert not any(heavy in line for line in imports)

    def test_predictions_are_unchanged_by_measuring_calibration(self, corpus, tmp_path):
        """The measurement labels a finished run. It must not be able to move
        one, and the check is that the links are identical either way."""
        bundle_path = tmp_path / "bundle.json"
        _fit_small_bundle(tmp_path, bundle_path)
        bundle = CalibrationBundle.load(bundle_path)
        measured = run_system(corpus, bundle=bundle, measure_calibration_quality=True)
        unmeasured = run_system(corpus, bundle=bundle, measure_calibration_quality=False)
        assert measured.matched.predictions == unmeasured.matched.predictions
        assert measured.metrics.auto_match_precision == unmeasured.metrics.auto_match_precision


class TestSplitIsolation:
    def test_a_bundle_from_another_generator_version_is_refused(self, corpus, tmp_path):
        """A probability fitted against generator 0.2.0 is not a probability
        about 0.3.0 data, and the bundle records the version precisely so a run
        can refuse a stale one rather than quietly applying it."""
        bundle_path = tmp_path / "bundle.json"
        _fit_small_bundle(tmp_path, bundle_path)
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        payload["provenance"]["generator_version"] = "9.9.9"
        bundle_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(StaleCalibrationError, match="generator"):
            load_bundle_for(bundle_path, load_manifest(corpus))

    def test_a_matching_version_is_accepted(self, corpus, tmp_path):
        bundle_path = tmp_path / "bundle.json"
        _fit_small_bundle(tmp_path, bundle_path)
        assert load_bundle_for(bundle_path, load_manifest(corpus)) is not None

    def test_a_bundle_can_never_be_fitted_on_test(self, tmp_path):
        """`CalibrationProvenance` refuses it, so the discipline is a type error
        rather than a convention someone has to remember."""
        from ledgerloop.matching.calibration import CalibrationProvenance

        with pytest.raises(ValueError):
            CalibrationProvenance(
                train_split=SplitName.TEST,
                train_seeds=(42,),
                calibration_split=SplitName.CALIBRATION,
                calibration_seeds=(47,),
                generator_version="0.2.0",
                top_k=3,
            )

    def test_the_two_halves_may_not_share_a_seed(self):
        from ledgerloop.matching.calibration import CalibrationProvenance

        with pytest.raises(ValueError):
            CalibrationProvenance(
                train_split=SplitName.TRAIN,
                train_seeds=(42, 43),
                calibration_split=SplitName.TRAIN,
                calibration_seeds=(43,),
                generator_version="0.2.0",
                top_k=3,
            )

    def test_the_generator_streams_are_split_scoped(self, tmp_path):
        """`make eval` reuses seeds 42-46 on both `train` and `test`. That is
        only sound because the generator seeds on `<seed>:<split>:<purpose>`, so
        the two corpora are independent rather than nested."""
        train, test = tmp_path / "train", tmp_path / "test"
        generate_to_disk(GeneratorConfig(split=SplitName.TRAIN, seed=42, order_count=40), train)
        generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42, order_count=40), test)
        left = (train / "ledger_orders.csv").read_text(encoding="utf-8")
        right = (test / "ledger_orders.csv").read_text(encoding="utf-8")
        assert left != right


class TestReproducibility:
    def test_two_runs_over_one_corpus_produce_the_same_links(self, corpus):
        first = run_system(corpus, measure_calibration_quality=False)
        second = run_system(corpus, measure_calibration_quality=False)
        assert first.matched.predictions == second.matched.predictions

    def test_and_the_same_summary_row(self, corpus):
        """The summary carries no wall clock, so the rows compare exactly."""
        first = run_system(corpus, measure_calibration_quality=False)
        second = run_system(corpus, measure_calibration_quality=False)
        assert first.summary() == second.summary()

    def test_and_the_same_exception_queue(self, corpus):
        first = run_system(corpus, measure_calibration_quality=False)
        second = run_system(corpus, measure_calibration_quality=False)
        assert [item.exception_class for item in first.exceptions] == [
            item.exception_class for item in second.exceptions
        ]
        assert [item.impact_minor for item in first.exceptions] == [
            item.impact_minor for item in second.exceptions
        ]

    def test_regeneration_is_byte_identical(self, tmp_path):
        """Fixed seed, deterministic generation. The whole reproducibility claim
        rests on this, so it is asserted rather than assumed."""
        left, right = tmp_path / "a", tmp_path / "b"
        for target in (left, right):
            generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=5), target)
        for name in sorted(p.name for p in left.iterdir()):
            assert (left / name).read_bytes() == (right / name).read_bytes()


class TestTierSelection:
    def test_t5_is_dropped_when_no_model_is_reachable(self, corpus):
        """A config listing T5 on a machine with no key ran T0-T4, and the run
        says T0-T4 rather than reporting a tier it never invoked."""
        client = LLMClient(config=LLMConfig(enabled=True), provider=None)
        run = run_system(
            corpus,
            client=client,
            enabled_tiers=DEFAULT_TIERS,
            measure_calibration_quality=False,
        )
        assert run.config.enabled_tiers == DETERMINISTIC_TIERS
        assert run.llm_available is False
        assert run.label == "T0-T4"

    def test_no_client_at_all_is_the_same_place(self, corpus):
        run = run_system(corpus, measure_calibration_quality=False)
        assert run.config.enabled_tiers == DETERMINISTIC_TIERS
        assert run.llm_available is False

    def test_an_ablation_row_below_t5_never_repairs_a_narration(self, corpus):
        """The narration repair is call site 1 and is gated on T5 as well as on
        the client. An ablation row that ran the deterministic ladder while a
        model quietly fixed its inputs would credit the tiers with the model's
        contribution."""
        run = run_system(
            corpus, enabled_tiers=(0, 1, 2, 3, 4), measure_calibration_quality=False
        )
        assert run.llm.narration.attempted == 0
        assert run.cost.llm_calls == 0

    def test_the_summary_counts_the_evaluation_unit_and_not_the_structural_edges(
        self, corpus
    ):
        """`MatchRun.auto_matched` counts every auto-matched decision, structural
        edges included. Reporting that beside a yield restricted to the
        evaluation unit would put a numerator and denominator from two different
        populations in adjacent columns."""
        run = run_system(corpus, measure_calibration_quality=False)
        row = run.summary()
        assert row.auto_matched == len(run.matched.predictions)
        assert row.auto_matched == row.true_positives + row.false_positives
        assert row.auto_matched <= run.matched.auto_matched


def _fit_small_bundle(tmp_path: Path, out: Path) -> None:
    """Fit a bundle from two small train and one small calibration corpus."""
    from ledgerloop.fitting import fit_from_corpora, harvest_corpora

    train_dirs, cal_dirs = [], []
    for seed in (101, 102):
        directory = tmp_path / f"train-{seed}"
        generate_to_disk(
            GeneratorConfig(split=SplitName.TRAIN, seed=seed, order_count=80), directory
        )
        train_dirs.append(directory)
    for seed in (201,):
        directory = tmp_path / f"cal-{seed}"
        generate_to_disk(
            GeneratorConfig(split=SplitName.CALIBRATION, seed=seed, order_count=80),
            directory,
        )
        cal_dirs.append(directory)
    config = RunConfig(run_id="fit", split=SplitName.TRAIN)
    bundle = fit_from_corpora(
        harvest_corpora(train_dirs, config=config),
        harvest_corpora(cal_dirs, config=config),
        target_precision=0.99,
    )
    bundle.save(out)
