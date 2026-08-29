"""`ledgerloop demo` -- the one command a reviewer runs.

It chains four stages that each have their own command and their own tests, so
what is tested here is the chaining: that it produces a readable run, that it
is idempotent, that it degrades honestly when something is missing, and that it
never quietly becomes a second implementation of any stage it calls.

Every test passes `--no-ui`. Launching Streamlit blocks, and a test that hung
waiting for a browser would be worse than no test.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.agent.graph import langgraph_available
from ledgerloop.agent.store import list_runs, load_run
from ledgerloop.cli import (
    DEMO_CALIBRATION_SEEDS,
    DEMO_TRAIN_SEEDS,
    _build_parser,
    main,
)
from ledgerloop.matching.calibration import CalibrationBundle
from ledgerloop.models.enums import SplitName

pytestmark = pytest.mark.skipif(
    not langgraph_available(), reason="langgraph is an optional extra"
)


def _demo(tmp_path, *extra: str) -> int:
    """The demo, entirely inside a temporary directory."""
    return main(
        [
            "demo",
            "--data-dir",
            str(tmp_path / "data"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--bundle",
            str(tmp_path / "calibration.json"),
            "--no-ui",
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """One full demo, run once. It is the slowest fixture in the suite."""
    root = tmp_path_factory.mktemp("demo")
    code = main(
        [
            "demo",
            "--data-dir",
            str(root / "data"),
            "--runs-dir",
            str(root / "runs"),
            "--bundle",
            str(root / "calibration.json"),
            "--no-ui",
        ]
    )
    assert code == 0
    return root


class TestItRunsEndToEnd:
    def test_it_exits_zero(self, completed):
        assert completed.is_dir()

    def test_it_generates_both_fitting_halves_and_the_demo_corpus(self, completed):
        data = completed / "data"
        for seed in DEMO_TRAIN_SEEDS:
            assert (data / f"train-standard-{seed}" / "manifest.json").is_file()
        for seed in DEMO_CALIBRATION_SEEDS:
            assert (data / f"calibration-standard-{seed}" / "manifest.json").is_file()
        assert (data / "test-standard-42" / "manifest.json").is_file()

    def test_it_does_not_generate_the_eval_only_corpora(self, completed):
        """The demo needs the demo corpus and the two fitting halves. Dragging
        in the fifteen sweep corpora would triple its wall clock for tables it
        does not show."""
        names = {path.name for path in (completed / "data").iterdir()}
        assert not any(name.startswith("test-easy") for name in names)
        assert not any(name.startswith("test-hard") for name in names)

    def test_it_writes_a_bundle_fitted_on_train_and_calibration(self, completed):
        bundle = CalibrationBundle.load(completed / "calibration.json")
        assert bundle.provenance.train_split.value == "train"
        assert bundle.provenance.calibration_split.value == "calibration"
        assert bundle.provenance.train_seeds == DEMO_TRAIN_SEEDS
        assert bundle.provenance.calibration_seeds == DEMO_CALIBRATION_SEEDS

    def test_the_bundle_was_never_fitted_on_test(self, completed):
        """The discipline the whole evaluation rests on, checked at the one
        place a demo could quietly break it."""
        bundle = CalibrationBundle.load(completed / "calibration.json")
        assert bundle.provenance.train_split.value != "test"
        assert bundle.provenance.calibration_split.value != "test"

    def test_it_writes_a_run_the_ui_can_read(self, completed):
        runs = list_runs(completed / "runs")
        assert len(runs) == 1
        stored = runs[0]
        assert stored.audit
        assert stored.decisions
        assert stored.summary["engine"] == "langgraph"

    def test_the_run_carries_the_four_files(self, completed):
        directory = (completed / "runs" / "t0t4-test-42")
        for name in ("run.json", "audit.jsonl", "exceptions.json", "decisions.json"):
            assert (directory / name).is_file()

    def test_the_demo_run_makes_no_wrong_auto_match(self, completed):
        """The headline claim, on the corpus the demo actually shows."""
        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.metrics["false_positives"] == 0
        assert stored.metrics["false_positive_cost_minor"] == 0
        assert stored.metrics["auto_match_precision"] == 1.0

    def test_it_ran_deterministically(self, completed):
        """No key in the test environment, so the demo must report a run with no
        model rather than silently behaving as though one were present."""
        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.summary["llm"]["available"] is False
        assert stored.summary["llm"]["calls"] == 0
        assert stored.summary["llm"]["total_tokens"] == 0

    def test_it_reports_the_unmatchable_floor_rather_than_hiding_it(self, completed):
        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.metrics["unmatchable_count"] > 0
        assert stored.summary["coverage"]["unmatchable"] > 0


class TestItIsIdempotent:
    def test_a_second_run_regenerates_nothing(self, completed, capsys):
        """Generation is a pure function of the seed, so an existing corpus is
        byte-identical and rewriting it buys nothing but wall clock."""
        code = _demo(completed)
        assert code == 0
        out = capsys.readouterr().out
        assert "0 generated, 10 already present" in out

    def test_a_second_run_reuses_the_bundle(self, completed, capsys):
        _demo(completed)
        assert "already exists" in capsys.readouterr().out

    def test_and_produces_the_same_numbers(self, completed):
        before = load_run(completed / "runs" / "t0t4-test-42")
        assert before is not None
        first = dict(before.metrics)
        assert _demo(completed) == 0
        after = load_run(completed / "runs" / "t0t4-test-42")
        assert after is not None
        # Wall clock is measured and legitimately moves; nothing else may.
        for key, value in first.items():
            if key in {"wall_clock_ms", "records_per_second"}:
                continue
            assert after.metrics[key] == value, key

    def test_refit_rewrites_the_bundle(self, completed, capsys):
        assert _demo(completed, "--refit") == 0
        assert "wrote" in capsys.readouterr().out


class TestItDegradesHonestly:
    def test_a_missing_langgraph_names_the_install(self, tmp_path, monkeypatch, capsys):
        """An ImportError from four frames down is not an error message."""
        import ledgerloop.cli as cli

        monkeypatch.setattr(cli, "langgraph_available", lambda: False)
        assert _demo(tmp_path) == 1
        error = capsys.readouterr().err
        assert "[demo]" in error
        assert "ledgerloop eval" in error

    def test_a_stale_bundle_is_refused_rather_than_applied(self, completed, tmp_path):
        """A probability fitted against one generator is not a probability about
        another's data, and a demo must not paper over that."""
        stale = tmp_path / "stale.json"
        payload = json.loads(
            (completed / "calibration.json").read_text(encoding="utf-8")
        )
        payload["provenance"]["generator_version"] = "9.9.9"
        stale.write_text(json.dumps(payload), encoding="utf-8")

        code = main(
            [
                "demo",
                "--data-dir",
                str(completed / "data"),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--bundle",
                str(stale),
                "--no-ui",
            ]
        )
        assert code == 1

    def test_no_llm_is_accepted_and_reported(self, tmp_path, capsys):
        """`--no-llm` must stay functional on every command that can run."""
        assert _demo(tmp_path, "--no-llm") == 0
        assert "--no-llm" in capsys.readouterr().out


class TestItChainsRatherThanReimplements:
    def test_the_demo_agrees_with_a_plain_run_of_the_same_corpus(self, completed):
        """The chaining must not change a number. If it does, the demo is
        showing a reviewer something the CLI does not produce."""
        from ledgerloop.eval.harness import load_bundle_for, run_system
        from ledgerloop.eval.truth_io import load_manifest

        corpus = completed / "data" / "test-standard-42"
        bundle = load_bundle_for(completed / "calibration.json", load_manifest(corpus))
        direct = run_system(corpus, bundle=bundle, measure_calibration_quality=True)

        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.metrics["auto_match_precision"] == direct.metrics.auto_match_precision
        assert stored.metrics["match_rate"] == direct.metrics.match_rate
        assert stored.metrics["exception_recall"] == direct.metrics.exception_recall
        assert stored.metrics["true_positives"] == direct.metrics.link_metrics.true_positives

    def test_the_seed_lists_are_declared_once(self):
        """The Makefile and the demo command fit on the same corpora. Declared
        as data in one place so the two cannot drift into fitting on different
        halves and reporting the same threshold."""
        assert DEMO_TRAIN_SEEDS == (42, 43, 44, 45, 46)
        assert DEMO_CALIBRATION_SEEDS == (47, 48, 49, 50)
        assert not set(DEMO_TRAIN_SEEDS) & set(DEMO_CALIBRATION_SEEDS)


class TestItOpensOnTheCorpusTheDocumentsQuote:
    """Phase 2.1, audit item 4. The most avoidable objection there is.

    Before this the demo opened on `dev`: 60 orders, recall 0.32, and an
    exception recall of 100% resting on **five records**. A reviewer would have
    seen the project's least meaningful numbers on screen and been unable to
    find any of them in README.md or EVALUATION.md, both of which are measured
    on `test`. Nothing was wrong with either document; the demo simply pointed
    somewhere else.
    """

    def test_the_default_split_is_test(self):
        """One flag, and it is the difference between a demo that corroborates
        the documents and one that quietly contradicts them."""
        namespace = _build_parser().parse_args(["demo"])
        assert namespace.split is SplitName.TEST

    def test_dev_is_still_reachable_for_a_faster_run(self):
        namespace = _build_parser().parse_args(["demo", "--split", "dev"])
        assert namespace.split is SplitName.DEV

    def test_the_run_it_stores_is_the_corpus_every_published_number_uses(
        self, completed
    ):
        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.summary["dataset"]["split"] == "test"
        assert stored.summary["dataset"]["seed"] == 42

    def test_its_numbers_are_the_single_seed_ones_the_report_publishes(
        self, completed
    ):
        """Not the multi-seed means -- those are in EVALUATION.md and are what a
        claim should quote -- but the seed-42 figures the report's single-seed
        tables carry, so a reviewer can find every number on screen in the
        document.
        """
        stored = load_run(completed / "runs" / "t0t4-test-42")
        assert stored is not None
        assert stored.metrics["auto_match_precision"] == 1.0
        assert stored.metrics["true_positives"] == 248
        assert stored.metrics["false_positives"] == 0
        assert stored.metrics["false_negatives"] == 46
        assert stored.metrics["match_rate"] == pytest.approx(0.7971, abs=5e-5)
