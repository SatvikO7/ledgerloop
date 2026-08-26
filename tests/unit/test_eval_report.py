"""``EVALUATION.md`` rendering, and the `ledgerloop eval` command.

PLAN.md §9.4 requires the whole report to come from one command with nothing
hand-typed. Two properties carry that requirement:

* **Determinism.** No timestamp, no path, no dict-order dependence -- so a diff
  between two runs shows a change in the system rather than in the clock.
* **Pending is not zero.** Metrics no implemented step produces yet must render
  as pending, never as ``0.00%``. A zero for an unbuilt component is a false
  measurement, and it is the easiest one to publish by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.cli import main
from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.baselines import run_b0
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.report import EvaluatedRun, render_report, write_report
from ledgerloop.eval.truth_io import load_ground_truth, load_manifest
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import AnomalyClass, SplitName

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


@pytest.fixture(scope="module")
def rendered():
    truth = load_ground_truth(FIXTURE)
    manifest = load_manifest(FIXTURE)
    baseline = run_b0(FIXTURE)
    metrics = evaluate(
        baseline.predictions,
        truth,
        run_id="b0-dev-42",
        wall_clock_ms=baseline.wall_clock_ms,
    )
    run = EvaluatedRun(system=baseline, metrics=metrics)
    return render_report([run], manifest=manifest, truth=truth), metrics


def _without_timings(text: str) -> str:
    """Drop the two measured rows -- see the module docstring."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("| Wall clock |", "| Throughput |"))
    )


class TestDeterminism:
    def test_two_renders_of_the_same_input_agree_apart_from_the_timings(self, rendered):
        text, _ = rendered
        truth = load_ground_truth(FIXTURE)
        manifest = load_manifest(FIXTURE)
        baseline = run_b0(FIXTURE)
        again = render_report(
            [
                EvaluatedRun(
                    system=baseline,
                    metrics=evaluate(baseline.predictions, truth, run_id="b0-dev-42"),
                )
            ],
            manifest=manifest,
            truth=truth,
        )
        assert _without_timings(again) == _without_timings(text)

    def test_the_varying_rows_are_confined_to_the_labelled_block(self, rendered):
        """A reader must be able to tell a real regression from scheduler noise."""
        text, _ = rendered
        assert "#### Measured timings" in text
        assert text.count("| Wall clock |") == 1
        assert text.count("| Throughput |") == 1
        timings_at = text.index("#### Measured timings")
        assert text.index("| Wall clock |") > timings_at

    def test_no_timestamp_leaks_into_the_document(self, rendered):
        """A date in the report makes every rerun a diff."""
        text, _ = rendered
        assert "202" not in text.replace("UTR202", "").replace("`0.2.0`", "")

    def test_it_ends_with_exactly_one_newline(self, rendered):
        text, _ = rendered
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


class TestPendingIsNotZero:
    def test_exception_recall_is_pending_before_step_8(self, rendered):
        text, _ = rendered
        assert "no exception classifier before Step 8" in text

    def test_calibration_is_pending_before_step_7(self, rendered):
        text, _ = rendered
        assert "no blender before Step 7" in text

    def test_unbuilt_baselines_are_listed_as_pending(self, rendered):
        text, _ = rendered
        for name in ("B1", "B2", "B3"):
            assert f"| {name} |" in text
        assert "_pending_ (Step 6)" in text

    def test_a_zero_denominator_renders_as_na_not_as_zero(self):
        """The single most common way an evaluation report misleads."""
        truth = load_ground_truth(FIXTURE)
        manifest = load_manifest(FIXTURE)
        baseline = run_b0(FIXTURE)
        empty = baseline.__class__(
            **{**baseline.__dict__, "predictions": ()}  # type: ignore[arg-type]
        )
        text = render_report(
            [
                EvaluatedRun(
                    system=empty,
                    metrics=evaluate((), truth, run_id="empty"),
                )
            ],
            manifest=manifest,
            truth=truth,
        )
        assert "| Auto-match precision | n/a |" in text


class TestContent:
    def test_it_names_the_dataset_identity(self, rendered):
        """A metric is only comparable to one from the same generator version."""
        text, _ = rendered
        assert "| Generator version | `0.2.0` |" in text
        assert "| Split | `dev` |" in text
        assert "| Seed | `42` |" in text

    def test_the_headline_three_are_present_with_their_targets(self, rendered):
        text, _ = rendered
        assert "Auto-match precision" in text
        assert "≥ 99.00%" in text
        assert "Match rate" in text
        assert "≥ 85.00%" in text

    def test_the_false_positive_cost_is_a_rupee_figure(self, rendered):
        text, _ = rendered
        assert "False-positive cost" in text
        assert "₹" in text

    def test_every_class_present_in_the_data_gets_a_row(self, rendered):
        text, metrics = rendered
        for anomaly in metrics.recall_by_anomaly_class:
            assert f"`{anomaly.value}`" in text

    def test_classes_scoring_zero_are_published(self, rendered):
        """PLAN.md §9.1: the table exists to show the bad rows."""
        text, metrics = rendered
        assert metrics.recall_by_anomaly_class[AnomalyClass.MISSING_REFERENCE] == 0.0
        assert "| `A07_MISSING_REFERENCE` |" in text
        assert "0.00%" in text

    def test_the_unmatchable_ceiling_is_reported(self, rendered):
        text, _ = rendered
        assert "Unmatchable records (the honest ceiling)" in text

    def test_the_wilson_choice_is_explained_in_the_document(self, rendered):
        """A reader must be able to see why the interval is what it is without
        reading the source."""
        text, _ = rendered
        assert "Wilson" in text

    def test_it_refuses_to_render_nothing(self):
        with pytest.raises(ValueError, match="at least one evaluated run"):
            render_report(
                [], manifest=load_manifest(FIXTURE), truth=load_ground_truth(FIXTURE)
            )


class TestWriteReport:
    def test_it_writes_lf_endings_on_every_platform(self, tmp_path, rendered):
        text, _ = rendered
        path = tmp_path / "nested" / "EVALUATION.md"
        write_report(path, text)
        assert b"\r\n" not in path.read_bytes()

    def test_it_creates_missing_parent_directories(self, tmp_path, rendered):
        text, _ = rendered
        path = tmp_path / "a" / "b" / "EVALUATION.md"
        write_report(path, text)
        assert path.is_file()


class TestEvalCommand:
    def test_it_writes_a_report_and_succeeds(self, tmp_path, capsys):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        out = tmp_path / "EVALUATION.md"

        code = main(["eval", "--data", str(data), "--out", str(out)])

        assert code == 0
        assert out.is_file()
        assert "EVALUATION.md" in out.read_text(encoding="utf-8")
        assert "B0" in capsys.readouterr().out

    def test_it_reports_the_interval_alongside_the_point_estimate(self, tmp_path, capsys):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        main(["eval", "--data", str(data), "--out", str(tmp_path / "out.md")])
        printed = capsys.readouterr().out
        assert "precision" in printed
        assert "[" in printed and "]" in printed

    def test_a_missing_directory_fails_loudly(self, tmp_path, capsys):
        code = main(["eval", "--data", str(tmp_path / "nope"), "--out", str(tmp_path / "o.md")])
        assert code == 1
        assert "no such dataset directory" in capsys.readouterr().err

    def test_the_report_defaults_to_evaluation_md(self, tmp_path, monkeypatch):
        """`make eval` relies on the default, so the default is tested."""
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        monkeypatch.chdir(tmp_path)
        assert main(["eval", "--data", str(data)]) == 0
        assert (tmp_path / "EVALUATION.md").is_file()


class TestTheSystemAndBaselineTogether:
    """`ledgerloop eval` scores the matcher and the floor it has to beat.

    Both go through the same renderer, so the comparison cannot be flattered by
    rendering one more generously than the other.
    """

    def test_the_command_reports_both_systems(self, tmp_path, capsys):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        code = main(["eval", "--data", str(data), "--out", str(tmp_path / "out.md")])
        printed = capsys.readouterr().out
        assert code == 0
        assert "T0+T1:" in printed
        assert "B0:" in printed

    def test_it_reports_decisions_and_settlement_dispositions(self, tmp_path, capsys):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        main(["eval", "--data", str(data), "--out", str(tmp_path / "out.md")])
        printed = capsys.readouterr().out
        assert "auto-matched" in printed
        assert "needs review" in printed
        assert "contested" in printed

    def test_the_report_carries_a_tier_contribution_table(self, tmp_path):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        out = tmp_path / "out.md"
        main(["eval", "--data", str(data), "--out", str(out)])
        text = out.read_text(encoding="utf-8")
        assert "### Tier contribution" in text
        assert "`T0_EXACT`" in text
        assert "`T1_TOLERANCE`" in text
        assert "Candidates proposed" in text

    def test_the_baseline_gets_no_tier_table(self, tmp_path):
        """A baseline has no tiers, and an empty table of zeros would be a lie."""
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        out = tmp_path / "out.md"
        main(["eval", "--data", str(data), "--out", str(out)])
        text = out.read_text(encoding="utf-8")
        assert text.count("### Tier contribution") == 1

    def test_quarantined_source_records_are_surfaced(self, tmp_path, capsys):
        data = tmp_path / "data"
        generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), data)
        ledger = data / "ledger_orders.csv"
        lines = ledger.read_text(encoding="utf-8").splitlines()
        parts = lines[1].split(",")
        parts[3] = "not-a-number"
        lines[1] = ",".join(parts)
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

        code = main(["eval", "--data", str(data), "--out", str(tmp_path / "out.md")])
        assert code == 0
        assert "malformed source records quarantined" in capsys.readouterr().err
