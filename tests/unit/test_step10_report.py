"""Step 10's report sections, and the commands that produce their artefacts.

Two rules are checked over and over here because they are the ones an
evaluation report breaks first:

* **An absent measurement is never a zero.** No sweep artefact means no
  multi-seed section, not a section of zeros; a B2 that could not reach a model
  is `_pending_`, not a precision of 0.00%.
* **No metric is prose.** Every number in `EVALUATION.md` is rendered from a
  typed artefact by deterministic code. The test for that is that the document
  regenerates identically, timings aside.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ledgerloop.cli import main
from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.ablation import AblationArtifact
from ledgerloop.eval.llm_baseline import LLMBaselineArtifact
from ledgerloop.eval.sweep import SweepArtifact
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import Difficulty, SplitName

#: The measured-timing lines, all of which live under `#### Measured timings`.
#: Everything else in the document is deterministic, and the two tests in
#: `TestDeterminism` are what hold that claim honest.
TIMING = re.compile(r"^\| (Wall clock|Throughput|T\d_[A-Z]+) \|")


def _stable(text: str) -> list[str]:
    """The document minus the two lines that legitimately vary between runs."""
    return [line for line in text.splitlines() if not TIMING.match(line)]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """One `dev` corpus at three difficulties and two seeds, plus the artefacts.

    Built once: this fixture runs the ablation (six ladders x two seeds), the
    sweep and B2, and the whole point of the artefact design is that a report
    can be rendered many times from them without repeating any of that.
    """
    root = tmp_path_factory.mktemp("step10")
    seeds = []
    for seed in (42, 43):
        directory = root / f"dev-standard-{seed}"
        generate_to_disk(
            GeneratorConfig(split=SplitName.DEV, seed=seed), directory
        )
        seeds.append(directory)
    hard = root / "dev-hard-42"
    generate_to_disk(
        GeneratorConfig(split=SplitName.DEV, difficulty=Difficulty.HARD, seed=42), hard
    )

    ablation = root / "ablation.json"
    sweep = root / "sweep.json"
    baseline = root / "b2.json"
    assert main(["ablation", "--data", *map(str, seeds), "--out", str(ablation)]) == 0
    assert main(["sweep", "--data", *map(str, [*seeds, hard]), "--out", str(sweep)]) == 0
    assert (
        main(
            [
                "baseline-llm",
                "--data",
                str(seeds[0]),
                "--cache-dir",
                str(root / "b2cache"),
                "--cold",
                "--offline-provider",
                "--out",
                str(baseline),
            ]
        )
        == 0
    )
    return root, seeds[0], ablation, sweep, baseline


def _render(workspace, tmp_path, *extra: str) -> str:
    _, data, _, _, _ = workspace
    out = tmp_path / "EVALUATION.md"
    assert main(["eval", "--data", str(data), "--out", str(out), *extra]) == 0
    return out.read_text(encoding="utf-8")


class TestTheBaselineComparison:
    def test_all_four_systems_appear(self, workspace, tmp_path):
        _, _, _, _, baseline = workspace
        text = _render(workspace, tmp_path, "--llm-baseline", str(baseline))
        for name in ("| B0 |", "| B1 |", "| B2", "| B3 ("):
            assert name in text

    def test_every_required_column_is_present(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        header = next(
            line for line in text.splitlines() if line.startswith("| # | Split |")
        )
        for column in (
            "Candidate yield",
            "Auto-matched",
            "Precision",
            "Recall",
            "Match rate",
            "FP",
            "FP cost",
            "Exception recall",
        ):
            assert column in header

    def test_b2_is_pending_when_its_artefact_is_absent(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "| B2 | `dev` | _pending_ |" in text

    def test_b2_carries_its_scope_in_the_row(self, workspace, tmp_path):
        """A comparison table whose rows come from different corpora has to say
        so in the row, not in a footnote nobody reads."""
        _, _, _, _, baseline = workspace
        text = _render(workspace, tmp_path, "--llm-baseline", str(baseline))
        assert "| B2 §" in text
        assert "`dev` ‡" in text

    def test_a_stand_in_row_is_labelled_as_one(self, workspace, tmp_path):
        _, _, _, _, baseline = workspace
        text = _render(workspace, tmp_path, "--llm-baseline", str(baseline))
        assert "**B2 was not answered by a language model.**" in text
        assert "not a claim about any model" in text

    def test_baselines_are_marked_as_having_no_proposal_stage(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "†" in text
        assert "no proposal stage separate from its output" in text


class TestTheAblationSection:
    def test_it_is_absent_without_its_artefact(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "## Ablation" not in text

    def test_it_renders_every_ladder(self, workspace, tmp_path):
        _, _, ablation, _, _ = workspace
        text = _render(workspace, tmp_path, "--ablation", str(ablation))
        for label in ("`T0`", "`T0-T1`", "`T0-T2`", "`T0-T3`", "`T0-T4`", "`T0-T5`"):
            assert f"| {label} |" in text

    def test_it_states_that_the_rows_were_re_run_not_subtracted(
        self, workspace, tmp_path
    ):
        _, _, ablation, _, _ = workspace
        text = _render(workspace, tmp_path, "--ablation", str(ablation))
        assert "**re-run, not subtracted**" in text

    def test_it_publishes_the_tuning_hash_that_holds_the_rows_together(
        self, workspace, tmp_path
    ):
        _, _, ablation, _, _ = workspace
        artifact = AblationArtifact.load(ablation)
        text = _render(workspace, tmp_path, "--ablation", str(ablation))
        assert f"`{artifact.tuning_hash}`" in text

    def test_a_model_less_t0_t5_row_says_so(self, workspace, tmp_path):
        """Its LLM columns are zero because no call was made, not because the
        tier was measured and found to contribute nothing."""
        _, _, ablation, _, _ = workspace
        text = _render(workspace, tmp_path, "--ablation", str(ablation))
        assert "No model was reachable when this table was produced" in text


class TestTheSweepSections:
    def test_they_are_absent_without_the_artefact(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "## Multi-seed evaluation" not in text
        assert "## Difficulty response" not in text

    def test_the_multi_seed_table_reports_mean_and_spread(self, workspace, tmp_path):
        _, _, _, sweep, _ = workspace
        text = _render(workspace, tmp_path, "--sweep", str(sweep))
        assert "## Multi-seed evaluation" in text
        assert "| Metric | Mean ± std | Min | Max |" in text
        assert "±" in text

    def test_the_single_seed_sections_say_they_are_single_seed(
        self, workspace, tmp_path
    ):
        """The multi-seed section points at them as "labelled as single-seed
        where they appear". That cross-reference has to be true."""
        text = _render(workspace, tmp_path)
        assert text.count("**Single seed.**") >= 3

    def test_it_names_the_sample_deviation_it_uses(self, workspace, tmp_path):
        _, _, _, sweep, _ = workspace
        text = _render(workspace, tmp_path, "--sweep", str(sweep))
        assert "`ddof = 1`" in text

    def test_the_difficulty_table_is_a_separate_section(self, workspace, tmp_path):
        _, _, _, sweep, _ = workspace
        text = _render(workspace, tmp_path, "--sweep", str(sweep))
        assert "## Difficulty response" in text
        assert "| Metric | `standard` | `hard` |" in text

    def test_it_states_that_the_threshold_was_never_refitted(self, workspace, tmp_path):
        _, _, _, sweep, _ = workspace
        text = _render(workspace, tmp_path, "--sweep", str(sweep))
        assert "**One threshold, not one per difficulty.**" in text
        assert "was selected against a\n`test` result" in text.replace("\r", "")


class TestTheHonestNegatives:
    def test_the_section_accounts_for_the_missing_recall(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "### Honest negative analysis" in text
        for label in (
            "Settlements left unresolved",
            "Settlements contested",
            "T2 subsets ambiguous",
            "T3 rejected on margin",
            "T4 inferences made",
            "Unmatchable records (the floor)",
        ):
            assert label in text

    def test_the_review_queue_is_counted_apart_from_the_misses(
        self, workspace, tmp_path
    ):
        text = _render(workspace, tmp_path)
        assert "routed to a human rather" in text

    def test_the_per_class_recall_table_includes_the_bad_rows(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "### Recall by anomaly class" in text
        assert "including the ones that score zero" in text

    def test_the_confusion_matrix_is_still_rendered(self, workspace, tmp_path):
        text = _render(workspace, tmp_path)
        assert "#### Anomaly → exception confusion" in text


class TestDeterminism:
    def test_the_document_regenerates_identically(self, workspace, tmp_path):
        """No metric anywhere is prose: the whole file is rendered from typed
        artefacts by deterministic code, so two runs differ only in the two
        measured timing lines."""
        _, _, ablation, sweep, baseline = workspace
        extra = [
            "--ablation",
            str(ablation),
            "--sweep",
            str(sweep),
            "--llm-baseline",
            str(baseline),
        ]
        first = _render(workspace, tmp_path / "a", *extra)
        second = _render(workspace, tmp_path / "b", *extra)
        assert _stable(first) == _stable(second)

    def test_the_only_lines_that_move_are_the_labelled_timings(
        self, workspace, tmp_path
    ):
        first = _render(workspace, tmp_path / "c")
        second = _render(workspace, tmp_path / "d")
        differing = [
            (left, right)
            for left, right in zip(
                first.splitlines(), second.splitlines(), strict=True
            )
            if left != right
        ]
        assert all(TIMING.match(left) for left, _ in differing)


class TestArtefactErrors:
    def test_a_missing_artefact_path_is_an_error_not_a_silent_omission(
        self, workspace, tmp_path
    ):
        """A path that was asked for and cannot be read must not produce a
        report quietly missing the table someone requested."""
        _, data, _, _, _ = workspace
        code = main(
            [
                "eval",
                "--data",
                str(data),
                "--ablation",
                str(tmp_path / "nope.json"),
                "--out",
                str(tmp_path / "out.md"),
            ]
        )
        assert code == 1

    def test_an_unparseable_artefact_is_an_error(self, workspace, tmp_path):
        _, data, _, _, _ = workspace
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        code = main(
            [
                "eval",
                "--data",
                str(data),
                "--sweep",
                str(broken),
                "--out",
                str(tmp_path / "out.md"),
            ]
        )
        assert code == 1


class TestTheCommands:
    def test_ablation_writes_a_loadable_artefact(self, workspace):
        _, _, ablation, _, _ = workspace
        artifact = AblationArtifact.load(ablation)
        assert len(artifact.rows) == 6
        assert artifact.seeds == (42, 43)

    def test_sweep_writes_a_loadable_artefact(self, workspace):
        _, _, _, sweep, _ = workspace
        artifact = SweepArtifact.load(sweep)
        assert {group.difficulty for group in artifact.groups} == {"standard", "hard"}

    def test_baseline_llm_writes_a_loadable_artefact(self, workspace):
        _, _, _, _, baseline = workspace
        artifact = LLMBaselineArtifact.load(baseline)
        assert artifact.ran is True
        assert artifact.is_standin is True
        assert artifact.cost.llm_calls > 0

    def test_a_missing_dataset_is_reported_rather_than_traced(self, tmp_path):
        assert main(["ablation", "--data", str(tmp_path / "nope")]) == 1
        assert main(["sweep", "--data", str(tmp_path / "nope")]) == 1
        assert main(["baseline-llm", "--data", str(tmp_path / "nope")]) == 1

    def test_no_llm_keeps_every_command_working(self, workspace, tmp_path):
        """`--no-llm` is the same code path with one branch taken. It has to
        stay functional on every command Step 10 added."""
        _, data, _, _, _ = workspace
        assert (
            main(["ablation", "--data", str(data), "--no-llm", "--out", str(tmp_path / "a.json")])
            == 0
        )
        assert (
            main(["sweep", "--data", str(data), "--no-llm", "--out", str(tmp_path / "s.json")])
            == 0
        )
        assert (
            main(
                [
                    "baseline-llm",
                    "--data",
                    str(data),
                    "--no-llm",
                    "--cache-dir",
                    str(tmp_path / "c"),
                    "--out",
                    str(tmp_path / "b.json"),
                ]
            )
            == 0
        )
        assert LLMBaselineArtifact.load(tmp_path / "b.json").ran is False

    def test_the_stand_in_is_never_a_silent_fallback(self, workspace, tmp_path):
        """Without the flag and without a key, B2 reports that it did not run.
        A run that quietly substituted a stand-in would publish a row nobody
        could tell apart from a live measurement."""
        _, data, _, _, _ = workspace
        out = tmp_path / "b.json"
        assert (
            main(
                [
                    "baseline-llm",
                    "--data",
                    str(data),
                    "--llm-key-env",
                    "LEDGERLOOP_DEFINITELY_UNSET",
                    "--cache-dir",
                    str(tmp_path / "c2"),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        artifact = LLMBaselineArtifact.load(out)
        assert artifact.ran is False
        assert artifact.is_standin is False


class TestTheEvaluationCommandItself:
    def test_it_writes_where_it_was_told(self, workspace, tmp_path):
        _, data, _, _, _ = workspace
        out = tmp_path / "nested" / "EVALUATION.md"
        assert main(["eval", "--data", str(data), "--out", str(out)]) == 0
        assert out.is_file()

    def test_a_missing_dataset_is_reported(self, tmp_path):
        assert main(["eval", "--data", str(tmp_path / "nope")]) == 1

    def test_a_missing_bundle_is_reported(self, workspace, tmp_path):
        _, data, _, _, _ = workspace
        code = main(
            [
                "eval",
                "--data",
                str(data),
                "--calibration",
                str(tmp_path / "nope.json"),
                "--out",
                str(tmp_path / "o.md"),
            ]
        )
        assert code == 1


def test_the_report_module_no_longer_promises_pending_baselines():
    """B1 and B2 were the pending list at Step 9. Step 10 built both, so the
    list is gone rather than left describing work that is done.

    `_PENDING` itself stays: it is still the right rendering for a B2 whose
    artefact was not supplied. What is gone is the table of *promises* -- rows
    naming a future step that would fill them in.
    """
    import ledgerloop.eval.report as report

    assert not hasattr(report, "PENDING_BASELINES")
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert "_pending_ (Step" not in source
