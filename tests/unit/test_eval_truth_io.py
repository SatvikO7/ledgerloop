"""Round-tripping ground truth through the emitted files.

The evaluator scores against truth read off disk rather than against the
generator's return value, so that ``EVALUATION.md`` and the committed CSVs
cannot disagree. That guarantee is only worth having if the reader is exactly
the inverse of the writer, which is what these tests pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.eval.truth_io import load_ground_truth, load_manifest
from ledgerloop.generator import generate_to_disk
from ledgerloop.models.enums import Difficulty, SplitName

#: Anchored on this file rather than the working directory, so the suite passes
#: from anywhere.
FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


def _config(**overrides) -> GeneratorConfig:
    kwargs = {"split": SplitName.DEV, "seed": 42, "ensure_class_coverage": True}
    kwargs.update(overrides)
    return GeneratorConfig(**kwargs)


@pytest.fixture
def written(tmp_path):
    directory = tmp_path / "dataset"
    dataset = generate_to_disk(_config(), directory)
    return directory, dataset


class TestRoundTrip:
    def test_evaluation_pairs_survive_the_write_and_read(self, written):
        """The atomic unit of evaluation must come back exactly."""
        directory, dataset = written
        assert load_ground_truth(directory).evaluation_pairs == dataset.truth.evaluation_pairs

    def test_every_link_survives(self, written):
        directory, dataset = written
        loaded = load_ground_truth(directory)
        assert loaded.links == dataset.truth.links

    def test_every_record_verdict_survives(self, written):
        directory, dataset = written
        loaded = load_ground_truth(directory)
        assert loaded.records == dataset.truth.records

    def test_the_denominators_survive(self, written):
        directory, dataset = written
        loaded = load_ground_truth(directory)
        assert loaded.reconcilable_refs == dataset.truth.reconcilable_refs
        assert loaded.unmatchable_refs == dataset.truth.unmatchable_refs

    def test_identity_and_draws_survive(self, written):
        """A metric is only comparable to one from the same generator version."""
        directory, dataset = written
        loaded = load_ground_truth(directory)
        assert loaded.split is dataset.truth.split
        assert loaded.difficulty is dataset.truth.difficulty
        assert loaded.seed == dataset.truth.seed
        assert loaded.generator_version == dataset.truth.generator_version
        assert loaded.scenario_draws == dataset.truth.scenario_draws

    def test_an_absent_note_reads_back_as_none_not_empty_string(self, written):
        """The emitter writes ``note or ""``. Reading that back as ``""`` would
        make the round-trip inexact and every equality test above misleading."""
        directory, _ = written
        loaded = load_ground_truth(directory)
        notes = {record.note for record in loaded.records}
        assert None in notes
        assert "" not in notes

    def test_impact_totals_survive(self, written):
        directory, dataset = written
        loaded = load_ground_truth(directory)
        assert loaded.impact_total_minor() == dataset.truth.impact_total_minor()


class TestManifest:
    def test_reports_the_dataset_identity(self, written):
        directory, dataset = written
        manifest = load_manifest(directory)
        assert manifest.split is SplitName.DEV
        assert manifest.difficulty is Difficulty.STANDARD
        assert manifest.seed == 42
        assert manifest.generator_version == dataset.config.generator_version

    def test_counts_agree_with_the_truth_that_was_written(self, written):
        directory, dataset = written
        manifest = load_manifest(directory)
        assert manifest.counts["evaluation_pairs"] == len(dataset.truth.evaluation_pairs)
        assert manifest.counts["unmatchable_records"] == len(dataset.truth.unmatchable_refs)

    def test_money_figures_are_integers_in_minor_units(self, written):
        directory, _ = written
        money = load_manifest(directory).money
        assert all(isinstance(value, int) for value in money.values())


class TestMalformedInput:
    def test_a_manifest_that_is_not_an_object_is_rejected(self, tmp_path):
        """Failing loudly here matters more than usual: a silently mis-read
        manifest would put the wrong split and seed on a real report."""
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_manifest(directory)


class TestCommittedFixture:
    """The fixture is committed, so the reader is pinned against real bytes
    rather than only against bytes it just produced itself."""

    def test_the_committed_fixture_loads(self):
        truth = load_ground_truth(FIXTURE)
        assert len(truth.evaluation_pairs) == 59
        assert len(truth.unmatchable_refs) == 13

    def test_the_committed_fixture_manifest_agrees_with_its_truth(self):
        manifest = load_manifest(FIXTURE)
        truth = load_ground_truth(FIXTURE)
        assert manifest.counts["evaluation_pairs"] == len(truth.evaluation_pairs)
        assert manifest.counts["truth_links"] == len(truth.links)
