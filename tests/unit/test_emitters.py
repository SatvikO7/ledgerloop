"""Emitter tests: byte-identical regeneration and source-format fidelity.

PLAN.md Phase 1 acceptance is explicit -- generate twice with the same seed and
get the same bytes. That is what lets a judge regenerate the corpus and rerun
every number in the report.
"""

from __future__ import annotations

import csv
import itertools
import json

import pytest

from ledgerloop.cli import main
from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.generator.emitters import (
    BANK_FILE,
    GROUND_TRUTH_LINKS_FILE,
    GROUND_TRUTH_RECORDS_FILE,
    LEDGER_FILE,
    MANIFEST_FILE,
    PSP_FILE,
)
from ledgerloop.models.enums import SplitName

ALL_FILES = (
    LEDGER_FILE,
    PSP_FILE,
    BANK_FILE,
    GROUND_TRUTH_LINKS_FILE,
    GROUND_TRUTH_RECORDS_FILE,
    MANIFEST_FILE,
)


def _config(**overrides) -> GeneratorConfig:
    kwargs = {"split": SplitName.DEV, "seed": 42, "ensure_class_coverage": True}
    kwargs.update(overrides)
    return GeneratorConfig(**kwargs)


@pytest.fixture
def written(tmp_path):
    directory = tmp_path / "dataset"
    dataset = generate_to_disk(_config(), directory)
    return directory, dataset


class TestByteIdenticalRegeneration:
    def test_two_runs_produce_identical_bytes(self, tmp_path):
        """The Phase 1 acceptance criterion, stated literally."""
        first, second = tmp_path / "a", tmp_path / "b"
        generate_to_disk(_config(), first)
        generate_to_disk(_config(), second)
        for name in ALL_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_a_different_seed_changes_the_bytes(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        generate_to_disk(_config(seed=1), first)
        generate_to_disk(_config(seed=2), second)
        assert (first / LEDGER_FILE).read_bytes() != (second / LEDGER_FILE).read_bytes()

    def test_files_use_lf_endings_regardless_of_platform(self, written):
        """Windows CRLF would make byte-identity platform-dependent."""
        directory, _ = written
        for name in (LEDGER_FILE, BANK_FILE, GROUND_TRUTH_LINKS_FILE):
            assert b"\r\n" not in (directory / name).read_bytes(), name

    def test_all_expected_files_are_written(self, written):
        directory, _ = written
        for name in ALL_FILES:
            assert (directory / name).is_file(), name


class TestLedgerSource:
    def test_header_and_row_count(self, written):
        directory, dataset = written
        with (directory / LEDGER_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(dataset.world.orders)
        assert set(rows[0]) == {
            "order_id",
            "merchant_id",
            "customer_ref",
            "amount_gross_paise",
            "currency",
            "booked_at",
            "status",
        }

    def test_amounts_are_integer_paise_text(self, written):
        directory, _ = written
        with (directory / LEDGER_FILE).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                assert "." not in row["amount_gross_paise"]
                assert int(row["amount_gross_paise"]) > 0

    def test_currency_is_inr_only(self, written):
        """A11 FX is cut; a USD row would be incoherent in a paise column."""
        directory, _ = written
        with (directory / LEDGER_FILE).open(encoding="utf-8") as handle:
            assert {row["currency"] for row in csv.DictReader(handle)} == {"INR"}


class TestPspSource:
    def test_is_nested_json_with_payments_inside_batches(self, written):
        directory, dataset = written
        payload = json.loads((directory / PSP_FILE).read_text(encoding="utf-8"))
        assert len(payload["settlements"]) == len(dataset.world.settlements)
        assert all("payments" in batch for batch in payload["settlements"])
        assert sum(len(b["payments"]) for b in payload["settlements"]) == len(
            dataset.world.payments
        )

    def test_order_refs_include_nulls_and_manglings(self, written):
        """PLAN.md §5.1: sometimes null, sometimes malformed. On purpose."""
        directory, _ = written
        payload = json.loads((directory / PSP_FILE).read_text(encoding="utf-8"))
        refs = [p["order_ref"] for b in payload["settlements"] for p in b["payments"]]
        assert any(ref is None for ref in refs)
        assert any(ref is not None and not ref.startswith("ORD-") for ref in refs)

    def test_declared_net_is_published_even_when_inconsistent(self, written):
        """A03 must survive to the file, or the anomaly does not exist."""
        directory, _ = written
        payload = json.loads((directory / PSP_FILE).read_text(encoding="utf-8"))
        inconsistent = [
            b
            for b in payload["settlements"]
            if b["net_paise"]
            != b["gross_paise"] - b["fee_paise"] - b["tax_paise"] + b["adjustments_paise"]
        ]
        assert inconsistent, "the fixture set must contain a fee/tax mismatch"

    def test_negative_adjustments_appear(self, written):
        """Chargebacks and refunds net off here."""
        directory, _ = written
        payload = json.loads((directory / PSP_FILE).read_text(encoding="utf-8"))
        assert any(b["adjustments_paise"] < 0 for b in payload["settlements"])


class TestBankSource:
    def test_dates_use_the_ambiguous_day_first_format(self, written):
        """DD/MM/YYYY, deliberately -- ingest has to resolve it in Step 3."""
        directory, _ = written
        with (directory / BANK_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            day, month, year = row["value_date"].split("/")
            assert len(day) == 2 and len(month) == 2 and len(year) == 4
            assert 1 <= int(month) <= 12

    def test_rows_are_ordered_by_value_date(self, written):
        """A statement arrives in date order, not generation order."""
        directory, _ = written
        with (directory / BANK_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = [
            (row["value_date"].split("/")[::-1], row["txn_id"]) for row in rows
        ]
        assert keys == sorted(keys)

    def test_running_balance_is_consistent(self, written):
        directory, _ = written
        with (directory / BANK_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for previous, current in itertools.pairwise(rows):
            expected = (
                int(previous["balance_paise"])
                + int(current["credit_paise"])
                - int(current["debit_paise"])
            )
            assert int(current["balance_paise"]) == expected

    def test_narrations_vary_and_include_unreferenced_rows(self, written):
        directory, _ = written
        with (directory / BANK_FILE).open(encoding="utf-8") as handle:
            narrations = [row["narration"] for row in csv.DictReader(handle)]
        assert len(set(narrations)) > 3
        assert any("UTR" not in n for n in narrations)

    def test_noise_rows_are_present(self, written):
        directory, _ = written
        with (directory / BANK_FILE).open(encoding="utf-8") as handle:
            narrations = [row["narration"] for row in csv.DictReader(handle)]
        assert any("SALARY" in n or "RENT" in n or "GST" in n for n in narrations)


class TestGroundTruthFiles:
    def test_links_file_matches_the_truth_object(self, written):
        directory, dataset = written
        with (directory / GROUND_TRUTH_LINKS_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(dataset.truth.links)

    def test_records_file_matches_the_truth_object(self, written):
        directory, dataset = written
        with (directory / GROUND_TRUTH_RECORDS_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(dataset.truth.records)

    def test_records_carry_status_class_and_impact(self, written):
        directory, _ = written
        with (directory / GROUND_TRUTH_RECORDS_FILE).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert any(row["expected_status"] == "UNMATCHABLE" for row in rows)
        assert any(row["expected_status"] == "EXCEPTION" for row in rows)
        assert any(int(row["impact_paise"]) > 0 for row in rows)
        assert any(row["note"] for row in rows)


class TestManifest:
    def test_records_provenance_and_conservation(self, written):
        directory, dataset = written
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        assert manifest["seed"] == dataset.config.seed
        assert manifest["split"] == dataset.config.split.value
        assert manifest["generator_version"] == dataset.config.generator_version
        money = manifest["money"]
        residual = (
            money["settled_credit_total_paise"]
            - money["declared_net_total_paise"]
            - money["declared_bank_delta_paise"]
        )
        assert residual == 0

    def test_contains_no_timestamp_or_absolute_path(self, written):
        """Either would break byte-identity across machines and runs."""
        directory, _ = written
        text = (directory / MANIFEST_FILE).read_text(encoding="utf-8")
        assert "C:\\" not in text
        assert "/home/" not in text
        assert "generated_at" not in text


class TestCli:
    def test_generate_writes_a_dataset_and_reports_success(self, tmp_path, capsys):
        exit_code = main(
            [
                "generate",
                "--split",
                "dev",
                "--seed",
                "42",
                "--out",
                str(tmp_path / "cli"),
                "--ensure-class-coverage",
            ]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "money conserved" in out
        for name in ALL_FILES:
            assert (tmp_path / "cli" / name).is_file()

    def test_order_count_override(self, tmp_path):
        assert main(["generate", "--orders", "45", "--out", str(tmp_path / "small")]) == 0
        with (tmp_path / "small" / LEDGER_FILE).open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 45

    def test_rupee_symbol_does_not_crash_on_a_windows_codepage(self, tmp_path, capsys):
        """cp1252 has no ₹; printing one unguarded takes the whole run down."""
        assert main(["generate", "--orders", "40", "--out", str(tmp_path / "utf8")]) == 0
        assert "₹" in capsys.readouterr().out
