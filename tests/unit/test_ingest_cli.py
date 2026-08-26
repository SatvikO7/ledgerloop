"""The ``ledgerloop ingest`` command.

A step whose output nobody can look at is a step nobody can check. This command
is how the Step 3 acceptance criteria are verified by hand -- and how ``make
ingest`` fails the build if the committed fixture ever starts losing rows.

What it prints is chosen to be the things that could silently go wrong: how the
date convention was decided, how many references normalisation recovered, and
how much the narration parser reached that a bare UTR regex would not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerloop.cli import main

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


@pytest.fixture
def run(capsys):
    def _run(*argv: str) -> tuple[int, str, str]:
        code = main(["ingest", *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


class TestTheHappyPath:
    def test_it_succeeds_on_the_committed_fixture(self, run):
        code, out, err = run("--data", str(FIXTURE))
        assert code == 0
        assert err == ""
        assert "0 malformed records" in out

    def test_it_reports_the_record_counts(self, run):
        _, out, _ = run("--data", str(FIXTURE))
        assert "60 orders" in out
        assert "5 settlements" in out
        assert "23 bank rows" in out

    def test_it_says_how_the_date_convention_was_decided(self, run):
        """Not just which convention -- on what basis. A claim needs its evidence."""
        _, out, _ = run("--data", str(FIXTURE))
        assert "DAY_FIRST proven by" in out

    def test_it_reports_what_normalisation_recovered(self, run):
        _, out, _ = run("--data", str(FIXTURE))
        assert "recovered by normalisation" in out
        assert "absent at source" in out

    def test_it_reports_the_narration_parser_reach(self, run):
        _, out, _ = run("--data", str(FIXTURE))
        assert "credits carry a" in out

    def test_strict_mode_also_succeeds_on_the_fixture(self, run):
        """What ``make ingest`` runs. A fixture that starts quarantining fails here."""
        code, out, _ = run("--data", str(FIXTURE), "--strict")
        assert code == 0
        assert "0 malformed records" in out


class TestFailures:
    def test_a_missing_directory_fails_loudly(self, run, tmp_path):
        code, _, err = run("--data", str(tmp_path / "nope"))
        assert code == 1
        assert "no such dataset directory" in err

    def test_an_incomplete_dataset_names_the_missing_file(self, run, tmp_path):
        code, _, err = run("--data", str(tmp_path))
        assert code == 1
        assert "ledger_orders.csv" in err

    def test_strict_mode_turns_a_quarantined_row_into_a_failure(self, run, tmp_path):
        _corrupt_fixture_into(tmp_path)
        code, _, err = run("--data", str(tmp_path), "--strict")
        assert code == 1
        assert "ingest failed" in err

    def test_lenient_mode_lists_the_quarantined_rows(self, run, tmp_path):
        _corrupt_fixture_into(tmp_path)
        code, out, err = run("--data", str(tmp_path))
        assert code == 0
        assert "59 orders" in out
        assert "1 malformed records quarantined" in err
        assert "amount_gross_paise" in err

    def test_the_problem_list_is_truncated_on_request(self, run, tmp_path):
        _corrupt_fixture_into(tmp_path, rows=3)
        _, _, err = run("--data", str(tmp_path), "--show-problems", "1")
        assert "and 2 more" in err

    def test_zero_shown_problems_still_reports_the_count(self, run, tmp_path):
        _corrupt_fixture_into(tmp_path, rows=2)
        _, _, err = run("--data", str(tmp_path), "--show-problems", "0")
        assert "2 malformed records quarantined" in err
        assert "and 2 more" in err


def _corrupt_fixture_into(directory: Path, *, rows: int = 1) -> None:
    """Copy the fixture and break the amount column on the first ``rows`` orders."""
    for source in FIXTURE.iterdir():
        (directory / source.name).write_bytes(source.read_bytes())

    path = directory / "ledger_orders.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(1, rows + 1):
        parts = lines[index].split(",")
        parts[3] = "not-a-number"
        lines[index] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
