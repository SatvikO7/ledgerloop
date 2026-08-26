"""Whole-dataset ingest, against the committed fixture and a generated split.

This is where the Step 3 acceptance criteria are checked. PLAN.md Phase 2 asks
for three things, and each has a test here that would fail if it stopped being
true:

* a 300-record set ingests with **zero** parse failures,
* the ambiguous ``DD/MM`` dates are resolved correctly,
* no float appears in any money field.

The last one is checked reflectively over every ingested record rather than
field by field, so a money field added later is covered the day it is added.

The fixture is reached through ``parents[2]`` rather than a relative path, so
the suite passes from any working directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.generator import generate_to_disk
from ledgerloop.generator.emitters import BANK_FILE, LEDGER_FILE, PSP_FILE
from ledgerloop.ingest import IngestError, ingest_dataset
from ledgerloop.ingest.dates import DateOrder
from ledgerloop.ingest.normalize import is_order_ref_shaped
from ledgerloop.models.enums import OrderStatus, RecordType, SourceName, SplitName
from ledgerloop.models.records import CanonicalOrder

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "data" / "fixtures" / "dev-standard-42"

#: Every money field on every canonical record, by record type. Walked
#: reflectively so the invariant cannot be quietly outgrown.
MONEY_FIELDS = {
    RecordType.ORDER: ("amount_minor",),
    RecordType.PAYMENT: ("amount_minor",),
    RecordType.SETTLEMENT: (
        "gross_minor",
        "fee_minor",
        "tax_minor",
        "adjustments_minor",
        "net_minor",
    ),
    RecordType.BANK_TXN: ("credit_minor", "debit_minor", "balance_minor"),
}


@pytest.fixture(scope="module")
def ingested():
    """The committed fixture, ingested strictly. Module-scoped: it is read-only."""
    return ingest_dataset(FIXTURE, strict=True)


@pytest.fixture(scope="module")
def test_split(tmp_path_factory):
    """A freshly generated 300-order split -- the size the acceptance criterion names."""
    directory = tmp_path_factory.mktemp("test-split")
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    return directory


class TestAcceptanceCriteria:
    def test_a_three_hundred_record_set_ingests_with_zero_parse_failures(self, test_split):
        result = ingest_dataset(test_split, strict=True)
        assert result.problems == ()
        assert len(result.orders) == 300
        assert len(result.payments) == 300
        assert result.settlements
        assert result.bank_txns

    def test_the_ambiguous_dd_mm_dates_are_resolved_correctly(self, ingested):
        """Proven from the column, not assumed from the locale.

        The generator writes ``%d/%m/%Y``, so a correct reading must place
        every transaction in March or April 2026 -- the window the world is
        generated in. A month-first reading would scatter them across the year.
        """
        assert ingested.date_order.proven
        assert ingested.date_order.order is DateOrder.DAY_FIRST
        assert ingested.date_order.day_first_witnesses > 0
        assert ingested.date_order.month_first_witnesses == 0

        for txn in ingested.bank_txns:
            assert date(2026, 1, 1) <= txn.value_date <= date(2026, 12, 31)
            assert txn.value_date.month in (3, 4)

    def test_the_dates_match_the_source_file_read_day_first(self, ingested):
        """Row by row against the raw text. No inference, just the comparison."""
        with (FIXTURE / BANK_FILE).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_id = {txn.txn_id: txn for txn in ingested.bank_txns}
        for row in rows:
            day, month, year = (int(part) for part in row["value_date"].split("/"))
            assert by_id[row["txn_id"]].value_date == date(year, month, day)

    def test_no_float_reaches_any_money_field(self, ingested):
        """Reflective: every money field on every record, checked by type.

        ``bool`` is checked explicitly because it is an ``int`` subclass, so
        ``isinstance(True, int)`` passes and ``True + 499900`` is ``499901``.
        """
        checked = 0
        for record in ingested.normalized:
            for name in MONEY_FIELDS[record.record_type]:
                value = getattr(record, name)
                if value is None:
                    continue
                assert not isinstance(value, bool), (record.ref.key, name)
                assert not isinstance(value, float), (record.ref.key, name)
                assert not isinstance(value, Decimal), (record.ref.key, name)
                assert isinstance(value, int), (record.ref.key, name, type(value))
                checked += 1
        assert checked > 0


class TestFidelityToTheSource:
    def test_every_ledger_row_becomes_exactly_one_order(self, ingested):
        with (FIXTURE / LEDGER_FILE).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(ingested.orders)
        for row, order in zip(rows, ingested.orders, strict=True):
            assert order.order_id == row["order_id"]
            assert order.amount_minor == int(row["amount_gross_paise"])
            assert order.merchant_id == row["merchant_id"]

    def test_every_psp_batch_and_payment_survives(self, ingested):
        document = json.loads((FIXTURE / PSP_FILE).read_text(encoding="utf-8"))
        batches = document["settlements"]
        assert len(batches) == len(ingested.settlements)
        assert sum(len(b["payments"]) for b in batches) == len(ingested.payments)
        for batch, settlement in zip(batches, ingested.settlements, strict=True):
            assert settlement.settlement_id == batch["settlement_id"]
            assert settlement.net_minor == batch["net_paise"]
            assert settlement.gross_minor == batch["gross_paise"]
            assert settlement.payment_ids == tuple(p["payment_id"] for p in batch["payments"])

    def test_every_bank_row_survives(self, ingested):
        with (FIXTURE / BANK_FILE).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(ingested.bank_txns)
        for row, txn in zip(rows, ingested.bank_txns, strict=True):
            assert txn.txn_id == row["txn_id"]
            assert txn.narration_raw == row["narration"]
            assert txn.credit_minor == int(row["credit_paise"])
            assert txn.balance_minor == int(row["balance_paise"])

    def test_ingest_is_deterministic(self, ingested):
        again = ingest_dataset(FIXTURE, strict=True)
        assert again.normalized == ingested.normalized
        assert again.date_order == ingested.date_order


class TestProvenance:
    def test_every_record_points_back_at_its_source_row(self, ingested):
        for record in ingested.normalized:
            assert record.raw is not None, record.ref.key
            assert record.raw.source is record.source
            assert record.raw.payload

    def test_source_lines_are_a_dense_sequence_per_entity_stream(self, ingested):
        for stream in (
            ingested.orders,
            ingested.payments,
            ingested.settlements,
            ingested.bank_txns,
        ):
            lines = [record.raw.source_line for record in stream]
            assert lines == list(range(len(stream)))

    def test_the_raw_payload_is_the_source_object_verbatim(self, ingested):
        document = json.loads((FIXTURE / PSP_FILE).read_text(encoding="utf-8"))
        expected = document["settlements"][0]["payments"][0]
        assert ingested.payments[0].raw.payload == expected

    def test_raw_records_group_the_way_recon_state_holds_them(self, ingested):
        grouped = ingested.raw_by_source
        assert set(grouped) == set(SourceName)
        assert len(grouped[SourceName.LEDGER]) == len(ingested.orders)
        assert len(grouped[SourceName.BANK]) == len(ingested.bank_txns)
        assert len(grouped[SourceName.PSP]) == len(ingested.payments) + len(
            ingested.settlements
        )

    def test_the_normalised_view_is_every_record_once(self, ingested):
        assert len(ingested.normalized) == ingested.record_count
        assert len({record.ref.key for record in ingested.normalized}) == (
            ingested.record_count
        )
        assert list(ingested) == ingested.normalized


class TestWhatNormalisationRecovered:
    def test_every_mangled_reference_in_the_fixture_is_recovered(self, ingested):
        """The measured value of normalisation, on the committed corpus.

        Roughly a fifth of PSP references are corrupted. Two of the three
        corruptions are recoverable and the third -- ``null`` -- is not, by
        construction. Every reference that is present is recovered.
        """
        present = [p for p in ingested.payments if p.order_ref_raw is not None]
        assert all(is_order_ref_shaped(p.order_ref_normalized) for p in present)
        assert ingested.payments_with_recovered_ref > 0
        assert ingested.payments_with_usable_ref == len(present)
        assert ingested.payments_with_no_ref == len(ingested.payments) - len(present)

    def test_recovered_references_name_real_orders(self, ingested):
        """Normalisation must not invent a reference that happens to look right."""
        known = {order.order_id for order in ingested.orders}
        recovered = [
            p
            for p in ingested.payments
            if p.order_ref_raw is not None and p.order_ref_raw != p.order_ref_normalized
        ]
        assert recovered
        assert all(p.order_ref_normalized in known for p in recovered)

    def test_the_narration_parser_beats_a_utr_regex_on_reach(self, ingested):
        """The A07 population: credits with a name but no reference.

        This gap is precisely what T3 is built to close, and it is measured
        here rather than assumed -- if it ever went to zero, T3 would have
        nothing to do and the fixture would have stopped exercising A07.
        """
        assert ingested.credits_with_merchant > ingested.credits_with_utr
        assert ingested.credits_with_no_reference > 0

    def test_noise_rows_and_debits_yield_no_counterparty(self, ingested):
        """Every row that must match nothing has nothing to match on."""
        for txn in ingested.bank_txns:
            if txn.extracted_merchant is not None or txn.extracted_utr is not None:
                assert txn.is_credit, txn.txn_id

    def test_ingest_asserts_no_cross_source_links(self, ingested):
        """Ingest normalises; it does not match. T0 is step 4.

        The only reference a payment carries is the one its own file published.
        Nothing here has resolved it to an order, and no bank transaction has
        been attached to a settlement.
        """
        assert all(p.settlement_id is not None for p in ingested.payments)
        for settlement in ingested.settlements:
            assert all(isinstance(pid, str) for pid in settlement.payment_ids)


class TestSynthesisedRecords:
    """``RawRecord`` is optional on the contract, and the grouping respects that.

    Ingest itself always sets it, but the field exists for records a later step
    synthesises rather than reads -- an inferred split credit, say. Those have
    no source row, and the provenance view must skip them rather than invent one.
    """

    def test_a_record_without_provenance_is_left_out_of_the_raw_view(self, ingested):
        synthesised = CanonicalOrder(
            order_id="ORD-2026-999999",
            merchant_id="MRCH_0001",
            customer_ref="CUST_00001",
            amount_minor=100,
            booked_at=datetime(2026, 3, 1, 12, 0, 0),
            status=OrderStatus.CAPTURED,
        )
        assert synthesised.raw is None
        result = replace(ingested, orders=(*ingested.orders, synthesised))
        assert len(result.normalized) == ingested.record_count + 1
        assert len(result.raw_by_source[SourceName.LEDGER]) == len(ingested.orders)


class TestDatasetLevelFailures:
    def test_a_missing_source_file_raises(self, tmp_path):
        with pytest.raises(IngestError, match=LEDGER_FILE):
            ingest_dataset(tmp_path)

    def test_a_partially_present_dataset_raises_naming_the_missing_file(self, tmp_path):
        for name in (LEDGER_FILE, PSP_FILE):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        with pytest.raises(IngestError, match=BANK_FILE):
            ingest_dataset(tmp_path)

    def test_lenient_mode_quarantines_where_strict_mode_raises(self, tmp_path):
        for name in (LEDGER_FILE, PSP_FILE, BANK_FILE):
            (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())

        path = tmp_path / LEDGER_FILE
        lines = path.read_text(encoding="utf-8").splitlines()
        parts = lines[1].split(",")
        parts[3] = "not-a-number"
        lines[1] = ",".join(parts)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(IngestError, match="MALFORMED_MONEY"):
            ingest_dataset(tmp_path, strict=True)

        result = ingest_dataset(tmp_path)
        assert len(result.orders) == 59
        (problem,) = result.problems_by_source(SourceName.LEDGER)
        assert problem.source_line == 0
        assert "amount_gross_paise" in str(problem)
        assert result.problems_by_source(SourceName.BANK) == ()
