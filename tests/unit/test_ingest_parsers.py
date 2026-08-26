"""The three source parsers, in isolation, on hand-built input.

Everything here is small enough to verify by eye. The parsers are exercised
against the real corpus in ``test_ingest_dataset.py``; this file is about the
inputs the corpus does *not* contain -- the malformed, the contradictory and
the out-of-scope -- because those are the paths a clean fixture never covers
and the ones a real month-end file will find first.

The contract under test is the same for all three: a row-level defect is
**quarantined with a diagnosis**, a file-level defect **raises**, and nothing
is ever silently defaulted.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.ingest.bank import parse_bank_rows, read_bank_rows
from ledgerloop.ingest.dates import DateOrder
from ledgerloop.ingest.fields import RESTKEY
from ledgerloop.ingest.ledger import parse_ledger_rows, read_ledger_rows
from ledgerloop.ingest.problems import IngestError, IngestProblemCode, ProblemLog
from ledgerloop.ingest.psp import parse_psp_payload, read_psp_payload
from ledgerloop.ingest.schemas import LEDGER_SCHEMA
from ledgerloop.models.enums import Currency, OrderStatus, SourceName


def _log() -> ProblemLog:
    return ProblemLog()


def _order_row(**overrides: str) -> dict[str, str]:
    row = {
        "order_id": "ORD-2026-000001",
        "merchant_id": "MRCH_0001",
        "customer_ref": "CUST_18977",
        "amount_gross_paise": "4265200",
        "currency": "INR",
        "booked_at": "2026-03-17T15:05:14",
        "status": "CAPTURED",
    }
    row.update(overrides)
    return row


def _bank_row(**overrides: str) -> dict[str, str]:
    row = {
        "txn_id": "BNK-00001",
        "value_date": "14/03/2026",
        "narration": "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026031412345-SETTLEMENT",
        "credit_paise": "3680323",
        "debit_paise": "0",
        "balance_paise": "19844210",
    }
    row.update(overrides)
    return row


def _batch(**overrides: object) -> dict[str, object]:
    batch: dict[str, object] = {
        "settlement_id": "SETL-0091",
        "merchant_id": "MRCH_0001",
        "utr": "UTR2026031412345",
        "settled_on": "2026-03-14",
        "gross_paise": 4210900,
        "fee_paise": 84218,
        "tax_paise": 15159,
        "adjustments_paise": -431200,
        "net_paise": 3680323,
        "payments": [
            {
                "payment_id": "PAY-88301",
                "order_ref": "ORD-2026-004821",
                "amount_paise": 499900,
                "captured_at": "2026-03-04T11:22:11+05:30",
            }
        ],
    }
    batch.update(overrides)
    return batch


# ----------------------------------------------------------------------
# Source A -- the ledger
# ----------------------------------------------------------------------


class TestLedger:
    def test_a_clean_row_becomes_a_canonical_order(self):
        log = _log()
        (order,) = parse_ledger_rows([_order_row()], log)
        assert not log
        assert order.order_id == "ORD-2026-000001"
        assert order.amount_minor == 4265200
        assert order.currency is Currency.INR
        assert order.status is OrderStatus.CAPTURED
        assert order.source is SourceName.LEDGER

    def test_provenance_points_back_at_the_source_row(self):
        log = _log()
        orders = parse_ledger_rows([_order_row(), _order_row(order_id="ORD-2026-000002")], log)
        assert [o.raw.source_line for o in orders] == [0, 1]
        assert orders[1].raw.payload["order_id"] == "ORD-2026-000002"
        assert orders[0].raw.source is SourceName.LEDGER

    @pytest.mark.parametrize(
        ("field", "value", "code"),
        [
            ("order_id", "", IngestProblemCode.EMPTY_IDENTIFIER),
            ("merchant_id", "   ", IngestProblemCode.EMPTY_IDENTIFIER),
            ("amount_gross_paise", "4265.20", IngestProblemCode.MALFORMED_MONEY),
            ("amount_gross_paise", "not a number", IngestProblemCode.MALFORMED_MONEY),
            ("amount_gross_paise", "", IngestProblemCode.MISSING_FIELD),
            ("booked_at", "17/03/2026", IngestProblemCode.MALFORMED_DATE),
            ("status", "SHIPPED", IngestProblemCode.UNKNOWN_ENUM_VALUE),
            ("currency", "USD", IngestProblemCode.UNSUPPORTED_CURRENCY),
            ("currency", "XYZ", IngestProblemCode.UNKNOWN_ENUM_VALUE),
        ],
    )
    def test_a_malformed_field_quarantines_the_row_with_a_diagnosis(self, field, value, code):
        log = _log()
        orders = parse_ledger_rows([_order_row(**{field: value})], log)
        assert orders == ()
        (problem,) = log.problems
        assert problem.code is code
        assert problem.field == field
        assert problem.source is SourceName.LEDGER
        assert problem.payload[field] == value

    def test_the_a11_fx_cut_is_testable_rather_than_merely_absent(self):
        """``Currency.USD.supported is False``, so ingest refuses it by name."""
        log = _log()
        parse_ledger_rows([_order_row(currency="USD")], log)
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.UNSUPPORTED_CURRENCY
        assert "INR-only" in problem.detail

    def test_a_missing_currency_column_defaults_to_inr(self):
        log = _log()
        row = _order_row()
        del row["currency"]
        (order,) = parse_ledger_rows([row], log)
        assert not log
        assert order.currency is Currency.INR

    def test_one_bad_row_does_not_stop_the_others(self):
        log = _log()
        orders = parse_ledger_rows(
            [
                _order_row(),
                _order_row(order_id="ORD-2026-000002", amount_gross_paise="oops"),
                _order_row(order_id="ORD-2026-000003"),
            ],
            log,
        )
        assert [o.order_id for o in orders] == ["ORD-2026-000001", "ORD-2026-000003"]
        assert len(log) == 1

    def test_a_duplicate_order_id_keeps_the_first(self):
        log = _log()
        orders = parse_ledger_rows(
            [_order_row(amount_gross_paise="100"), _order_row(amount_gross_paise="200")], log
        )
        assert [o.amount_minor for o in orders] == [100]
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.DUPLICATE_ID

    def test_strict_mode_raises_on_the_first_problem(self):
        with pytest.raises(IngestError, match="MALFORMED_MONEY"):
            parse_ledger_rows(
                [_order_row(amount_gross_paise="oops")], ProblemLog(strict=True)
            )

    def test_a_missing_required_column_is_a_file_level_failure(self, tmp_path):
        path = tmp_path / "ledger_orders.csv"
        path.write_text("order_id,merchant_id\nORD-1,MRCH_1\n", encoding="utf-8")
        with pytest.raises(IngestError, match="amount_gross_paise"):
            read_ledger_rows(path)

    def test_an_extra_column_is_ignored_not_rejected(self, tmp_path):
        path = tmp_path / "ledger_orders.csv"
        header = ",".join([*_order_row().keys(), "bank_ref"])
        values = ",".join([*_order_row().values(), "XYZ"])
        path.write_text(f"{header}\n{values}\n", encoding="utf-8")
        rows = read_ledger_rows(path)
        log = _log()
        (order,) = parse_ledger_rows(rows, log)
        assert not log
        assert order.raw.payload["bank_ref"] == "XYZ"  # kept in provenance


# ----------------------------------------------------------------------
# Source B -- the PSP report
# ----------------------------------------------------------------------


class TestPsp:
    def test_one_batch_yields_a_settlement_and_its_payments(self):
        log = _log()
        records = parse_psp_payload([_batch()], log)
        assert not log
        (settlement,) = records.settlements
        (payment,) = records.payments
        assert settlement.settlement_id == "SETL-0091"
        assert settlement.payment_ids == ("PAY-88301",)
        assert payment.settlement_id == "SETL-0091"
        assert payment.amount_minor == 499900

    def test_the_settlement_arithmetic_is_exposed_never_enforced(self):
        """A03 ``FEE_TAX_MISMATCH`` breaks this identity on purpose."""
        log = _log()
        (settlement,) = parse_psp_payload([_batch(net_paise=3680000)], log).settlements
        assert not log  # parsed, not rejected
        assert settlement.net_delta_minor != 0

    def test_the_three_reference_corruptions_are_handled_distinctly(self):
        log = _log()
        batch = _batch(
            payments=[
                {
                    "payment_id": "PAY-1",
                    "order_ref": "ORD-2026-004821",
                    "amount_paise": 1,
                    "captured_at": "2026-03-04T11:22:11",
                },
                {
                    "payment_id": "PAY-2",
                    "order_ref": None,
                    "amount_paise": 1,
                    "captured_at": "2026-03-04T11:22:11",
                },
                {
                    "payment_id": "PAY-3",
                    "order_ref": "ord 2026 004821",
                    "amount_paise": 1,
                    "captured_at": "2026-03-04T11:22:11",
                },
                {
                    "payment_id": "PAY-4",
                    "order_ref": f"ORD{chr(0x2011)}2026{chr(0x2011)}004821",
                    "amount_paise": 1,
                    "captured_at": "2026-03-04T11:22:11",
                },
            ]
        )
        payments = parse_psp_payload([batch], log).payments
        assert not log  # a corrupt reference is expected input, not a defect
        assert [p.order_ref_normalized for p in payments] == [
            "ORD-2026-004821",
            None,
            "ORD-2026-004821",
            "ORD-2026-004821",
        ]

    def test_the_raw_reference_is_preserved_alongside_the_recovered_one(self):
        """What lets an exception explain *why* an exact join missed."""
        log = _log()
        batch = _batch(
            payments=[
                {
                    "payment_id": "PAY-1",
                    "order_ref": "ord 2026 004821",
                    "amount_paise": 1,
                    "captured_at": "2026-03-04T11:22:11",
                }
            ]
        )
        (payment,) = parse_psp_payload([batch], log).payments
        assert payment.order_ref_raw == "ord 2026 004821"
        assert payment.order_ref_normalized == "ORD-2026-004821"

    def test_an_offset_aware_capture_time_is_normalised_to_naive_ist(self):
        log = _log()
        (payment,) = parse_psp_payload([_batch()], log).payments
        assert payment.captured_at.tzinfo is None
        assert payment.captured_at.hour == 11

    def test_payments_are_numbered_across_the_whole_file(self):
        log = _log()
        def _payment(pid):
            return {
                "payment_id": pid,
                "order_ref": None,
                "amount_paise": 1,
                "captured_at": "2026-03-04T11:22:11",
            }

        records = parse_psp_payload(
            [
                _batch(settlement_id="SETL-1", payments=[_payment("P1"), _payment("P2")]),
                _batch(settlement_id="SETL-2", payments=[_payment("P3")]),
            ],
            log,
        )
        assert [s.raw.source_line for s in records.settlements] == [0, 1]
        assert [p.raw.source_line for p in records.payments] == [0, 1, 2]

    def test_optional_amounts_default_to_zero(self):
        log = _log()
        batch = _batch()
        del batch["tax_paise"]
        del batch["adjustments_paise"]
        (settlement,) = parse_psp_payload([batch], log).settlements
        assert not log
        assert settlement.tax_minor == 0
        assert settlement.adjustments_minor == 0

    def test_a_missing_utr_is_allowed(self):
        log = _log()
        (settlement,) = parse_psp_payload([_batch(utr=None)], log).settlements
        assert not log
        assert settlement.utr is None

    def test_a_bad_payment_does_not_take_down_its_settlement(self):
        log = _log()
        batch = _batch(
            payments=[
                {
                    "payment_id": "PAY-1",
                    "amount_paise": "oops",
                    "captured_at": "2026-03-04T11:22:11",
                },
                {"payment_id": "PAY-2", "amount_paise": 1, "captured_at": "2026-03-04T11:22:11"},
            ]
        )
        records = parse_psp_payload([batch], log)
        assert len(records.settlements) == 1
        assert records.settlements[0].payment_ids == ("PAY-2",)
        assert len(log) == 1

    def test_a_bad_settlement_takes_its_payments_with_it(self):
        """No payment may survive with a ``settlement_id`` pointing at nothing."""
        log = _log()
        records = parse_psp_payload([_batch(settlement_id="")], log)
        assert records.settlements == ()
        assert records.payments == ()

    def test_a_payment_missing_a_required_field_is_quarantined(self):
        log = _log()
        batch = _batch(payments=[{"payment_id": "PAY-1", "amount_paise": 1}])
        records = parse_psp_payload([batch], log)
        assert records.payments == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MISSING_FIELD
        assert "captured_at" in problem.detail

    def test_a_payment_that_is_not_an_object_is_quarantined(self):
        log = _log()
        records = parse_psp_payload([_batch(payments=["PAY-1"])], log)
        assert records.payments == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_STRUCTURE

    def test_a_payments_field_that_is_not_an_array_rejects_the_batch(self):
        log = _log()
        records = parse_psp_payload([_batch(payments={"a": 1})], log)
        assert records.settlements == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_STRUCTURE

    def test_a_duplicate_payment_id_keeps_the_first(self):
        log = _log()
        payment = {
            "payment_id": "PAY-1",
            "amount_paise": 1,
            "captured_at": "2026-03-04T11:22:11",
        }
        records = parse_psp_payload(
            [
                _batch(settlement_id="SETL-1", payments=[payment]),
                _batch(settlement_id="SETL-2", payments=[payment]),
            ],
            log,
        )
        assert len(records.payments) == 1
        assert records.settlements[1].payment_ids == ()
        assert log.problems[0].code is IngestProblemCode.DUPLICATE_ID

    def test_a_duplicate_settlement_id_keeps_the_first(self):
        log = _log()
        records = parse_psp_payload([_batch(), _batch(net_paise=1)], log)
        assert [s.net_minor for s in records.settlements] == [3680323]
        assert log.problems[0].code is IngestProblemCode.DUPLICATE_ID

    def test_a_float_amount_is_refused_by_the_money_gate(self):
        """JSON can carry a float where minor units were meant. Nothing else can."""
        log = _log()
        records = parse_psp_payload([_batch(gross_paise=4210900.0)], log)
        assert records.settlements == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_MONEY
        assert "float" in problem.detail

    @pytest.mark.parametrize(
        "document",
        ['["not an object"]', '{"batches": []}', '{"settlements": {"a": 1}}'],
    )
    def test_a_structurally_wrong_document_raises(self, tmp_path, document):
        path = tmp_path / "psp_settlements.json"
        path.write_text(document, encoding="utf-8")
        with pytest.raises(IngestError):
            read_psp_payload(path)

    def test_a_settlement_that_is_not_an_object_raises(self, tmp_path):
        path = tmp_path / "psp_settlements.json"
        path.write_text(json.dumps({"settlements": [1]}), encoding="utf-8")
        with pytest.raises(IngestError, match=r"settlements\[0\]"):
            read_psp_payload(path)

    def test_a_batch_missing_a_required_field_is_quarantined(self):
        log = _log()
        batch = _batch()
        del batch["net_paise"]
        records = parse_psp_payload([batch], log)
        assert records.settlements == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MISSING_FIELD
        assert "net_paise" in problem.detail


# ----------------------------------------------------------------------
# Source C -- the bank statement
# ----------------------------------------------------------------------


class TestBank:
    def test_a_clean_credit_row_is_fully_extracted(self):
        log = _log()
        records = parse_bank_rows([_bank_row()], log)
        (txn,) = records.transactions
        assert not log
        assert txn.credit_minor == 3680323
        assert txn.debit_minor == 0
        assert txn.is_credit
        assert txn.extracted_utr == "UTR2026031412345"
        assert txn.extracted_merchant == "RAZORPAY SOFTWARE PVT"
        assert txn.narration_normalized.startswith("NEFT CR RAZORPAY")

    def test_the_date_order_is_inferred_from_the_column_not_assumed(self):
        log = _log()
        records = parse_bank_rows(
            [_bank_row(value_date="06/03/2026"), _bank_row(txn_id="B2", value_date="14/03/2026")],
            log,
        )
        assert records.date_order.proven
        assert records.date_order.order is DateOrder.DAY_FIRST
        assert records.transactions[0].value_date.month == 3
        assert records.transactions[0].value_date.day == 6

    def test_a_month_first_column_is_read_month_first(self):
        """The inference is real: the same digits parse differently on evidence."""
        log = _log()
        records = parse_bank_rows(
            [_bank_row(value_date="06/03/2026"), _bank_row(txn_id="B2", value_date="03/14/2026")],
            log,
        )
        assert records.date_order.order is DateOrder.MONTH_FIRST
        assert records.transactions[0].value_date.month == 6

    def test_an_undecidable_column_uses_the_convention_and_says_so(self):
        log = _log()
        records = parse_bank_rows([_bank_row(value_date="06/03/2026")], log)
        assert not records.date_order.proven
        assert records.date_order.order is DateOrder.DAY_FIRST
        assert not log  # a convention is evidence, not a malformed record

    def test_a_contradictory_date_column_is_a_file_level_failure(self):
        with pytest.raises(IngestError, match="both orders"):
            parse_bank_rows(
                [
                    _bank_row(value_date="14/03/2026"),
                    _bank_row(txn_id="B2", value_date="03/14/2026"),
                ],
                _log(),
            )

    def test_a07_leaves_no_utr_but_keeps_the_merchant(self):
        log = _log()
        records = parse_bank_rows(
            [_bank_row(narration="NEFT CR-RZRPAY SFTWR P L-SETTLEMENT")], log
        )
        (txn,) = records.transactions
        assert txn.extracted_utr is None
        assert txn.extracted_merchant == "RZRPAY SFTWR P L"

    def test_a_noise_row_extracts_nothing(self):
        log = _log()
        records = parse_bank_rows(
            [_bank_row(narration="RENT PAYMENT COMMERCIAL PREMISES", credit_paise="0",
                       debit_paise="500000")],
            log,
        )
        (txn,) = records.transactions
        assert txn.extracted_utr is None
        assert txn.extracted_merchant is None
        assert not txn.is_credit

    def test_a_row_asserting_both_a_credit_and_a_debit_is_refused(self):
        log = _log()
        records = parse_bank_rows([_bank_row(credit_paise="100", debit_paise="200")], log)
        assert records.transactions == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.CONTRADICTORY_SIGNS

    @pytest.mark.parametrize("field", ["credit_paise", "debit_paise"])
    def test_a_negative_amount_is_refused_rather_than_flipped(self, field):
        log = _log()
        records = parse_bank_rows([_bank_row(**{field: "-100"})], log)
        assert records.transactions == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.NEGATIVE_AMOUNT
        assert problem.field == field

    def test_an_impossible_date_quarantines_the_row(self):
        log = _log()
        records = parse_bank_rows(
            [_bank_row(value_date="14/03/2026"), _bank_row(txn_id="B2", value_date="31/02/2026")],
            log,
        )
        assert len(records.transactions) == 1
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_DATE
        assert problem.record_id == "B2"

    def test_a_missing_balance_column_leaves_the_field_none(self):
        log = _log()
        row = _bank_row()
        del row["balance_paise"]
        (txn,) = parse_bank_rows([row], log).transactions
        assert not log
        assert txn.balance_minor is None

    def test_a_duplicate_txn_id_keeps_the_first(self):
        log = _log()
        records = parse_bank_rows(
            [_bank_row(credit_paise="100"), _bank_row(credit_paise="200")], log
        )
        assert [t.credit_minor for t in records.transactions] == [100]
        assert log.problems[0].code is IngestProblemCode.DUPLICATE_ID

    def test_the_narration_parses_travel_with_the_records(self):
        log = _log()
        records = parse_bank_rows([_bank_row()], log)
        assert records.narrations["BNK-00001"].rail == "NEFT"

    def test_a_missing_required_column_is_a_file_level_failure(self, tmp_path):
        path = tmp_path / "bank_statement.csv"
        path.write_text("txn_id,value_date\nBNK-1,14/03/2026\n", encoding="utf-8")
        with pytest.raises(IngestError, match="narration"):
            read_bank_rows(path)

    def test_an_empty_narration_quarantines_the_row(self):
        log = _log()
        records = parse_bank_rows([_bank_row(narration="  ")], log)
        assert records.transactions == ()
        assert log.problems[0].field == "narration"


# ----------------------------------------------------------------------
# Shared machinery: the schema objects and the problem log
# ----------------------------------------------------------------------


class TestRaggedRows:
    """A row with more values than the header. Usually broken quoting."""

    def test_a_ledger_row_with_extra_values_is_quarantined(self, tmp_path):
        path = tmp_path / "ledger_orders.csv"
        header = ",".join(_order_row().keys())
        values = ",".join(_order_row().values()) + ",stray"
        path.write_text(f"{header}\n{values}\n", encoding="utf-8")
        log = _log()
        assert parse_ledger_rows(read_ledger_rows(path), log) == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_STRUCTURE
        assert "beyond the header" in problem.detail

    def test_an_unescaped_comma_in_a_narration_is_quarantined(self, tmp_path):
        path = tmp_path / "bank_statement.csv"
        header = ",".join(_bank_row().keys())
        values = ",".join(_bank_row(narration="NEFT CR-ACME, LTD-SETTLEMENT").values())
        path.write_text(f"{header}\n{values}\n", encoding="utf-8")
        log = _log()
        assert parse_bank_rows(read_bank_rows(path), log).transactions == ()
        assert log.problems[0].code is IngestProblemCode.MALFORMED_STRUCTURE

    def test_the_stray_values_survive_into_the_problem_payload(self, tmp_path):
        path = tmp_path / "ledger_orders.csv"
        header = ",".join(_order_row().keys())
        path.write_text(
            f"{header}\n{','.join(_order_row().values())},stray\n", encoding="utf-8"
        )
        log = _log()
        parse_ledger_rows(read_ledger_rows(path), log)
        assert log.problems[0].payload[RESTKEY] == ["stray"]


class TestTheContractViolationGuard:
    """The last line of defence: a record the *model* refuses.

    Nothing a well-formed file contains reaches here -- the field readers catch
    every defect first. It exists so that a validator added to the contract
    layer later quarantines a record rather than crashing the run, and it is
    exercised by handing a parser a payload the model cannot hold.
    """

    def test_a_ledger_row_the_model_rejects_is_quarantined(self):
        log = _log()
        row = _order_row()
        row[42] = "provenance payloads are keyed by strings"  # type: ignore[index]
        assert parse_ledger_rows([row], log) == ()
        assert log.problems[0].code is IngestProblemCode.CONTRACT_VIOLATION

    def test_a_bank_row_the_model_rejects_is_quarantined(self):
        log = _log()
        row = _bank_row()
        row[42] = "not a string key"  # type: ignore[index]
        assert parse_bank_rows([row], log).transactions == ()
        assert log.problems[0].code is IngestProblemCode.CONTRACT_VIOLATION

    def test_a_settlement_the_model_rejects_is_quarantined(self):
        log = _log()
        batch = _batch()
        batch[42] = "not a string key"  # type: ignore[index]
        assert parse_psp_payload([batch], log).settlements == ()
        assert any(
            p.code is IngestProblemCode.CONTRACT_VIOLATION for p in log.problems
        )

    def test_a_payment_the_model_rejects_is_quarantined(self):
        log = _log()
        payment: dict[object, object] = {
            "payment_id": "PAY-1",
            "amount_paise": 1,
            "captured_at": "2026-03-04T11:22:11",
            42: "not a string key",
        }
        records = parse_psp_payload([_batch(payments=[payment])], log)
        assert records.payments == ()
        assert log.problems[0].code is IngestProblemCode.CONTRACT_VIOLATION


class TestMoreMalformedInput:
    def test_a_json_array_where_an_amount_belongs(self):
        log = _log()
        assert parse_psp_payload([_batch(gross_paise=[1, 2])], log).settlements == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_MONEY
        assert "list is not a minor-unit amount" in problem.detail

    def test_a_settled_on_in_the_wrong_format(self):
        log = _log()
        assert parse_psp_payload([_batch(settled_on="14/03/2026")], log).settlements == ()
        (problem,) = log.problems
        assert problem.code is IngestProblemCode.MALFORMED_DATE
        assert problem.field == "settled_on"

    def test_a_captured_at_in_the_wrong_format(self):
        log = _log()
        batch = _batch(
            payments=[
                {
                    "payment_id": "PAY-1",
                    "amount_paise": 1,
                    "captured_at": "04/03/2026 11:22",
                }
            ]
        )
        assert parse_psp_payload([batch], log).payments == ()
        assert log.problems[0].code is IngestProblemCode.MALFORMED_DATE


class TestSchemasAndProblems:
    def test_a_schema_lists_required_then_optional_fields(self):
        assert LEDGER_SCHEMA.fields == LEDGER_SCHEMA.required + LEDGER_SCHEMA.optional
        assert "order_id" in LEDGER_SCHEMA.fields
        assert "currency" in LEDGER_SCHEMA.optional

    def test_a_complete_header_reports_no_mismatch(self):
        assert LEDGER_SCHEMA.describe_mismatch(LEDGER_SCHEMA.fields) is None
        assert LEDGER_SCHEMA.missing_from(LEDGER_SCHEMA.required) == ()

    def test_a_mismatch_names_every_missing_field_in_declaration_order(self):
        message = LEDGER_SCHEMA.describe_mismatch(["order_id", "status"])
        assert message is not None
        assert message.index("merchant_id") < message.index("booked_at")

    def test_header_whitespace_is_tolerated(self):
        assert LEDGER_SCHEMA.missing_from([f" {f} " for f in LEDGER_SCHEMA.required]) == ()

    def test_a_problem_renders_everything_it_knows(self):
        log = _log()
        parse_ledger_rows([_order_row(amount_gross_paise="oops")], log)
        rendered = str(log.problems[0])
        assert "ledger[0]" in rendered
        assert "ORD-2026-000001" in rendered
        assert "amount_gross_paise" in rendered
        assert "MALFORMED_MONEY" in rendered

    def test_a_problem_with_no_record_or_field_still_renders(self):
        log = _log()
        problem = log.record(
            source=SourceName.PSP,
            source_line=3,
            code=IngestProblemCode.MALFORMED_STRUCTURE,
            detail="something structural",
        )
        assert str(problem) == "psp[3]: MALFORMED_STRUCTURE -- something structural"

    def test_the_log_counts_and_is_falsy_when_empty(self):
        log = _log()
        assert not log
        assert len(log) == 0
        parse_ledger_rows([_order_row(status="SHIPPED")], log)
        assert log
        assert len(log) == 1
