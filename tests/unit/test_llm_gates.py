"""The gates that stand between a model's answer and this system's data.

A schema proves the *shape* of an answer. These prove its *provenance*, and
that is the entire safety argument for letting a model near a reconciliation:

* a UTR is a **join key**, so an invented one creates a match out of nothing;
* a cited record the prompt never contained is a hallucination, not a discovery;
* an invented id in prose sends a controller looking for a record that does not
  exist.

``verify_arithmetic`` is tested here too, because it is the same kind of thing:
a deterministic re-derivation that does not care who proposed the link. Its
signature is checked for that -- a gate that could be told "this one came from a
confident model" would eventually be told exactly that.
"""

from __future__ import annotations

import inspect

import pytest

from ledgerloop.llm.gates import (
    RECORD_ID_PATTERN,
    grounded_in_text,
    grounded_refs,
    prose_names_only_known_records,
)
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.verify import verify_arithmetic
from ledgerloop.models.refs import payment_ref, settlement_ref
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus, debit_row

NARRATION = "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT"
EPSILON = 300


class TestGroundingInText:
    def test_a_value_that_is_in_the_text_passes(self):
        assert grounded_in_text("UTR2026030412345", NARRATION)
        assert grounded_in_text("RAZORPAY SOFTWARE PVT", NARRATION)

    def test_casing_and_separator_noise_do_not_cause_a_false_rejection(self):
        assert grounded_in_text("razorpay software pvt", NARRATION)

    def test_an_invented_reference_is_refused(self):
        """The expensive one: a UTR is a join key."""
        gate = grounded_in_text("UTR2026039999999", NARRATION)
        assert not gate
        assert "does not occur" in gate.reason
        assert gate.offending == ("UTR2026039999999",)

    def test_an_invented_merchant_is_refused(self):
        assert not grounded_in_text("PAYTM PAYMENTS BANK", NARRATION)

    def test_none_is_grounded_because_absence_is_an_honest_answer(self):
        assert grounded_in_text(None, NARRATION)

    def test_an_empty_string_is_not_an_absence(self):
        gate = grounded_in_text("   ", NARRATION)
        assert not gate
        assert "empty string" in gate.reason


class TestGroundingReferences:
    def test_citing_only_supplied_records_passes(self):
        assert grounded_refs(["settlement:SETL-1"], ["settlement:SETL-1", "bank_txn:B"])

    def test_citing_nothing_passes(self):
        assert grounded_refs([], ["settlement:SETL-1"])

    def test_citing_a_record_that_was_not_in_the_pack_is_refused(self):
        gate = grounded_refs(["settlement:SETL-9"], ["settlement:SETL-1"])
        assert not gate
        assert "settlement:SETL-9" in gate.reason
        assert gate.offending == ("settlement:SETL-9",)

    def test_every_unknown_reference_is_named_not_just_the_first(self):
        gate = grounded_refs(["a", "b"], [])
        assert gate.offending == ("a", "b")

    def test_one_bad_reference_condemns_the_whole_citation(self):
        assert not grounded_refs(["settlement:SETL-1", "settlement:SETL-9"], ["settlement:SETL-1"])


class TestGroundingProse:
    def test_prose_naming_the_records_involved_passes(self):
        assert prose_names_only_known_records(
            "SETL-0001 is short by the amount of PAY-00002.",
            (settlement_ref("SETL-0001"), payment_ref("PAY-00002")),
        )

    def test_prose_naming_no_records_at_all_passes(self):
        assert prose_names_only_known_records(
            "The payout did not arrive.", (settlement_ref("SETL-0001"),)
        )

    def test_prose_inventing_a_record_is_refused(self):
        gate = prose_names_only_known_records(
            "SETL-0001 was netted against SETL-0099.",
            (settlement_ref("SETL-0001"),),
        )
        assert not gate
        assert "SETL-0099" in gate.reason

    @pytest.mark.parametrize(
        "identifier",
        ["ORD-2026-000123", "PAY-00042", "SETL-0091", "BNK-00007", "UTR2026030412345"],
    )
    def test_every_id_shape_the_generator_writes_is_recognised(self, identifier):
        assert RECORD_ID_PATTERN.findall(f"see {identifier} for detail") == [identifier]

    def test_ordinary_words_are_not_mistaken_for_ids(self):
        assert RECORD_ID_PATTERN.findall("the settlement was short by 500") == []


class TestVerifyArithmetic:
    @pytest.fixture
    def solved(self):
        """One batch paid in two tranches, allocated exactly as truth is."""
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        grosses = [payment.amount_minor for payment in only.payments]
        amounts = allocate_minor(only.net_minor, [grosses[0], grosses[1] + grosses[2]])
        credits = [
            bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
        ]
        return only, MatchContext.from_ingest(
            corpus(batches=[only], bank_txns=credits)
        )

    def test_a_partition_that_reconciles_is_verified(self, solved):
        only, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[0].payment_id,),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert check
        assert check.settlement_id == "SETL-0001"
        assert check.residual_minor == 0

    def test_a_subset_whose_money_does_not_close_is_refused(self, solved):
        only, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[1].payment_id,),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "which is not" in check.reason

    def test_a_credit_that_does_not_exist_is_refused(self, solved):
        only, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[0].payment_id,),
            bank_txn_id="BNK-99999",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "not a row in the statement" in check.reason

    def test_a_payment_that_does_not_exist_is_refused(self, solved):
        _, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=("PAY-99999",),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "no such payment" in check.reason

    def test_naming_no_payments_is_refused(self, solved):
        _, context = solved
        check = verify_arithmetic(
            context, payment_ids=(), bank_txn_id="BNK-00001", epsilon_minor=EPSILON
        )
        assert not check
        assert "no payments were named" in check.reason

    def test_naming_one_payment_twice_is_refused(self, solved):
        only, context = solved
        payment = only.payments[0].payment_id
        check = verify_arithmetic(
            context,
            payment_ids=(payment, payment),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "travels once" in check.reason

    def test_a_debit_can_never_settle_a_payout(self):
        only = batch()
        context = MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[debit_row("BNK-09002", utr=only.settlement.utr)],
            )
        )
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[0].payment_id,),
            bank_txn_id="BNK-09002",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "outgoing row" in check.reason

    def test_payments_from_two_batches_are_refused_before_any_arithmetic(self):
        first = batch("SETL-0001")
        second = batch("SETL-0002", first_index=5, utr="UTR2026031099999")
        context = MatchContext.from_ingest(
            corpus(batches=[first, second], bank_txns=[first.credit()])
        )
        check = verify_arithmetic(
            context,
            payment_ids=(first.payments[0].payment_id, second.payments[0].payment_id),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert not check
        assert "not all nested in one settlement" in check.reason

    def test_a_settlement_the_payments_do_not_belong_to_is_refused(self, solved):
        only, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[0].payment_id,),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
            settlement_id="SETL-9999",
        )
        assert not check
        assert "not in SETL-9999" in check.reason

    def test_the_check_is_truthy_and_carries_its_working(self, solved):
        only, context = solved
        check = verify_arithmetic(
            context,
            payment_ids=(only.payments[0].payment_id,),
            bank_txn_id="BNK-00001",
            epsilon_minor=EPSILON,
        )
        assert bool(check) is True
        assert check.gross_minor == only.payments[0].amount_minor
        assert check.credit is not None
        assert len(check.payments) == 1
        assert "allocate to" in check.reason

    def test_it_cannot_be_told_where_a_proposal_came_from(self):
        """A gate with a "this one is from a model" parameter would be given one."""
        parameters = set(inspect.signature(verify_arithmetic).parameters)
        assert parameters == {
            "context",
            "payment_ids",
            "bank_txn_id",
            "epsilon_minor",
            "settlement_id",
        }
