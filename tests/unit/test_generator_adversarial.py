"""Model B -- the second construction model, tested as a source of evidence.

A validation corpus is only worth what its construction is worth. These tests
are about the *builder*, not about the matcher: that money is conserved, that
truth still comes from what was constructed rather than from what was emitted,
that the case layout is fixed while the money moves with the seed, and above all
that model B actually expresses the degree of freedom model A cannot -- a credit
that is both unreferenced and off its net.

The matcher's behaviour on this corpus is measured in
``test_t3_exactness_validation.py``.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import GeneratorConfig
from ledgerloop.generator.adversarial import (
    ADVERSARIAL_VERSION,
    BANK_CHARGE_MINOR,
    Case,
    build_adversarial_corpus,
    write_adversarial_corpus,
)
from ledgerloop.generator.generate import generate
from ledgerloop.models.enums import AnomalyClass, LinkType, SplitName


@pytest.fixture(scope="module")
def corpus():
    return build_adversarial_corpus(42)


class TestItIsAWorldAndNotJustARow:
    def test_money_is_conserved_modulo_declared_deductions(self, corpus):
        """The same statement model A makes, and the same reason it matters.

        Every bank-side deduction declares a ``bank_delta_minor``. A residual
        would mean the builder moved money without saying so, and every number
        measured on this corpus would describe a different world than the one
        the report claims.
        """
        assert corpus.conservation_residual_minor == 0

    def test_every_case_shape_is_present(self, corpus):
        assert set(corpus.by_case()) == set(Case)

    def test_every_merchant_carries_the_whole_case_list(self, corpus):
        merchants = {record.merchant_id for record in corpus.cases}
        for case, records in corpus.by_case().items():
            covered = {record.merchant_id for record in records}
            assert covered == merchants, f"{case} is missing merchants {merchants - covered}"

    def test_a_merchant_identity_is_shared_by_many_settlements(self, corpus):
        """The condition under which same-merchant amounts collide.

        Phase 2.6's twenty-two false positives needed a merchant with enough
        payouts that two of them fall inside each other's band. A corpus of
        one settlement per merchant could not reproduce that.
        """
        counts = {}
        for record in corpus.cases:
            counts[record.merchant_id] = counts.get(record.merchant_id, 0) + 1
        assert min(counts.values()) >= 16

    def test_it_is_labelled_as_model_b(self, corpus):
        assert corpus.truth.generator_version == ADVERSARIAL_VERSION

    def test_it_rejects_a_merchant_count_it_cannot_seat(self):
        with pytest.raises(ValueError, match="between 1 and"):
            build_adversarial_corpus(42, merchants=99)


class TestTruthStillComesFromConstruction:
    def test_a_charged_back_payment_earns_no_link(self, corpus):
        """Its money never reached the bank, so no credit can carry it."""
        credited = {
            link.source_ref.key
            for link in corpus.truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        charged_back = {
            record.record_ref.key
            for record in corpus.truth.records
            if record.anomaly_class is AnomalyClass.CHARGEBACK_NETTED
        }
        assert charged_back
        assert not (charged_back & credited)

    def test_an_orphan_credit_earns_no_link(self, corpus):
        """A credit with no settlement behind it. Linking it is a false positive."""
        targets = {
            link.target_ref.key
            for link in corpus.truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        orphans = {
            record.record_ref.key
            for record in corpus.truth.records
            if record.anomaly_class is AnomalyClass.ORPHAN_BANK_CREDIT
        }
        assert orphans
        assert not (orphans & targets)

    def test_the_link_amounts_sum_to_the_credit(self, corpus):
        """No paise created by a deduction, none destroyed by a split."""
        by_credit: dict[str, int] = {}
        for link in corpus.truth.links:
            if link.link_type is LinkType.PAYMENT_CREDITED_AS:
                by_credit[link.target_ref.key] = (
                    by_credit.get(link.target_ref.key, 0) + link.amount_minor
                )
        for txn in corpus.world.bank_txns:
            if not txn.covered_payment_ids:
                continue
            assert by_credit[f"bank_txn:{txn.txn_id}"] == txn.credit_minor

    def test_no_case_declares_what_the_matcher_should_do(self):
        """`Case` names a shape. An expectation field would make the outcome
        table a restatement of the builder's opinion rather than a measurement."""
        fields = set(Case.__members__)
        assert not any("expect" in name.lower() or "should" in name.lower() for name in fields)


class TestTheDegreeOfFreedomModelACannotExpress:
    """The whole reason model B exists.

    In model A, ``A02_ROUNDING_DRIFT`` is the only class that moves a whole-net
    credit off its net, and it cannot land on the same settlement as
    ``A07_MISSING_REFERENCE``: both take ``ASPECT_PRIMARY``, because a
    ``GroundTruthRecord`` carries one anomaly class per record. So "no reference"
    and "off the net" are mutually exclusive there -- for a truth-representation
    reason, not a financial one.
    """

    def test_model_a_never_produces_an_inexact_unreferenced_credit(self):
        """The artefact, asserted on model A itself so it cannot drift silently."""
        dataset = generate(GeneratorConfig(split=SplitName.DEV, seed=42))
        world = dataset.world
        payments = world.payments_by_id()
        inexact_unreferenced = 0
        for settlement in world.settlements:
            linked = [
                txn
                for txn in world.credits_for_settlement(settlement.settlement_id)
                if txn.covered_payment_ids
            ]
            if len(linked) != 1:
                continue
            credit = linked[0]
            if "UTR" in credit.narration:
                continue
            if credit.credit_minor != settlement.declared_net_minor(payments):
                inexact_unreferenced += 1
        assert inexact_unreferenced == 0

    def test_model_b_does(self, corpus):
        """And it produces them in quantity, across four different shapes."""
        inexact = [
            record
            for record in corpus.cases
            if not record.referenced and len(record.credit_ids) == 1 and record.delta_minor != 0
        ]
        assert {record.case for record in inexact} == {
            Case.UNREF_CHARGE,
            Case.UNREF_CHARGEBACK_CHARGE,
            Case.UNREF_HAIRCUT,
            Case.UNREF_ROUNDED,
        }
        assert len(inexact) >= 20

    def test_the_reference_and_the_deduction_are_independent(self, corpus):
        """The same flat charge appears with a reference and without one.

        That independence is what makes the corpus a test rather than a
        demonstration: nothing about the deduction predicts the reference.
        """
        charged = {
            record.referenced
            for record in corpus.cases
            if record.delta_minor == -BANK_CHARGE_MINOR and len(record.credit_ids) == 1
        }
        assert charged == {True, False}


class TestTheCasesAreWhatTheyClaim:
    def test_exact_cases_are_exact_to_the_paise(self, corpus):
        exact = {
            Case.REFERENCED_EXACT,
            Case.UNREF_EXACT,
            Case.UNREF_TWIN_EXACT,
            Case.UNREF_CHARGEBACK_EXACT,
            Case.UNREF_LATE_ONLY,
            Case.UNREF_SPLIT_EXACT,
            Case.UNREF_TRANCHE_BAIT,
            Case.UNREF_TRANCHE_HOST,
            Case.UNREF_ORPHAN_NEAR,
            Case.UNREF_ORPHAN_EXACT,
        }
        for case in exact:
            assert {record.delta_minor for record in corpus.by_case()[case]} == {0}

    def test_the_deducted_cases_are_short_by_the_flat_charge(self, corpus):
        for case in (Case.REFERENCED_DEDUCTED, Case.UNREF_CHARGE, Case.UNREF_CHARGEBACK_CHARGE):
            assert {record.delta_minor for record in corpus.by_case()[case]} == {
                -BANK_CHARGE_MINOR
            }

    def test_a_deduction_is_never_bigger_than_the_band_that_admits_it(self, corpus):
        """Otherwise the refusals would be date- or pool-gated rather than
        amount-gated, and the measurement would be of the wrong thing."""
        for record in corpus.cases:
            if record.referenced or record.delta_minor == 0:
                continue
            band = max(100, record.net_minor * 50 // 10_000)
            assert abs(record.delta_minor) <= band, record

    def test_the_twins_are_genuinely_indistinguishable(self, corpus):
        """Same net, same value date, same merchant, no reference on either."""
        twins = corpus.by_case()[Case.UNREF_TWIN_EXACT]
        by_merchant: dict[str, list[int]] = {}
        for record in twins:
            by_merchant.setdefault(record.merchant_id, []).append(record.net_minor)
        for nets in by_merchant.values():
            assert len(nets) == 2 and nets[0] == nets[1]
        dates = {
            txn.value_date
            for txn in corpus.world.bank_txns
            for record in twins
            if txn.settlement_id == record.settlement_id
        }
        assert len(dates) == len(by_merchant)

    def test_the_lookalike_tranche_is_inside_the_band_and_not_equal(self, corpus):
        """The Phase 2.10 geometry: close enough to be considered, never equal."""
        baits = corpus.by_case()[Case.UNREF_TRANCHE_BAIT]
        hosts = {
            record.merchant_id: record for record in corpus.by_case()[Case.UNREF_TRANCHE_HOST]
        }
        credits = {txn.txn_id: txn for txn in corpus.world.bank_txns}
        for bait in baits:
            host = hosts[bait.merchant_id]
            tranche = credits[host.credit_ids[0]].credit_minor
            band = max(100, bait.net_minor * 50 // 10_000)
            assert tranche != bait.net_minor
            assert abs(tranche - bait.net_minor) <= band
            # A fraction of the host's net, so the host never claims it back.
            assert tranche * 2 < host.net_minor

    def test_the_baits_own_payout_is_outside_t3s_window(self, corpus):
        """Which is what leaves the lookalike alone in the pool."""
        credits = {txn.txn_id: txn for txn in corpus.world.bank_txns}
        settlements = {s.settlement_id: s for s in corpus.world.settlements}
        for record in corpus.by_case()[Case.UNREF_TRANCHE_BAIT]:
            txn = credits[record.credit_ids[0]]
            gap = (txn.value_date - settlements[record.settlement_id].settled_on).days
            assert gap > 7

    def test_the_unreferenced_credits_carry_no_recoverable_reference(self, corpus):
        for record in corpus.cases:
            if record.referenced:
                continue
            for txn_id in record.credit_ids:
                txn = next(t for t in corpus.world.bank_txns if t.txn_id == txn_id)
                assert "UTR" not in txn.narration

    def test_the_referenced_credits_do(self, corpus):
        settlements = {s.settlement_id: s for s in corpus.world.settlements}
        for record in corpus.cases:
            if not record.referenced:
                continue
            for txn_id in record.credit_ids:
                txn = next(t for t in corpus.world.bank_txns if t.txn_id == txn_id)
                assert settlements[record.settlement_id].utr in txn.narration


class TestReproducibility:
    def test_the_same_seed_produces_the_same_world(self):
        left = build_adversarial_corpus(43)
        right = build_adversarial_corpus(43)
        assert [txn.credit_minor for txn in left.world.bank_txns] == [
            txn.credit_minor for txn in right.world.bank_txns
        ]
        assert left.truth.links == right.truth.links

    def test_the_seed_moves_the_money_but_not_the_layout(self):
        """So a re-run at another seed asks the same questions of different money,
        and the per-case outcome table stays comparable across seeds."""
        left = build_adversarial_corpus(42)
        right = build_adversarial_corpus(43)
        assert [record.case for record in left.cases] == [
            record.case for record in right.cases
        ]
        assert [record.net_minor for record in left.cases] != [
            record.net_minor for record in right.cases
        ]

    def test_it_emits_the_same_five_files_model_a_does(self, tmp_path):
        corpus = write_adversarial_corpus(tmp_path / "adv", 42, merchants=2)
        written = {path.name for path in (tmp_path / "adv").iterdir()}
        assert written == {
            "ledger_orders.csv",
            "psp_settlements.json",
            "bank_statement.csv",
            "ground_truth_links.csv",
            "ground_truth_records.csv",
            "manifest.json",
        }
        assert corpus.conservation_residual_minor == 0
