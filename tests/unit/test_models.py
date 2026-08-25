"""Contract tests for the Pydantic models.

These assert the *invariants encoded in the schema* -- the ones that must hold
however the rest of the system is written, because they are enforced at the
type boundary rather than by convention.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError

from ledgerloop.models import (
    AuditEvent,
    AuditEventType,
    CanonicalBankTxn,
    CanonicalOrder,
    CanonicalPayment,
    CanonicalSettlement,
    Currency,
    DecisionOutcome,
    Evidence,
    EvidenceKind,
    ExceptionClass,
    FeatureVector,
    Hypothesis,
    LinkType,
    MatchCandidate,
    MatchDecision,
    OrderStatus,
    ReconException,
    RecordRef,
    RecordType,
    Severity,
    Tier,
    bank_ref,
    order_ref,
    payment_ref,
    settlement_ref,
)

NOW = datetime(2026, 3, 4, 11, 22, 9, tzinfo=UTC)


# ----------------------------------------------------------------------
# RecordRef
# ----------------------------------------------------------------------


class TestRecordRef:
    def test_key_round_trip(self):
        ref = payment_ref("PAY-88301")
        assert ref.key == "payment:PAY-88301"
        assert RecordRef.parse(ref.key) == ref

    def test_is_hashable_for_use_as_a_set_member(self):
        """The subset-sum solver and the graph rules need set membership."""
        assert len({payment_ref("PAY-1"), payment_ref("PAY-1"), payment_ref("PAY-2")}) == 2

    def test_type_prevents_cross_type_confusion(self):
        """`ORD-1` as an order is not `ORD-1` as a payment."""
        assert order_ref("X-1") != payment_ref("X-1")

    def test_rejects_empty_id(self):
        with pytest.raises(ValidationError):
            RecordRef(record_type=RecordType.ORDER, record_id="   ")

    def test_rejects_separator_in_id(self):
        with pytest.raises(ValidationError, match="ambiguous"):
            RecordRef(record_type=RecordType.ORDER, record_id="ORD:1")

    def test_parse_rejects_unkeyed_string(self):
        with pytest.raises(ValueError, match="not a record key"):
            RecordRef.parse("PAY-88301")

    def test_str_is_the_key(self):
        """Refs land in log lines and Cytoscape node ids as their key."""
        assert str(payment_ref("PAY-88301")) == "payment:PAY-88301"

    def test_every_record_type_has_a_constructor(self):
        assert order_ref("O").record_type is RecordType.ORDER
        assert payment_ref("P").record_type is RecordType.PAYMENT
        assert settlement_ref("S").record_type is RecordType.SETTLEMENT
        assert bank_ref("B").record_type is RecordType.BANK_TXN


# ----------------------------------------------------------------------
# Money fields at the schema boundary
# ----------------------------------------------------------------------


class TestMoneyFieldsRejectFloats:
    def test_order_amount_rejects_float(self):
        with pytest.raises(ValidationError, match="float is forbidden"):
            CanonicalOrder(
                order_id="ORD-1",
                merchant_id="MRCH_0007",
                customer_ref="CUST_1",
                amount_minor=4999.0,
                booked_at=NOW,
                status=OrderStatus.CAPTURED,
            )

    def test_order_amount_rejects_bool(self):
        with pytest.raises(ValidationError, match="bool is not a money value"):
            CanonicalOrder(
                order_id="ORD-1",
                merchant_id="MRCH_0007",
                customer_ref="CUST_1",
                amount_minor=True,
                booked_at=NOW,
                status=OrderStatus.CAPTURED,
            )

    def test_money_survives_a_json_round_trip_exactly(self):
        order = CanonicalOrder(
            order_id="ORD-2026-004821",
            merchant_id="MRCH_0007",
            customer_ref="CUST_11902",
            amount_minor=499900,
            booked_at=NOW,
            status=OrderStatus.CAPTURED,
        )
        restored = CanonicalOrder.model_validate_json(order.model_dump_json())
        assert restored == order
        assert isinstance(restored.amount_minor, int)


class TestCanonicalRecordsExposeTheirRef:
    """Every record self-addresses, so nothing has to build keys by hand."""

    def test_order(self):
        order = CanonicalOrder(
            order_id="ORD-1",
            merchant_id="M",
            customer_ref="C",
            amount_minor=1,
            booked_at=NOW,
            status=OrderStatus.CAPTURED,
        )
        assert order.ref.key == "order:ORD-1"

    def test_payment(self):
        payment = CanonicalPayment(payment_id="PAY-1", amount_minor=1, captured_at=NOW)
        assert payment.ref.key == "payment:PAY-1"

    def test_settlement(self):
        settlement = CanonicalSettlement(
            settlement_id="SETL-1",
            settled_on=date(2026, 3, 6),
            gross_minor=100,
            fee_minor=0,
            tax_minor=0,
            net_minor=100,
        )
        assert settlement.ref.key == "settlement:SETL-1"

    def test_bank_txn(self):
        txn = CanonicalBankTxn(
            txn_id="BNK-1",
            value_date=date(2026, 3, 6),
            narration_raw="NEFT CR",
            credit_minor=100,
        )
        assert txn.ref.key == "bank_txn:BNK-1"


class TestMoneyFieldCoverage:
    """Structural guard: every `_minor` field really is a guarded money field.

    Catches the failure mode where someone adds `refund_minor: float` or a
    plain `int` to a model six weeks from now. Convention would not catch it;
    reflection over the schema does.
    """

    MODELS: ClassVar[list[type[BaseModel]]] = [
        CanonicalOrder,
        CanonicalPayment,
        CanonicalSettlement,
        CanonicalBankTxn,
        Evidence,
        FeatureVector,
        MatchCandidate,
        ReconException,
    ]

    @staticmethod
    def _is_guarded(field) -> bool:
        from typing import get_args

        from ledgerloop.models.base import _MinorUnitsAnnotation

        def search(node: object) -> bool:
            if isinstance(node, _MinorUnitsAnnotation):
                return True
            return any(search(arg) for arg in get_args(node))

        if any(isinstance(m, _MinorUnitsAnnotation) for m in field.metadata):
            return True
        return search(field.annotation)

    @pytest.mark.parametrize("model", MODELS)
    def test_every_minor_suffixed_field_is_guarded(self, model):
        offenders = [
            name
            for name, field in model.model_fields.items()
            if name.endswith("_minor") and not self._is_guarded(field)
        ]
        assert not offenders, f"{model.__name__}: unguarded money fields {offenders}"

    @pytest.mark.parametrize("model", MODELS)
    def test_no_money_field_is_typed_float(self, model):
        offenders = [
            name
            for name, field in model.model_fields.items()
            if name.endswith("_minor") and field.annotation is float
        ]
        assert not offenders, f"{model.__name__}: float money fields {offenders}"


# ----------------------------------------------------------------------
# Canonical records
# ----------------------------------------------------------------------


class TestCanonicalSettlement:
    def _settlement(self, **overrides) -> CanonicalSettlement:
        kwargs = {
            "settlement_id": "SETL-0091",
            "utr": "UTR2026030412345",
            "settled_on": date(2026, 3, 6),
            "gross_minor": 4_210_900,
            "fee_minor": 84_218,
            "tax_minor": 15_159,
            "adjustments_minor": -431_200,
            "net_minor": 3_680_323,
        }
        kwargs.update(overrides)
        return CanonicalSettlement(**kwargs)

    def test_consistent_settlement_has_zero_delta(self):
        assert self._settlement().net_delta_minor == 0

    def test_fee_tax_mismatch_is_reported_not_rejected(self):
        """A03 breaks this invariant on purpose -- a validator would delete the anomaly."""
        settlement = self._settlement(net_minor=3_680_000)
        assert settlement.net_delta_minor == -323
        assert settlement.expected_net_minor == 3_680_323

    def test_negative_adjustments_are_allowed(self):
        """Chargebacks (A08) and post-settlement refunds (A06) are negative."""
        assert self._settlement().adjustments_minor == -431_200


class TestCanonicalBankTxn:
    def _txn(self, **overrides) -> CanonicalBankTxn:
        kwargs = {
            "txn_id": "BNK-77120",
            "value_date": date(2026, 3, 6),
            "narration_raw": "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT",
            "credit_minor": 3_680_323,
        }
        kwargs.update(overrides)
        return CanonicalBankTxn(**kwargs)

    def test_credit_is_positive_signed(self):
        assert self._txn().signed_amount_minor == 3_680_323
        assert self._txn().is_credit is True

    def test_debit_is_negative_signed(self):
        txn = self._txn(credit_minor=0, debit_minor=50_000)
        assert txn.signed_amount_minor == -50_000
        assert txn.is_credit is False

    def test_missing_reference_leaves_extraction_none(self):
        """A07 MISSING_REFERENCE: no UTR anywhere in the narration."""
        txn = self._txn(narration_raw="NEFT CR-SOME COUNTERPARTY-PAYMENT")
        assert txn.extracted_utr is None


class TestCanonicalPayment:
    def test_raw_and_normalized_order_refs_are_both_kept(self):
        """Keeping both is what lets the audit trail explain why T0 missed."""
        payment = CanonicalPayment(
            payment_id="PAY-88301",
            settlement_id="SETL-0091",
            order_ref_raw="ord 2026 004821",
            order_ref_normalized="ORD-2026-004821",
            amount_minor=499_900,
            captured_at=NOW,
        )
        assert payment.order_ref_raw != payment.order_ref_normalized

    def test_null_order_ref_is_permitted(self):
        payment = CanonicalPayment(
            payment_id="PAY-88302", amount_minor=100, captured_at=NOW
        )
        assert payment.order_ref_raw is None
        assert payment.settlement_id is None


class TestUnsupportedCurrency:
    def test_usd_is_declared_but_flagged_unsupported(self):
        """A11 FX is cut. The enum member exists so the cut is explicit and testable."""
        assert Currency.USD.supported is False
        assert Currency.INR.supported is True


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------


def _features(tier: Tier = Tier.T0_EXACT, **overrides) -> FeatureVector:
    return FeatureVector(tier=tier, **overrides)


def _candidate(**overrides) -> MatchCandidate:
    kwargs = {
        "candidate_id": "CAND-1",
        "link_type": LinkType.PAYMENT_CREDITED_AS,
        "source_ref": payment_ref("PAY-88301"),
        "target_ref": bank_ref("BNK-77120"),
        "tier": Tier.T0_EXACT,
        "features": _features(),
    }
    kwargs.update(overrides)
    return MatchCandidate(**kwargs)


class TestMatchCandidate:
    def test_tier_must_agree_with_feature_tier(self):
        with pytest.raises(ValidationError, match="disagrees with feature tier"):
            _candidate(tier=Tier.T2_AGGREGATION, features=_features(Tier.T0_EXACT))

    def test_subset_members_require_t2(self):
        with pytest.raises(ValidationError, match="only meaningful for T2"):
            _candidate(subset_members=(payment_ref("PAY-1"),))

    def test_subset_size_must_match_member_count(self):
        with pytest.raises(ValidationError, match="disagrees with"):
            _candidate(
                tier=Tier.T2_AGGREGATION,
                features=_features(Tier.T2_AGGREGATION, subset_size=3),
                subset_members=(payment_ref("PAY-1"), payment_ref("PAY-2")),
            )

    def test_valid_t2_candidate(self):
        candidate = _candidate(
            tier=Tier.T2_AGGREGATION,
            features=_features(Tier.T2_AGGREGATION, subset_size=2),
            subset_members=(payment_ref("PAY-1"), payment_ref("PAY-2")),
        )
        assert candidate.features.subset_size == 2

    def test_only_payment_credited_as_links_are_evaluated(self):
        """The atomic unit of evaluation -- ARCHITECTURE.md §2."""
        assert _candidate().is_evaluable is True
        assert (
            _candidate(
                link_type=LinkType.PAYMENT_SETTLED_IN,
                target_ref=settlement_ref("SETL-1"),
            ).is_evaluable
            is False
        )

    def test_scores_start_unset(self):
        candidate = _candidate()
        assert candidate.raw_score is None
        assert candidate.calibrated_p is None
        assert candidate.arithmetic_verified is False

    def test_pair_is_comparable_against_ground_truth(self):
        """The candidate's pair must be the same shape as GroundTruth.evaluation_pairs."""
        assert _candidate().pair == ("payment:PAY-88301", "bank_txn:BNK-77120")

    def test_truth_label_defaults_to_none(self):
        """Must stay None during evaluation; only training and calibration set it."""
        assert _candidate().is_truth_positive is None

    def test_llm_confidence_is_optional_and_bounded(self):
        with pytest.raises(ValidationError):
            _features(Tier.T5_LLM, llm_confidence=1.4)


class TestTierSemantics:
    def test_t0_and_t1_bypass_the_blender(self):
        assert Tier.T0_EXACT.is_deterministic_certain
        assert Tier.T1_TOLERANCE.is_deterministic_certain

    @pytest.mark.parametrize(
        "tier", [Tier.T2_AGGREGATION, Tier.T3_FUZZY, Tier.T4_GRAPH, Tier.T5_LLM]
    )
    def test_residual_tiers_are_blended_and_calibrated(self, tier: Tier):
        assert not tier.is_deterministic_certain

    def test_tiers_order_cheapest_first(self):
        assert Tier.T0_EXACT < Tier.T2_AGGREGATION < Tier.T5_LLM


# ----------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------


def _decision(**overrides) -> MatchDecision:
    kwargs = {
        "decision_id": "DEC-1",
        "candidate_id": "CAND-1",
        "link_type": LinkType.PAYMENT_CREDITED_AS,
        "source_ref": payment_ref("PAY-88301"),
        "target_ref": bank_ref("BNK-77120"),
        "tier": Tier.T0_EXACT,
        "outcome": DecisionOutcome.AUTO_MATCHED,
        "calibrated_p": 0.99,
        "arithmetic_verified": True,
        "decided_at": NOW,
        "reason": "p=0.99 >= tau_high=0.95",
    }
    kwargs.update(overrides)
    return MatchDecision(**kwargs)


class TestMatchDecision:
    def test_auto_match_requires_verified_arithmetic(self):
        """The most expensive failure mode, blocked at the type boundary."""
        with pytest.raises(ValidationError, match="requires arithmetic_verified"):
            _decision(arithmetic_verified=False)

    def test_unverified_link_may_still_be_referred_for_review(self):
        decision = _decision(
            outcome=DecisionOutcome.NEEDS_REVIEW,
            arithmetic_verified=False,
            reason="demoted: verify_arithmetic failed",
        )
        assert decision.outcome is DecisionOutcome.NEEDS_REVIEW

    def test_needs_review_is_not_a_positive_prediction(self):
        """Counting referrals as matches is the precision-inflating trap."""
        assert _decision().is_positive_prediction is True
        assert (
            _decision(
                outcome=DecisionOutcome.NEEDS_REVIEW, arithmetic_verified=False
            ).is_positive_prediction
            is False
        )
        assert (
            _decision(
                outcome=DecisionOutcome.EXCEPTION, arithmetic_verified=False
            ).is_positive_prediction
            is False
        )

    def test_pair_matches_the_candidate_pair(self):
        """Decision and candidate address the same edge the same way."""
        assert _decision().pair == _candidate().pair

    def test_decisions_are_immutable(self):
        """Append-only audit: a revision writes a new record."""
        decision = _decision()
        with pytest.raises(ValidationError):
            decision.outcome = DecisionOutcome.EXCEPTION


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------


def _exception(**overrides) -> ReconException:
    kwargs = {
        "exception_id": "EXC-1",
        "exception_class": ExceptionClass.CHARGEBACK_NETTED,
        "severity": Severity.HIGH,
        "impact_minor": 431_200,
        "involved_refs": (settlement_ref("SETL-0091"),),
        "root_cause": "A chargeback of ₹4,312.00 was netted off payout SETL-0091.",
        "suggested_action": "Request chargeback detail for SETL-0091.",
        "classification_confidence": 0.88,
    }
    kwargs.update(overrides)
    return ReconException(**kwargs)


class TestReconException:
    def test_is_a_data_record_not_a_raisable_error(self):
        """Named off the builtin, and deliberately not a BaseException."""
        assert not issubclass(ReconException, BaseException)

    def test_requires_at_least_one_involved_record(self):
        with pytest.raises(ValidationError):
            _exception(involved_refs=())

    def test_ambiguity_requires_two_hypotheses(self):
        """An ambiguity with one explanation is not an ambiguity."""
        with pytest.raises(ValidationError, match="at least two competing hypotheses"):
            _exception(
                exception_class=ExceptionClass.AMBIGUOUS_AGGREGATION,
                hypotheses=(Hypothesis(summary="subset A", probability=0.5),),
            )

    def test_ambiguity_with_two_hypotheses_is_accepted(self):
        exc = _exception(
            exception_class=ExceptionClass.AMBIGUOUS_AGGREGATION,
            hypotheses=(
                Hypothesis(summary="subset A", probability=0.55),
                Hypothesis(summary="subset B", probability=0.45),
            ),
        )
        assert len(exc.hypotheses) == 2

    def test_hypotheses_must_be_ordered_by_descending_probability(self):
        with pytest.raises(ValidationError, match="descending probability"):
            _exception(
                exception_class=ExceptionClass.AMBIGUOUS_AGGREGATION,
                hypotheses=(
                    Hypothesis(summary="subset A", probability=0.4),
                    Hypothesis(summary="subset B", probability=0.6),
                ),
            )

    def test_unmatchable_can_never_be_agent_resolvable(self):
        """The honest floor: the data to resolve it does not exist."""
        with pytest.raises(ValidationError, match="irreconcilable by construction"):
            _exception(
                exception_class=ExceptionClass.UNMATCHABLE, resolvable_by_agent=True
            )

    def test_prose_source_defaults_to_template(self):
        """--no-llm runs must be honestly labelled in the report."""
        exc = _exception()
        assert exc.root_cause_source.value == "TEMPLATE"
        assert exc.suggested_action_source.value == "TEMPLATE"


class TestTaxonomiesAreDistinct:
    def test_exception_classes_outnumber_anomaly_classes(self):
        """13 vs 11 -- the confusion matrix is rectangular, not square."""
        from ledgerloop.models import AnomalyClass

        assert len(AnomalyClass) == 11
        assert len(ExceptionClass) == 13

    def test_clean_is_an_anomaly_label_but_never_an_exception(self):
        assert "CLEAN" not in {e.name for e in ExceptionClass}

    def test_system_only_states_have_no_anomaly_counterpart(self):
        from ledgerloop.models import AnomalyClass

        anomaly_names = {a.name for a in AnomalyClass}
        assert "AMBIGUOUS_AGGREGATION" not in anomaly_names
        assert "UNKNOWN_RESIDUAL" not in anomaly_names
        assert "UNMATCHABLE" not in anomaly_names


# ----------------------------------------------------------------------
# Evidence and audit
# ----------------------------------------------------------------------


class TestEvidence:
    def test_evidence_points_back_at_source_records(self):
        """Every claim must be checkable rather than trusted."""
        evidence = Evidence(
            kind=EvidenceKind.SUBSET_SUM,
            detail="14 payments sum to ₹42,109.00 against a net credit of ₹36,803.23",
            refs=(settlement_ref("SETL-0091"), bank_ref("BNK-77120")),
            amount_minor=3_680_323,
        )
        assert len(evidence.refs) == 2
        assert evidence.amount_minor == 3_680_323

    def test_evidence_is_immutable(self):
        evidence = Evidence(kind=EvidenceKind.EXACT_KEY, detail="UTR matched")
        with pytest.raises(ValidationError):
            evidence.detail = "changed"


class TestAuditEvent:
    def test_jsonl_line_omits_unset_fields(self):
        event = AuditEvent(
            run_id="RUN-1",
            sequence=0,
            event_type=AuditEventType.RUN_STARTED,
            node="ingest",
            timestamp=NOW,
        )
        line = event.to_jsonl()
        assert "\n" not in line
        assert "prompt_hash" not in line

    def test_llm_provenance_is_recorded_when_present(self):
        event = AuditEvent(
            run_id="RUN-1",
            sequence=7,
            event_type=AuditEventType.LLM_CALL,
            node="llm_adjudicate",
            timestamp=NOW,
            prompt_hash="a1b2c3",
            provider="groq",
            prompt_tokens=1200,
            completion_tokens=180,
            latency_ms=340,
        )
        assert '"prompt_hash":"a1b2c3"' in event.to_jsonl()

    def test_events_are_immutable(self):
        event = AuditEvent(
            run_id="RUN-1",
            sequence=0,
            event_type=AuditEventType.RUN_STARTED,
            node="ingest",
            timestamp=NOW,
        )
        with pytest.raises(ValidationError):
            event.sequence = 1


class TestExtraFieldsAreForbidden:
    def test_unexpected_key_is_rejected(self):
        """LLM output is validated against these schemas; a hallucinated field
        must be caught, not silently ignored."""
        with pytest.raises(ValidationError):
            Evidence(kind=EvidenceKind.EXACT_KEY, detail="ok", hallucinated_field=1)
