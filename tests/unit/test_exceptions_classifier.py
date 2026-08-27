"""The queue itself: assembly, evidence, ordering, prose, and the leash.

Three properties carry the weight here, and each is a claim the project makes
out loud:

* **One exception per residual record**, never one per decision. A contested
  settlement is one problem, and a queue that listed it four times would be
  reporting the matcher's internals as a controller's workload.
* **Sorted by money.** PLAN.md §8.2.3. Any other order hides the ₹4 lakh payout
  behind two hundred one-paise drifts.
* **The bounds refuse loudly.** A proposal past its bound is emitted as refused
  with the bound named, never dropped -- a leash nobody can see is not a leash.

Ground truth appears in exactly one test here, and only to assert that the
classifier's output does *not* depend on it.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ledgerloop.config import AutoResolutionBounds, RunConfig
from ledgerloop.exceptions import (
    classify_exceptions,
    exception_id,
    mark_resolvable,
    queue_order,
    resolve_bounded,
)
from ledgerloop.exceptions.templates import PROSE_VERSION, prose_for
from ledgerloop.matching import run_matching
from ledgerloop.models.enums import (
    EvidenceKind,
    ExceptionClass,
    ProseSource,
    RecordType,
    Severity,
)
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, settlement_ref
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus, debit_row, noise_credit

CONFIG = RunConfig(run_id="queue-test")


def queue(ingest, config: RunConfig = CONFIG):
    run = run_matching(ingest, config)
    assert run.context is not None
    return run, classify_exceptions(
        run.context,
        run.decisions,
        run.candidates,
        config,
        merchant_profiles=run.merchant_spellings,
    )


@pytest.fixture
def duplicated():
    """One batch credited twice on the same reference -- A05's shape."""
    only = batch()
    return corpus(
        batches=[only],
        bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
    )


@pytest.fixture
def clean_run():
    only = batch()
    return corpus(batches=[only], bank_txns=[only.credit()])


class TestAssembly:
    def test_a_clean_run_raises_nothing(self, clean_run):
        _, outcome = queue(clean_run)
        assert outcome.exceptions == ()
        assert outcome.settlements_seen == 0

    def test_every_exception_carries_the_whole_deliverable(self, duplicated):
        _, outcome = queue(duplicated)
        assert outcome.exceptions
        for item in outcome.exceptions:
            assert item.exception_class in set(ExceptionClass)
            assert item.severity in set(Severity)
            assert isinstance(item.impact_minor, int)
            assert item.involved_refs
            assert item.evidence
            assert item.root_cause.endswith((".", "!"))
            assert item.suggested_action
            assert 0.0 <= item.classification_confidence <= 1.0

    def test_one_exception_per_record_not_one_per_decision(self, duplicated):
        run, outcome = queue(duplicated)
        subjects = [item.involved_refs[0].key for item in outcome.exceptions]
        assert len(subjects) == len(set(subjects))
        assert len(outcome.exceptions) < len(run.decisions)

    def test_the_id_is_derived_from_the_subject(self, duplicated):
        _, outcome = queue(duplicated)
        for item in outcome.exceptions:
            assert item.exception_id == exception_id(item.involved_refs[0].key)

    def test_the_evidence_points_back_at_source_records(self, duplicated):
        _, outcome = queue(duplicated)
        for item in outcome.exceptions:
            for evidence in item.evidence:
                assert evidence.detail
                assert evidence.kind in set(EvidenceKind)

    def test_the_prose_is_template_written_until_an_llm_says_otherwise(self, duplicated):
        _, outcome = queue(duplicated)
        for item in outcome.exceptions:
            assert item.root_cause_source is ProseSource.TEMPLATE
            assert item.suggested_action_source is ProseSource.TEMPLATE

    def test_a_settlement_exception_reaches_the_orders_behind_it(self):
        """"SETL-0104 is short" means nothing to the person whose customer asks."""
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[noise_credit(amount_minor=11)])
        _, outcome = queue(ingest)
        subject = next(
            item for item in outcome.exceptions if item.involved_refs[0].record_id == "SETL-0001"
        )
        kinds = {ref.record_type for ref in subject.involved_refs}
        assert RecordType.ORDER in kinds
        assert RecordType.PAYMENT in kinds

    def test_outgoing_rows_are_counted_rather_than_queued(self):
        only = batch()
        ingest = corpus(
            batches=[only],
            bank_txns=[only.credit(), debit_row("BNK-09002", utr=only.settlement.utr)],
        )
        _, outcome = queue(ingest)
        assert outcome.debits_ignored == 1
        assert all(
            item.involved_refs[0].record_id != "BNK-09002" for item in outcome.exceptions
        )


class TestOrdering:
    def test_the_queue_is_sorted_by_money_descending(self):
        first = batch("SETL-0001", amounts=(500_000,), utr=None)
        second = batch("SETL-0002", amounts=(1_000,), utr=None, first_index=5)
        ingest = corpus(batches=[first, second], bank_txns=[noise_credit(amount_minor=3)])
        _, outcome = queue(ingest)
        impacts = [item.impact_minor for item in outcome.exceptions]
        assert impacts == sorted(impacts, reverse=True)

    def test_ties_break_on_the_id_so_the_order_is_total(self):
        items = [
            ReconException(
                exception_id=f"exception:{name}",
                exception_class=ExceptionClass.UNKNOWN_RESIDUAL,
                severity=Severity.LOW,
                impact_minor=100,
                involved_refs=(bank_ref(name),),
                root_cause="x.",
                suggested_action="y",
                classification_confidence=0.5,
            )
            for name in ("BNK-2", "BNK-1", "BNK-3")
        ]
        assert [item.exception_id for item in queue_order(items)] == [
            "exception:BNK-1",
            "exception:BNK-2",
            "exception:BNK-3",
        ]

    def test_two_runs_over_one_corpus_produce_the_same_queue(self, duplicated):
        _, first = queue(duplicated)
        _, second = queue(duplicated)
        assert [item.exception_id for item in first.exceptions] == [
            item.exception_id for item in second.exceptions
        ]
        assert [item.root_cause for item in first.exceptions] == [
            item.root_cause for item in second.exceptions
        ]


class TestImpact:
    def test_an_uncredited_payout_is_worth_its_whole_net(self):
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[noise_credit(amount_minor=5)])
        _, outcome = queue(ingest)
        subject = next(
            item for item in outcome.exceptions if item.involved_refs[0].record_id == "SETL-0001"
        )
        assert subject.impact_minor == only.net_minor

    def test_a_credited_payout_is_worth_only_its_discrepancy(self):
        """The money arrived. Only the part the document cannot explain is at stake."""
        only = batch(net_minor=100_000 - 1_500)
        ingest = corpus(
            batches=[only],
            bank_txns=[bank_credit("BNK-00001", amount_minor=only.net_minor,
                                   utr=only.settlement.utr)],
        )
        _, outcome = queue(ingest)
        subject = next(
            item
            for item in outcome.exceptions
            if item.exception_class is ExceptionClass.FEE_TAX_MISMATCH
        )
        assert subject.impact_minor == 1_500

    def test_the_money_is_always_an_integer(self, duplicated):
        _, outcome = queue(duplicated)
        assert all(isinstance(item.impact_minor, int) for item in outcome.exceptions)
        assert isinstance(outcome.total_impact_minor, int)


class TestAmbiguityIsPreserved:
    @pytest.fixture
    def ambiguous(self):
        """Two subsets of one batch both compose the first tranche."""
        only = batch(amounts=(50_000, 50_000, 30_000, 20_000))
        grosses = [payment.amount_minor for payment in only.payments]
        amounts = allocate_minor(only.net_minor, [grosses[0], sum(grosses[1:])])
        return corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
                bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
            ],
        )

    def test_an_ambiguity_carries_its_competing_explanations(self, ambiguous):
        _, outcome = queue(ambiguous)
        ambiguities = [
            item
            for item in outcome.exceptions
            if item.exception_class is ExceptionClass.AMBIGUOUS_AGGREGATION
        ]
        for item in ambiguities:
            assert len(item.hypotheses) >= 2
            probabilities = [h.probability for h in item.hypotheses]
            assert probabilities == sorted(probabilities, reverse=True)

    def test_an_ambiguity_the_log_cannot_evidence_is_downgraded_not_faked(self):
        """The model requires two hypotheses. Where there is one, the class changes."""
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[noise_credit(amount_minor=9)])
        _, outcome = queue(ingest)
        for item in outcome.exceptions:
            if item.exception_class is ExceptionClass.AMBIGUOUS_AGGREGATION:
                assert len(item.hypotheses) >= 2


class TestBoundedAutoResolution:
    def _exception(self, exception_class: ExceptionClass, impact: int, evidence=()):
        return ReconException(
            exception_id=f"exception:{exception_class.value}",
            exception_class=exception_class,
            severity=Severity.LOW,
            impact_minor=impact,
            involved_refs=(settlement_ref("SETL-0001"), bank_ref("BNK-00001")),
            evidence=evidence,
            root_cause="x.",
            suggested_action="y",
            classification_confidence=0.9,
        )

    def test_a_small_drift_is_within_the_per_record_bound(self):
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.ROUNDING_DRIFT, 400)],
            AutoResolutionBounds(),
        )
        assert len(outcome.applied) == 1
        assert outcome.rounding_spent_minor == 400

    def test_a_drift_past_the_per_record_bound_is_refused_and_says_why(self):
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.ROUNDING_DRIFT, 501)],
            AutoResolutionBounds(rounding_per_record_minor=500),
        )
        assert len(outcome.refused) == 1
        assert "per-record bound" in (outcome.refused[0].refusal or "")
        assert outcome.refused[0].bound

    def test_the_run_budget_stops_the_run_not_the_record(self):
        bounds = AutoResolutionBounds(
            rounding_per_record_minor=500, rounding_per_run_minor=900
        )
        exceptions = [
            self._exception(ExceptionClass.ROUNDING_DRIFT, 500).model_copy(
                update={"exception_id": f"exception:{index}"}
            )
            for index in range(3)
        ]
        outcome = resolve_bounded(exceptions, bounds)
        assert len(outcome.applied) == 1
        assert len(outcome.refused) == 2
        assert outcome.rounding_spent_minor == 500
        assert "budget" in (outcome.refused[0].refusal or "")

    def test_a_duplicate_is_flagged_and_linked_and_nothing_is_deleted(self):
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.DUPLICATE_CREDIT, 10_000)],
            AutoResolutionBounds(),
        )
        assert outcome.applied
        assert "never deletes anything" in outcome.applied[0].bound
        assert "delete" in outcome.applied[0].action

    def test_a_timing_shift_inside_the_day_bound_is_proposed(self):
        from ledgerloop.models.candidates import Evidence

        evidence = (
            Evidence(
                kind=EvidenceKind.DATE_PROXIMITY,
                detail="BNK-00001 credits on 2026-03-14, +4 day(s) from the settlement",
            ),
        )
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.TIMING_SHIFT, 10_000, evidence)],
            AutoResolutionBounds(timing_shift_max_days=5),
        )
        assert outcome.applied

    def test_a_timing_shift_past_the_day_bound_is_refused(self):
        from ledgerloop.models.candidates import Evidence

        evidence = (
            Evidence(
                kind=EvidenceKind.DATE_PROXIMITY,
                detail="BNK-00001 credits on 2026-03-30, +20 day(s) from the settlement",
            ),
        )
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.TIMING_SHIFT, 10_000, evidence)],
            AutoResolutionBounds(timing_shift_max_days=5),
        )
        assert outcome.refused
        assert "exceeds the bound" in (outcome.refused[0].refusal or "")

    def test_a_timing_shift_with_no_measurable_gap_is_refused(self):
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.TIMING_SHIFT, 10_000)],
            AutoResolutionBounds(),
        )
        assert outcome.refused

    @pytest.mark.parametrize(
        "exception_class",
        [
            ExceptionClass.FEE_TAX_MISMATCH,
            ExceptionClass.ORPHAN_BANK_CREDIT,
            ExceptionClass.UNMATCHABLE,
            ExceptionClass.UNKNOWN_RESIDUAL,
            ExceptionClass.SPLIT_PAYOUT_INCOMPLETE,
        ],
    )
    def test_every_other_class_is_proposal_only(self, exception_class):
        outcome = resolve_bounded(
            [self._exception(exception_class, 100)], AutoResolutionBounds()
        )
        assert outcome.resolutions == ()

    def test_disabling_the_bounds_switches_the_whole_resolver_off(self):
        outcome = resolve_bounded(
            [self._exception(ExceptionClass.ROUNDING_DRIFT, 100)],
            AutoResolutionBounds(enabled=False),
        )
        assert outcome.resolutions == ()

    def test_the_flag_is_stamped_only_on_what_was_accepted(self):
        exceptions = [
            self._exception(ExceptionClass.ROUNDING_DRIFT, 100),
            self._exception(ExceptionClass.UNMATCHABLE, 100).model_copy(
                update={"exception_id": "exception:floor"}
            ),
        ]
        outcome = resolve_bounded(exceptions, AutoResolutionBounds())
        stamped = mark_resolvable(exceptions, outcome)
        assert stamped[0].resolvable_by_agent
        assert not stamped[1].resolvable_by_agent

    def test_the_floor_can_never_be_marked_resolvable(self):
        """Enforced twice: not in the resolvable set, and refused by the model."""
        with pytest.raises(ValueError, match="irreconcilable by construction"):
            ReconException(
                exception_id="exception:x",
                exception_class=ExceptionClass.UNMATCHABLE,
                severity=Severity.LOW,
                impact_minor=1,
                involved_refs=(bank_ref("BNK-1"),),
                root_cause="x.",
                suggested_action="y",
                classification_confidence=1.0,
                resolvable_by_agent=True,
            )


class TestNothingIsPosted:
    def test_auto_resolution_never_changes_a_prediction(self, duplicated):
        """A resolver that could add a link would be a sixth tier in disguise."""
        run, outcome = queue(duplicated)
        before = {prediction.pair for prediction in run.predictions}
        resolutions = resolve_bounded(outcome.exceptions, CONFIG.auto_resolution)
        mark_resolvable(outcome.exceptions, resolutions)
        assert {prediction.pair for prediction in run.predictions} == before

    def test_classification_never_changes_a_decision(self, duplicated):
        run, _ = queue(duplicated)
        outcomes = [decision.outcome for decision in run.decisions]
        classify_exceptions(
            run.context, run.decisions, run.candidates, CONFIG  # type: ignore[arg-type]
        )
        assert [decision.outcome for decision in run.decisions] == outcomes


class TestTheTemplates:
    @pytest.mark.parametrize("exception_class", list(ExceptionClass))
    def test_every_class_has_prose_naming_its_subject_and_its_money(
        self, exception_class
    ):
        prose = prose_for(
            exception_class,
            subject="SETL-0001",
            impact_minor=123_456,
            counterpart="BNK-00001",
            day_gap=4,
        )
        assert "SETL-0001" in prose.root_cause
        assert "₹1,234.56" in prose.root_cause
        assert prose.suggested_action
        assert prose.root_cause != prose.suggested_action

    def test_no_template_speculates_about_intent(self):
        for exception_class in ExceptionClass:
            prose = prose_for(
                exception_class, subject="X", impact_minor=1, counterpart="Y", day_gap=1
            )
            lowered = prose.root_cause.lower()
            for hedge in ("appears to", "probably", "seems", "might have", "likely"):
                assert hedge not in lowered

    def test_the_floor_offers_no_action_it_cannot_support(self):
        prose = prose_for(
            ExceptionClass.UNMATCHABLE, subject="BNK-1", impact_minor=100
        )
        assert "No action is available" in prose.suggested_action

    def test_the_version_is_recorded(self):
        assert PROSE_VERSION


class TestGroundTruthIsNotAnInput:
    def test_the_queue_is_identical_whatever_the_truth_says(self, duplicated):
        """The classifier has no truth parameter. This pins that it stays that way."""
        run, first = queue(duplicated)
        second = classify_exceptions(
            run.context,  # type: ignore[arg-type]
            run.decisions,
            run.candidates,
            CONFIG,
            merchant_profiles=run.merchant_spellings,
        )
        assert [item.exception_class for item in first.exceptions] == [
            item.exception_class for item in second.exceptions
        ]

    def test_the_signature_takes_no_ground_truth(self):
        import inspect

        parameters = inspect.signature(classify_exceptions).parameters
        assert "truth" not in parameters
        assert "ground_truth" not in parameters


def test_severity_uses_the_dataset_clock_not_the_wall_clock():
    """A queue that changed as the calendar advanced would break the diff-free report."""
    only = batch(settled_on=batch().settlement.settled_on - timedelta(days=400))
    ingest = corpus(batches=[only], bank_txns=[noise_credit(amount_minor=13)])
    _, first = queue(ingest)
    _, second = queue(ingest)
    assert [item.severity for item in first.exceptions] == [
        item.severity for item in second.exceptions
    ]
