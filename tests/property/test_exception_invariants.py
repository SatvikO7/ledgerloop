"""Property tests for the exception queue (PLAN.md §13).

Four statements that have to hold for every corpus, not just the ones somebody
wrote down. Each is a promise the report makes on the strength of the queue:

* **Every residual record is accounted for exactly once.** A record the ladder
  did not match either appears in the queue or is outside the unit; nothing
  falls between.
* **The queue is a total order by money.** The report prints it that way and a
  controller works down it.
* **Money is exact and non-negative.** Impact is a rupee figure a person acts
  on, so a negative or fractional one is not a bad metric, it is a broken
  contract.
* **The bounds hold whatever the queue contains.** No sequence of exceptions
  can push the resolver past its per-record or per-run leash.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ledgerloop.config import AutoResolutionBounds, RunConfig, SeverityThresholds
from ledgerloop.exceptions import (
    classify_exceptions,
    mark_resolvable,
    queue_order,
    resolve_bounded,
    severity_for,
)
from ledgerloop.exceptions.taxonomy import AGENT_RESOLVABLE_CLASSES
from ledgerloop.matching import run_matching
from ledgerloop.models.enums import ExceptionClass, RecordType, Severity
from ledgerloop.models.recon_exception import Hypothesis, ReconException
from ledgerloop.models.refs import bank_ref
from tests.unit.conftest import bank_credit, batch, corpus, debit_row, noise_credit

CONFIG = RunConfig(run_id="prop-exceptions")

amounts = st.integers(min_value=1_000, max_value=5_000_000)
impacts = st.integers(min_value=0, max_value=10**10)
classes = st.sampled_from(list(ExceptionClass))


def build(exception_class: ExceptionClass, impact: int, index: int) -> ReconException:
    """One synthetic queue row.

    ``AMBIGUOUS_AGGREGATION`` gets two hypotheses because the model refuses to
    exist without them -- an ambiguity with one explanation is not an ambiguity.
    That refusal is itself a Step 0 invariant, tested in ``test_models.py``;
    here it just has to be satisfied so the generator can reach every class.
    """
    hypotheses: tuple[Hypothesis, ...] = ()
    if exception_class is ExceptionClass.AMBIGUOUS_AGGREGATION:
        hypotheses = (
            Hypothesis(summary="first", probability=0.5),
            Hypothesis(summary="second", probability=0.5),
        )
    return ReconException(
        exception_id=f"exception:{index:04d}",
        exception_class=exception_class,
        severity=Severity.LOW,
        impact_minor=impact,
        involved_refs=(bank_ref(f"BNK-{index:05d}"), bank_ref(f"BNK-{index + 1:05d}")),
        root_cause="x.",
        suggested_action="y",
        classification_confidence=0.5,
        hypotheses=hypotheses,
    )


queues = st.lists(st.tuples(classes, impacts), min_size=0, max_size=25).map(
    lambda rows: [
        build(exception_class, impact, index)
        for index, (exception_class, impact) in enumerate(rows)
    ]
)


@st.composite
def corpora(draw):
    """A small three-way corpus with the knobs that drive the cascade."""
    gross = draw(st.lists(amounts, min_size=1, max_size=4))
    fee = draw(st.integers(min_value=0, max_value=5_000))
    adjustment = draw(st.sampled_from([0, -gross[0], -777]))
    keyed = draw(st.booleans())
    delta = draw(st.sampled_from([0, -3, 5_000, -5_000]))
    days = draw(st.integers(min_value=0, max_value=30))
    extra = draw(st.sampled_from(["none", "noise", "debit", "twin"]))

    only = batch(
        amounts=tuple(gross),
        fee_minor=fee,
        adjustments_minor=adjustment,
        utr="UTR2026031012345" if keyed else None,
    )
    rows = [
        bank_credit(
            "BNK-00001",
            amount_minor=max(1, only.net_minor + delta),
            utr=only.settlement.utr,
            value_date=only.settlement.settled_on.replace() + __import__(
                "datetime"
            ).timedelta(days=days),
        )
    ]
    if extra == "noise":
        rows.append(noise_credit(amount_minor=draw(amounts)))
    elif extra == "debit":
        rows.append(debit_row("BNK-09002", utr=only.settlement.utr))
    elif extra == "twin":
        rows.append(
            bank_credit(
                "BNK-00002",
                amount_minor=max(1, only.net_minor + delta),
                utr=only.settlement.utr,
            )
        )
    return only, corpus(batches=[only], bank_txns=rows)


def run_queue(ingest):
    run = run_matching(ingest, CONFIG)
    assert run.context is not None
    return run, classify_exceptions(
        run.context,
        run.decisions,
        run.candidates,
        CONFIG,
        merchant_profiles=run.merchant_spellings,
    )


class TestTheQueueAccountsForTheResidual:
    @given(corpora())
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_every_unmatched_credit_is_in_the_queue_or_outside_the_unit(self, made):
        _, ingest = made
        run, outcome = run_queue(ingest)
        matched = {
            decision.target_ref.record_id
            for decision in run.decisions
            if decision.is_positive_prediction
        }
        queued = {
            ref.record_id
            for item in outcome.exceptions
            for ref in item.involved_refs
            if ref.record_type is RecordType.BANK_TXN
        }
        for txn in run.context.bank_txns:  # type: ignore[union-attr]
            if not txn.is_credit:
                continue  # outgoing money is outside the unit, by design
            assert txn.txn_id in matched or txn.txn_id in queued

    @given(corpora())
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_record_is_the_subject_of_two_exceptions(self, made):
        _, ingest = made
        _, outcome = run_queue(ingest)
        subjects = [item.involved_refs[0].key for item in outcome.exceptions]
        assert len(subjects) == len(set(subjects))

    @given(corpora())
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_a_debit_is_never_queued(self, made):
        _, ingest = made
        _, outcome = run_queue(ingest)
        assert all(
            item.involved_refs[0].record_id != "BNK-09002"
            for item in outcome.exceptions
        )

    @given(corpora())
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_every_exception_is_complete_and_actionable(self, made):
        _, ingest = made
        _, outcome = run_queue(ingest)
        for item in outcome.exceptions:
            assert item.involved_refs
            assert item.evidence
            assert item.root_cause.strip()
            assert item.suggested_action.strip()
            assert item.impact_minor >= 0
            assert isinstance(item.impact_minor, int)

    @given(corpora())
    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_classifying_twice_gives_the_same_queue(self, made):
        _, ingest = made
        _, first = run_queue(ingest)
        _, second = run_queue(ingest)
        assert [item.exception_id for item in first.exceptions] == [
            item.exception_id for item in second.exceptions
        ]
        assert [item.exception_class for item in first.exceptions] == [
            item.exception_class for item in second.exceptions
        ]


class TestTheOrdering:
    @given(queues)
    def test_it_is_descending_by_money(self, items):
        ordered = queue_order(items)
        impacts = [item.impact_minor for item in ordered]
        assert impacts == sorted(impacts, reverse=True)

    @given(queues)
    def test_it_is_a_total_order(self, items):
        first = queue_order(items)
        second = queue_order(list(reversed(items)))
        assert [item.exception_id for item in first] == [
            item.exception_id for item in second
        ]

    @given(queues)
    def test_it_neither_adds_nor_drops_a_row(self, items):
        assert len(queue_order(items)) == len(items)


class TestSeverity:
    @given(impacts, st.integers(min_value=0, max_value=400))
    def test_it_is_always_one_of_the_four(self, impact, age):
        assert severity_for(
            impact, age_days=age, thresholds=SeverityThresholds()
        ) in set(Severity)

    @given(impacts, st.integers(min_value=0, max_value=400))
    def test_age_never_lowers_it(self, impact, age):
        thresholds = SeverityThresholds()
        ladder = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        fresh = severity_for(impact, age_days=0, thresholds=thresholds)
        aged = severity_for(impact, age_days=age, thresholds=thresholds)
        assert ladder.index(aged) >= ladder.index(fresh)

    @given(impacts, impacts)
    def test_more_money_never_means_less_severity(self, first, second):
        thresholds = SeverityThresholds()
        ladder = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        low, high = sorted((first, second))
        assert ladder.index(
            severity_for(high, age_days=0, thresholds=thresholds)
        ) >= ladder.index(severity_for(low, age_days=0, thresholds=thresholds))


class TestTheBoundsAlwaysHold:
    @given(queues)
    def test_no_queue_can_push_the_run_past_its_rounding_budget(self, items):
        bounds = AutoResolutionBounds()
        outcome = resolve_bounded(items, bounds)
        rounding = [
            item
            for item in outcome.applied
            if item.exception_class is ExceptionClass.ROUNDING_DRIFT
        ]
        assert sum(item.amount_minor for item in rounding) <= bounds.rounding_per_run_minor
        assert all(
            item.amount_minor <= bounds.rounding_per_record_minor for item in rounding
        )

    @given(queues)
    def test_only_the_three_named_classes_are_ever_touched(self, items):
        outcome = resolve_bounded(items, AutoResolutionBounds())
        assert all(
            item.exception_class in AGENT_RESOLVABLE_CLASSES
            for item in outcome.resolutions
        )

    @given(queues)
    def test_a_refusal_always_names_the_bound_it_broke(self, items):
        outcome = resolve_bounded(items, AutoResolutionBounds())
        for item in outcome.refused:
            assert item.bound
            assert item.refusal

    @given(queues)
    def test_the_floor_is_never_marked_resolvable(self, items):
        outcome = resolve_bounded(items, AutoResolutionBounds())
        stamped = mark_resolvable(items, outcome)
        for item in stamped:
            if item.exception_class is ExceptionClass.UNMATCHABLE:
                assert not item.resolvable_by_agent

    @given(queues)
    def test_disabling_the_bounds_resolves_nothing(self, items):
        outcome = resolve_bounded(items, AutoResolutionBounds(enabled=False))
        assert outcome.resolutions == ()

    @given(queues)
    def test_stamping_never_changes_the_queue_length_or_order(self, items):
        outcome = resolve_bounded(items, AutoResolutionBounds())
        stamped = mark_resolvable(items, outcome)
        assert [item.exception_id for item in stamped] == [
            item.exception_id for item in items
        ]
