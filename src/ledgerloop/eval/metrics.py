"""Link-level scoring. The scoreboard every later step is measured against.

THE UNIT
--------
``ARCHITECTURE.md`` §2 fixes the atomic unit of evaluation as the
``PAYMENT_CREDITED_AS`` link -- a ``(payment, bank_txn)`` pair. Everything here
scores against :attr:`GroundTruth.evaluation_pairs` and nothing else. Structural
edges are excluded because the sources assert them; counting edges the system
never had to work for would inflate every number in the report.

WHY A CONFIDENCE INTERVAL, AND WHY WILSON
-----------------------------------------
(:func:`~ledgerloop.stats.wilson_interval` itself moved to
:mod:`ledgerloop.stats` at Step 7, so the calibrator could use it without
``matching`` depending on ``eval``. It is re-exported here unchanged.)

The target is precision >= 0.99 on a test split carrying roughly 250 evaluable
links. At that size one wrong decision moves precision from 1.000 to 0.996 --
a headline "0.99" is indistinguishable from a lucky 0.97. So the point estimate
travels with an interval.

The interval is **Wilson**, not the normal approximation, because this project
lives in exactly the regime where the normal approximation fails. At 250
successes out of 250, ``p +- z*sqrt(p(1-p)/n)`` collapses to ``[1.0, 1.0]``:
it claims certainty that the true precision is perfect, from 250 samples. The
Wilson interval returns roughly ``[0.985, 1.0]``, which is the honest statement.
An evaluator built to make this project's central claim credible cannot use the
estimator that breaks precisely where the claim lives.

DEGENERATE DENOMINATORS
-----------------------
Precision with no predictions, recall with no truth, and F1 with both at zero
all return **0.0**, never 1.0. A system that predicted nothing has not achieved
perfect precision, and the report renders a zero denominator as ``n/a`` rather
than as a score. The confidence interval, by contrast, widens to the full
``[0.0, 1.0]`` on no evidence, which is the correct statement of ignorance.

PRECISION AND MATCH RATE ARE ORTHOGONAL ON PURPOSE
--------------------------------------------------
:func:`match_rate` counts records the system made *any* positive assertion
about; it does not check whether the assertion was right. Correctness is what
precision measures. Folding correctness into the match rate would double-count
it and hide the trade-off the decision policy exists to make.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import (
    AnomalyClass,
    ExceptionClass,
    ExpectedStatus,
    LinkType,
    RecordType,
)
from ledgerloop.models.metrics import LinkMetrics, RunMetrics, TierContribution
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import RecordRef
from ledgerloop.models.truth import GroundTruth, TruthPair
from ledgerloop.money import sum_minor
from ledgerloop.stats import Z_95, wilson_interval

__all__ = [
    "EVALUATED_RECORD_TYPES",
    "EXCEPTION_RECORD_TYPES",
    "Z_95",
    "ExceptionCoverage",
    "LinkConfusion",
    "MatchRateResult",
    "MoneyView",
    "PredictedLink",
    "confusion",
    "covered_refs",
    "evaluate",
    "evaluation_links_by_class",
    "exception_confusion",
    "exception_coverage",
    "exception_impact_minor",
    "exceptions_by_class",
    "link_metrics",
    "match_rate",
    "money_view",
    "recall_by_anomaly_class",
    "wilson_interval",
]

#: The record types the evaluation unit touches.
#:
#: ``PAYMENT_CREDITED_AS`` runs between a payment and a bank transaction, so
#: those are the only records a matcher can resolve by producing one. Orders and
#: settlements are attached by structural edges the sources already assert;
#: including them in the match-rate denominator would charge the system for
#: records it was never asked to match, and including them in the numerator
#: would hand it credit for edges it was given.
EVALUATED_RECORD_TYPES = frozenset({RecordType.PAYMENT, RecordType.BANK_TXN})


class PredictedLink(FrozenLedgerModel):
    """One link a system asserts exists. The evaluator's input contract.

    Mirrors :class:`~ledgerloop.models.truth.GroundTruthLink` deliberately --
    same endpoints, same ``pair`` key -- so predictions and truth are compared
    without either side reshaping the other. From Step 7 the decision policy
    produces these from ``AUTO_MATCHED`` :class:`~ledgerloop.models.decisions.
    MatchDecision` records; until then the baselines produce them directly.

    ``amount_minor`` is what the predictor *claims* was credited, which is not
    necessarily what the truth says. That gap is the point: it is what makes
    :attr:`LinkMetrics.false_positive_cost_minor` a rupee figure rather than a
    ratio.
    """

    source_ref: RecordRef
    target_ref: RecordRef
    amount_minor: MinorUnits = 0

    @property
    def pair(self) -> TruthPair:
        return (self.source_ref.key, self.target_ref.key)


@dataclass(frozen=True)
class LinkConfusion:
    """The three sets, kept as sets rather than counts.

    The report names specific wrong links -- "B0 credited PAY-00013 to both
    BNK-00002 and BNK-00018" -- and an exception queue eventually has to point
    at them. Returning counts alone would throw away the only part of the
    result a human can act on.
    """

    true_positives: frozenset[TruthPair]
    false_positives: frozenset[TruthPair]
    false_negatives: frozenset[TruthPair]

    @property
    def predicted_count(self) -> int:
        return len(self.true_positives) + len(self.false_positives)

    @property
    def truth_count(self) -> int:
        return len(self.true_positives) + len(self.false_negatives)

    @property
    def precision(self) -> float:
        return _ratio(len(self.true_positives), self.predicted_count)

    @property
    def recall(self) -> float:
        return _ratio(len(self.true_positives), self.truth_count)

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        if total == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / total


@dataclass(frozen=True)
class MatchRateResult:
    """Match rate with its denominator exposed.

    A bare ratio invites the reader to assume a denominator. This one is
    ``reconcilable_refs`` restricted to :data:`EVALUATED_RECORD_TYPES`:
    ``UNMATCHABLE`` records are excluded -- PLAN.md §8.2.5's honest floor --
    and reported separately instead of being counted as failures.
    """

    resolved_refs: frozenset[str]
    denominator_refs: frozenset[str]

    @property
    def rate(self) -> float:
        return _ratio(len(self.resolved_refs), len(self.denominator_refs))


@dataclass(frozen=True)
class MoneyView:
    """Rupees reconciled against rupees still outstanding.

    Both are measured over truth link amounts, so ``reconciled + outstanding``
    is exactly the money the evaluation unit covers -- no paise created by a
    system that over-asserts, none destroyed by one that under-asserts. What an
    over-asserting system produces instead is a large
    :attr:`LinkMetrics.false_positive_cost_minor`.
    """

    reconciled_minor: int
    outstanding_minor: int

    @property
    def total_minor(self) -> int:
        return self.reconciled_minor + self.outstanding_minor


def _ratio(numerator: int, denominator: int) -> float:
    """Zero denominator yields 0.0 -- see the module docstring."""
    return numerator / denominator if denominator > 0 else 0.0


def confusion(
    predicted_pairs: Iterable[TruthPair], truth_pairs: AbstractSet[TruthPair]
) -> LinkConfusion:
    """Partition predictions and truth into TP / FP / FN.

    Predictions are de-duplicated: asserting the same link twice is one claim,
    not two, and counting it twice would let a system inflate or deflate its own
    denominator by repeating itself.
    """
    predicted = frozenset(predicted_pairs)
    return LinkConfusion(
        true_positives=predicted & truth_pairs,
        false_positives=predicted - truth_pairs,
        false_negatives=frozenset(truth_pairs) - predicted,
    )


def link_metrics(
    predicted_pairs: Iterable[TruthPair],
    truth_pairs: AbstractSet[TruthPair],
    *,
    asserted_amount_by_pair: Mapping[TruthPair, int] | None = None,
) -> LinkMetrics:
    """Precision / recall / F1 with a Wilson interval on precision.

    ``asserted_amount_by_pair`` supplies what the predictor claimed each link
    was worth. The false positives' claimed amounts sum into
    ``false_positive_cost_minor``: the money the system declared reconciled
    that was not.
    """
    matrix = confusion(predicted_pairs, truth_pairs)
    true_positives = len(matrix.true_positives)
    false_positives = len(matrix.false_positives)

    ci_low, ci_high = wilson_interval(true_positives, matrix.predicted_count)

    cost = 0
    if asserted_amount_by_pair is not None:
        cost = sum_minor(
            (asserted_amount_by_pair.get(pair, 0) for pair in sorted(matrix.false_positives)),
            field="false_positive_cost",
        )

    return LinkMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=len(matrix.false_negatives),
        precision=matrix.precision,
        recall=matrix.recall,
        f1=matrix.f1,
        precision_ci_low=ci_low,
        precision_ci_high=ci_high,
        false_positive_cost_minor=cost,
    )


def match_rate(predicted_pairs: Iterable[TruthPair], truth: GroundTruth) -> MatchRateResult:
    """Fraction of reconcilable payment/bank records the system asserted a link for."""
    denominator = frozenset(
        ref
        for ref in truth.reconcilable_refs
        if RecordRef.parse(ref).record_type in EVALUATED_RECORD_TYPES
    )
    touched: set[str] = set()
    for source_key, target_key in predicted_pairs:
        touched.add(source_key)
        touched.add(target_key)
    return MatchRateResult(
        resolved_refs=frozenset(touched) & denominator, denominator_refs=denominator
    )


def evaluation_links_by_class(truth: GroundTruth) -> dict[AnomalyClass, frozenset[TruthPair]]:
    """Group the evaluation links by the anomaly classes touching their endpoints.

    **Attribution goes through the record verdicts, not the link label.**
    :attr:`GroundTruthLink.anomaly_class` exists on the model but the generator
    only ever labels *records*: every link it emits carries the default
    ``A01_CLEAN``. Grouping on the link field would therefore report one row of
    100% clean links and silently hide all ten other classes -- which is the
    exact failure mode PLAN.md §9.1 asks this table to prevent.

    **A link may appear under more than one class.** Settled decision 14 is that
    anomalies compose along independent aspects: a batch can credit the wrong
    amount *and* arrive late *and* lose its narration reference. A link whose
    payment is charged back and whose bank row is a split payout is genuinely
    both, and forcing it into one row would misreport whichever class lost the
    tiebreak. The rows therefore do not partition the link set, and the report
    says so rather than inviting the reader to add the columns up.

    ``CLEAN`` collects only the links where *both* endpoints are clean.
    """
    verdicts = truth.verdict_by_ref
    grouped: dict[AnomalyClass, set[TruthPair]] = {}
    for link in truth.links:
        if link.link_type is not LinkType.PAYMENT_CREDITED_AS:
            continue
        touched = {
            verdicts[ref].anomaly_class
            for ref in link.pair
            if ref in verdicts and verdicts[ref].anomaly_class is not AnomalyClass.CLEAN
        }
        for anomaly in touched or {AnomalyClass.CLEAN}:
            grouped.setdefault(anomaly, set()).add(link.pair)
    return {anomaly: frozenset(pairs) for anomaly, pairs in grouped.items()}


def recall_by_anomaly_class(
    predicted_pairs: Iterable[TruthPair], truth: GroundTruth
) -> dict[AnomalyClass, float]:
    """Per-class recall over the evaluation links, **including the bad rows**.

    Every class with at least one evaluation link appears, even at 0.0.
    Publishing only the classes that score well is precisely what this project
    is trying not to do (PLAN.md §9.1), so the omission is not available here:
    the caller gets whatever the data contains.
    """
    predicted = frozenset(predicted_pairs)
    return {
        anomaly: _ratio(len(pairs & predicted), len(pairs))
        for anomaly, pairs in sorted(
            evaluation_links_by_class(truth).items(), key=lambda item: item[0].value
        )
    }


def money_view(predicted_pairs: Iterable[TruthPair], truth: GroundTruth) -> MoneyView:
    """Rupees on correctly found links versus rupees on links that were missed."""
    predicted = frozenset(predicted_pairs)
    reconciled: list[int] = []
    outstanding: list[int] = []
    for link in truth.links:
        if link.link_type is not LinkType.PAYMENT_CREDITED_AS:
            continue
        bucket = reconciled if link.pair in predicted else outstanding
        bucket.append(link.amount_minor)
    return MoneyView(
        reconciled_minor=sum_minor(reconciled, field="reconciled"),
        outstanding_minor=sum_minor(outstanding, field="outstanding"),
    )


# ---------------------------------------------------------------------------
# Exception metrics (Step 8)
#
# A different unit from everything above. Precision and recall over links ask
# "did the system find the money?"; these ask "did the system *say something
# useful* about the money it could not find?", and the answer is a property of
# records, not of pairs.
# ---------------------------------------------------------------------------

#: Record types the exception queue is accountable for.
#:
#: The same restriction the match rate uses, plus settlements: a payout that
#: never arrived is exactly the kind of item a controller expects in the queue,
#: even though no ``PAYMENT_CREDITED_AS`` link runs through it.
EXCEPTION_RECORD_TYPES = frozenset(
    {RecordType.ORDER, RecordType.PAYMENT, RecordType.SETTLEMENT, RecordType.BANK_TXN}
)


@dataclass(frozen=True)
class ExceptionCoverage:
    """Which records the queue accounted for, and which it left silent.

    Two denominators, kept apart on purpose:

    * **``expected``** -- records ground truth calls ``EXCEPTION``: resolvable
      items the system failed to resolve. Missing one of these is a real failure
      and it is what :attr:`recall` measures.
    * **``unmatchable``** -- records ground truth calls ``UNMATCHABLE``. Raising
      an exception for one is the *correct* behaviour (it is how the honest
      floor gets reported), but crediting it inside the headline recall would
      let a system inflate the number by describing items nobody could resolve.
      It is reported as its own line.
    """

    expected: frozenset[str]
    covered_expected: frozenset[str]
    unmatchable: frozenset[str]
    covered_unmatchable: frozenset[str]
    raised: int = 0
    out_of_scope: int = 0
    """Records excluded from both denominators, counted so the exclusion is visible."""

    @property
    def recall(self) -> float:
        return _ratio(len(self.covered_expected), len(self.expected))

    @property
    def missed(self) -> frozenset[str]:
        return self.expected - self.covered_expected

    @property
    def unmatchable_recall(self) -> float:
        return _ratio(len(self.covered_unmatchable), len(self.unmatchable))


def covered_refs(exceptions: Iterable[ReconException]) -> frozenset[str]:
    """Every record key any exception in the queue names.

    A record is covered when an exception *mentions* it, not only when it is the
    subject. A chargeback exception whose subject is the payment and whose chain
    names the settlement and the order has told the controller about all three,
    and a metric that only counted subjects would report two of them as silence.
    """
    return frozenset(
        ref.key for exception in exceptions for ref in exception.involved_refs
    )


def exception_coverage(
    exceptions: Sequence[ReconException],
    truth: GroundTruth,
    *,
    record_types: AbstractSet[RecordType] = EXCEPTION_RECORD_TYPES,
    out_of_scope: AbstractSet[str] = frozenset(),
) -> ExceptionCoverage:
    """How much of what should have been reported actually was.

    ``out_of_scope`` carries one honest exclusion. A bank **debit** is money
    leaving the account, not a payout this system reconciles, and ground truth
    marks every debit ``UNMATCHABLE`` because no settlement claims it. Listing
    thirty-four outgoing rows in a controller's queue would be noise, so they sit
    outside the unit -- and the count is reported rather than the rows being
    dropped quietly. Passing nothing includes them, which is the conservative
    default: an exclusion has to be asked for.
    """
    covered = covered_refs(exceptions)
    expected: set[str] = set()
    unmatchable: set[str] = set()
    ignored = 0
    for record in truth.records:
        if record.record_ref.record_type not in record_types:
            continue
        if record.record_ref.key in out_of_scope:
            ignored += 1
            continue
        if record.expected_status is ExpectedStatus.EXCEPTION:
            expected.add(record.record_ref.key)
        elif record.expected_status is ExpectedStatus.UNMATCHABLE:
            unmatchable.add(record.record_ref.key)
    return ExceptionCoverage(
        expected=frozenset(expected),
        covered_expected=frozenset(expected & covered),
        unmatchable=frozenset(unmatchable),
        covered_unmatchable=frozenset(unmatchable & covered),
        raised=len(exceptions),
        out_of_scope=ignored,
    )


def exception_confusion(
    exceptions: Sequence[ReconException], truth: GroundTruth
) -> dict[str, dict[str, int]]:
    """True anomaly class -> predicted exception class -> count.

    **Rectangular, not square**, and that is the point (ARCHITECTURE.md 6,
    decision 5). Eleven anomaly classes describe what the generator did; thirteen
    exception classes describe what the system concluded; the two vocabularies
    answer different questions and the mapping between them is many-to-many. A
    square matrix would be an identity nobody measured.

    An exception is attributed to every anomaly touching any record it names, so
    -- as with the per-class recall table -- the rows do not partition the queue
    and the report says so.
    """
    verdicts = truth.verdict_by_ref
    matrix: dict[str, dict[str, int]] = {}
    for exception in exceptions:
        touched = {
            verdicts[ref.key].anomaly_class
            for ref in exception.involved_refs
            if ref.key in verdicts
        }
        for anomaly in touched or {AnomalyClass.CLEAN}:
            row = matrix.setdefault(anomaly.value, {})
            predicted = exception.exception_class.value
            row[predicted] = row.get(predicted, 0) + 1
    return {
        anomaly.value: dict(sorted(matrix[anomaly.value].items()))
        for anomaly in AnomalyClass
        if anomaly.value in matrix
    }


def exceptions_by_class(
    exceptions: Sequence[ReconException],
) -> dict[ExceptionClass, int]:
    """Queue size per class, in taxonomy order."""
    counts: dict[ExceptionClass, int] = {}
    for exception in exceptions:
        counts[exception.exception_class] = counts.get(exception.exception_class, 0) + 1
    return {
        exception_class: counts[exception_class]
        for exception_class in ExceptionClass
        if exception_class in counts
    }


def exception_impact_minor(exceptions: Sequence[ReconException]) -> int:
    """Total money the queue is about, through the money gate."""
    return sum_minor(
        (exception.impact_minor for exception in exceptions), field="exception_impact"
    )


def evaluate(
    predictions: Sequence[PredictedLink],
    truth: GroundTruth,
    *,
    run_id: str,
    wall_clock_ms: int = 0,
    tier_contributions: Sequence[TierContribution] = (),
    exceptions: Sequence[ReconException] = (),
    out_of_scope_refs: AbstractSet[str] = frozenset(),
) -> RunMetrics:
    """Score one system's predictions against one dataset's truth.

    The single entry point the report consumes. ``record_count`` is taken from
    the truth's verdict list, which has exactly one entry per record, so it
    cannot disagree with the denominators computed beside it.

    ``tier_contributions`` is what each tier proposed and what survived the
    decision policy. It defaults to empty because a baseline has no tiers, and
    an empty tuple renders as an absent section rather than as a table of
    zeros -- a zero for an unbuilt component is a false measurement.

    ``exceptions`` is the queue Step 8's classifier produced. Same rule: a
    baseline has no exception classifier, so the empty default renders as an
    absent section and never as a recall of zero.
    """
    pairs = [prediction.pair for prediction in predictions]
    truth_pairs = truth.evaluation_pairs

    # Last assertion wins on a repeated pair, matching `confusion`'s
    # de-duplication so the cost table and the counts describe the same set.
    asserted = {prediction.pair: prediction.amount_minor for prediction in predictions}

    links = link_metrics(pairs, truth_pairs, asserted_amount_by_pair=asserted)
    coverage = match_rate(pairs, truth)
    queue = exception_coverage(exceptions, truth, out_of_scope=out_of_scope_refs)
    money = money_view(pairs, truth)
    record_count = len(truth.records)

    seconds = wall_clock_ms / 1000.0
    return RunMetrics(
        run_id=run_id,
        record_count=record_count,
        auto_match_precision=links.precision,
        match_rate=coverage.rate,
        exception_recall=queue.recall,
        exceptions_by_class=exceptions_by_class(exceptions),
        exception_confusion=exception_confusion(exceptions, truth),
        link_metrics=links,
        recall_by_anomaly_class=recall_by_anomaly_class(pairs, truth),
        unmatchable_count=len(truth.unmatchable_refs),
        unmatchable_impact_minor=truth.impact_total_minor(truth.unmatchable_refs),
        reconciled_minor=money.reconciled_minor,
        outstanding_minor=money.outstanding_minor,
        records_per_second=record_count / seconds if seconds > 0 else 0.0,
        tier_contributions=tuple(tier_contributions),
    )
