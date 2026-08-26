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
from math import sqrt

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import AnomalyClass, LinkType, RecordType
from ledgerloop.models.metrics import LinkMetrics, RunMetrics, TierContribution
from ledgerloop.models.refs import RecordRef
from ledgerloop.models.truth import GroundTruth, TruthPair
from ledgerloop.money import sum_minor

__all__ = [
    "EVALUATED_RECORD_TYPES",
    "Z_95",
    "LinkConfusion",
    "MatchRateResult",
    "MoneyView",
    "PredictedLink",
    "confusion",
    "evaluate",
    "evaluation_links_by_class",
    "link_metrics",
    "match_rate",
    "money_view",
    "recall_by_anomaly_class",
    "wilson_interval",
]

#: Two-sided 95% normal quantile.
Z_95 = 1.959963984540054

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


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(0.0, 1.0)`` on zero trials: with no evidence, every proportion
    remains possible, and that is the interval that says so.
    """
    if successes < 0 or trials < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if trials == 0:
        return (0.0, 1.0)

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        z
        / denominator
        * sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4 * trials * trials))
    )
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)

    # At the boundaries the algebra is exact -- a run with no failures has an
    # upper bound of exactly 1, and one with no successes a lower bound of
    # exactly 0 -- but the floating-point evaluation lands a few ulps short
    # (0.9999999999999998 at 250/250). Those two cases are the ones this project
    # reports most often, so they are pinned rather than left to round.
    if successes == trials:
        high = 1.0
    if successes == 0:
        low = 0.0
    return (low, high)


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


def evaluate(
    predictions: Sequence[PredictedLink],
    truth: GroundTruth,
    *,
    run_id: str,
    wall_clock_ms: int = 0,
    tier_contributions: Sequence[TierContribution] = (),
) -> RunMetrics:
    """Score one system's predictions against one dataset's truth.

    The single entry point the report consumes. ``record_count`` is taken from
    the truth's verdict list, which has exactly one entry per record, so it
    cannot disagree with the denominators computed beside it.

    ``tier_contributions`` is what each tier proposed and what survived the
    decision policy. It defaults to empty because a baseline has no tiers, and
    an empty tuple renders as an absent section rather than as a table of
    zeros -- a zero for an unbuilt component is a false measurement.
    """
    pairs = [prediction.pair for prediction in predictions]
    truth_pairs = truth.evaluation_pairs

    # Last assertion wins on a repeated pair, matching `confusion`'s
    # de-duplication so the cost table and the counts describe the same set.
    asserted = {prediction.pair: prediction.amount_minor for prediction in predictions}

    links = link_metrics(pairs, truth_pairs, asserted_amount_by_pair=asserted)
    coverage = match_rate(pairs, truth)
    money = money_view(pairs, truth)
    record_count = len(truth.records)

    seconds = wall_clock_ms / 1000.0
    return RunMetrics(
        run_id=run_id,
        record_count=record_count,
        auto_match_precision=links.precision,
        match_rate=coverage.rate,
        link_metrics=links,
        recall_by_anomaly_class=recall_by_anomaly_class(pairs, truth),
        unmatchable_count=len(truth.unmatchable_refs),
        unmatchable_impact_minor=truth.impact_total_minor(truth.unmatchable_refs),
        reconciled_minor=money.reconciled_minor,
        outstanding_minor=money.outstanding_minor,
        records_per_second=record_count / seconds if seconds > 0 else 0.0,
        tier_contributions=tuple(tier_contributions),
    )
