"""Top-k candidate emission, labelled against ground truth.

The blender cannot be fitted on what the tiers *assert*, because on this corpus
everything they assert is right. Measured on `train`, seed 42: 307 evaluation
links proposed by the residual tiers, **307 of them true positives and not one
negative**. A logistic regression fitted on that has no contrast to learn from
and would return a base rate wearing the shape of a model.

The negatives exist -- they are simply the pairings the tiers *considered and
rejected*. T3 scores every unreferenced credit inside the amount and date
windows and keeps the best; the rest are labelled examples of what a wrong
pairing looks like, and they were discarded before Step 7 because nothing
needed them. T2 counts subsets and refuses when two fit; the second subset is
likewise a labelled negative.

So this module re-runs the residual ladder in a **top-k** mode: every decision
point emits up to ``top_k`` contenders instead of its single pick, each carries
the same feature vector the tier would have built for it, and each is labelled
against ``GroundTruth.evaluation_pairs``. Rank 0 is what the tier would assert;
ranks 1 and beyond are what it passed over.

WHAT THIS IS NOT
----------------
**Not a second matcher.** The harvester never decides anything, never consumes
a record on its own account, and never feeds the evaluation. It runs *beside*
the real ladder over the same pool -- contenders are collected from the pool
state each tier is about to see, then the real tier runs and consumes exactly
what it always would. Removing this module changes no decision on any split.

**Not a way to widen recall.** A contender is not a proposal. Nothing here can
cause a link to be auto-matched that the tiers did not propose; the harvest
only supplies the labels a model is fitted from.

WHICH ROWS ARE COLLECTED
------------------------
Evaluation-unit links only -- ``PAYMENT_CREDITED_AS``, ARCHITECTURE.md 2 -- and
residual tiers only. T0 and T1 run first so the pool the harvester sees is the
production residual, but their own candidates are never collected: they bypass
the blender, so training on them would fit a model to a population it will
never score.

A HONEST LIMITATION
-------------------
The feature vector is per-pairing: it says how well *this* credit fits *this*
settlement, and it carries no field for "and it beat its rivals by 0.12". So the
model cannot distinguish a contender that won from one that lost except through
its own feature values. The margin lives in the tier's refusal instead --
``arithmetic_verified=False`` on an ambiguity, which the blender is forbidden to
overturn (see :func:`~ledgerloop.matching.calibration.apply_bundle`). Adding a
margin feature would mean changing :class:`~ledgerloop.models.candidates.
FeatureVector`, which is a Step 0 contract, and it is not needed to make the
blender work -- so it is named here rather than done quietly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ledgerloop.config import MatchingTolerances, RunConfig
from ledgerloop.ingest.dataset import IngestResult
from ledgerloop.matching.bank_leg import attribute_clawback, candidate_id
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.matching.subset_sum import find_subsets
from ledgerloop.matching.tier0_exact import run_tier0
from ledgerloop.matching.tier1_tolerance import run_tier1
from ledgerloop.matching.tier2_aggregation import (
    Assignment,
    accept_for,
    credit_bucket,
    expected_credit_minor,
    payment_bucket,
    run_tier2,
    search_window,
)
from ledgerloop.matching.tier2_aggregation import features_for as aggregation_features
from ledgerloop.matching.tier3_lexical import (
    MerchantProfile,
    NameMatch,
    build_profiles,
    candidate_credits,
    rank_credits,
    run_tier3,
)
from ledgerloop.matching.tier3_lexical import features_for as lexical_features
from ledgerloop.matching.tier4_graph import run_tier4
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.records import CanonicalBankTxn, CanonicalPayment
from ledgerloop.models.refs import bank_ref, payment_ref
from ledgerloop.models.truth import GroundTruth
from ledgerloop.money import allocate_minor, format_minor, sum_minor

__all__ = ["DEFAULT_TOP_K", "HarvestResult", "LabelledCandidate", "harvest"]

#: Contenders kept per decision point. Three rather than one because one is the
#: pick (almost always correct, so almost always a positive) and the rest are
#: the rejections. Beyond three the extra rows are pairings no scorer ranked
#: anywhere near the answer -- easy negatives that only inflate the row count
#: and drag the fitted intercept towards the base rate of the harvest rather
#: than the base rate of the decisions.
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class LabelledCandidate:
    """One contender: where it came from, what rank it took, what truth says.

    ``accepted`` is the field the fit turns on. It records whether the tier
    **resolved** the decision point this contender came from -- not whether this
    particular pairing was the one it picked. A rival that lost at a resolved
    decision point is still part of the population the blender scores, because
    the winner from that same decision point is; a contender from a decision
    point the tier *refused* is not, because nothing from that point ever
    reaches the blender.
    """

    candidate: MatchCandidate
    rank: int
    is_positive: bool
    settlement_id: str
    accepted: bool = False
    asserted: bool = False

    @property
    def tier(self) -> Tier:
        return self.candidate.tier

    @property
    def pair(self) -> tuple[str, str]:
        return self.candidate.pair


@dataclass(frozen=True)
class HarvestResult:
    """Every labelled contender from one dataset, with the counts to explain it.

    Two populations, and the difference between them is the whole methodology:

    * :attr:`fit_rows` -- contenders from decision points the tiers **resolved**.
      This is exactly the population :func:`~ledgerloop.matching.calibration.
      apply_bundle` scores at run time, so it is the only population a model
      may be fitted on. Anything else is fitted on one distribution and applied
      to another.
    * :attr:`rows` -- every contender, resolved and refused alike. Larger, and
      it is where the wrong pairings live, so it is the population a reliability
      diagram has something to show on. Used as a **diagnostic** and never for
      fitting.
    """

    rows: tuple[LabelledCandidate, ...] = ()
    top_k: int = DEFAULT_TOP_K
    passes: int = 0
    decision_points: int = 0
    resolved_points: int = 0
    duplicates_dropped: int = 0

    @property
    def fit_rows(self) -> tuple[LabelledCandidate, ...]:
        """Contenders from decision points a tier resolved. The fit population."""
        return tuple(row for row in self.rows if row.accepted)

    @property
    def refused_rows(self) -> tuple[LabelledCandidate, ...]:
        """Contenders from decision points a tier refused. Never fitted on."""
        return tuple(row for row in self.rows if not row.accepted)

    @property
    def positives(self) -> int:
        return sum(1 for row in self.rows if row.is_positive)

    @property
    def negatives(self) -> int:
        return len(self.rows) - self.positives

    @property
    def features(self) -> tuple[FeatureVector, ...]:
        return tuple(row.candidate.features for row in self.fit_rows)

    @property
    def labels(self) -> tuple[bool, ...]:
        return tuple(row.is_positive for row in self.fit_rows)

    @property
    def diagnostic_features(self) -> tuple[FeatureVector, ...]:
        return tuple(row.candidate.features for row in self.rows)

    @property
    def diagnostic_labels(self) -> tuple[bool, ...]:
        return tuple(row.is_positive for row in self.rows)

    def by_tier(self, *, fit_only: bool = True) -> dict[str, int]:
        return _count_by_tier(self.fit_rows if fit_only else self.rows)

    def positives_by_tier(self, *, fit_only: bool = True) -> dict[str, int]:
        rows = self.fit_rows if fit_only else self.rows
        return _count_by_tier(tuple(row for row in rows if row.is_positive))


def _count_by_tier(rows: Sequence[LabelledCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.tier.name] = counts.get(row.tier.name, 0) + 1
    return {tier.name: counts[tier.name] for tier in Tier if tier.name in counts}


@dataclass
class _Collector:
    """Accumulates contenders, keeping the best rank seen for each link.

    The residual loop runs several passes over the same pool, and a settlement
    that stayed open is re-examined each time. Without de-duplication the same
    pairing would enter the training set once per pass, silently weighting it by
    how long it took the ladder to settle -- which is a property of the loop,
    not of the evidence.
    """

    rows: dict[tuple[str, str, Tier], LabelledCandidate] = field(default_factory=dict)
    decision_points: int = 0
    resolved_points: int = 0
    duplicates: int = 0

    def add(
        self,
        candidate: MatchCandidate,
        rank: int,
        truth: GroundTruth,
        settlement_id: str,
    ) -> None:
        key = (*candidate.pair, candidate.tier)
        is_positive = candidate.pair in truth.evaluation_pairs
        labelled = LabelledCandidate(
            candidate=candidate.model_copy(update={"is_truth_positive": is_positive}),
            rank=rank,
            is_positive=is_positive,
            settlement_id=settlement_id,
        )
        existing = self.rows.get(key)
        if existing is not None:
            self.duplicates += 1
            if existing.rank <= labelled.rank:
                return
        self.rows[key] = labelled

    def resolve(
        self, settlements: set[str], asserted: set[tuple[str, str]], tier: Tier
    ) -> None:
        """Mark what the tier concluded, once it has actually run.

        Acceptance is read off the tier's own output rather than re-derived from
        its gates. Re-implementing "would T3 have taken this?" beside the tier
        is how a training population drifts away from the population it is meant
        to describe -- and the drift would be invisible, because both sides
        would look reasonable in isolation.
        """
        for key, row in list(self.rows.items()):
            if row.tier is not tier or row.settlement_id not in settlements:
                continue
            self.rows[key] = replace(
                row, accepted=True, asserted=row.pair in asserted
            )

    def ordered(self) -> tuple[LabelledCandidate, ...]:
        """Rows in a stable order: tier, then rank, then the pair itself.

        The fit is deterministic given its input order, so the input order has
        to be deterministic too -- a dict insertion order that depends on which
        pass first saw a settlement would make two identical runs produce two
        different models.
        """
        return tuple(
            sorted(
                self.rows.values(),
                key=lambda row: (row.tier, row.rank, row.pair),
            )
        )


def _contender(
    *,
    tier: Tier,
    payment: CanonicalPayment,
    credit: CanonicalBankTxn,
    features: FeatureVector,
    detail: str,
    share_minor: int,
    verified: bool,
    subset: tuple[CanonicalPayment, ...] = (),
) -> MatchCandidate:
    """Build one evaluation-unit contender.

    The candidate id is the same content-derived id the tier would produce for
    this pairing, so a harvested contender and the tier's own candidate for the
    same link are recognisably the same object rather than two rows about one
    fact.
    """
    return MatchCandidate(
        candidate_id=candidate_id(
            tier,
            LinkType.PAYMENT_CREDITED_AS,
            payment_ref(payment.payment_id).key,
            bank_ref(credit.txn_id).key,
        ),
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref(payment.payment_id),
        target_ref=bank_ref(credit.txn_id),
        tier=tier,
        features=features,
        evidence=(
            Evidence(
                kind=(
                    EvidenceKind.SUBSET_SUM
                    if tier is Tier.T2_AGGREGATION
                    else EvidenceKind.LEXICAL_SIMILARITY
                ),
                detail=detail,
                refs=(payment_ref(payment.payment_id), bank_ref(credit.txn_id)),
                amount_minor=share_minor,
            ),
        ),
        subset_members=tuple(payment_ref(p.payment_id) for p in subset),
        arithmetic_verified=verified,
    )


def _harvest_aggregation(
    context: MatchContext,
    config: RunConfig,
    truth: GroundTruth,
    collector: _Collector,
    top_k: int,
) -> None:
    """T2's contenders: up to ``top_k`` subsets per open, keyed credit.

    The search runs over the settlement's whole payment bucket for *each*
    credit, rather than over the sequential remainder the tier uses. That is
    deliberate: the tier solves a batch transactionally and never revisits a
    tranche, so its remainder shrinks as it goes, whereas the training set wants
    the alternatives that existed at the moment of the decision -- including the
    subsets that would have explained a different tranche.
    """
    tolerances = config.tolerances
    epsilon = tolerances.aggregation_epsilon_minor
    for view in list(context.open_settlements()):
        credits = credit_bucket(view, context)
        if len(credits) < 2:
            continue
        payments = payment_bucket(view, context)
        gross = view.payment_gross_minor
        if not payments or gross <= 0 or view.net_minor <= 0:
            continue

        for credit in credits:
            collector.decision_points += 1
            low, high = search_window(credit.credit_minor, gross, view.net_minor, epsilon)
            search = find_subsets(
                [p.amount_minor for p in payments],
                low,
                high,
                want=top_k,
                max_exact_items=tolerances.max_subset_size,
                timeout_ms=tolerances.subset_solver_timeout_ms,
                accept=accept_for(credit.credit_minor, gross, view.net_minor, epsilon),
            )
            for rank, solution in enumerate(search.solutions):
                taken = tuple(payments[i] for i in solution.indices)
                if not taken:
                    continue
                assignment = Assignment(
                    credit=credit,
                    payments=taken,
                    gross_minor=solution.total_minor,
                    expected_minor=expected_credit_minor(
                        solution.total_minor, gross, view.net_minor
                    ),
                    search=search,
                )
                features = aggregation_features(assignment, view, epsilon)
                shares = allocate_minor(
                    credit.credit_minor, [p.amount_minor for p in taken]
                )
                verified = (
                    sum_minor(shares, field=f"{credit.txn_id}.harvest")
                    == credit.credit_minor
                )
                for payment, share in zip(taken, shares, strict=True):
                    collector.add(
                        _contender(
                            tier=Tier.T2_AGGREGATION,
                            payment=payment,
                            credit=credit,
                            features=features,
                            detail=(
                                f"contender rank {rank}: {len(taken)} payment(s) of "
                                f"{view.settlement_id} carrying gross "
                                f"{format_minor(solution.total_minor)} would compose "
                                f"{credit.txn_id}"
                            ),
                            share_minor=share,
                            verified=verified,
                            subset=taken,
                        ),
                        rank,
                        truth,
                        view.settlement_id,
                    )


def _harvest_lexical(
    context: MatchContext,
    config: RunConfig,
    truth: GroundTruth,
    collector: _Collector,
    profiles: dict[str, MerchantProfile],
    top_k: int,
) -> None:
    """T3's contenders: the top ``top_k`` credits by name score, gate removed.

    ``gate=0.0`` is the whole point. The tier keeps only credits scoring at or
    above ``min_score``, and ARCHITECTURE.md 6 decision 28 records that the gate
    sits at 0.90 for precision even though true variant pairs score as low as
    0.667. Every credit the gate turns away is a labelled example of a wrong
    pairing at a known score, and those are the rows that teach the blender what
    the lexical column is worth.
    """
    tolerances = config.tolerances
    lexical = config.lexical
    for view in list(context.open_settlements()):
        merchant = context.merchant_of(view)
        profile = profiles.get(merchant) if merchant is not None else None
        if profile is None or not profile:
            continue
        if not view.payments or view.net_minor <= 0:
            continue
        pool = candidate_credits(view, context, tolerances, lexical)
        if not pool:
            continue

        collector.decision_points += 1
        scored, _, _ = rank_credits(view, profile, pool, lexical, gate=0.0)
        for rank, match in enumerate(scored[:top_k]):
            _collect_lexical_match(view, match, rank, tolerances, truth, collector)


def _collect_lexical_match(
    view: SettlementView,
    match: NameMatch,
    rank: int,
    tolerances: MatchingTolerances,
    truth: GroundTruth,
    collector: _Collector,
) -> None:
    """Expand one scored credit into evaluation-unit contenders.

    The **charged-back payment is excluded**, exactly as
    :mod:`~ledgerloop.matching.tier3_lexical` excludes it: A08 nets a payment
    off the settlement, so its money never reached the bank and no credit
    carries it. Harvesting it would put a link into the training set that ground
    truth calls false for a reason no feature explains -- the model would be
    asked to learn an exclusion the arithmetic already performs. Found by the
    fit population disagreeing with the tier's own output on four rows.
    """
    credit = match.credit
    clawback = attribute_clawback(view)
    excluded = clawback.excluded.payment_id if clawback.excluded is not None else None
    covered = tuple(p for p in view.payments if p.payment_id != excluded)
    if not covered:
        return
    shares = allocate_minor(credit.credit_minor, [p.amount_minor for p in covered])
    verified = view.gross_reconciles and (
        sum_minor(shares, field=f"{credit.txn_id}.harvest") == credit.credit_minor
    )
    features = lexical_features(view, match, tolerances)
    for payment, share in zip(covered, shares, strict=True):
        collector.add(
            _contender(
                tier=Tier.T3_FUZZY,
                payment=payment,
                credit=credit,
                features=features,
                detail=(
                    f"contender rank {rank}: {credit.txn_id} names "
                    f"{credit.extracted_merchant!r}, scoring {match.score:.3f} against "
                    f"{match.matched_spelling!r} for {view.settlement_id}"
                ),
                share_minor=share,
                verified=verified,
            ),
            rank,
            truth,
            view.settlement_id,
        )


def _harvest_graph(
    candidates: Sequence[MatchCandidate], truth: GroundTruth, collector: _Collector
) -> None:
    """T4 contributes what it inferred, at rank 0.

    No top-k for the graph tier: its rules are constraint propagation, so a
    conclusion either follows from the established links or it does not. There
    is no runner-up to a deduction. On this corpus both inference rules fire
    zero times (ARCHITECTURE.md 6 decision 31), so this collects nothing -- and
    it is written rather than skipped so that a corpus where they *do* fire
    trains the model on them instead of silently omitting a tier.
    """
    for candidate in candidates:
        if not candidate.is_evaluable:
            continue
        collector.decision_points += 1
        collector.add(candidate, 0, truth, candidate.candidate_id)
        if not candidate.arithmetic_verified:
            # Same rule as everywhere else: an inference whose allocation does
            # not conserve is a refusal, the blender never scores it, and so it
            # is collected as a diagnostic rather than fitted on.
            continue
        collector.resolve({candidate.candidate_id}, {candidate.pair}, Tier.T4_GRAPH)
        collector.resolved_points += 1


def _mark_concluded(
    collector: _Collector, candidates: Sequence[MatchCandidate], tier: Tier
) -> None:
    """Tell the collector which decision points the tier just resolved.

    A settlement counts as resolved when the tier emitted an **arithmetically
    verified** settlement edge for it. That is the same flag
    :func:`~ledgerloop.matching.calibration.apply_bundle` uses to decide whether
    a candidate reaches the blender at all, so the two agree by construction
    rather than by two developers remembering the same rule.
    """
    settlements = {
        candidate.source_ref.record_id
        for candidate in candidates
        if candidate.link_type is LinkType.SETTLEMENT_CREDITED_AS
        and candidate.arithmetic_verified
    }
    asserted = {
        candidate.pair
        for candidate in candidates
        if candidate.is_evaluable and candidate.arithmetic_verified
    }
    collector.resolved_points += len(settlements)
    collector.resolve(settlements, asserted, tier)


def harvest(
    ingest: IngestResult,
    truth: GroundTruth,
    config: RunConfig,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> HarvestResult:
    """Collect labelled top-k contenders from the residual tiers of one dataset.

    The ladder is run exactly as :func:`~ledgerloop.matching.pipeline.
    run_matching` runs it -- T0, T1, then the bounded T2/T3/T4 loop -- so the
    pool at each decision point is the production residual and not an artefact
    of the harvest. The contenders are read off the pool *before* each tier
    consumes from it.

    ``truth`` is used for labelling and nothing else. It is never consulted to
    decide what to collect, so a link the ladder cannot reach is absent from the
    training set because no tier proposed it, not because it was filtered out
    for being hard.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    context = MatchContext.from_ingest(
        ingest,
        detect_duplicates=config.duplicates.enabled,
        duplicate_window_days=config.duplicates.window_days,
    )
    order_leg, t0_bank = run_tier0(context)
    t1_bank = run_tier1(context, config.tolerances)

    profiles = build_profiles(context)
    collector = _Collector()
    residual: list[MatchCandidate] = []
    passes = 0

    for _ in range(config.graph.max_rerun_passes):
        passes += 1
        before = (len(collector.rows), len(residual))

        _harvest_aggregation(context, config, truth, collector, top_k)
        aggregation = run_tier2(context, config.tolerances)
        _mark_concluded(collector, aggregation.candidates, Tier.T2_AGGREGATION)

        _harvest_lexical(context, config, truth, collector, profiles, top_k)
        lexical = run_tier3(context, config.tolerances, config.lexical, profiles=profiles)
        _mark_concluded(collector, lexical.candidates, Tier.T3_FUZZY)

        # The same premise set the pipeline hands T4: everything established so
        # far, keyed tiers included. A narrower set would let the harvest see
        # inferences production would never draw, or miss ones it would.
        established = (
            *order_leg.candidates,
            *t0_bank.candidates,
            *t1_bank.candidates,
            *residual,
            *aggregation.candidates,
            *lexical.candidates,
        )
        graph = run_tier4(context, established, config.graph, config.thresholds)
        _harvest_graph(graph.candidates, truth, collector)

        residual.extend(aggregation.candidates)
        residual.extend(lexical.candidates)
        residual.extend(graph.candidates)
        if (len(collector.rows), len(residual)) == before:
            break

    return HarvestResult(
        rows=collector.ordered(),
        top_k=top_k,
        passes=passes,
        decision_points=collector.decision_points,
        resolved_points=collector.resolved_points,
        duplicates_dropped=collector.duplicates,
    )
