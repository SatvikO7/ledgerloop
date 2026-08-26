"""T4 -- graph inference. Constraint propagation over what is already known.

PLAN.md §6.4 gives four rules. Three of them reason over the entity graph
``Order -> Payment -> Settlement -> BankTxn``; the fourth is a fraud signal that
produces no matches at all.

**Exclusivity pruning** (negative)
    A credit whose links already absorb its full amount cannot take on more.
    Produces no matches; it blocks the other two.
**Path closure** (deductive)
    ``P -> S`` known and ``S -> C`` known implies ``P -> C``.
**Sibling completion** (inductive)
    Most of a settlement's payments point at credit B, so the rest are
    constrained to B.
**Ring detection** (diagnostic)
    One customer refunding across several merchants. Never a match decision.

THE THREE ARE NOT EQUALLY STRONG, AND THEY DO NOT GET THE SAME CONFIDENCE
--------------------------------------------------------------------------
Path closure is a deduction: the settlement edge is established and the file
itself says which payments are in the batch, so the conclusion is as certain as
the premise. It gets ``p = 1.0``.

Sibling completion is an *induction* -- "most of them went there, so the rest
did" is a good guess and not a proof. Its confidence is the fraction of
siblings that actually support it, so an 80% majority produces ``p = 0.80``,
which the configured ``tau_high`` routes to review rather than auto-matching.
Giving an inductive rule the same certainty as a deductive one would be the
single easiest way to turn this tier into a false-positive generator.

Exclusivity is neither: it produces no candidates, only refusals, and every
refusal is counted so the tier can show what it stopped.

WHY THIS TIER FINDS LITTLE ON THIS CORPUS, AND WHY THAT IS REPORTED
--------------------------------------------------------------------
Every earlier tier matches at **settlement granularity** -- it establishes
``S -> C`` and expands the whole batch in one go. So the partial assignments
that path closure and sibling completion exist to finish do not arise, and both
rules correctly fire zero times on the generated corpus.

That is reported rather than engineered around. A tier that manufactured work
by loosening a rule until it fired would be trading precision for the
appearance of contribution, which is the trap this project is built to avoid.
Exclusivity still does real work: it is what guarantees the other two can never
overfill a credit, and it is measured by what it blocks.

NO NEO4J, AND NO NETWORKX EITHER
---------------------------------
``graph/interface.py`` settled Neo4j at Step 0: PLAN.md §4 required the fallback
to produce *identical* decisions, which concedes the database buys nothing. The
same argument disposes of NetworkX -- these four rules are adjacency lookups
over a few hundred nodes, and :mod:`ledgerloop.graph.memory_repo` implements the
Protocol in about as many lines as the import would cost. The Protocol is what
keeps either a drop-in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ledgerloop.config import DecisionThresholds, GraphInference
from ledgerloop.graph.memory_repo import MemoryGraphRepo
from ledgerloop.matching.bank_leg import allocated_share_minor, attribute_clawback, candidate_id
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, OrderStatus, Tier
from ledgerloop.models.records import CanonicalPayment
from ledgerloop.models.refs import bank_ref, order_ref, payment_ref, settlement_ref
from ledgerloop.money import allocate_minor, format_minor, sum_minor

__all__ = [
    "GraphOutcome",
    "RingFinding",
    "build_graph",
    "detect_rings",
    "run_tier4",
]


@dataclass(frozen=True)
class RingFinding:
    """A customer reference refunding across several merchants.

    PLAN.md §6.4 is explicit that this is "a bonus signal in the exception
    report, **not a match decision**", so it produces no candidates. It is
    surfaced for the Step 8 exception queue to carry.
    """

    customer_ref: str
    events: int
    merchants: tuple[str, ...]
    orders: tuple[str, ...]


@dataclass(frozen=True)
class GraphOutcome:
    """What one T4 pass produced, and what it refused to."""

    candidates: tuple[MatchCandidate, ...] = ()
    nodes: int = 0
    edges: int = 0
    credits_fully_absorbed: int = 0
    inferences_blocked: int = 0
    path_closures: int = 0
    sibling_completions: int = 0
    rings: tuple[RingFinding, ...] = field(default=())

    @property
    def tier(self) -> Tier:
        return Tier.T4_GRAPH

    @property
    def payment_links(self) -> int:
        return sum(1 for c in self.candidates if c.is_evaluable)


def build_graph(
    context: MatchContext, established: tuple[MatchCandidate, ...]
) -> MemoryGraphRepo:
    """Assemble the entity graph from the sources plus what the tiers established.

    Two kinds of edge, and the distinction is the whole point of the tier:

    * **Asserted** -- ``ORDER_PAID_BY`` and ``PAYMENT_SETTLED_IN`` come from the
      sources' own references and nesting. No tier earned them.
    * **Inferred** -- ``SETTLEMENT_CREDITED_AS`` and ``PAYMENT_CREDITED_AS``
      come from candidates an earlier tier proposed *and* the policy would
      accept. A candidate the policy would route to review is not a premise.
    """
    repo = MemoryGraphRepo()
    for order in context.orders:
        repo.add_node(order_ref(order.order_id), merchant_id=order.merchant_id)
    for view in context.settlements:
        repo.add_node(settlement_ref(view.settlement_id))
        for payment in view.payments:
            repo.add_node(payment_ref(payment.payment_id))
            repo.add_edge(
                payment_ref(payment.payment_id),
                settlement_ref(view.settlement_id),
                LinkType.PAYMENT_SETTLED_IN,
            )
            reference = payment.order_ref_normalized
            if reference is not None and reference in context.orders_by_id:
                repo.add_edge(
                    order_ref(reference),
                    payment_ref(payment.payment_id),
                    LinkType.ORDER_PAID_BY,
                )
    for txn in context.bank_txns:
        if txn.is_credit:
            repo.add_node(bank_ref(txn.txn_id), amount_minor=txn.credit_minor)

    for candidate in established:
        repo.add_edge(candidate.source_ref, candidate.target_ref, candidate.link_type)
    return repo


def _absorbed(established: tuple[MatchCandidate, ...]) -> dict[str, int]:
    """How much of each credit the established payment links already claim."""
    totals: dict[str, int] = defaultdict(int)
    for candidate in established:
        if candidate.link_type is not LinkType.PAYMENT_CREDITED_AS:
            continue
        totals[candidate.target_ref.record_id] += allocated_share_minor(candidate)
    return dict(totals)


def detect_rings(context: MatchContext, graph: GraphInference) -> tuple[RingFinding, ...]:
    """Customer references refunding across several merchants.

    Read off the ledger's own ``REFUNDED`` status -- the only refund signal in
    the three sources, and the one anomaly A06 sets. Grouping by
    ``customer_ref`` and counting *distinct merchants* is what separates a
    difficult customer from a pattern worth a controller's attention.

    Produces no candidates. PLAN.md §6.4 calls this a bonus signal, not a match
    decision, and a fraud heuristic that could move a match would be exactly
    the kind of unaccountable behaviour the project argues against.
    """
    by_customer: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for order in context.orders:
        if order.status is OrderStatus.REFUNDED:
            by_customer[order.customer_ref].append((order.merchant_id, order.order_id))

    findings: list[RingFinding] = []
    for customer, events in sorted(by_customer.items()):
        merchants = sorted({merchant for merchant, _ in events})
        if len(events) >= graph.ring_min_events and len(merchants) >= graph.ring_min_merchants:
            findings.append(
                RingFinding(
                    customer_ref=customer,
                    events=len(events),
                    merchants=tuple(merchants),
                    orders=tuple(sorted(order for _, order in events)),
                )
            )
    return tuple(findings)


def _covered_payments(view: SettlementView) -> tuple[CanonicalPayment, ...]:
    """The payments whose money actually travelled, A08's exclusion applied.

    A charged-back payment must never be *completed* onto a credit: its money
    never reached the bank, so inferring a link for it would be the tier
    inventing the very error the earlier tiers took care to avoid.
    """
    clawback = attribute_clawback(view)
    excluded = clawback.excluded.payment_id if clawback.excluded is not None else None
    return tuple(p for p in view.payments if p.payment_id != excluded)


def _infer_candidates(
    view: SettlementView,
    credit_id: str,
    missing: tuple[CanonicalPayment, ...],
    *,
    residual_minor: int,
    probability: float,
    rule: str,
    detail: str,
    support: float,
) -> list[MatchCandidate]:
    """Turn one rule's conclusion into payment links over the credit's residual."""
    shares = allocate_minor(residual_minor, [p.amount_minor for p in missing])
    conserved = sum_minor(shares, field=f"{credit_id}.t4") == residual_minor
    candidates: list[MatchCandidate] = []
    for payment, share in zip(missing, shares, strict=True):
        candidates.append(
            MatchCandidate(
                candidate_id=candidate_id(
                    Tier.T4_GRAPH,
                    LinkType.PAYMENT_CREDITED_AS,
                    payment_ref(payment.payment_id).key,
                    bank_ref(credit_id).key,
                ),
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(payment.payment_id),
                target_ref=bank_ref(credit_id),
                tier=Tier.T4_GRAPH,
                features=FeatureVector(
                    tier=Tier.T4_GRAPH,
                    amount_delta_minor=0,
                    graph_support=support,
                ),
                evidence=(
                    Evidence(
                        kind=EvidenceKind.GRAPH_RULE,
                        detail=f"{rule}: {detail}",
                        refs=(
                            payment_ref(payment.payment_id),
                            settlement_ref(view.settlement_id),
                            bank_ref(credit_id),
                        ),
                        score=support,
                    ),
                    Evidence(
                        kind=EvidenceKind.ARITHMETIC_CHECK,
                        detail=(
                            f"allocated {format_minor(share)} of the "
                            f"{format_minor(residual_minor)} still unclaimed on "
                            f"{credit_id}, by gross weight"
                        ),
                        refs=(payment_ref(payment.payment_id), bank_ref(credit_id)),
                        amount_minor=share,
                    ),
                ),
                calibrated_p=probability,
                arithmetic_verified=conserved,
            )
        )
    return candidates


def run_tier4(
    context: MatchContext,
    established: tuple[MatchCandidate, ...],
    graph_config: GraphInference,
    thresholds: DecisionThresholds,
) -> GraphOutcome:
    """Propagate constraints over what the earlier tiers established.

    ``established`` is filtered to the candidates the policy would accept --
    an inference built on a premise headed for review would inherit its doubt
    without inheriting its caveat.
    """
    premises = tuple(
        candidate
        for candidate in established
        if candidate.calibrated_p is not None
        and candidate.calibrated_p >= thresholds.tau_high
        and candidate.arithmetic_verified
    )
    repo = build_graph(context, premises)
    absorbed = _absorbed(premises)

    # -- exclusivity pruning ---------------------------------------------
    fully_absorbed = 0
    capacity: dict[str, int] = {}
    for txn in context.bank_txns:
        if not txn.is_credit:
            continue
        remaining = txn.credit_minor - absorbed.get(txn.txn_id, 0)
        capacity[txn.txn_id] = remaining
        if remaining <= 0 and absorbed.get(txn.txn_id, 0) > 0:
            repo.mark_consumed(bank_ref(txn.txn_id))
            fully_absorbed += 1

    linked_payments: dict[str, str] = {
        candidate.source_ref.record_id: candidate.target_ref.record_id
        for candidate in premises
        if candidate.link_type is LinkType.PAYMENT_CREDITED_AS
    }
    settlement_credits: dict[str, list[str]] = defaultdict(list)
    for candidate in premises:
        if candidate.link_type is LinkType.SETTLEMENT_CREDITED_AS:
            settlement_credits[candidate.source_ref.record_id].append(
                candidate.target_ref.record_id
            )

    candidates: list[MatchCandidate] = []
    blocked = closures = completions = 0

    for view in context.settlements:
        covered = _covered_payments(view)
        if not covered:
            continue
        missing = tuple(p for p in covered if p.payment_id not in linked_payments)
        if not missing:
            continue

        # -- path closure: S -> C established, so every payment of S is in C --
        credits = settlement_credits.get(view.settlement_id, [])
        if len(credits) == 1:
            credit_id = credits[0]
            if capacity.get(credit_id, 0) <= 0:
                blocked += 1
            else:
                candidates.extend(
                    _infer_candidates(
                        view,
                        credit_id,
                        missing,
                        residual_minor=capacity[credit_id],
                        probability=1.0,
                        rule="path closure",
                        detail=(
                            f"{view.settlement_id} is credited as {credit_id}, and these "
                            f"{len(missing)} payment(s) sit in that settlement by the "
                            "PSP's own nesting, so their money is in that credit"
                        ),
                        support=1.0,
                    )
                )
                capacity[credit_id] = 0
                closures += 1
            continue

        # -- sibling completion: most of the batch already points at one credit --
        votes: dict[str, int] = defaultdict(int)
        for payment in covered:
            target = linked_payments.get(payment.payment_id)
            if target is not None:
                votes[target] += 1
        if not votes:
            continue
        winner, support_count = max(sorted(votes.items()), key=lambda item: item[1])
        support = support_count / len(covered)
        if support < graph_config.sibling_completion_threshold:
            continue
        if capacity.get(winner, 0) <= 0:
            blocked += 1
            continue
        candidates.extend(
            _infer_candidates(
                view,
                winner,
                missing,
                residual_minor=capacity[winner],
                probability=support,
                rule="sibling completion",
                detail=(
                    f"{support_count} of {len(covered)} payment(s) in "
                    f"{view.settlement_id} are already credited to {winner} "
                    f"({support:.0%} >= "
                    f"{graph_config.sibling_completion_threshold:.0%}), constraining the "
                    f"remaining {len(missing)}"
                ),
                support=support,
            )
        )
        capacity[winner] = 0
        completions += 1

    return GraphOutcome(
        candidates=tuple(candidates),
        nodes=len(repo),
        edges=sum(
            len(repo.edges_of_type(link_type))
            for link_type in LinkType
        ),
        credits_fully_absorbed=fully_absorbed,
        inferences_blocked=blocked,
        path_closures=closures,
        sibling_completions=completions,
        rings=detect_rings(context, graph_config),
    )
