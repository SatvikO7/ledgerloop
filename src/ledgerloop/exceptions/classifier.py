"""Turning a finished run's residual into a typed, priced, evidenced queue.

PLAN.md 8.2.2: "a bare 'unmatched' count is not a deliverable". This module is
what makes that true -- every item the ladder could not resolve comes out of it
carrying a class, a severity, a rupee figure, an evidence chain that points back
at source records, a root cause and an action.

WHAT IT READS, AND WHAT IT MAY NOT
----------------------------------
It reads the :class:`~ledgerloop.matching.context.MatchContext` (the sources)
and the run's decisions. It does **not** read ground truth: an anomaly label is
the answer key, and a classifier that consulted it would make every number in
the confusion matrix meaningless. The confusion matrix is assembled afterwards,
by the evaluator, from two independently produced vocabularies.

It also does not call an LLM. Class, severity and impact are deterministic
functions of the three source documents, and Step 9 may only *rewrite the prose*
on an exception this module already built -- recorded on the exception itself as
``root_cause_source`` / ``suggested_action_source``.

ONE EXCEPTION PER RESIDUAL RECORD
---------------------------------
Not one per decision. A contested settlement produces several decisions and one
problem, and a queue that listed it four times would be measuring the matcher's
internals rather than the controller's workload. Decisions are folded in as
evidence and, for an ambiguity, as competing hypotheses.

THE QUEUE IS SORTED BY MONEY
----------------------------
PLAN.md 8.2.3. Descending ``impact_minor``, then by id so the order is total
and a rerun produces the same document. A controller cares about the one ₹4
lakh payout, not the two hundred one-paise drifts, and a queue sorted by count
or by class hides exactly that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from ledgerloop.config import RunConfig
from ledgerloop.exceptions.taxonomy import (
    ClawbackItem,
    CreditItem,
    PaymentItem,
    SettlementItem,
    classify_credit,
    classify_payment,
    classify_settlement,
    clawback_items,
    residual_items,
    severity_for,
)
from ledgerloop.exceptions.templates import PROSE_VERSION, prose_for
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.candidates import Evidence, MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import (
    DecisionOutcome,
    EvidenceKind,
    ExceptionClass,
    LinkType,
    Severity,
)
from ledgerloop.models.recon_exception import Hypothesis, ReconException
from ledgerloop.models.records import CanonicalPayment
from ledgerloop.models.refs import (
    RecordRef,
    bank_ref,
    order_ref,
    payment_ref,
    settlement_ref,
)
from ledgerloop.money import format_minor

__all__ = [
    "ExceptionOutcome",
    "classify_exceptions",
    "exception_id",
    "queue_order",
]


def exception_id(subject_key: str) -> str:
    """A stable id derived from the record the exception is about.

    Content-derived rather than a counter, for the same reason
    :func:`~ledgerloop.matching.policy.decision_id` is: two runs over the same
    data must produce comparable queues, and a counter renumbers everything the
    moment one item appears or disappears.
    """
    return f"exception:{subject_key}"


@dataclass(frozen=True)
class ExceptionOutcome:
    """The queue, plus the counters that explain how it was built."""

    exceptions: tuple[ReconException, ...] = ()
    settlements_seen: int = 0
    credits_seen: int = 0
    payments_seen: int = 0
    clawbacks_seen: int = 0
    """Claw-backs traced to a refund from an earlier batch (Phase 2.3)."""
    ambiguities: int = 0
    debits_ignored: int = 0
    """Outgoing rows the queue does not cover.

    Reported rather than absorbed. A debit is money leaving the account, not a
    payout being reconciled, so it is outside the unit -- and saying so with a
    number is the difference between a scope decision and a silent omission.
    """

    @property
    def total_impact_minor(self) -> int:
        return sum(item.impact_minor for item in self.exceptions)

    @property
    def by_class(self) -> dict[ExceptionClass, int]:
        counts: dict[ExceptionClass, int] = {}
        for item in self.exceptions:
            counts[item.exception_class] = counts.get(item.exception_class, 0) + 1
        return {
            exception_class: counts[exception_class]
            for exception_class in ExceptionClass
            if exception_class in counts
        }

    @property
    def by_severity(self) -> dict[Severity, int]:
        counts: dict[Severity, int] = {}
        for item in self.exceptions:
            counts[item.severity] = counts.get(item.severity, 0) + 1
        return {
            severity: counts[severity] for severity in Severity if severity in counts
        }

    @property
    def unmatchable(self) -> tuple[ReconException, ...]:
        """The honest floor: items no system could resolve from these sources."""
        return tuple(
            item
            for item in self.exceptions
            if item.exception_class is ExceptionClass.UNMATCHABLE
        )

    @property
    def resolvable(self) -> tuple[ReconException, ...]:
        return tuple(item for item in self.exceptions if item.resolvable_by_agent)

    def covering(self, ref_key: str) -> tuple[ReconException, ...]:
        """Every exception naming this record. The evaluator's lookup."""
        return tuple(
            item
            for item in self.exceptions
            if any(ref.key == ref_key for ref in item.involved_refs)
        )


@dataclass
class _DecisionIndex:
    """What the policy concluded, indexed by the record each ruling touches."""

    by_settlement: dict[str, list[MatchDecision]] = field(default_factory=dict)
    by_credit: dict[str, list[MatchDecision]] = field(default_factory=dict)
    candidates: dict[str, MatchCandidate] = field(default_factory=dict)
    matched_settlements: set[str] = field(default_factory=set)
    matched_credits: set[str] = field(default_factory=set)
    matched_payments: set[str] = field(default_factory=set)
    ambiguous_settlements: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        decisions: Sequence[MatchDecision],
        candidates: Sequence[MatchCandidate],
    ) -> _DecisionIndex:
        index = cls(candidates={c.candidate_id: c for c in candidates})
        for decision in decisions:
            if decision.link_type is LinkType.SETTLEMENT_CREDITED_AS:
                index.by_settlement.setdefault(
                    decision.source_ref.record_id, []
                ).append(decision)
            if decision.link_type in (
                LinkType.SETTLEMENT_CREDITED_AS,
                LinkType.PAYMENT_CREDITED_AS,
            ):
                index.by_credit.setdefault(decision.target_ref.record_id, []).append(
                    decision
                )
            if decision.outcome is DecisionOutcome.AUTO_MATCHED:
                if decision.link_type is LinkType.SETTLEMENT_CREDITED_AS:
                    index.matched_settlements.add(decision.source_ref.record_id)
                if decision.link_type is LinkType.PAYMENT_CREDITED_AS:
                    index.matched_payments.add(decision.source_ref.record_id)
                if decision.link_type in (
                    LinkType.SETTLEMENT_CREDITED_AS,
                    LinkType.PAYMENT_CREDITED_AS,
                ):
                    index.matched_credits.add(decision.target_ref.record_id)

        # An ambiguity is what the *tier* concluded, read off the evidence it
        # attached rather than re-derived here. Re-running the subset search
        # could reach a different verdict from the tier that actually declined,
        # which would put one story in the queue and another in the audit log.
        for candidate in candidates:
            if candidate.link_type is not LinkType.SETTLEMENT_CREDITED_AS:
                continue
            if any(
                item.kind is EvidenceKind.NEGATIVE_EVIDENCE
                and "different subsets" in item.detail
                for item in candidate.evidence
            ):
                index.ambiguous_settlements.add(candidate.source_ref.record_id)
        return index

    def hypotheses_for(self, settlement_id: str) -> tuple[Hypothesis, ...]:
        """Competing explanations, preserved rather than collapsed (PLAN.md 8.2.4)."""
        found: list[Hypothesis] = []
        for decision in self.by_settlement.get(settlement_id, ()):
            candidate = self.candidates.get(decision.candidate_id)
            if candidate is None:  # pragma: no cover - every decision has one
                continue
            subset = candidate.subset_members
            summary = (
                f"{settlement_id} was credited as {decision.target_ref.record_id}"
                + (f" carrying {len(subset)} payment(s)" if subset else "")
            )
            found.append(
                Hypothesis(
                    summary=summary,
                    probability=decision.calibrated_p,
                    implied_refs=(decision.source_ref, decision.target_ref, *subset),
                    evidence=candidate.evidence,
                )
            )
        found.sort(key=lambda item: (-item.probability, item.summary))
        return tuple(found)

    def evidence_for(self, refs: Sequence[str]) -> tuple[Evidence, ...]:
        """Every piece of evidence the tiers attached to a ruling on these records."""
        seen: set[str] = set()
        chain: list[Evidence] = []
        for decision in (
            decision
            for ref in refs
            for decision in (
                *self.by_settlement.get(ref, ()),
                *self.by_credit.get(ref, ()),
            )
        ):
            candidate = self.candidates.get(decision.candidate_id)
            if candidate is None:  # pragma: no cover
                continue
            for item in candidate.evidence:
                if item.detail not in seen:
                    seen.add(item.detail)
                    chain.append(item)
        return tuple(chain)


def _settlement_evidence(item: SettlementItem) -> tuple[Evidence, ...]:
    """What the sources themselves say about an uncredited payout."""
    view = item.view
    settlement = view.settlement
    chain: list[Evidence] = [
        Evidence(
            kind=EvidenceKind.ARITHMETIC_CHECK,
            detail=(
                f"{view.settlement_id} declares gross "
                f"{format_minor(settlement.gross_minor)}, fee "
                f"{format_minor(settlement.fee_minor)}, tax "
                f"{format_minor(settlement.tax_minor)}, adjustments "
                f"{format_minor(settlement.adjustments_minor)} and net "
                f"{format_minor(settlement.net_minor)} -- an identity that is off by "
                f"{format_minor(settlement.net_delta_minor)}"
            ),
            refs=(settlement_ref(view.settlement_id),),
            amount_minor=settlement.net_minor,
        ),
        Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{len(item.keyed_credits)} unclaimed credit(s) carry the reference "
                f"{settlement.utr or '(none published)'}, and "
                f"{len(item.near_amount_credits)} unclaimed credit(s) match the net "
                "on amount alone"
            ),
            refs=(settlement_ref(view.settlement_id),),
        ),
    ]
    clawback = item.clawback
    if clawback.present and clawback.excluded is not None:
        chain.append(
            Evidence(
                kind=EvidenceKind.ARITHMETIC_CHECK,
                detail=(
                    f"the negative adjustment of {format_minor(clawback.amount_minor)} "
                    f"equals the gross of {clawback.excluded.payment_id} exactly"
                ),
                refs=(payment_ref(clawback.excluded.payment_id),),
                amount_minor=clawback.amount_minor,
            )
        )
    for credit in item.keyed_credits:
        chain.append(
            Evidence(
                kind=EvidenceKind.EXACT_KEY,
                detail=(
                    f"{credit.txn_id} credits {format_minor(credit.credit_minor)} on "
                    f"{credit.value_date.isoformat()} under the same reference, "
                    f"{(credit.value_date - settlement.settled_on).days:+d} day(s) from "
                    "the settlement date"
                ),
                refs=(bank_ref(credit.txn_id),),
                amount_minor=credit.credit_minor,
            )
        )
    return tuple(chain)


def _credit_evidence(item: CreditItem) -> tuple[Evidence, ...]:
    credit = item.credit
    chain: list[Evidence] = [
        Evidence(
            kind=EvidenceKind.AMOUNT_MATCH,
            detail=(
                f"{credit.txn_id} credits {format_minor(credit.credit_minor)} on "
                f"{credit.value_date.isoformat()} and no settlement claimed it"
            ),
            refs=(bank_ref(credit.txn_id),),
            amount_minor=credit.credit_minor,
        ),
        Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                "narration published "
                + (
                    f"reference {credit.extracted_utr}"
                    if credit.extracted_utr
                    else "no reference"
                )
                + " and "
                + (
                    f"merchant {credit.extracted_merchant!r}"
                    if credit.extracted_merchant
                    else "no merchant name"
                )
            ),
            refs=(bank_ref(credit.txn_id),),
        ),
    ]
    original = item.reposting_of
    if original is not None:
        chain.append(
            Evidence(
                kind=EvidenceKind.EXACT_KEY,
                detail=(
                    f"{original.txn_id} credits the same "
                    f"{format_minor(original.credit_minor)} under the same narration "
                    f"on {original.value_date.isoformat()}, "
                    f"{(credit.value_date - original.value_date).days} day(s) earlier "
                    f"-- so {credit.txn_id} is a re-posting of it, not a second payout"
                ),
                refs=(bank_ref(original.txn_id),),
                amount_minor=original.credit_minor,
            )
        )
    for other in item.twin_credits:
        chain.append(
            Evidence(
                kind=EvidenceKind.EXACT_KEY,
                detail=(
                    f"{other.txn_id} carries the same reference and credits "
                    f"{format_minor(other.credit_minor)}"
                ),
                refs=(bank_ref(other.txn_id),),
                amount_minor=other.credit_minor,
            )
        )
    for view in item.keyed_settlements:
        chain.append(
            Evidence(
                kind=EvidenceKind.EXACT_KEY,
                detail=(
                    f"the reference names {view.settlement_id}, whose declared net is "
                    f"{format_minor(view.net_minor)}"
                ),
                refs=(settlement_ref(view.settlement_id),),
                amount_minor=view.net_minor,
            )
        )
    return tuple(chain)


def _order_refs(
    context: MatchContext, payments: Sequence[CanonicalPayment]
) -> tuple[RecordRef, ...]:
    """The orders behind a set of payments, where the reference resolves.

    An exception has to reach the order, because that is the record a
    controller recognises -- "SETL-0104 is short" means nothing to the person
    whose customer is asking about their refund. The chain payment -> order is
    the one T0's order leg already verifies, so following it here asserts
    nothing new.
    """
    found: list[RecordRef] = []
    seen: set[str] = set()
    for payment in payments:
        reference = payment.order_ref_normalized
        if reference is None or reference in seen or reference not in context.orders_by_id:
            continue
        seen.add(reference)
        found.append(order_ref(reference))
    return tuple(found)


def _payment_evidence(item: PaymentItem) -> tuple[Evidence, ...]:
    view = item.view
    settlement = view.settlement
    chain: list[Evidence] = [
        Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{item.payment.payment_id} is nested in {view.settlement_id}, which was "
                f"credited, but no bank row carries its "
                f"{format_minor(item.payment.amount_minor)}"
            ),
            refs=(payment_ref(item.key), settlement_ref(view.settlement_id)),
            amount_minor=item.payment.amount_minor,
        )
    ]
    if item.clawed_back:
        chain.append(
            Evidence(
                kind=EvidenceKind.ARITHMETIC_CHECK,
                detail=(
                    f"{view.settlement_id} declares adjustments of "
                    f"{format_minor(settlement.adjustments_minor)}, equal to the gross of "
                    f"{item.payment.payment_id} exactly -- the payment was netted off and "
                    "its money never reached the bank"
                ),
                refs=(settlement_ref(view.settlement_id), payment_ref(item.key)),
                amount_minor=item.payment.amount_minor,
            )
        )
    return tuple(chain)


def _confidence(exception_class: ExceptionClass, evidence_count: int) -> float:
    """Confidence in the **class**, never in a match.

    Two things move it, and both are properties of the classification rather
    than of any pairing:

    * ``UNKNOWN_RESIDUAL`` is by definition the class assigned when no rule
      fired, so asserting it confidently would be incoherent.
    * A rule that fired on the settlement's own arithmetic (the identity, an
      adjustment equal to a payment) is a deduction from the source document
      and gets 1.0; a rule that fired on the *absence* of something is weaker.
    """
    if exception_class is ExceptionClass.UNKNOWN_RESIDUAL:
        return 0.30
    deductive = {
        ExceptionClass.FEE_TAX_MISMATCH,
        ExceptionClass.CHARGEBACK_NETTED,
        ExceptionClass.DUPLICATE_CREDIT,
        ExceptionClass.AMBIGUOUS_AGGREGATION,
        ExceptionClass.UNMATCHABLE,
    }
    if exception_class in deductive:
        return 1.0
    return 0.90 if evidence_count >= 3 else 0.75


def _age_days(as_of: date, latest: date) -> int:
    return max(0, (latest - as_of).days)


def _latest_date(context: MatchContext) -> date:
    """The dataset's own latest date -- the clock this run ages items against.

    Never ``date.today()``. The report carries no timestamp anywhere so that two
    runs over the same data diff to nothing, and a severity that drifted upward
    as the calendar advanced would break that for the one table a controller
    reads first.
    """
    dates = [txn.value_date for txn in context.bank_txns]
    dates.extend(view.settlement.settled_on for view in context.settlements)
    return max(dates) if dates else date(1970, 1, 1)


def _build(
    *,
    subject: RecordRef,
    exception_class: ExceptionClass,
    impact_minor: int,
    involved: Sequence[RecordRef],
    evidence: Sequence[Evidence],
    severity: Severity,
    detail: str,
    counterpart: str | None,
    day_gap: int | None,
    hypotheses: Sequence[Hypothesis] = (),
) -> ReconException:
    prose = prose_for(
        exception_class,
        subject=subject.record_id,
        impact_minor=impact_minor,
        detail=detail,
        counterpart=counterpart,
        day_gap=day_gap,
    )
    return ReconException(
        exception_id=exception_id(subject.key),
        exception_class=exception_class,
        severity=severity,
        impact_minor=impact_minor,
        involved_refs=tuple(involved),
        evidence=tuple(evidence),
        root_cause=prose.root_cause,
        suggested_action=prose.suggested_action,
        classification_confidence=_confidence(exception_class, len(evidence)),
        resolvable_by_agent=False,
        hypotheses=tuple(hypotheses),
    )


def classify_exceptions(
    context: MatchContext,
    decisions: Sequence[MatchDecision],
    candidates: Sequence[MatchCandidate],
    config: RunConfig,
    *,
    merchant_profiles: frozenset[str] = frozenset(),
) -> ExceptionOutcome:
    """Build the exception queue for one finished run.

    ``merchant_profiles`` is the set of narration spellings T3 learned from the
    statement's own references. It is what separates "this credit lost its
    reference" from "this credit is from outside the ledger" -- and it is
    derived from the sources, so passing it in adds no information the
    classifier could not have computed itself.
    """
    index = _DecisionIndex.build(decisions, candidates)
    latest = _latest_date(context)
    settlement_items, credit_items, payment_items = residual_items(
        context,
        matched_settlements=frozenset(index.matched_settlements),
        matched_credits=frozenset(index.matched_credits),
        matched_payments=frozenset(index.matched_payments),
        merchant_profiles=merchant_profiles,
        tolerances=config.tolerances,
    )

    raised: list[ReconException] = []
    ambiguities = 0

    for item in settlement_items:
        ambiguous = item.key in index.ambiguous_settlements
        exception_class = classify_settlement(
            item,
            date_window_days=config.tolerances.date_window_days,
            ambiguous=ambiguous,
        )
        hypotheses = index.hypotheses_for(item.key) if ambiguous else ()
        if ambiguous and len(hypotheses) < 2:
            # The model requires two competing explanations for an ambiguity and
            # is right to: an ambiguity with one explanation is not one. Where
            # the log cannot supply a second, the item is reported as the
            # unexplained residual it has become rather than forced into a class
            # whose contract it does not satisfy.
            exception_class = ExceptionClass.UNKNOWN_RESIDUAL
            hypotheses = ()
        if hypotheses:
            ambiguities += 1

        clawback = item.clawback
        counterpart: str | None = None
        day_gap: int | None = None
        if exception_class is ExceptionClass.CHARGEBACK_NETTED and clawback.excluded:
            counterpart = clawback.excluded.payment_id
        elif item.keyed_credits:
            counterpart = item.keyed_credits[0].txn_id
            day_gap = (
                item.keyed_credits[0].value_date - item.view.settlement.settled_on
            ).days
        elif item.near_amount_credits:
            counterpart = item.near_amount_credits[0].txn_id
            day_gap = (
                item.near_amount_credits[0].value_date - item.view.settlement.settled_on
            ).days

        involved: list[RecordRef] = [settlement_ref(item.key)]
        involved.extend(payment_ref(p.payment_id) for p in item.view.payments)
        involved.extend(_order_refs(context, item.view.payments))
        involved.extend(bank_ref(txn.txn_id) for txn in item.keyed_credits)

        # A credited payout is only in the queue because its own identity does
        # not close, so the money at stake is the discrepancy -- the rest of the
        # payout arrived and is not at risk. Sorting a queue by a figure that
        # includes money nobody has lost would put the wrong item at the top.
        impact = abs(
            item.view.settlement.net_delta_minor
            if exception_class is ExceptionClass.FEE_TAX_MISMATCH
            else item.impact_minor
        )
        raised.append(
            _build(
                subject=settlement_ref(item.key),
                exception_class=exception_class,
                impact_minor=impact,
                involved=involved,
                evidence=(
                    *_settlement_evidence(item),
                    *index.evidence_for([item.key]),
                ),
                severity=severity_for(
                    impact,
                    age_days=_age_days(item.as_of, latest),
                    thresholds=config.severity,
                ),
                detail=f"{len(item.view.payments)} payment(s) are nested in this batch.",
                counterpart=counterpart,
                day_gap=day_gap,
                hypotheses=hypotheses,
            )
        )

    for credit in credit_items:
        exception_class = classify_credit(credit)
        involved = [bank_ref(credit.key)]
        if credit.reposting_of is not None:
            involved.append(bank_ref(credit.reposting_of.txn_id))
        involved.extend(bank_ref(other.txn_id) for other in credit.twin_credits)
        involved.extend(
            settlement_ref(view.settlement_id) for view in credit.keyed_settlements
        )
        counterpart = (
            credit.reposting_of.txn_id
            if credit.reposting_of is not None
            else (
                credit.twin_credits[0].txn_id
                if credit.twin_credits
                else (
                    credit.keyed_settlements[0].settlement_id
                    if credit.keyed_settlements
                    else None
                )
            )
        )
        # Which of two twins is the duplicate is only established when the
        # ladder credited one of them. Where it refused both -- T0's contested
        # case -- saying "already settled" would assert an ordering nothing
        # measured, so the prose says what is actually known instead.
        detail = ""
        if (
            exception_class is ExceptionClass.DUPLICATE_CREDIT
            and credit.reposting_of is None
            and not credit.matched_twins
        ):
            detail = (
                "Neither credit was matched, so which of the two is the genuine "
                "payout is not established by the sources."
            )
        raised.append(
            _build(
                subject=bank_ref(credit.key),
                exception_class=exception_class,
                impact_minor=credit.impact_minor,
                involved=involved,
                evidence=(
                    *_credit_evidence(credit),
                    *index.evidence_for([credit.key]),
                ),
                severity=severity_for(
                    credit.impact_minor,
                    age_days=_age_days(credit.as_of, latest),
                    thresholds=config.severity,
                ),
                detail=detail,
                counterpart=counterpart,
                # Present only for a re-posting the duplicate pass identified,
                # where the ordering IS the evidence and there may be no shared
                # reference to cite. `prose_for` branches on exactly that.
                day_gap=(
                    (credit.credit.value_date - credit.reposting_of.value_date).days
                    if credit.reposting_of is not None
                    else None
                ),
            )
        )

    clawbacks = clawback_items(
        context, matched_settlements=frozenset(index.matched_settlements)
    )
    for clawback_item in clawbacks:
        raised.append(_clawback_exception(clawback_item, config, latest=latest))

    for payment_item in payment_items:
        exception_class = classify_payment(payment_item)
        involved = [
            payment_ref(payment_item.key),
            settlement_ref(payment_item.view.settlement_id),
            *_order_refs(context, [payment_item.payment]),
        ]
        raised.append(
            _build(
                subject=payment_ref(payment_item.key),
                exception_class=exception_class,
                impact_minor=payment_item.impact_minor,
                involved=involved,
                evidence=_payment_evidence(payment_item),
                severity=severity_for(
                    payment_item.impact_minor,
                    age_days=_age_days(payment_item.as_of, latest),
                    thresholds=config.severity,
                ),
                detail="",
                counterpart=payment_item.view.settlement_id,
                day_gap=None,
            )
        )

    return ExceptionOutcome(
        exceptions=queue_order(raised),
        settlements_seen=len(settlement_items),
        credits_seen=len(credit_items),
        payments_seen=len(payment_items),
        clawbacks_seen=len(clawbacks),
        ambiguities=ambiguities,
        debits_ignored=sum(1 for txn in context.bank_txns if not txn.is_credit),
    )


def queue_order(exceptions: Sequence[ReconException]) -> tuple[ReconException, ...]:
    """Descending money, then id. PLAN.md 8.2.3, and a total order for replay."""
    return tuple(
        sorted(exceptions, key=lambda item: (-item.impact_minor, item.exception_id))
    )


#: The prose version every exception in a queue was written against.
PROSE = PROSE_VERSION
__all__ += ["PROSE"]


def _clawback_exception(
    item: ClawbackItem, config: RunConfig, *, latest: date
) -> ReconException:
    """The queue row for a refund taken out of somebody else's batch.

    The class is ``POST_SETTLEMENT_REFUND`` in both the attributed and the
    ambiguous case -- what the system concluded is the same thing either way,
    and only the *subject* differs. Softening the ambiguous one to
    ``UNKNOWN_RESIDUAL`` would throw away the part that is known.
    """
    attributed = item.attributed
    subject = (
        order_ref(attributed.order_id)
        if attributed is not None
        else settlement_ref(item.view.settlement_id)
    )
    involved: list[RecordRef] = [
        settlement_ref(item.view.settlement_id),
        *(order_ref(order.order_id) for order in item.refunded_orders),
        *(settlement_ref(view.settlement_id) for view in item.source_settlements),
    ]
    if attributed is not None:
        involved.insert(0, subject)

    origin = item.source_settlements[0] if item.source_settlements else None
    detail = (
        f"{len(item.refunded_orders)} refunded orders carry this amount, so the "
        "adjustment does not name one of them."
        if attributed is None
        else ""
    )
    return _build(
        subject=subject,
        exception_class=ExceptionClass.POST_SETTLEMENT_REFUND,
        impact_minor=item.impact_minor,
        involved=involved,
        evidence=_clawback_evidence(item),
        severity=severity_for(
            item.impact_minor,
            age_days=_age_days(item.as_of, latest),
            thresholds=config.severity,
        ),
        detail=detail,
        counterpart=origin.settlement_id if origin is not None else item.view.settlement_id,
        day_gap=None,
    )


def _clawback_evidence(item: ClawbackItem) -> tuple[Evidence, ...]:
    """The two source facts, each pointing at the document that states it."""
    settlement = settlement_ref(item.view.settlement_id)
    evidence = [
        Evidence(
            kind=EvidenceKind.ARITHMETIC_CHECK,
            detail=(
                f"{item.view.settlement_id} declares adjustments of "
                f"{format_minor(-item.amount_minor)}, which matches none of its "
                f"{len(item.view.payments)} nested payment(s) -- so the money it "
                "accounts for was not paid out in this batch"
            ),
            refs=(settlement,),
            amount_minor=item.amount_minor,
        )
    ]
    for order in item.refunded_orders:
        origin = next(
            (
                view
                for view in item.source_settlements
                if any(
                    payment.order_ref_normalized == order.order_id
                    for payment in view.payments
                )
            ),
            None,
        )
        where = (
            f" and was paid out in {origin.settlement_id}" if origin is not None else ""
        )
        evidence.append(
            Evidence(
                kind=EvidenceKind.EXACT_KEY,
                detail=(
                    f"the ledger marks {order.order_id} REFUNDED for "
                    f"{format_minor(order.amount_minor)}{where}"
                ),
                refs=(order_ref(order.order_id), settlement),
                amount_minor=order.amount_minor,
            )
        )
    return tuple(evidence)
