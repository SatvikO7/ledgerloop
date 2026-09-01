"""Construction model B -- the second corpus, built to disagree with the first.

WHY THIS EXISTS
---------------
Phase 2.10 removed T3's amount tolerance band on the strength of one empirical
claim, measured over 49 corpora:

    Where the match is made **without a reference**, the credit equals the
    settlement's net **exactly**.  (543 correct / 0 wrong at ``delta == 0``;
    0 correct / 1 wrong at ``delta != 0``.)

Every one of those 49 corpora came from :mod:`ledgerloop.generator.baseline`
plus :mod:`ledgerloop.generator.scenarios`. So the claim had one source of
evidence, and it is load-bearing for precision. This module is the second
source.

WHY THE FIRST GENERATOR CANNOT TEST ITS OWN ASSUMPTION
------------------------------------------------------
Model A defines the bank credit as ``settlement.declared_net_minor()``. Only
three things ever move it off that value:

* **A02 rounding drift** -- moves it by a few paise;
* **A05 duplicate credit** and **A09 split payout** -- change the *number* of
  credits, not the whole-net relation.

Nothing else can. A03, A06 and A08 all re-derive the credit *from* the declared
net, so they are exact by construction.

A02 is therefore the only class that can make a whole-net credit inexact -- and
A02 and A07 ``MISSING_REFERENCE`` **cannot land on the same settlement**. Both
call ``_claim_settlement(..., claims_primary_label=True)``, and that claim exists
because a :class:`~ledgerloop.models.truth.GroundTruthRecord` carries exactly one
``anomaly_class`` per record: two effects naming the same bank row would
overwrite each other's label.

So "the credit carries no reference" and "the credit is off its net" are mutually
exclusive in model A **for a truth-representation reason**, not a financial one.
Measured over those same 49 corpora, on whole-net settlements:

    referenced   : 1720 exact, 818 inexact   (32% of referenced credits drift)
    unreferenced :  654 exact,   0 inexact

A third of referenced credits legitimately miss their declared net. None of the
unreferenced ones do. That gap is the artefact, and no amount of re-running
model A can close it.

WHAT MODEL B DOES DIFFERENTLY
-----------------------------
One change, stated as a rule:

    **The bank decides what lands.** A credit is ``declared_net - deduction``,
    where the deduction belongs to the *bank relationship* and never appears in
    the PSP file: an inward-remittance charge and its GST, a rounding
    convention, a cross-border haircut.

Whether the bank wrote the UTR into the narration and whether it took a charge
are **independent** here. That independence is the whole point: it is the degree
of freedom model A cannot express, and it is the one the exactness rule is a
claim about.

Everything else is deliberately shared with model A -- the file format, the
merchant vocabulary, the fee identity, ``build_ground_truth``. Those are the
*interface*, not the construction model. Inventing a second merchant list would
add code without adding independence on the axis under test.

TRUTH STILL COMES FROM CONSTRUCTION
-----------------------------------
Links are read off ``DraftBankTxn.covered_payment_ids`` and verdicts off the
:class:`~ledgerloop.generator.world.ScenarioEffect` list, through the same
:func:`~ledgerloop.generator.ground_truth.build_ground_truth` model A uses.
Nothing is inferred back out of the emitted files, and **no case declares what
the matcher should do** -- :class:`Case` names the shape that was built, and
every outcome is measured afterwards.

No new :class:`~ledgerloop.models.enums.AnomalyClass` is introduced. Every case
is labelled with an existing one, because a twelfth class would move the
prevalence dial, the taxonomy, the confusion matrix and every published table --
for a corpus that is a probe, not a published split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from ledgerloop.generator.baseline import EPOCH, make_utr
from ledgerloop.generator.emitters import write_dataset
from ledgerloop.generator.ground_truth import build_ground_truth
from ledgerloop.generator.vocab import MERCHANTS, NARRATION_WITH_UTR, NARRATION_WITHOUT_UTR
from ledgerloop.generator.world import (
    DraftBankTxn,
    DraftOrder,
    DraftPayment,
    DraftSettlement,
    DraftWorld,
    ScenarioEffect,
)
from ledgerloop.models.enums import AnomalyClass, Difficulty, ExpectedStatus, OrderStatus, SplitName
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.models.truth import GroundTruth

__all__ = [
    "ADVERSARIAL_VERSION",
    "BANK_CHARGE_MINOR",
    "HAIRCUT_BPS",
    "LATE_DAYS",
    "LOOKALIKE_BPS",
    "AdversarialCorpus",
    "Case",
    "CaseRecord",
    "build_adversarial_corpus",
    "write_adversarial_corpus",
]

#: Recorded as the generator version, so a model-B corpus can never be mistaken
#: for a model-A one in a manifest, a run artefact or a report.
ADVERSARIAL_VERSION = "adversarial-0.1.0"

#: An inward-remittance charge of Rs 500 plus 18% GST, in paise. A *flat* fee on
#: the transfer, so it is the same on a Rs 2 lakh payout as on a Rs 6 lakh one --
#: which is exactly why it cannot be folded into the PSP's proportional fee
#: identity, and why the PSP file cannot know about it.
BANK_CHARGE_MINOR = 59_000

#: A cross-border payout haircut, in basis points. Proportional, unlike the flat
#: charge, so the corpus exercises both shapes a real deduction takes.
HAIRCUT_BPS = 35

#: How late "too late for T3" is. T3's window is +/-7 days; three weeks is
#: outside it by a margin no configuration change would quietly close.
LATE_DAYS = 21

#: How far a lookalike tranche sits from the net it imitates, in basis points.
#: 6 bps is the distance of the Phase 2.10 false positive -- ``SETL-0015`` taking
#: a tranche of ``SETL-0018``'s split payout 0.059% away. Inside the 50 bps pool
#: band, and nowhere near equal.
LOOKALIKE_BPS = 6


class Case(StrEnum):
    """The shape a settlement was built in. **Not** a prediction of the outcome.

    Every member names what model B constructed, so the per-case outcome table
    is a measurement rather than a restatement of an expectation.
    """

    REFERENCED_EXACT = "referenced_exact"
    """A UTR in the narration and a credit equal to the net. Also what teaches
    T3 the merchant's spellings -- profiles are built from referenced credits."""

    REFERENCED_DEDUCTED = "referenced_deducted"
    """A UTR *and* a bank charge. The control that makes the asymmetry visible:
    the same drift, with a reference standing in front of it."""

    UNREF_EXACT = "unref_exact"
    """No reference, credit equal to the net. The case Phase 2.10 kept."""

    UNREF_CHARGE = "unref_charge"
    """No reference, credit short by the flat inward charge."""

    UNREF_ROUNDED = "unref_rounded"
    """No reference, credit rounded down to the whole rupee."""

    UNREF_HAIRCUT = "unref_haircut"
    """No reference, credit short by a proportional cross-border haircut."""

    UNREF_TWIN_EXACT = "unref_twin_exact"
    """Two settlements of one merchant with identical nets, both unreferenced,
    both credited exactly, on the same day. Two exact candidates each."""

    UNREF_LATE_ONLY = "unref_late_only"
    """No reference, and the payout landed three weeks later -- so the pool is
    empty. No candidate of any kind."""

    UNREF_TRANCHE_BAIT = "unref_tranche_bait"
    """The Phase 2.10 false positive, rebuilt: unreferenced, its own payout
    outside the window, and a sibling's split tranche sitting 6 bps from its
    net."""

    UNREF_TRANCHE_HOST = "unref_tranche_host"
    """The sibling that owns that tranche. Unreferenced and split, so no keyed
    tier consumes its credits before T3 runs."""

    UNREF_CHARGEBACK_EXACT = "unref_chargeback_exact"
    """A chargeback netted off an unreferenced payout, credited exactly."""

    UNREF_CHARGEBACK_CHARGE = "unref_chargeback_charge"
    """A chargeback netted off an unreferenced payout, and a bank charge on top.
    Two independent reasons the money is short."""

    UNREF_SPLIT_EXACT = "unref_split_exact"
    """An unreferenced split payout whose two tranches sum to the net exactly."""

    UNREF_ORPHAN_NEAR = "unref_orphan_near"
    """An unreferenced settlement whose payout is late, beside an orphan credit
    carrying the merchant's name and an amount inside the band but not equal."""

    UNREF_ORPHAN_EXACT = "unref_orphan_exact"
    """The same, with the orphan equal to the paise -- constructed to be
    indistinguishable from a true match given only the three sources."""


@dataclass(frozen=True)
class CaseRecord:
    """What was built for one settlement. Carries no expectation about matching."""

    case: Case
    settlement_id: str
    merchant_id: str
    net_minor: int
    credit_ids: tuple[str, ...]
    credit_total_minor: int
    referenced: bool

    @property
    def delta_minor(self) -> int:
        """Credit total minus net. Zero for every exact case, by construction."""
        return self.credit_total_minor - self.net_minor


@dataclass(frozen=True)
class AdversarialCorpus:
    """A model-B world, its truth, and the case each settlement was built in."""

    world: DraftWorld
    truth: GroundTruth
    cases: tuple[CaseRecord, ...]
    seed: int

    def by_case(self) -> dict[Case, tuple[CaseRecord, ...]]:
        grouped: dict[Case, list[CaseRecord]] = {}
        for record in self.cases:
            grouped.setdefault(record.case, []).append(record)
        return {key: tuple(grouped[key]) for key in sorted(grouped, key=lambda c: c.value)}

    def case_of(self, settlement_id: str) -> Case | None:
        for record in self.cases:
            if record.settlement_id == settlement_id:
                return record.case
        return None

    @property
    def conservation_residual_minor(self) -> int:
        """Money unaccounted for after declared deductions. **Must be zero.**

        The same statement model A's :class:`GeneratedDataset` makes, and for the
        same reason: a deduction that moved money without declaring a
        ``bank_delta_minor`` would make every metric on this corpus a
        measurement of the wrong world.
        """
        return (
            self.world.settled_credit_total_minor()
            - self.world.declared_net_total_minor()
            - self.world.declared_bank_delta_minor()
        )


def _bps(amount_minor: int, bps: int) -> int:
    """Integer basis-point share, rounded up. Mirrors ``world._bps``."""
    return -(-amount_minor * bps // 10_000)


class _Builder:
    """Lays out one model-B world. Mutable, used once, discarded."""

    def __init__(self, rng: random.Random, merchant_count: int) -> None:
        self.rng = rng
        self.merchant_count = merchant_count
        self.world = DraftWorld()
        self.cases: list[CaseRecord] = []
        self._order_seq = 0
        self._payment_seq = 0
        self._settlement_seq = 0
        self._bank_seq = 0

    # -- primitives -----------------------------------------------------

    def _next_bank_id(self) -> str:
        self._bank_seq += 1
        return f"BNK-{self._bank_seq:05d}"

    def _variant(self, merchant_id: str) -> str:
        merchant = next(m for m in MERCHANTS if m.merchant_id == merchant_id)
        return merchant.variants[self.rng.randrange(len(merchant.variants))]

    def _batch(
        self,
        merchant_id: str,
        *,
        day: int,
        payments: int,
        amounts: list[int] | None = None,
    ) -> DraftSettlement:
        """One merchant batch: its orders, its payments, and the settlement.

        Order references are written cleanly. Model A corrupts a fifth of them to
        give T0 something to fail at; this corpus is a probe of one T3 rule, and
        a corruption rate would only add variance to the measurement it exists to
        make.
        """
        self._settlement_seq += 1
        booked = datetime.combine(EPOCH + timedelta(days=day), datetime.min.time())
        settled_on = (booked + timedelta(days=1)).date()
        settlement = DraftSettlement(
            settlement_id=f"SETL-{self._settlement_seq:04d}",
            merchant_id=merchant_id,
            utr=make_utr(self.rng, settled_on),
            settled_on=settled_on,
        )
        chosen = (
            amounts
            if amounts is not None
            else [self.rng.randrange(30_000, 4_800_000, 100) for _ in range(payments)]
        )
        for index, amount in enumerate(chosen):
            self._order_seq += 1
            self._payment_seq += 1
            order_id = f"ORD-2026-{self._order_seq:06d}"
            self.world.orders.append(
                DraftOrder(
                    order_id=order_id,
                    merchant_id=merchant_id,
                    customer_ref=f"CUST_{self.rng.randint(10_000, 19_999)}",
                    amount_minor=amount,
                    booked_at=booked + timedelta(hours=6 + index % 12),
                    status=OrderStatus.CAPTURED,
                )
            )
            payment_id = f"PAY-{self._payment_seq:05d}"
            self.world.payments.append(
                DraftPayment(
                    payment_id=payment_id,
                    order_id=order_id,
                    settlement_id=settlement.settlement_id,
                    amount_minor=amount,
                    captured_at=booked + timedelta(hours=6 + index % 12, seconds=17),
                    order_ref_raw=order_id,
                )
            )
            settlement.payment_ids.append(payment_id)
        self.world.settlements.append(settlement)
        # Model A reindexes once, after its baseline world is complete. Model B
        # reads a settlement's net *while it is laying out the next case* -- the
        # tranche lookalike is positioned against a net computed a moment
        # earlier -- so the index has to be current at every step, not at the end.
        self.world.reindex()
        return settlement

    def _narration(self, merchant_id: str, utr: str | None) -> str:
        variant = self._variant(merchant_id)
        if utr is None:
            template = NARRATION_WITHOUT_UTR[self.rng.randrange(len(NARRATION_WITHOUT_UTR))]
            return template.format(variant=variant)
        template = NARRATION_WITH_UTR[self.rng.randrange(len(NARRATION_WITH_UTR))]
        return template.format(variant=variant, utr=utr)

    def _credit(
        self,
        settlement: DraftSettlement,
        amount_minor: int,
        payment_ids: list[str],
        *,
        referenced: bool,
        days_late: int = 1,
    ) -> DraftBankTxn:
        txn = DraftBankTxn(
            txn_id=self._next_bank_id(),
            value_date=settlement.settled_on + timedelta(days=days_late),
            narration=self._narration(
                settlement.merchant_id, settlement.utr if referenced else None
            ),
            credit_minor=amount_minor,
            settlement_id=settlement.settlement_id,
            covered_payment_ids=list(payment_ids),
        )
        self.world.add_bank_txn(txn)
        return txn

    def _orphan(self, merchant_id: str, amount_minor: int, value_date: date) -> DraftBankTxn:
        """A credit with no settlement behind it -- A10, in model A's vocabulary.

        It carries the merchant's name and no reference, so it is exactly what
        T3 is allowed to look at, and it belongs to nothing. Linking it is a
        false positive by construction.
        """
        txn = DraftBankTxn(
            txn_id=self._next_bank_id(),
            value_date=value_date,
            narration=self._narration(merchant_id, None),
            credit_minor=amount_minor,
            settlement_id=None,
        )
        self.world.add_bank_txn(txn)
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.ORPHAN_BANK_CREDIT,
                primary_ref=bank_ref(txn.txn_id),
                expected_status=ExpectedStatus.UNMATCHABLE,
                impact_minor=amount_minor,
                note=(
                    f"{txn.txn_id} credits {amount_minor} paise under {merchant_id}'s name "
                    "with no settlement behind it"
                ),
            )
        )
        return txn

    def _net(self, settlement: DraftSettlement) -> int:
        return settlement.declared_net_minor(self.world.payments_by_id())

    def _record(
        self,
        case: Case,
        settlement: DraftSettlement,
        credits: list[DraftBankTxn],
        *,
        referenced: bool,
    ) -> None:
        self.cases.append(
            CaseRecord(
                case=case,
                settlement_id=settlement.settlement_id,
                merchant_id=settlement.merchant_id,
                net_minor=self._net(settlement),
                credit_ids=tuple(txn.txn_id for txn in credits),
                credit_total_minor=sum(txn.credit_minor for txn in credits),
                referenced=referenced,
            )
        )

    # -- declared effects -----------------------------------------------

    def _unreferenced_effect(self, settlement: DraftSettlement, txn: DraftBankTxn) -> None:
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.MISSING_REFERENCE,
                primary_ref=bank_ref(txn.txn_id),
                expected_status=ExpectedStatus.MATCHED,
                impact_minor=0,
                note=f"UTR {settlement.utr} absent from narration on {txn.txn_id}",
            )
        )

    def _deduction_effect(
        self, settlement: DraftSettlement, txn: DraftBankTxn, amount_minor: int, what: str
    ) -> None:
        """Declare a bank-side deduction, so conservation stays checkable.

        Labelled ``A02_ROUNDING_DRIFT`` -- the existing class for "the bank
        credited something other than the declared net, and the link is still
        real". The label is model A's vocabulary; the *magnitude* and the
        independence from the reference are model B's, and the note says so.
        """
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.ROUNDING_DRIFT,
                primary_ref=settlement_ref(settlement.settlement_id),
                expected_status=ExpectedStatus.MATCHED,
                impact_minor=amount_minor,
                bank_delta_minor=-amount_minor,
                note=(
                    f"bank deducted {amount_minor} paise ({what}) from "
                    f"{settlement.settlement_id}'s payout on {txn.txn_id}; the PSP file "
                    "never sees it"
                ),
            )
        )

    def _late_effect(self, settlement: DraftSettlement, txn: DraftBankTxn) -> None:
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.LATE_ARRIVAL,
                primary_ref=settlement_ref(settlement.settlement_id),
                expected_status=ExpectedStatus.MATCHED,
                impact_minor=0,
                note=(
                    f"{settlement.settlement_id}'s payout credited {LATE_DAYS} days after "
                    f"settlement, on {txn.txn_id}"
                ),
            )
        )

    def _chargeback(self, settlement: DraftSettlement) -> int:
        """Net a chargeback off the batch, exactly as model A's A08 does."""
        payments = self.world.payments_by_id()
        payment_id = settlement.payment_ids[0]
        amount = payments[payment_id].amount_minor
        settlement.adjustments_minor -= amount
        self.world.claimed_payments.add(payment_id)
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.CHARGEBACK_NETTED,
                primary_ref=payment_ref(payment_id),
                expected_status=ExpectedStatus.EXCEPTION,
                impact_minor=amount,
                note=(
                    f"chargeback of {amount} paise on {payment_id} netted off "
                    f"{settlement.settlement_id}; money never reached the bank"
                ),
            )
        )
        return amount

    def _carried(self, settlement: DraftSettlement) -> list[str]:
        """The payments whose money actually travelled -- chargebacks excluded."""
        return [pid for pid in settlement.payment_ids if pid not in self.world.claimed_payments]

    # -- the cases ------------------------------------------------------

    def _plain(
        self,
        case: Case,
        merchant_id: str,
        day: int,
        *,
        referenced: bool,
        deduction: int = 0,
        what: str = "",
        days_late: int = 1,
    ) -> DraftSettlement:
        """One batch, one credit, optionally short by a declared deduction."""
        settlement = self._batch(merchant_id, day=day, payments=self.rng.randint(6, 12))
        net = self._net(settlement)
        txn = self._credit(
            settlement,
            net - deduction,
            settlement.payment_ids,
            referenced=referenced,
            days_late=days_late,
        )
        if not referenced:
            self._unreferenced_effect(settlement, txn)
        if deduction:
            self._deduction_effect(settlement, txn, deduction, what)
        if days_late > 7:
            self._late_effect(settlement, txn)
        self._record(case, settlement, [txn], referenced=referenced)
        return settlement

    def _twins(self, merchant_id: str, day: int) -> None:
        """Two settlements a matcher has no way to tell apart.

        Identical nets, identical value dates, both unreferenced, both credited
        exactly. Model A cannot produce this: its amounts are drawn independently
        per order and collide only by accident.
        """
        amounts = [self.rng.randrange(30_000, 4_800_000, 100) for _ in range(8)]
        for _ in range(2):
            settlement = self._batch(merchant_id, day=day, payments=8, amounts=list(amounts))
            txn = self._credit(
                settlement, self._net(settlement), settlement.payment_ids, referenced=False
            )
            self._unreferenced_effect(settlement, txn)
            self._record(Case.UNREF_TWIN_EXACT, settlement, [txn], referenced=False)

    def _tranche_bait(self, merchant_id: str, day: int) -> None:
        """The Phase 2.10 false positive, rebuilt from the other direction.

        ``bait`` is unreferenced and its own payout arrives three weeks late, so
        the only same-merchant credit inside its window belongs to ``host`` -- a
        split payout whose first tranche is set 6 bps above ``bait``'s net.

        The tranche boundary is chosen directly rather than allocated from the
        payment weights, which is model B's other departure and the more
        realistic one: a bank pays out in the tranches *it* decides on, not in
        the proportions the PSP's payment table happens to imply.
        """
        # The bait is deliberately a small batch and the host a large one, so the
        # tranche that imitates the bait's net is genuinely a *fraction* of the
        # host's -- which is what made the Phase 2.10 case invisible to the
        # contention check. The host's own net is nowhere near the tranche, so
        # the host never claims it, and there is no rival to refuse over.
        bait = self._batch(
            merchant_id,
            day=day,
            payments=6,
            amounts=[self.rng.randrange(30_000, 600_000, 100) for _ in range(6)],
        )
        bait_net = self._net(bait)
        late = self._credit(
            bait, bait_net, bait.payment_ids, referenced=False, days_late=LATE_DAYS
        )
        self._unreferenced_effect(bait, late)
        self._late_effect(bait, late)
        self._record(Case.UNREF_TRANCHE_BAIT, bait, [late], referenced=False)

        lookalike = bait_net + _bps(bait_net, LOOKALIKE_BPS)
        host = self._batch(
            merchant_id,
            day=day,
            payments=14,
            amounts=[self.rng.randrange(2_400_000, 4_800_000, 100) for _ in range(14)],
        )
        host_net = self._net(host)
        if host_net - lookalike <= lookalike:  # pragma: no cover - the ranges guarantee it
            raise ValueError(
                f"model B could not seat a lookalike tranche of {lookalike} inside "
                f"{host.settlement_id}'s net of {host_net}"
            )
        cut = len(host.payment_ids) // 2
        first = self._credit(host, lookalike, host.payment_ids[:cut], referenced=False)
        second = self._credit(
            host,
            host_net - lookalike,
            host.payment_ids[cut:],
            referenced=False,
            days_late=2,
        )
        self._unreferenced_effect(host, first)
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.SPLIT_PAYOUT,
                primary_ref=bank_ref(second.txn_id),
                expected_status=ExpectedStatus.MATCHED,
                impact_minor=0,
                note=(
                    f"{host.settlement_id} paid out as {first.txn_id} + {second.txn_id} "
                    f"({lookalike} + {host_net - lookalike} paise); the first tranche sits "
                    f"{LOOKALIKE_BPS} bps from {bait.settlement_id}'s net"
                ),
            )
        )
        self._record(Case.UNREF_TRANCHE_HOST, host, [first, second], referenced=False)

    def _chargeback_case(self, case: Case, merchant_id: str, day: int, deduction: int) -> None:
        settlement = self._batch(merchant_id, day=day, payments=self.rng.randint(7, 12))
        self._chargeback(settlement)
        net = self._net(settlement)
        txn = self._credit(
            settlement, net - deduction, self._carried(settlement), referenced=False
        )
        self._unreferenced_effect(settlement, txn)
        if deduction:
            self._deduction_effect(settlement, txn, deduction, "inward remittance charge")
        self._record(case, settlement, [txn], referenced=False)

    def _split_exact(self, merchant_id: str, day: int) -> None:
        settlement = self._batch(merchant_id, day=day, payments=10)
        net = self._net(settlement)
        first_amount = net // 2
        first = self._credit(settlement, first_amount, settlement.payment_ids[:5], referenced=False)
        second = self._credit(
            settlement,
            net - first_amount,
            settlement.payment_ids[5:],
            referenced=False,
            days_late=2,
        )
        self._unreferenced_effect(settlement, first)
        self.world.effects.append(
            ScenarioEffect(
                anomaly=AnomalyClass.SPLIT_PAYOUT,
                primary_ref=bank_ref(second.txn_id),
                expected_status=ExpectedStatus.MATCHED,
                impact_minor=0,
                note=(
                    f"{settlement.settlement_id} paid out as {first.txn_id} + "
                    f"{second.txn_id} ({first_amount} + {net - first_amount} paise)"
                ),
            )
        )
        self._record(Case.UNREF_SPLIT_EXACT, settlement, [first, second], referenced=False)

    def _orphan_case(self, case: Case, merchant_id: str, day: int, *, exact: bool) -> None:
        """A late payout, and an orphan sitting where the payout should have been.

        The settlement's own credit is outside T3's window, so the only thing in
        its pool is a credit that belongs to nobody. ``exact`` decides whether
        the exactness rule can see the difference.
        """
        settlement = self._batch(merchant_id, day=day, payments=self.rng.randint(6, 10))
        net = self._net(settlement)
        late = self._credit(
            settlement, net, settlement.payment_ids, referenced=False, days_late=LATE_DAYS
        )
        self._unreferenced_effect(settlement, late)
        self._late_effect(settlement, late)
        amount = net if exact else net - _bps(net, LOOKALIKE_BPS)
        self._orphan(merchant_id, amount, settlement.settled_on + timedelta(days=1))
        self._record(case, settlement, [late], referenced=False)

    # -- layout ---------------------------------------------------------

    def build(self) -> None:
        """Every case, for every merchant, in a fixed order.

        The case list is the same for each merchant, so a merchant's identity is
        shared by sixteen settlements and same-merchant amounts are dense --
        which is the condition under which Phase 2.6's twenty-two false
        positives appeared at 5,000 orders.
        """
        for index in range(self.merchant_count):
            merchant_id = MERCHANTS[index].merchant_id
            day = index * 2
            # Two referenced batches first: T3 cannot build a profile for a
            # merchant the bank has never named alongside a UTR, and a corpus
            # where it could not would measure nothing.
            self._plain(Case.REFERENCED_EXACT, merchant_id, day, referenced=True)
            self._plain(Case.REFERENCED_EXACT, merchant_id, day + 1, referenced=True)
            self._plain(
                Case.REFERENCED_DEDUCTED,
                merchant_id,
                day + 2,
                referenced=True,
                deduction=BANK_CHARGE_MINOR,
                what="inward remittance charge",
            )
            self._plain(Case.UNREF_EXACT, merchant_id, day + 3, referenced=False)
            self._plain(
                Case.UNREF_CHARGE,
                merchant_id,
                day + 4,
                referenced=False,
                deduction=BANK_CHARGE_MINOR,
                what="inward remittance charge",
            )
            rounded = self._batch(merchant_id, day=day + 5, payments=9)
            rounded_net = self._net(rounded)
            paise = rounded_net % 100
            txn = self._credit(
                rounded, rounded_net - paise, rounded.payment_ids, referenced=False
            )
            self._unreferenced_effect(rounded, txn)
            if paise:
                self._deduction_effect(rounded, txn, paise, "rounded down to the rupee")
            self._record(Case.UNREF_ROUNDED, rounded, [txn], referenced=False)

            haircut = self._batch(merchant_id, day=day + 6, payments=9)
            haircut_net = self._net(haircut)
            cut = _bps(haircut_net, HAIRCUT_BPS)
            txn = self._credit(
                haircut, haircut_net - cut, haircut.payment_ids, referenced=False
            )
            self._unreferenced_effect(haircut, txn)
            self._deduction_effect(haircut, txn, cut, f"{HAIRCUT_BPS} bps cross-border haircut")
            self._record(Case.UNREF_HAIRCUT, haircut, [txn], referenced=False)

            self._twins(merchant_id, day + 7)
            self._plain(
                Case.UNREF_LATE_ONLY, merchant_id, day + 8, referenced=False, days_late=LATE_DAYS
            )
            self._tranche_bait(merchant_id, day + 9)
            self._chargeback_case(Case.UNREF_CHARGEBACK_EXACT, merchant_id, day + 10, 0)
            self._chargeback_case(
                Case.UNREF_CHARGEBACK_CHARGE, merchant_id, day + 11, BANK_CHARGE_MINOR
            )
            self._split_exact(merchant_id, day + 12)
            self._orphan_case(Case.UNREF_ORPHAN_NEAR, merchant_id, day + 13, exact=False)
            self._orphan_case(Case.UNREF_ORPHAN_EXACT, merchant_id, day + 14, exact=True)

        self.world.reindex()


def build_adversarial_corpus(seed: int = 42, *, merchants: int = 6) -> AdversarialCorpus:
    """Build one model-B corpus in memory.

    ``seed`` moves the amounts and the narration variants; it never moves the
    case layout, which is fixed. So a re-run at a different seed asks the same
    fifteen questions of different money, and the per-case outcome table stays
    comparable across seeds.
    """
    if not 1 <= merchants <= len(MERCHANTS):
        raise ValueError(
            f"model B needs between 1 and {len(MERCHANTS)} merchants; got {merchants}"
        )
    builder = _Builder(random.Random(f"{seed}:adversarial"), merchants)
    builder.build()
    truth = build_ground_truth(
        builder.world,
        split=SplitName.TEST,
        difficulty=Difficulty.STANDARD,
        seed=seed,
        generator_version=ADVERSARIAL_VERSION,
    )
    return AdversarialCorpus(
        world=builder.world, truth=truth, cases=tuple(builder.cases), seed=seed
    )


def write_adversarial_corpus(
    directory: Path, seed: int = 42, *, merchants: int = 6
) -> AdversarialCorpus:
    """Build a model-B corpus and emit the same five files model A emits.

    The corpus therefore goes through the real ingest, the real ladder and the
    real scorer -- there is no second run path, and no chance of measuring
    something the production pipeline would not do.
    """
    corpus = build_adversarial_corpus(seed, merchants=merchants)
    write_dataset(directory, corpus.world, corpus.truth)
    return corpus
