"""T3 -- lexical. Finding the payout when the reference is gone.

Anomaly A07 ``MISSING_REFERENCE`` strips the UTR out of a bank narration and
leaves only a merchant-name variant. Every keyed tier is blind to it: T0 and T1
join on the reference, and T2 needs the settlement's key on two credits before
it will look at a split. On the `test` split that is nine settlements and
**99 links, ₹21.9 lakh** -- by a wide margin the largest bucket left, and the
class PLAN.md §6.3 exists to close.

WHAT T3 CAN ACTUALLY MATCH ON
------------------------------
Not what it looks like. The bank writes ``RZRPAY SFTWR P L``; the ledger writes
``MRCH_0001``. **No source file maps one to the other** -- there is no merchant
master among the three inputs, and the strings share no characters. So the
comparison T3 needs does not exist until it is built.

It is built from the corpus's own evidence. Every credit that *does* carry a
UTR names a settlement, and that settlement's payments name orders, and those
orders name a merchant. So a keyed credit is a labelled example: *this* spelling
belongs to *that* merchant id. Collect them and the system has a merchant master
it derived rather than was given -- a profile per merchant of the spellings the
bank has actually used for it.

The profile is learned from the **reference**, not from our own matches. That
matters: a profile bootstrapped off T0/T1/T2's output would compound any error
they made, and would make T3's reach depend on how well earlier tiers happened
to do. The reference is the source's own assertion, so the profile is evidence
even where the match was declined.

THE THREE GATES
---------------
A name alone is not a match -- one merchant has many payouts. All three must
hold:

1. **Amount**, within T1's band. The credit is the whole net, so this is the
   same test T1 applies, and it is what separates one payout from the next.
2. **Date**, within :attr:`~ledgerloop.config.LexicalMatching.date_window_days`.
   Wider than T1's ±3 on purpose: T3's residual is precisely the batches other
   anomalies have already moved.
3. **Name**, at or above ``min_score``, and beating the runner-up by
   ``min_margin``.

WITHOUT A REFERENCE, THE AMOUNT *IS* THE IDENTITY CLAIM
--------------------------------------------------------
T0 and T1 match on a reference and then check the money, and their tolerance
band absorbs fee rounding around an identity the reference has already proven.
T3 has no reference. The amount is not a check on the match here -- it **is**
the match, together with a merchant name that every batch of that merchant
shares. An approximate identity claim, over a pool of same-merchant amounts, is
exactly where a false positive comes from.

That is decision 61's argument and this tier was not applying it. `find_tranche_set`
targets ``[net, net]`` with no band for the same reason: a payout conserves money
by construction, so an epsilon there would not absorb drift, it would admit sets
that are merely close.

Measured before it was written, over 49 corpora -- the 29 committed ones and the
20 (size, seed) scale corpora:

    delta == 0 :  543 correct,   0 wrong
    delta != 0 :    0 correct,   1 wrong

Every legitimate whole-net match is exact **to the paise**, and the one inexact
match in the entire corpus family is the false positive: `SETL-0015` taking
`BNK-00018` at 0.059% away, when that credit is a *tranche* of `SETL-0018`'s
split payout. So the band was admitting exactly one thing, and it was wrong.

The band still governs which credits are *considered* -- a near-miss rival must
still be able to contest a credit, and throwing that evidence away is the
mistake Phase 2.9 corrected. What exactness governs is the **assignment**.

UNIQUENESS HAS TWO DIRECTIONS
-----------------------------
Gate 3's margin asks *one* question: does this settlement have two credits the
scorer cannot separate? That leaves the mirror-image question unasked -- do two
**settlements** both have a claim on this credit? They are different failures
and only the first was being caught.

The second one cannot arise on a small corpus and is close to certain on a large
one. A merchant's payouts are a few lakh apart on a base of several crore, so two
settlements of the *same* merchant land inside T1's tolerance band of each other
as soon as that merchant has enough of them; the date window then admits both,
and the name is by construction identical because it is the same merchant. On the
committed corpora (60-400 orders) this happens **zero** times. At 2,500 orders it
happens once and at 5,000 it happens nine times -- and a settlement-ordered greedy
loop hands the credit to whichever settlement it reached first, at ``p = 1.0``
with the arithmetic verifying, because the amounts really do agree.

So the credit side carries the same test as the settlement side, against the same
``min_margin`` and with no new constant: a credit claimed by a second settlement
that the score cannot put behind the first is **contested**, and every claimant
refuses it. Two settlements of one merchant always score identically, so a
same-merchant contest can never be resolved on the name -- which is precisely
why refusing is the only honest answer available to a *lexical* tier.

ALREADY-REFERENCED SETTLEMENTS ARE NOT T3'S WORK
-------------------------------------------------
The rule below says a *credit* carrying a reference is already explained. The
same evidence runs the other way and was not being applied: a **settlement**
whose UTR the bank has written on some credit has already been told where its
money went, and a whole-net match to a different, unreferenced credit contradicts
the statement rather than reading it.

The case that makes this concrete is A09 composed with A07 on one side only. A
settlement pays out in two tranches; one tranche keeps the settlement's UTR and
the other loses it. T2 will not close it, because the keyed tranches do not sum
to the whole net -- correctly, half the money is missing. The settlement is
therefore still open when T3 reaches it, and T3's amount gate compares a *whole*
net against single credits. On a large statement some other settlement's tranche
lands inside that band: at 5,000 orders two do, 0.27% and 0.22% away from a net
several crore wide, inside the date window, spelt with the same merchant name
because it *is* the same merchant. Every gate passes and the tier auto-matches at
``p = 1.0`` with the arithmetic verifying, because the amounts genuinely agree.

Twenty-two wrong links on one corpus came from exactly that, and none of them is
reachable by tightening a threshold: the evidence T3 reads really does point that
way. What rules them out is evidence T3 was not consulting -- the settlement's own
UTR, sitting on a credit elsewhere in the statement. Finding the missing tranche
of a partly-referenced payout is T2's arithmetic, where the sum has to close; it
is not something a name and an amount band should be allowed to guess at.

WHY THE CANDIDATE MUST BE UNREFERENCED
---------------------------------------
T3 only considers credits whose ``extracted_utr`` is ``None``. A credit that
carries a reference is already explained by whatever that reference names --
its amount agreeing with some other settlement is a coincidence, and matching
it on a name would be overruling the bank's own statement of where the money
went. This is the same rule Step 4's rival check applies, for the same reason.

THE PROBABILITY
---------------
``p = score / n`` -- the similarity, divided uniformly among candidates the
scorer could not separate. At an exact skeleton match with one candidate this
is 1.0 and the configured ``tau_high`` auto-matches it; a name that only
resembles routes to review, and a two-way tie lands below ``tau_low`` and
becomes an exception. Same convention as every tier before it, and no new
constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from ledgerloop.config import LexicalMatching, MatchingTolerances
from ledgerloop.ingest.normalize import merchant_skeleton, normalize_merchant_name
from ledgerloop.matching.bank_leg import attribute_clawback, candidate_id
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.models.candidates import Evidence, FeatureVector, MatchCandidate
from ledgerloop.models.enums import EvidenceKind, LinkType, Tier
from ledgerloop.models.records import CanonicalBankTxn
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.money import (
    allocate_minor,
    delta_ratio,
    format_minor,
    sum_minor,
    tolerance_minor,
    within_tolerance,
)

__all__ = [
    "LexicalOutcome",
    "MerchantProfile",
    "NameMatch",
    "build_profiles",
    "candidate_credits",
    "features_for",
    "rank_credits",
    "run_tier3",
    "score_names",
]


@dataclass(frozen=True)
class MerchantProfile:
    """The spellings a bank has been seen to use for one merchant."""

    merchant_id: str
    spellings: tuple[str, ...]
    skeletons: tuple[str, ...]
    witnesses: int

    def __bool__(self) -> bool:
        return bool(self.skeletons)


@dataclass(frozen=True)
class NameMatch:
    """One credit scored against one merchant's profile."""

    credit: CanonicalBankTxn
    score: float
    matched_spelling: str


@dataclass(frozen=True)
class _Claim:
    """One settlement's above-gate claim on one credit."""

    settlement_id: str
    score: float


@dataclass(frozen=True)
class LexicalOutcome:
    """What one T3 pass produced."""

    candidates: tuple[MatchCandidate, ...] = ()
    profiles_built: int = 0
    profile_witnesses: int = 0
    settlements_seen: int = 0
    settlements_resolved: int = 0
    settlements_ambiguous: int = 0
    settlements_unsolved: int = 0
    settlements_without_profile: int = 0
    credits_matched: int = 0
    payments_matched: int = 0
    names_scored: int = 0
    rejected_below_score: int = 0
    rejected_on_margin: int = 0
    rejected_on_contention: int = 0
    rejected_inexact: int = 0
    settlements_already_referenced: int = 0

    @property
    def tier(self) -> Tier:
        return Tier.T3_FUZZY

    @property
    def settlement_links(self) -> int:
        return sum(
            1 for c in self.candidates if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
        )

    @property
    def payment_links(self) -> int:
        return sum(1 for c in self.candidates if c.is_evaluable)


def score_names(left: str, right: str) -> float:
    """Similarity of two merchant spellings, in ``[0, 1]``.

    Two scorers, and the better of the two wins:

    * **Consonant skeletons**, compared with ``fuzz.ratio``. This is the one
      that does the work. ``generator/vocab.py`` argues that embeddings cannot
      relate ``RZRPAY SFTWR P L`` to ``Razorpay Software Private Limited``;
      Step 3's :func:`~ledgerloop.ingest.normalize.merchant_skeleton` maps both
      to ``RZRPYSFTWR``, and eight of the twelve merchants collapse exactly.
    * **Normalised names**, compared with ``fuzz.token_set_ratio`` -- PLAN.md
      §6.3's scorer, and the one that survives a reordered or truncated name
      where the skeleton has been shortened by a whole missing word
      (``TECH`` for ``TECHNOLOGIES``).

    Taking the maximum rather than a weighted blend keeps the score
    interpretable: it is "how alike are these on the best available reading",
    and a blend would need weights nothing here can justify.
    """
    skeleton = fuzz.ratio(merchant_skeleton(left), merchant_skeleton(right))
    tokens = fuzz.token_set_ratio(normalize_merchant_name(left), normalize_merchant_name(right))
    return max(skeleton, tokens) / 100.0


def build_profiles(context: MatchContext) -> dict[str, MerchantProfile]:
    """Derive a merchant master from the credits that carry a reference.

    Each keyed credit is a labelled example: its UTR names a settlement, the
    settlement's payments name orders, and those orders name a merchant. So the
    spelling on that narration is attributable without any tier having matched
    anything.

    Built over **every** credit with a resolvable key, consumed or not -- the
    evidence is the reference, and a settlement being already matched does not
    make its narration less informative.
    """
    spellings: dict[str, list[str]] = {}
    for txn in context.bank_txns:
        if not txn.is_credit or txn.extracted_utr is None or txn.extracted_merchant is None:
            continue
        for view in context.settlements_by_utr.get(txn.extracted_utr, ()):
            merchant = context.merchant_of(view)
            if merchant is None:
                continue
            seen = spellings.setdefault(merchant, [])
            if txn.extracted_merchant not in seen:
                seen.append(txn.extracted_merchant)

    return {
        merchant: MerchantProfile(
            merchant_id=merchant,
            spellings=tuple(names),
            skeletons=tuple(merchant_skeleton(name) for name in names),
            witnesses=len(names),
        )
        for merchant, names in sorted(spellings.items())
    }


def rank_credits(
    view: SettlementView,
    profile: MerchantProfile,
    credits: tuple[CanonicalBankTxn, ...],
    lexical: LexicalMatching,
    *,
    gate: float | None = None,
) -> tuple[list[NameMatch], int, int]:
    """Score every candidate credit against the profile, keeping those that pass.

    ``gate`` overrides :attr:`LexicalMatching.min_score`. The tier always uses
    the configured gate; Step 7's candidate harvester passes ``0.0`` so the
    credits the gate *rejects* are collected too. Those rejections are the
    blender's negatives -- a credit the tier declined is a labelled example of
    what a wrong pairing looks like, and discarding it here would leave the
    training set with nothing but the pairings that were right.
    """
    minimum = lexical.min_score if gate is None else gate
    scored: list[NameMatch] = []
    below = 0
    for txn in credits:
        assert txn.extracted_merchant is not None  # filtered by the caller
        best = 0.0
        best_spelling = profile.spellings[0]
        for spelling in profile.spellings:
            score = score_names(txn.extracted_merchant, spelling)
            if score > best:
                best, best_spelling = score, spelling
        if best < minimum:
            below += 1
            continue
        scored.append(NameMatch(credit=txn, score=best, matched_spelling=best_spelling))

    # Descending score, then txn_id: a total order, so the run is reproducible
    # and the runner-up is always the same credit.
    scored.sort(key=lambda match: (-match.score, match.credit.txn_id))
    del view
    return scored, len(credits), below


def candidate_credits(
    view: SettlementView, context: MatchContext, tolerances: MatchingTolerances,
    lexical: LexicalMatching,
) -> tuple[CanonicalBankTxn, ...]:
    """Unreferenced credits that could be this settlement's payout on money alone."""
    band = tolerance_minor(
        view.net_minor,
        floor_minor=tolerances.amount_floor_minor,
        bps=tolerances.amount_bps,
    )
    picked: list[CanonicalBankTxn] = []
    for txn in context.open_credits():
        if txn.extracted_utr is not None or txn.extracted_merchant is None:
            continue
        if not within_tolerance(txn.credit_minor, view.net_minor, band):
            continue
        gap = (txn.value_date - view.settlement.settled_on).days
        if abs(gap) > lexical.date_window_days:
            continue
        picked.append(txn)
    return tuple(picked)


def features_for(
    view: SettlementView, match: NameMatch, tolerances: MatchingTolerances
) -> FeatureVector:
    delta = match.credit.credit_minor - view.net_minor
    return FeatureVector(
        tier=Tier.T3_FUZZY,
        amount_delta_minor=delta,
        tolerance_band_minor=tolerance_minor(
            view.net_minor,
            floor_minor=tolerances.amount_floor_minor,
            bps=tolerances.amount_bps,
        ),
        amount_delta_ratio=delta_ratio(delta, view.net_minor),
        date_delta_days=(match.credit.value_date - view.settlement.settled_on).days,
        lexical_score=match.score,
        # semantic_score stays 0.0: ChromaDB is cut, and the semantic path is
        # scheduled as an ablation row rather than a dependency.
    )


def _evidence(
    view: SettlementView,
    match: NameMatch,
    profile: MerchantProfile,
    tolerances: MatchingTolerances,
    lexical: LexicalMatching,
    runner_up: NameMatch | None,
) -> tuple[Evidence, ...]:
    band = tolerance_minor(
        view.net_minor,
        floor_minor=tolerances.amount_floor_minor,
        bps=tolerances.amount_bps,
    )
    gap = (match.credit.value_date - view.settlement.settled_on).days
    items = [
        Evidence(
            kind=EvidenceKind.LEXICAL_SIMILARITY,
            detail=(
                f"{match.credit.txn_id} names {match.credit.extracted_merchant!r}, which "
                f"scores {match.score:.3f} against {match.matched_spelling!r} -- a "
                f"spelling the bank used for {profile.merchant_id} on a credit that did "
                f"carry a reference ({profile.witnesses} spelling(s) on file)"
            ),
            refs=(settlement_ref(view.settlement_id), bank_ref(match.credit.txn_id)),
            score=match.score,
        ),
        Evidence(
            kind=EvidenceKind.NEGATIVE_EVIDENCE,
            detail=(
                f"{match.credit.txn_id} carries no reference of its own, so nothing else "
                "in the statement claims it; the narration lost its UTR (A07) and the "
                "merchant name is all that survived"
            ),
            refs=(bank_ref(match.credit.txn_id),),
        ),
        Evidence(
            kind=EvidenceKind.AMOUNT_MATCH,
            detail=(
                f"credit {format_minor(match.credit.credit_minor)} matches the net "
                f"declared by {view.settlement_id} within {format_minor(band)}"
            ),
            refs=(settlement_ref(view.settlement_id), bank_ref(match.credit.txn_id)),
            amount_minor=match.credit.credit_minor,
        ),
        Evidence(
            kind=EvidenceKind.DATE_PROXIMITY,
            detail=(
                f"credited {gap:+d} day(s) from the "
                f"{view.settlement.settled_on.isoformat()} settlement date, inside the "
                f"+/-{lexical.date_window_days} day window"
            ),
            refs=(settlement_ref(view.settlement_id), bank_ref(match.credit.txn_id)),
        ),
    ]
    if runner_up is not None:
        items.append(
            Evidence(
                kind=EvidenceKind.NEGATIVE_EVIDENCE,
                detail=(
                    f"the closest rival was {runner_up.credit.txn_id} at "
                    f"{runner_up.score:.3f}, beaten by "
                    f"{match.score - runner_up.score:.3f} against a required margin of "
                    f"{lexical.min_margin:.3f}"
                ),
                refs=(bank_ref(runner_up.credit.txn_id),),
                score=runner_up.score,
            )
        )
    return tuple(items)


def _ambiguity_evidence(
    view: SettlementView, best: NameMatch, runner_up: NameMatch, lexical: LexicalMatching
) -> Evidence:
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{best.credit.txn_id} ({best.score:.3f}) and {runner_up.credit.txn_id} "
            f"({runner_up.score:.3f}) both match {view.settlement_id} on name, amount "
            f"and date, and are {best.score - runner_up.score:.3f} apart against a "
            f"required margin of {lexical.min_margin:.3f}; the name cannot separate them"
        ),
        refs=(
            settlement_ref(view.settlement_id),
            bank_ref(best.credit.txn_id),
            bank_ref(runner_up.credit.txn_id),
        ),
        amount_minor=best.credit.credit_minor,
    )


def _closest_rival(
    claimants: tuple[_Claim, ...], settlement_id: str
) -> _Claim | None:
    """The strongest *other* settlement claiming the same credit.

    Ties break on ``settlement_id`` so the rival named in the evidence is the
    same one on every run over the same data.
    """
    others = [claim for claim in claimants if claim.settlement_id != settlement_id]
    if not others:
        return None
    return max(others, key=lambda claim: (claim.score, claim.settlement_id))


def _contention_evidence(
    view: SettlementView,
    best: NameMatch,
    rival: _Claim,
    claimants: tuple[_Claim, ...],
    lexical: LexicalMatching,
) -> Evidence:
    """Why a credit that passed all three gates is still not assignable."""
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{best.credit.txn_id} is claimed by {len(claimants)} settlements of the "
            f"same merchant -- {view.settlement_id} at {best.score:.3f} and "
            f"{rival.settlement_id} at {rival.score:.3f}, {best.score - rival.score:.3f} "
            f"apart against a required margin of {lexical.min_margin:.3f}. Their nets "
            "agree inside the amount band and the name is the same string for both, so "
            "nothing in a lexical reading says which settlement this credit paid"
        ),
        refs=(
            settlement_ref(view.settlement_id),
            settlement_ref(rival.settlement_id),
            bank_ref(best.credit.txn_id),
        ),
        amount_minor=best.credit.credit_minor,
    )


def _inexact_evidence(view: SettlementView, best: NameMatch) -> Evidence:
    """Why a credit that cleared every gate is still not this settlement's.

    The rupee figures are named because the whole refusal is the gap between
    them: without a reference, "close" is not a match, it is a different amount.
    """
    delta = best.credit.credit_minor - view.net_minor
    return Evidence(
        kind=EvidenceKind.NEGATIVE_EVIDENCE,
        detail=(
            f"{best.credit.txn_id} credits {format_minor(best.credit.credit_minor)} "
            f"against {view.settlement_id}'s net of {format_minor(view.net_minor)} -- "
            f"{format_minor(abs(delta))} apart. The name and the date agree and the "
            "reference is gone, so the amount is the only identity claim available, "
            "and an amount that is merely close is not one. A payout conserves money "
            "by construction; a credit that does not equal the net is a different "
            "payout, or part of one"
        ),
        refs=(settlement_ref(view.settlement_id), bank_ref(best.credit.txn_id)),
        amount_minor=best.credit.credit_minor,
    )


def _settlement_candidate(
    view: SettlementView,
    match: NameMatch,
    profile: MerchantProfile,
    *,
    probability: float,
    verified: bool,
    tolerances: MatchingTolerances,
    lexical: LexicalMatching,
    runner_up: NameMatch | None,
    extra: tuple[Evidence, ...] = (),
) -> MatchCandidate:
    return MatchCandidate(
        candidate_id=candidate_id(
            Tier.T3_FUZZY,
            LinkType.SETTLEMENT_CREDITED_AS,
            settlement_ref(view.settlement_id).key,
            bank_ref(match.credit.txn_id).key,
        ),
        link_type=LinkType.SETTLEMENT_CREDITED_AS,
        source_ref=settlement_ref(view.settlement_id),
        target_ref=bank_ref(match.credit.txn_id),
        tier=Tier.T3_FUZZY,
        features=features_for(view, match, tolerances),
        evidence=(
            *_evidence(view, match, profile, tolerances, lexical, runner_up),
            *extra,
        ),
        calibrated_p=probability,
        arithmetic_verified=verified,
    )


def _payment_candidates(
    view: SettlementView,
    match: NameMatch,
    profile: MerchantProfile,
    *,
    probability: float,
    verified: bool,
    tolerances: MatchingTolerances,
    lexical: LexicalMatching,
    runner_up: NameMatch | None,
) -> list[MatchCandidate]:
    """Expand a lexically identified credit into the evaluation unit.

    Identical allocation to every other tier: the credit is split across the
    payments it carries by gross weight, with the charged-back payment (A08)
    left out because its money never reached the bank. Sharing the rule keeps
    a T3 link worth exactly what a T0 link is worth in rupees.
    """
    clawback = attribute_clawback(view)
    excluded = clawback.excluded.payment_id if clawback.excluded is not None else None
    covered = tuple(p for p in view.payments if p.payment_id != excluded)
    if not covered:
        return []

    credit = match.credit
    shares = allocate_minor(credit.credit_minor, [p.amount_minor for p in covered])
    conserved = (
        sum_minor(shares, field=f"{credit.txn_id}.allocation") == credit.credit_minor
    )
    features = features_for(view, match, tolerances)
    lexical_note = _evidence(view, match, profile, tolerances, lexical, runner_up)[0]

    candidates: list[MatchCandidate] = []
    for payment, share in zip(covered, shares, strict=True):
        candidates.append(
            MatchCandidate(
                candidate_id=candidate_id(
                    Tier.T3_FUZZY,
                    LinkType.PAYMENT_CREDITED_AS,
                    payment_ref(payment.payment_id).key,
                    bank_ref(credit.txn_id).key,
                ),
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(payment.payment_id),
                target_ref=bank_ref(credit.txn_id),
                tier=Tier.T3_FUZZY,
                features=features,
                evidence=(
                    lexical_note,
                    Evidence(
                        kind=EvidenceKind.ARITHMETIC_CHECK,
                        detail=(
                            f"allocated {format_minor(share)} of the "
                            f"{format_minor(credit.credit_minor)} credit, by gross "
                            f"weight; the {len(covered)} share(s) sum to the credit "
                            "exactly"
                        ),
                        refs=(payment_ref(payment.payment_id), bank_ref(credit.txn_id)),
                        amount_minor=share,
                    ),
                ),
                calibrated_p=probability,
                arithmetic_verified=verified and conserved,
            )
        )
    return candidates


def run_tier3(
    context: MatchContext,
    tolerances: MatchingTolerances,
    lexical: LexicalMatching,
    *,
    profiles: dict[str, MerchantProfile] | None = None,
) -> LexicalOutcome:
    """Run T3 over whatever the keyed tiers left in the pool.

    ``profiles`` is accepted so the re-run loop can build the merchant master
    once rather than per pass -- it is derived from references, which do not
    change as tiers consume records.
    """
    master = build_profiles(context) if profiles is None else profiles
    candidates: list[MatchCandidate] = []
    seen = resolved = ambiguous = unsolved = without_profile = 0
    credits_matched = payments_matched = scored_count = below = margin_rejects = 0
    contention_rejects = already_referenced = inexact_rejects = 0

    # Pass 1 -- score, commit to nothing. The claim map has to be complete before
    # any credit is handed out, because contention is a property of the whole
    # statement and a loop that consumed as it went would only ever see the
    # claims it had not yet satisfied. Nothing is consumed here, so the map does
    # not depend on the order the settlements are visited in.
    ranked: list[tuple[SettlementView, MerchantProfile, list[NameMatch]]] = []
    claims: dict[str, list[_Claim]] = {}
    # Who may *claim*, and who may be *assigned*, are different sets.
    #
    # A claim is evidence that a credit is spoken for; an assignment is an act.
    # A settlement that is open may do both. One the ladder **refused** may only
    # claim -- it still has an outstanding claim on the credit it was refused
    # over, and that is exactly the evidence contention needs. One that was
    # **resolved** may do neither: its money is accounted for, so a further
    # claim would be a refusal manufactured out of nothing.
    open_ids = {view.settlement_id for view in context.open_settlements()}
    claimant_ids = open_ids | context.refused_settlements
    for view in context.settlements:
        if view.settlement_id not in claimant_ids:
            continue
        merchant = context.merchant_of(view)
        profile = master.get(merchant) if merchant is not None else None
        if profile is None or not profile:
            without_profile += 1
            continue
        if not view.payments or view.net_minor <= 0:
            continue

        pool = candidate_credits(view, context, tolerances, lexical)
        if not pool:
            continue

        scored, examined, rejected = rank_credits(view, profile, pool, lexical)

        # A settlement the bank has already named is not T3's work -- see
        # ALREADY-REFERENCED, above -- so it is never *assigned* a credit. It
        # still **claims** one, and that distinction is the whole point: its
        # claim is evidence about the credit, not a request for it.
        #
        # Dropping the claim as well is what let a false positive through at
        # 5,000 orders on seed 45. SETL-0231's own UTR sat on BNK-01528, so this
        # gate excluded it entirely; it therefore never claimed BNK-00231, whose
        # amount equalled its net **to the paise**; contention below saw a single
        # claimant and handed that credit to a different settlement 0.03% away.
        # The gate meant to protect precision had removed the only evidence the
        # contention test needed.
        assignable = view.settlement_id in open_ids and not (
            view.utr is not None and context.credits_by_utr.get(view.utr)
        )
        if not assignable:
            if view.settlement_id in open_ids:
                already_referenced += 1
            for match in scored:
                claims.setdefault(match.credit.txn_id, []).append(
                    _Claim(settlement_id=view.settlement_id, score=match.score)
                )
            continue

        seen += 1
        scored_count += examined
        below += rejected

        if not scored:
            unsolved += 1
            continue

        ranked.append((view, profile, scored))
        for match in scored:
            claims.setdefault(match.credit.txn_id, []).append(
                _Claim(settlement_id=view.settlement_id, score=match.score)
            )

    # Pass 2 -- assign, against a claim map that no longer moves.
    for view, profile, scored in ranked:
        best = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        if runner_up is not None and best.score - runner_up.score < lexical.min_margin:
            # Two credits the scorer cannot separate. Same refusal as every
            # tier before this one, and the same uniform prior over them.
            margin_rejects += 1
            candidates.append(
                _settlement_candidate(
                    view,
                    best,
                    profile,
                    probability=best.score / 2.0,
                    verified=False,
                    tolerances=tolerances,
                    lexical=lexical,
                    runner_up=None,
                    extra=(_ambiguity_evidence(view, best, runner_up, lexical),),
                )
            )
            context.consume(view.settlement_id)
            ambiguous += 1
            continue

        # The mirror-image test: does a second settlement have as good a claim on
        # this credit? Same margin, same refusal, same uniform prior -- the only
        # difference is which side of the pairing the rival sits on.
        claimants = tuple(claims.get(best.credit.txn_id, ()))
        rival = _closest_rival(claimants, view.settlement_id)
        if rival is not None and best.score - rival.score < lexical.min_margin:
            contention_rejects += 1
            candidates.append(
                _settlement_candidate(
                    view,
                    best,
                    profile,
                    probability=best.score / len(claimants),
                    verified=False,
                    tolerances=tolerances,
                    lexical=lexical,
                    runner_up=None,
                    extra=(_contention_evidence(view, best, rival, claimants, lexical),),
                )
            )
            context.consume(view.settlement_id)
            ambiguous += 1
            continue

        # The money has to be exact -- see WITHOUT A REFERENCE, above. The band
        # got this credit into the pool, where it could be contested; it does
        # not get it assigned.
        if best.credit.credit_minor != view.net_minor:
            inexact_rejects += 1
            candidates.append(
                _settlement_candidate(
                    view,
                    best,
                    profile,
                    probability=best.score / 2.0,
                    verified=False,
                    tolerances=tolerances,
                    lexical=lexical,
                    runner_up=None,
                    extra=(_inexact_evidence(view, best),),
                )
            )
            context.consume(view.settlement_id)
            ambiguous += 1
            continue

        verified = view.gross_reconciles
        candidates.append(
            _settlement_candidate(
                view,
                best,
                profile,
                probability=best.score,
                verified=verified,
                tolerances=tolerances,
                lexical=lexical,
                runner_up=runner_up,
            )
        )
        expanded = _payment_candidates(
            view,
            best,
            profile,
            probability=best.score,
            verified=verified,
            tolerances=tolerances,
            lexical=lexical,
            runner_up=runner_up,
        )
        candidates.extend(expanded)
        context.consume(view.settlement_id, [best.credit.txn_id])
        resolved += 1
        credits_matched += 1
        payments_matched += len(expanded)

    return LexicalOutcome(
        candidates=tuple(candidates),
        profiles_built=len(master),
        profile_witnesses=sum(p.witnesses for p in master.values()),
        settlements_seen=seen,
        settlements_resolved=resolved,
        settlements_ambiguous=ambiguous,
        settlements_unsolved=unsolved,
        settlements_without_profile=without_profile,
        credits_matched=credits_matched,
        payments_matched=payments_matched,
        names_scored=scored_count,
        rejected_below_score=below,
        rejected_on_margin=margin_rejects,
        rejected_on_contention=contention_rejects,
        rejected_inexact=inexact_rejects,
        settlements_already_referenced=already_referenced,
    )
