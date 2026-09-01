"""Phase 2.11 -- T3's exactness rule, measured against a second construction model.

WHAT THIS FILE PINS
-------------------
Phase 2.10 removed T3's amount tolerance band because, over 49 corpora, every
legitimate reference-free whole-net match was exact to the paise and the band was
admitting exactly one thing: the last false positive. All 49 came from one
generator, and in that generator "unreferenced" and "off the net" are mutually
exclusive **by a truth-representation constraint** -- see
:mod:`ledgerloop.generator.adversarial`.

So the rule had one source of evidence and was load-bearing for precision. This
file runs the unchanged matcher over a corpus built under the opposite
assumption -- where the bank deducts a charge the PSP file never sees, and the
deduction is independent of whether the narration kept its UTR -- and pins what
happens.

Two things happen, and both matter:

* **The Phase 2.10 false positive, rebuilt independently, is refused.** Every
  ``UNREF_TRANCHE_BAIT`` settlement is offered its sibling's split tranche a few
  basis points away, and every one is declined. So is every near-miss orphan.
* **Legitimate reference-free drift is refused too, and it is real here.** The
  four deduction shapes carry true links the system does not find. That is the
  measured cost of the rule, and it is pinned rather than described.

The counterfactual -- re-admitting the pre-2.10 band on this same corpus -- was
measured in ``.local/steps/phase-2-11.md`` rather than here: it costs precision
0.8469 -> 0.7729 and multiplies false positives 226 -> 639 across five seeds. It
is not shipped, so there is no production behaviour for a test to hold.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ledgerloop.eval.harness import run_system
from ledgerloop.eval.metrics import confusion
from ledgerloop.generator.adversarial import Case, write_adversarial_corpus
from ledgerloop.models.enums import Tier

#: The phrase :func:`~ledgerloop.matching.tier3_lexical._inexact_evidence` writes
#: and nothing else does. Matching on it rather than on a counter is what lets a
#: refusal be attributed to a *settlement*, which is the unit the case table is in.
INEXACT_MARKER = "the amount is the only identity claim available"


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """One model-B corpus, run through the production path, with its case table.

    No calibration bundle: the fitted one lives under ``reports/``, which is not
    committed, and the run is identical without it -- T3's own scores clear the
    default threshold, and the exactness gate gets there first either way.
    """
    directory = tmp_path_factory.mktemp("adversarial") / "corpus"
    corpus = write_adversarial_corpus(directory, 42)
    run = run_system(directory, measure_calibration_quality=False)
    return corpus, run


def _case_of_payment(corpus):
    payment_to_settlement = {p.payment_id: p.settlement_id for p in corpus.world.payments}
    settlement_to_case = {r.settlement_id: r.case for r in corpus.cases}
    return lambda payment_id: settlement_to_case.get(payment_to_settlement.get(payment_id))


def _confusion(corpus, run):
    return confusion([p.pair for p in run.matched.predictions], run.truth.evaluation_pairs)


def _fp_cases(corpus, run) -> Counter:
    case_of = _case_of_payment(corpus)
    tally: Counter = Counter()
    for pair in _confusion(corpus, run).false_positives:
        tally[case_of(pair[0].split(":")[-1])] += 1
    return tally


def _tp_cases(corpus, run) -> Counter:
    case_of = _case_of_payment(corpus)
    tally: Counter = Counter()
    for pair in _confusion(corpus, run).true_positives:
        tally[case_of(pair[0].split(":")[-1])] += 1
    return tally


def _inexact_refusals(corpus, run) -> dict[Case, set[str]]:
    """Which settlements T3 declined *on the amount*, grouped by case shape."""
    settlement_to_case = {r.settlement_id: r.case for r in corpus.cases}
    refused: dict[Case, set[str]] = {}
    for candidate in run.matched.candidates:
        if candidate.tier is not Tier.T3_FUZZY:
            continue
        if not any(INEXACT_MARKER in item.detail for item in candidate.evidence):
            continue
        settlement_id = candidate.source_ref.key.split(":")[-1]
        case = settlement_to_case.get(settlement_id)
        if case is not None:
            refused.setdefault(case, set()).add(settlement_id)
    return refused


class TestThePhase210FalsePositiveIsRefusedByASecondGenerator:
    """The regression that matters, rebuilt from the other direction.

    Phase 2.10's seven false positives were settlements taking a *tranche* of a
    different split payout a few basis points away. Model B constructs that shape
    deliberately: the bait's own payout lands outside T3's window, so the only
    same-merchant credit in its pool is the host's first tranche, sitting 6 bps
    from its net -- the measured distance of the original defect.
    """

    def test_every_bait_is_offered_the_tranche_and_declines_it(self, scored):
        corpus, run = scored
        baits = {r.settlement_id for r in corpus.by_case()[Case.UNREF_TRANCHE_BAIT]}
        refused = _inexact_refusals(corpus, run).get(Case.UNREF_TRANCHE_BAIT, set())
        assert refused == baits

    def test_the_credit_it_declined_belongs_to_the_sibling(self, scored):
        """A refusal on the wrong credit would be the right answer for the wrong
        reason, and would not be a regression test for anything."""
        corpus, run = scored
        owner_of = {t.txn_id: t.settlement_id for t in corpus.world.bank_txns}
        hosts = {r.settlement_id for r in corpus.by_case()[Case.UNREF_TRANCHE_HOST]}
        baits = {r.settlement_id for r in corpus.by_case()[Case.UNREF_TRANCHE_BAIT]}
        seen = 0
        for candidate in run.matched.candidates:
            if candidate.tier is not Tier.T3_FUZZY:
                continue
            if candidate.source_ref.key.split(":")[-1] not in baits:
                continue
            if not any(INEXACT_MARKER in item.detail for item in candidate.evidence):
                continue
            assert owner_of[candidate.target_ref.key.split(":")[-1]] in hosts
            seen += 1
        assert seen == len(baits)

    def test_no_bait_payment_is_linked_to_anything(self, scored):
        corpus, run = scored
        assert _fp_cases(corpus, run)[Case.UNREF_TRANCHE_BAIT] == 0
        assert _tp_cases(corpus, run)[Case.UNREF_TRANCHE_BAIT] == 0


class TestWhatTheRuleStillCannotCatch:
    """Exactness is necessary, not sufficient, and the corpus says so.

    ``UNREF_ORPHAN_EXACT`` puts a credit carrying the merchant's name and the
    settlement's net **to the paise** where that settlement's payout should have
    been. Given the three sources there is nothing to distinguish it from a true
    payout, so it is taken -- and no amount rule could refuse it without refusing
    every true T3 match. Re-admitting the band makes this case no better and
    every other case worse.
    """

    def test_every_false_positive_comes_from_the_indistinguishable_orphan(self, scored):
        corpus, run = scored
        assert set(_fp_cases(corpus, run)) == {Case.UNREF_ORPHAN_EXACT}

    def test_the_orphan_that_is_merely_close_is_refused(self, scored):
        """The same shape one basis point off the paise, and the rule holds."""
        corpus, run = scored
        assert _fp_cases(corpus, run)[Case.UNREF_ORPHAN_NEAR] == 0
        assert Case.UNREF_ORPHAN_NEAR in _inexact_refusals(corpus, run)

    def test_two_settlements_it_cannot_tell_apart_are_both_refused(self, scored):
        corpus, run = scored
        assert _fp_cases(corpus, run)[Case.UNREF_TWIN_EXACT] == 0
        assert _tp_cases(corpus, run)[Case.UNREF_TWIN_EXACT] == 0


class TestTheCostOfTheRuleIsRealAndMeasured:
    """Legitimate reference-free matches with non-exact amounts **do exist**.

    Model A cannot produce one; model B produces four shapes of them. The rule
    refuses all four, and these tests hold that cost visible rather than letting
    it be described away. Do not delete them to make a recall number look better:
    the counterfactual band that recovers these links costs 2.8x the false
    positives on the same corpus.
    """

    @pytest.mark.parametrize(
        "case",
        [
            Case.UNREF_CHARGE,
            Case.UNREF_ROUNDED,
            Case.UNREF_HAIRCUT,
            Case.UNREF_CHARGEBACK_CHARGE,
        ],
    )
    def test_a_legitimate_deduction_is_never_matched(self, scored, case):
        corpus, run = scored
        assert _tp_cases(corpus, run)[case] == 0
        assert _fp_cases(corpus, run)[case] == 0

    def test_the_refusal_is_on_the_amount_and_is_recorded_as_such(self, scored):
        corpus, run = scored
        refused = _inexact_refusals(corpus, run)
        assert Case.UNREF_CHARGE in refused
        assert Case.UNREF_HAIRCUT in refused
        assert run.matched.lexical.rejected_inexact >= len(
            corpus.by_case()[Case.UNREF_TRANCHE_BAIT]
        )

    def test_the_same_drift_with_a_reference_in_front_of_it_is_matched(self, scored):
        """The asymmetry is the rule, stated as a measurement.

        T0 and T1 match on a reference and *then* check the money, so their band
        absorbs the identical bank charge. T3 has no reference, so the amount is
        the identity claim. Same drift, same corpus, opposite outcome.
        """
        corpus, run = scored
        assert _tp_cases(corpus, run)[Case.REFERENCED_DEDUCTED] > 0
        assert _tp_cases(corpus, run)[Case.UNREF_CHARGE] == 0


class TestWhatStillWorks:
    def test_an_exact_reference_free_payout_is_still_matched(self, scored):
        corpus, run = scored
        assert _tp_cases(corpus, run)[Case.UNREF_EXACT] > 0
        assert _fp_cases(corpus, run)[Case.UNREF_EXACT] == 0

    def test_a_chargeback_does_not_break_exactness(self, scored):
        """The credit is short by the chargeback, and equals the *adjusted* net.
        A rule comparing against the gross would refuse this."""
        corpus, run = scored
        assert _tp_cases(corpus, run)[Case.UNREF_CHARGEBACK_EXACT] > 0
        assert _fp_cases(corpus, run)[Case.UNREF_CHARGEBACK_EXACT] == 0

    def test_the_exception_queue_still_covers_everything_it_should(self, scored):
        _, run = scored
        assert run.metrics.exception_recall == 1.0


class TestThePhase211RegressionArm:
    """The fourth exact arm, alongside Steps 4-9 130/0/164, Phase 2.3 248/0/46
    and Phase 2.5 283/0/11.

    Model B seed 42, full ladder, no calibration bundle. If a change moves this
    triple it has moved behaviour on reference-free matching, and that belongs in
    the step notes with its reason -- not in a quiet edit to this number.
    """

    def test_the_arm_is_exact(self, scored):
        _, run = scored
        links = run.metrics.link_metrics
        assert (links.true_positives, links.false_positives, links.false_negatives) == (
            249,
            43,
            641,
        )

    def test_the_false_positive_cost_is_entirely_the_indistinguishable_orphan(self, scored):
        corpus, run = scored
        orphans = {
            txn.txn_id for txn in corpus.world.bank_txns if txn.settlement_id is None
        }
        for pair in _confusion(corpus, run).false_positives:
            assert pair[1].split(":")[-1] in orphans
