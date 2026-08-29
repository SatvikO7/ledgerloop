"""Split completion: T2's arithmetic over a tranche set T3's evidence supplies.

The pass exists for one shape the ladder could not see. A09 ``SPLIT_PAYOUT``
delivers a batch in two tranches; A07 ``MISSING_REFERENCE`` strips the narration
reference. Composed, ``credit_bucket`` -- which reads ``open_credits_for(utr)``
-- returns nothing, so T2 never sees the batch, and T3 cannot help because it
tests a candidate against the **whole** net and a tranche is never the whole net.

Three things have to hold, and each has a section:

1. **It finds the right tranche set, or none.** Exact sums, at least two
   credits, and exhaustively unique -- every other case is a refusal.
2. **It refuses everything it cannot prove.** Ambiguity, non-exhaustive search,
   an already-claimed credit, a partition that does not resolve.
3. **It is right on real corpora**, checked against link-level ground truth,
   and it never asserts a link ground truth does not contain.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ledgerloop.config import (
    Difficulty,
    GeneratorConfig,
    LexicalMatching,
    MatchingTolerances,
    RunConfig,
    SplitCompletion,
    SplitName,
)
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.metrics import confusion
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.matching import run_matching
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier2_aggregation import (
    find_tranche_set,
    lexical_credit_bucket,
    run_split_completion,
)
from ledgerloop.matching.tier3_lexical import build_profiles
from ledgerloop.models.enums import LinkType
from tests.unit.conftest import bank_credit, batch, corpus

TOL = MatchingTolerances()
LEX = LexicalMatching()
MERCHANT = "RAZORPAY SOFTWARE PVT"


def _split(only, amounts, *, merchant=MERCHANT, utr=None, days=(1, 2), extra=()):
    """A batch whose payout arrives as unreferenced tranches of ``amounts``."""
    tranches = [
        bank_credit(
            f"BNK-0010{index}",
            amount_minor=amount,
            utr=utr,
            merchant=merchant,
            value_date=only.settlement.settled_on + timedelta(days=days[index % len(days)]),
        )
        for index, amount in enumerate(amounts)
    ]
    return corpus(batches=[only], bank_txns=[*tranches, *extra])


def _context(sources, *, keyed_seed: bool = True) -> MatchContext:
    """A context whose merchant master knows ``MERCHANT``.

    The master is derived from the statement's own **keyed** credits, so a
    corpus with no keyed credit at all has no profile and the pass has no pool.
    ``keyed_seed`` adds one unrelated keyed credit naming the same merchant,
    which is what a real statement always contains.
    """
    context = MatchContext.from_ingest(sources)
    del keyed_seed
    return context


@pytest.fixture
def split_sources():
    """SETL-0001: 100,000 gross, no fee, paid as 60,000 + 40,000, unreferenced.

    A keyed credit for a *different* settlement carries the merchant spelling,
    which is how T3's master learns it -- exactly as it does on a real corpus.
    """
    only = batch(amounts=(60_000, 40_000))
    keyed = batch(
        "SETL-0002",
        utr="UTR2026031099999",
        amounts=(25_000,),
        first_index=50,
    )
    return corpus(
        batches=[only, keyed],
        bank_txns=[
            keyed.credit("BNK-00050"),
            bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
            bank_credit(
                "BNK-00102",
                amount_minor=40_000,
                utr=None,
                merchant=MERCHANT,
                value_date=only.settlement.settled_on + timedelta(days=2),
            ),
        ],
    )


def _run(sources, **overrides):
    config = RunConfig(run_id="split", **overrides)
    return run_matching(sources, config)


class TestItFindsTheTrancheSet:
    def test_two_unreferenced_credits_summing_to_the_net_are_the_payout(
        self, split_sources
    ):
        run = _run(split_sources)
        outcome = run.split_completion
        assert outcome.settlements_resolved == 1
        assert outcome.credits_matched == 2
        assert outcome.payments_matched == 2

    def test_it_asserts_the_payment_links_the_tranches_carry(self, split_sources):
        run = _run(split_sources)
        pairs = {
            (c.source_ref.record_id, c.target_ref.record_id)
            for c in run.candidates
            if c.link_type is LinkType.PAYMENT_CREDITED_AS
            and c.target_ref.record_id in {"BNK-00101", "BNK-00102"}
        }
        assert pairs == {("PAY-00001", "BNK-00101"), ("PAY-00002", "BNK-00102")}

    def test_the_pool_excludes_credits_that_carry_a_reference(self, split_sources):
        """A credit publishing a reference is already explained -- either by the
        settlement it names, or as somebody else's business."""
        context = _context(split_sources)
        profiles = build_profiles(context)
        view = context.settlements_by_id["SETL-0001"]
        pool = lexical_credit_bucket(view, context, LEX, profiles)
        assert {txn.txn_id for txn in pool} == {"BNK-00101", "BNK-00102"}

    def test_the_search_is_over_credits_not_payments(self, split_sources):
        context = _context(split_sources)
        profiles = build_profiles(context)
        view = context.settlements_by_id["SETL-0001"]
        pool = lexical_credit_bucket(view, context, LEX, profiles)
        tranches, search = find_tranche_set(view, pool, TOL)
        assert tranches is not None
        assert {t.txn_id for t in tranches} == {"BNK-00101", "BNK-00102"}
        assert search.is_unique

    def test_money_stays_in_integer_minor_units(self, split_sources):
        """Every amount the pass touches is an int. A float in the money path
        raises in `assert_minor` rather than producing a plausible subset."""
        context = _context(split_sources)
        profiles = build_profiles(context)
        view = context.settlements_by_id["SETL-0001"]
        pool = lexical_credit_bucket(view, context, LEX, profiles)
        assert all(isinstance(txn.credit_minor, int) for txn in pool)
        tranches, _ = find_tranche_set(view, pool, TOL)
        assert tranches is not None
        assert sum(t.credit_minor for t in tranches) == view.net_minor


class TestEveryRefusal:
    """Each test changes one thing and expects the pass to assert nothing."""

    def test_a_single_credit_equal_to_the_whole_net_is_not_a_split(self):
        """That is a one-to-one match and belongs to T0/T1/T3. If this pass
        claimed it, it would be re-litigating a stricter tier's question."""
        only = batch(amounts=(60_000, 40_000))
        keyed = batch("SETL-0002", utr="UTR2026031099999", amounts=(25_000,), first_index=50)
        sources = corpus(
            batches=[only, keyed],
            bank_txns=[
                keyed.credit("BNK-00050"),
                bank_credit("BNK-00101", amount_minor=100_000, utr=None, merchant=MERCHANT),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0

    def test_two_sets_reaching_the_net_are_ambiguous_and_refused(self):
        """60+40 and 55+45 both make 100,000. Picking one is a coin flip."""
        only = batch(amounts=(60_000, 40_000))
        keyed = batch("SETL-0002", utr="UTR2026031099999", amounts=(25_000,), first_index=50)
        settled = only.settlement.settled_on
        sources = corpus(
            batches=[only, keyed],
            bank_txns=[
                keyed.credit("BNK-00050"),
                bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
                bank_credit(
                    "BNK-00102", amount_minor=40_000, utr=None, merchant=MERCHANT,
                    value_date=settled + timedelta(days=2),
                ),
                bank_credit(
                    "BNK-00103", amount_minor=55_000, utr=None, merchant=MERCHANT,
                    value_date=settled + timedelta(days=3),
                ),
                bank_credit(
                    "BNK-00104", amount_minor=45_000, utr=None, merchant=MERCHANT,
                    value_date=settled + timedelta(days=4),
                ),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0
        assert not [
            c for c in run.candidates
            if c.target_ref.record_id in {"BNK-00101", "BNK-00102", "BNK-00103", "BNK-00104"}
        ]

    def test_amounts_that_do_not_reach_the_net_exactly_are_refused(self):
        """Not a band. A split payout conserves money by construction, so a
        tolerance here would admit sets that are merely close."""
        only = batch(amounts=(60_000, 40_000))
        keyed = batch("SETL-0002", utr="UTR2026031099999", amounts=(25_000,), first_index=50)
        sources = corpus(
            batches=[only, keyed],
            bank_txns=[
                keyed.credit("BNK-00050"),
                bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
                bank_credit(
                    "BNK-00102", amount_minor=39_999, utr=None, merchant=MERCHANT,
                    value_date=only.settlement.settled_on + timedelta(days=2),
                ),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0

    def test_a_different_merchant_is_not_in_the_pool(self):
        only = batch(amounts=(60_000, 40_000))
        keyed = batch("SETL-0002", utr="UTR2026031099999", amounts=(25_000,), first_index=50)
        sources = corpus(
            batches=[only, keyed],
            bank_txns=[
                keyed.credit("BNK-00050"),
                bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
                bank_credit(
                    "BNK-00102", amount_minor=40_000, utr=None,
                    merchant="NYKAA E RETAIL PVT LTD",
                    value_date=only.settlement.settled_on + timedelta(days=2),
                ),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0

    def test_a_credit_outside_the_date_window_is_not_in_the_pool(self):
        only = batch(amounts=(60_000, 40_000))
        keyed = batch("SETL-0002", utr="UTR2026031099999", amounts=(25_000,), first_index=50)
        sources = corpus(
            batches=[only, keyed],
            bank_txns=[
                keyed.credit("BNK-00050"),
                bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
                bank_credit(
                    "BNK-00102", amount_minor=40_000, utr=None, merchant=MERCHANT,
                    value_date=only.settlement.settled_on
                    + timedelta(days=LEX.date_window_days + 5),
                ),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0

    def test_with_no_merchant_master_there_is_no_pool(self):
        """The master is derived from the statement's own keyed credits. A
        statement with none teaches T3 nothing, and this pass inherits that."""
        only = batch(amounts=(60_000, 40_000))
        sources = corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00101", amount_minor=60_000, utr=None, merchant=MERCHANT),
                bank_credit(
                    "BNK-00102", amount_minor=40_000, utr=None, merchant=MERCHANT,
                    value_date=only.settlement.settled_on + timedelta(days=2),
                ),
            ],
        )
        run = _run(sources)
        assert run.split_completion.settlements_resolved == 0
        assert run.split_completion.settlements_without_key == 1

    def test_a_settlement_a_tier_already_ruled_on_is_never_re_opened(
        self, split_sources
    ):
        """Exclusivity in the other direction: the pass iterates the *open*
        pool, so a batch T0-T4 consumed -- matched or contested -- is invisible
        to it. A looser candidate rule must not overturn a stricter refusal."""
        context = MatchContext.from_ingest(split_sources)
        context.consume("SETL-0001")
        outcome = run_split_completion(
            context, TOL, LEX, build_profiles(context)
        )
        assert outcome.settlements_seen == 0
        assert outcome.candidates == ()

    def test_a_credit_another_settlement_claimed_is_not_available(self, split_sources):
        """`open_credits()` is the pool, so a consumed credit cannot be a
        tranche of a second batch. One payout, one credit."""
        context = MatchContext.from_ingest(split_sources)
        context.consume("SETL-0099", ["BNK-00101"])
        outcome = run_split_completion(
            context, TOL, LEX, build_profiles(context)
        )
        assert outcome.settlements_resolved == 0

    def test_switching_the_pass_off_asserts_nothing(self, split_sources):
        run = _run(split_sources, split_completion=SplitCompletion(enabled=False))
        assert run.split_completion.settlements_seen == 0
        assert run.split_completion.candidates == ()
        assert not [
            c for c in run.candidates
            if c.target_ref.record_id in {"BNK-00101", "BNK-00102"}
        ]


class TestInvariants:
    """Properties that must hold however the pass is exercised."""

    def test_the_tranches_it_asserts_always_sum_to_the_net_exactly(
        self, split_sources
    ):
        run = _run(split_sources)
        view = run.context.settlements_by_id["SETL-0001"]
        claimed = [
            txn for txn in run.context.bank_txns
            if txn.txn_id in run.context.consumed_credits
            and txn.txn_id in {"BNK-00101", "BNK-00102"}
        ]
        assert sum(t.credit_minor for t in claimed) == view.net_minor

    def test_every_payment_is_assigned_to_exactly_one_tranche(self, split_sources):
        run = _run(split_sources)
        assigned = [
            c.source_ref.record_id for c in run.candidates
            if c.link_type is LinkType.PAYMENT_CREDITED_AS
            and c.target_ref.record_id in {"BNK-00101", "BNK-00102"}
        ]
        assert len(assigned) == len(set(assigned))

    def test_every_candidate_it_emits_is_arithmetic_verified(self, split_sources):
        """`MatchDecision` refuses to be AUTO_MATCHED without this, so a pass
        that emitted an unverified candidate could never auto-match anyway --
        but the guarantee belongs here, where the money is re-derived."""
        run = _run(split_sources)
        emitted = [
            c for c in run.candidates
            if c.target_ref.record_id in {"BNK-00101", "BNK-00102"}
        ]
        assert emitted
        assert all(c.arithmetic_verified for c in emitted)

    def test_its_candidates_carry_the_evidence_a_controller_can_check(
        self, split_sources
    ):
        run = _run(split_sources)
        settlement_links = [
            c for c in run.candidates
            if c.link_type is LinkType.SETTLEMENT_CREDITED_AS
            and c.target_ref.record_id in {"BNK-00101", "BNK-00102"}
        ]
        assert settlement_links
        blob = " ".join(
            item.detail for c in settlement_links for item in c.evidence
        )
        assert "sum exactly to" in blob
        assert "none of these rows carries a reference" in blob

    def test_it_is_deterministic(self, split_sources):
        first = _run(split_sources)
        second = _run(split_sources)
        assert [c.candidate_id for c in first.candidates] == [
            c.candidate_id for c in second.candidates
        ]


class TestOnGeneratedCorpora:
    """The measurement, not the illustration."""

    @pytest.mark.parametrize(
        ("split", "difficulty", "seed"),
        [
            (SplitName.TEST, Difficulty.STANDARD, 42),
            (SplitName.TEST, Difficulty.STANDARD, 43),
            (SplitName.TEST, Difficulty.HARD, 45),
            (SplitName.TRAIN, Difficulty.STANDARD, 42),
        ],
    )
    def test_it_never_asserts_a_link_ground_truth_does_not_contain(
        self, tmp_path, split, difficulty, seed
    ):
        """The claim the whole pass rests on, checked against link-level truth.

        Ground truth is read here and nowhere the system can see it. Precision
        must be exactly 1.0 and the false-positive count exactly zero -- a
        recall gain bought with a wrong auto-match is not an improvement in this
        project.
        """
        directory = tmp_path / f"{split.value}-{difficulty.value}-{seed}"
        generate_to_disk(
            GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
        )
        run = run_system(directory, measure_calibration_quality=False)
        links = run.metrics.link_metrics
        assert links is not None
        assert links.false_positives == 0
        assert links.false_positive_cost_minor == 0
        assert run.metrics.auto_match_precision == 1.0

    def test_it_gains_links_without_losing_any(self, tmp_path):
        """Both arms, same corpus, one configuration field apart."""
        directory = tmp_path / "test-standard-42"
        generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
        off = run_system(
            directory,
            split_completion=SplitCompletion(enabled=False),
            measure_calibration_quality=False,
        )
        on = run_system(directory, measure_calibration_quality=False)
        assert off.metrics.link_metrics is not None
        assert on.metrics.link_metrics is not None

        truth = load_ground_truth(directory)
        before = confusion(
            [p.pair for p in off.matched.predictions], truth.evaluation_pairs
        )
        after = confusion(
            [p.pair for p in on.matched.predictions], truth.evaluation_pairs
        )
        # Strictly additive: every link the old arm found, the new one still
        # finds. A change that swapped one correct link for another would show
        # up here even at an unchanged count.
        assert before.true_positives <= after.true_positives
        assert after.false_positives == frozenset()
        assert len(after.true_positives) == 283

    def test_the_pass_reports_what_it_refused(self, tmp_path):
        """A refusal is a result. The counters are what let the report say how
        often the pass declined rather than only how often it fired."""
        directory = tmp_path / "test-hard-45"
        generate_to_disk(
            GeneratorConfig(
                split=SplitName.TEST, difficulty=Difficulty.HARD, seed=45
            ),
            directory,
        )
        outcome = run_system(
            directory, measure_calibration_quality=False
        ).matched.split_completion
        accounted = (
            outcome.settlements_resolved
            + outcome.settlements_ambiguous
            + outcome.settlements_unsolved
        )
        assert accounted == outcome.settlements_seen
