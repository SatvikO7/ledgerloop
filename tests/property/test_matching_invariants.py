"""Property tests for the tier ladder.

Three families, in ascending order of what they would catch.

**Tolerance algebra** -- the admission test is monotone in the band and
symmetric in the sign of the drift. A rule that accepted ``+3`` and rejected
``-3``, or that accepted a wide band but not a wider one, would be wrong in a
way no single example is likely to find.

**Money conservation** -- whatever the matcher asserts about a credit, the
allocated shares sum to that credit exactly. This is the no-paise-created
invariant crossing from the money module into the matcher.

**Ladder discipline** -- across generated corpora at several splits, seeds and
difficulties: no link is ever asserted twice, no record is claimed twice, T1
never touches what T0 ruled on, and a rerun decides identically.

The corpora are memoised, as in the ingest property tests: generation is the
slow half.
"""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ledgerloop.config import GeneratorConfig, MatchingTolerances, RunConfig
from ledgerloop.eval.metrics import confusion
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import IngestResult, ingest_dataset
from ledgerloop.matching import MatchRun, run_matching
from ledgerloop.matching.bank_leg import BankLegRule, allocated_share_minor, resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.matching.tier0_exact import T0_RULE
from ledgerloop.matching.tier1_tolerance import rule_for
from ledgerloop.models.enums import Difficulty, LinkType, SplitName, Tier
from ledgerloop.money import tolerance_minor
from tests.unit.conftest import batch, corpus

# Bounded to realistic payout sizes: ₹10 to ₹10 lakh, in whole paise.
nets = st.integers(min_value=1_000, max_value=100_000_000)
drifts = st.integers(min_value=-50_000, max_value=50_000)
day_gaps = st.integers(min_value=-30, max_value=30)


def _run_rule(net: int, delta: int, days: int, rule: BankLegRule) -> int:
    only = batch(amounts=(net,))
    built = corpus(
        batches=[only], bank_txns=[only.credit(delta_minor=delta, days_after=days)]
    )
    return resolve_bank_leg(MatchContext.from_ingest(built), rule).resolved_settlements


class TestToleranceAlgebra:
    @given(nets, drifts)
    @settings(max_examples=60, deadline=None)
    def test_the_band_decides_and_nothing_else_does(self, net: int, delta: int):
        rule = rule_for(MatchingTolerances())
        band = tolerance_minor(net, floor_minor=100, bps=50)
        expected = 1 if abs(delta) <= band else 0
        assert _run_rule(net, delta, 1, rule) == expected

    @given(nets, st.integers(min_value=1, max_value=50_000))
    @settings(max_examples=40, deadline=None)
    def test_the_rule_is_symmetric_in_the_sign_of_the_drift(self, net: int, size: int):
        rule = rule_for(MatchingTolerances())
        assert _run_rule(net, size, 1, rule) == _run_rule(net, -size, 1, rule)

    @given(nets, drifts, st.integers(min_value=0, max_value=200))
    @settings(max_examples=40, deadline=None)
    def test_a_wider_band_never_matches_less(self, net: int, delta: int, extra_bps: int):
        narrow = rule_for(MatchingTolerances(amount_bps=50))
        wide = rule_for(MatchingTolerances(amount_bps=50 + extra_bps))
        assert _run_rule(net, delta, 1, wide) >= _run_rule(net, delta, 1, narrow)

    @given(nets, day_gaps, st.integers(min_value=0, max_value=40))
    @settings(max_examples=40, deadline=None)
    def test_a_wider_window_never_matches_less(self, net: int, days: int, extra: int):
        narrow = rule_for(MatchingTolerances(date_window_days=3))
        wide = rule_for(MatchingTolerances(date_window_days=3 + extra))
        assert _run_rule(net, 1, days, wide) >= _run_rule(net, 1, days, narrow)

    @given(nets, day_gaps)
    @settings(max_examples=40, deadline=None)
    def test_t0_ignores_the_calendar_entirely(self, net: int, days: int):
        assert _run_rule(net, 0, days, T0_RULE) == 1

    @given(nets, st.integers(min_value=1, max_value=50_000))
    @settings(max_examples=40, deadline=None)
    def test_t0_accepts_nothing_but_equality(self, net: int, size: int):
        assert _run_rule(net, size, 1, T0_RULE) == 0
        assert _run_rule(net, -size, 1, T0_RULE) == 0

    @given(nets)
    @settings(max_examples=40, deadline=None)
    def test_anything_t0_matches_t1_would_have_matched_too(self, net: int):
        """T0 is strictly the tighter rule on amount. The ladder relies on it."""
        wide = rule_for(MatchingTolerances())
        assert _run_rule(net, 0, 1, T0_RULE) <= _run_rule(net, 0, 1, wide)


class TestMoneyConservation:
    @given(
        st.lists(
            st.integers(min_value=100, max_value=5_000_000), min_size=1, max_size=12
        ),
        st.integers(min_value=0, max_value=200_000),
    )
    @settings(max_examples=60, deadline=None)
    def test_allocated_shares_sum_to_the_credit_exactly(
        self, amounts: list[int], fee: int
    ):
        gross = sum(amounts)
        if fee >= gross:
            fee = gross - 1
        only = batch(amounts=tuple(amounts), fee_minor=fee)
        built = corpus(batches=[only], bank_txns=[only.credit()])
        outcome = resolve_bank_leg(MatchContext.from_ingest(built), T0_RULE)
        shares = [allocated_share_minor(c) for c in outcome.candidates if c.is_evaluable]
        assert sum(shares) == only.net_minor
        assert all(isinstance(share, int) and not isinstance(share, bool) for share in shares)

    @given(st.lists(st.integers(min_value=100, max_value=900_000), min_size=1, max_size=10))
    @settings(max_examples=40, deadline=None)
    def test_no_share_is_negative(self, amounts: list[int]):
        only = batch(amounts=tuple(amounts))
        built = corpus(batches=[only], bank_txns=[only.credit()])
        outcome = resolve_bank_leg(MatchContext.from_ingest(built), T0_RULE)
        assert all(
            allocated_share_minor(c) >= 0 for c in outcome.candidates if c.is_evaluable
        )


@cache
def _matched(
    split: SplitName, difficulty: Difficulty, seed: int
) -> tuple[Path, IngestResult, MatchRun]:
    """Generate, ingest and match one corpus. Memoised -- generation is the slow half."""
    directory = Path(tempfile.mkdtemp(prefix="ll-match-"))
    generate_to_disk(
        GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
    )
    ingested = ingest_dataset(directory, strict=True)
    run = run_matching(ingested, RunConfig(run_id=f"{split.value}-{seed}"))
    return directory, ingested, run


@pytest.mark.parametrize("split", [SplitName.DEV, SplitName.CALIBRATION])
@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", [7, 42])
class TestLadderDisciplineAcrossTheCorpus:
    """Across split x difficulty x seed, not just on the committed fixture."""

    def test_no_link_is_ever_asserted_twice(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        pairs = [prediction.pair for prediction in run.predictions]
        assert len(pairs) == len(set(pairs))

    def test_no_bank_credit_is_claimed_by_two_settlements(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        claimed: dict[str, str] = {}
        for candidate in run.candidates:
            if candidate.link_type is not LinkType.SETTLEMENT_CREDITED_AS:
                continue
            if candidate.calibrated_p != 1.0:
                continue  # contested edges assert nothing and claim nothing
            target = candidate.target_ref.key
            assert claimed.setdefault(target, candidate.source_ref.key) == (
                candidate.source_ref.key
            )

    def test_t1_never_touches_a_settlement_t0_ruled_on(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        by_tier: dict[Tier, set[str]] = {Tier.T0_EXACT: set(), Tier.T1_TOLERANCE: set()}
        for candidate in run.candidates:
            if candidate.link_type is LinkType.SETTLEMENT_CREDITED_AS:
                by_tier[candidate.tier].add(candidate.source_ref.key)
        assert by_tier[Tier.T0_EXACT] & by_tier[Tier.T1_TOLERANCE] == set()

    def test_no_debit_row_is_ever_matched(self, split, difficulty, seed):
        _, ingested, run = _matched(split, difficulty, seed)
        debits = {txn.txn_id for txn in ingested.bank_txns if not txn.is_credit}
        touched = {p.target_ref.record_id for p in run.predictions}
        assert touched & debits == set()

    def test_every_prediction_is_correct(self, split, difficulty, seed):
        """The step's headline claim, across the configuration space."""
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
        assert matrix.false_positives == frozenset()

    def test_predicted_amounts_equal_the_truth_amounts(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        amounts = {
            link.pair: link.amount_minor
            for link in truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        for prediction in run.predictions:
            assert prediction.amount_minor == amounts[prediction.pair]

    def test_no_unmatchable_record_is_asserted_about(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        touched = {ref for prediction in run.predictions for ref in prediction.pair}
        assert touched & truth.unmatchable_refs == set()

    def test_an_auto_match_always_verified_its_arithmetic(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        for decision in run.decisions:
            if decision.is_positive_prediction:
                assert decision.arithmetic_verified

    def test_rerunning_decides_identically(self, split, difficulty, seed):
        _, ingested, run = _matched(split, difficulty, seed)
        again = run_matching(
            ingested,
            RunConfig(run_id=f"{split.value}-{seed}"),
            decided_at=run.decisions[0].decided_at,
        )
        assert again.predictions == run.predictions
        assert [d.decision_id for d in again.decisions] == [
            d.decision_id for d in run.decisions
        ]


@settings(deadline=None, max_examples=6)
@given(seed=st.integers(min_value=0, max_value=400))
def test_precision_is_perfect_at_an_arbitrary_seed(seed: int):
    directory, _, run = _matched(SplitName.DEV, Difficulty.STANDARD, seed)
    truth = load_ground_truth(directory)
    matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
    assert matrix.false_positives == frozenset()
