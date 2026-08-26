"""Property tests for T3 and T4.

Three families.

**The scorer** -- bounded, symmetric, reflexive, and stable under the
normalisation Step 3 built. A scorer that drifted on case or punctuation would
make T3's gate mean something different for every narration.

**The graph** -- reachability and adjacency behave, whatever order the edges
arrive in. Non-deterministic traversal would make seeded runs irreproducible.

**The ladder with five tiers** -- across generated corpora at several splits,
seeds and difficulties: still no false positive, no payment credited twice, no
tier revisiting what an earlier one ruled on, and the re-run loop terminating.
"""

from __future__ import annotations

import tempfile
from functools import cache
from itertools import combinations
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ledgerloop.config import GeneratorConfig, LexicalMatching, RunConfig
from ledgerloop.eval.metrics import confusion
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.generator.vocab import MERCHANTS
from ledgerloop.graph.memory_repo import MemoryGraphRepo
from ledgerloop.ingest import IngestResult, ingest_dataset
from ledgerloop.matching import MatchRun, run_matching
from ledgerloop.matching.bank_leg import allocated_share_minor
from ledgerloop.matching.tier3_lexical import score_names
from ledgerloop.models.enums import Difficulty, LinkType, SplitName, Tier
from ledgerloop.models.refs import bank_ref, order_ref, payment_ref, settlement_ref

LEXICAL = LexicalMatching()

names = st.one_of(
    st.sampled_from(
        [variant for merchant in MERCHANTS for variant in merchant.variants]
        + [merchant.legal_name for merchant in MERCHANTS]
    ),
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ ", max_size=30),
)


class TestTheScorer:
    @given(names, names)
    @settings(max_examples=200, deadline=None)
    def test_it_is_bounded(self, left: str, right: str):
        assert 0.0 <= score_names(left, right) <= 1.0

    @given(names, names)
    @settings(max_examples=200, deadline=None)
    def test_it_is_symmetric(self, left: str, right: str):
        assert score_names(left, right) == score_names(right, left)

    @given(names)
    @settings(max_examples=100, deadline=None)
    def test_a_name_matches_itself_perfectly(self, name: str):
        from ledgerloop.ingest.normalize import normalize_merchant_name

        if not normalize_merchant_name(name):
            return
        assert score_names(name, name) == 1.0

    @given(names)
    @settings(max_examples=150, deadline=None)
    def test_case_and_punctuation_make_no_difference(self, name: str):
        noisy = name.lower().replace(" ", " - ")
        assert score_names(name, name) == pytest.approx(score_names(name, noisy), abs=0.0)

    @given(names)
    @settings(max_examples=100, deadline=None)
    def test_a_trailing_legal_form_makes_no_difference(self, name: str):
        from ledgerloop.ingest.normalize import normalize_merchant_name

        if not normalize_merchant_name(name):
            return
        assert score_names(name, name) == score_names(name, f"{name} PRIVATE LIMITED")


class TestTheScoreSeparation:
    """The measured claim T3's precision rests on."""

    def test_the_gate_clears_every_cross_merchant_pairing(self):
        worst = max(
            score_names(left, right)
            for first, second in combinations(MERCHANTS, 2)
            for left in (*first.variants, first.legal_name)
            for right in (*second.variants, second.legal_name)
        )
        assert worst < LEXICAL.min_score

    def test_every_merchant_is_reachable_by_at_least_one_pair_of_spellings(self):
        """The comparison T3 actually makes is variant against variant.

        A legal name never appears in a narration -- the profile is built from
        the spellings the *bank* used -- so the question is not whether a
        variant resembles the registered name but whether two variants resemble
        each other. Every merchant has at least one pair that clears the gate,
        so none is unreachable in principle.

        Which pair the corpus happens to present is chance, and for the four
        merchants with a below-gate pair that is where T3's reach is partial.
        Named in ``test_matching_tier3.py`` rather than tuned away.
        """
        for merchant in MERCHANTS:
            best = max(
                score_names(left, right)
                for left, right in combinations(merchant.variants, 2)
            )
            assert best >= LEXICAL.min_score, merchant.merchant_id


refs = st.sampled_from(
    [order_ref("O1"), payment_ref("P1"), settlement_ref("S1"), bank_ref("B1")]
)


class TestTheGraph:
    @given(st.lists(st.tuples(refs, refs), max_size=12))
    @settings(max_examples=150, deadline=None)
    def test_every_inserted_edge_is_found(self, pairs):
        repo = MemoryGraphRepo()
        for source, target in pairs:
            repo.add_edge(source, target, LinkType.PAYMENT_CREDITED_AS)
        for source, target in pairs:
            assert repo.has_edge(source, target, LinkType.PAYMENT_CREDITED_AS)

    @given(st.lists(st.tuples(refs, refs), max_size=12))
    @settings(max_examples=150, deadline=None)
    def test_an_edge_implies_a_path(self, pairs):
        repo = MemoryGraphRepo()
        for source, target in pairs:
            repo.add_edge(source, target, LinkType.PAYMENT_CREDITED_AS)
        for source, target in pairs:
            assert repo.path_exists(source, target)

    @given(st.lists(st.tuples(refs, refs), max_size=12))
    @settings(max_examples=100, deadline=None)
    def test_reachability_never_hangs_on_a_cycle(self, pairs):
        repo = MemoryGraphRepo()
        for source, target in pairs:
            repo.add_edge(source, target, LinkType.PAYMENT_CREDITED_AS)
        assert isinstance(repo.path_exists(order_ref("O1"), bank_ref("ZZZ")), bool)

    @given(st.lists(st.tuples(refs, refs), max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_inserting_the_same_edges_twice_changes_nothing(self, pairs):
        once, twice = MemoryGraphRepo(), MemoryGraphRepo()
        for source, target in pairs:
            once.add_edge(source, target, LinkType.ORDER_PAID_BY)
            twice.add_edge(source, target, LinkType.ORDER_PAID_BY)
            twice.add_edge(source, target, LinkType.ORDER_PAID_BY)
        assert len(once) == len(twice)
        assert once.edges_of_type(LinkType.ORDER_PAID_BY) == twice.edges_of_type(
            LinkType.ORDER_PAID_BY
        )


@cache
def _matched(
    split: SplitName, difficulty: Difficulty, seed: int
) -> tuple[Path, IngestResult, MatchRun]:
    directory = Path(tempfile.mkdtemp(prefix="ll-t34-"))
    generate_to_disk(
        GeneratorConfig(split=split, difficulty=difficulty, seed=seed), directory
    )
    ingested = ingest_dataset(directory, strict=True)
    run = run_matching(ingested, RunConfig(run_id=f"{split.value}-{seed}"))
    return directory, ingested, run


@pytest.mark.parametrize("split", [SplitName.DEV, SplitName.CALIBRATION])
@pytest.mark.parametrize("difficulty", list(Difficulty))
@pytest.mark.parametrize("seed", [7, 42])
class TestTheFiveTierLadder:
    def test_the_whole_ladder_asserts_nothing_wrong(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
        assert matrix.false_positives == frozenset()

    def test_no_t3_prediction_is_wrong(self, split, difficulty, seed):
        """The lexical tier's own precision, isolated from the tiers before it."""
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        lexical_pairs = {
            (c.source_ref.key, c.target_ref.key)
            for c in run.candidates
            if c.tier is Tier.T3_FUZZY and c.is_evaluable
        }
        predicted = {p.pair for p in run.predictions} & lexical_pairs
        assert predicted <= truth.evaluation_pairs

    def test_no_payment_is_credited_twice(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        seen: dict[str, str] = {}
        for prediction in run.predictions:
            source = prediction.source_ref.key
            assert seen.setdefault(source, prediction.target_ref.key) == (
                prediction.target_ref.key
            )

    def test_no_settlement_is_ruled_on_by_two_tiers(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        owner: dict[str, Tier] = {}
        for candidate in run.candidates:
            if candidate.link_type is not LinkType.SETTLEMENT_CREDITED_AS:
                continue
            key = candidate.source_ref.key
            assert owner.setdefault(key, candidate.tier) is candidate.tier

    def test_no_credit_is_over_absorbed(self, split, difficulty, seed):
        """Exclusivity, checked over the whole run rather than inside one tier."""
        _, ingested, run = _matched(split, difficulty, seed)
        amounts = {t.txn_id: t.credit_minor for t in ingested.bank_txns if t.is_credit}
        absorbed: dict[str, int] = {}
        for prediction in run.predictions:
            key = prediction.target_ref.record_id
            absorbed[key] = absorbed.get(key, 0) + prediction.amount_minor
        for txn_id, total in absorbed.items():
            assert total <= amounts[txn_id]

    def test_predicted_amounts_equal_the_truth_amounts(self, split, difficulty, seed):
        directory, _, run = _matched(split, difficulty, seed)
        truth = load_ground_truth(directory)
        expected = {
            link.pair: link.amount_minor
            for link in truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        for prediction in run.predictions:
            assert prediction.amount_minor == expected[prediction.pair]

    def test_the_rerun_loop_terminates_well_inside_its_cap(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        assert 1 <= run.passes <= RunConfig(run_id="x").graph.max_rerun_passes

    def test_every_t4_candidate_carries_graph_evidence(self, split, difficulty, seed):
        _, _, run = _matched(split, difficulty, seed)
        for candidate in run.candidates:
            if candidate.tier is Tier.T4_GRAPH:
                assert any(
                    item.kind.value == "GRAPH_RULE" for item in candidate.evidence
                )
                assert allocated_share_minor(candidate) >= 0

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
def test_the_five_tier_ladder_is_precise_at_an_arbitrary_seed(seed: int):
    directory, _, run = _matched(SplitName.DEV, Difficulty.STANDARD, seed)
    truth = load_ground_truth(directory)
    matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
    assert matrix.false_positives == frozenset()
