"""Top-k candidate emission, and the population rule that governs the fit.

The harvester's job is to produce labelled rows for the blender, and the only
way it can be wrong that a passing fit would still hide is by producing rows
from a *different population* than the one the blender scores at run time. So
the central test here is not "does it collect contenders" but:

    the rows it offers the fit are exactly the evaluation links the pipeline
    would send through the blender -- no more, no fewer

Everything else follows from that: the clawback exclusion, the refused decision
points, the de-duplication across passes.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import RunConfig
from ledgerloop.matching import run_matching
from ledgerloop.matching.harvest import (
    DEFAULT_TOP_K,
    _Collector,
    _harvest_graph,
    harvest,
)
from ledgerloop.models.enums import (
    AnomalyClass,
    Difficulty,
    ExpectedStatus,
    LinkType,
    SplitName,
    Tier,
)
from ledgerloop.models.refs import bank_ref, payment_ref
from ledgerloop.models.truth import GroundTruth, GroundTruthLink, GroundTruthRecord
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus

CONFIG = RunConfig(run_id="harvest-test")


def truth_for(*pairs: tuple[str, str], amount_minor: int = 1) -> GroundTruth:
    """Ground truth carrying exactly the given ``(payment, credit)`` links."""
    return GroundTruth(
        split=SplitName.TRAIN,
        difficulty=Difficulty.STANDARD,
        seed=1,
        generator_version="0.2.0",
        links=tuple(
            GroundTruthLink(
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref(payment),
                target_ref=bank_ref(credit),
                amount_minor=amount_minor,
            )
            for payment, credit in pairs
        ),
        records=(),
    )


def truth_from_run(ingest, config: RunConfig = CONFIG) -> GroundTruth:
    """Truth that agrees with whatever the ladder asserts. Isolates the plumbing."""
    run = run_matching(ingest, config)
    return truth_for(
        *(
            (candidate.source_ref.record_id, candidate.target_ref.record_id)
            for candidate in run.candidates
            if candidate.is_evaluable and candidate.arithmetic_verified
        )
    )


def split_credits(only, first: tuple[int, ...], second: tuple[int, ...]):
    """Two tranches of one batch, allocated exactly as the truth links are."""
    grosses = [payment.amount_minor for payment in only.payments]
    left = sum(grosses[index] for index in first)
    right = sum(grosses[index] for index in second)
    amounts = allocate_minor(only.net_minor, [left, right])
    return (
        bank_credit("BNK-00001", amount_minor=amounts[0], utr=only.settlement.utr),
        bank_credit("BNK-00002", amount_minor=amounts[1], utr=only.settlement.utr),
    )


@pytest.fixture
def split_corpus():
    """One batch paid in two tranches -- the shape T2 exists for."""
    only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
    return only, corpus(batches=[only], bank_txns=list(split_credits(only, (0,), (1, 2))))


class TestThePopulationRule:
    """The fit population must equal the population the blender scores."""

    def test_the_fit_rows_are_exactly_what_the_pipeline_would_score(self, split_corpus):
        _, ingest = split_corpus
        truth = truth_from_run(ingest)
        harvested = harvest(ingest, truth, CONFIG)
        run = run_matching(ingest, CONFIG)

        scored = {
            candidate.pair
            for candidate in run.candidates
            if candidate.is_evaluable
            and candidate.arithmetic_verified
            and not candidate.tier.is_deterministic_certain
        }
        assert {row.pair for row in harvested.fit_rows} == scored

    def test_the_deterministic_tiers_never_enter_the_training_set(self, simple):
        """T0 and T1 bypass the blender, so fitting on them fits the wrong shape."""
        truth = truth_from_run(simple)
        harvested = harvest(simple, truth, CONFIG)
        assert all(not row.tier.is_deterministic_certain for row in harvested.rows)

    def test_a_refused_decision_point_is_collected_but_never_fitted_on(self):
        """Two credits the name score cannot separate: T3 refuses, so no fit rows."""
        only = batch(utr=None, amounts=(60_000, 40_000))
        rival = batch(
            "SETL-0002",
            utr="UTR2026031099999",
            amounts=(60_000, 40_000),
            first_index=3,
        )
        twins = [
            bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None),
            bank_credit("BNK-00002", amount_minor=only.net_minor, utr=None),
            bank_credit("BNK-00003", amount_minor=rival.net_minor, utr=rival.settlement.utr),
        ]
        ingest = corpus(batches=[only, rival], bank_txns=twins)
        harvested = harvest(ingest, truth_for(), CONFIG)
        assert harvested.rows  # the contenders were seen
        assert not harvested.fit_rows  # and none of them was fitted on

    def test_fit_rows_are_a_subset_of_every_row(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        every = {(row.pair, row.tier) for row in harvested.rows}
        assert {(row.pair, row.tier) for row in harvested.fit_rows} <= every
        assert len(harvested.fit_rows) + len(harvested.refused_rows) == len(
            harvested.rows
        )


class TestLabelling:
    def test_a_link_ground_truth_asserts_is_positive(self, split_corpus):
        _, ingest = split_corpus
        truth = truth_from_run(ingest)
        harvested = harvest(ingest, truth, CONFIG)
        assert harvested.fit_rows
        assert all(row.is_positive for row in harvested.fit_rows)

    def test_a_link_ground_truth_denies_is_negative(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        assert harvested.positives == 0
        assert harvested.negatives == len(harvested.rows)

    def test_the_label_travels_on_the_candidate_itself(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_from_run(ingest), CONFIG)
        for row in harvested.fit_rows:
            assert row.candidate.is_truth_positive is row.is_positive

    def test_features_and_labels_line_up(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_from_run(ingest), CONFIG)
        assert len(harvested.features) == len(harvested.labels)
        assert len(harvested.diagnostic_features) == len(harvested.diagnostic_labels)


class TestTheClawbackExclusion:
    """A08 nets a payment off the batch; its money never reached the bank."""

    def test_a_charged_back_payment_is_not_harvested_as_a_link(self):
        only = batch(
            utr=None,
            amounts=(60_000, 40_000),
            adjustments_minor=-40_000,
        )
        credit = bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None)
        ingest = corpus(batches=[only], bank_txns=[credit])
        harvested = harvest(ingest, truth_for(), CONFIG)
        clawed_back = only.payments[1].payment_id
        assert all(
            row.candidate.source_ref.record_id != clawed_back for row in harvested.rows
        )


class TestDeterminism:
    def test_two_harvests_of_one_corpus_agree_exactly(self, split_corpus):
        _, ingest = split_corpus
        truth = truth_from_run(ingest)
        first = harvest(ingest, truth, CONFIG)
        second = harvest(ingest, truth, CONFIG)
        assert [row.pair for row in first.rows] == [row.pair for row in second.rows]
        assert first.labels == second.labels

    def test_rows_come_out_in_tier_then_rank_order(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        keys = [(row.tier, row.rank, row.pair) for row in harvested.rows]
        assert keys == sorted(keys)

    def test_a_pairing_seen_in_two_passes_is_recorded_once(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        pairs = [(row.pair, row.tier) for row in harvested.rows]
        assert len(pairs) == len(set(pairs))


class TestTheHarvestChangesNothing:
    def test_the_ladder_decides_identically_either_side_of_a_harvest(self, split_corpus):
        """A harvest must be observationally invisible to the evaluation."""
        _, ingest = split_corpus
        before = run_matching(ingest, CONFIG)
        harvest(ingest, truth_for(), CONFIG)
        after = run_matching(ingest, CONFIG)
        assert [d.candidate_id for d in before.decisions] == [
            d.candidate_id for d in after.decisions
        ]
        assert [d.outcome for d in before.decisions] == [
            d.outcome for d in after.decisions
        ]


class TestTheCounts:
    def test_it_reports_the_decision_points_it_examined(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        assert harvested.decision_points > 0
        assert harvested.resolved_points <= harvested.decision_points
        assert harvested.passes >= 1

    def test_counts_by_tier_come_out_in_ladder_order(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_from_run(ingest), CONFIG)
        counts = harvested.by_tier()
        assert list(counts) == sorted(counts, key=lambda name: Tier[name])
        assert sum(counts.values()) == len(harvested.fit_rows)

    def test_the_diagnostic_counts_cover_every_contender(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_for(), CONFIG)
        assert sum(harvested.by_tier(fit_only=False).values()) == len(harvested.rows)

    def test_positives_by_tier_never_exceeds_rows_by_tier(self, split_corpus):
        _, ingest = split_corpus
        harvested = harvest(ingest, truth_from_run(ingest), CONFIG)
        rows = harvested.by_tier()
        for tier, count in harvested.positives_by_tier().items():
            assert count <= rows[tier]

    def test_the_default_top_k_is_carried_on_the_result(self, split_corpus):
        _, ingest = split_corpus
        assert harvest(ingest, truth_for(), CONFIG).top_k == DEFAULT_TOP_K


class TestRefusals:
    def test_a_top_k_below_one_is_refused(self, split_corpus):
        _, ingest = split_corpus
        with pytest.raises(ValueError, match="at least 1"):
            harvest(ingest, truth_for(), CONFIG, top_k=0)

    def test_an_empty_corpus_yields_no_rows(self):
        assert harvest(corpus(), truth_for(), CONFIG).rows == ()


class TestGroundTruthIsForLabellingOnly:
    def test_the_rows_collected_do_not_depend_on_the_labels(self, split_corpus):
        """A link the ladder cannot reach is absent because no tier proposed it."""
        _, ingest = split_corpus
        everything = harvest(ingest, truth_from_run(ingest), CONFIG)
        nothing = harvest(ingest, truth_for(), CONFIG)
        assert [row.pair for row in everything.rows] == [
            row.pair for row in nothing.rows
        ]

    def test_an_unreachable_truth_link_is_not_invented(self, simple):
        truth = truth_for(("PAY-99999", "BNK-99999"))
        harvested = harvest(simple, truth, CONFIG)
        assert all(row.pair != ("payment:PAY-99999", "bank_txn:BNK-99999")
                   for row in harvested.rows)


def test_records_are_not_required_for_labelling():
    """The harvester reads ``evaluation_pairs`` and nothing else from the truth."""
    truth = GroundTruth(
        split=SplitName.TRAIN,
        difficulty=Difficulty.STANDARD,
        seed=1,
        generator_version="0.2.0",
        links=(),
        records=(
            GroundTruthRecord(
                record_ref=payment_ref("PAY-00001"),
                expected_status=ExpectedStatus.MATCHED,
                anomaly_class=AnomalyClass.CLEAN,
            ),
        ),
    )
    only = batch()
    ingest = corpus(batches=[only], bank_txns=[only.credit()])
    assert harvest(ingest, truth, CONFIG).positives == 0


class TestShapesThatOfferNothingToHarvest:
    """Each of these is a decision point that does not exist. None may raise."""

    def test_a_settlement_with_one_credit_is_not_an_aggregation_question(self, simple):
        """One credit is a whole-batch match, which T0 and T1 own."""
        harvested = harvest(simple, truth_for(), CONFIG)
        assert all(row.tier is not Tier.T2_AGGREGATION for row in harvested.rows)

    def test_a_batch_with_no_payments_is_skipped(self):
        only = batch(amounts=())
        ingest = corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=1, utr=only.settlement.utr),
                bank_credit("BNK-00002", amount_minor=1, utr=only.settlement.utr),
            ],
        )
        assert harvest(ingest, truth_for(), CONFIG).rows == ()

    def test_a_merchant_with_no_learned_spellings_yields_no_lexical_contenders(self):
        """T3 needs a profile, and a profile needs a keyed credit to learn from."""
        only = batch(utr=None)
        ingest = corpus(
            batches=[only],
            bank_txns=[bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None)],
        )
        assert harvest(ingest, truth_for(), CONFIG).rows == ()

    def test_a_batch_whose_every_payment_was_clawed_back_yields_nothing(self):
        only = batch(utr=None, amounts=(40_000,), adjustments_minor=-40_000)
        keyed = batch("SETL-0002", amounts=(40_000,), first_index=5)
        ingest = corpus(
            batches=[only, keyed],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None),
                keyed.credit("BNK-00002"),
            ],
        )
        rows = harvest(ingest, truth_for(), CONFIG).rows
        assert all(
            row.candidate.source_ref.record_id != only.payments[0].payment_id
            for row in rows
        )


class TestTheGraphTierIsHarvestedAtRankZero:
    """T4 fires zero times on this corpus, so its path is exercised directly.

    Same treatment Step 6 gave the inference rules themselves: a rule that
    cannot be made to fire on the corpus is tested against a constructed state
    rather than reported as covered because nothing ran through it.
    """

    def _candidate(self, verified: bool):
        from ledgerloop.models.candidates import FeatureVector, MatchCandidate

        return MatchCandidate(
            candidate_id="t4-1",
            link_type=LinkType.PAYMENT_CREDITED_AS,
            source_ref=payment_ref("PAY-00001"),
            target_ref=bank_ref("BNK-00001"),
            tier=Tier.T4_GRAPH,
            features=FeatureVector(tier=Tier.T4_GRAPH, graph_support=0.9),
            calibrated_p=0.9,
            arithmetic_verified=verified,
        )

    def test_a_verified_inference_becomes_a_fit_row(self):
        collector = _Collector()
        truth = truth_for(("PAY-00001", "BNK-00001"))
        _harvest_graph((self._candidate(verified=True),), truth, collector)
        rows = collector.ordered()
        assert len(rows) == 1
        assert rows[0].accepted and rows[0].asserted and rows[0].is_positive
        assert collector.resolved_points == 1

    def test_an_unverified_inference_is_collected_but_not_fitted_on(self):
        collector = _Collector()
        _harvest_graph((self._candidate(verified=False),), truth_for(), collector)
        rows = collector.ordered()
        assert len(rows) == 1
        assert not rows[0].accepted
        assert collector.resolved_points == 0

    def test_a_structural_edge_is_not_an_evaluation_link(self):
        from ledgerloop.models.candidates import FeatureVector, MatchCandidate
        from ledgerloop.models.refs import settlement_ref

        collector = _Collector()
        structural = MatchCandidate(
            candidate_id="t4-2",
            link_type=LinkType.SETTLEMENT_CREDITED_AS,
            source_ref=settlement_ref("SETL-0001"),
            target_ref=bank_ref("BNK-00001"),
            tier=Tier.T4_GRAPH,
            features=FeatureVector(tier=Tier.T4_GRAPH),
            calibrated_p=1.0,
            arithmetic_verified=True,
        )
        _harvest_graph((structural,), truth_for(), collector)
        assert collector.ordered() == ()
