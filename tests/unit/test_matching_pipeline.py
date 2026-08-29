"""The whole ladder, end to end, and against the real corpus.

Two halves. The synthetic half checks the wiring on datasets small enough to
verify by eye: what reaches ``ReconState``, what becomes a prediction, what a
rerun produces. The measured half runs the committed fixture and a generated
split and pins the properties that make the step's claim -- **zero false
positives** -- checkable rather than asserted.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ledgerloop.config import (
    GeneratorConfig,
    GraphInference,
    MatchingTolerances,
    RunConfig,
)
from ledgerloop.eval.baselines import run_b0
from ledgerloop.eval.metrics import confusion, evaluate
from ledgerloop.eval.truth_io import load_ground_truth
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import ingest_dataset
from ledgerloop.matching import MATCHER_NAME, run_matching
from ledgerloop.models.enums import DecisionOutcome, LinkType, SourceName, SplitName, Tier
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus, noise_credit

WHEN = datetime(2026, 4, 1, 9, 0, 0)
FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "dev-standard-42"


def config(**kwargs: object) -> RunConfig:
    return RunConfig(run_id="test-run", **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def fixture_scored():
    """The committed fixture, matched and scored once for the whole module."""
    truth = load_ground_truth(FIXTURE)
    ingested = ingest_dataset(FIXTURE, strict=True)
    run = run_matching(ingested, config(), decided_at=WHEN)
    metrics = evaluate(
        run.predictions,
        truth,
        run_id="fixture",
        tier_contributions=run.tier_contributions,
    )
    return run, truth, metrics


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("t0t1-split")
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def split_scored(split_dir):
    truth = load_ground_truth(split_dir)
    run = run_matching(ingest_dataset(split_dir, strict=True), config(), decided_at=WHEN)
    return run, truth, evaluate(run.predictions, truth, run_id="split")


@pytest.fixture
def clean_run(simple):
    return run_matching(simple, config(), decided_at=WHEN)


class TestStateIntegration:
    def test_the_run_carries_a_populated_recon_state(self, clean_run):
        state = clean_run.state
        assert state.run_id == "test-run"
        assert state.config.config_hash
        assert state.candidates
        assert state.decisions

    def test_the_normalised_records_reach_the_state(self, clean_run, simple):
        assert len(clean_run.state.normalized) == len(simple.normalized)

    def test_provenance_reaches_the_state_grouped_by_source(self, clean_run):
        raw = clean_run.state.raw
        assert set(raw) == set(SourceName)
        assert raw[SourceName.LEDGER]
        assert raw[SourceName.BANK]

    def test_open_decisions_are_all_of_them_before_any_revision(self, clean_run):
        assert len(clean_run.state.open_decisions) == len(clean_run.state.decisions)

    def test_every_decision_points_at_a_candidate_in_the_state(self, clean_run):
        ids = {candidate.candidate_id for candidate in clean_run.state.candidates}
        assert all(decision.candidate_id in ids for decision in clean_run.state.decisions)

    def test_every_decision_carries_the_run_timestamp(self, clean_run):
        assert {d.decided_at for d in clean_run.state.decisions} == {WHEN}


class TestWhatBecomesAPrediction:
    def test_only_the_evaluation_link_type_is_predicted(self, clean_run):
        assert all(
            p.source_ref.record_type.value == "payment" for p in clean_run.predictions
        )
        assert all(p.target_ref.record_type.value == "bank_txn" for p in clean_run.predictions)

    def test_structural_edges_are_decided_but_never_predicted(self, clean_run):
        decided = {d.link_type for d in clean_run.state.decisions}
        assert LinkType.ORDER_PAID_BY in decided
        assert LinkType.SETTLEMENT_CREDITED_AS in decided
        assert len(clean_run.predictions) == 2

    def test_a_referral_never_becomes_a_prediction(self):
        """A contested settlement asserts nothing to the evaluator.

        Both credits land on the **same day**, which is what keeps the contest a
        contest: the Phase 2.3 duplicate-posting pass needs a unique earliest
        row to call one of them a re-posting, and two same-day postings give it
        no ordering. So this exercises the tier's refusal rather than the pass's.
        """
        only = batch()
        run = run_matching(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
            ),
            config(),
            decided_at=WHEN,
        )
        assert run.predictions == ()
        assert run.exceptions == 2
        assert run.auto_matched == 2  # the order leg still resolved

    def test_predictions_assert_the_allocated_share_not_the_gross(self):
        """B0 asserts each payment's gross and overstates the money it reconciled."""
        only = batch(amounts=(60_000, 40_000), fee_minor=10_000)
        run = run_matching(
            corpus(batches=[only], bank_txns=[only.credit()]), config(), decided_at=WHEN
        )
        assert sum(p.amount_minor for p in run.predictions) == only.net_minor
        assert sum(p.amount_minor for p in run.predictions) < sum(
            p.amount_minor for p in only.payments
        )

    def test_an_unverified_link_is_demoted_and_not_predicted(self):
        only = batch(amounts=(60_000, 40_000), gross_minor=1)
        run = run_matching(
            corpus(batches=[only], bank_txns=[only.credit()]), config(), decided_at=WHEN
        )
        assert run.predictions == ()
        assert run.needs_review >= 2


class TestCandidateYieldVersusConviction:
    def test_the_two_are_reported_separately(self, clean_run):
        rows = {row.tier: row for row in clean_run.tier_contributions}
        assert set(rows) == {
            Tier.T0_EXACT,
            Tier.T1_TOLERANCE,
            Tier.T2_AGGREGATION,
            Tier.T3_FUZZY,
            Tier.T4_GRAPH,
        }
        assert rows[Tier.T0_EXACT].candidates_proposed == 5  # 2 orders + 1 settlement + 2 payments
        assert rows[Tier.T0_EXACT].auto_matched == 5

    def test_a_tier_with_nothing_to_do_reports_zero_rather_than_vanishing(self, clean_run):
        """T0 took the whole batch, so T2 saw an empty pool. Zero is the measurement.

        Distinct from an *unbuilt* tier, which is absent from the table entirely.
        """
        rows = {row.tier: row for row in clean_run.tier_contributions}
        assert rows[Tier.T2_AGGREGATION].candidates_proposed == 0
        assert rows[Tier.T2_AGGREGATION].auto_matched == 0

    def test_yield_exceeds_auto_matches_when_the_policy_declines(self):
        # Same-day rivals: an unorderable pair the duplicate-posting pass leaves
        # to the tier, so the contested path is the one under test.
        only = batch()
        run = run_matching(
            corpus(
                batches=[only],
                bank_txns=[only.credit("BNK-00001"), only.credit("BNK-00002")],
            ),
            config(),
            decided_at=WHEN,
        )
        rows = {row.tier: row for row in run.tier_contributions}
        assert rows[Tier.T0_EXACT].candidates_proposed == 4
        assert rows[Tier.T0_EXACT].auto_matched == 2

    def test_marginal_equals_total_on_a_strictly_residual_ladder(self, clean_run):
        for row in clean_run.tier_contributions:
            assert row.marginal_auto_matched == row.auto_matched

    def test_no_tier_reports_an_llm_call(self, clean_run):
        assert all(row.llm_calls == 0 for row in clean_run.tier_contributions)

    def test_only_implemented_tiers_appear(self, clean_run):
        """A zero row for an unbuilt tier would be a false measurement."""
        assert {row.tier for row in clean_run.tier_contributions} == {
            Tier.T0_EXACT,
            Tier.T1_TOLERANCE,
            Tier.T2_AGGREGATION,
            Tier.T3_FUZZY,
            Tier.T4_GRAPH,
        }
        assert Tier.T5_LLM not in {row.tier for row in clean_run.tier_contributions}

    def test_evaluable_candidates_counts_only_the_scored_link_type(self, clean_run):
        assert clean_run.evaluable_candidates == 2
        assert clean_run.evaluable_candidates < len(clean_run.candidates)

    def test_the_decision_counts_partition_the_decisions(self, clean_run):
        total = clean_run.auto_matched + clean_run.needs_review + clean_run.exceptions
        rejected = len(clean_run.decisions_with(DecisionOutcome.REJECTED))
        assert total + rejected == len(clean_run.decisions)


class TestReproducibility:
    def test_two_runs_over_the_same_data_decide_identically(self, simple):
        first = run_matching(simple, config(), decided_at=WHEN)
        second = run_matching(simple, config(), decided_at=WHEN)
        assert first.state.decisions == second.state.decisions
        assert first.state.candidates == second.state.candidates
        assert first.predictions == second.predictions

    def test_candidate_ids_are_content_derived_not_counters(self, simple):
        run = run_matching(simple, config(), decided_at=WHEN)
        ids = [c.candidate_id for c in run.candidates]
        assert len(set(ids)) == len(ids)
        assert all("|" in identifier for identifier in ids)
        assert any("payment:PAY-00001" in identifier for identifier in ids)

    def test_a_reordered_corpus_produces_the_same_decisions(self):
        """Order of bank rows in the file must not change what is matched."""
        first_batch = batch("SETL-0001", utr="UTR2026031000001", amounts=(50_000,), first_index=1)
        second_batch = batch("SETL-0002", utr="UTR2026031000002", amounts=(70_000,), first_index=10)
        rows = [first_batch.credit("BNK-00001"), second_batch.credit("BNK-00002")]

        forward = run_matching(
            corpus(batches=[first_batch, second_batch], bank_txns=rows), config(), decided_at=WHEN
        )
        backward = run_matching(
            corpus(batches=[first_batch, second_batch], bank_txns=list(reversed(rows))),
            config(),
            decided_at=WHEN,
        )
        assert {d.pair for d in forward.decisions} == {d.pair for d in backward.decisions}

    def test_the_configuration_reaches_the_run_and_changes_the_result(self):
        only = batch(amounts=(60_000, 40_000))
        drifted = corpus(batches=[only], bank_txns=[only.credit(delta_minor=400)])
        default = run_matching(drifted, config(), decided_at=WHEN)
        tight = run_matching(
            drifted,
            config(tolerances=MatchingTolerances(amount_floor_minor=1, amount_bps=1)),
            decided_at=WHEN,
        )
        assert len(default.predictions) == 2
        assert tight.predictions == ()


class TestDiagnostics:
    def test_the_run_reports_what_it_saw(self):
        only = batch()
        run = run_matching(
            corpus(batches=[only], bank_txns=[only.credit(), noise_credit(amount_minor=777)]),
            config(),
            decided_at=WHEN,
        )
        assert run.credits_seen == 2
        assert run.credits_with_utr == 1
        assert run.credits_without_utr == 1
        assert run.settlements_seen == 1
        assert run.settlements_with_utr == 1
        assert run.credits_joined == 1

    def test_settlement_dispositions_partition_the_settlements(self):
        resolved = batch("SETL-0001", utr="UTR2026031000001", amounts=(50_000,), first_index=1)
        contested = batch("SETL-0002", utr="UTR2026031000002", amounts=(70_000,), first_index=10)
        unreachable = batch("SETL-0003", utr="UTR2026031000003", amounts=(90_000,), first_index=20)
        run = run_matching(
            corpus(
                batches=[resolved, contested, unreachable],
                bank_txns=[
                    resolved.credit("BNK-00001"),
                    contested.credit("BNK-00002"),
                    contested.credit("BNK-00003"),
                ],
            ),
            config(),
            decided_at=WHEN,
        )
        assert run.settlements_resolved == 1
        assert run.settlements_contested == 1
        assert run.settlements_unresolved == 1
        total = (
            run.settlements_resolved + run.settlements_contested + run.settlements_unresolved
        )
        assert total == run.settlements_seen

    def test_the_matcher_names_itself(self, clean_run):
        assert clean_run.name == MATCHER_NAME
        assert "tolerance" in clean_run.description


class TestAgainstTheRealCorpus:
    """The measured claim. If these regress, the step's headline is wrong."""

    def test_it_asserts_no_false_positive_on_the_fixture(self, fixture_scored):
        _, _, metrics = fixture_scored
        links = metrics.link_metrics
        assert links is not None
        assert links.false_positives == 0
        assert links.false_positive_cost_minor == 0

    def test_every_predicted_link_is_in_the_truth_set(self, fixture_scored):
        run, truth, _ = fixture_scored
        matrix = confusion([p.pair for p in run.predictions], truth.evaluation_pairs)
        assert matrix.false_positives == frozenset()

    def test_predicted_amounts_agree_with_the_truth_amounts_exactly(self, fixture_scored):
        """The allocation is the generator's own, so a correct link is correct in rupees."""
        run, truth, _ = fixture_scored
        truth_amounts = {
            link.pair: link.amount_minor
            for link in truth.links
            if link.link_type is LinkType.PAYMENT_CREDITED_AS
        }
        for prediction in run.predictions:
            assert prediction.amount_minor == truth_amounts[prediction.pair]

    def test_no_unmatchable_record_is_ever_asserted_about(self, fixture_scored):
        """Orphan credits and noise rows are irreconcilable by construction."""
        run, truth, _ = fixture_scored
        touched = {ref for p in run.predictions for ref in p.pair}
        assert touched & truth.unmatchable_refs == set()

    def test_the_tier_contributions_are_populated(self, fixture_scored):
        _, _, metrics = fixture_scored
        assert len(metrics.tier_contributions) == 5
        assert sum(row.candidates_proposed for row in metrics.tier_contributions) > 0

    def test_running_twice_gives_the_same_predictions(self, fixture_scored):
        run, _, _ = fixture_scored
        again = run_matching(
            ingest_dataset(FIXTURE, strict=True), config(), decided_at=WHEN
        )
        assert again.predictions == run.predictions
        assert again.state.decisions == run.state.decisions


class TestAgainstAGeneratedSplit:
    def test_precision_is_perfect_on_the_test_split(self, split_scored):
        _, _, metrics = split_scored
        links = metrics.link_metrics
        assert links is not None
        assert links.false_positives == 0
        assert metrics.auto_match_precision == 1.0

    def test_it_beats_the_baseline_on_precision_by_a_wide_margin(
        self, split_dir, split_scored
    ):
        """The comparison that justifies the step, measured rather than claimed."""
        _, truth, metrics = split_scored
        baseline = run_b0(split_dir)
        baseline_metrics = evaluate(baseline.predictions, truth, run_id="b0")
        assert metrics.auto_match_precision > baseline_metrics.auto_match_precision + 0.3
        assert baseline_metrics.link_metrics is not None
        assert baseline_metrics.link_metrics.false_positive_cost_minor > 0

    def test_the_precision_interval_lower_bound_is_credible(self, split_scored):
        """A perfect run still carries an honest lower bound, not [1.0, 1.0]."""
        _, _, metrics = split_scored
        links = metrics.link_metrics
        assert links is not None
        assert links.precision_ci_high == 1.0
        assert 0.9 < links.precision_ci_low < 1.0

    def test_recall_is_reported_honestly_and_is_still_short_of_the_target(
        self, split_scored
    ):
        """Five tiers reach nearly every link, and the rest is recorded as missed.

        Phase 2.3 moved this from 0.4422 to 0.8435 by de-duplicating the
        statement before the ladder ran, and Phase 2.5 to 0.9626 by completing
        the split payouts whose tranches had lost their reference. What remains
        is **one** settlement whose payments do not partition uniquely across
        its two tranches -- a refusal rather than a failure to look.

        A step that quietly optimised recall would have to give up precision to
        do it, and the precision column says it did not: still exactly 1.0,
        still zero false positives, across every one of those changes.
        """
        _, _, metrics = split_scored
        links = metrics.link_metrics
        assert links is not None
        assert 0.95 < links.recall < 1.0
        assert links.precision == 1.0
        assert links.false_positives == 0


class TestTheResidualLoop:
    """T2/T3/T4 repeat until a pass changes nothing, or the cap stops them."""

    @staticmethod
    def _split_corpus():
        """A split payout, so the residual tiers actually produce something."""
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        grosses = [p.amount_minor for p in only.payments]
        parts = allocate_minor(only.net_minor, [grosses[0], grosses[1] + grosses[2]])
        rows = [
            bank_credit("BNK-00001", amount_minor=parts[0], utr=only.settlement.utr),
            bank_credit("BNK-00002", amount_minor=parts[1], utr=only.settlement.utr),
        ]
        return corpus(batches=[only], bank_txns=rows)

    def test_it_stops_after_one_pass_when_nothing_is_left_to_do(self, clean_run):
        """T0 took the whole batch, so the first residual pass finds nothing."""
        assert clean_run.passes == 1

    def test_a_productive_pass_earns_another(self):
        """T2 resolves the split in pass one, so the loop looks again."""
        run = run_matching(self._split_corpus(), config(), decided_at=WHEN)
        assert run.passes == 2
        assert run.aggregation.settlements_resolved == 1

    def test_the_cap_bounds_a_run_that_would_otherwise_look_again(self):
        """An unbounded loop in a reconciliation run is a hang, not a slow answer."""
        capped = run_matching(
            self._split_corpus(),
            config(graph=GraphInference(max_rerun_passes=1)),
            decided_at=WHEN,
        )
        assert capped.passes == 1

    def test_capping_the_loop_does_not_change_what_the_first_pass_found(self):
        built = self._split_corpus()
        full = run_matching(built, config(), decided_at=WHEN)
        capped = run_matching(
            built, config(graph=GraphInference(max_rerun_passes=1)), decided_at=WHEN
        )
        assert capped.predictions == full.predictions

    def test_the_loop_reports_how_many_passes_it_used(self, clean_run):
        assert 1 <= clean_run.passes <= clean_run.state.config.graph.max_rerun_passes
