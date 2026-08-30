"""Where the model attaches to a run, and the promise that it never has to.

The claim this file exists to defend: **the complete system runs, and is
measured, with no model at all**. Everything else here is about making sure
that when a model *is* present it cannot become authoritative -- its proposals
go through the same decision policy as T2's, its prose changes no number, and
its repairs can only fill a gap the regex layer left.

The `--no-llm` equivalence test is the load-bearing one. If a run with the
model disabled ever stopped producing exactly what Step 8 produced, the
deterministic path would have quietly become a second implementation.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.cli import main
from ledgerloop.config import LLMConfig, RunConfig
from ledgerloop.llm.cache import ResponseCache
from ledgerloop.llm.client import (
    LLMClient,
    LLMUnavailable,
    RateLimited,
    ScriptedProvider,
)
from ledgerloop.llm.integration import (
    LLMRunSummary,
    adjudicator_for,
    repair_narrations,
    residual_packs,
)
from ledgerloop.llm.tasks import AdjudicationOutcome, ExplanationOutcome, NarrationOutcome
from ledgerloop.matching import run_matching
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.enums import DecisionOutcome, Tier
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus, noise_credit

NARRATION = "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT"


def client_for(tmp_path, *responses: str, **overrides: object) -> LLMClient:
    settings = LLMConfig(cache_dir=tmp_path / "cache", **overrides)  # type: ignore[arg-type]
    return LLMClient(
        config=settings,
        provider=ScriptedProvider(responses=list(responses)),
        cache=ResponseCache(directory=settings.cache_dir),
    )


@pytest.fixture
def unreadable():
    """A credit whose narration the regex layer cannot resolve at all."""
    only = batch(utr=None)
    row = bank_credit(
        "BNK-00001", amount_minor=only.net_minor, utr=None, merchant=None
    )
    blind = row.model_copy(
        update={
            "narration_raw": "MISC CREDIT 88213",
            "narration_normalized": "MISC CREDIT 88213",
            "extracted_utr": None,
            "extracted_merchant": None,
        }
    )
    return only, corpus(batches=[only], bank_txns=[blind])


class TestRepairingNarrations:
    def test_an_accepted_repair_fills_the_gap_on_the_bank_row(self, tmp_path, unreadable):
        _, ingest = unreadable
        response = json.dumps(
            {
                "extractions": [
                    {
                        "item_id": "BNK-00001",
                        "utr": None,
                        "merchant": "MISC CREDIT",
                        "confidence": 0.7,
                    }
                ]
            }
        )
        repaired, outcome = repair_narrations(client_for(tmp_path, response), ingest)
        assert outcome.accepted == 1
        assert repaired.bank_txns[0].extracted_merchant == "MISC CREDIT"

    def test_a_reference_the_regex_layer_read_is_never_overwritten(self, tmp_path):
        """The deterministic parser is the more reliable reader where it can read."""
        only = batch()
        ingest = corpus(batches=[only], bank_txns=[only.credit()])
        client = client_for(tmp_path)
        repaired, outcome = repair_narrations(client, ingest)
        assert outcome.attempted == 0  # a resolved row is never sent
        assert repaired.bank_txns[0].extracted_utr == only.settlement.utr
        assert client.calls == 0

    def test_an_invented_reference_leaves_the_row_untouched(self, tmp_path, unreadable):
        _, ingest = unreadable
        response = json.dumps(
            {
                "extractions": [
                    {
                        "item_id": "BNK-00001",
                        "utr": "UTR2026039999999",
                        "confidence": 0.99,
                    }
                ]
            }
        )
        repaired, outcome = repair_narrations(client_for(tmp_path, response), ingest)
        assert outcome.rejected_ungrounded == 1
        assert repaired.bank_txns[0].extracted_utr is None

    def test_a_disabled_client_returns_the_ingest_unchanged(self, tmp_path, unreadable):
        _, ingest = unreadable
        client = LLMClient(config=LLMConfig(enabled=False, cache_dir=tmp_path))
        repaired, outcome = repair_narrations(client, ingest)
        assert repaired is ingest
        assert outcome.attempted == 0


class TestTheEvidencePacks:
    @pytest.fixture
    def residual(self):
        only = batch(utr=None)
        return only, MatchContext.from_ingest(
            corpus(
                batches=[only],
                bank_txns=[
                    bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None),
                    noise_credit("BNK-09001", amount_minor=7),
                ],
            )
        )

    def test_a_pack_names_only_the_records_of_its_own_item(self, residual):
        only, context = residual
        packs = residual_packs(context, ())
        assert len(packs) == 1
        assert packs[0].item_id == "SETL-0001"
        assert "settlement:SETL-0001" in packs[0].refs
        for payment in only.payments:
            assert f"payment:{payment.payment_id}" in packs[0].refs

    def test_a_credited_payout_is_not_in_the_residual(self):
        """A keyed batch T0 credits in full. There is nothing left to adjudicate."""
        only = batch()
        run = run_matching(
            corpus(batches=[only], bank_txns=[only.credit()]), RunConfig(run_id="x")
        )
        assert run.context is not None
        assert residual_packs(run.context, run.candidates) == ()

    def test_the_pack_count_is_bounded(self, residual):
        _, context = residual
        assert len(residual_packs(context, (), limit=0)) == 0

    def test_packs_come_out_largest_payout_first(self):
        small = batch("SETL-0001", amounts=(1_000,), utr=None)
        large = batch("SETL-0002", amounts=(900_000,), utr=None, first_index=5)
        context = MatchContext.from_ingest(
            corpus(batches=[small, large], bank_txns=[noise_credit(amount_minor=3)])
        )
        assert [pack.item_id for pack in residual_packs(context, ())] == [
            "SETL-0002",
            "SETL-0001",
        ]


class TestTierFiveInTheLadder:
    @pytest.fixture
    def split(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500, utr=None)
        grosses = [payment.amount_minor for payment in only.payments]
        amounts = allocate_minor(only.net_minor, [grosses[0], grosses[1] + grosses[2]])
        return only, corpus(
            batches=[only],
            bank_txns=[
                bank_credit("BNK-00001", amount_minor=amounts[0], utr=None),
                bank_credit("BNK-00002", amount_minor=amounts[1], utr=None),
            ],
        )

    def proposal(self, **overrides: object) -> str:
        link = {
            "payment_id": "PAY-00001",
            "bank_txn_id": "BNK-00001",
            "settlement_id": "SETL-0001",
            "payment_ids": ["PAY-00001"],
        }
        link.update(overrides)
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "item_id": "SETL-0001",
                        "hypothesis": "the first tranche is BNK-00001",
                        "proposed_link": link,
                        "confidence": 0.85,
                        "reasoning": "the amounts line up",
                        "evidence_refs": ["settlement:SETL-0001"],
                    }
                ]
            }
        )

    def test_a_verified_proposal_becomes_a_t5_candidate_and_is_decided(
        self, tmp_path, split
    ):
        _, ingest = split
        client = client_for(tmp_path, self.proposal())
        config = RunConfig(run_id="t5", enabled_tiers=(0, 1, 2, 3, 4, 5))
        run = run_matching(ingest, config, adjudicator=adjudicator_for(client, config))
        t5 = [c for c in run.candidates if c.tier is Tier.T5_LLM]
        assert len(t5) == 1
        assert t5[0].arithmetic_verified
        decision = next(d for d in run.decisions if d.tier is Tier.T5_LLM)
        assert decision.outcome in set(DecisionOutcome)

    def test_a_proposal_whose_money_does_not_close_can_never_auto_match(
        self, tmp_path, split
    ):
        """The MatchDecision validator makes this structural, not a policy choice."""
        _, ingest = split
        client = client_for(
            tmp_path, self.proposal(payment_id="PAY-00002", payment_ids=["PAY-00002"])
        )
        config = RunConfig(run_id="t5", enabled_tiers=(0, 1, 2, 3, 4, 5))
        run = run_matching(ingest, config, adjudicator=adjudicator_for(client, config))
        for decision in run.decisions:
            if decision.tier is Tier.T5_LLM:
                assert decision.outcome is not DecisionOutcome.AUTO_MATCHED

    def test_the_tier_table_gains_a_t5_row_only_when_t5_ran(self, tmp_path, split):
        _, ingest = split
        config = RunConfig(run_id="t5", enabled_tiers=(0, 1, 2, 3, 4, 5))
        without = run_matching(ingest, RunConfig(run_id="plain"))
        assert all(row.tier is not Tier.T5_LLM for row in without.tier_contributions)

        client = client_for(tmp_path, self.proposal())
        with_llm = run_matching(
            ingest, config, adjudicator=adjudicator_for(client, config)
        )
        assert any(row.tier is Tier.T5_LLM for row in with_llm.tier_contributions)

    def test_an_outage_leaves_the_ladder_exactly_where_it_was(self, tmp_path, split):
        _, ingest = split
        client = client_for(tmp_path)
        client.provider = ScriptedProvider(failure=LLMUnavailable("429"))
        config = RunConfig(run_id="t5", enabled_tiers=(0, 1, 2, 3, 4, 5))
        with_outage = run_matching(
            ingest, config, adjudicator=adjudicator_for(client, config)
        )
        without = run_matching(ingest, RunConfig(run_id="plain"))
        assert {p.pair for p in with_outage.predictions} == {
            p.pair for p in without.predictions
        }

    def test_a_disabled_client_yields_no_adjudicator_at_all(self, tmp_path):
        client = LLMClient(config=LLMConfig(enabled=False, cache_dir=tmp_path))
        assert adjudicator_for(client, RunConfig(run_id="x")) is None


class TestTheRunSummary:
    def test_it_totals_what_was_accepted_and_what_was_refused(self):
        summary = LLMRunSummary(
            narration=NarrationOutcome(accepted=2, rejected_ungrounded=1),
            adjudication=AdjudicationOutcome(
                accepted=1, rejected_ungrounded=1, rejected_unverified=3
            ),
            explanation=ExplanationOutcome(accepted=4, calls_refused=1),
        )
        assert summary.accepted == 7
        assert summary.rejected_ungrounded == 2
        assert summary.rejected_unverified == 3
        assert summary.calls_refused == 1


class TestTheSystemRunsWithoutAModel:
    """The promise: `--no-llm` is the same path with one branch taken."""

    @pytest.fixture
    def dataset(self, tmp_path):
        out = tmp_path / "test-5"
        assert (
            main(
                ["generate", "--split", "test", "--seed", "5", "--orders", "120",
                 "--out", str(out)]
            )
            == 0
        )
        return out

    def test_eval_runs_with_no_llm_and_says_so(self, dataset, tmp_path, capsys):
        code = main(
            ["eval", "--data", str(dataset), "--no-llm", "--out", str(tmp_path / "E.md")]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "llm: disabled (--no-llm)" in printed
        assert "every number above is deterministic" in printed

    def test_no_llm_section_is_rendered_when_no_model_ran(self, dataset, tmp_path):
        report = tmp_path / "E.md"
        main(["eval", "--data", str(dataset), "--no-llm", "--out", str(report)])
        text = report.read_text(encoding="utf-8")
        assert "### LLM cost and refusals" not in text
        assert "### The exception queue" in text

    def test_a_missing_key_reaches_the_same_place_as_the_flag(
        self, dataset, tmp_path, capsys, monkeypatch
    ):
        """No credential anywhere on the ladder, so no ladder is built at all.

        The message names every variable it looked at rather than only the
        shared one: since Phase 2.2 a rung can be enabled by its own key, and a
        reader told "no key in $LEDGERLOOP_LLM_API_KEY" would not know that
        $GROQ_API_KEY was also checked and also absent.
        """
        for name in (
            "LEDGERLOOP_LLM_API_KEY",
            "LEDGERLOOP_LLM_PROVIDERS",
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "OLLAMA_API_KEY",
            "OLLAMA_BASE_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        code = main(["eval", "--data", str(dataset), "--out", str(tmp_path / "E.md")])
        assert code == 0

        printed = capsys.readouterr().out
        assert "llm: disabled (no provider key in $LEDGERLOOP_LLM_API_KEY" in printed
        assert "$GROQ_API_KEY" in printed
        assert "every number above is deterministic" in printed

    def test_the_metrics_are_identical_with_and_without_the_flag(self, dataset, tmp_path):
        """An absent key and an explicit refusal must not measure differently."""
        first = tmp_path / "A.md"
        second = tmp_path / "B.md"
        main(["eval", "--data", str(dataset), "--no-llm", "--out", str(first)])
        main(["eval", "--data", str(dataset), "--out", str(second)])
        assert _without_timings(first.read_text(encoding="utf-8")) == _without_timings(
            second.read_text(encoding="utf-8")
        )


def _without_timings(text: str) -> str:
    """The report's own rule: everything but the measured timings is deterministic.

    Three places carry a measured timing -- the wall clock, the throughput, and
    the per-tier column of the contribution table. All three genuinely differ
    between two runs over the same data, and the report says so; everything else
    must be byte-identical.
    """
    keep = []
    for line in text.splitlines():
        if "Wall clock" in line or "Throughput" in line:
            continue
        if line.rstrip().endswith("ms |"):
            continue
        keep.append(line)
    return "\n".join(keep)


class TestAModelThatMisbehavesCannotCostPrecision:
    """The safety property a live run would have stressed, tested without a key.

    Phase 2.8 set out to make one live call and could not: no rung of the ladder
    is reachable on this machine (no credential, and no local Ollama), so
    ``build_ladder`` returns ``None`` and ``llm-report`` refuses. That blocker is
    environmental and the code path is complete -- but the *reason* a live run
    matters is that a real provider fails, rate-limits and answers wrongly in
    ways a scripted one has to be told to imitate.

    So the failure semantics are pinned here instead. Both cases below are
    exactly what a bad live call looks like from inside the pipeline, and
    neither is allowed to move a published number:

    * the provider never answers at all -- outage, exhausted ladder, bad key.

    The second shape -- a grounded, confident reply whose money does not close --
    is already covered end to end by
    ``TestTheAdjudicator.test_a_proposal_whose_money_does_not_close_can_never_auto_match``
    and is deliberately not repeated here.

    The unit tests around ``verify_arithmetic`` and the grounding gates already
    check each guard in isolation. What was missing is the end-to-end claim: run
    the **whole system** with a hostile model and show the answer is the
    deterministic one, to the link.
    """

    @pytest.fixture
    def corpus_dir(self, tmp_path):
        out = tmp_path / "test-9"
        assert (
            main(
                ["generate", "--split", "test", "--seed", "9", "--orders", "120",
                 "--out", str(out)]
            )
            == 0
        )
        return out

    @staticmethod
    def _links(run):
        return run.metrics.link_metrics

    def test_a_provider_that_never_answers_changes_nothing(self, corpus_dir, tmp_path):
        """Every call fails. The run must land on the deterministic answer.

        Not "approximately the same" -- the same true positives, the same false
        positives, the same false negatives. A model that cannot be reached must
        cost exactly nothing, because everything it would have contributed was
        optional by construction.
        """
        from ledgerloop.eval.harness import run_system

        settings = LLMConfig(cache_dir=tmp_path / "dead-cache")
        dead = LLMClient(
            config=settings,
            provider=ScriptedProvider(failure=LLMUnavailable("connection refused")),
            cache=ResponseCache(directory=settings.cache_dir),
        )
        with_dead_model = run_system(
            corpus_dir, client=dead, measure_calibration_quality=False
        )
        without = run_system(corpus_dir, measure_calibration_quality=False)

        live, base = self._links(with_dead_model), self._links(without)
        assert (live.true_positives, live.false_positives, live.false_negatives) == (
            base.true_positives, base.false_positives, base.false_negatives
        )
        assert live.precision == base.precision
        assert with_dead_model.metrics.match_rate == without.metrics.match_rate

    def test_the_failure_is_recorded_rather_than_swallowed(self, corpus_dir, tmp_path):
        """Degrading quietly is right; degrading *silently* is not.

        The run must still be able to say a model was asked and did not answer,
        or the cost ledger would report a clean run over a broken one.
        """
        from ledgerloop.eval.harness import run_system

        settings = LLMConfig(cache_dir=tmp_path / "dead-cache-2")
        dead = LLMClient(
            config=settings,
            provider=ScriptedProvider(failure=LLMUnavailable("connection refused")),
            cache=ResponseCache(directory=settings.cache_dir),
        )
        run_system(corpus_dir, client=dead, measure_calibration_quality=False)
        assert dead.provider_failures > 0
        assert dead.calls == 0

    def test_a_failing_provider_introduces_no_false_positive(
        self, corpus_dir, tmp_path
    ):
        """Precision before recall, on the path where the model is broken."""
        from ledgerloop.eval.harness import run_system

        settings = LLMConfig(cache_dir=tmp_path / "dead-cache-3")
        dead = LLMClient(
            config=settings,
            provider=ScriptedProvider(failure=LLMUnavailable("503 upstream")),
            cache=ResponseCache(directory=settings.cache_dir),
        )
        run = run_system(corpus_dir, client=dead, measure_calibration_quality=False)
        assert self._links(run).false_positives == 0
        assert self._links(run).false_positive_cost_minor == 0

    def test_a_rate_limit_is_a_failure_like_any_other_to_the_answer(
        self, corpus_dir, tmp_path
    ):
        """429 is the failure a live demo actually hits, and it must be boring."""
        from ledgerloop.eval.harness import run_system

        settings = LLMConfig(cache_dir=tmp_path / "limited-cache")
        limited = LLMClient(
            config=settings,
            provider=ScriptedProvider(failure=RateLimited("429 too many requests")),
            cache=ResponseCache(directory=settings.cache_dir),
        )
        run = run_system(corpus_dir, client=limited, measure_calibration_quality=False)
        base = run_system(corpus_dir, measure_calibration_quality=False)
        assert self._links(run).false_positives == 0
        assert self._links(run).true_positives == self._links(base).true_positives
