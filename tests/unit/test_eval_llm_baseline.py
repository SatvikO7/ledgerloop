"""B2 -- the LLM-only baseline, its cost accounting, and its stand-in reasoner.

Three things are tested, and only the first is about accuracy:

1. **The path.** Prompt, schema, cache, budget, retry, ledger -- B2 goes through
   the same client as the production call sites, so a failure anywhere in that
   chain degrades rather than crashes.
2. **The isolation.** B2's output reaches `evaluate` and the report and nothing
   else. It cannot enter the ladder, the decision policy or the calibrator.
3. **The absence of the gates.** An invented id becomes a false positive rather
   than a refusal, because that is what B2 means.

The stand-in reasoner is tested for the property that makes it publishable: it
reads the prompt and nothing else.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.config import GeneratorConfig, LLMConfig
from ledgerloop.eval.harness import run_system
from ledgerloop.eval.llm_baseline import (
    B2_PROMPT_VERSION,
    LLMBaselineArtifact,
    LLMOnlyBatch,
    render_reconciliation,
    run_b2,
)
from ledgerloop.eval.metrics import evaluate
from ledgerloop.eval.offline_provider import OFFLINE_PROVIDER_NAME, OfflineReasoner
from ledgerloop.eval.truth_io import load_ground_truth, load_manifest
from ledgerloop.generator import generate_to_disk
from ledgerloop.ingest import ingest_dataset
from ledgerloop.llm.cache import ResponseCache
from ledgerloop.llm.client import LLMClient, LLMUnavailable, ScriptedProvider
from ledgerloop.models.enums import SplitName


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("b2") / "dev"
    generate_to_disk(GeneratorConfig(split=SplitName.DEV, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def ingested(corpus):
    return ingest_dataset(corpus, strict=False)


def _client(tmp_path, provider, **config):
    return LLMClient(
        config=LLMConfig(enabled=True, cache_dir=tmp_path / "cache", **config),
        provider=provider,
    )


def _answer(pairs):
    return json.dumps(
        {"links": [{"payment_id": p, "bank_txn_id": b, "confidence": 0.9} for p, b in pairs]}
    )


class TestItRunsOrSaysItDidNot:
    def test_a_disabled_client_produces_a_not_run_artefact(self, ingested, corpus):
        """Not a precision of zero. A baseline that made no attempt has not
        achieved anything, and the report renders that as `_pending_`."""
        client = LLMClient(config=LLMConfig(enabled=False), provider=None)
        predictions, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert predictions == ()
        assert artifact.ran is False
        assert "no model was reachable" in artifact.reason
        assert artifact.precision == 0.0
        assert artifact.calls_attempted == 0

    def test_it_still_records_what_it_would_have_offered(self, ingested, corpus):
        client = LLMClient(config=LLMConfig(enabled=False), provider=None)
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.payments_offered > 0
        assert artifact.credits_offered > 0


class TestThePrompt:
    def test_the_whole_statement_goes_into_every_prompt(self, ingested):
        """A payment can be credited by any row, so the payment side is chunked
        and the bank side is not. That is where B2's token cost comes from."""
        payments = list(ingested.payments)[:3]
        credits = [txn for txn in ingested.bank_txns if txn.is_credit]
        prompt = render_reconciliation(payments, credits)
        for txn in credits:
            assert txn.txn_id in prompt
        assert prompt.count("PAY-") >= 3

    def test_it_asks_for_nothing_but_the_pairs(self, ingested):
        """B2 has no arithmetic gate, so it does not ask for an amount -- there
        would be nothing to check one against."""
        prompt = render_reconciliation(list(ingested.payments)[:2], [])
        assert "amount" not in prompt.lower().split("Schema:")[-1]
        assert '"payment_id": str, "bank_txn_id": str' in prompt

    def test_the_prompt_version_is_its_own(self):
        """A cache keyed without a distinct version would serve a production
        adjudication answer to a B2 question."""
        assert B2_PROMPT_VERSION == "reconcile-all/1.0.0"


class TestNoGates:
    def test_an_invented_payment_id_is_asserted_and_becomes_a_false_positive(
        self, ingested, corpus, tmp_path
    ):
        """The grounding gate's absence, measured. In production `llm/gates.py`
        refuses the citation and discards the whole extraction."""
        credit = next(txn for txn in ingested.bank_txns if txn.is_credit)
        provider = ScriptedProvider(
            responses=[_answer([("PAY-99999", credit.txn_id)])] * 10
        )
        client = _client(tmp_path, provider)
        predictions, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.unknown_payment_ids >= 1
        truth = load_ground_truth(corpus)
        metrics = evaluate(predictions, truth, run_id="b2")
        assert metrics.link_metrics is not None
        assert metrics.link_metrics.false_positives >= 1

    def test_an_invented_bank_id_is_counted_too(self, ingested, corpus, tmp_path):
        payment = next(p for p in ingested.payments if p.settlement_id is not None)
        provider = ScriptedProvider(
            responses=[_answer([(payment.payment_id, "BNK-99999")])] * 10
        )
        client = _client(tmp_path, provider)
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.unknown_bank_txn_ids >= 1

    def test_a_repeated_link_is_de_duplicated_and_counted(self, ingested, corpus, tmp_path):
        """Asserting the same link twice is one claim, not two. Counting it
        twice would let the baseline inflate its own denominator."""
        payment = next(p for p in ingested.payments if p.settlement_id is not None)
        credit = next(txn for txn in ingested.bank_txns if txn.is_credit)
        pair = (payment.payment_id, credit.txn_id)
        provider = ScriptedProvider(responses=[_answer([pair, pair])] * 10)
        client = _client(tmp_path, provider)
        predictions, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.links_duplicated >= 1
        assert len({link.pair for link in predictions}) == len(predictions)

    def test_a_malformed_id_is_dropped_rather_than_crashing_the_baseline(
        self, ingested, corpus, tmp_path
    ):
        """An id carrying the record-key separator cannot be scored as a link at
        all. It is counted as unknown and dropped, because a baseline that
        crashed on the evidence against it would report nothing."""
        provider = ScriptedProvider(
            responses=[_answer([("PAY:BAD", "BNK-00001")])] * 10
        )
        client = _client(tmp_path, provider)
        predictions, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.unknown_payment_ids >= 1
        assert all(":" not in link.source_ref.record_id for link in predictions)


class TestFailuresDegradeRatherThanCrash:
    def test_a_provider_outage_is_a_failed_batch_and_nothing_else(
        self, ingested, corpus, tmp_path
    ):
        """B2 has no deterministic answer to fall back to -- that is the whole
        difference between it and the production system -- so a failed batch is
        recorded as one rather than papered over."""
        provider = ScriptedProvider(failure=LLMUnavailable("429"))
        client = _client(tmp_path, provider)
        predictions, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert predictions == ()
        assert artifact.ran is True
        assert artifact.calls_failed == artifact.calls_attempted > 0

    def test_unparseable_output_is_a_failed_batch(self, ingested, corpus, tmp_path):
        provider = ScriptedProvider(responses=["not json at all"] * 20)
        client = _client(tmp_path, provider, validation_retries=0)
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.calls_failed > 0

    def test_the_budget_stops_it_before_the_network(self, ingested, corpus, tmp_path):
        provider = ScriptedProvider(responses=[_answer([])] * 50)
        client = _client(tmp_path, provider, max_calls_per_run=1)
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.cost.llm_calls == 1
        assert artifact.calls_failed > 0


class TestCostAccounting:
    def test_calls_and_tokens_are_the_client_ledger(self, ingested, corpus, tmp_path):
        provider = ScriptedProvider(
            responses=[_answer([])] * 50, prompt_tokens=100, completion_tokens=40
        )
        client = _client(tmp_path, provider)
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.cost.llm_calls == artifact.calls_attempted
        assert artifact.cost.total_tokens == artifact.cost.llm_calls * 140
        assert artifact.cost.actual_cost_inr == 0.0
        assert artifact.cost.equivalent_paid_cost_inr > 0.0

    def test_a_rerun_over_a_warm_cache_makes_zero_live_calls(
        self, ingested, corpus, tmp_path
    ):
        """The published figure is the cold cost, because a warm cache reports
        zero and that is true of any rerun. This is the guarantee that claim
        rests on, asserted rather than stated in prose."""
        cache = tmp_path / "cache"
        first = LLMClient(
            config=LLMConfig(enabled=True, cache_dir=cache),
            provider=OfflineReasoner(),
        )
        _, cold = run_b2(first, ingested, load_manifest(corpus))
        assert cold.cost.llm_calls > 0
        assert cold.cost.cache_hits == 0

        second = LLMClient(
            config=LLMConfig(enabled=True, cache_dir=cache),
            provider=OfflineReasoner(),
        )
        _, warm = run_b2(second, ingested, load_manifest(corpus))
        assert warm.cost.llm_calls == 0
        assert warm.cost.cache_hits == cold.cost.llm_calls
        assert warm.cost.cache_hit_rate == 1.0

    def test_the_cache_is_keyed_on_the_b2_prompt_version(self, tmp_path):
        """Otherwise a production adjudication answer could be served to a B2
        question, and neither run would notice."""
        from ledgerloop.llm.cache import CacheKey

        cache = ResponseCache(directory=tmp_path / "cache")
        left = CacheKey("p", "m", 0.0, B2_PROMPT_VERSION, "same prompt")
        right = CacheKey("p", "m", 0.0, "adjudicate/1.0.0", "same prompt")
        cache.put(left, "b2 answer")
        assert cache.get(left) == "b2 answer"
        assert cache.get(right) is None

    def test_the_token_multiple_has_no_value_without_a_denominator(self):
        artifact = LLMBaselineArtifact(ran=True, system_ran=False)
        assert artifact.token_multiple == 0.0

    def test_and_is_a_ratio_when_it_has_one(self):
        from ledgerloop.models.metrics import CostLedger

        artifact = LLMBaselineArtifact(
            ran=True,
            system_ran=True,
            cost=CostLedger(prompt_tokens=8_000),
            system_cost=CostLedger(prompt_tokens=200),
        )
        assert artifact.token_multiple == 40.0


class TestIsolationFromTheProductionSystem:
    def test_a_b2_run_does_not_change_the_pipeline_run(self, corpus, ingested, tmp_path):
        """The whole architectural claim: B2's answers reach the evaluator and
        the report, and nothing else."""
        before = run_system(corpus, measure_calibration_quality=False)
        client = _client(tmp_path, OfflineReasoner())
        run_b2(client, ingested, load_manifest(corpus))
        after = run_system(corpus, measure_calibration_quality=False)
        assert before.matched.predictions == after.matched.predictions

    def test_b2_output_never_carries_a_calibrated_probability(
        self, ingested, corpus, tmp_path
    ):
        """A `PredictedLink` has no probability field at all, which is the point:
        B2 has no calibrator, so there is nothing honest for it to report."""
        client = _client(tmp_path, OfflineReasoner())
        predictions, _ = run_b2(client, ingested, load_manifest(corpus))
        assert predictions
        for link in predictions:
            assert not hasattr(link, "calibrated_p")

    def test_the_contract_cannot_express_an_amount(self):
        """`LLMOnlyLink` is deliberately not `llm.contracts.ProposedLink`: that
        one is a proposal headed for `verify_arithmetic`."""
        assert set(LLMOnlyBatch.model_fields) == {"links"}
        from ledgerloop.eval.llm_baseline import LLMOnlyLink

        assert set(LLMOnlyLink.model_fields) == {
            "payment_id",
            "bank_txn_id",
            "confidence",
        }


class TestTheStandInReasoner:
    def test_it_reads_the_prompt_and_nothing_else(self, ingested):
        """The property that makes its row publishable. It takes one string; it
        opens no file and sees no ground truth."""
        import inspect

        signature = inspect.signature(OfflineReasoner.complete)
        assert list(signature.parameters) == ["self", "prompt", "timeout_s"]
        source = inspect.getsource(OfflineReasoner)
        assert "open(" not in source
        assert "truth" not in source.lower().replace("truth_io", "")

    def test_it_answers_a_real_prompt_with_valid_json(self, ingested):
        payments = [p for p in ingested.payments if p.settlement_id is not None][:5]
        credits = [txn for txn in ingested.bank_txns if txn.is_credit]
        prompt = render_reconciliation(payments, credits)
        completion = OfflineReasoner().complete(prompt, timeout_s=1.0)
        parsed = LLMOnlyBatch.model_validate_json(completion.text)
        assert parsed.links

    def test_it_never_names_a_record_that_was_not_in_the_prompt(self, ingested):
        """It cannot invent an id, which is why B2's invented-id counters read
        zero here. A real model can, and the report says so rather than letting
        the zero look like a result."""
        payments = [p for p in ingested.payments if p.settlement_id is not None][:8]
        credits = [txn for txn in ingested.bank_txns if txn.is_credit]
        prompt = render_reconciliation(payments, credits)
        parsed = LLMOnlyBatch.model_validate_json(
            OfflineReasoner().complete(prompt, timeout_s=1.0).text
        )
        known_payments = {payment.payment_id for payment in payments}
        known_credits = {txn.txn_id for txn in credits}
        for link in parsed.links:
            assert link.payment_id in known_payments
            assert link.bank_txn_id in known_credits

    def test_it_is_deterministic(self, ingested):
        payments = [p for p in ingested.payments if p.settlement_id is not None][:6]
        credits = [txn for txn in ingested.bank_txns if txn.is_credit]
        prompt = render_reconciliation(payments, credits)
        first = OfflineReasoner().complete(prompt, timeout_s=1.0).text
        second = OfflineReasoner().complete(prompt, timeout_s=1.0).text
        assert first == second

    def test_it_answers_nothing_when_there_are_no_credits(self, ingested):
        prompt = render_reconciliation(list(ingested.payments)[:3], [])
        parsed = LLMOnlyBatch.model_validate_json(
            OfflineReasoner().complete(prompt, timeout_s=1.0).text
        )
        assert parsed.links == ()

    def test_its_provider_name_marks_the_row(self, ingested, corpus, tmp_path):
        """The artefact records which reasoner answered, so the report can print
        the banner rather than leaving a reader to assume a model."""
        client = _client(tmp_path, OfflineReasoner())
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.cost.provider_used == OFFLINE_PROVIDER_NAME
        marked = artifact.model_copy(update={"provider_kind": OFFLINE_PROVIDER_NAME})
        assert marked.is_standin is True
        assert artifact.model_copy(update={"provider_kind": "live"}).is_standin is False


class TestTheArtefact:
    def test_it_round_trips_through_disk(self, ingested, corpus, tmp_path):
        client = _client(tmp_path, OfflineReasoner())
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        path = tmp_path / "b2.json"
        artifact.save(path)
        assert LLMBaselineArtifact.load(path).model_dump_json() == artifact.model_dump_json()

    def test_it_carries_the_corpus_it_was_measured_on(self, ingested, corpus, tmp_path):
        """B2 runs on `dev` and the rest of the table on `test`. A row whose
        scope is not recorded is a row that will be compared to the wrong thing."""
        client = _client(tmp_path, OfflineReasoner())
        _, artifact = run_b2(client, ingested, load_manifest(corpus))
        assert artifact.split == "dev"
        assert artifact.seed == 42
        assert artifact.generator_version == "0.2.0"
