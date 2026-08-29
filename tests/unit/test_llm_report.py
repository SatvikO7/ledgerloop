"""The measured run of the production LLM path, and the stand-in that drives it.

Two things are under test and they are different claims:

* the **stand-in** reads the prompt and nothing else, and answers each of the
  three production prompt shapes with something the contract accepts;
* the **report** measures the real path when driven by it, and carries the
  ``--no-llm`` control that makes "the model proposes, deterministic code
  decides" a measurement rather than an assertion.

Nothing here claims anything about a language model's answer quality, and one
test asserts the artefact says so.
"""

from __future__ import annotations

import inspect

import pytest

from ledgerloop.config import GeneratorConfig, LLMConfig, SplitName
from ledgerloop.eval.llm_report import run_llm_report, score_of
from ledgerloop.generator import generate_to_disk
from ledgerloop.llm.client import LLMClient
from ledgerloop.llm.contracts import AdjudicationBatch, ExplanationBatch, NarrationBatch
from ledgerloop.llm.offline_analyst import OFFLINE_ANALYST_NAME, OfflineAnalyst
from ledgerloop.llm.prompts import (
    EvidencePack,
    render_adjudication,
    render_explanation,
    render_narration,
)


def _answer(prompt: str) -> str:
    return OfflineAnalyst().complete(prompt, timeout_s=1.0).text


class TestTheStandInReadsOnlyThePrompt:
    def test_it_takes_no_argument_but_the_prompt_and_a_timeout(self):
        """The claim that it cannot see ground truth, as a signature."""
        signature = inspect.signature(OfflineAnalyst.complete)
        assert list(signature.parameters) == ["self", "prompt", "timeout_s"]

    def test_a_narration_answer_satisfies_the_narration_contract(self):
        prompt = render_narration(
            [
                ("item-0", "NEFT-UTR2026031012345-RAZORPAY SOFTWARE PVT-PAYOUT"),
                ("item-1", "RENT PAYMENT COMMERCIAL PREMISES"),
            ]
        )
        batch = NarrationBatch.model_validate_json(_answer(prompt))
        by_id = {item.item_id: item for item in batch.extractions}

        assert by_id["item-0"].utr == "UTR2026031012345"
        assert by_id["item-1"].utr is None, "no reference in that text to find"

    def test_every_value_it_returns_is_present_in_the_text_it_was_given(self):
        """Which is what the grounding gate checks. It cannot pass by inventing."""
        narration = "NEFT-UTR2026031099999-NYKAA E RETAIL PVT LTD-SETTLEMENT"
        batch = NarrationBatch.model_validate_json(
            _answer(render_narration([("item-0", narration)]))
        )
        item = batch.extractions[0]
        assert item.utr is not None and item.utr in narration
        assert item.merchant is not None and item.merchant in narration

    def test_an_adjudication_answer_cites_only_the_packs_own_records(self):
        pack = EvidencePack(
            item_id="SETL-0001",
            summary="payout of ₹1,00,000.00 declared on 2026-03-10, not credited",
            refs=("settlement:SETL-0001", "payment:PAY-00001", "bank_txn:BNK-00001"),
            facts=("reference published: none",),
            candidates=("BNK-00001 credits ₹1,00,000.00 on 2026-03-11, narration '...'",),
        )
        batch = AdjudicationBatch.model_validate_json(
            _answer(render_adjudication([pack]))
        )
        hypothesis = batch.hypotheses[0]

        assert hypothesis.item_id == "SETL-0001"
        assert set(hypothesis.evidence_refs) <= set(pack.refs)
        assert hypothesis.proposed_link is not None
        assert hypothesis.proposed_link.payment_id == "PAY-00001"
        assert hypothesis.proposed_link.bank_txn_id == "BNK-00001"

    def test_an_explanation_answer_says_two_different_things(self):
        """The contract refuses a diagnosis repeated as an instruction."""
        batch = ExplanationBatch.model_validate_json(
            _answer(
                render_explanation(
                    [("EXC-0001", "a payout that never arrived", ("evidence line",))]
                )
            )
        )
        item = batch.explanations[0]
        assert item.root_cause.strip() != item.suggested_action.strip()

    def test_an_unrecognised_prompt_shape_produces_an_empty_batch_not_a_guess(self):
        assert _answer("what is the capital of France?") == "{}"

    def test_it_records_what_it_was_asked(self):
        analyst = OfflineAnalyst()
        analyst.complete("prompt one", timeout_s=1.0)
        assert analyst.prompts == ["prompt one"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("llm-report") / "test-standard-42"
    generate_to_disk(GeneratorConfig(split=SplitName.TEST, seed=42), directory)
    return directory


@pytest.fixture(scope="module")
def artifact(corpus, tmp_path_factory):
    client = LLMClient(
        config=LLMConfig(cache_dir=tmp_path_factory.mktemp("cache")),
        provider=OfflineAnalyst(),
    )
    return run_llm_report(corpus, client=client, live=False)


class TestTheMeasuredRun:
    def test_it_ran_and_says_it_was_not_live(self, artifact):
        """The one field a reader must not have to infer."""
        assert artifact.ran
        assert artifact.live is False
        assert artifact.is_standin
        assert artifact.provider_used == OFFLINE_ANALYST_NAME

    def test_the_machinery_columns_are_measured_and_non_trivial(self, artifact):
        """Calls, tokens and latency came off the real client, not a constant."""
        assert artifact.cost.llm_calls > 0
        assert artifact.cost.total_tokens > 0
        assert artifact.calls_per_100_records > 0

    def test_the_budget_was_respected(self, artifact):
        assert artifact.cost.llm_calls <= LLMConfig().max_calls_per_run

    def test_actual_spend_is_zero_and_the_equivalent_paid_cost_is_not(self, artifact):
        assert artifact.cost.actual_cost_inr == 0.0
        assert artifact.cost.equivalent_paid_cost_inr > 0.0

    def test_the_gates_are_exercised_rather_than_bypassed(self, artifact):
        """The residual reaching T5 is what the ladder refused, so most of the
        stand-in's proposals do not reconcile -- and watching them be demoted is
        what this run exists to measure."""
        assert artifact.proposals_returned > 0
        assert artifact.demoted > 0
        assert artifact.proposals_accepted <= artifact.proposals_returned

    def test_the_control_ran_and_is_recorded_beside_the_model_run(self, artifact):
        assert artifact.without_llm.precision == 1.0
        assert artifact.with_llm.precision == 1.0
        assert artifact.with_llm.false_positives == 0
        assert artifact.without_llm.false_positives == 0

    def test_no_proposal_reached_an_auto_match_without_its_arithmetic(self, artifact):
        """The one thing the design forbids outright. Precision would not be 1.0
        if a demoted proposal had been committed."""
        assert artifact.with_llm.precision == 1.0
        assert artifact.with_llm.false_positives == 0

    def test_a_disabled_client_produces_a_did_not_run_artefact_not_zeros(self):
        client = LLMClient(config=LLMConfig(enabled=False), provider=None)
        result = run_llm_report(  # type: ignore[call-arg]
            __import__("pathlib").Path("."), client=client, live=False
        )
        assert result.ran is False
        assert result.reason
        assert result.cost.llm_calls == 0

    def test_score_of_reads_the_four_headline_figures(self, corpus):
        from ledgerloop.eval.harness import run_system

        score = score_of(run_system(corpus, measure_calibration_quality=False))
        assert score.precision == 1.0
        assert 0.0 < score.recall <= 1.0
        assert score.auto_matched > 0
