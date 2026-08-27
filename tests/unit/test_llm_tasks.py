"""The three call sites, and everything each of them refuses.

PLAN.md §7.1: *the LLM never decides a match by itself, and never does
arithmetic.* These tests are that rule stated as behaviour. For each site the
same three questions are asked:

1. Does the good path work?
2. Does a malformed, absent, over-budget or unreachable model leave the
   deterministic answer standing?
3. Does a **plausible but invented** answer get refused -- an unreferenced UTR,
   a cited record that was not in the pack, a settlement id conjured into prose,
   a proposed link whose money does not close?

The third is the one that matters. The first two fail loudly; the third would
fail silently and would look like the system working.
"""

from __future__ import annotations

import json

import pytest

from ledgerloop.config import LLMConfig, RunConfig
from ledgerloop.llm.cache import ResponseCache
from ledgerloop.llm.client import LLMClient, LLMUnavailable, ScriptedProvider
from ledgerloop.llm.prompts import (
    render_adjudication,
    render_explanation,
    render_narration,
)
from ledgerloop.llm.tasks import (
    adjudicate_residual,
    evidence_pack_for,
    explain_exceptions,
    parse_narrations,
    unmeasured_probability,
)
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.candidates import Evidence
from ledgerloop.models.enums import (
    EvidenceKind,
    ExceptionClass,
    ProseSource,
    Severity,
    Tier,
)
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from ledgerloop.money import allocate_minor
from tests.unit.conftest import bank_credit, batch, corpus

CONFIG = RunConfig(run_id="llm-tasks")
NARRATION = "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT"


def client_for(tmp_path, *responses: str, **overrides: object) -> LLMClient:
    settings = LLMConfig(cache_dir=tmp_path / "cache", **overrides)  # type: ignore[arg-type]
    return LLMClient(
        config=settings,
        provider=ScriptedProvider(responses=list(responses)),
        cache=ResponseCache(directory=settings.cache_dir),
    )


def disabled_client(tmp_path) -> LLMClient:
    return LLMClient(config=LLMConfig(enabled=False, cache_dir=tmp_path))


class TestParseNarration:
    def test_a_grounded_extraction_is_accepted(self, tmp_path):
        response = json.dumps(
            {
                "extractions": [
                    {
                        "item_id": "BNK-1",
                        "utr": "UTR2026030412345",
                        "merchant": "RAZORPAY SOFTWARE PVT",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        outcome = parse_narrations(client_for(tmp_path, response), [("BNK-1", NARRATION)])
        assert outcome.accepted == 1
        assert outcome.repairs[0].utr == "UTR2026030412345"

    def test_an_invented_reference_is_refused_and_counted(self, tmp_path):
        """A UTR is a join key: an invented one would create a match from nothing."""
        response = json.dumps(
            {
                "extractions": [
                    {"item_id": "BNK-1", "utr": "UTR2026039999999", "confidence": 0.99}
                ]
            }
        )
        outcome = parse_narrations(client_for(tmp_path, response), [("BNK-1", NARRATION)])
        assert outcome.accepted == 0
        assert outcome.rejected_ungrounded == 1
        assert outcome.repairs == ()

    def test_a_partial_hallucination_discards_the_whole_extraction(self, tmp_path):
        """Keeping the half that passed would be partial trust in an invented answer."""
        response = json.dumps(
            {
                "extractions": [
                    {
                        "item_id": "BNK-1",
                        "utr": "UTR2026030412345",
                        "merchant": "PAYTM PAYMENTS BANK",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        outcome = parse_narrations(client_for(tmp_path, response), [("BNK-1", NARRATION)])
        assert outcome.accepted == 0
        assert outcome.rejected_ungrounded == 1

    def test_an_honest_absence_is_accepted_without_a_repair(self, tmp_path):
        response = json.dumps(
            {"extractions": [{"item_id": "BNK-1", "utr": None, "confidence": 0.1}]}
        )
        outcome = parse_narrations(client_for(tmp_path, response), [("BNK-1", NARRATION)])
        assert outcome.accepted == 0
        assert outcome.rejected_ungrounded == 0
        assert outcome.repairs == ()

    def test_an_answer_about_an_item_never_sent_is_refused(self, tmp_path):
        response = json.dumps(
            {"extractions": [{"item_id": "BNK-9", "utr": "UTR1", "confidence": 0.9}]}
        )
        outcome = parse_narrations(client_for(tmp_path, response), [("BNK-1", NARRATION)])
        assert outcome.rejected_ungrounded == 1

    def test_an_outage_leaves_the_deterministic_answer_standing(self, tmp_path):
        client = client_for(tmp_path)
        client.provider = ScriptedProvider(failure=LLMUnavailable("429"))
        outcome = parse_narrations(client, [("BNK-1", NARRATION)])
        assert outcome.repairs == ()
        assert outcome.calls_refused == 1
        assert outcome.failures

    def test_a_disabled_client_does_nothing_at_all(self, tmp_path):
        outcome = parse_narrations(disabled_client(tmp_path), [("BNK-1", NARRATION)])
        assert outcome.attempted == 0
        assert outcome.repairs == ()

    def test_nothing_to_read_makes_no_call(self, tmp_path):
        client = client_for(tmp_path)
        assert parse_narrations(client, []).attempted == 0
        assert client.calls == 0

    def test_items_are_batched(self, tmp_path):
        responses = [json.dumps({"extractions": []})] * 3
        client = client_for(tmp_path, *responses)
        parse_narrations(
            client, [(f"BNK-{i}", f"{NARRATION} {i}") for i in range(5)], batch_size=2
        )
        assert client.calls == 3

    def test_the_prompt_shows_the_text_and_never_an_amount(self):
        prompt = render_narration([("BNK-1", NARRATION)])
        assert NARRATION in prompt
        assert "Do not calculate anything" in prompt
        assert "₹" not in prompt


class TestAdjudicateResidual:
    @pytest.fixture
    def solved(self):
        only = batch(amounts=(60_000, 40_000, 25_000), fee_minor=2_500)
        grosses = [payment.amount_minor for payment in only.payments]
        amounts = allocate_minor(only.net_minor, [grosses[0], grosses[1] + grosses[2]])
        credits = [
            bank_credit("BNK-00001", amount_minor=amounts[0], utr=None),
            bank_credit("BNK-00002", amount_minor=amounts[1], utr=None),
        ]
        return only, MatchContext.from_ingest(corpus(batches=[only], bank_txns=credits))

    def pack(self, only):
        return evidence_pack_for(
            "SETL-0001",
            summary="payout not credited",
            refs=[
                settlement_ref("SETL-0001").key,
                *(payment_ref(p.payment_id).key for p in only.payments),
                bank_ref("BNK-00001").key,
            ],
        )

    def hypothesis(self, **overrides: object) -> str:
        body = {
            "item_id": "SETL-0001",
            "hypothesis": "the first tranche is BNK-00001",
            "proposed_link": {
                "payment_id": "PAY-00001",
                "bank_txn_id": "BNK-00001",
                "settlement_id": "SETL-0001",
                "payment_ids": ["PAY-00001"],
            },
            "confidence": 0.8,
            "reasoning": "amounts line up",
            "evidence_refs": ["settlement:SETL-0001", "bank_txn:BNK-00001"],
        }
        body.update(overrides)
        return json.dumps({"hypotheses": [body]})

    def test_a_proposal_that_reconciles_becomes_a_verified_candidate(self, tmp_path, solved):
        only, context = solved
        outcome = adjudicate_residual(
            client_for(tmp_path, self.hypothesis()), context, [self.pack(only)], CONFIG
        )
        assert outcome.accepted == 1
        assert len(outcome.candidates) == 1
        candidate = outcome.candidates[0]
        assert candidate.tier is Tier.T5_LLM
        assert candidate.arithmetic_verified

    def test_the_model_confidence_lands_in_the_features_not_the_probability(
        self, tmp_path, solved
    ):
        """Raw self-reported confidence is overconfident. It is a feature."""
        only, context = solved
        outcome = adjudicate_residual(
            client_for(tmp_path, self.hypothesis()), context, [self.pack(only)], CONFIG
        )
        candidate = outcome.candidates[0]
        assert candidate.features.llm_confidence == pytest.approx(0.8)
        # Not the confidence, and not high enough to auto-match: an unmeasured
        # probability sits in the review band until a calibrator prices it.
        assert candidate.calibrated_p == unmeasured_probability(CONFIG.thresholds)
        assert candidate.calibrated_p < CONFIG.thresholds.tau_high

    def test_a_proposal_whose_money_does_not_close_is_demoted_not_dropped(
        self, tmp_path, solved
    ):
        """PLAN.md §7.4: the disagreement is information a controller wants."""
        only, context = solved
        wrong = self.hypothesis(
            proposed_link={
                "payment_id": "PAY-00002",
                "bank_txn_id": "BNK-00001",
                "settlement_id": "SETL-0001",
                "payment_ids": ["PAY-00002"],
            }
        )
        outcome = adjudicate_residual(
            client_for(tmp_path, wrong), context, [self.pack(only)], CONFIG
        )
        assert outcome.accepted == 0
        assert outcome.demoted == 1
        assert len(outcome.candidates) == 1
        candidate = outcome.candidates[0]
        assert not candidate.arithmetic_verified
        kinds = {item.kind for item in candidate.evidence}
        assert EvidenceKind.NEGATIVE_EVIDENCE in kinds

    def test_a_hypothesis_citing_a_record_it_was_not_given_is_discarded_whole(
        self, tmp_path, solved
    ):
        only, context = solved
        outcome = adjudicate_residual(
            client_for(tmp_path, self.hypothesis(evidence_refs=["settlement:SETL-9999"])),
            context,
            [self.pack(only)],
            CONFIG,
        )
        assert outcome.rejected_ungrounded == 1
        assert outcome.candidates == ()
        assert outcome.hypotheses == ()

    def test_a_proposal_naming_records_that_do_not_exist_yields_no_candidate(
        self, tmp_path, solved
    ):
        only, context = solved
        outcome = adjudicate_residual(
            client_for(
                tmp_path,
                self.hypothesis(
                    proposed_link={
                        "payment_id": "PAY-99999",
                        "bank_txn_id": "BNK-00001",
                        "settlement_id": None,
                        "payment_ids": ["PAY-99999"],
                    }
                ),
            ),
            context,
            [self.pack(only)],
            CONFIG,
        )
        assert outcome.candidates == ()
        assert outcome.rejected_unverified == 1

    def test_a_hypothesis_with_no_link_is_kept_as_reasoning_only(self, tmp_path, solved):
        only, context = solved
        outcome = adjudicate_residual(
            client_for(tmp_path, self.hypothesis(proposed_link=None)),
            context,
            [self.pack(only)],
            CONFIG,
        )
        assert outcome.candidates == ()
        assert len(outcome.hypotheses) == 1

    def test_an_answer_about_an_item_never_sent_is_refused(self, tmp_path, solved):
        only, context = solved
        outcome = adjudicate_residual(
            client_for(tmp_path, self.hypothesis(item_id="SETL-9999")),
            context,
            [self.pack(only)],
            CONFIG,
        )
        assert outcome.rejected_ungrounded == 1

    def test_a_half_specified_link_never_validates(self, tmp_path, solved):
        """The schema refuses it; the retry refuses it; nothing reaches the gate."""
        only, context = solved
        broken = self.hypothesis(
            proposed_link={"payment_id": "", "bank_txn_id": "BNK-00001"}
        )
        outcome = adjudicate_residual(
            client_for(tmp_path, broken, broken), context, [self.pack(only)], CONFIG
        )
        assert outcome.candidates == ()
        assert outcome.calls_refused == 1

    def test_an_outage_leaves_the_ladder_where_it_was(self, tmp_path, solved):
        only, context = solved
        client = client_for(tmp_path)
        client.provider = ScriptedProvider(failure=LLMUnavailable("timeout"))
        outcome = adjudicate_residual(client, context, [self.pack(only)], CONFIG)
        assert outcome.candidates == ()
        assert outcome.calls_refused == 1

    def test_a_disabled_client_proposes_nothing(self, tmp_path, solved):
        only, context = solved
        outcome = adjudicate_residual(
            disabled_client(tmp_path), context, [self.pack(only)], CONFIG
        )
        assert outcome.attempted == 0

    def test_the_pack_is_rendered_with_sorted_refs_so_the_cache_key_is_stable(self):
        first = evidence_pack_for("X", summary="s", refs=["b", "a"])
        second = evidence_pack_for("X", summary="s", refs=["a", "b", "a"])
        assert first.refs == second.refs == ("a", "b")
        assert render_adjudication([first]) == render_adjudication([second])

    def test_the_prompt_states_that_the_arithmetic_will_be_rechecked(self):
        prompt = render_adjudication([evidence_pack_for("X", summary="s", refs=["a"])])
        assert "re-derives the money" in prompt
        assert "Do not calculate amounts" in prompt


class TestExplainExceptions:
    def exception(self, **overrides: object) -> ReconException:
        base = {
            "exception_id": "exception:settlement:SETL-0001",
            "exception_class": ExceptionClass.LATE_ARRIVAL,
            "severity": Severity.HIGH,
            "impact_minor": 500_000,
            "involved_refs": (settlement_ref("SETL-0001"), payment_ref("PAY-00001")),
            "evidence": (
                Evidence(kind=EvidenceKind.AMOUNT_MATCH, detail="declared ₹5,000.00"),
            ),
            "root_cause": "template cause.",
            "suggested_action": "template action.",
            "classification_confidence": 0.9,
        }
        base.update(overrides)
        return ReconException(**base)  # type: ignore[arg-type]

    def response(self, **overrides: object) -> str:
        body = {
            "exception_id": "exception:settlement:SETL-0001",
            "root_cause": "SETL-0001 has not been credited in this period.",
            "suggested_action": "Check the next statement for SETL-0001.",
        }
        body.update(overrides)
        return json.dumps({"explanations": [body]})

    def test_accepted_prose_replaces_the_template_and_says_who_wrote_it(self, tmp_path):
        outcome = explain_exceptions(
            client_for(tmp_path, self.response()), [self.exception()]
        )
        assert outcome.rewritten == 1
        rewritten = outcome.exceptions[0]
        assert rewritten.root_cause_source is ProseSource.LLM
        assert rewritten.suggested_action_source is ProseSource.LLM
        assert "not been credited" in rewritten.root_cause

    def test_the_class_severity_and_money_are_never_touched(self, tmp_path):
        original = self.exception()
        outcome = explain_exceptions(
            client_for(tmp_path, self.response()), [original]
        )
        rewritten = outcome.exceptions[0]
        assert rewritten.exception_class is original.exception_class
        assert rewritten.severity is original.severity
        assert rewritten.impact_minor == original.impact_minor
        assert rewritten.involved_refs == original.involved_refs

    def test_prose_inventing_a_record_is_refused_and_the_template_stands(self, tmp_path):
        outcome = explain_exceptions(
            client_for(
                tmp_path,
                self.response(root_cause="SETL-0001 was netted against SETL-0099."),
            ),
            [self.exception()],
        )
        assert outcome.rewritten == 0
        assert outcome.rejected_ungrounded == 1
        assert outcome.exceptions[0].root_cause == "template cause."
        assert outcome.exceptions[0].root_cause_source is ProseSource.TEMPLATE

    def test_an_answer_about_an_exception_never_sent_is_refused(self, tmp_path):
        outcome = explain_exceptions(
            client_for(tmp_path, self.response(exception_id="exception:other")),
            [self.exception()],
        )
        assert outcome.rejected_ungrounded == 1
        assert outcome.rewritten == 0

    def test_a_cause_repeated_as_an_action_never_validates(self, tmp_path):
        same = self.response(root_cause="Same text.", suggested_action="Same text.")
        outcome = explain_exceptions(client_for(tmp_path, same, same), [self.exception()])
        assert outcome.rewritten == 0
        assert outcome.calls_refused == 1

    def test_an_outage_leaves_every_row_with_its_template(self, tmp_path):
        client = client_for(tmp_path)
        client.provider = ScriptedProvider(failure=LLMUnavailable("500"))
        outcome = explain_exceptions(client, [self.exception()])
        assert outcome.exceptions[0].root_cause == "template cause."
        assert outcome.calls_refused == 1

    def test_a_disabled_client_leaves_the_queue_exactly_as_it_was(self, tmp_path):
        original = [self.exception()]
        outcome = explain_exceptions(disabled_client(tmp_path), original)
        assert outcome.exceptions == tuple(original)
        assert outcome.attempted == 0

    def test_an_empty_queue_makes_no_call(self, tmp_path):
        client = client_for(tmp_path)
        assert explain_exceptions(client, []).exceptions == ()
        assert client.calls == 0

    def test_exceptions_are_batched_by_class(self, tmp_path):
        """PLAN.md §7.3: one call per cluster, not one per exception."""
        client = client_for(
            tmp_path, json.dumps({"explanations": []}), json.dumps({"explanations": []})
        )
        explain_exceptions(
            client,
            [
                self.exception(),
                self.exception(
                    exception_id="exception:bank_txn:BNK-1",
                    exception_class=ExceptionClass.ORPHAN_BANK_CREDIT,
                    involved_refs=(bank_ref("BNK-1"),),
                ),
            ],
        )
        assert client.calls == 2

    def test_the_prompt_gives_the_class_and_money_as_settled_facts(self):
        prompt = render_explanation([("exception:1", "E_LATE_ARRIVAL, HIGH", ["fact"])])
        assert "already decided and are not yours to change" in prompt
        assert "Do not speculate about intent" in prompt
