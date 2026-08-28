"""Property tests for the LLM layer (PLAN.md §13).

The unit tests check the failure modes somebody thought of. These check the
statements that must hold for **every** answer a model could return -- which is
the right shape for this component, because the input is adversarial by nature:
a model can return any schema-valid string, and "we tested the ones we imagined"
is not a safety argument.

Four invariants, each one a sentence from PLAN.md §7 made checkable:

* *the LLM never decides a match by itself* -- no answer produces an auto-match
  without ``verify_arithmetic`` passing first;
* *the LLM never does arithmetic* -- no answer changes a rupee figure;
* every value it claims to have read is in the text it was shown;
* every reference it cites was in the pack it was sent.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ledgerloop.config import DecisionThresholds, LLMConfig, RunConfig
from ledgerloop.llm.cache import CacheKey, ResponseCache
from ledgerloop.llm.client import LLMClient, ScriptedProvider
from ledgerloop.llm.contracts import NarrationBatch
from ledgerloop.llm.gates import (
    grounded_in_text,
    grounded_refs,
    prose_names_only_known_records,
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
from ledgerloop.models.enums import EvidenceKind, ExceptionClass, ProseSource, Severity
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.models.refs import bank_ref, payment_ref, settlement_ref
from tests.unit.conftest import bank_credit, batch, corpus

CONFIG = RunConfig(run_id="llm-props")
NARRATION = "NEFT CR-RAZORPAY SOFTWARE PVT-UTR2026030412345-SETTLEMENT"

#: ``tmp_path_factory`` is function-scoped by Hypothesis's reckoning; these
#: tests want a fresh cache directory per example, which is exactly what it is
#: for. The health check is suppressed rather than the directory shared,
#: because a shared cache would make the second example a cache hit and the
#: property would stop testing the path it claims to.
SLOW_FIXTURE = [HealthCheck.function_scoped_fixture]

text = st.text(min_size=1, max_size=24)
identifiers = st.sampled_from(
    ["PAY-00001", "PAY-00002", "BNK-00001", "BNK-00002", "SETL-0001", "SETL-9999"]
)
probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def client_for(tmp_path, *responses: str) -> LLMClient:
    settings_ = LLMConfig(cache_dir=tmp_path / "cache")
    return LLMClient(
        config=settings_,
        provider=ScriptedProvider(responses=list(responses)),
        cache=ResponseCache(directory=settings_.cache_dir),
    )


def context_and_pack():
    only = batch(amounts=(60_000, 40_000), utr=None)
    context = MatchContext.from_ingest(
        corpus(
            batches=[only],
            bank_txns=[bank_credit("BNK-00001", amount_minor=only.net_minor, utr=None)],
        )
    )
    pack = evidence_pack_for(
        "SETL-0001",
        summary="not credited",
        refs=[
            settlement_ref("SETL-0001").key,
            payment_ref("PAY-00001").key,
            payment_ref("PAY-00002").key,
            bank_ref("BNK-00001").key,
        ],
    )
    return context, pack


class TestTheGroundingGates:
    @given(text, text)
    def test_a_value_passes_exactly_when_it_occurs_in_the_source(self, value, source):
        expected = value.strip().lower().replace(" ", "") != "" and _folds_into(
            value, source
        )
        assert bool(grounded_in_text(value, source)) is expected or not value.strip()

    @given(st.lists(text, max_size=6), st.lists(text, max_size=6))
    def test_references_pass_exactly_when_every_one_was_supplied(self, claimed, supplied):
        gate = grounded_refs(claimed, supplied)
        assert bool(gate) is set(claimed).issubset(set(supplied))

    @given(st.lists(text, max_size=6), st.lists(text, max_size=6))
    def test_a_refusal_always_names_what_was_wrong(self, claimed, supplied):
        gate = grounded_refs(claimed, supplied)
        if not gate:
            assert gate.offending
            assert gate.reason

    @given(st.lists(identifiers, max_size=4), st.lists(identifiers, max_size=3))
    def test_prose_passes_exactly_when_it_names_only_involved_records(
        self, mentioned, involved
    ):
        refs = tuple(_ref(name) for name in involved)
        prose = "The payout " + " and ".join(mentioned) + " is short."
        gate = prose_names_only_known_records(prose, refs)
        assert bool(gate) is set(mentioned).issubset(set(involved))


class TestNoAnswerCanDecideAMatch:
    @given(
        payment=st.sampled_from(["PAY-00001", "PAY-00002", "PAY-99999"]),
        credit=st.sampled_from(["BNK-00001", "BNK-99999"]),
        confidence=probabilities,
        refs=st.lists(
            st.sampled_from(
                ["settlement:SETL-0001", "payment:PAY-00001", "settlement:SETL-9999"]
            ),
            max_size=3,
        ),
    )
    @settings(max_examples=120, deadline=None, suppress_health_check=SLOW_FIXTURE)
    def test_a_candidate_is_verified_only_when_the_money_closes(
        self, tmp_path_factory, payment, credit, confidence, refs
    ):
        """Whatever the model says, ``verify_arithmetic`` has the last word."""
        context, pack = context_and_pack()
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "item_id": "SETL-0001",
                        "hypothesis": "h",
                        "proposed_link": {
                            "payment_id": payment,
                            "bank_txn_id": credit,
                            "settlement_id": None,
                            "payment_ids": [payment],
                        },
                        "confidence": confidence,
                        "reasoning": "r",
                        "evidence_refs": refs,
                    }
                ]
            }
        )
        outcome = adjudicate_residual(
            client_for(tmp_path_factory.mktemp("c"), response), context, [pack], CONFIG
        )
        for candidate in outcome.candidates:
            if candidate.arithmetic_verified:
                # The only pairing whose money actually closes on this corpus.
                assert candidate.source_ref.record_id in {"PAY-00001", "PAY-00002"}
                assert candidate.target_ref.record_id == "BNK-00001"

    @given(confidence=probabilities)
    @settings(max_examples=40, deadline=None, suppress_health_check=SLOW_FIXTURE)
    def test_the_model_confidence_never_becomes_the_probability(
        self, tmp_path_factory, confidence
    ):
        """A 0.99 that set its own ``calibrated_p`` would auto-match itself."""
        context, pack = context_and_pack()
        response = json.dumps(
            {
                "hypotheses": [
                    {
                        "item_id": "SETL-0001",
                        "hypothesis": "h",
                        "proposed_link": {
                            "payment_id": "PAY-00001",
                            "bank_txn_id": "BNK-00001",
                            "settlement_id": None,
                            "payment_ids": ["PAY-00001", "PAY-00002"],
                        },
                        "confidence": confidence,
                        "reasoning": "r",
                        "evidence_refs": [],
                    }
                ]
            }
        )
        outcome = adjudicate_residual(
            client_for(tmp_path_factory.mktemp("c"), response), context, [pack], CONFIG
        )
        unmeasured = unmeasured_probability(CONFIG.thresholds)
        for candidate in outcome.candidates:
            assert candidate.features.llm_confidence == confidence
            assert candidate.calibrated_p == unmeasured
            assert candidate.calibrated_p < CONFIG.thresholds.tau_high

    @given(
        low=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        high=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_an_unmeasured_probability_can_never_reach_tau_high(self, low, high):
        """The invariant that keeps a T5 proposal from auto-matching itself.

        Two collapses are allowed and both go the safe way. A degenerate band
        (`tau_low == tau_high`) returns `tau_low`, routing to an exception; so
        does a band too narrow to hold a float, which Hypothesis found at
        `[0.0, 5e-324]`. What is never allowed is a value at or above
        `tau_high`, and that is asserted unconditionally.
        """
        if low > high:
            low, high = high, low
        thresholds = DecisionThresholds(tau_low=low, tau_high=high)
        value = unmeasured_probability(thresholds)
        assert value <= thresholds.tau_high
        if low < value:
            # A real band. The value sits strictly inside it, so the policy
            # routes it to review rather than to a ledger.
            assert value < high
        else:
            assert value == thresholds.tau_low


class TestNoAnswerCanChangeAmountOrClass:
    @given(
        cause=st.text(min_size=1, max_size=60),
        action=st.text(min_size=1, max_size=60),
    )
    @settings(max_examples=60, deadline=None, suppress_health_check=SLOW_FIXTURE)
    def test_prose_never_moves_the_class_the_severity_or_the_money(
        self, tmp_path_factory, cause, action
    ):
        original = ReconException(
            exception_id="exception:settlement:SETL-0001",
            exception_class=ExceptionClass.LATE_ARRIVAL,
            severity=Severity.HIGH,
            impact_minor=500_000,
            involved_refs=(settlement_ref("SETL-0001"),),
            evidence=(Evidence(kind=EvidenceKind.AMOUNT_MATCH, detail="d"),),
            root_cause="template cause.",
            suggested_action="template action.",
            classification_confidence=0.9,
        )
        response = json.dumps(
            {
                "explanations": [
                    {
                        "exception_id": original.exception_id,
                        "root_cause": cause,
                        "suggested_action": action,
                    }
                ]
            }
        )
        outcome = explain_exceptions(
            client_for(tmp_path_factory.mktemp("c"), response, response), [original]
        )
        after = outcome.exceptions[0]
        assert after.exception_class is original.exception_class
        assert after.severity is original.severity
        assert after.impact_minor == original.impact_minor
        assert after.involved_refs == original.involved_refs
        assert after.exception_id == original.exception_id
        if after.root_cause != original.root_cause:
            assert after.root_cause_source is ProseSource.LLM

    @given(utr=st.text(min_size=1, max_size=20))
    @settings(max_examples=60, deadline=None, suppress_health_check=SLOW_FIXTURE)
    def test_a_repair_is_accepted_only_when_it_is_in_the_narration(
        self, tmp_path_factory, utr
    ):
        response = json.dumps(
            {"extractions": [{"item_id": "BNK-1", "utr": utr, "confidence": 0.9}]}
        )
        outcome = parse_narrations(
            client_for(tmp_path_factory.mktemp("c"), response), [("BNK-1", NARRATION)]
        )
        if outcome.repairs:
            assert _folds_into(utr, NARRATION)


class TestTheCache:
    @given(text, text, text)
    def test_the_digest_is_stable_and_hex(self, provider, model, prompt):
        key = CacheKey(provider, model, 0.0, "v", prompt)
        assert key.digest == CacheKey(provider, model, 0.0, "v", prompt).digest
        assert len(key.digest) == 64
        assert key.filename.endswith(".json")

    @given(text, text)
    def test_a_round_trip_returns_what_was_stored(self, tmp_path_factory, prompt, answer):
        cache = ResponseCache(directory=tmp_path_factory.mktemp("cache"))
        key = CacheKey("p", "m", 0.0, "v", prompt)
        cache.put(key, answer)
        assert cache.get(key) == answer

    @given(st.lists(text, min_size=1, max_size=6))
    @settings(max_examples=40, deadline=None, suppress_health_check=SLOW_FIXTURE)
    def test_a_second_run_over_the_same_prompts_makes_no_live_call(
        self, tmp_path_factory, prompts
    ):
        directory = tmp_path_factory.mktemp("cache")
        answers = ['{"extractions": []}'] * len(prompts)
        settings_ = LLMConfig(cache_dir=directory, max_calls_per_run=99)
        first = LLMClient(
            config=settings_,
            provider=ScriptedProvider(responses=list(answers)),
            cache=ResponseCache(directory=directory),
        )
        for prompt in prompts:
            first.complete_json(prompt, NarrationBatch, prompt_version="v")

        second = LLMClient(
            config=settings_,
            provider=ScriptedProvider(responses=[]),
            cache=ResponseCache(directory=directory),
        )
        for prompt in prompts:
            second.complete_json(prompt, NarrationBatch, prompt_version="v")
        assert second.calls == 0
        assert second.ledger().cache_hit_rate == 1.0


def _folds_into(value: str, source: str) -> bool:
    from ledgerloop.ingest.normalize import fold_text

    folded = fold_text(value)
    return bool(folded) and folded in fold_text(source)


def _ref(name: str):
    if name.startswith("PAY"):
        return payment_ref(name)
    if name.startswith("BNK"):
        return bank_ref(name)
    return settlement_ref(name)
