"""The failover ladder, its retry policy, and what a keyless machine gets.

Every test here uses a fake rung and an injected sleep. Nothing opens a socket
and nothing waits: the assertions are about *how many* attempts happened, *in
what order*, and *how long the policy asked to wait*, and a real sleep would
make each of those slow to check and none of them easier to trust.
"""

from __future__ import annotations

import pytest

from ledgerloop.config import LLMConfig
from ledgerloop.llm.client import (
    Completion,
    LLMClient,
    LLMUnavailable,
    RateLimited,
)
from ledgerloop.llm.providers import (
    DEFAULT_LADDER,
    FailoverProvider,
    build_ladder,
    configured_rungs,
)
from ledgerloop.models.base import FrozenLedgerModel


class _Rung:
    """A provider that answers, or raises whatever it was given, per attempt."""

    def __init__(self, name: str, *outcomes: object) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        del prompt, timeout_s
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else LLMUnavailable("exhausted")
        if isinstance(outcome, Exception):
            raise outcome
        return Completion(text=str(outcome), provider=self.name, prompt_tokens=10)


class _Clock:
    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _ladder(*rungs: _Rung, **kwargs: object) -> FailoverProvider:
    clock = _Clock()
    provider = FailoverProvider(rungs=tuple(rungs), sleep=clock, **kwargs)  # type: ignore[arg-type]
    provider.waits = clock.waits  # type: ignore[attr-defined]
    return provider


class TestWalkingTheLadder:
    def test_the_first_rung_that_answers_is_the_one_used(self):
        ladder = _ladder(_Rung("groq", "ok"), _Rung("gemini", "never"))
        completion = ladder.complete("p", timeout_s=1.0)

        assert completion.provider == "groq"
        assert ladder.fallback_depth == 0
        assert ladder.rungs[1].calls == 0  # type: ignore[attr-defined]

    def test_an_outage_moves_down_a_rung_and_records_how_far(self):
        ladder = _ladder(
            _Rung("groq", LLMUnavailable("down")),
            _Rung("gemini", LLMUnavailable("down")),
            _Rung("openrouter", "ok"),
        )
        completion = ladder.complete("p", timeout_s=1.0)

        assert completion.provider == "openrouter"
        assert ladder.fallback_depth == 2
        assert ladder.max_fallback_depth == 2

    def test_max_fallback_depth_remembers_the_worst_call_not_the_last(self):
        """The report prints one number for a run of many calls, and the number
        a reader wants is how bad it got."""
        ladder = _ladder(_Rung("groq", LLMUnavailable("down"), "ok"), _Rung("gemini", "ok"))
        ladder.complete("first", timeout_s=1.0)  # falls to gemini
        ladder.complete("second", timeout_s=1.0)  # groq answers
        assert ladder.fallback_depth == 0
        assert ladder.max_fallback_depth == 1

    def test_exhausting_the_ladder_raises_naming_every_rung(self):
        ladder = _ladder(
            _Rung("groq", LLMUnavailable("groq is down")),
            _Rung("ollama", LLMUnavailable("nothing listening")),
        )
        with pytest.raises(LLMUnavailable) as raised:
            ladder.complete("p", timeout_s=1.0)
        assert "groq" in str(raised.value)
        assert "ollama" in str(raised.value)

    def test_the_trail_records_each_rung_that_declined(self):
        ladder = _ladder(_Rung("groq", LLMUnavailable("down")), _Rung("gemini", "ok"))
        ladder.complete("p", timeout_s=1.0)

        declined = [rung for rung in ladder.trail if not rung.answered]
        assert [rung.provider for rung in declined] == ["groq"]
        assert ladder.trail[-1].provider == "gemini"
        assert ladder.trail[-1].answered

    def test_an_empty_ladder_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="at least one provider"):
            FailoverProvider(rungs=())


class TestTheRetryPolicy:
    def test_a_rate_limit_is_retried_in_place_before_moving_down(self):
        first = _Rung("groq", RateLimited("429"), "ok")
        ladder = _ladder(first, _Rung("gemini", "never"))
        completion = ladder.complete("p", timeout_s=1.0)

        assert completion.provider == "groq"
        assert first.calls == 2
        assert ladder.rungs[1].calls == 0  # type: ignore[attr-defined]

    def test_an_outage_is_not_retried_because_waiting_will_not_fix_it(self):
        """A misconfigured endpoint and an exhausted quota wear the same shape
        at the transport; only one of them is worth half a second."""
        first = _Rung("groq", LLMUnavailable("no route to host"), "ok")
        ladder = _ladder(first, _Rung("gemini", "ok"))
        ladder.complete("p", timeout_s=1.0)
        assert first.calls == 1

    def test_the_providers_own_retry_after_is_honoured_when_it_sends_one(self):
        ladder = _ladder(_Rung("groq", RateLimited("429", retry_after_s=2.0), "ok"))
        ladder.complete("p", timeout_s=1.0)
        assert ladder.waits == [2.0]  # type: ignore[attr-defined]

    def test_a_retry_after_of_an_hour_is_capped_rather_than_obeyed(self):
        """True and unusable inside one run. Obeying it literally would hang a
        demo on a header."""
        ladder = _ladder(
            _Rung("groq", RateLimited("429", retry_after_s=3600.0), "ok"),
            max_backoff_s=5.0,
        )
        ladder.complete("p", timeout_s=1.0)
        assert ladder.waits == [5.0]  # type: ignore[attr-defined]

    def test_without_a_header_the_backoff_is_the_ladders_own_and_bounded(self):
        ladder = _ladder(
            _Rung("groq", RateLimited("429"), RateLimited("429"), "ok"),
            retries_per_rung=2,
            backoff_s=0.5,
            max_backoff_s=5.0,
        )
        ladder.complete("p", timeout_s=1.0)
        assert ladder.waits == [0.5, 1.0]  # type: ignore[attr-defined]

    def test_retries_are_exhausted_before_the_next_rung_is_tried(self):
        first = _Rung("groq", RateLimited("429"), RateLimited("429"))
        second = _Rung("gemini", "ok")
        ladder = _ladder(first, second, retries_per_rung=1)
        ladder.complete("p", timeout_s=1.0)
        assert first.calls == 2
        assert second.calls == 1

    def test_a_negative_retry_count_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="retries_per_rung"):
            FailoverProvider(rungs=(_Rung("groq"),), retries_per_rung=-1)


class TestWhichRungsAMachineHas:
    def test_a_machine_with_no_credentials_gets_no_ladder_at_all(self):
        """Not an error, and not a localhost timeout either -- `None` is the
        deterministic path, exactly as it was before the ladder existed."""
        assert build_ladder(LLMConfig(), environ={}) is None

    def test_a_per_provider_key_enables_exactly_that_rung(self):
        ladder = build_ladder(LLMConfig(), environ={"OPENROUTER_API_KEY": "k"})
        assert ladder is not None
        assert ladder.ladder == ("openrouter",)

    def test_the_shared_step_9_variable_still_enables_every_keyed_rung(self):
        ladder = build_ladder(LLMConfig(), environ={"LEDGERLOOP_LLM_API_KEY": "k"})
        assert ladder is not None
        assert ladder.ladder == DEFAULT_LADDER

    def test_a_per_provider_key_beats_the_shared_one(self):
        rungs = configured_rungs(
            LLMConfig(),
            environ={"LEDGERLOOP_LLM_API_KEY": "shared", "GROQ_API_KEY": "mine"},
        )
        assert next(rung.api_key for rung in rungs if rung.name == "groq") == "mine"

    def test_ollama_needs_no_key_and_is_therefore_only_used_when_asked_for(self):
        """The one rung that would otherwise make a keyless machine wait for a
        connection to localhost before reaching the deterministic path."""
        assert build_ladder(LLMConfig(), environ={}) is None
        asked = build_ladder(LLMConfig(), environ={"LEDGERLOOP_LLM_PROVIDERS": "ollama"})
        assert asked is not None and asked.ladder == ("ollama",)

    def test_pointing_ollama_somewhere_counts_as_asking_for_it(self):
        ladder = build_ladder(LLMConfig(), environ={"OLLAMA_BASE_URL": "http://h:1/v1"})
        assert ladder is not None
        assert ladder.ladder == ("ollama",)

    def test_the_environment_can_reorder_the_ladder(self):
        ladder = build_ladder(
            LLMConfig(),
            environ={
                "LEDGERLOOP_LLM_PROVIDERS": "openrouter,groq",
                "LEDGERLOOP_LLM_API_KEY": "k",
            },
        )
        assert ladder is not None
        assert ladder.ladder == ("openrouter", "groq")

    def test_an_unknown_provider_name_is_dropped_and_never_raises(self):
        """A typo in an environment variable should leave the run deterministic,
        not stop it."""
        rungs = configured_rungs(
            LLMConfig(), environ={"LEDGERLOOP_LLM_API_KEY": "k"}, order=("grok", "groq")
        )
        assert [rung.name for rung in rungs] == ["groq"]

    def test_each_rung_is_asked_for_a_model_it_actually_serves(self):
        """Sending Groq's model id to OpenRouter fails for a reason that has
        nothing to do with quota, on every rung below the first."""
        rungs = configured_rungs(LLMConfig(), environ={"LEDGERLOOP_LLM_API_KEY": "k"})
        models = {rung.name: rung.model for rung in rungs}
        assert models["groq"] == LLMConfig().model
        assert models["openrouter"] != models["groq"]

    def test_no_ladder_is_built_for_a_run_that_asked_for_no_model(self):
        assert build_ladder(LLMConfig(enabled=False), environ={"GROQ_API_KEY": "k"}) is None


class _Echo(FrozenLedgerModel):
    text: str


class TestTheClientAboveTheLadder:
    """Nothing above the provider protocol knows a ladder is there."""

    def test_the_cache_budget_and_validation_are_unchanged_by_failover(self, tmp_path):
        ladder = _ladder(
            _Rung("groq", LLMUnavailable("down")),
            _Rung("gemini", '{"text": "hello"}'),
        )
        client = LLMClient(
            config=LLMConfig(cache_dir=tmp_path / "cache"), provider=ladder
        )
        answer, _digest = client.complete_json(
            "prompt", _Echo, prompt_version="test/1.0.0"
        )
        assert answer.text == "hello"
        assert client.calls == 1

        again, _ = client.complete_json("prompt", _Echo, prompt_version="test/1.0.0")
        assert again.text == "hello"
        assert client.calls == 1
        assert client.cache_hits == 1

    def test_the_ledger_reports_how_far_down_the_ladder_the_run_went(self, tmp_path):
        ladder = _ladder(
            _Rung("groq", LLMUnavailable("down")),
            _Rung("gemini", '{"text": "hi"}'),
        )
        client = LLMClient(
            config=LLMConfig(cache_dir=tmp_path / "cache"), provider=ladder
        )
        client.complete_json("prompt", _Echo, prompt_version="test/1.0.0")
        ledger = client.ledger()

        assert ledger.fallback_depth == 1
        assert ledger.provider_used == "gemini"
        assert client.provider_failure_detail  # the rung that declined is named

    def test_actual_spend_is_computed_from_the_rung_not_asserted_to_be_zero(
        self, tmp_path
    ):
        """Every rung this project is configured for is free, so the arithmetic
        produces zero -- but it is arithmetic, and a paid rung would show."""
        ladder = _ladder(_Rung("groq", '{"text": "hi"}'))
        client = LLMClient(
            config=LLMConfig(cache_dir=tmp_path / "cache"), provider=ladder
        )
        client.complete_json("prompt", _Echo, prompt_version="test/1.0.0")
        ledger = client.ledger()

        assert client.price_inr_per_mtok == (0.0, 0.0)
        assert ledger.actual_cost_inr == 0.0
        assert ledger.equivalent_paid_cost_inr > 0.0
