"""The one door to a model: cache, budget, retry, validation, ledger.

Everything a call site is not allowed to think about lives in the client, so
this is where the failure modes a network introduces are pinned. The rule being
tested throughout is PLAN.md §7.4's, generalised: **never a crash, never a
silent default.** A timeout, a 429, a 500, a body that is not JSON, a body that
is JSON but not the schema -- each raises a typed error that a call site catches
and falls back from, and none of them returns something that looks like an
answer.

The provider is scripted rather than mocked. It implements the same protocol and
goes through the same cache, budget, retry and validation path, so these tests
exercise every line the real provider would except the socket itself.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from ledgerloop.config import LLMConfig
from ledgerloop.llm.cache import CacheKey, ResponseCache
from ledgerloop.llm.client import (
    COMPLETION_INR_PER_MTOK,
    PROMPT_INR_PER_MTOK,
    BudgetExhausted,
    LLMClient,
    LLMDisabled,
    LLMUnavailable,
    LLMValidationError,
    OpenAICompatibleProvider,
    ScriptedProvider,
    batched,
    build_provider,
)
from ledgerloop.llm.contracts import NarrationBatch

VERSION = "test/1.0.0"


def answer(**extractions: object) -> str:
    return json.dumps({"extractions": [extractions]}) if extractions else '{"extractions": []}'


def client_for(
    tmp_path, *, responses: list[str] | None = None, **config: object
) -> tuple[LLMClient, ScriptedProvider]:
    provider = ScriptedProvider(responses=list(responses or []))
    settings = LLMConfig(cache_dir=tmp_path / "cache", **config)  # type: ignore[arg-type]
    return (
        LLMClient(
            config=settings,
            provider=provider,
            cache=ResponseCache(directory=settings.cache_dir),
        ),
        provider,
    )


class TestTheCache:
    def test_the_key_covers_everything_that_could_change_an_answer(self):
        base = CacheKey("groq", "m", 0.0, VERSION, "prompt")
        for changed in (
            CacheKey("gemini", "m", 0.0, VERSION, "prompt"),
            CacheKey("groq", "other", 0.0, VERSION, "prompt"),
            CacheKey("groq", "m", 0.7, VERSION, "prompt"),
            CacheKey("groq", "m", 0.0, "other/2.0.0", "prompt"),
            CacheKey("groq", "m", 0.0, VERSION, "different"),
        ):
            assert changed.digest != base.digest

    def test_the_same_request_hashes_the_same_way(self):
        assert (
            CacheKey("groq", "m", 0.0, VERSION, "p").digest
            == CacheKey("groq", "m", 0.0, VERSION, "p").digest
        )

    def test_a_stored_answer_comes_back(self, tmp_path):
        cache = ResponseCache(directory=tmp_path)
        key = CacheKey("groq", "m", 0.0, VERSION, "p")
        cache.put(key, "hello")
        assert cache.get(key) == "hello"
        assert cache.hits == 1

    def test_the_prompt_is_written_beside_the_answer_so_a_human_can_audit_it(
        self, tmp_path
    ):
        cache = ResponseCache(directory=tmp_path)
        key = CacheKey("groq", "m", 0.0, VERSION, "the prompt")
        path = cache.put(key, "the answer")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["prompt"] == "the prompt"
        assert payload["completion"] == "the answer"
        assert payload["prompt_version"] == VERSION

    def test_a_missing_entry_is_a_miss(self, tmp_path):
        cache = ResponseCache(directory=tmp_path)
        assert cache.get(CacheKey("groq", "m", 0.0, VERSION, "p")) is None
        assert cache.misses == 1

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        """A half-written file after an interrupted run is a real thing."""
        cache = ResponseCache(directory=tmp_path)
        key = CacheKey("groq", "m", 0.0, VERSION, "p")
        cache.path_for(key).write_text("{not json", encoding="utf-8")
        assert cache.get(key) is None

    def test_an_entry_with_the_wrong_shape_is_a_miss(self, tmp_path):
        cache = ResponseCache(directory=tmp_path)
        key = CacheKey("groq", "m", 0.0, VERSION, "p")
        cache.path_for(key).write_text('{"completion": 12}', encoding="utf-8")
        assert cache.get(key) is None

    def test_the_hit_rate_is_zero_before_anything_is_asked(self, tmp_path):
        assert ResponseCache(directory=tmp_path).hit_rate == 0.0

    def test_a_second_identical_run_makes_no_live_call(self, tmp_path):
        """The claim a demo rests on, checked."""
        first, provider = client_for(tmp_path, responses=[answer(item_id="1")])
        first.complete_json("prompt", NarrationBatch, prompt_version=VERSION)
        assert first.calls == 1

        second, _ = client_for(tmp_path)  # a provider with no responses left
        second.complete_json("prompt", NarrationBatch, prompt_version=VERSION)
        assert second.calls == 0
        assert second.cache_hits == 1
        assert second.ledger().cache_hit_rate == 1.0
        assert provider.responses == []


class TestTheBudget:
    def test_it_stops_the_run_before_the_network(self, tmp_path):
        client, provider = client_for(
            tmp_path, responses=[answer(item_id="1"), answer(item_id="2")],
            max_calls_per_run=1,
        )
        client.complete_json("first", NarrationBatch, prompt_version=VERSION)
        with pytest.raises(BudgetExhausted, match="budget"):
            client.complete_json("second", NarrationBatch, prompt_version=VERSION)
        assert len(provider.responses) == 1  # the second call never happened
        assert client.budget_refusals == 1

    def test_a_cache_hit_costs_no_budget(self, tmp_path):
        """Otherwise a fully cached rerun would look like a full-price one."""
        client, _ = client_for(
            tmp_path, responses=[answer(item_id="1")], max_calls_per_run=1
        )
        client.complete_json("prompt", NarrationBatch, prompt_version=VERSION)
        client.complete_json("prompt", NarrationBatch, prompt_version=VERSION)
        assert client.calls == 1
        assert client.cache_hits == 1

    def test_the_remaining_budget_is_reported(self, tmp_path):
        client, _ = client_for(
            tmp_path, responses=[answer(item_id="1")], max_calls_per_run=3
        )
        assert client.remaining_budget == 3
        client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert client.remaining_budget == 2


class TestValidation:
    def test_a_valid_answer_parses(self, tmp_path):
        client, _ = client_for(
            tmp_path,
            responses=[
                json.dumps(
                    {"extractions": [{"item_id": "1", "utr": "UTR1", "confidence": 0.9}]}
                )
            ],
        )
        parsed, digest = client.complete_json(
            "p", NarrationBatch, prompt_version=VERSION
        )
        assert parsed.extractions[0].utr == "UTR1"
        assert len(digest) == 64

    def test_a_malformed_answer_is_retried_with_the_error_appended(self, tmp_path):
        client, provider = client_for(
            tmp_path, responses=["not json at all", answer(item_id="1")],
            validation_retries=1,
        )
        parsed, _ = client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert parsed.extractions[0].item_id == "1"
        assert client.validation_failures == 1
        assert "rejected by the schema" in provider.prompts[1]

    def test_it_gives_up_after_the_configured_retries(self, tmp_path):
        client, _ = client_for(
            tmp_path, responses=["garbage", "still garbage"], validation_retries=1
        )
        with pytest.raises(LLMValidationError, match="did not validate"):
            client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert client.validation_failures == 2

    def test_an_unexpected_key_is_rejected_rather_than_ignored(self, tmp_path):
        """`extra="forbid"`. A hallucinated field silently dropped is the bug."""
        client, _ = client_for(
            tmp_path,
            responses=[
                json.dumps({"extractions": [], "confidence_overall": 0.9}),
                json.dumps({"extractions": []}),
            ],
        )
        parsed, _ = client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert parsed.extractions == ()
        assert client.validation_failures == 1

    def test_a_fenced_answer_is_unwrapped(self, tmp_path):
        client, _ = client_for(
            tmp_path, responses=['```json\n{"extractions": []}\n```']
        )
        parsed, _ = client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert parsed.extractions == ()
        assert client.validation_failures == 0


class TestProviderFailures:
    @pytest.mark.parametrize(
        "failure",
        [
            LLMUnavailable("groq returned HTTP 429"),
            LLMUnavailable("groq did not respond: timed out"),
            LLMUnavailable("groq returned HTTP 500"),
        ],
    )
    def test_every_outage_raises_the_same_typed_error(self, tmp_path, failure):
        client, provider = client_for(tmp_path)
        provider.failure = failure
        with pytest.raises(LLMUnavailable):
            client.complete_json("p", NarrationBatch, prompt_version=VERSION)
        assert client.provider_failures == 1

    def test_running_out_of_scripted_answers_is_an_outage_not_a_crash(self, tmp_path):
        client, _ = client_for(tmp_path)
        with pytest.raises(LLMUnavailable):
            client.complete_json("p", NarrationBatch, prompt_version=VERSION)


class TestTheDisabledPath:
    def test_a_disabled_client_refuses_loudly(self, tmp_path):
        settings = LLMConfig(enabled=False, cache_dir=tmp_path)
        client = LLMClient(config=settings, provider=ScriptedProvider())
        assert not client.enabled
        with pytest.raises(LLMDisabled, match="--no-llm"):
            client.complete_json("p", NarrationBatch, prompt_version=VERSION)

    def test_a_client_with_no_provider_refuses_loudly(self, tmp_path):
        client = LLMClient(config=LLMConfig(cache_dir=tmp_path), provider=None)
        assert not client.enabled
        with pytest.raises(LLMDisabled, match="no provider"):
            client.complete_json("p", NarrationBatch, prompt_version=VERSION)

    def test_a_disabled_client_still_reports_a_ledger_of_zeros(self, tmp_path):
        client = LLMClient(config=LLMConfig(enabled=False, cache_dir=tmp_path))
        ledger = client.ledger()
        assert ledger.llm_calls == 0
        assert ledger.equivalent_paid_cost_inr == 0.0
        assert ledger.provider_used is None


class TestTheLedger:
    def test_it_counts_tokens_calls_and_hits(self, tmp_path):
        client, _ = client_for(
            tmp_path, responses=[answer(item_id="1"), answer(item_id="2")]
        )
        client.complete_json("a", NarrationBatch, prompt_version=VERSION)
        client.complete_json("b", NarrationBatch, prompt_version=VERSION)
        client.complete_json("a", NarrationBatch, prompt_version=VERSION)
        ledger = client.ledger()
        assert ledger.llm_calls == 2
        assert ledger.cache_hits == 1
        assert ledger.prompt_tokens == 200
        assert ledger.completion_tokens == 80
        assert ledger.total_tokens == 280

    def test_actual_spend_is_zero_and_the_equivalent_is_priced(self, tmp_path):
        client, _ = client_for(tmp_path, responses=[answer(item_id="1")])
        client.complete_json("a", NarrationBatch, prompt_version=VERSION)
        ledger = client.ledger()
        expected = (
            100 * PROMPT_INR_PER_MTOK + 40 * COMPLETION_INR_PER_MTOK
        ) / 1_000_000
        assert ledger.actual_cost_inr == 0.0
        assert ledger.equivalent_paid_cost_inr == pytest.approx(expected, abs=1e-4)

    def test_calls_per_hundred_records_is_the_reported_discipline(self, tmp_path):
        client, _ = client_for(tmp_path, responses=[answer(item_id="1")])
        client.complete_json("a", NarrationBatch, prompt_version=VERSION)
        assert client.ledger().calls_per_100_records(300) == pytest.approx(1 / 3)


class TestBatching:
    def test_it_splits_evenly(self):
        assert batched(list(range(5)), 2) == [[0, 1], [2, 3], [4]]

    def test_a_short_list_is_one_batch(self):
        assert batched([1], 20) == [[1]]

    def test_an_empty_list_is_no_batches(self):
        assert batched([], 20) == []

    def test_a_batch_size_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            batched([1], 0)


class TestBuildingAProvider:
    def test_no_key_means_the_deterministic_path(self, tmp_path):
        assert build_provider(LLMConfig(cache_dir=tmp_path), api_key=None) is None
        assert build_provider(LLMConfig(cache_dir=tmp_path), api_key="") is None

    def test_disabled_means_the_deterministic_path_even_with_a_key(self, tmp_path):
        settings = LLMConfig(enabled=False, cache_dir=tmp_path)
        assert build_provider(settings, api_key="secret") is None

    def test_a_key_builds_an_openai_compatible_provider(self, tmp_path):
        provider = build_provider(LLMConfig(cache_dir=tmp_path), api_key="secret")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == "groq"
        assert provider.base_url.startswith("https://")

    def test_an_unknown_provider_name_falls_back_to_a_known_base_url(self, tmp_path):
        settings = LLMConfig(provider="somethingelse", cache_dir=tmp_path)
        provider = build_provider(settings, api_key="secret")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.base_url


class TestTheHttpTransport:
    """The socket layer, with the socket replaced. Error mapping is the point."""

    def _provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            name="groq", base_url="https://example.invalid/v1", api_key="k", model="m"
        )

    def test_a_good_response_is_parsed_with_its_usage(self, monkeypatch):
        payload = {
            "choices": [{"message": {"content": '{"extractions": []}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen(json.dumps(payload))
        )
        completion = self._provider().complete("p", timeout_s=1.0)
        assert completion.text == '{"extractions": []}'
        assert completion.prompt_tokens == 11
        assert completion.completion_tokens == 7
        assert completion.provider == "groq"

    def test_a_rate_limit_becomes_an_outage(self, monkeypatch):
        def raise_http(*args: object, **kwargs: object):
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr("urllib.request.urlopen", raise_http)
        with pytest.raises(LLMUnavailable, match="429"):
            self._provider().complete("p", timeout_s=1.0)

    def test_a_timeout_becomes_an_outage(self, monkeypatch):
        def raise_timeout(*args: object, **kwargs: object):
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", raise_timeout)
        with pytest.raises(LLMUnavailable, match="did not respond"):
            self._provider().complete("p", timeout_s=0.01)

    def test_an_unreachable_host_becomes_an_outage(self, monkeypatch):
        def raise_url(*args: object, **kwargs: object):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("urllib.request.urlopen", raise_url)
        with pytest.raises(LLMUnavailable, match="did not respond"):
            self._provider().complete("p", timeout_s=1.0)

    def test_a_body_that_is_not_json_becomes_an_outage(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen("<html>502</html>"))
        with pytest.raises(LLMUnavailable, match="not JSON"):
            self._provider().complete("p", timeout_s=1.0)

    def test_a_body_with_no_completion_becomes_an_outage(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen('{"choices": []}'))
        with pytest.raises(LLMUnavailable, match="no completion"):
            self._provider().complete("p", timeout_s=1.0)

    def test_missing_usage_counts_as_zero_tokens_rather_than_failing(self, monkeypatch):
        payload = {"choices": [{"message": {"content": "{}"}}]}
        monkeypatch.setattr(
            "urllib.request.urlopen", _fake_urlopen(json.dumps(payload))
        )
        completion = self._provider().complete("p", timeout_s=1.0)
        assert completion.prompt_tokens == 0


def _fake_urlopen(body: str):
    class _Response:
        def read(self) -> bytes:
            return body.encode("utf-8")

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def opener(*args: object, **kwargs: object) -> _Response:
        return _Response()

    return opener
