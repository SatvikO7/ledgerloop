"""One client, one provider protocol, and the budget that stops it.

PLAN.md 7.3: "every call goes through one OpenAI-compatible client". This is
that client. It owns four things a call site should never have to think about
-- the cache, the budget, the retry, and the audit record -- and it owns them in
one place so that no call site can accidentally skip one.

WHY THE TRANSPORT IS ``urllib``
-------------------------------
The obvious dependency here is ``openai`` or ``httpx``. Neither is added, for
the same reason NetworkX and pandas were not (ARCHITECTURE.md 6, decisions 9,
29, 32): an OpenAI-compatible chat completion is one POST of one JSON body, and
``urllib.request`` in the standard library does it in thirty lines. The project's
claim is that it runs on nothing; an HTTP client for one endpoint is not where
that claim should be spent.

The trade is named: ``urllib`` has no connection pooling and a plainer error
surface. Neither matters at fewer than thirty calls per run, and the retry and
timeout policy this module needs is explicit either way.

WHAT "NEVER A CRASH" MEANS HERE
-------------------------------
PLAN.md 7.4 requires that validation failure never crashes a run. This module
generalises that to every failure mode a network introduces: a timeout, a 429,
a 500, a malformed body and a schema violation all raise
:class:`LLMUnavailable` or :class:`LLMValidationError`, and every call site
catches them and falls back to what it would have done with ``--no-llm``.
A rate limit slows a run; it never fails one.

THE BUDGET IS A HARD STOP, NOT A TARGET
---------------------------------------
``LLMConfig.max_calls_per_run`` is checked before every live call. Exceeding it
raises :class:`BudgetExhausted`, which the call sites treat exactly like an
outage -- the deterministic answer stands. Free-tier quota that runs out
mid-demo is the failure this exists to prevent, and a budget that logged a
warning and carried on would not prevent it.

Cache hits do **not** consume budget. The budget prices API calls, and a hit is
not one; counting hits would make a fully cached rerun of a 30-call run look
like a 30-call run.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import ValidationError

from ledgerloop.config import LLMConfig
from ledgerloop.llm.cache import CacheKey, ResponseCache
from ledgerloop.models.base import LedgerModel
from ledgerloop.models.metrics import CostLedger

__all__ = [
    "BudgetExhausted",
    "Completion",
    "LLMClient",
    "LLMDisabled",
    "LLMError",
    "LLMUnavailable",
    "LLMValidationError",
    "OpenAICompatibleProvider",
    "Provider",
    "ScriptedProvider",
]


class LLMError(RuntimeError):
    """Base for every LLM failure. Always caught at the call site."""


class LLMUnavailable(LLMError):
    """The provider could not be reached, or refused: timeout, 429, 5xx, garbage."""


class LLMValidationError(LLMError):
    """The response did not satisfy its schema, after the configured retries."""


class BudgetExhausted(LLMError):
    """The run has spent ``max_calls_per_run``. Treated exactly like an outage."""


class LLMDisabled(LLMError):
    """``--no-llm``. Raised only if something asks for a call anyway."""


@dataclass(frozen=True)
class Completion:
    """One provider response, with what it cost."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    provider: str = ""


class Provider(Protocol):
    """The one thing a backend has to do.

    Deliberately narrower than any vendor SDK: a string in, a
    :class:`Completion` out. Everything else -- caching, budget, retry,
    validation, audit -- lives above this line, so a second provider is a
    thirty-line class and not a second policy.
    """

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str, *, timeout_s: float) -> Completion: ...


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """A chat-completions endpoint reached with the standard library.

    Groq, Gemini's compatibility layer, OpenRouter and Ollama all speak this
    shape, which is why PLAN.md 10 chose it: swapping provider is a base URL
    and a key, not a rewrite.

    The provider **failover ladder** (Groq -> Gemini -> OpenRouter -> Ollama) is
    a Step 14 stretch item and is not implemented here. What is implemented is
    the thing the ladder would need: a single narrow protocol, an error type
    that means "this backend did not answer", and call sites that already
    degrade gracefully when it is raised.
    """

    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # rate limits and server errors
            raise LLMUnavailable(f"{self.name} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailable(f"{self.name} did not respond: {exc}") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise LLMUnavailable(f"{self.name} returned a body that is not JSON") from exc

        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(
                f"{self.name} returned a body with no completion in it"
            ) from exc
        usage = payload.get("usage") or {}
        return Completion(
            text=text if isinstance(text, str) else "",
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=int(latency_ms),
            provider=self.name,
        )


@dataclass
class ScriptedProvider:
    """A provider that replays a list of answers. Tests and offline demos.

    Not a mock in the usual sense -- it implements the same protocol and goes
    through the same cache, budget, retry and validation path, so a test using
    it exercises every line the real provider would except the socket.
    """

    responses: list[str] = field(default_factory=list)
    name: str = "scripted"
    prompt_tokens: int = 100
    completion_tokens: int = 40
    failure: Exception | None = None
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        del timeout_s
        self.prompts.append(prompt)
        if self.failure is not None:
            raise self.failure
        if not self.responses:
            raise LLMUnavailable("scripted provider ran out of responses")
        return Completion(
            text=self.responses.pop(0),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=1,
            provider=self.name,
        )


_T = TypeVar("_T", bound=LedgerModel)

#: Batching is shape-agnostic: narrations are tuples, evidence packs are
#: dataclasses, exceptions are models. A second variable rather than reusing the
#: schema one, which is deliberately bounded to a validated contract.
_ItemT = TypeVar("_ItemT")

#: Paid-frontier prices, rupees per million tokens, for the equivalent-cost
#: figure PLAN.md 7.3 asks for. Actual spend is ₹0 on the free tier; this is
#: what the same run would have cost had it not been, which is the number that
#: quantifies what deterministic-first buys.
PROMPT_INR_PER_MTOK = 250.0
COMPLETION_INR_PER_MTOK = 1250.0


@dataclass
class LLMClient:
    """The single door to a model. Cache, budget, retry, validation, ledger.

    ``provider`` is ``None`` under ``--no-llm``: the client still exists, still
    reports a ledger of zeros, and raises :class:`LLMDisabled` if anything asks
    it for a completion. That is deliberately louder than returning a canned
    answer -- a disabled run that silently produced model-shaped output would be
    the one bug this whole design exists to prevent.
    """

    config: LLMConfig
    provider: Provider | None = None
    cache: ResponseCache | None = None
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    validation_failures: int = 0
    provider_failures: int = 0
    budget_refusals: int = 0
    prompt_hashes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = ResponseCache(directory=self.config.cache_dir)

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.provider is not None

    @property
    def remaining_budget(self) -> int:
        return max(0, self.config.max_calls_per_run - self.calls)

    def complete_json(
        self,
        prompt: str,
        schema: type[_T],
        *,
        prompt_version: str,
        timeout_s: float = 30.0,
    ) -> tuple[_T, str]:
        """One validated call. Returns the parsed response and its prompt hash.

        The order is the policy: **cache first, budget second, provider third,
        validation last**. A cached answer costs no budget and no network; a
        budget refusal never reaches the network; a network answer is never
        trusted before it validates.

        On a validation failure the prompt is retried
        ``LLMConfig.validation_retries`` times with the error appended, exactly
        as PLAN.md 7.4 specifies, and then :class:`LLMValidationError` is raised
        for the call site to fall back from. A retry is a fresh cache key
        because the prompt genuinely differs.
        """
        if not self.config.enabled:
            raise LLMDisabled("this run was started with --no-llm")
        if self.provider is None:
            raise LLMDisabled("no provider is configured")

        attempt_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(self.config.validation_retries + 1):
            key = CacheKey(
                provider=self.provider.name,
                model=self.config.model,
                temperature=self.config.temperature,
                prompt_version=prompt_version,
                prompt=attempt_prompt,
            )
            digest = key.digest
            assert self.cache is not None  # set in __post_init__
            text = self.cache.get(key)
            if text is not None:
                self.cache_hits += 1
            else:
                if self.calls >= self.config.max_calls_per_run:
                    self.budget_refusals += 1
                    raise BudgetExhausted(
                        f"this run has spent its budget of "
                        f"{self.config.max_calls_per_run} call(s)"
                    )
                try:
                    completion = self.provider.complete(
                        attempt_prompt, timeout_s=timeout_s
                    )
                except LLMError:
                    self.provider_failures += 1
                    raise
                self.calls += 1
                self.prompt_tokens += completion.prompt_tokens
                self.completion_tokens += completion.completion_tokens
                self.latency_ms += completion.latency_ms
                text = completion.text
                self.cache.put(key, text)
            self.prompt_hashes.append(digest)

            try:
                return schema.model_validate_json(_strip_fences(text)), digest
            except ValidationError as exc:
                self.validation_failures += 1
                last_error = exc
                attempt_prompt = (
                    f"{prompt}\n\nYour previous answer was rejected by the schema:\n"
                    f"{exc.errors(include_url=False)!r}\n"
                    "Reply with JSON that satisfies the schema exactly."
                )
                if attempt == self.config.validation_retries:
                    break
        raise LLMValidationError(
            f"the model's answer did not validate after "
            f"{self.config.validation_retries + 1} attempt(s): {last_error}"
        )

    def ledger(self) -> CostLedger:
        """The cost record for this run.

        ``actual_cost_inr`` is ₹0 by construction -- the free tier is the whole
        point -- so the interesting figure is what the same tokens would have
        cost on a paid frontier API. Reporting both is what turns "we used an
        LLM sparingly" from a claim into a number.
        """
        equivalent = (
            self.prompt_tokens * PROMPT_INR_PER_MTOK
            + self.completion_tokens * COMPLETION_INR_PER_MTOK
        ) / 1_000_000
        return CostLedger(
            llm_calls=self.calls,
            cache_hits=self.cache_hits,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            wall_clock_ms=self.latency_ms,
            actual_cost_inr=0.0,
            equivalent_paid_cost_inr=round(equivalent, 4),
            provider_used=self.provider.name if self.provider is not None else None,
            fallback_depth=0,
        )


def _strip_fences(text: str) -> str:
    """Remove a markdown code fence a model wrapped its JSON in.

    Not leniency about the schema -- the parsed object still has to satisfy it
    exactly. This is leniency about a formatting habit that has nothing to do
    with the content, and refusing over it would spend a retry on punctuation.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def build_provider(config: LLMConfig, *, api_key: str | None) -> Provider | None:
    """The provider a run should use, or ``None`` for the deterministic path.

    Returns ``None`` rather than raising when there is no key: a machine without
    credentials should run the whole pipeline deterministically and say so, not
    fail at start-up. ``--no-llm`` and "no key configured" reach the same place
    by design -- the second is the accident the first is the choice.
    """
    if not config.enabled or not api_key:
        return None
    return OpenAICompatibleProvider(
        name=config.provider,
        base_url=_BASE_URLS.get(config.provider, _BASE_URLS["groq"]),
        api_key=api_key,
        model=config.model,
        temperature=config.temperature,
    )


#: Base URLs for the OpenAI-compatible endpoints PLAN.md 10 names.
_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}

__all__ += ["build_provider"]


def batched(items: Sequence[_ItemT], size: int) -> list[Sequence[_ItemT]]:
    """Split a work list into prompt-sized batches.

    Batching is the cost control PLAN.md 7.3 relies on: twenty narrations in one
    call rather than twenty calls. It lives here so every call site batches the
    same way and the budget means the same thing at each of them.
    """
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")
    return [items[start : start + size] for start in range(0, len(items), size)]


__all__ += ["batched"]
