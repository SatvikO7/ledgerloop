"""The provider ladder: Groq -> Gemini -> OpenRouter -> Ollama, and the retry.

PLAN.md 10 named this ladder and Step 9 shipped the thing it needs -- one narrow
:class:`~ledgerloop.llm.client.Provider` protocol and call sites that already
degrade when a backend raises. What was missing was the walk itself, and the
``client.py`` docstring said so rather than implying otherwise. This module is
the walk.

WHY A LADDER AT ALL
-------------------
Every rung here is a free tier, and a free tier's failure mode is not an outage
-- it is a **429 at the least convenient moment**, which for this project is the
middle of a demo. One provider plus ``--no-llm`` survives that by giving up the
model entirely. A ladder survives it by moving down a rung, and records how far
it had to move so the run's cost ledger says what actually answered.

The rungs are ordered by what they cost to be wrong about: Groq first because it
is fastest, Ollama last because it is local and therefore the one that keeps
working when the network does not.

WHAT COUNTS AS "TRY THE NEXT RUNG"
-----------------------------------
Anything that means *this backend did not answer*: a timeout, a 429, a 5xx, a
body that is not JSON, a body with no completion in it. All of them arrive as
:class:`~ledgerloop.llm.client.LLMUnavailable` from the transport, which is why
this module needs no vendor-specific error handling.

What does **not** move down a rung is a schema violation. That is a property of
the answer, not of the backend, and it is retried in place by
:meth:`~ledgerloop.llm.client.LLMClient.complete_json` with the error appended.
Handing a malformed answer to the next provider would spend a second free tier
on the same mistake.

RETRY, AND WHY IT IS SO SMALL
------------------------------
One retry per rung, after a short backoff, and only for a rate limit. Two
reasons it is not more:

* A run has a **budget** (``LLMConfig.max_calls_per_run``) and a demo has a
  clock. Four rungs times three attempts times a thirty-second timeout is six
  minutes of a reviewer's attention spent on a tier that contributes prose.
* The deterministic answer is already correct. Retrying hard is what you do when
  the model is the only path; here it is the last one, and giving up is cheap.

``Retry-After`` is honoured when the provider sends it, capped, because a
provider that says "come back in an hour" is telling the truth and waiting for
it is not an option inside one run.

THE SLEEP IS INJECTED
---------------------
:attr:`FailoverProvider.sleep` defaults to :func:`time.sleep` and is replaced in
tests. A retry policy that could only be tested by actually waiting would be
tested loosely or not at all, and this one has to be exact: the assertions are
about *how many* attempts happened and *in what order*, which is precisely what
a real sleep makes slow to check.

NO KEY IS NOT AN ERROR
----------------------
:func:`build_ladder` returns ``None`` when no rung is configured, exactly as
:func:`~ledgerloop.llm.client.build_provider` did for the single-provider case.
A machine with no credentials runs the whole pipeline deterministically and says
so; it does not fail at start-up. ``--no-llm`` reaches the same place on purpose
rather than by accident.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ledgerloop.config import LLMConfig
from ledgerloop.llm.client import (
    Completion,
    LLMUnavailable,
    OpenAICompatibleProvider,
    Provider,
    RateLimited,
)

__all__ = [
    "DEFAULT_LADDER",
    "PROVIDER_BASE_URLS",
    "PROVIDER_KEY_ENVS",
    "PROVIDER_PRICES_INR_PER_MTOK",
    "FailoverProvider",
    "LadderRung",
    "ProviderSpec",
    "build_ladder",
    "configured_rungs",
]

#: Base URLs for the OpenAI-compatible endpoints PLAN.md 10 names.
#:
#: Shared with :mod:`ledgerloop.llm.client` rather than duplicated -- a second
#: copy of a URL is a second thing to get wrong, and the single-provider path
#: and the ladder must reach the same endpoint for the same name.
PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

#: Where each rung's credential lives. One variable per provider, so a machine
#: can hold a Groq key and an OpenRouter key at once and the ladder uses both.
#:
#: ``LEDGERLOOP_LLM_API_KEY`` remains a fallback for every rung, which is what
#: keeps the Step 9 invocation working unchanged.
PROVIDER_KEY_ENVS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

#: The model each rung is asked for when the run's configured model is not the
#: rung's own. A ladder that sent Groq's model id to OpenRouter would fail on
#: every rung below the first for a reason that had nothing to do with quota.
PROVIDER_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "ollama": "llama3.1",
}

#: (prompt, completion) rupees per million tokens, for the **actual** spend
#: figure. Zero where the rung is a free tier or runs locally, which is every
#: rung this project is configured for -- so ``actual_cost_inr`` stays ₹0 and is
#: true rather than assumed. A paid rung would put a real number here and the
#: ledger would report it without any other change.
PROVIDER_PRICES_INR_PER_MTOK: dict[str, tuple[float, float]] = {
    "groq": (0.0, 0.0),
    "gemini": (0.0, 0.0),
    "openrouter": (0.0, 0.0),
    "ollama": (0.0, 0.0),
}

#: The order PLAN.md 10 specifies. Fastest first, local last.
DEFAULT_LADDER: tuple[str, ...] = ("groq", "gemini", "openrouter", "ollama")

#: Ollama needs no credential: it is a process on this machine, and requiring a
#: key for it would make the one rung that works offline the one rung a keyless
#: machine cannot reach.
#:
#: It is therefore also the one rung that must be **asked for**. A machine with
#: no credentials at all has to reach the deterministic path immediately -- not
#: after a connection to ``localhost:11434`` times out -- so a keyless rung
#: joins the ladder only when the operator names it in
#: ``LEDGERLOOP_LLM_PROVIDERS`` or points ``OLLAMA_BASE_URL`` somewhere. Absent
#: both, "no key" means what it has always meant here: run deterministically and
#: say so.
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"ollama"})


@dataclass(frozen=True)
class ProviderSpec:
    """One rung, resolved: which endpoint, which model, which credential."""

    name: str
    base_url: str
    model: str
    api_key: str

    @property
    def prices_inr_per_mtok(self) -> tuple[float, float]:
        return PROVIDER_PRICES_INR_PER_MTOK.get(self.name, (0.0, 0.0))


@dataclass(frozen=True)
class LadderRung:
    """What happened on one rung of one call. Kept for the report."""

    provider: str
    attempts: int
    error: str | None = None

    @property
    def answered(self) -> bool:
        return self.error is None


@dataclass
class FailoverProvider:
    """Walks the ladder until a rung answers, and records how far it walked.

    It **is** a :class:`~ledgerloop.llm.client.Provider`, so it drops into the
    existing client with no change above it: the cache, the budget, the schema
    validation, the audit record and the cost ledger all sit where they were.
    That is the point of the protocol being one method wide.
    """

    rungs: tuple[Provider, ...]
    retries_per_rung: int = 1
    backoff_s: float = 0.5
    max_backoff_s: float = 5.0
    sleep: Callable[[float], None] = time.sleep
    fallback_depth: int = 0
    """How far down the ladder the **last** answer came from. 0 is the first rung."""

    max_fallback_depth: int = 0
    """The deepest rung any call in this run needed. What the report prints."""

    trail: list[LadderRung] = field(default_factory=list)
    """One entry per rung tried, across the whole run. The audit of the walk."""

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("a failover ladder needs at least one provider")
        if self.retries_per_rung < 0:
            raise ValueError(
                f"retries_per_rung must not be negative, got {self.retries_per_rung}"
            )

    answered_by: str | None = None
    """The rung that answered the most recent call, for the cost ledger."""

    @property
    def name(self) -> str:
        """The **ladder's** identity, and deliberately not the rung that answered.

        This string goes into the response cache key
        (:class:`~ledgerloop.llm.cache.CacheKey`), so it has to be stable for the
        life of the run. Returning the last rung that answered would change the
        key the moment a rate limit pushed one call down a rung, and every
        subsequent identical prompt would miss a cache entry it had already
        paid for -- turning a transient 429 into a permanent extra cost.

        The rung that actually replied is :attr:`answered_by`, and that is what
        ``CostLedger.provider_used`` reports.
        """
        return "ladder:" + ">".join(rung.name for rung in self.rungs)

    @property
    def ladder(self) -> tuple[str, ...]:
        return tuple(rung.name for rung in self.rungs)

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        """One answer, from the highest rung that gives one.

        Raises :class:`~ledgerloop.llm.client.LLMUnavailable` naming every rung
        it tried once the ladder is exhausted. The call site catches it and
        falls back to what it would have done with ``--no-llm`` -- which is the
        behaviour that makes the whole ladder optional rather than load-bearing.
        """
        failures: list[str] = []
        for depth, rung in enumerate(self.rungs):
            error = self._try_rung(rung, prompt, timeout_s=timeout_s)
            if isinstance(error, Completion):
                self.fallback_depth = depth
                self.max_fallback_depth = max(self.max_fallback_depth, depth)
                self.answered_by = rung.name
                return error
            failures.append(f"{rung.name}: {error}")
        raise LLMUnavailable(
            "every provider on the ladder declined -- " + "; ".join(failures)
        )

    def _try_rung(
        self, rung: Provider, prompt: str, *, timeout_s: float
    ) -> Completion | str:
        """One rung, with its retries. A completion, or why it did not answer."""
        attempts = 0
        message = "no attempt was made"
        for attempt in range(self.retries_per_rung + 1):
            attempts += 1
            try:
                completion = rung.complete(prompt, timeout_s=timeout_s)
            except RateLimited as exc:
                message = str(exc)
                if attempt == self.retries_per_rung:
                    break
                self.sleep(self._wait_for(exc, attempt))
            except LLMUnavailable as exc:
                # Not a quota problem: the rung is down, misconfigured or
                # unreachable, and waiting half a second will not change that.
                # Move on rather than spending the retry on it.
                message = str(exc)
                break
            else:
                self.trail.append(LadderRung(provider=rung.name, attempts=attempts))
                return completion
        self.trail.append(
            LadderRung(provider=rung.name, attempts=attempts, error=message)
        )
        return message

    def _wait_for(self, error: RateLimited, attempt: int) -> float:
        """How long to wait: the provider's own answer, or an exponential guess.

        Capped either way. A ``Retry-After`` of an hour is true and unusable
        inside one run, and honouring it literally would hang a demo on a header.
        """
        hinted = error.retry_after_s
        if hinted is not None:
            return min(max(float(hinted), 0.0), self.max_backoff_s)
        return min(self.backoff_s * (2.0**attempt), self.max_backoff_s)


def configured_rungs(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
) -> tuple[ProviderSpec, ...]:
    """Which rungs this machine can actually reach, in ladder order.

    A rung is included when its credential is present -- its own variable, or
    the shared ``LEDGERLOOP_LLM_API_KEY`` -- or when it needs none, which is
    Ollama. Everything else is skipped silently: an absent key is the ordinary
    state of a machine, not a misconfiguration to complain about.

    The run's configured model is used for the rung it names and each other
    rung's own default elsewhere, because a model id is provider-specific and
    sending Groq's to OpenRouter fails for a reason that is not quota.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    requested = order is not None or bool(env.get("LEDGERLOOP_LLM_PROVIDERS"))
    names = tuple(order) if order is not None else _ladder_from(env)
    shared = env.get("LEDGERLOOP_LLM_API_KEY", "")

    specs: list[ProviderSpec] = []
    for name in names:
        if name not in PROVIDER_BASE_URLS:
            continue
        key = env.get(PROVIDER_KEY_ENVS.get(name, ""), "") or shared
        if not key and name not in KEYLESS_PROVIDERS:
            continue
        if not key and not (requested or env.get(f"{name.upper()}_BASE_URL")):
            # A keyless rung nobody asked for. See KEYLESS_PROVIDERS.
            continue
        specs.append(
            ProviderSpec(
                name=name,
                base_url=env.get(
                    f"{name.upper()}_BASE_URL", PROVIDER_BASE_URLS[name]
                ),
                model=(
                    config.model
                    if name == config.provider
                    else PROVIDER_MODELS.get(name, config.model)
                ),
                api_key=key,
            )
        )
    return tuple(specs)


def _ladder_from(env: Mapping[str, str]) -> tuple[str, ...]:
    """``LEDGERLOOP_LLM_PROVIDERS=groq,ollama`` overrides the default order.

    An unknown name is dropped by :func:`configured_rungs` rather than raising:
    the variable is a convenience for trying a rung, and a typo in it should
    leave the run deterministic, not stop it.
    """
    raw = env.get("LEDGERLOOP_LLM_PROVIDERS", "")
    if not raw:
        return DEFAULT_LADDER
    names = tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())
    return names or DEFAULT_LADDER


def build_ladder(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FailoverProvider | None:
    """The provider a run should use, or ``None`` for the deterministic path.

    ``None`` when the run asked for no model, or when no rung is reachable. Both
    reach the same place as ``--no-llm`` by design -- see the module docstring.
    """
    if not config.enabled:
        return None
    specs = configured_rungs(config, environ=environ, order=order)
    if not specs:
        return None
    return FailoverProvider(
        rungs=tuple(
            OpenAICompatibleProvider(
                name=spec.name,
                base_url=spec.base_url,
                api_key=spec.api_key,
                model=spec.model,
                temperature=config.temperature,
            )
            for spec in specs
        ),
        sleep=sleep,
    )
