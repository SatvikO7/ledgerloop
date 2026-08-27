"""The LLM layer: optional, gated, cached, and never authoritative.

PLAN.md 7.1, the hard rule: **the LLM never decides a match by itself, and never
does arithmetic.** Everything in this package is arranged so that rule is
structural rather than remembered:

* :mod:`ledgerloop.llm.contracts` -- what the model may say, as strict schemas
  that cannot express an amount, a class, a severity or a decision.
* :mod:`ledgerloop.llm.gates` -- provenance checks. A schema proves the shape of
  an answer; these prove that every value in it came from what the model was
  shown.
* :mod:`ledgerloop.llm.cache` -- content-hash disk cache, so a rerun is free and
  a demo can prove it consumed zero live calls.
* :mod:`ledgerloop.llm.client` -- one client for every call: budget, retry,
  validation, ledger. Failures raise; call sites fall back.
* :mod:`ledgerloop.llm.prompts` -- the three prompts PLAN.md 7.2 allows,
  versioned into the cache key and the audit trail.
* :mod:`ledgerloop.llm.tasks` -- the three call sites, each with its
  deterministic fallback beside it.

The arithmetic gate itself lives outside this package, at
:func:`ledgerloop.matching.verify.verify_arithmetic`, and takes no argument
saying where a proposal came from. A gate that could be told "this one is from a
very confident model" would eventually be told exactly that.

``--no-llm`` is not a separate path through any of this. It is the same path
with the first branch taken, which is why the whole system stays runnable, and
measurable, with no key at all.
"""

from __future__ import annotations

from ledgerloop.llm.cache import CacheKey, ResponseCache
from ledgerloop.llm.client import (
    BudgetExhausted,
    Completion,
    LLMClient,
    LLMDisabled,
    LLMError,
    LLMUnavailable,
    LLMValidationError,
    OpenAICompatibleProvider,
    Provider,
    ScriptedProvider,
    batched,
    build_provider,
)
from ledgerloop.llm.contracts import (
    AdjudicationBatch,
    ExceptionExplanation,
    ExplanationBatch,
    NarrationBatch,
    NarrationExtraction,
    ProposedLink,
    ResidualHypothesis,
)
from ledgerloop.llm.gates import (
    GateResult,
    grounded_in_text,
    grounded_refs,
    prose_names_only_known_records,
)
from ledgerloop.llm.prompts import (
    ADJUDICATION_VERSION,
    EXPLANATION_VERSION,
    NARRATION_VERSION,
    EvidencePack,
)
from ledgerloop.llm.tasks import (
    AdjudicationOutcome,
    ExplanationOutcome,
    NarrationOutcome,
    adjudicate_residual,
    evidence_pack_for,
    explain_exceptions,
    parse_narrations,
)

__all__ = [
    "ADJUDICATION_VERSION",
    "EXPLANATION_VERSION",
    "NARRATION_VERSION",
    "AdjudicationBatch",
    "AdjudicationOutcome",
    "BudgetExhausted",
    "CacheKey",
    "Completion",
    "EvidencePack",
    "ExceptionExplanation",
    "ExplanationBatch",
    "ExplanationOutcome",
    "GateResult",
    "LLMClient",
    "LLMDisabled",
    "LLMError",
    "LLMUnavailable",
    "LLMValidationError",
    "NarrationBatch",
    "NarrationExtraction",
    "NarrationOutcome",
    "OpenAICompatibleProvider",
    "ProposedLink",
    "Provider",
    "ResidualHypothesis",
    "ResponseCache",
    "ScriptedProvider",
    "adjudicate_residual",
    "batched",
    "build_provider",
    "evidence_pack_for",
    "explain_exceptions",
    "grounded_in_text",
    "grounded_refs",
    "parse_narrations",
    "prose_names_only_known_records",
]
