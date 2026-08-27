"""The three prompts, versioned and rendered deterministically.

PLAN.md 7.2 names exactly three call sites and this module holds all three.
Two properties matter:

**Versioned.** Every template carries a version string that goes into the cache
key and into the audit trail. A prompt edited without a version bump would
serve yesterday's answers to today's question, and the failure would look like
a model regression rather than a cache bug.

**Deterministic rendering.** The same items in the same order always produce the
same string, so the same cache entry. Nothing here reads a clock, a set's
iteration order, or a dictionary the caller built without sorting.

WHAT EVERY PROMPT SAYS, AND WHAT NONE OF THEM ASKS FOR
-------------------------------------------------------
Each prompt states the boundary in the instructions themselves -- that the
answer will be checked against the sources, that inventing a reference is worse
than returning null, that no arithmetic is wanted. That is not a substitute for
the gates in :mod:`ledgerloop.llm.gates`; the gates are what enforce it. It is
there because a model told what will be checked returns null more often, and a
null is the cheapest correct answer in the system.

None of them asks for a rupee figure, an exception class, a severity, or a
decision. Those are not the model's to give.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ADJUDICATION_VERSION",
    "EXPLANATION_VERSION",
    "NARRATION_VERSION",
    "EvidencePack",
    "render_adjudication",
    "render_explanation",
    "render_narration",
]

NARRATION_VERSION = "narration/1.0.0"
ADJUDICATION_VERSION = "adjudicate/1.0.0"
EXPLANATION_VERSION = "explain/1.0.0"

_GROUNDING = (
    "Rules:\n"
    "- Every value you return is checked against the text above. A value that "
    "does not appear there is discarded and counted as a hallucination.\n"
    "- Returning null is correct and expected when the text does not say. It is "
    "always better than a plausible guess.\n"
    "- Do not calculate anything. Amounts are re-derived from the source "
    "documents by code that ignores what you say about them.\n"
    "- Reply with JSON only, matching the schema exactly."
)


def render_narration(items: Sequence[tuple[str, str]]) -> str:
    """``parse_narration``: read a reference and a merchant out of free text.

    The model is shown the narration and nothing else -- no amounts, no
    settlement ids, no candidate matches. It cannot propose a link even by
    accident, because it is not told that links exist. PLAN.md 7.3 calls this
    site "regex first, LLM only on regex miss", so every item here is one the
    deterministic parser already failed on.
    """
    lines = [
        "You are reading bank statement narrations that a regular-expression "
        "parser could not resolve.",
        "",
        "For each item, extract the transfer reference (a UTR) and the "
        "counterparty merchant name, if the text contains them.",
        "",
        "Items:",
    ]
    lines.extend(f'{item_id}: "{narration}"' for item_id, narration in items)
    lines.extend(
        [
            "",
            _GROUNDING,
            "",
            'Schema: {"extractions": [{"item_id": str, "utr": str|null, '
            '"merchant": str|null, "confidence": float}]}',
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class EvidencePack:
    """The compact context one residual item is adjudicated against.

    PLAN.md 7.3 calls for "a compact evidence pack (candidate links, amount
    deltas, graph neighbourhood)". Compact is a cost decision and a safety one:
    a prompt carrying the whole corpus would let the model reach for records
    the item has nothing to do with, and every reference it returns is checked
    against exactly this pack.
    """

    item_id: str
    summary: str
    refs: tuple[str, ...]
    facts: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"Item {self.item_id}: {self.summary}"]
        lines.extend(f"  fact: {fact}" for fact in self.facts)
        lines.extend(f"  candidate: {candidate}" for candidate in self.candidates)
        lines.append(f"  records: {', '.join(self.refs)}")
        return "\n".join(lines)


def render_adjudication(packs: Sequence[EvidencePack]) -> str:
    """``adjudicate_residual``: a hypothesis about what happened, and maybe a link.

    The prompt states plainly that a proposed link is a *proposal*: it will be
    re-derived from the source documents by ``verify_arithmetic`` and refused if
    the money does not close. Saying so up front is not decoration -- a model
    that knows the arithmetic will be checked proposes fewer links that do not
    add up.
    """
    lines = [
        "You are a reconciliation analyst looking at items an automated ladder "
        "could not resolve. For each item, give one hypothesis about what "
        "happened, and a proposed link only if the evidence supports one.",
        "",
    ]
    lines.extend(pack.render() for pack in packs)
    lines.extend(
        [
            "",
            "Rules:",
            "- A proposed link is a proposal. Code re-derives the money from the "
            "source documents and refuses any link that does not reconcile.",
            "- Cite only the record ids listed under the item you are discussing. "
            "Citing anything else is counted as a hallucination and the whole "
            "hypothesis is discarded.",
            "- Do not calculate amounts and do not state any.",
            "- confidence is your own estimate. It is treated as one feature among "
            "many and is recalibrated against measured outcomes, not trusted.",
            "- Reply with JSON only, matching the schema exactly.",
            "",
            'Schema: {"hypotheses": [{"item_id": str, "hypothesis": str, '
            '"proposed_link": {"payment_id": str, "bank_txn_id": str, '
            '"settlement_id": str|null, "payment_ids": [str]}|null, '
            '"confidence": float, "reasoning": str, "evidence_refs": [str]}]}',
        ]
    )
    return "\n".join(lines)


def render_explanation(items: Sequence[tuple[str, str, Sequence[str]]]) -> str:
    """``explain_exception`` / ``suggest_action``: prose for a finished exception.

    The class, the severity and the rupee figure are **given to** the model as
    settled facts, not asked of it. All it may do is say the same thing in
    better English and name a next step. That is the narrowest useful role, and
    it is the only one where a wrong answer costs a sentence rather than a
    reconciliation.
    """
    lines = [
        "You are writing the explanation a finance controller reads for each "
        "exception below. The classification, the severity and the amount are "
        "already decided and are not yours to change.",
        "",
    ]
    for exception_id, summary, evidence in items:
        lines.append(f"Exception {exception_id}: {summary}")
        lines.extend(f"  evidence: {item}" for item in evidence)
    lines.extend(
        [
            "",
            "Rules:",
            "- Write a root cause grounded in the evidence lines, and one concrete "
            "next action naming the record to look at or the document to request.",
            "- Mention only the record ids that appear above. Any other id is "
            "counted as a hallucination and your text is discarded in favour of "
            "the deterministic template.",
            "- Do not restate the amount as a calculation and do not change it.",
            "- Do not speculate about intent or blame.",
            "- Reply with JSON only, matching the schema exactly.",
            "",
            'Schema: {"explanations": [{"exception_id": str, "root_cause": str, '
            '"suggested_action": str}]}',
        ]
    )
    return "\n".join(lines)
