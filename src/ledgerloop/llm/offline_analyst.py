"""A stand-in that answers the three **production** prompts, from the prompt.

WHY A SECOND STAND-IN
---------------------
:mod:`ledgerloop.eval.offline_provider` answers B2's one prompt -- the whole
corpus in, links out -- and nothing else. The three prompts the production path
sends are different shapes with different schemas, and none of them is a
corpus dump. So Phase 2.2 needs a reasoner for those, and the honest way to get
one is to write it and label it, not to reuse a rule built for another question.

WHAT IT IS
----------
A real :class:`~ledgerloop.llm.client.Provider`. Its answers travel through the
same prompts, the same content-hash cache, the same budget, the same schema
validation, the same grounding gate, the same ``verify_arithmetic`` and the same
cost ledger a live provider's would. Everything in ``ledgerloop llm-report``'s
**machinery** columns -- calls, cache hits, tokens, latency, failures, budget
refusals, hallucinations refused, proposals demoted -- is measured on the real
code path when it drives this.

WHAT IT IS NOT
--------------
It is not a language model, and no figure it produces is a claim about one. Its
acceptance rate is a property of the rules below. ``llm-report`` records
``live: false`` when it answered, and the report prints a banner. **No claim is
made here about any model's answer quality.**

THE RULES, IN FULL
------------------
It reads only the prompt string. It never opens a source file, never sees ground
truth, and takes no argument but the text -- there is a test asserting the
signature.

``narration``
    Returns the first ``UTR<digits>`` substring **present in that item's own
    text**, and the longest run of letters and spaces as the merchant. Both are
    then checked by the grounding gate against the same string, so an item whose
    text holds no reference produces a null and an item whose text holds one
    produces an accepted repair. Confidence is fixed at 0.5, because a constant
    is honest about a rule that has no opinion and a varying number would look
    like one.

``adjudication``
    For each item, proposes the **first** candidate credit the pack lists,
    linked to the item's first payment. That is a deliberately weak proposal:
    the residual reaching T5 is precisely the set the deterministic ladder
    refused, so most of these do not reconcile -- and watching
    ``verify_arithmetic`` demote them is the measurement this exists for.
    Evidence refs are copied from the pack's own ``records:`` line, so the
    grounding gate passes and the *arithmetic* gate is the one under test.

``explanation``
    Rewrites the summary line into a root cause and a next action naming the
    first record id the item listed. Two different sentences, because the schema
    refuses a diagnosis repeated as an instruction.

It **does not invent record ids**, so the hallucination counters read zero when
it drives a run. A zero from a reasoner incapable of the failure is not evidence
that the failure does not happen, and the report says so rather than letting the
zero look like a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ledgerloop.llm.client import Completion

__all__ = ["OFFLINE_ANALYST_NAME", "OfflineAnalyst"]

OFFLINE_ANALYST_NAME = "offline-analyst"

#: ``item-0: "NEFT CR-DIRECT TRANSFER-370162-INWARD"``
_NARRATION_ITEM = re.compile(r'^(?P<id>[A-Za-z0-9\-]+): "(?P<text>.*)"$')

#: ``Item SETL-0001: payout of ... not credited``
_ADJUDICATION_ITEM = re.compile(r"^Item (?P<id>[A-Za-z0-9\-]+): (?P<summary>.*)$")

#: ``Exception exception:bank_txn:BNK-00107: E_DUPLICATE_CREDIT, CRITICAL, ...``
#:
#: The id carries colons -- it is ``exception:<record ref>`` -- so the character
#: class has to admit them and the ``: `` separator does the splitting. A class
#: that stopped at the first colon would echo ``exception`` as every item's id,
#: and every rewrite would then be refused as naming an item the batch does not
#: contain. Found by the acceptance count reading zero.
_EXPLANATION_ITEM = re.compile(r"^Exception (?P<id>[A-Za-z0-9:_\-]+): (?P<summary>.*)$")

_RECORDS_LINE = re.compile(r"^\s*records: (?P<refs>.+)$")
_EVIDENCE_LINE = re.compile(r"^\s*evidence: (?P<text>.+)$")
_UTR = re.compile(r"UTR\d{6,20}")
_MERCHANT = re.compile(r"[A-Z][A-Z ]{4,}")
_PAYMENT_KEY = re.compile(r"payment:([A-Za-z0-9\-]+)")
_BANK_KEY = re.compile(r"bank_txn:([A-Za-z0-9\-]+)")
_SETTLEMENT_KEY = re.compile(r"settlement:([A-Za-z0-9\-]+)")


@dataclass
class OfflineAnalyst:
    """Answers a narration, adjudication or explanation prompt by shape.

    ``prompts`` records everything it was asked, so a test can assert it was
    handed the production prompt rather than a convenience shape.
    """

    name: str = OFFLINE_ANALYST_NAME
    prompts: list[str] = field(default_factory=list)
    prompt_tokens_per_char: float = 0.25
    """Tokens per prompt character, for the ledger.

    Roughly four characters to a token, the usual English ratio. It is an
    **estimate and is labelled as one** wherever it is reported: this provider
    has no tokeniser, and a precise-looking count would be worse than an honest
    approximation.
    """

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        del timeout_s
        self.prompts.append(prompt)
        text = self._answer(prompt)
        return Completion(
            text=text,
            prompt_tokens=int(len(prompt) * self.prompt_tokens_per_char),
            completion_tokens=int(len(text) * self.prompt_tokens_per_char),
            latency_ms=1,
            provider=self.name,
        )

    def _answer(self, prompt: str) -> str:
        """Route by the prompt's own opening line. No other state is consulted."""
        if "bank statement narrations" in prompt:
            return json.dumps({"extractions": _narrations(prompt)})
        if "reconciliation analyst" in prompt:
            return json.dumps({"hypotheses": _hypotheses(prompt)})
        if "finance controller reads" in prompt:
            return json.dumps({"explanations": _explanations(prompt)})
        # An unrecognised shape is answered with an empty batch rather than a
        # guess. Every contract's collection field defaults to empty, so this
        # validates, contributes nothing, and is visible as a zero.
        return json.dumps({})


def _narrations(prompt: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in prompt.splitlines():
        match = _NARRATION_ITEM.match(line.strip())
        if match is None:
            continue
        text = match.group("text")
        reference = _UTR.search(text)
        merchant = _MERCHANT.search(text)
        out.append(
            {
                "item_id": match.group("id"),
                "utr": reference.group(0) if reference else None,
                "merchant": merchant.group(0).strip() if merchant else None,
                "confidence": 0.5,
            }
        )
    return out


@dataclass
class _Block:
    """One item's lines, gathered as the prompt is walked once."""

    item_id: str
    summary: str
    lines: list[str] = field(default_factory=list)


def _blocks(prompt: str, pattern: re.Pattern[str]) -> list[_Block]:
    """Split a prompt into per-item blocks. Everything below an item is its own."""
    blocks: list[_Block] = []
    for line in prompt.splitlines():
        header = pattern.match(line)
        if header is not None:
            blocks.append(
                _Block(item_id=header.group("id"), summary=header.group("summary"))
            )
        elif blocks and line.startswith("  "):
            blocks[-1].lines.append(line)
    return blocks


def _hypotheses(prompt: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for block in _blocks(prompt, _ADJUDICATION_ITEM):
        refs = _refs_of(block)
        payments = [
            payment for ref in refs for payment in _PAYMENT_KEY.findall(ref)
        ]
        credits = [txn for ref in refs for txn in _BANK_KEY.findall(ref)]
        settlements = [item for ref in refs for item in _SETTLEMENT_KEY.findall(ref)]
        link: dict[str, object] | None = None
        if payments and credits:
            link = {
                "payment_id": payments[0],
                "bank_txn_id": credits[0],
                "settlement_id": settlements[0] if settlements else None,
                "payment_ids": payments,
            }
        out.append(
            {
                "item_id": block.item_id,
                "hypothesis": (
                    f"The payout described as {block.summary[:200]} has no credit "
                    "the ladder would accept; the nearest unclaimed credit in the "
                    "pack is the only candidate the evidence names."
                ),
                "proposed_link": link,
                "confidence": 0.5,
                "reasoning": "Nearest unclaimed candidate named in the pack.",
                "evidence_refs": tuple(refs),
            }
        )
    return out


def _refs_of(block: _Block) -> list[str]:
    for line in block.lines:
        match = _RECORDS_LINE.match(line)
        if match is not None:
            return [ref.strip() for ref in match.group("refs").split(",") if ref.strip()]
    return []


def _explanations(prompt: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for block in _blocks(prompt, _EXPLANATION_ITEM):
        evidence = [
            match.group("text")
            for line in block.lines
            if (match := _EVIDENCE_LINE.match(line)) is not None
        ]
        first = evidence[0] if evidence else block.summary
        out.append(
            {
                "exception_id": block.item_id,
                "root_cause": f"{block.summary[:400]} The evidence states: {first[:300]}",
                "suggested_action": (
                    "Open the records named in the evidence chain and confirm the "
                    "figure against the source document before adjusting anything."
                ),
            }
        )
    return out
