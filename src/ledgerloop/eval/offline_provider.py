"""A stand-in reasoner for B2 on a machine with no API key.

WHAT THIS IS, AND WHAT IT IS NOT
-------------------------------
There is no provider key in this environment (Step 9, §5), so B2 cannot be run
against a model here. This module is what makes it runnable anyway, and its
limits are the first thing to state:

* It **is** a real :class:`~ledgerloop.llm.client.Provider`. Its answers go
  through the same prompt, the same content-hash cache, the same budget, the
  same schema validation and the same cost ledger as a live provider's. Every
  number in B2's cost, cache and failure columns is measured machinery.
* It **is not** a language model, and no row it produces is a claim about one.
  Its precision and recall are properties of the rule below, not of any model.

WHAT THE B2 ROW STILL DEMONSTRATES
----------------------------------
The architectural point, which does not depend on which reasoner answered:

* output asserted with no ``verify_arithmetic`` behind it is asserted wrong as
  readily as right, and the report prices that in rupees;
* an id the answer invents becomes a false positive rather than a refusal,
  because B2 has no grounding gate to catch it;
* the token cost scales with the **corpus**, not with the residual, because the
  whole statement goes into every prompt.

THE RULE, STATED IN FULL SO NOBODY HAS TO GUESS
------------------------------------------------
It reads **only the prompt it is handed** -- the same text a model would see.
It never opens a source file and never sees ground truth; there is a test
asserting it takes no argument but the prompt string. Given that text it:

1. groups the payments in the prompt by the settlement id each one declares;
2. sums each group's payment amounts -- a **gross** figure, because the prompt
   states no fee and neither would a model reading it;
3. picks, for each group, the bank credit whose amount is nearest that sum, and
   asserts every payment in the group against it;
4. takes the argmax with no uniqueness check and no margin, and lets two groups
   claim the same credit if that is what nearest-amount says.

Steps 2 and 4 are where it goes wrong, and they are the two things the
deterministic ladder does differently: T2 inverts gross into net through the
same conserving allocation the generator used, and refuses when two subsets
both fit. This reasoner does neither, which is what B2 means.

It **does not invent record ids.** A real model does; the counters in
:class:`~ledgerloop.eval.llm_baseline.LLMBaselineArtifact` exist to catch that
and will read zero here. A zero from a reasoner incapable of the failure is not
evidence that the failure does not happen, and the report says so rather than
letting the zero look like a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ledgerloop.llm.client import Completion

__all__ = ["OFFLINE_PROVIDER_NAME", "OfflineReasoner"]

OFFLINE_PROVIDER_NAME = "offline-standin"

#: ``PAY-00001: ₹14,545.00 captured 2026-03-02, settlement SETL-0001, order ...``
_PAYMENT = re.compile(
    r"^(?P<id>[A-Za-z0-9\-]+): (?P<amount>[₹0-9,.]+) captured "
    r"(?P<date>\d{4}-\d{2}-\d{2}), settlement (?P<settlement>[^,]+),"
)

#: ``BNK-00002: ₹1,71,336.23 on 2026-03-18, narration '...'``
_CREDIT = re.compile(
    r"^(?P<id>[A-Za-z0-9\-]+): (?P<amount>[₹0-9,.]+) on (?P<date>\d{4}-\d{2}-\d{2}), "
    r"narration "
)


def _paise(rendered: str) -> int:
    """``₹1,71,336.23`` -> ``17133623``. Integer paise, never a float rupee."""
    digits = rendered.replace("₹", "").replace(",", "").strip()
    whole, _, fraction = digits.partition(".")
    fraction = (fraction + "00")[:2]
    return int(whole or "0") * 100 + int(fraction or "0")


@dataclass
class _Payment:
    payment_id: str
    settlement_id: str
    amount_minor: int


@dataclass
class _Credit:
    txn_id: str
    amount_minor: int


@dataclass
class OfflineReasoner:
    """Answers a B2 prompt by nearest-amount, from the prompt alone.

    ``prompts`` records everything it was asked, so a test can assert it was
    handed the production prompt rather than a convenience shape, and that it
    never received anything else.
    """

    name: str = OFFLINE_PROVIDER_NAME
    prompts: list[str] = field(default_factory=list)
    prompt_tokens_per_char: float = 0.25
    """Tokens per prompt character, for the ledger.

    Roughly four characters to a token, which is the usual English ratio and is
    what a real tokeniser would return within a few per cent on text this
    repetitive. It is an **estimate and is labelled as one** wherever the number
    is reported: this provider has no tokeniser, and inventing a precise-looking
    count would be worse than an honest approximation.
    """

    def complete(self, prompt: str, *, timeout_s: float) -> Completion:
        del timeout_s
        self.prompts.append(prompt)
        payments, credits = _parse(prompt)
        links = _nearest_amount_links(payments, credits)
        text = json.dumps({"links": links}, separators=(",", ":"), sort_keys=True)
        return Completion(
            text=text,
            prompt_tokens=int(len(prompt) * self.prompt_tokens_per_char),
            completion_tokens=int(len(text) * self.prompt_tokens_per_char),
            latency_ms=0,
            provider=self.name,
        )


def _parse(prompt: str) -> tuple[list[_Payment], list[_Credit]]:
    """Pull the two record lists out of the prompt text and nothing else."""
    payments: list[_Payment] = []
    credits: list[_Credit] = []
    section = ""
    for raw in prompt.splitlines():
        line = raw.strip()
        if line.startswith("Payments:"):
            section = "payments"
            continue
        if line.startswith("Bank credits:"):
            section = "credits"
            continue
        if section == "payments":
            found = _PAYMENT.match(line)
            if found is not None:
                payments.append(
                    _Payment(
                        payment_id=found["id"],
                        settlement_id=found["settlement"].strip(),
                        amount_minor=_paise(found["amount"]),
                    )
                )
        elif section == "credits":
            found = _CREDIT.match(line)
            if found is not None:
                credits.append(
                    _Credit(txn_id=found["id"], amount_minor=_paise(found["amount"]))
                )
    return payments, credits


def _nearest_amount_links(
    payments: list[_Payment], credits: list[_Credit]
) -> list[dict[str, object]]:
    """Group by declared settlement, sum gross, take the nearest credit.

    No uniqueness check and no arithmetic inversion -- see the module docstring.
    The confidence is a fixed 0.8: a self-reported number that nothing measured,
    which is exactly how much weight B2 gives it (none, because B2 has no
    calibrator) and exactly the overconfidence PLAN.md §7.4 warns about.
    """
    if not credits:
        return []
    groups: dict[str, list[_Payment]] = {}
    for payment in payments:
        groups.setdefault(payment.settlement_id, []).append(payment)

    links: list[dict[str, object]] = []
    for settlement_id in sorted(groups):
        members = groups[settlement_id]
        if settlement_id.lower() in {"none", ""}:
            continue
        total = sum(member.amount_minor for member in members)
        best = min(
            credits, key=lambda credit: (abs(credit.amount_minor - total), credit.txn_id)
        )
        links.extend(
            {
                "payment_id": member.payment_id,
                "bank_txn_id": best.txn_id,
                "confidence": 0.8,
            }
            for member in members
        )
    return links
