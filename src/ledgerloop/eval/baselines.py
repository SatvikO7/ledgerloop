"""B0 -- exact join on UTR. The "why not just SQL" answer.

PLAN.md §9.2 opens the baseline table with the question every reviewer asks
first: the bank narration contains a UTR and the PSP report publishes one, so
why is this not a ``JOIN``? B0 is that join, implemented honestly and scored on
the same test set as everything else. Its number is the floor the rest of the
system has to beat, and the specific ways it fails are the specification for
tiers T1-T5.

WHAT IT DOES
------------
1. Extract a UTR from each bank narration with one regex.
2. Join it to the settlement publishing that UTR.
3. Assert that every payment in that settlement was credited by that bank row.

Step 3 of :doc:`IMPLEMENTATION_PLAN` builds the real ingest layer. B0
deliberately does not wait for it and does not use it: a baseline that shares
the system's normalisation is measuring the system twice. These readers are as
naive as the baseline they serve, which is the point.

B0 IS GIVEN EVERY ADVANTAGE
---------------------------
The regex knows the generator's exact UTR shape, the reader is handed
well-formed files, and no tolerance or ambiguity check is applied that could
cause it to decline a join. A baseline built to lose proves nothing. This one is
built to do as well as an exact join possibly can, and it still cannot:

* **A07 MISSING_REFERENCE** strips the UTR from the narration -- nothing to
  join on, so every payment in that settlement is a false negative.
* **A05 DUPLICATE_CREDIT** repeats a UTR on a second bank row. The join has no
  notion of "already credited", so it credits the whole batch twice and every
  link on the duplicate is a false positive.
* **A09 SPLIT_PAYOUT** delivers one settlement as two credits. The join emits
  the full cross product -- every payment against both rows -- where the truth
  partitions the payments between them.

Those three are not edge cases bolted on to embarrass the baseline; they are
three of the eleven generated anomaly classes, at their configured prevalence.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ledgerloop.eval.metrics import PredictedLink
from ledgerloop.generator.emitters import BANK_FILE, PSP_FILE
from ledgerloop.models.refs import bank_ref, payment_ref
from ledgerloop.money import parse_minor_units

__all__ = ["B0_DESCRIPTION", "B0_NAME", "UTR_PATTERN", "BaselineRun", "run_b0"]

B0_NAME = "B0"
B0_DESCRIPTION = "Exact join on UTR (narration regex -> settlement.utr -> its payments)"

#: The UTR as the generator writes it: ``UTR`` + ``YYYYMMDD`` + a 5-digit tail.
#:
#: Left at ``{8,}`` rather than pinned to the exact 13 digits so a near-variant
#: still joins. Being generous here is deliberate -- see the module docstring.
UTR_PATTERN = re.compile(r"UTR\d{8,}")


@dataclass(frozen=True)
class _Settlement:
    settlement_id: str
    utr: str | None
    payment_amounts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _BankCredit:
    txn_id: str
    narration: str
    credit_minor: int


@dataclass(frozen=True)
class BaselineRun:
    """One baseline's output, plus the diagnostics the report explains it with."""

    name: str
    description: str
    predictions: tuple[PredictedLink, ...]
    wall_clock_ms: int

    credits_seen: int
    credits_with_utr: int
    credits_joined: int
    settlements_seen: int
    settlements_joined: int

    @property
    def credits_without_utr(self) -> int:
        """Credits carrying no recoverable reference. A07, plus the noise rows."""
        return self.credits_seen - self.credits_with_utr

    @property
    def settlements_unjoined(self) -> int:
        """Settlements whose UTR appeared in no narration. Pure false negatives."""
        return self.settlements_seen - self.settlements_joined


def _read_settlements(directory: Path) -> list[_Settlement]:
    """Read the PSP report. Nested JSON: settlements outside, payments inside."""
    with (directory / PSP_FILE).open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)

    settlements: list[_Settlement] = []
    for batch in payload["settlements"]:
        utr = batch.get("utr")
        settlements.append(
            _Settlement(
                settlement_id=str(batch["settlement_id"]),
                utr=str(utr) if utr else None,
                payment_amounts=tuple(
                    (
                        str(payment["payment_id"]),
                        parse_minor_units(payment["amount_paise"], field="psp.amount_paise"),
                    )
                    for payment in batch["payments"]
                ),
            )
        )
    return settlements


def _read_bank_credits(directory: Path) -> list[_BankCredit]:
    """Read the bank statement, keeping only incoming money.

    Debits are excluded rather than joined and rejected. A settlement is money
    arriving; matching a payout batch to an outgoing payment is not a mistake an
    exact join would plausibly make, and letting B0 make it would inflate its
    false-positive count with an error no real implementation would commit.
    """
    with (directory / BANK_FILE).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    credits: list[_BankCredit] = []
    for row in rows:
        credit_minor = parse_minor_units(row["credit_paise"], field="bank.credit_paise")
        if credit_minor <= 0:
            continue
        credits.append(
            _BankCredit(
                txn_id=row["txn_id"],
                narration=row["narration"],
                credit_minor=credit_minor,
            )
        )
    return credits


def run_b0(directory: Path) -> BaselineRun:
    """Run the exact-join baseline over a generated dataset directory."""
    started_ns = time.perf_counter_ns()

    settlements = _read_settlements(directory)
    credits = _read_bank_credits(directory)

    # A UTR *should* identify one settlement. Nothing enforces that, so the
    # index holds a list: if two settlements ever shared one, the join would
    # return both, and silently keeping only the last would hide the collision
    # rather than let the metrics show it.
    by_utr: dict[str, list[_Settlement]] = {}
    for settlement in settlements:
        if settlement.utr is not None:
            by_utr.setdefault(settlement.utr, []).append(settlement)

    predictions: list[PredictedLink] = []
    credits_with_utr = 0
    credits_joined = 0
    joined_settlement_ids: set[str] = set()

    for credit in credits:
        found = UTR_PATTERN.search(credit.narration)
        if found is None:
            continue
        credits_with_utr += 1

        matches = by_utr.get(found.group(0))
        if not matches:
            continue
        credits_joined += 1

        for settlement in matches:
            joined_settlement_ids.add(settlement.settlement_id)
            for payment_id, amount_minor in settlement.payment_amounts:
                # The join asserts the payment's full gross amount. It has no
                # model of fees, and no way to allocate a credit across the
                # payments it carries -- which is why B0's reconciled-rupee
                # figure runs above the truth even where the links are right.
                predictions.append(
                    PredictedLink(
                        source_ref=payment_ref(payment_id),
                        target_ref=bank_ref(credit.txn_id),
                        amount_minor=amount_minor,
                    )
                )

    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
    return BaselineRun(
        name=B0_NAME,
        description=B0_DESCRIPTION,
        predictions=tuple(predictions),
        wall_clock_ms=int(elapsed_ms),
        credits_seen=len(credits),
        credits_with_utr=credits_with_utr,
        credits_joined=credits_joined,
        settlements_seen=len(settlements),
        settlements_joined=len(joined_settlement_ids),
    )
