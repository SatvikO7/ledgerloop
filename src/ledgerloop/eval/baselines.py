"""B0 and B1 -- the two deterministic baselines the system has to beat.

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

B1 -- EXACT + FUZZY, "THE TYPICAL HACKATHON SUBMISSION"
--------------------------------------------------------
PLAN.md §9.2's second row. B1 is B0's join plus the two things a reconciliation
script reaches for when the join misses: fuzzy recovery of a damaged reference,
and a nearest-amount match inside a date window.

It is built the way that script would be built, which means it is built to
*succeed* -- and the specific ways it then fails are the argument for the rest
of the ladder:

* It takes the **argmax** and never checks whether the runner-up was just as
  good. LedgerLoop's tiers refuse a decision point two candidates fit equally
  (T2's ``AMBIGUOUS_AGGREGATION``, T3's margin gate); B1 has nowhere to put an
  ambiguity, so it picks one.
* It has **no aggregation solver**, so a settlement paid out as two credits
  (A09) can only ever be half-explained -- and the nearest-amount stage will
  happily attach the whole batch to a tranche that carries part of it.
* Its amount comparison is against the settlement **net**, so a batch whose fee
  was misdeclared (A03) falls outside the band and a batch whose net collides
  with an unrelated one inside the band is matched to the wrong merchant.

**What B1 deliberately does not do: fuzzy-match merchant names.** There is no
merchant master among the three sources -- the ledger publishes ``MRCH_0001``
and the narration says ``RZRPAY SFTWR P L``, and the two strings share no
characters. Deriving one from the statement's own references is what T3 does
(ARCHITECTURE.md §6, decision 27), and handing it to a baseline would be
measuring LedgerLoop twice. Reading the generator's merchant vocabulary would
be worse: it is not one of the three inputs a real system gets.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from ledgerloop.eval.metrics import PredictedLink
from ledgerloop.generator.emitters import BANK_FILE, PSP_FILE
from ledgerloop.models.refs import bank_ref, payment_ref
from ledgerloop.money import parse_minor_units

__all__ = [
    "B0_DESCRIPTION",
    "B0_NAME",
    "B1_AMOUNT_BPS",
    "B1_DATE_WINDOW_DAYS",
    "B1_DESCRIPTION",
    "B1_FUZZY_THRESHOLD",
    "B1_NAME",
    "UTR_PATTERN",
    "BaselineRun",
    "run_b0",
    "run_b1",
]

B0_NAME = "B0"
B0_DESCRIPTION = "Exact join on UTR (narration regex -> settlement.utr -> its payments)"

B1_NAME = "B1"
B1_DESCRIPTION = "Exact join + fuzzy reference recovery + nearest-amount match"

#: RapidFuzz similarity a damaged reference must reach to be treated as a
#: settlement's UTR. 85 is the value a reconciliation script reaches for -- high
#: enough to look careful, low enough to admit a transposition. It is not tuned
#: on any split: a baseline whose threshold was optimised against the test set
#: would be a second system, not a floor.
B1_FUZZY_THRESHOLD = 85.0

#: Nearest-amount band, in basis points of the settlement net. 100 bps = 1%,
#: twice LedgerLoop's T1 band, because a script with no fee model has to absorb
#: the fee misdeclaration it cannot see.
B1_AMOUNT_BPS = 100

#: Date window for the nearest-amount stage, in days either side of the
#: settlement's declared date.
B1_DATE_WINDOW_DAYS = 5

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
    net_minor: int = 0
    settled_on: date | None = None


@dataclass(frozen=True)
class _BankCredit:
    txn_id: str
    narration: str
    credit_minor: int
    value_date: date | None = None


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
                net_minor=parse_minor_units(
                    batch.get("net_paise", 0), field="psp.net_paise"
                ),
                settled_on=_iso_date(batch.get("settled_on")),
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
                value_date=_day_first_date(row.get("value_date")),
            )
        )
    return credits


def _iso_date(value: object) -> date | None:
    """``2026-03-18`` -> a date. Anything else is absent, never a guess."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:  # pragma: no cover - the emitter always writes ISO
        return None


def _day_first_date(value: object) -> date | None:
    """``18/03/2026`` -> a date, assuming day-first.

    The baseline **assumes** the Indian convention rather than inferring it from
    the column, which is exactly what a reconciliation script does and exactly
    what ``ingest/dates.py`` refuses to do. On this corpus the assumption is
    correct, so B1 is not penalised for it -- the difference is that B1 could
    not have known.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError:  # pragma: no cover - the emitter always writes DD/MM/YYYY
        return None


def _exact_join(
    credits: list[_BankCredit], settlements: list[_Settlement]
) -> tuple[list[PredictedLink], set[str], set[str], int]:
    """B0's join, factored out so B1 provably starts from the same place.

    Returns the links, the credit ids it consumed, the settlement ids it
    reached, and how many credits carried a recoverable UTR. B1 layers on top
    of exactly this rather than reimplementing it, so any difference between the
    two rows is the fuzzy stages and nothing else.

    A UTR *should* identify one settlement. Nothing enforces that, so the index
    holds a list: if two settlements ever shared one, the join would return
    both, and silently keeping only the last would hide the collision rather
    than let the metrics show it.
    """
    by_utr: dict[str, list[_Settlement]] = {}
    for settlement in settlements:
        if settlement.utr is not None:
            by_utr.setdefault(settlement.utr, []).append(settlement)

    links: list[PredictedLink] = []
    joined_credits: set[str] = set()
    joined_settlements: set[str] = set()
    with_utr = 0

    for credit in credits:
        found = UTR_PATTERN.search(credit.narration)
        if found is None:
            continue
        with_utr += 1
        matches = by_utr.get(found.group(0))
        if not matches:
            continue
        joined_credits.add(credit.txn_id)
        for settlement in matches:
            joined_settlements.add(settlement.settlement_id)
            # The join asserts the payment's full gross amount. It has no model
            # of fees, and no way to allocate a credit across the payments it
            # carries -- which is why the reconciled-rupee figure for both
            # baselines runs above the truth even where the links are right.
            links.extend(
                PredictedLink(
                    source_ref=payment_ref(payment_id),
                    target_ref=bank_ref(credit.txn_id),
                    amount_minor=amount_minor,
                )
                for payment_id, amount_minor in settlement.payment_amounts
            )
    return links, joined_credits, joined_settlements, with_utr


def run_b0(directory: Path) -> BaselineRun:
    """Run the exact-join baseline over a generated dataset directory."""
    started_ns = time.perf_counter_ns()

    settlements = _read_settlements(directory)
    credits = _read_bank_credits(directory)
    links, joined_credits, joined_settlements, with_utr = _exact_join(credits, settlements)

    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
    return BaselineRun(
        name=B0_NAME,
        description=B0_DESCRIPTION,
        predictions=tuple(links),
        wall_clock_ms=int(elapsed_ms),
        credits_seen=len(credits),
        credits_with_utr=with_utr,
        credits_joined=len(joined_credits),
        settlements_seen=len(settlements),
        settlements_joined=len(joined_settlements),
    )


def _fuzzy_reference_stage(
    credits: list[_BankCredit],
    settlements: list[_Settlement],
    *,
    joined_credits: set[str],
    joined_settlements: set[str],
    threshold: float,
) -> list[tuple[_BankCredit, _Settlement, float]]:
    """Recover a damaged reference: the best UTR-shaped token above ``threshold``.

    Scored with ``fuzz.partial_ratio`` against each open settlement's UTR, over
    the alphanumeric runs in the narration. That is the shape of the code a
    reconciliation script actually contains -- and it has no way to notice that
    two references differing by a transposition are two different payouts,
    which is the point of measuring it.
    """
    open_settlements = [
        settlement
        for settlement in settlements
        if settlement.utr is not None
        and settlement.settlement_id not in joined_settlements
    ]
    proposals: list[tuple[_BankCredit, _Settlement, float]] = []
    for credit in credits:
        if credit.txn_id in joined_credits:
            continue
        tokens = [
            token
            for token in re.split(r"[^A-Za-z0-9]+", credit.narration.upper())
            if len(token) >= 8 and any(character.isdigit() for character in token)
        ]
        if not tokens:
            continue
        best: tuple[_Settlement, float] | None = None
        for settlement in open_settlements:
            utr = settlement.utr
            assert utr is not None  # filtered above
            score = max(float(fuzz.partial_ratio(token, utr)) for token in tokens)
            if score >= threshold and (best is None or score > best[1]):
                best = (settlement, score)
        if best is not None:
            proposals.append((credit, best[0], best[1]))
    return proposals


def _nearest_amount_stage(
    credits: list[_BankCredit],
    settlements: list[_Settlement],
    *,
    joined_credits: set[str],
    joined_settlements: set[str],
    amount_bps: int,
    date_window_days: int,
) -> list[tuple[_BankCredit, _Settlement, float]]:
    """Nearest net inside a band and a date window. Argmax, no ambiguity check.

    The absence of the check is the measurement. LedgerLoop's T2 counts the
    subsets that fit and refuses when two do; T3 requires the winner to beat the
    runner-up by a margin. This stage does neither, because the script it stands
    in for does neither -- it sorts by ``abs(delta)`` and takes the first row.
    """
    open_settlements = [
        settlement
        for settlement in settlements
        if settlement.settlement_id not in joined_settlements
        and settlement.net_minor > 0
    ]
    proposals: list[tuple[_BankCredit, _Settlement, float]] = []
    for credit in credits:
        if credit.txn_id in joined_credits:
            continue
        best: tuple[_Settlement, int] | None = None
        for settlement in open_settlements:
            band = max(settlement.net_minor * amount_bps // 10_000, 100)
            delta = abs(credit.credit_minor - settlement.net_minor)
            if delta > band:
                continue
            if (
                credit.value_date is not None
                and settlement.settled_on is not None
                and abs((credit.value_date - settlement.settled_on).days)
                > date_window_days
            ):
                continue
            if best is None or delta < best[1]:
                best = (settlement, delta)
        if best is not None:
            # The score is "how close", negated so the greedy commit below ranks
            # the tightest fits first. It is not a probability and is never
            # reported as one -- a baseline that produced a confidence would be
            # claiming the calibration this project spends a whole step earning.
            proposals.append((credit, best[0], -float(best[1])))
    return proposals


def _commit(
    proposals: list[tuple[_BankCredit, _Settlement, float]],
    *,
    links: list[PredictedLink],
    joined_credits: set[str],
    joined_settlements: set[str],
) -> None:
    """Greedy best-score-first commit, one settlement to one credit.

    Greedy rather than a cross product: it is the *charitable* reading of what a
    script does, and it keeps B1 from inflating its own false-positive count
    with an error a careless implementation would make but a typical one would
    not. Ties are broken on the record ids so the run is reproducible.
    """
    ordered = sorted(
        proposals, key=lambda item: (-item[2], item[0].txn_id, item[1].settlement_id)
    )
    for credit, settlement, _ in ordered:
        if credit.txn_id in joined_credits:
            continue
        if settlement.settlement_id in joined_settlements:
            continue
        joined_credits.add(credit.txn_id)
        joined_settlements.add(settlement.settlement_id)
        links.extend(
            PredictedLink(
                source_ref=payment_ref(payment_id),
                target_ref=bank_ref(credit.txn_id),
                amount_minor=amount_minor,
            )
            for payment_id, amount_minor in settlement.payment_amounts
        )


def run_b1(
    directory: Path,
    *,
    fuzzy_threshold: float = B1_FUZZY_THRESHOLD,
    amount_bps: int = B1_AMOUNT_BPS,
    date_window_days: int = B1_DATE_WINDOW_DAYS,
) -> BaselineRun:
    """Run the exact + fuzzy baseline over a generated dataset directory.

    Three stages, in the order a script would write them: the exact join, then
    fuzzy reference recovery over what it missed, then a nearest-amount match
    over what is still open. Each later stage sees only the residual of the
    earlier ones, so the same credit is never claimed twice by B1 itself --
    though the exact stage can still credit one batch to two rows, exactly as
    B0 does, because that failure belongs to the join rather than to the script
    wrapped around it.

    The parameters are arguments rather than constants so a test can drive the
    stages independently. They are **not** swept: the reported B1 row uses the
    module defaults on every split, and tuning a baseline against the test set
    is the way a baseline stops being one.
    """
    started_ns = time.perf_counter_ns()

    settlements = _read_settlements(directory)
    credits = _read_bank_credits(directory)
    links, joined_credits, joined_settlements, with_utr = _exact_join(credits, settlements)

    _commit(
        _fuzzy_reference_stage(
            credits,
            settlements,
            joined_credits=joined_credits,
            joined_settlements=joined_settlements,
            threshold=fuzzy_threshold,
        ),
        links=links,
        joined_credits=joined_credits,
        joined_settlements=joined_settlements,
    )
    _commit(
        _nearest_amount_stage(
            credits,
            settlements,
            joined_credits=joined_credits,
            joined_settlements=joined_settlements,
            amount_bps=amount_bps,
            date_window_days=date_window_days,
        ),
        links=links,
        joined_credits=joined_credits,
        joined_settlements=joined_settlements,
    )

    elapsed_ms = (time.perf_counter_ns() - started_ns) // 1_000_000
    return BaselineRun(
        name=B1_NAME,
        description=B1_DESCRIPTION,
        predictions=tuple(links),
        wall_clock_ms=int(elapsed_ms),
        credits_seen=len(credits),
        credits_with_utr=with_utr,
        credits_joined=len(joined_credits),
        settlements_seen=len(settlements),
        settlements_joined=len(joined_settlements),
    )
