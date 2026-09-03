"""The same run, in words a finance person uses.

WHY THIS MODULE EXISTS
----------------------
:mod:`ledgerloop.ui.views` shapes the run for someone who already knows what a
tier, a Wilson interval and a residual pass are. This module shapes the *same
stored run* for someone who does not, and who wants four answers in five
seconds: how much money was reconciled, how many payments were matched, what
needs looking at, and whether anything was matched wrongly.

**It translates; it never computes.** Every number here is read off the run
record that :mod:`ledgerloop.agent.store` loaded, exactly as ``views.py`` does.
Nothing is re-derived, no metric is recalculated, and there is no second
definition of "matched" anywhere in this file. Where a plain word maps onto a
technical one the mapping is written down once, in :data:`GLOSSARY`, so the
dashboard and the report cannot drift apart.

WHAT IT REFUSES TO DO
---------------------
* **It does not invent per-transaction detail the run does not hold.** The run
  store keeps decisions (which payment, which bank row, which stage, what
  confidence, whether the arithmetic closed) and exceptions (amount, plain-English
  cause, suggested action, evidence chain). It does **not** keep a merchant name
  or a rupee figure per matched link, so no screen shows one. A column of
  plausible-looking blanks would be worse than an honest, narrower table.
* **It does not soften a refusal into a match.** "Needs review" and "not matched"
  stay separate from "matched" here for the same reason ``views.OUTCOME_HELP``
  keeps the four outcomes apart: a referral is not a match.
* **It does not overstate precision.** :func:`safety_note` says *zero incorrect
  matches were found*, which is what was measured. Whether that clears a 99%
  target at this sample size is a statistical ruling, and it stays in the
  technical report with its interval, where it can be read properly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ledgerloop.agent.store import StoredRun
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, Tier
from ledgerloop.models.recon_exception import ReconException
from ledgerloop.money import format_minor
from ledgerloop.ui.views import Headline, headline

__all__ = [
    "GLOSSARY",
    "REPORT_NAMES",
    "STAGES",
    "AssistantActivity",
    "AttentionItem",
    "Bucket",
    "GlossaryEntry",
    "JourneyStep",
    "MatchStory",
    "Snapshot",
    "Status",
    "assistant_activity",
    "attention_items",
    "attention_items_from",
    "buckets",
    "confidence_word",
    "glossary",
    "journey",
    "match_story",
    "match_story_from",
    "report_labels",
    "safety_note",
    "snapshot",
    "stage_of",
    "status_of",
    "transaction_rows",
    "transaction_rows_from",
]


@dataclass(frozen=True)
class _Stage:
    """One rung of the ladder, named for a reader who has never seen a tier."""

    plain: str
    """What to call it on screen."""

    because: str
    """What this stage required before it would commit, in one sentence.

    True by construction rather than per-record: it describes the rule the stage
    applies, which is why it can be stated without inspecting the record. The
    per-record facts -- which two records, what confidence, whether the
    arithmetic closed -- come off the decision itself.
    """


#: Tier -> plain name and plain reason. The only place the mapping is written.
STAGES: dict[Tier, _Stage] = {
    Tier.T0_EXACT: _Stage(
        "Exact reference match",
        "The payment's own reference appears on the bank transaction, and the "
        "amounts agree exactly.",
    ),
    Tier.T1_TOLERANCE: _Stage(
        "Reference match, small fee difference",
        "The reference matches, and the amounts agree once the payment "
        "processor's fee is allowed for.",
    ),
    Tier.T2_AGGREGATION: _Stage(
        "Grouped payout",
        "Several payments were paid out together as one bank credit, and this "
        "payment is one of them. Only one grouping fitted the amount.",
    ),
    Tier.T3_FUZZY: _Stage(
        "Matched on name and amount",
        "The bank transaction carried no reference, so the business name and "
        "the exact amount were used instead.",
    ),
    Tier.T4_GRAPH: _Stage(
        "Matched from surrounding evidence",
        "Everything else about this batch was already settled, and only one "
        "possibility was left.",
    ),
    Tier.T5_LLM: _Stage(
        "Suggested by the assistant, then checked",
        "An AI assistant suggested this link. It was accepted only after the "
        "amounts were re-checked against the source documents.",
    ),
}

#: A fallback that says what it is rather than guessing a name.
_UNKNOWN_STAGE = _Stage("Matched", "Recorded by the reconciliation, without a named stage.")


def stage_of(tier: Tier) -> _Stage:
    return STAGES.get(tier, _UNKNOWN_STAGE)


@dataclass(frozen=True)
class GlossaryEntry:
    """A technical term, what it means in plain words, and this run's value."""

    term: str
    plain: str
    value: str


#: The translation table, shown beside the technical report so the two
#: vocabularies are visibly the same numbers rather than two sets of claims.
GLOSSARY: tuple[tuple[str, str], ...] = (
    ("Precision", "Of the matches LedgerLoop made, how many were correct."),
    (
        "Recall",
        "Of the real payment-to-bank connections that exist, how many LedgerLoop found.",
    ),
    ("Match rate", "The share of all records that ended up resolved."),
    ("False positives", "Matches LedgerLoop made that were wrong."),
    (
        "Exception recall",
        "Of the problems that were really there, how many reached the review queue.",
    ),
    (
        "Unmatchable",
        "Records that cannot be resolved from the three source files at all. "
        "A real floor, not a failure.",
    ),
    (
        "Wilson interval",
        "The range the true figure is likely to sit in, given how few items were "
        "measured. A perfect score on a small sample is still uncertain.",
    ),
)


@dataclass(frozen=True)
class Snapshot:
    """The four answers the first screen exists to give."""

    matched: int
    """Links the system committed without a human."""

    needs_attention: int
    """Items in the review queue -- the controller's actual workday."""

    not_matched: int
    """Real connections the system did not find. Left for a person."""

    incorrect: int
    """Committed links that were wrong. The number that must stay zero."""

    reconciled: str
    outstanding: str
    incorrect_cost: str
    unmatchable: int
    unmatchable_impact: str
    records: int
    referred: int
    """Found and deliberately not committed. Not a miss -- a referral."""

    checked: int | None
    """Transactions a perfect system could have reconciled from these files.

    The **match-rate denominator**, taken from the run's stored interval, so the
    three figures below it are one unit and genuinely add up: ``checked`` equals
    ``resolved + unresolved``. ``records`` is a larger number -- everything read,
    including rows nothing could ever resolve -- and quoting it beside
    ``resolved`` would invent a failure rate the run never measured.

    ``None`` on a run recorded before intervals were stored. The screen falls
    back to ``records`` and says which it is showing rather than guessing.
    """

    resolved: int | None
    """How many of ``checked`` ended up reconciled."""

    unresolved: int | None
    """How many did not, and are left for a person."""

    @property
    def is_clean(self) -> bool:
        return self.incorrect == 0

    @property
    def counts_add_up(self) -> bool:
        """Whether the same-unit triple is available and consistent."""
        if self.checked is None or self.resolved is None or self.unresolved is None:
            return False
        return self.resolved + self.unresolved == self.checked


def snapshot(run: StoredRun) -> Snapshot:
    """The overview numbers, read straight off the stored run.

    ``matched`` is the count of **committed decisions**, not of correct ones, and
    ``incorrect`` is how many of those were wrong. Presenting them that way round
    is deliberate: it lets a reader see the whole claim -- *this many were
    committed, this many of them were mistakes* -- instead of a single number
    that has already had its errors quietly removed.
    """
    view = headline(run)
    interval = run.metrics.get("intervals", {}).get("match_rate_interval") or {}
    checked = interval.get("trials")
    resolved = interval.get("successes")
    return Snapshot(
        checked=int(checked) if checked is not None else None,
        resolved=int(resolved) if resolved is not None else None,
        unresolved=(
            int(checked) - int(resolved)
            if checked is not None and resolved is not None
            else None
        ),
        matched=view.auto_matched,
        needs_attention=view.queue_size,
        not_matched=view.false_negatives,
        incorrect=view.false_positives,
        reconciled=view.reconciled,
        outstanding=view.outstanding,
        incorrect_cost=view.false_positive_cost,
        unmatchable=view.unmatchable_count,
        unmatchable_impact=view.unmatchable_impact,
        records=view.records,
        referred=view.needs_review,
    )


@dataclass(frozen=True)
class Bucket:
    """One column of the "where did everything go" picture."""

    icon: str
    label: str
    count: int
    note: str
    tone: str


def buckets(run: StoredRun) -> list[Bucket]:
    """Matched / needs attention / not matched, as three plain columns.

    Three and not one. Collapsing them would hide the distinction the whole
    system is built on: an item left for a person is a *decision*, not a failure.
    """
    view = snapshot(run)
    return [
        Bucket(
            "check",
            "Matched",
            view.matched,
            "Enough evidence to be sure. Committed automatically.",
            "good",
        ),
        Bucket(
            "review",
            "Needs attention",
            view.needs_attention,
            "Something is wrong, or the evidence was not strong enough. "
            "Sent to a person with a reason.",
            "warn",
        ),
        Bucket(
            "open",
            "Not matched",
            view.not_matched,
            "No safe match was found. Nothing was guessed.",
            "muted",
        ),
    ]


def safety_note(run: StoredRun) -> tuple[str, str]:
    """The headline claim in plain words, and only as strongly as measured.

    Returns the number and the sentence. When something *was* matched wrongly
    the sentence says so and prices it -- a safety banner that only ever
    congratulates is not a safety banner.
    """
    view = snapshot(run)
    if view.is_clean:
        return (
            "0 incorrect matches",
            "When the evidence is not strong enough, LedgerLoop leaves the "
            "transaction for a person instead of guessing.",
        )
    return (
        f"{view.incorrect} incorrect matches",
        f"These were committed and should not have been, costing "
        f"{view.incorrect_cost}. Every one is listed in the technical report.",
    )



#: Split name -> what to call that report on screen.
#:
#: The run ids the store writes (`t0t4-test-42`, `ui-demo`) encode a ladder, a
#: split and a seed, which is exactly right for reproducing a figure and exactly
#: wrong as the first thing a reader sees. The split is the only part of that
#: identity a non-technical reader has any use for, so it is the part that
#: survives. The full id stays in the sidebar's details expander.
REPORT_NAMES: dict[str, str] = {
    "dev": "Demo report",
    "test": "Test report",
    "calibration": "Calibration report",
    "train": "Training report",
    "scale": "Large-scale report",
}


def report_labels(runs: Sequence[StoredRun]) -> dict[str, str]:
    """A friendly name per run id, unique within the list.

    Two runs of the same split would otherwise share a label and the picker
    would show the same words twice, so repeats are numbered in list order --
    deterministic, and stable between reloads.
    """
    used: dict[str, int] = {}
    labels: dict[str, str] = {}
    for run in runs:
        split = str(run.summary.get("dataset", {}).get("split", ""))
        base = REPORT_NAMES.get(split, "Reconciliation report")
        used[base] = used.get(base, 0) + 1
        labels[run.run_id] = base if used[base] == 1 else f"{base} ({used[base]})"
    return labels


@dataclass(frozen=True)
class Status:
    """The one-line verdict the first viewport leads with."""

    icon: str
    title: str
    body: str
    tone: str


def status_of(run: StoredRun) -> Status:
    """Did this reconciliation finish safely, and what does that mean?

    Three states, and the ordering matters: a wrong match outranks a queue,
    because a queue is work and a wrong match is a mistake already in the books.
    """
    view = snapshot(run)
    if not view.is_clean:
        return Status(
            "!",
            f"{view.incorrect} match(es) were wrong",
            f"{view.incorrect} transaction(s) were reconciled that should not have "
            f"been, involving {view.incorrect_cost}. Check these before relying on "
            "this report.",
            "bad",
        )
    if view.needs_attention:
        # `unresolved` where the run stored it, so this sentence agrees with the
        # "Need review" figure beside it. The queue is a different, larger count
        # -- it also holds records outside the reconcilable set -- and naming
        # both as "needs review" on one screen is what made 29 and 67 look like
        # a contradiction.
        waiting = view.unresolved if view.unresolved is not None else view.needs_attention
        return Status(
            "OK",
            "Reconciliation completed safely",
            f"Nothing was matched incorrectly. {waiting:,} transaction(s) could "
            "not be settled from the evidence available and are waiting for a "
            "person.",
            "good",
        )
    return Status(
        "OK",
        "Reconciliation completed safely",
        "Nothing was matched incorrectly, and nothing needs a person.",
        "good",
    )


@dataclass(frozen=True)
class JourneyStep:
    """One stop on the picture of what LedgerLoop actually does."""

    label: str
    note: str
    tone: str


def journey(run: StoredRun) -> tuple[list[JourneyStep], list[JourneyStep]]:
    """Two paths through the same three files: settled, and not settled.

    Drawn side by side because the second one is the product's argument. A
    reconciliation tool that only showed the happy path would be hiding the
    behaviour that makes this one trustworthy.
    """
    view = snapshot(run)
    resolved = view.resolved if view.resolved is not None else view.matched
    settled = [
        JourneyStep("Payment", "what your system says was taken", "muted"),
        JourneyStep("Bank transaction", "what actually arrived", "muted"),
        JourneyStep("Settlement record", "what the processor says it paid", "muted"),
        JourneyStep("Reconciled", f"{resolved:,} of these", "good"),
    ]
    unresolved = view.unresolved if view.unresolved is not None else view.needs_attention
    stuck = [
        JourneyStep("Payment", "what your system says was taken", "muted"),
        JourneyStep("Bank transaction", "no reference, or two possible matches", "muted"),
        JourneyStep("Not enough evidence", "nothing was guessed", "warn"),
        JourneyStep("Sent for review", f"{unresolved:,} of these", "warn"),
    ]
    return settled, stuck

@dataclass(frozen=True)
class AttentionItem:
    """One queue entry, led by what it costs and what to do about it."""

    amount: str
    amount_minor: int
    subject: str
    found: str
    """Plain English, written by the exception classifier from its evidence."""

    action: str
    severity: str
    technical_class: str
    exception_id: str
    evidence_count: int
    agent_may_resolve: bool


def attention_items_from(
    exceptions: Sequence[ReconException],
) -> list[AttentionItem]:
    """The queue, from the exceptions themselves.

    Split out from :func:`attention_items` so uploaded files -- which produce
    exceptions but no stored run -- go through exactly this code and not a
    second copy of it.
    """
    items = sorted(exceptions, key=lambda item: (-item.impact_minor, item.exception_id))
    return [
        AttentionItem(
            amount=format_minor(item.impact_minor),
            amount_minor=item.impact_minor,
            subject=item.involved_refs[0].record_id if item.involved_refs else "",
            found=item.root_cause,
            action=item.suggested_action,
            severity=item.severity.value.title(),
            technical_class=item.exception_class.value,
            exception_id=item.exception_id,
            evidence_count=len(item.evidence),
            agent_may_resolve=item.resolvable_by_agent,
        )
        for item in items
    ]


def attention_items(run: StoredRun) -> list[AttentionItem]:
    """The review queue, biggest rupee impact first.

    Sorted by money for the same reason ``views.exception_rows`` is: one large
    payout matters more than two hundred one-paise drifts, and any other order
    buries that.

    ``found`` and ``action`` are the classifier's own sentences. They are already
    plain English and grounded in the evidence chain, so they are passed through
    untouched rather than paraphrased -- a paraphrase here would be this module
    inventing a claim the run did not make.
    """
    return attention_items_from(run.exceptions)


def confidence_word(probability: float, *, verified: bool) -> str:
    """A probability as a word, and the arithmetic check folded in.

    Deliberately coarse. A controller does not act differently at 0.94 than at
    0.96, and printing four decimals to a non-technical reader invites a
    precision that the calibration section of the report is careful *not* to
    claim. The exact figure stays one expander away.
    """
    if not verified:
        return "Not confirmed"
    if probability >= 0.99:
        return "Very strong evidence"
    if probability >= 0.9:
        return "Strong evidence"
    if probability >= 0.7:
        return "Reasonable evidence"
    return "Weak evidence"


#: Plain labels for the policy's four outcomes, in the same order the report
#: uses. The technical name travels beside each one in the table.
_OUTCOME_PLAIN: dict[str, tuple[str, str]] = {
    DecisionOutcome.AUTO_MATCHED.value: ("Matched", "good"),
    DecisionOutcome.NEEDS_REVIEW.value: ("Needs review", "warn"),
    DecisionOutcome.EXCEPTION.value: ("Sent to queue", "warn"),
    DecisionOutcome.REJECTED.value: ("Refused", "muted"),
}


def transaction_rows_from(
    decisions: Sequence[MatchDecision], *, status: str | None = None
) -> list[dict[str, object]]:
    """Every decision as a scannable row, from the decisions themselves.

    Uploaded files produce decisions without a stored run, and they deserve the
    same table rather than a second one written for them.
    """
    rows: list[dict[str, object]] = []
    for decision in decisions:
        label, tone = _OUTCOME_PLAIN.get(decision.outcome.value, ("Recorded", "muted"))
        if status is not None and label != status:
            continue
        stage = stage_of(decision.tier)
        rows.append(
            {
                "Status": label,
                "tone": tone,
                "Payment": decision.source_ref.record_id,
                "Bank transaction": decision.target_ref.record_id,
                "How it was matched": stage.plain,
                "Confidence": confidence_word(
                    decision.calibrated_p, verified=decision.arithmetic_verified
                ),
                "record_key": decision.source_ref.key,
                "tier": decision.tier.name,
                "calibrated_p": decision.calibrated_p,
            }
        )
    return rows


def transaction_rows(run: StoredRun, *, status: str | None = None) -> list[dict[str, object]]:
    """Every decision the run committed, as a table a person can scan.

    Columns are what the run actually holds. There is **no amount and no
    merchant column**, because the run store keeps neither per decision -- see
    the module docstring. The money lives on the overview and on the review
    queue, where the figures are real.

    ``status`` filters by the plain label. Filtering narrows what is shown and
    nothing else; the overview's counts are read from the run's own summary and
    cannot be moved by it.
    """
    return transaction_rows_from(run.decisions, status=status)


def transaction_search(rows: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    """Free-text filter over the identifiers a person would actually paste in.

    Matches a payment id, a bank id or a stage name, case-insensitively. Kept
    separate from :func:`transaction_rows` so the filter is testable without a
    run and cannot accidentally change what the table contains.
    """
    needle = query.strip().lower()
    if not needle:
        return rows
    return [
        row
        for row in rows
        if needle
        in " ".join(
            str(row.get(field, ""))
            for field in ("Payment", "Bank transaction", "How it was matched", "Status")
        ).lower()
    ]


__all__ += ["transaction_search"]


@dataclass(frozen=True)
class MatchStory:
    """Why one record ended where it did, told without jargon."""

    record_key: str
    matched: bool
    headline: str
    partner: str
    """The **other** end of the link.

    Whichever of the two refs is not the record being looked at. A decision
    names a source and a target, and a record can be either -- printing the
    target unconditionally showed "BNK-00002 to BNK-00002" whenever a bank row
    was the one selected.
    """

    stage: str
    reasons: tuple[str, ...]
    confidence: str
    caveat: str
    """Empty when there is nothing to warn about."""

    technical: tuple[tuple[str, str], ...]
    """Label/value pairs for the expander. The jargon lives here and only here."""


def match_story_from(
    decisions: Sequence[MatchDecision],
    exceptions: Sequence[ReconException],
    record_key: str,
) -> MatchStory:
    """The plain account of one record, from decisions and exceptions.

    ``decisions`` must already be the ones naming ``record_key``; the caller
    knows how to select them and a stored run and an uploaded result select them
    differently. Everything after that selection is identical, which is why this
    is one function rather than two.
    """
    naming = [
        item
        for item in exceptions
        if any(ref.key == record_key for ref in item.involved_refs)
    ]

    if not decisions:
        cause = (
            naming[0].root_cause
            if naming
            else (
                "No stage found evidence strong enough to link this record, and "
                "nothing was guessed. It may also be one of the records that "
                "cannot be resolved from the three source files at all."
            )
        )
        return MatchStory(
            record_key=record_key,
            matched=False,
            headline="Not matched",
            partner="",
            stage="",
            reasons=(cause,),
            confidence="",
            caveat="",
            technical=(
                ("Decisions recorded", "0"),
                ("Exceptions naming it", str(len(naming))),
            ),
        )

    decision = decisions[-1]
    stage = stage_of(decision.tier)
    committed = decision.outcome is DecisionOutcome.AUTO_MATCHED

    reasons = [stage.because]
    if decision.arithmetic_verified:
        reasons.append(
            "The amounts were re-added from the original ledger, settlement and "
            "bank files, and they agree."
        )
    else:
        reasons.append(
            "The amounts could not be confirmed against the source files, which "
            "is why this was not committed automatically."
        )

    caveat = ""
    if not committed:
        caveat = (
            "This was found but deliberately not committed. LedgerLoop hands a "
            "case to a person rather than guess between two possibilities."
        )

    other = (
        decision.source_ref
        if decision.target_ref.key == record_key
        else decision.target_ref
    )
    return MatchStory(
        record_key=record_key,
        matched=committed,
        headline="Match confirmed" if committed else "Left for a person",
        partner=other.record_id,
        stage=stage.plain,
        reasons=tuple(reasons),
        confidence=confidence_word(
            decision.calibrated_p, verified=decision.arithmetic_verified
        ),
        caveat=caveat,
        technical=(
            ("Tier", decision.tier.name),
            ("Outcome", decision.outcome.value),
            ("Calibrated probability", f"{decision.calibrated_p:.4f}"),
            ("Arithmetic verified", str(decision.arithmetic_verified)),
            ("Link type", decision.link_type.value),
            ("Policy reason", decision.reason),
            ("Decision id", decision.decision_id),
        ),
    )


def match_story(run: StoredRun, record_key: str) -> MatchStory:
    """The plain-English account of one record.

    Assembled by *selecting* from the stored decisions and the stored exceptions
    -- the same discipline ``views.trace_record`` follows. The reasons are the
    stage's own rule plus two facts the decision itself records: whether the
    arithmetic closed against the source documents, and how confident the
    calibrated model was. Nothing is inferred about the record beyond what it
    stored.
    """
    return match_story_from(
        run.decisions_for(record_key), run.exceptions, record_key
    )


@dataclass(frozen=True)
class AssistantActivity:
    """What the AI assistant actually did on this run, if anything.

    Every field is read off the stored run. ``used`` is **calls made**, never a
    credential existing -- the distinction the whole gate exists to protect.

    The refusal counts matter more than the acceptances, and they are the reason
    this is worth showing at all. A model that proposes twelve links and has
    nine of them thrown out for citing records it was never given is a model
    being held to something; a dashboard that showed only the three would be
    describing a different system.
    """

    available: bool
    """Whether a model was wired up for this run at all."""

    calls: int
    tokens: int
    cost_inr: float
    cache_hits: int
    accepted: int
    """Proposals that survived every gate and became candidates."""

    refused_ungrounded: int
    """Thrown out for citing a record that was not in the evidence pack."""

    demoted_unverified: int
    """The money did not close under ``verify_arithmetic``. Demoted, not
    dropped: it becomes a candidate for a person, because "the model suggested
    this and the arithmetic disagrees" is information a controller wants."""

    prose_rewritten: int
    """Exception explanations reworded. The class, the severity and the rupee
    figure were already decided and are never sent back for revision."""

    @property
    def used(self) -> bool:
        return self.calls > 0

    @property
    def refused(self) -> int:
        return self.refused_ungrounded + self.demoted_unverified

    @property
    def did_anything_visible(self) -> bool:
        return bool(self.accepted or self.refused or self.prose_rewritten)


def assistant_activity(run: StoredRun) -> AssistantActivity:
    """Read the model's contribution off the run record. Nothing recomputed."""
    llm = run.summary.get("llm", {})
    return AssistantActivity(
        available=bool(llm.get("available", False)),
        calls=int(llm.get("calls", 0)),
        tokens=int(llm.get("total_tokens", 0)),
        cost_inr=float(llm.get("equivalent_paid_cost_inr", 0.0)),
        cache_hits=int(llm.get("cache_hits", 0)),
        accepted=int(llm.get("accepted", 0)),
        refused_ungrounded=int(llm.get("rejected_ungrounded", 0)),
        demoted_unverified=int(llm.get("rejected_unverified", 0)),
        prose_rewritten=int(llm.get("prose_rewritten", 0)),
    )


def glossary(view: Headline) -> list[GlossaryEntry]:
    """The technical terms with their plain meaning and this run's figure.

    Placed beside the numbers rather than in a separate page, so a reader who
    meets "precision" for the first time can see immediately which number on the
    screen it names.
    """
    values = {
        "Precision": f"{view.precision:.4f}",
        "Recall": f"{view.recall:.4f}",
        "Match rate": f"{view.match_rate:.4f}",
        "False positives": str(view.false_positives),
        "Exception recall": f"{view.exception_recall:.4f}",
        "Unmatchable": f"{view.unmatchable_count} records ({view.unmatchable_impact})",
        "Wilson interval": (
            f"precision {view.precision_ci[0]:.4f} to {view.precision_ci[1]:.4f}"
        ),
    }
    return [
        GlossaryEntry(term=term, plain=plain, value=values.get(term, ""))
        for term, plain in GLOSSARY
    ]
