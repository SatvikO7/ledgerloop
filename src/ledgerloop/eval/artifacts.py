"""The serialised results of the evaluation's expensive commands.

The models only. The code that *produces* them lives in
:mod:`ledgerloop.eval.ablation`, :mod:`ledgerloop.eval.sweep`,
:mod:`ledgerloop.eval.llm_baseline`, :mod:`ledgerloop.eval.comparison` and
:mod:`ledgerloop.eval.llm_report`, and the separation is load-bearing rather
than tidy.

Phase 2 added two more artefacts and proved the rule the hard way: putting
``ComparisonArtifact`` beside ``run_comparison`` closed exactly the cycle this
docstring warns about, and the import error surfaced four files from the cause.
The models came back here.

WHY THE MODELS ARE SPLIT FROM THE RUNNERS
-----------------------------------------
:mod:`ledgerloop.eval.report` renders these tables and must be able to import
them. The runners import :mod:`ledgerloop.eval.harness`, which imports
:mod:`ledgerloop.llm`; and :mod:`ledgerloop.matching.pipeline` imports
``eval.metrics`` for the ``PredictedLink`` contract, which initialises the
``eval`` package, which imports the report.

Put the models in the runners and that chain closes into a cycle -- ``matching``
-> ``eval`` -> ``report`` -> ``ablation`` -> ``harness`` -> ``llm`` ->
``matching``. Python reports it as a partially-initialised module several files
from the cause. Keeping the models here breaks it at the only place it can be
broken cleanly: **a document needs the shape of a result, never the machinery
that produced one.**

It also keeps a second promise. ``matching`` must never depend on ``llm``
(ARCHITECTURE.md §6, decision 43) -- that is what makes ``--no-llm`` one code
path with a branch rather than a second implementation. Without this split it
would have acquired that dependency transitively, through a document renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import Field

from ledgerloop.eval.summary import Aggregate, RunSummary
from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.metrics import CostLedger

__all__ = [
    "COMPARED_METRICS",
    "AblationArtifact",
    "AblationRow",
    "ComparisonArm",
    "ComparisonArtifact",
    "ComparisonRow",
    "LLMBaselineArtifact",
    "LLMReportArtifact",
    "RunScore",
    "ScaleArtifact",
    "ScalePoint",
    "SweepArtifact",
    "SweepGroup",
]

_ArtifactT = TypeVar("_ArtifactT", bound="_SavedArtifact")


class _SavedArtifact(FrozenLedgerModel):
    """One JSON file on disk, written with ``\\n`` endings on every platform.

    Indented and sorted so a human can read the artefact a table was rendered
    from -- the same reason the LLM response cache stores its prompt beside its
    completion rather than only hashing it.
    """

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.model_dump_json(indent=2))
            handle.write("\n")

    @classmethod
    def load(cls: type[_ArtifactT], path: Path) -> _ArtifactT:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class AblationRow(FrozenLedgerModel):
    """One ladder, aggregated across the seeds it was run on."""

    label: str
    tiers: tuple[int, ...]
    seeds: tuple[int, ...]
    tuning_hashes: tuple[str, ...] = Field(
        default=(),
        description="The distinct tuning hashes this row ran under. One entry "
        "here, and one across every row of the table, is what witnesses that the "
        "rows differ in their ladder and in nothing else.",
    )
    runs: tuple[RunSummary, ...] = ()

    precision: Aggregate
    recall: Aggregate
    match_rate: Aggregate
    f1: Aggregate
    exception_recall: Aggregate
    candidate_yield: Aggregate
    auto_matched: Aggregate
    false_positives: Aggregate
    false_positive_cost_minor: Aggregate
    llm_calls: Aggregate
    llm_tokens: Aggregate
    equivalent_paid_cost_inr: Aggregate

    llm_available: bool = Field(
        default=False,
        description="Whether a model was reachable when this row ran. A T0-T5 "
        "row with this False ran T0-T4 and the report says so, rather than "
        "printing a zero LLM contribution as though the tier had been measured.",
    )


class AblationArtifact(_SavedArtifact):
    """The whole ablation table, serialisable so the report never re-runs it."""

    split: str
    difficulty: str
    seeds: tuple[int, ...]
    generator_version: str
    calibrated: bool
    tuning_hash: str = ""
    """The one tuning configuration every row ran under.

    :func:`~ledgerloop.eval.ablation.run_ablation` refuses to build an artefact
    whose rows disagree here, so its presence is a guarantee rather than a note.
    """

    rows: tuple[AblationRow, ...]

    def marginal(self, index: int, metric: str) -> float:
        """A row's mean minus the previous row's. The marginal contribution.

        The first row has no predecessor and reports its own value: T0 is
        measured against doing nothing, which is a floor of zero.
        """
        current: Aggregate = getattr(self.rows[index], metric)
        if index == 0:
            return current.mean
        previous: Aggregate = getattr(self.rows[index - 1], metric)
        return current.mean - previous.mean


class SweepGroup(FrozenLedgerModel):
    """One difficulty, aggregated across its seeds."""

    difficulty: str
    split: str
    seeds: tuple[int, ...]
    runs: tuple[RunSummary, ...]
    aggregates: dict[str, Aggregate]

    def of(self, metric: str) -> Aggregate:
        """One metric's aggregate, or an empty one if it was not swept."""
        return self.aggregates.get(metric, Aggregate(metric=metric, count=0))

    @property
    def config_hashes(self) -> tuple[str, ...]:
        """The distinct **tuning** hashes across this group's seeds.

        One entry is the claim the group depends on: the spread it reports is
        corpus variance, not configuration drift. ``config_hash`` would differ on
        every row here by construction -- the seed is part of it -- so comparing
        that would witness nothing.
        """
        return tuple(dict.fromkeys(run.tuning_hash for run in self.runs))


class SweepArtifact(_SavedArtifact):
    """The headline configuration's behaviour across seeds and difficulties."""

    split: str
    generator_version: str
    calibrated: bool
    headline_difficulty: str = Field(
        default="standard",
        description="The difficulty whose group is the multi-seed headline. Every "
        "single-seed table elsewhere in the report is measured at this setting.",
    )
    groups: tuple[SweepGroup, ...]

    @property
    def headline(self) -> SweepGroup | None:
        """The group the headline claims come from, or ``None`` if it was not run."""
        for group in self.groups:
            if group.difficulty == self.headline_difficulty:
                return group
        return None


class ScalePoint(_SavedArtifact):
    """One corpus size, run once.

    Quality and cost are kept in one row but they are not the same kind of
    number and the report must not present them as if they were. Everything
    down to ``false_positive_cost_minor`` is deterministic and reproduces
    exactly on any machine; ``wall_clock_ms`` and ``records_per_second``
    describe *this* machine on *this* run and reproduce nowhere. See
    :class:`ScaleArtifact`.
    """

    orders: int
    records: int
    settlements: int
    bank_rows: int

    # Deterministic.
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    match_rate: float
    exception_recall: float
    false_positive_cost_minor: MinorUnits

    # Measured, and only on the machine that ran it.
    wall_clock_ms: int
    generate_ms: int
    records_per_second: float


class ScaleArtifact(_SavedArtifact):
    """The throughput run PLAN.md left as a stretch item, with its quality beside it.

    **Why quality is here and not only throughput.** The question a scale run is
    usually asked to answer is "how fast", and that was the item as written. It
    is the less interesting half. Precision is this system's headline claim and
    every published measurement of it comes from corpora of 60 to 400 orders; a
    run at 5,000 is the first evidence that the claim survives a statement large
    enough for two of a merchant's payouts to look alike. The first one found
    twenty-two wrong links, so the answer was not free.

    **Why the timings are quarantined in their own fields.** They are the only
    numbers in any artefact this project writes that a second run will not
    reproduce, and they carry a machine's identity rather than the system's.
    ``eval/report.py`` keeps every such figure inside one labelled block for the
    same reason; a document whose diff shows scheduler noise cannot be diffed
    for anything else.
    """

    split: str
    difficulty: str
    seed: int
    generator_version: str
    tuning_hash: str
    calibrated: bool
    machine: str = Field(
        description="What produced the timings. Present so a throughput figure "
        "can never be read as a property of the system alone."
    )
    points: tuple[ScalePoint, ...]

    @property
    def largest(self) -> ScalePoint | None:
        """The headline point: the biggest corpus that was run."""
        return max(self.points, key=lambda p: p.orders) if self.points else None


class LLMBaselineArtifact(_SavedArtifact):
    """B2's result, serialisable so ``make eval`` can render it without re-calling.

    A separate artefact rather than a section computed inside ``eval`` because
    B2 is the only part of the evaluation that touches a network. Running it is
    an explicit command with its own budget; rendering the report is not, and a
    report regenerated twice must not spend quota twice.
    """

    ran: bool = Field(
        description="False when no model was reachable. Every metric below is "
        "then meaningless and the report says so rather than printing zeros."
    )
    reason: str = Field(default="", description="Why it did not run, when it did not.")

    split: str = ""
    difficulty: str = ""
    seed: int = 0
    generator_version: str = ""
    record_count: int = Field(default=0, ge=0)
    evaluation_links: int = Field(default=0, ge=0)

    payments_offered: int = Field(default=0, ge=0)
    credits_offered: int = Field(default=0, ge=0)
    calls_attempted: int = Field(default=0, ge=0)
    calls_failed: int = Field(
        default=0,
        ge=0,
        description="Batches that produced no usable answer: a timeout, a refusal, "
        "a budget stop, or output that would not validate after its retry.",
    )

    links_returned: int = Field(default=0, ge=0)
    links_asserted: int = Field(default=0, ge=0)
    links_duplicated: int = Field(
        default=0,
        ge=0,
        description="Links the model returned more than once across batches. "
        "De-duplicated before scoring, and counted so the repetition is visible.",
    )
    unknown_payment_ids: int = Field(
        default=0,
        ge=0,
        description="Payment ids that appear in no source document. Asserted "
        "anyway -- the absence of a grounding gate is what B2 is measuring -- "
        "and therefore scored as false positives.",
    )
    unknown_bank_txn_ids: int = Field(default=0, ge=0)

    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_ci_low: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_ci_high: float = Field(default=1.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)
    match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    false_positive_cost_minor: MinorUnits = Field(default=0)

    cost: CostLedger = Field(default_factory=CostLedger)
    wall_clock_ms: int = Field(default=0, ge=0)

    provider_kind: str = Field(
        default="",
        description="`live` for a real provider; `offline-standin` for the "
        "prompt-reading reasoner in `eval/offline_provider.py`. The report "
        "renders the second with a banner: its cost and cache columns are "
        "measured machinery, and its accuracy columns are a property of a "
        "documented rule rather than a claim about any language model.",
    )

    system_cost: CostLedger = Field(
        default_factory=CostLedger,
        description="What the production pipeline spent on the SAME dataset. "
        "The token multiple PLAN.md §9.2 predicts is only a number if both "
        "halves are measured on identical data.",
    )
    system_ran: bool = Field(default=False)

    @property
    def is_standin(self) -> bool:
        return self.provider_kind == "offline-standin"

    @property
    def token_multiple(self) -> float:
        """B2's tokens divided by the production run's, on the same corpus.

        Zero when either side has no tokens to divide, which the report renders
        as an explicit statement rather than a ratio. A quotient with a
        denominator of zero is not a large number, it is an undefined one.
        """
        theirs = self.system_cost.total_tokens
        if theirs <= 0 or not self.system_ran:
            return 0.0
        return self.cost.total_tokens / theirs


#: What the before/after table reports. A subset of the sweep's metrics, chosen
#: because these are the ones the change could plausibly move -- and including
#: the two it must **not** move (precision, false positives) is the point.
COMPARED_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "match_rate",
    "exception_recall",
    "false_positives",
    "false_positive_cost_minor",
    "auto_matched",
)


class ComparisonArm(FrozenLedgerModel):
    """One configuration, over one difficulty's seeds."""

    label: str
    difficulty: str
    seeds: tuple[int, ...]
    tuning_hash: str = Field(
        default="",
        description="The tunables alone. Two arms differing here by more than "
        "the switch under test are two experiments, not one comparison.",
    )
    runs: tuple[RunSummary, ...] = ()
    aggregates: dict[str, Aggregate] = Field(default_factory=dict)

    def of(self, metric: str) -> Aggregate:
        return self.aggregates.get(metric, Aggregate(metric=metric, count=0))


class ComparisonRow(FrozenLedgerModel):
    """One difficulty, both arms, ready to render as a before/after line."""

    difficulty: str
    before: ComparisonArm
    after: ComparisonArm

    def delta(self, metric: str) -> float:
        """After minus before, on the mean. A difference, not a test."""
        return self.after.of(metric).mean - self.before.of(metric).mean

    @property
    def precision_held(self) -> bool:
        """Whether the change cost any precision at all, on any seed.

        The one question the whole comparison exists to answer honestly. A
        recall gain bought with a false positive is not an improvement in this
        project, and this reads the false-positive **count** rather than the
        precision ratio -- a ratio can round, a count cannot.
        """
        return all(row.false_positives == 0 for row in self.after.runs)


class ComparisonArtifact(_SavedArtifact):
    """The whole before/after study, serialisable so the report never re-runs it."""

    change: str = Field(description="What differs between the arms, in one line.")
    before_label: str
    after_label: str
    split: str
    generator_version: str
    calibrated: bool
    rows: tuple[ComparisonRow, ...] = ()

    @property
    def headline(self) -> ComparisonRow | None:
        """The standard-difficulty row -- the one every headline claim quotes."""
        for row in self.rows:
            if row.difficulty == "standard":
                return row
        return self.rows[0] if self.rows else None

    @property
    def precision_held_everywhere(self) -> bool:
        """Zero false positives in the after arm, at every difficulty."""
        return bool(self.rows) and all(row.precision_held for row in self.rows)

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.rows[0].after.seeds if self.rows else ()


class RunScore(_SavedArtifact):
    """The four headline figures of one run, so two runs can be compared."""

    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    match_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    exception_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    auto_matched: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)



class LLMReportArtifact(_SavedArtifact):
    """What one measured LLM run cost, was refused, and did not change."""

    ran: bool = Field(
        description="False when no provider was reachable and none was asked "
        "for. Every column below is then meaningless and the report says so "
        "rather than printing zeros."
    )
    reason: str = Field(default="", description="Why it did not run, when it did not.")

    live: bool = Field(
        default=False,
        description="True only when a real provider on the ladder answered. "
        "False means the offline analyst did, and no figure here is a claim "
        "about any language model's answer quality.",
    )
    provider_used: str | None = Field(
        default=None, description="The rung that answered, or the stand-in's name."
    )
    ladder: tuple[str, ...] = Field(
        default=(),
        description="The rungs that were reachable, in the order they would be "
        "tried. One entry is not a ladder and the report says so.",
    )
    fallback_depth: int = Field(
        default=0,
        ge=0,
        description="How far down the ladder the run had to go at its worst. "
        "Zero means the first rung answered every time.",
    )
    provider_failures: tuple[str, ...] = Field(
        default=(),
        description="One line per rung that declined, with its error and how "
        "many attempts it took to give up. Empty for a single-rung run.",
    )

    split: str = ""
    difficulty: str = ""
    seed: int = 0
    generator_version: str = ""
    record_count: int = Field(default=0, ge=0)

    # --- what the model was asked, and what survived ---
    narrations_offered: int = Field(
        default=0, ge=0, description="Narrations the regex layer could not read."
    )
    narrations_accepted: int = Field(default=0, ge=0)
    proposals_returned: int = Field(default=0, ge=0)
    proposals_accepted: int = Field(default=0, ge=0)
    rejected_ungrounded: int = Field(
        default=0,
        ge=0,
        description="References the model returned that were not in the pack it "
        "was given. Refused by the grounding gate.",
    )
    rejected_unverified: int = Field(
        default=0,
        ge=0,
        description="Proposals whose money did not close when re-derived from "
        "the sources. Demoted to review, not dropped.",
    )
    demoted: int = Field(
        default=0,
        ge=0,
        description="The same proposals, counted where the pipeline records "
        "them: a candidate carrying its own arithmetic failure as evidence.",
    )
    explanations_accepted: int = Field(default=0, ge=0)
    calls_refused: int = Field(
        default=0,
        ge=0,
        description="Calls the budget, an outage or a schema failure stopped. A "
        "run survives every one of them.",
    )
    validation_failures: int = Field(default=0, ge=0)

    cost: CostLedger = Field(default_factory=CostLedger)
    wall_clock_ms: int = Field(default=0, ge=0)

    # --- the control ---
    with_llm: RunScore = Field(default_factory=RunScore)
    without_llm: RunScore = Field(default_factory=RunScore)

    @property
    def metrics_unchanged(self) -> bool:
        """Whether removing the model moved any headline figure.

        See the module docstring: ``False`` means an accepted narration repair
        changed what the ladder read, not that the model decided anything.
        """
        return self.with_llm == self.without_llm

    @property
    def calls_per_100_records(self) -> float:
        return self.cost.calls_per_100_records(self.record_count)

    @property
    def is_standin(self) -> bool:
        return self.ran and not self.live
