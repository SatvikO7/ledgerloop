"""The serialised results of Step 10's three expensive commands.

The models only. The code that *produces* them lives in
:mod:`ledgerloop.eval.ablation`, :mod:`ledgerloop.eval.sweep` and
:mod:`ledgerloop.eval.llm_baseline`, and the separation is load-bearing rather
than tidy.

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
    "AblationArtifact",
    "AblationRow",
    "LLMBaselineArtifact",
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
