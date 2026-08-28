"""Run configuration -- every tunable in one frozen, hashable object.

Two properties matter beyond holding values:

* **Frozen.** A run's configuration cannot drift mid-run, so the audit trail's
  ``config_hash`` genuinely identifies what produced a result.
* **Hashable and serialisable.** ``EVALUATION.md`` prints the config alongside
  the numbers, because a metric without its thresholds is not reproducible.

Thresholds carry defaults, but ``tau_high`` is **not meant to stay at its
default**. PLAN.md §6.5 selects it on the calibration split as the lowest
threshold achieving auto-match precision >= 0.99. The default here is a
placeholder for a value the calibrator computes; :attr:`DecisionThresholds.
tau_high_is_fitted` records whether that has happened, so a report can never
silently present a hand-picked threshold as a fitted one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field, model_validator

from ledgerloop.models.base import FrozenLedgerModel, MinorUnits
from ledgerloop.models.enums import AnomalyClass, Difficulty, SplitName

__all__ = [
    "SPLIT_SIZES",
    "STANDARD_PREVALENCE",
    "AutoResolutionBounds",
    "DecisionThresholds",
    "GeneratorConfig",
    "GraphInference",
    "LLMConfig",
    "LexicalMatching",
    "MatchingTolerances",
    "RunConfig",
    "SeverityThresholds",
    "prevalence_for",
]


class MatchingTolerances(FrozenLedgerModel):
    """Amount and date bands for T1 and T2 (PLAN.md §6.1)."""

    amount_floor_minor: MinorUnits = Field(
        default=100, description="₹1 in paise. The absolute floor of the tolerance band."
    )
    amount_bps: int = Field(
        default=50, ge=0, description="50 bps = 0.5%. The proportional part of the band."
    )
    date_window_days: int = Field(
        default=3, ge=0, description="T1 date tolerance, ± this many days."
    )
    aggregation_epsilon_minor: MinorUnits = Field(
        default=300,
        description="T2 residual tolerance. Wider than the T1 floor because a subset "
        "accumulates per-payment rounding drift (A02) across its members.",
    )
    max_subset_size: int = Field(
        default=40, ge=1, description="Bucket cap above which T2 falls back to greedy search."
    )
    subset_solver_timeout_ms: int = Field(
        default=200, ge=1, description="Hard per-credit cap (PLAN.md §6.2)."
    )


class LexicalMatching(FrozenLedgerModel):
    """T3's similarity gates (PLAN.md 6.3).

    T3 is the first tier that scores one string against another, so it is the
    first that needs a *threshold* rather than an exact test. Both gates are
    here rather than in the tier so a reported number cannot be separated from
    the values that produced it -- they travel in ``RunConfig.config_hash``.
    """

    min_score: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Similarity a merchant name must reach before T3 will consider "
        "the credit at all. High because the corpus's own abbreviations score in the "
        "high nineties against their expansions once the consonant skeleton is taken "
        "-- so a name that only reaches the eighties is a different merchant, not a "
        "harder spelling of the same one.",
    )
    min_margin: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="How far the best candidate must beat the runner-up. Two merchants "
        "the scorer cannot separate are an ambiguity, and picking the higher of two "
        "indistinguishable scores is the coin flip every tier here refuses.",
    )
    date_window_days: int = Field(
        default=7,
        ge=0,
        description="Credit-to-settlement date window for a lexical match. Wider than "
        "T1's +/-3 because T3's residual is precisely the batches other anomalies have "
        "already moved -- A04 shifts a credit two days and A12 shifts a settlement by up "
        "to nine, and a window that excluded them would lose the links T3 exists for.",
    )


class GraphInference(FrozenLedgerModel):
    """T4's constraint-propagation parameters (PLAN.md 6.4)."""

    sibling_completion_threshold: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description="Fraction of a settlement's payments that must already point at one "
        "credit before the remainder are constrained to it. PLAN.md 6.4 names 80%.",
    )
    ring_min_events: int = Field(
        default=3,
        ge=1,
        description="Refund events on one customer reference before it is worth a look.",
    )
    ring_min_merchants: int = Field(
        default=2,
        ge=1,
        description="Distinct merchants those events must span. One merchant is a "
        "difficult customer; several is a pattern.",
    )
    max_rerun_passes: int = Field(
        default=4,
        ge=1,
        description="Cap on the T2/T3/T4 re-run loop. The loop stops early when a pass "
        "changes nothing, so this only bounds a pathological case -- but an unbounded "
        "loop in a reconciliation run is a hang, not a slow answer.",
    )


class DecisionThresholds(FrozenLedgerModel):
    """The routing policy (PLAN.md §6.5)."""

    tau_high: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="p >= tau_high -> AUTO_MATCHED. Fitted on the calibration split, "
        "not hand-picked. See tau_high_is_fitted.",
    )
    tau_low: float = Field(
        default=0.60, ge=0.0, le=1.0, description="p <= tau_low -> EXCEPTION."
    )
    target_auto_match_precision: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="What tau_high is selected to achieve. In finance ops a wrong "
        "auto-match costs far more than a human reviewing an extra item, so the "
        "objective is precision, not coverage.",
    )
    tau_high_is_fitted: bool = Field(
        default=False,
        description="True only after threshold selection has run on the calibration "
        "split. Reports must state which value they used and how it was chosen.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> DecisionThresholds:
        if self.tau_low > self.tau_high:
            raise ValueError(
                f"tau_low ({self.tau_low}) must not exceed tau_high ({self.tau_high})"
            )
        return self


class SeverityThresholds(FrozenLedgerModel):
    """Rupee bands for exception severity, plus the age escalation (PLAN.md 8.1).

    Severity is "driven by rupee impact + age", which leaves the bands
    unstated. They live here rather than in the classifier so that a queue
    ordering cannot be separated from the values that produced it -- they
    travel in ``RunConfig.config_hash`` like every other tunable.

    Age escalates by **one step and never more**, and never downwards. An old
    twelve-rupee drift is still a twelve-rupee drift.
    """

    critical_minor: MinorUnits = Field(
        default=10_000_000, description="₹1,00,000. A payout at this size is a phone call."
    )
    high_minor: MinorUnits = Field(default=1_000_000, description="₹10,000.")
    medium_minor: MinorUnits = Field(default=100_000, description="₹1,000.")
    escalate_after_days: int = Field(
        default=14,
        ge=0,
        description="Age past which severity rises one step. Measured against the "
        "dataset's own latest date, never the wall clock, so a queue is "
        "reproducible and the report stays timestamp-free.",
    )

    @model_validator(mode="after")
    def _bands_descend(self) -> SeverityThresholds:
        if not self.critical_minor > self.high_minor > self.medium_minor:
            raise ValueError(
                "severity bands must strictly descend: "
                f"critical {self.critical_minor} > high {self.high_minor} > "
                f"medium {self.medium_minor}"
            )
        return self


class AutoResolutionBounds(FrozenLedgerModel):
    """Hard caps on what the agent may resolve by itself (PLAN.md §8.3).

    The agent *proposes* journal adjustments and never posts them to any real
    system (§1.3). These bounds cap the proposals it is willing to auto-approve
    within the run's own books, and they are printed in the report so a reader
    can see the leash rather than take it on trust.
    """

    rounding_per_record_minor: MinorUnits = Field(
        default=500, description="₹5 per record."
    )
    rounding_per_run_minor: MinorUnits = Field(
        default=50_000, description="₹500 per run, across all rounding adjustments."
    )
    timing_shift_max_days: int = Field(default=5, ge=0)
    enabled: bool = Field(
        default=True, description="Set False to make every exception proposal-only."
    )


class LLMConfig(FrozenLedgerModel):
    """LLM wiring. The MVP ships one provider plus ``--no-llm``.

    The provider failover ladder (Groq -> Gemini -> OpenRouter -> Ollama) is
    scheduled after the deterministic system is complete and measured; the
    field is here so the audit trail's shape does not change when it lands.
    """

    enabled: bool = Field(
        default=True,
        description="False is the --no-llm path: the entire pipeline runs "
        "deterministically with no network. Powers the ablation and guarantees "
        "the demo survives a rate limit.",
    )
    provider: str = Field(default="groq")
    model: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    cache_dir: Path = Field(
        default=Path("tests/fixtures/llm_cache"),
        description="Content-hash response cache, committed as fixtures so CI and "
        "reruns consume zero API calls.",
    )
    max_calls_per_run: int = Field(
        default=30,
        ge=0,
        description="Hard budget. Exceeding it aborts the LLM path rather than "
        "quietly burning free-tier quota.",
    )
    narration_batch_size: int = Field(default=20, ge=1)
    adjudication_batch_size: int = Field(default=10, ge=1)
    validation_retries: int = Field(
        default=1,
        ge=0,
        description="Retries on Pydantic validation failure before falling through to "
        "an exception. Never a crash, never a silent default.",
    )


#: Anomaly prevalence at standard difficulty.
#:
#: A11 FX_MULTICURRENCY is cut from the MVP; its 2% is reassigned to CLEAN,
#: which is why CLEAN is 67% here against PLAN.md §5.2's 65%. The weights must
#: sum to exactly 1.0 and there is a test that says so -- a generator that
#: silently normalises a broken distribution would make prevalence
#: unverifiable, and every downstream metric is conditioned on prevalence.
STANDARD_PREVALENCE: dict[AnomalyClass, float] = {
    AnomalyClass.CLEAN: 0.67,
    AnomalyClass.ROUNDING_DRIFT: 0.05,
    AnomalyClass.FEE_TAX_MISMATCH: 0.04,
    AnomalyClass.TIMING_SHIFT: 0.05,
    AnomalyClass.DUPLICATE_CREDIT: 0.02,
    AnomalyClass.POST_SETTLEMENT_REFUND: 0.04,
    AnomalyClass.MISSING_REFERENCE: 0.04,
    AnomalyClass.CHARGEBACK_NETTED: 0.03,
    AnomalyClass.SPLIT_PAYOUT: 0.03,
    AnomalyClass.ORPHAN_BANK_CREDIT: 0.02,
    AnomalyClass.LATE_ARRIVAL: 0.01,
}

#: Share of scenario draws that are CLEAN at each difficulty (PLAN.md §5.2's
#: "difficulty dial"). The non-clean classes keep their relative proportions and
#: are rescaled to fill the remainder, so turning the dial changes *how much*
#: goes wrong without changing *what* goes wrong -- which is what makes the
#: three difficulty columns comparable to each other.
_CLEAN_SHARE_BY_DIFFICULTY: dict[Difficulty, float] = {
    Difficulty.EASY: 0.85,
    Difficulty.STANDARD: 0.67,
    Difficulty.HARD: 0.50,
}


def prevalence_for(difficulty: Difficulty) -> dict[AnomalyClass, float]:
    """Anomaly prevalence at the given difficulty.

    ``CLEAN`` is computed as ``1 - sum(others)`` rather than taken from the
    table, so the result sums to exactly 1.0 in floating point and the
    :class:`GeneratorConfig` validator cannot be tripped by scaling residue.
    """
    clean_share = _CLEAN_SHARE_BY_DIFFICULTY[difficulty]
    others = {
        anomaly: weight
        for anomaly, weight in STANDARD_PREVALENCE.items()
        if anomaly is not AnomalyClass.CLEAN
    }
    scale = (1.0 - clean_share) / sum(others.values())
    scaled: dict[AnomalyClass, float] = {
        anomaly: weight * scale for anomaly, weight in others.items()
    }
    return {AnomalyClass.CLEAN: 1.0 - sum(scaled.values()), **scaled}


#: Order counts per split. ``TRAIN`` is the addition the plan lacked -- the
#: blender needs its own fitting data so the calibrator never sees in-sample
#: scores. See SplitName's docstring.
SPLIT_SIZES: dict[SplitName, int] = {
    SplitName.DEV: 60,
    SplitName.TRAIN: 400,
    SplitName.CALIBRATION: 200,
    SplitName.TEST: 300,
    SplitName.SCALE: 5_000,
}


class GeneratorConfig(FrozenLedgerModel):
    """Synthetic data generation (PLAN.md §5)."""

    split: SplitName = SplitName.DEV
    difficulty: Difficulty = Difficulty.STANDARD
    seed: int = Field(default=42, ge=0)
    order_count: int | None = Field(
        default=None, description="Overrides the split default when set."
    )
    prevalence: dict[AnomalyClass, float] = Field(
        default_factory=lambda: dict(STANDARD_PREVALENCE),
        description="Left unset, this is derived from `difficulty`. Set it "
        "explicitly to override the dial entirely.",
    )
    generator_version: str = Field(default="0.2.0")
    ensure_class_coverage: bool = Field(
        default=False,
        description="After the prevalence draw, force-apply any anomaly class that "
        "produced no effect. This DISTORTS prevalence, so it stays off for every "
        "evaluated split. It exists for the committed fixture set, whose job is to "
        "exercise every code path rather than to be statistically representative -- "
        "a 60-order dataset simply has too few settlements to be near-certain of "
        "containing a 2%-prevalence class.",
    )

    @model_validator(mode="before")
    @classmethod
    def _prevalence_follows_difficulty(cls, data: object) -> object:
        """Derive prevalence from the difficulty dial unless it was given explicitly.

        A ``before`` validator rather than an ``after`` one because the model is
        frozen: by the time an ``after`` validator runs, the field can no longer
        be assigned.
        """
        if isinstance(data, dict) and data.get("prevalence") is None:
            difficulty = Difficulty(data.get("difficulty", Difficulty.STANDARD))
            return {**data, "prevalence": prevalence_for(difficulty)}
        return data

    @model_validator(mode="after")
    def _prevalence_sums_to_one(self) -> GeneratorConfig:
        total = sum(self.prevalence.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"anomaly prevalence must sum to 1.0, got {total!r}")
        if any(weight < 0 for weight in self.prevalence.values()):
            raise ValueError("anomaly prevalence weights must be non-negative")
        return self

    @property
    def effective_order_count(self) -> int:
        return self.order_count if self.order_count is not None else SPLIT_SIZES[self.split]


class RunConfig(FrozenLedgerModel):
    """Everything one reconciliation run needs to be reproducible."""

    run_id: str
    split: SplitName = SplitName.DEV
    difficulty: Difficulty = Difficulty.STANDARD
    seed: int = Field(default=42, ge=0)

    tolerances: MatchingTolerances = Field(default_factory=MatchingTolerances)
    lexical: LexicalMatching = Field(default_factory=LexicalMatching)
    graph: GraphInference = Field(default_factory=GraphInference)
    thresholds: DecisionThresholds = Field(default_factory=DecisionThresholds)
    severity: SeverityThresholds = Field(default_factory=SeverityThresholds)
    auto_resolution: AutoResolutionBounds = Field(default_factory=AutoResolutionBounds)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    enabled_tiers: tuple[int, ...] = Field(
        default=(0, 1, 2, 3, 4, 5),
        description="Which tiers run. The ablation study drives this: (0,) then "
        "(0,1) then (0,1,2) and so on, each row priced in LLM calls.",
    )

    data_dir: Path = Field(default=Path("data/generated"))
    audit_dir: Path = Field(default=Path("reports/audit"))

    @model_validator(mode="after")
    def _tiers_valid(self) -> RunConfig:
        if not self.enabled_tiers:
            raise ValueError("at least one tier must be enabled")
        if sorted(self.enabled_tiers) != list(self.enabled_tiers):
            raise ValueError("enabled_tiers must be ascending")
        if len(set(self.enabled_tiers)) != len(self.enabled_tiers):
            raise ValueError("enabled_tiers must not repeat")
        if any(tier < 0 or tier > 5 for tier in self.enabled_tiers):
            raise ValueError("tiers must be in range 0..5")
        if 5 in self.enabled_tiers and not self.llm.enabled:
            raise ValueError(
                "T5 is the LLM adjudication tier; enabling it with llm.enabled=False "
                "would report an LLM contribution that never happened"
            )
        return self

    @property
    def config_hash(self) -> str:
        """Stable hash of the configuration, minus the run's own identity.

        ``run_id`` and output paths are excluded so that two runs of the same
        configuration hash identically -- that is what lets the evaluator prove
        a rerun reproduced a result rather than merely resembling it.

        The **corpus** is part of this hash: a run over ``test`` seed 43 is not
        the same run as one over seed 42, and an audit trail that said it was
        would be wrong. :attr:`tuning_hash` is the one to compare across corpora.
        """
        payload = self.model_dump(mode="json", exclude={"run_id", "data_dir", "audit_dir"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def tuning_hash(self) -> str:
        """Hash of the **tunables** alone: every knob, and nothing else.

        Excluded on top of :attr:`config_hash`'s exclusions:

        * ``split``, ``difficulty`` and ``seed`` -- which corpus this is, not how
          it was processed. A multi-seed table exists precisely to run one
          configuration over several corpora, so a hash that changed with the
          seed could never witness that the configuration was held fixed.
        * ``enabled_tiers`` -- which *ladder* ran. An ablation table exists
          precisely to vary that, and it reports the ladder explicitly as the row
          label; folding it into the hash would make the hash restate the label
          and stop witnessing anything else.

        What remains is the tolerances, the lexical gates, the graph parameters,
        the thresholds, the severity bands, the resolution bounds and the LLM
        configuration. One ``tuning_hash`` across the rows of a table is what
        turns "nothing else changed" from a claim into a check.
        """
        payload = self.model_dump(
            mode="json",
            exclude={
                "run_id",
                "data_dir",
                "audit_dir",
                "split",
                "difficulty",
                "seed",
                "enabled_tiers",
            },
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
