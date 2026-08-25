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
    "LLMConfig",
    "MatchingTolerances",
    "RunConfig",
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
    prevalence: dict[AnomalyClass, float] = Field(default_factory=lambda: dict(STANDARD_PREVALENCE))
    generator_version: str = Field(default="0.1.0")

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
    thresholds: DecisionThresholds = Field(default_factory=DecisionThresholds)
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
        """
        payload = self.model_dump(mode="json", exclude={"run_id", "data_dir", "audit_dir"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
