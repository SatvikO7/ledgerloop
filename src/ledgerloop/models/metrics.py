"""Run metrics and the cost ledger.

Every field here is a number that will appear in ``EVALUATION.md``, and none of
it is hand-typed -- ``make eval`` regenerates the whole document. The contract
exists so that the evaluator and the report writer cannot drift apart.

ON SAMPLE SIZE
--------------
:class:`LinkMetrics` carries confidence intervals, not bare point estimates.
On a 300-record test split roughly 250 links get auto-matched, so one wrong
decision moves precision from 1.000 to 0.996 and two moves it to 0.992. A
headline "0.99 precision" from that sample cannot be distinguished from 0.97.
Reporting the interval alongside the point estimate is the difference between
a measured claim and a lucky one.

**Phase 2.1 applies that rule to every headline proportion, not to precision
alone.** Until then the argument was made at length and then honoured once,
which is the sharpest methodological criticism the project's own standard
invites. The metric with the smallest denominator in the whole report is
exception recall -- thirty records on ``test`` seed 42 -- and it was the one
printed to four significant figures with no interval at all. Its 95% Wilson
interval spans roughly twenty points, and the target sits inside it, so the
interval changes what the number *means*: at n = 30 the run neither meets nor
conclusively misses ≥ 0.95, and a bare point estimate cannot say that.

Every interval here is **Wilson at 95%**, computed by
:func:`~ledgerloop.stats.wilson_interval`, and every one travels with the
integer numerator and denominator it was computed from -- so a reader can
re-derive it, and a proportion can never be reported without its sample size.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ledgerloop.models.base import FrozenLedgerModel, LedgerModel, MinorUnits
from ledgerloop.models.enums import AnomalyClass, ExceptionClass, Tier

__all__ = [
    "CalibrationMetrics",
    "CostLedger",
    "LinkMetrics",
    "Proportion",
    "RunMetrics",
    "TierContribution",
    "Verdict",
]


class Verdict(StrEnum):
    """Whether a proportion clears a floor, read off its interval.

    Every target in this project is a floor (``>= x``), so the reading is
    one-sided and there are three answers rather than two. The third is the
    one this exists for: a sample can be too small to separate a pass from a
    failure, and saying so is the honest report.

    It lives beside :class:`Proportion` rather than in the report writer
    because two renderers now need it -- ``EVALUATION.md`` and the dashboard --
    and a verdict rule stated twice is a verdict rule that can disagree with
    itself. The prose differs between them; the ruling does not.
    """

    MET = "met"
    MISSED = "missed"
    UNDECIDED = "undecided"
    UNTARGETED = "untargeted"
    """Reported, with no floor to clear. Not a pass and not a failure."""


#: The floors the headline proportions are judged against, by field name.
#:
#: Data rather than literals so the report and the dashboard cannot drift, and
#: ``None`` is a deliberate entry rather than an absence: link recall and
#: unmatchable coverage are *reported and not targeted*, which is a different
#: statement from having a target nobody wrote down.
METRIC_TARGETS: dict[str, float | None] = {
    "precision_interval": 0.99,
    "recall_interval": None,
    "match_rate_interval": 0.85,
    "exception_recall_interval": 0.95,
    "unmatchable_coverage_interval": None,
}


class Proportion(FrozenLedgerModel):
    """A ratio that cannot be reported without its sample size or its interval.

    The type exists so the omission is not available. A bare ``float`` field can
    be printed alone; this one carries ``successes``, ``trials`` and a 95%
    Wilson interval in the same object, so any renderer that shows the estimate
    has the other three to hand and no excuse for dropping them.
    """

    successes: int = Field(ge=0)
    trials: int = Field(ge=0)
    value: float = Field(ge=0.0, le=1.0)
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)

    @classmethod
    def of(cls, successes: int, trials: int) -> Proportion:
        """Build one from its counts. The only sanctioned constructor.

        A zero denominator yields ``0.0`` with the full ``[0, 1]`` interval:
        a system that predicted nothing has not achieved perfect precision, and
        with no evidence every proportion remains possible.
        """
        from ledgerloop.stats import wilson_interval

        low, high = wilson_interval(successes, trials)
        return cls(
            successes=successes,
            trials=trials,
            value=successes / trials if trials > 0 else 0.0,
            ci_low=low,
            ci_high=high,
        )

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def verdict(self, target: float | None) -> Verdict:
        """Met, missed, or a sample too small to tell the two apart.

        Read off the **interval**, one-sided, because every target here is a
        floor:

        * ``MET`` -- the lower bound clears the target. The data supports it.
        * ``MISSED`` -- the upper bound is below it. The data rules it out.
        * ``UNDECIDED`` -- the interval straddles it, so the sample cannot
          separate the two.

        The third cuts both ways, which is the point. Exception recall at 30 of
        30 does not *demonstrate* >= 95% from thirty records, and 0.9333 from
        the same thirty is not a clean miss -- both are the same statement about
        the same denominator, and reporting the flattering one as a pass is the
        same error as reporting the other as a failure.
        """
        if target is None:
            return Verdict.UNTARGETED
        if self.ci_low >= target:
            return Verdict.MET
        if self.ci_high < target:
            return Verdict.MISSED
        return Verdict.UNDECIDED

    def covers(self, target: float) -> bool:
        """Whether ``target`` lies inside the interval.

        The question a target-versus-result line has to answer honestly. At
        n = 30 an estimate of 0.9333 against a target of 0.95 is not a miss --
        it is a sample that cannot separate the two, and this says so.
        """
        return self.ci_low <= target <= self.ci_high

    def render(self, places: int = 4) -> str:
        """``0.9333 [0.7868, 0.9815] (n=30)`` -- estimate, interval, sample."""
        return (
            f"{self.value:.{places}f} "
            f"[{self.ci_low:.{places}f}, {self.ci_high:.{places}f}] "
            f"(n={self.trials})"
        )


class LinkMetrics(FrozenLedgerModel):
    """Precision/recall over ``PAYMENT_CREDITED_AS`` links -- the atomic unit."""

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    precision_ci_low: float = Field(ge=0.0, le=1.0)
    precision_ci_high: float = Field(ge=0.0, le=1.0)

    recall_ci_low: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Wilson lower bound on recall. Added in Phase 2.1: recall "
        "has a smaller denominator than precision on every corpus here, so it "
        "needed the interval more and had it less.",
    )
    recall_ci_high: float = Field(default=1.0, ge=0.0, le=1.0)

    false_positive_cost_minor: MinorUnits = Field(
        default=0,
        description="Total impact of incorrect auto-matches. A rupee figure, not a "
        "ratio -- this is the number that makes the precision-over-match-rate "
        "argument concrete.",
    )


class TierContribution(FrozenLedgerModel):
    """One row of the ablation table: what this tier added, and what it cost."""

    tier: Tier
    candidates_proposed: int = Field(ge=0)
    auto_matched: int = Field(ge=0)
    marginal_auto_matched: int = Field(
        ge=0, description="Auto-matches this tier added over the ladder without it."
    )
    llm_calls: int = Field(ge=0, default=0)
    wall_clock_ms: int = Field(ge=0, default=0)


class CostLedger(LedgerModel):
    """Tokens, calls, latency, money.

    ``actual_cost_inr`` is ₹0 by construction (free tier). The interesting
    figure is ``equivalent_paid_cost_inr`` -- what the same run would cost on a
    frontier paid API -- because it quantifies what deterministic-first buys.
    """

    llm_calls: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    wall_clock_ms: int = Field(default=0, ge=0)

    actual_cost_inr: float = Field(
        default=0.0, ge=0.0, description="Real spend. Zero on the free tier."
    )
    equivalent_paid_cost_inr: float = Field(
        default=0.0, ge=0.0, description="Same token count priced at a paid frontier API."
    )

    provider_used: str | None = None
    fallback_depth: int = Field(
        default=0,
        ge=0,
        description="How far down the provider ladder the run had to go. Recorded so "
        "a rate-limited run is visible in the audit trail rather than silent.",
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def calls_per_100_records(self, record_count: int) -> float:
        """The headline cost-discipline metric. Target: < 10 (i.e. <30 per 300)."""
        if record_count <= 0:
            return 0.0
        return self.llm_calls * 100.0 / record_count

    @property
    def cache_hit_rate(self) -> float:
        """A second identical run must reach 1.0 -- zero live API calls."""
        attempts = self.llm_calls + self.cache_hits
        return self.cache_hits / attempts if attempts else 0.0


class CalibrationMetrics(FrozenLedgerModel):
    """Calibration quality, measured where uncertainty actually lives.

    ``residual_only`` is ``True`` by default and should stay that way. T0 and
    T1 contribute ~70% of volume at a probability of essentially 1.0; including
    them produces a reliability diagram with one populated bin and an ECE near
    zero that measures the *shape of the corpus* rather than the quality of the
    calibrator. Restricting to T2-T5 is the honest measurement.
    """

    ece: float = Field(ge=0.0, le=1.0, description="Expected calibration error.")
    brier: float = Field(ge=0.0, le=1.0)
    bin_count: int = Field(ge=1)
    populated_bins: int = Field(
        ge=0, description="Bins with at least one sample. A low count invalidates the ECE."
    )
    sample_count: int = Field(ge=0)
    residual_only: bool = Field(
        default=True, description="Whether T0/T1 were excluded from the measurement."
    )


class RunMetrics(LedgerModel):
    """Everything ``EVALUATION.md`` reports for one run."""

    run_id: str
    record_count: int = Field(ge=0)

    # --- the headline three (PLAN.md §9.1) ---
    auto_match_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    match_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="auto-matched / reconcilable. Denominator excludes UNMATCHABLE.",
    )
    exception_recall: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- the same three, with their sample sizes and intervals (Phase 2.1) ---
    #
    # The bare floats above are kept because every existing caller reads them and
    # a rename would touch the UI, the artefacts and the ablation for no gain.
    # What is added is the honest form of each, and the report renders these.
    precision_interval: Proportion | None = Field(
        default=None, description="Auto-match precision with n and its 95% Wilson CI."
    )
    recall_interval: Proportion | None = Field(
        default=None, description="Link recall with n and its 95% Wilson CI."
    )
    match_rate_interval: Proportion | None = Field(
        default=None, description="Match rate with n and its 95% Wilson CI."
    )
    exception_recall_interval: Proportion | None = Field(
        default=None,
        description="Exception recall with n and its 95% Wilson CI. The smallest "
        "denominator in the report, and the reason Phase 2.1 exists.",
    )
    unmatchable_coverage_interval: Proportion | None = Field(
        default=None,
        description="Share of the UNMATCHABLE floor the queue accounted for, "
        "with n and its interval. Reported on its own line and never inside "
        "exception recall.",
    )

    link_metrics: LinkMetrics | None = None
    calibration: CalibrationMetrics | None = None
    cost: CostLedger = Field(default_factory=CostLedger)

    tier_contributions: tuple[TierContribution, ...] = ()

    # --- honest negatives (PLAN.md §9.1, D8) ---
    recall_by_anomaly_class: dict[AnomalyClass, float] = Field(
        default_factory=dict,
        description="Per-class recall INCLUDING the classes that do badly. Publishing "
        "only the good rows is what this project is trying not to do.",
    )
    exception_confusion: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="True anomaly class -> predicted exception class -> count. Keyed by "
        "enum .value because JSON object keys must be strings. The two vocabularies "
        "differ (11 vs 13), so this matrix is rectangular, not square.",
    )

    # --- the reported ceiling ---
    unmatchable_count: int = Field(default=0, ge=0)
    unmatchable_impact_minor: MinorUnits = Field(default=0)

    # --- money view ---
    reconciled_minor: MinorUnits = Field(default=0)
    outstanding_minor: MinorUnits = Field(default=0)

    exceptions_by_class: dict[ExceptionClass, int] = Field(default_factory=dict)
    records_per_second: float = Field(default=0.0, ge=0.0)
