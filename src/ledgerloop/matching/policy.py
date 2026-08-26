"""The decision policy: candidate in, immutable ruling out.

PLAN.md 6.5 states the routing rule, and this module is that rule and nothing
more::

    p >= tau_high  -> AUTO_MATCHED
    tau_low < p < tau_high -> NEEDS_REVIEW
    p <= tau_low   -> EXCEPTION

with one gate in front of it, from PLAN.md 7.4: an unverified link can never be
auto-matched. :class:`~ledgerloop.models.decisions.MatchDecision` enforces that
with a model validator, so a policy that forgot it would raise rather than
publish a wrong auto-match. This module demotes explicitly instead, and records
the demotion in the reason string, because "AUTO_MATCHED was refused" is
information a controller needs and an exception a run cannot survive is not.

WHERE THE PROBABILITY COMES FROM AT T0 AND T1
----------------------------------------------
Not from the blender. :attr:`~ledgerloop.models.enums.Tier.
is_deterministic_certain` excludes T0 and T1 from blender fitting and from the
calibration report, because ~70% of volume at p ~= 1.0 produces a reliability
diagram with one populated bin and an ECE that measures the corpus rather than
the calibrator. The tiers set ``calibrated_p`` themselves: 1.0 for a key that
resolved uniquely, ``1/n`` where ``n`` contenders were indistinguishable.

The thresholds are still applied. That is the point of setting a real
probability rather than stamping 1.0 on everything: a contested pair at p = 0.5
falls below ``tau_low`` and routes to ``EXCEPTION`` **through the configured
policy**, not through a special case written into the tier.

The default ``tau_high`` is a placeholder that Step 7 replaces with a value
fitted on the calibration split; :attr:`DecisionThresholds.tau_high_is_fitted`
records which it was, so a report can never present a hand-picked threshold as
a fitted one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from ledgerloop.config import DecisionThresholds
from ledgerloop.models.candidates import MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome

__all__ = ["decide", "decide_all", "decision_id"]

_DECISION_PREFIX = "decision"


def decision_id(candidate_id: str) -> str:
    """A stable decision identifier, derived from the candidate's own id.

    Content-derived rather than a counter, so two runs over the same data
    produce identical decision logs and a reproducibility test can compare them
    directly. Revisions (Step 7 onwards) will carry a suffix and point back
    through ``supersedes``; nothing is ever renumbered.
    """
    return f"{_DECISION_PREFIX}:{candidate_id}"


def decide(
    candidate: MatchCandidate, thresholds: DecisionThresholds, *, decided_at: datetime
) -> MatchDecision:
    """Route one candidate. Pure: same inputs, same ruling, every time."""
    probability = candidate.calibrated_p
    if probability is None:  # pragma: no cover - tiers T0/T1 always set it
        raise ValueError(
            f"{candidate.candidate_id} reached the policy with no calibrated_p; "
            "T0/T1 set it directly and residual tiers get it from the calibrator"
        )

    if probability >= thresholds.tau_high:
        if candidate.arithmetic_verified:
            outcome = DecisionOutcome.AUTO_MATCHED
            reason = (
                f"p={probability:.4f} >= tau_high={thresholds.tau_high:.4f}, "
                "arithmetic verified"
            )
        else:
            # PLAN.md 7.4. The most expensive failure in the system is a
            # confident wrong auto-match, so the gate demotes rather than trusts.
            outcome = DecisionOutcome.NEEDS_REVIEW
            reason = (
                f"demoted: p={probability:.4f} >= tau_high={thresholds.tau_high:.4f} "
                "but the settlement arithmetic does not verify"
            )
    elif probability > thresholds.tau_low:
        outcome = DecisionOutcome.NEEDS_REVIEW
        reason = (
            f"tau_low={thresholds.tau_low:.4f} < p={probability:.4f} < "
            f"tau_high={thresholds.tau_high:.4f}"
        )
    else:
        outcome = DecisionOutcome.EXCEPTION
        reason = f"p={probability:.4f} <= tau_low={thresholds.tau_low:.4f}"

    return MatchDecision(
        decision_id=decision_id(candidate.candidate_id),
        candidate_id=candidate.candidate_id,
        link_type=candidate.link_type,
        source_ref=candidate.source_ref,
        target_ref=candidate.target_ref,
        tier=candidate.tier,
        outcome=outcome,
        calibrated_p=probability,
        arithmetic_verified=candidate.arithmetic_verified,
        decided_at=decided_at,
        reason=reason,
    )


def decide_all(
    candidates: Iterable[MatchCandidate],
    thresholds: DecisionThresholds,
    *,
    decided_at: datetime,
) -> tuple[MatchDecision, ...]:
    """Route every candidate, preserving order."""
    return tuple(decide(candidate, thresholds, decided_at=decided_at) for candidate in candidates)


def positive_decisions(decisions: Sequence[MatchDecision]) -> tuple[MatchDecision, ...]:
    """The decisions that assert a link exists.

    ``AUTO_MATCHED`` only. ``NEEDS_REVIEW`` is a referral to a human, not a
    prediction, and counting referrals as matches is the precision-inflating
    trap PLAN.md 9.1 warns about -- so the filter lives here, once, rather than
    being re-derived at each call site.
    """
    return tuple(decision for decision in decisions if decision.is_positive_prediction)


__all__ += ["positive_decisions"]
