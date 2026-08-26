"""The tier ladder. Cheapest and most certain first.

Two tiers today -- T0 exact key, T1 tolerance -- sharing one resolver and one
residual pool. T2 aggregation, T3 fuzzy, T4 graph and T5 LLM adjudication are
later steps, and are absent rather than stubbed: a tier that exists and returns
nothing would show as a zero in the contribution table, and a zero for an
unbuilt component is a false measurement.

Read :mod:`ledgerloop.matching.pipeline` first; the rest is the machinery it
composes.
"""

from ledgerloop.matching.bank_leg import (
    BankLegOutcome,
    BankLegRule,
    allocated_share_minor,
    candidate_id,
    resolve_bank_leg,
)
from ledgerloop.matching.context import MatchContext, SettlementView
from ledgerloop.matching.pipeline import (
    MATCHER_DESCRIPTION,
    MATCHER_NAME,
    MatchRun,
    run_matching,
)
from ledgerloop.matching.policy import decide, decide_all, decision_id, positive_decisions
from ledgerloop.matching.tier0_exact import (
    T0_RULE,
    OrderLegOutcome,
    resolve_order_leg,
    run_tier0,
)
from ledgerloop.matching.tier1_tolerance import rule_for, run_tier1

__all__ = [
    "MATCHER_DESCRIPTION",
    "MATCHER_NAME",
    "T0_RULE",
    "BankLegOutcome",
    "BankLegRule",
    "MatchContext",
    "MatchRun",
    "OrderLegOutcome",
    "SettlementView",
    "allocated_share_minor",
    "candidate_id",
    "decide",
    "decide_all",
    "decision_id",
    "positive_decisions",
    "resolve_bank_leg",
    "resolve_order_leg",
    "rule_for",
    "run_matching",
    "run_tier0",
    "run_tier1",
]
