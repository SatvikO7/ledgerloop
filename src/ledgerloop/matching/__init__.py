"""The tier ladder. Cheapest and most certain first.

Three tiers today -- T0 exact key, T1 tolerance, T2 aggregation -- sharing one
residual pool. T0 and T1 share a resolver; T2 has its own, because the question
it asks is different in kind: not "is this credit this payout?" but "which
payments travelled in this tranche?". T3 fuzzy, T4 graph and T5 LLM
adjudication are later steps, and are absent rather than stubbed: a tier that
exists and returns nothing would show as a zero in the contribution table, and a
zero for an unbuilt component is a false measurement.

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
from ledgerloop.matching.subset_sum import (
    SubsetSearch,
    SubsetSolution,
    find_subsets,
    greedy_subset,
    meet_in_the_middle,
)
from ledgerloop.matching.tier0_exact import (
    T0_RULE,
    OrderLegOutcome,
    resolve_order_leg,
    run_tier0,
)
from ledgerloop.matching.tier1_tolerance import rule_for, run_tier1
from ledgerloop.matching.tier2_aggregation import (
    AggregationOutcome,
    Assignment,
    credit_bucket,
    expected_credit_minor,
    payment_bucket,
    run_tier2,
)

__all__ = [
    "MATCHER_DESCRIPTION",
    "MATCHER_NAME",
    "T0_RULE",
    "AggregationOutcome",
    "Assignment",
    "BankLegOutcome",
    "BankLegRule",
    "MatchContext",
    "MatchRun",
    "OrderLegOutcome",
    "SettlementView",
    "SubsetSearch",
    "SubsetSolution",
    "allocated_share_minor",
    "candidate_id",
    "credit_bucket",
    "decide",
    "decide_all",
    "decision_id",
    "expected_credit_minor",
    "find_subsets",
    "greedy_subset",
    "meet_in_the_middle",
    "payment_bucket",
    "positive_decisions",
    "resolve_bank_leg",
    "resolve_order_leg",
    "rule_for",
    "run_matching",
    "run_tier0",
    "run_tier1",
    "run_tier2",
]
