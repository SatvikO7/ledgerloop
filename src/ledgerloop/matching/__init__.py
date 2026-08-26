"""The tier ladder. Cheapest and most certain first.

Five tiers -- T0 exact key, T1 tolerance, T2 aggregation, T3 lexical, T4 graph
-- sharing one residual pool. T0 and T1 share a resolver; T2, T3 and T4 each
have their own, because they ask different questions: which payments composed
this tranche, which credit is this payout when the reference is gone, and what
follows from what is already known. T2/T3/T4 repeat in a bounded loop until a
pass changes nothing.

T5 LLM adjudication is a later step and is absent rather than stubbed: a tier
that exists and returns nothing would show as a zero in the contribution table,
and a zero for an unbuilt component is a false measurement.

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
from ledgerloop.matching.tier3_lexical import (
    LexicalOutcome,
    MerchantProfile,
    build_profiles,
    run_tier3,
    score_names,
)
from ledgerloop.matching.tier4_graph import (
    GraphOutcome,
    RingFinding,
    build_graph,
    detect_rings,
    run_tier4,
)

__all__ = [
    "MATCHER_DESCRIPTION",
    "MATCHER_NAME",
    "T0_RULE",
    "AggregationOutcome",
    "Assignment",
    "BankLegOutcome",
    "BankLegRule",
    "GraphOutcome",
    "LexicalOutcome",
    "MatchContext",
    "MatchRun",
    "MerchantProfile",
    "OrderLegOutcome",
    "RingFinding",
    "SettlementView",
    "SubsetSearch",
    "SubsetSolution",
    "allocated_share_minor",
    "build_graph",
    "build_profiles",
    "candidate_id",
    "credit_bucket",
    "decide",
    "decide_all",
    "decision_id",
    "detect_rings",
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
    "run_tier3",
    "run_tier4",
    "score_names",
]
