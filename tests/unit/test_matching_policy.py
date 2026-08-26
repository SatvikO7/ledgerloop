"""The decision policy: thresholds, the arithmetic gate, and what counts as a match.

Three properties carry the whole step:

* the routing follows the *configured* thresholds, not constants in the code;
* an unverified link can never be auto-matched, whatever its probability;
* ``NEEDS_REVIEW`` is not a positive prediction, so a referral can never reach
  the evaluator as a match.

The third is the precision-inflating trap PLAN.md §9.1 names, and it is
enforced on the model rather than in the metric, so a future caller cannot
forget it.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from ledgerloop.config import DecisionThresholds
from ledgerloop.matching.policy import decide, decide_all, decision_id, positive_decisions
from ledgerloop.models.candidates import FeatureVector, MatchCandidate
from ledgerloop.models.decisions import MatchDecision
from ledgerloop.models.enums import DecisionOutcome, LinkType, Tier
from ledgerloop.models.refs import bank_ref, payment_ref

WHEN = datetime(2026, 4, 1, 9, 0, 0)
DEFAULTS = DecisionThresholds()


def candidate(
    *, probability: float, verified: bool = True, tier: Tier = Tier.T0_EXACT
) -> MatchCandidate:
    return MatchCandidate(
        candidate_id=f"{tier.name}|test|{probability}|{verified}",
        link_type=LinkType.PAYMENT_CREDITED_AS,
        source_ref=payment_ref("PAY-00001"),
        target_ref=bank_ref("BNK-00001"),
        tier=tier,
        features=FeatureVector(tier=tier),
        calibrated_p=probability,
        arithmetic_verified=verified,
    )


class TestRouting:
    @pytest.mark.parametrize("probability", [1.0, 0.99, 0.95])
    def test_at_or_above_tau_high_auto_matches(self, probability):
        decision = decide(candidate(probability=probability), DEFAULTS, decided_at=WHEN)
        assert decision.outcome is DecisionOutcome.AUTO_MATCHED
        assert decision.is_positive_prediction

    @pytest.mark.parametrize("probability", [0.9499, 0.8, 0.601])
    def test_between_the_thresholds_needs_review(self, probability):
        decision = decide(candidate(probability=probability), DEFAULTS, decided_at=WHEN)
        assert decision.outcome is DecisionOutcome.NEEDS_REVIEW
        assert not decision.is_positive_prediction

    @pytest.mark.parametrize("probability", [0.6, 0.5, 0.0])
    def test_at_or_below_tau_low_is_an_exception(self, probability):
        decision = decide(candidate(probability=probability), DEFAULTS, decided_at=WHEN)
        assert decision.outcome is DecisionOutcome.EXCEPTION
        assert not decision.is_positive_prediction

    def test_a_contested_pair_at_one_half_routes_to_exception(self):
        """The A05 path, reached through the configured policy and no special case.

        Ground truth's verdict for a duplicated credit is ``EXCEPTION``, and a
        uniform prior over two indistinguishable contenders lands there on its
        own -- ``0.5 <= tau_low``.
        """
        decision = decide(candidate(probability=0.5), DEFAULTS, decided_at=WHEN)
        assert decision.outcome is DecisionOutcome.EXCEPTION

    def test_the_thresholds_are_read_from_the_configuration(self):
        permissive = DecisionThresholds(tau_high=0.4, tau_low=0.2)
        decision = decide(candidate(probability=0.5), permissive, decided_at=WHEN)
        assert decision.outcome is DecisionOutcome.AUTO_MATCHED

    def test_the_reason_names_the_rule_that_fired(self):
        decision = decide(candidate(probability=1.0), DEFAULTS, decided_at=WHEN)
        assert "tau_high" in decision.reason
        assert "1.0000" in decision.reason


class TestTheArithmeticGate:
    def test_an_unverified_candidate_is_demoted_however_certain(self):
        decision = decide(
            candidate(probability=1.0, verified=False), DEFAULTS, decided_at=WHEN
        )
        assert decision.outcome is DecisionOutcome.NEEDS_REVIEW
        assert "demoted" in decision.reason
        assert not decision.is_positive_prediction

    def test_the_demotion_keeps_the_probability_it_was_given(self):
        """The gate changes the routing, never the measurement."""
        decision = decide(
            candidate(probability=1.0, verified=False), DEFAULTS, decided_at=WHEN
        )
        assert decision.calibrated_p == 1.0
        assert decision.arithmetic_verified is False

    def test_the_model_refuses_an_unverified_auto_match_outright(self):
        """Belt and braces: the invariant lives on the contract, not the policy."""
        with pytest.raises(ValidationError, match="arithmetic_verified"):
            MatchDecision(
                decision_id="d1",
                candidate_id="c1",
                link_type=LinkType.PAYMENT_CREDITED_AS,
                source_ref=payment_ref("PAY-00001"),
                target_ref=bank_ref("BNK-00001"),
                tier=Tier.T0_EXACT,
                outcome=DecisionOutcome.AUTO_MATCHED,
                calibrated_p=1.0,
                arithmetic_verified=False,
                decided_at=WHEN,
                reason="should not be constructible",
            )

    def test_an_unverified_low_probability_candidate_is_still_an_exception(self):
        decision = decide(
            candidate(probability=0.2, verified=False), DEFAULTS, decided_at=WHEN
        )
        assert decision.outcome is DecisionOutcome.EXCEPTION


class TestIdentityAndReproducibility:
    def test_the_decision_id_derives_from_the_candidate(self):
        item = candidate(probability=1.0)
        assert decision_id(item.candidate_id) in decide(
            item, DEFAULTS, decided_at=WHEN
        ).decision_id

    def test_the_same_candidate_decides_identically_every_time(self):
        item = candidate(probability=1.0)
        first = decide(item, DEFAULTS, decided_at=WHEN)
        second = decide(item, DEFAULTS, decided_at=WHEN)
        assert first == second

    def test_decide_all_preserves_order(self):
        items = [candidate(probability=p) for p in (1.0, 0.5, 0.8)]
        decisions = decide_all(items, DEFAULTS, decided_at=WHEN)
        assert [d.candidate_id for d in decisions] == [i.candidate_id for i in items]

    def test_the_decision_carries_the_candidates_endpoints_and_tier(self):
        item = candidate(probability=1.0, tier=Tier.T1_TOLERANCE)
        decision = decide(item, DEFAULTS, decided_at=WHEN)
        assert decision.pair == item.pair
        assert decision.tier is Tier.T1_TOLERANCE
        assert decision.link_type is item.link_type


class TestPositiveDecisions:
    def test_only_auto_matched_decisions_are_predictions(self):
        decisions = decide_all(
            [candidate(probability=p) for p in (1.0, 0.8, 0.3)], DEFAULTS, decided_at=WHEN
        )
        assert len(positive_decisions(decisions)) == 1
        assert positive_decisions(decisions)[0].outcome is DecisionOutcome.AUTO_MATCHED

    def test_an_empty_log_yields_no_predictions(self):
        assert positive_decisions([]) == ()
