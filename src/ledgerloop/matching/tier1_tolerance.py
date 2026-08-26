"""T1 -- tolerance. The same key, a band instead of an equality.

PLAN.md 6.1 gives T1 as "amount within +/-max(₹1, 0.5%), date within +/-3
days". Every one of those numbers is already a field on
:class:`~ledgerloop.config.MatchingTolerances`, so this module reads them and
hard-codes nothing: change ``amount_bps`` in the config and T1 changes with it,
and the value that produced a reported number travels in the run's
``config_hash``.

WHAT T1 IS ACTUALLY FOR
-----------------------
Exactly one thing, on this corpus: **A02 ``ROUNDING_DRIFT``**, where the bank
credits one to three paise more or less than the declared net. T0's exact
amount declines it; the ₹1 tolerance floor is three hundred times wider than
the drift, so T1 takes it comfortably.

That narrowness is worth stating rather than hiding. The other classes that
survive T0 are not T1's to solve and it must not pretend otherwise:

* **A09 ``SPLIT_PAYOUT``** -- one settlement paid as two tranches. Neither
  tranche is within 0.5% of the whole net (the cuts are tens of percent), so
  T1 declines both. That is the correct answer; the subset arithmetic is T2's.
* **A05 ``DUPLICATE_CREDIT``** -- T0 has already ruled it contested and taken it
  out of the pool, so T1 never sees it. This is the "T1 cannot silently
  override a stronger T0 result" property, and it is structural: the pool
  enforces it, not a convention in this file.
* **A07 ``MISSING_REFERENCE``** -- no UTR in the narration means no key, and T1
  is a keyed tier. T3's fuzzy name matching owns it.

DATE PROXIMITY IS A CONSTRAINT, NOT A SCORE
--------------------------------------------
The window narrows T1 relative to T0, which reads backwards until you see the
trade: T0 buys its confidence with an exact amount and can afford to ignore the
calendar, so it still matches A04 ``TIMING_SHIFT`` and A12 ``LATE_ARRIVAL``
where the money arrived intact but late. T1 has spent that exactness on the
band, so it needs the date back to keep the band from reaching a coincidence.
The anchor is ``settlement.settled_on`` against ``bank_txn.value_date`` --
payout date against value date, which is the pair the edge is actually about.
"""

from __future__ import annotations

from ledgerloop.config import MatchingTolerances
from ledgerloop.matching.bank_leg import BankLegOutcome, BankLegRule, resolve_bank_leg
from ledgerloop.matching.context import MatchContext
from ledgerloop.models.enums import Tier

__all__ = ["rule_for", "run_tier1"]


def rule_for(tolerances: MatchingTolerances) -> BankLegRule:
    """Build T1's admission test from the run configuration.

    Nothing here invents a threshold. ``amount_floor_minor``, ``amount_bps`` and
    ``date_window_days`` come from :class:`~ledgerloop.config.MatchingTolerances`
    and are hashed into ``RunConfig.config_hash``, so a reported metric cannot
    be separated from the band that produced it.
    """
    return BankLegRule(
        tier=Tier.T1_TOLERANCE,
        amount_floor_minor=tolerances.amount_floor_minor,
        amount_bps=tolerances.amount_bps,
        date_window_days=tolerances.date_window_days,
    )


def run_tier1(context: MatchContext, tolerances: MatchingTolerances) -> BankLegOutcome:
    """Run T1 over whatever T0 left in the pool."""
    return resolve_bank_leg(context, rule_for(tolerances))
