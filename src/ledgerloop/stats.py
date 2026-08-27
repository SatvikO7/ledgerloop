"""Small-sample statistics shared by the evaluator and the calibrator.

One function today, and it lives here rather than in :mod:`ledgerloop.eval.
metrics` -- where Step 2 wrote it -- because Step 7's threshold selection needs
it too. Importing it from ``eval`` inside ``matching`` would point the
dependency arrow backwards: ``eval`` scores what ``matching`` produces, so
``matching`` must not depend on ``eval``. ``eval.metrics`` re-exports both
names, so every existing import keeps working.

Nothing here knows about reconciliation. It is arithmetic on counts.
"""

from __future__ import annotations

from math import sqrt

__all__ = ["Z_95", "wilson_interval"]

#: Two-sided 95% normal quantile.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    **Wilson rather than the normal approximation**, because this project lives
    exactly where the normal approximation fails. At 250 successes out of 250,
    ``p +- z*sqrt(p(1-p)/n)`` collapses to ``[1.0, 1.0]``: it claims certainty
    that the true precision is perfect, from 250 samples. Wilson returns roughly
    ``[0.985, 1.0]``, which is the honest statement -- and an evaluator built to
    make a precision claim credible cannot use the estimator that breaks
    precisely where the claim lives.

    Returns ``(0.0, 1.0)`` on zero trials: with no evidence every proportion
    remains possible, and that is the interval which says so.
    """
    if successes < 0 or trials < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if trials == 0:
        return (0.0, 1.0)

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        z
        / denominator
        * sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4 * trials * trials))
    )
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)

    # At the boundaries the algebra is exact -- a run with no failures has an
    # upper bound of exactly 1, and one with no successes a lower bound of
    # exactly 0 -- but the floating-point evaluation lands a few ulps short
    # (0.9999999999999998 at 250/250). Those two cases are the ones this project
    # reports most often, so they are pinned rather than left to round.
    if successes == trials:
        high = 1.0
    if successes == 0:
        low = 0.0
    return (low, high)
