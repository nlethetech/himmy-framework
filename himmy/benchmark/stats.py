"""Statistics for benchmark scoring — confidence intervals + percentiles.

Model runs are non-deterministic, so a single pass/fail is noise. We run each task N
times and report a **Wilson score interval** for the pass rate: a small-sample-correct
95% confidence interval on a binomial proportion (the right tool for "k of n trials
passed"). Latency is summarized with percentiles (p50/p95) — robust to the long tail.
"""

from __future__ import annotations

import math

#: z for a 95% two-sided confidence interval.
Z_95 = 1.959963984540054


def wilson_interval(
    successes: int, trials: int, *, z: float = Z_95
) -> tuple[float, float]:
    """Wilson score confidence interval for ``successes``/``trials`` (a proportion).

    Returns ``(low, high)`` in [0, 1]. Unlike the naive normal interval, it stays
    inside [0, 1] and is accurate for small ``trials`` and rates near 0 or 1.
    """
    if trials <= 0:
        return (0.0, 0.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    margin = (
        z
        / denom
        * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4 * trials * trials))
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated ``q``-quantile (q in [0, 1]) of ``values`` (0.0 if empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def mean(values: list[float]) -> float:
    """Arithmetic mean (0.0 if empty)."""
    return sum(values) / len(values) if values else 0.0


__all__ = ["wilson_interval", "percentile", "mean", "Z_95"]
