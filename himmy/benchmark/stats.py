"""Statistics for benchmark scoring — confidence intervals + percentiles.

Model runs are non-deterministic, so a single pass/fail is noise. We run each task N
times and report a **Wilson score interval** for the pass rate: a small-sample-correct
95% confidence interval on a binomial proportion (the right tool for "k of n trials
passed"). Latency is summarized with percentiles (p50/p95) — robust to the long tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of an exact (sign-test) McNemar comparison of two paired classifiers.

    ``b`` = pairs where A passed and B failed; ``c`` = pairs where A failed and B
    passed. Only the *discordant* pairs (``b + c``) carry signal — concordant pairs
    (both pass or both fail) say nothing about which model is better. ``p_value`` is
    the exact two-sided binomial probability under H0 (a discordant pair is equally
    likely to favour either model). ``leader`` names the model with more wins among
    discordant pairs (``None`` on a tie or when there are no discordant pairs).
    """

    n_pairs: int
    b: int
    c: int
    p_value: float
    leader: str | None = None
    discordant_task_ids: list[str] = field(default_factory=list)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided p-value of McNemar's test from discordant counts ``b``/``c``.

    Uses the **binomial (sign-test) form**, correct for the small trial counts a
    benchmark produces — not the chi-square approximation, which is unreliable when
    ``b + c`` is small. Under H0 each of the ``n = b + c`` discordant pairs is a fair
    coin; the two-sided p-value is ``min(1, 2 * P(X <= min(b, c)))`` for
    ``X ~ Binomial(n, 0.5)``. With no discordant pairs the result is ``1.0`` (no
    evidence of a difference).
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) for X ~ Binomial(n, 0.5) == sum of C(n, i) / 2**n for i in 0..k.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def mcnemar_from_outcomes(
    label_a: str,
    outcomes_a: dict[tuple[str, int], bool],
    label_b: str,
    outcomes_b: dict[tuple[str, int], bool],
) -> McNemarResult:
    """Pair two models' per-trial pass/fail grids by ``(task_id, trial_index)``.

    ``outcomes_*`` map ``(task_id, trial_index) -> passed``. Only keys present in
    **both** grids are paired (the intersection); trials one model ran and the other
    did not are ignored, so a partial overlap still yields an honest comparison over
    the shared trials. Pairing is deterministic (sorted keys). The returned
    :class:`McNemarResult` carries the discordant counts, the exact two-sided
    p-value, the leader, and the task ids that contributed any discordant pair (for
    diagnosing *where* the models diverge).
    """
    shared = sorted(set(outcomes_a) & set(outcomes_b))
    b = c = 0
    discordant: list[str] = []
    for key in shared:
        pa, pb = outcomes_a[key], outcomes_b[key]
        if pa and not pb:
            b += 1
            discordant.append(key[0])
        elif pb and not pa:
            c += 1
            discordant.append(key[0])
    p = mcnemar_exact(b, c)
    leader = label_a if b > c else label_b if c > b else None
    # De-dup task ids while preserving first-seen order (a task can be discordant on
    # several trials but should appear once in the diagnostic list).
    seen: dict[str, None] = {}
    for tid in discordant:
        seen.setdefault(tid, None)
    return McNemarResult(
        n_pairs=len(shared),
        b=b,
        c=c,
        p_value=p,
        leader=leader,
        discordant_task_ids=list(seen),
    )


__all__ = [
    "wilson_interval",
    "percentile",
    "mean",
    "Z_95",
    "McNemarResult",
    "mcnemar_exact",
    "mcnemar_from_outcomes",
]
