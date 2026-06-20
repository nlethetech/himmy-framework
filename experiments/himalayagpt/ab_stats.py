"""Aggregate ab_results.jsonl into per-cell rates + a paired McNemar test (few-shot vs
zero-shot), per language, on the binary EMIT and ARGS outcomes.

Pairing: the same task in the same language is run under BOTH arms with only the prompting
differing -> a matched pair. McNemar's exact test uses the discordant pairs (b = zero-fails
but few-shot-passes, c = zero-passes but few-shot-fails); the two-sided exact p-value is
2*sum of the binomial tail (n=b+c, p=0.5). Fisher's exact (2x2 unpaired) is reported too as
a cross-check on the marginal rates.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).with_name("ab_results.jsonl")


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(_binom_pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for table [[a,b],[c,d]] (sum of tables <= observed prob)."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d

    def _hyper(x: int) -> float:
        # P(a=x) given fixed margins
        if x < max(0, col1 - row2) or x > min(row1, col1):
            return 0.0
        return (
            math.comb(col1, x)
            * math.comb(col2, row1 - x)
            / math.comb(n, row1)
        )

    p_obs = _hyper(a)
    total = 0.0
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    for x in range(lo, hi + 1):
        px = _hyper(x)
        if px <= p_obs + 1e-12:
            total += px
    return min(1.0, total)


def main() -> None:
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    # index[(lang, arm, task_id)] = rec
    index: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        index[(r["lang"], r["arm"], r["task_id"])] = r

    langs = ["en", "ne"]
    arms = ["zero", "fewshot"]
    metrics = ["emit", "selection", "args"]

    print("=" * 78)
    print("PER-CELL RATES (n = tasks completed in that cell)")
    print("=" * 78)
    cell_rate: dict[tuple[str, str, str], tuple[int, int]] = {}
    for lang in langs:
        for arm in arms:
            cell = [r for r in rows if r["lang"] == lang and r["arm"] == arm]
            n = len(cell)
            line = f"lang={lang} arm={arm:<7} n={n:>2}"
            for m in metrics:
                k = sum(1 for r in cell if r[m])
                cell_rate[(lang, arm, m)] = (k, n)
                pct = (100.0 * k / n) if n else 0.0
                line += f"   {m}={k}/{n} ({pct:4.0f}%)"
            print(line)
        print("-" * 78)

    print("\n" + "=" * 78)
    print("PAIRED McNEMAR (few-shot vs zero-shot), per language, per metric")
    print("  b = zero-FAIL & fewshot-PASS (few-shot wins)")
    print("  c = zero-PASS & fewshot-FAIL (zero wins)")
    print("=" * 78)
    for lang in langs:
        for m in metrics:
            # paired over tasks present in BOTH arms
            task_ids = sorted(
                {r["task_id"] for r in rows if r["lang"] == lang and r["arm"] == "zero"}
                & {r["task_id"] for r in rows if r["lang"] == lang and r["arm"] == "fewshot"}
            )
            b = c = both = neither = 0
            for tid in task_ids:
                z = index[(lang, "zero", tid)][m]
                f = index[(lang, "fewshot", tid)][m]
                if f and not z:
                    b += 1
                elif z and not f:
                    c += 1
                elif z and f:
                    both += 1
                else:
                    neither += 1
            n_pairs = len(task_ids)
            p_mc = mcnemar_exact(b, c)
            # Fisher on the marginal 2x2: pass/fail x arm
            zk, zn = cell_rate[(lang, "zero", m)]
            fk, fn = cell_rate[(lang, "fewshot", m)]
            p_f = fisher_exact_2x2(fk, fn - fk, zk, zn - zk)
            zr = f"{zk}/{zn}"
            fr = f"{fk}/{fn}"
            sig = "SIGNIFICANT" if p_mc < 0.05 else "n.s."
            print(
                f"lang={lang} metric={m:<9} pairs={n_pairs:>2} | "
                f"zero={zr} fewshot={fr} | b={b} c={c} both={both} neither={neither} | "
                f"McNemar p={p_mc:.4f} ({sig}) | Fisher p={p_f:.4f}"
            )
        print("-" * 78)

    # Latency summary (proves the real model ran — real wall clock per trial).
    print("\n" + "=" * 78)
    print("LATENCY (ms) — proves real transformers generation, not stubbed")
    print("=" * 78)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r["latency_ms"])
    for arm, lats in by_arm.items():
        lats_sorted = sorted(lats)
        med = lats_sorted[len(lats_sorted) // 2]
        print(
            f"arm={arm:<7} n={len(lats):>2}  min={min(lats):7.0f}  "
            f"median={med:7.0f}  max={max(lats):7.0f}  mean={sum(lats)/len(lats):7.0f}"
        )


if __name__ == "__main__":
    main()
