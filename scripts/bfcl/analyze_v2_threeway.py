"""Three-way qwen BFCL analysis: RAW (v1 baseline, reused) vs HIMMY-v1 vs HIMMY-v2.

Reads ONLY official BFCL evaluator score files off disk (no fabrication). For each
category it recovers per-task pass/fail (header line = summary; subsequent lines =
FAILED task ids; a generated task not in the failure set passed), pairs by task id
across the three arms, and reports:

  rawAcc, himmyV1Acc, himmyV2Acc, liftV2 (v2-raw), improvement (v2-v1),
  McNemar p for (raw vs v2) and (v1 vs v2) via himmy/benchmark/stats.py.

All three arms ran the IDENTICAL fixed 170-task subset (temperature 0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from himmy.benchmark.stats import mcnemar_from_outcomes, wilson_interval  # noqa: E402

_WORKDIR = Path(__file__).resolve().parent / "workdir"
_V1_HIMMY_DIR = _WORKDIR / "_v1_qwen_backup_20260622_185850"
_VP = "BFCL_v4"

_SECTION = {
    "simple_python": "non_live",
    "multiple": "non_live",
    "parallel": "non_live",
    "parallel_multiple": "non_live",
    "irrelevance": "non_live",
    "multi_turn_base": "multi_turn",
}
_CATS = list(_SECTION.keys())

RAW = "or-raw-qwen2.5-7b-prompting"
HIMMY = "or-himmy-qwen2.5-7b"


def _score_path(base: Path, model: str, cat: str) -> Path:
    return base / "score" / model / _SECTION[cat] / f"{_VP}_{cat}_score.json"


def _result_path(base: Path, model: str, cat: str) -> Path:
    return base / "result" / model / _SECTION[cat] / f"{_VP}_{cat}_result.json"


def _result_ids(base: Path, model: str, cat: str) -> set[str]:
    f = _result_path(base, model, cat)
    ids: set[str] = set()
    if not f.exists():
        return ids
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def verdicts(base: Path, model: str, cat: str) -> tuple[dict[str, bool], dict]:
    """Return (task_id -> passed, header). Raises if score file missing/empty."""
    sf = _score_path(base, model, cat)
    if not sf.exists():
        raise FileNotFoundError(str(sf))
    lines = [ln for ln in sf.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty: {sf}")
    header = json.loads(lines[0])
    failures: set[str] = set()
    for ln in lines[1:]:
        rec = json.loads(ln)
        tid = rec.get("id")
        if tid is not None:
            failures.add(tid)
    gen = _result_ids(base, model, cat)
    if not gen:
        raise ValueError(f"no result ids: {model}/{cat}")
    return {tid: (tid not in failures) for tid in gen}, header


def main() -> None:
    per_cat = []
    pool_raw: dict[tuple[str, int], bool] = {}
    pool_v1: dict[tuple[str, int], bool] = {}
    pool_v2: dict[tuple[str, int], bool] = {}
    disk = {"raw": [], "v1": [], "v2": []}

    for cat in _CATS:
        rv, rh = verdicts(_WORKDIR, RAW, cat)
        v1, v1h = verdicts(_V1_HIMMY_DIR, HIMMY, cat)
        v2, v2h = verdicts(_WORKDIR, HIMMY, cat)

        disk["raw"].append((str(_score_path(_WORKDIR, RAW, cat)), rh.get("total_count")))
        disk["v1"].append((str(_score_path(_V1_HIMMY_DIR, HIMMY, cat)), v1h.get("total_count")))
        disk["v2"].append((str(_score_path(_WORKDIR, HIMMY, cat)), v2h.get("total_count")))

        shared = sorted(set(rv) & set(v1) & set(v2))
        n = len(shared)
        rk = sum(1 for t in shared if rv[t])
        k1 = sum(1 for t in shared if v1[t])
        k2 = sum(1 for t in shared if v2[t])
        raw_acc, v1_acc, v2_acc = rk / n, k1 / n, k2 / n

        oa = {(t, 0): rv[t] for t in shared}
        ob1 = {(t, 0): v1[t] for t in shared}
        ob2 = {(t, 0): v2[t] for t in shared}
        mc_lift = mcnemar_from_outcomes(RAW, oa, HIMMY + "-v2", ob2)
        mc_impr = mcnemar_from_outcomes(HIMMY + "-v1", ob1, HIMMY + "-v2", ob2)

        for t in shared:
            pool_raw[(f"{cat}:{t}", 0)] = rv[t]
            pool_v1[(f"{cat}:{t}", 0)] = v1[t]
            pool_v2[(f"{cat}:{t}", 0)] = v2[t]

        per_cat.append({
            "category": cat, "n": n,
            "rawAcc": raw_acc, "himmyV1Acc": v1_acc, "himmyV2Acc": v2_acc,
            "raw_correct": rk, "v1_correct": k1, "v2_correct": k2,
            "liftV2_pp": (v2_acc - raw_acc) * 100,
            "improvement_pp": (v2_acc - v1_acc) * 100,
            "mcnemar_p_lift": mc_lift.p_value,
            "mcnemar_p_improvement": mc_impr.p_value,
            "v2_ci": list(wilson_interval(k2, n)),
        })

    shared = sorted(set(pool_raw) & set(pool_v1) & set(pool_v2))
    n = len(shared)
    rk = sum(1 for k in shared if pool_raw[k])
    k1 = sum(1 for k in shared if pool_v1[k])
    k2 = sum(1 for k in shared if pool_v2[k])
    mc_lift = mcnemar_from_outcomes(RAW, pool_raw, HIMMY + "-v2", pool_v2)
    mc_impr = mcnemar_from_outcomes(HIMMY + "-v1", pool_v1, HIMMY + "-v2", pool_v2)
    pooled = {
        "n": n, "raw_correct": rk, "v1_correct": k1, "v2_correct": k2,
        "rawAcc": rk / n, "himmyV1Acc": k1 / n, "himmyV2Acc": k2 / n,
        "liftV2_pp": (k2 - rk) / n * 100,
        "improvement_pp": (k2 - k1) / n * 100,
        "mcnemar_p_lift": mc_lift.p_value,
        "mcnemar_p_improvement": mc_impr.p_value,
    }

    out = {"per_category": per_cat, "pooled": pooled, "disk": disk}
    print(json.dumps(out, indent=2))
    (_WORKDIR / "analysis_v2_qwen_threeway.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
