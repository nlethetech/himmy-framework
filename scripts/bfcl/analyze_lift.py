"""Compute same-model tool-calling LIFT from paired BFCL runs (RAW vs HIMMY).

This is the stats layer of the harness. It reads the OFFICIAL BFCL per-task verdicts
(written by ``bfcl evaluate``) for a RAW arm and a HIMMY arm on IDENTICAL tasks, pairs
them by task id, and reports per-category:

  * RAW accuracy and HIMMY accuracy (with a Wilson 95% CI on each),
  * LIFT = (HIMMY - RAW) in percentage points,
  * an exact (sign-test) McNemar p-value over the discordant pairs,

reusing himmy's own benchmark stats (``himmy/benchmark/stats.py``) — the same Wilson
and exact-McNemar code himmy uses everywhere else — so the lift numbers are computed
with himmy's audited statistics, while the *grading* stays the official BFCL evaluator.

himmy is a framework, not a model: every number here is "model X RAW" vs "model X under
himmy (same-model lift)". We NEVER print a bare himmy leaderboard row.

How a per-task verdict is recovered
-----------------------------------
``bfcl evaluate`` writes ``score/<model>/<section>/BFCL_v4_<category>_score.json`` as a
JSON-lines file whose FIRST line is the summary header
(``{"accuracy", "correct_count", "total_count"}``) and whose SUBSEQUENT lines are one
record per FAILED task (each carrying an ``id``). So for the subset of task ids that were
actually run (from ``test_case_ids_to_generate.json``), a task PASSED iff it is not in the
failure-id set. We additionally intersect with the ids present in the model's *result*
file, so a task that errored out of generation (no result row) is not silently counted.

Usage
-----
    python scripts/bfcl/analyze_lift.py \
        --raw or-raw-llama3.2-3b-prompting \
        --himmy or-himmy-llama3.2-3b \
        --categories simple_python multiple parallel parallel_multiple irrelevance multi_turn_base

Outputs a human table to stdout and (``--json out.json``) a machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from himmy.benchmark.stats import (  # noqa: E402  (himmy's audited stats)
    mcnemar_from_outcomes,
    wilson_interval,
)

_WORKDIR = Path(__file__).resolve().parent / "workdir"
_VERSION_PREFIX = "BFCL_v4"

# BFCL groups categories into score subdirectories. We only use non-live + multi_turn.
_SECTION = {
    "simple_python": "non_live",
    "simple_java": "non_live",
    "simple_javascript": "non_live",
    "multiple": "non_live",
    "parallel": "non_live",
    "parallel_multiple": "non_live",
    "irrelevance": "non_live",
    "multi_turn_base": "multi_turn",
    "multi_turn_miss_func": "multi_turn",
    "multi_turn_miss_param": "multi_turn",
    "multi_turn_long_context": "multi_turn",
}


def _section_for(category: str) -> str:
    return _SECTION.get(category, "non_live")


def _result_ids(model: str, category: str) -> set[str]:
    """Task ids that have a result row (i.e. were actually generated)."""
    f = (
        _WORKDIR
        / "result"
        / model
        / _section_for(category)
        / f"{_VERSION_PREFIX}_{category}_result.json"
    )
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


def _verdicts(model: str, category: str) -> dict[str, bool]:
    """Map ``task_id -> passed`` from a BFCL score file.

    The score file's first line is the summary header; subsequent lines are the FAILED
    tasks. A task that was generated (has a result row) and is NOT in the failure set
    passed. We pair over the generated-result ids so a generation error is not counted
    as a silent pass.
    """
    score_f = (
        _WORKDIR
        / "score"
        / model
        / _section_for(category)
        / f"{_VERSION_PREFIX}_{category}_score.json"
    )
    if not score_f.exists():
        raise FileNotFoundError(
            f"No score file for {model}/{category}: {score_f}. "
            "Run `bfcl evaluate --partial-eval` for this arm first."
        )
    lines = [ln for ln in score_f.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty score file: {score_f}")
    header = json.loads(lines[0])
    total = header.get("total_count")
    failures: set[str] = set()
    for ln in lines[1:]:
        rec = json.loads(ln)
        tid = rec.get("id")
        if tid is not None:
            failures.add(tid)

    generated = _result_ids(model, category)
    if not generated:
        # Fall back to score header count if result file unreadable; verdicts then
        # cannot be paired by id, so we cannot proceed honestly.
        raise ValueError(
            f"No generated result ids for {model}/{category}; cannot pair verdicts."
        )
    verdicts = {tid: (tid not in failures) for tid in generated}
    # Sanity: the count of passes must equal the header's correct_count when the
    # graded set == the generated set (the normal --partial-eval case).
    n_pass = sum(1 for v in verdicts.values() if v)
    cc = header.get("correct_count")
    if cc is not None and total == len(generated) and n_pass != cc:
        print(
            f"  [warn] {model}/{category}: recovered passes={n_pass} != header "
            f"correct_count={cc} (graded set may differ from generated set)",
            file=sys.stderr,
        )
    return verdicts


def _acc(verdicts: dict[str, bool]) -> tuple[int, int]:
    n = len(verdicts)
    k = sum(1 for v in verdicts.values() if v)
    return k, n


def analyze(raw_model: str, himmy_model: str, categories: list[str]) -> dict:
    rows = []
    agg_raw: dict[tuple[str, int], bool] = {}
    agg_himmy: dict[tuple[str, int], bool] = {}
    for cat in categories:
        try:
            rv = _verdicts(raw_model, cat)
            hv = _verdicts(himmy_model, cat)
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"category": cat, "error": str(exc)})
            continue

        shared = sorted(set(rv) & set(hv))
        if not shared:
            rows.append({"category": cat, "error": "no shared task ids"})
            continue

        rk = sum(1 for t in shared if rv[t])
        hk = sum(1 for t in shared if hv[t])
        n = len(shared)
        raw_acc = rk / n
        himmy_acc = hk / n
        r_lo, r_hi = wilson_interval(rk, n)
        h_lo, h_hi = wilson_interval(hk, n)

        # McNemar over single-trial pairs (trial index 0). himmy's stats keys on
        # (task_id, trial_index); we use trial 0 (temperature 0, one pass).
        oa = {(t, 0): rv[t] for t in shared}
        ob = {(t, 0): hv[t] for t in shared}
        mc = mcnemar_from_outcomes(raw_model, oa, himmy_model, ob)

        # Accumulate for an overall (pooled) line.
        for t in shared:
            agg_raw[(f"{cat}:{t}", 0)] = rv[t]
            agg_himmy[(f"{cat}:{t}", 0)] = hv[t]

        rows.append(
            {
                "category": cat,
                "n": n,
                "raw_correct": rk,
                "himmy_correct": hk,
                "raw_acc": raw_acc,
                "himmy_acc": himmy_acc,
                "raw_ci": [r_lo, r_hi],
                "himmy_ci": [h_lo, h_hi],
                "lift_pp": (himmy_acc - raw_acc) * 100.0,
                "mcnemar_b_raw_only": mc.b,
                "mcnemar_c_himmy_only": mc.c,
                "mcnemar_p": mc.p_value,
                "mcnemar_leader": mc.leader,
                "discordant_task_ids": mc.discordant_task_ids,
            }
        )

    overall = None
    if agg_raw:
        shared = sorted(set(agg_raw) & set(agg_himmy))
        rk = sum(1 for k in shared if agg_raw[k])
        hk = sum(1 for k in shared if agg_himmy[k])
        n = len(shared)
        mc = mcnemar_from_outcomes(raw_model, agg_raw, himmy_model, agg_himmy)
        overall = {
            "n": n,
            "raw_correct": rk,
            "himmy_correct": hk,
            "raw_acc": rk / n,
            "himmy_acc": hk / n,
            "raw_ci": list(wilson_interval(rk, n)),
            "himmy_ci": list(wilson_interval(hk, n)),
            "lift_pp": (hk - rk) / n * 100.0,
            "mcnemar_b_raw_only": mc.b,
            "mcnemar_c_himmy_only": mc.c,
            "mcnemar_p": mc.p_value,
            "mcnemar_leader": mc.leader,
        }

    return {
        "raw_model": raw_model,
        "himmy_model": himmy_model,
        "categories": rows,
        "overall": overall,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_table(report: dict) -> None:
    raw = report["raw_model"]
    him = report["himmy_model"]
    print()
    print(f"SAME-MODEL LIFT  RAW={raw}  vs  HIMMY={him}")
    print("(both arms: same OpenRouter model, same tasks, temperature 0, prompting "
          "mode; only difference = himmy manifest+tolerant-parse+coercion in the loop)")
    print("-" * 108)
    print(
        f"{'category':<20}{'n':>4}  {'RAW acc':>9}  {'HIMMY acc':>10}  "
        f"{'lift pp':>8}  {'b(R>H)':>7}{'c(H>R)':>7}  {'McNemar p':>11}  leader"
    )
    print("-" * 108)
    for row in report["categories"]:
        if "error" in row:
            print(f"{row['category']:<20}  ERROR: {row['error']}")
            continue
        leader = row["mcnemar_leader"] or "tie"
        leader_short = (
            "HIMMY" if leader == him else "RAW" if leader == raw else "tie"
        )
        print(
            f"{row['category']:<20}{row['n']:>4}  "
            f"{_fmt_pct(row['raw_acc']):>9}  {_fmt_pct(row['himmy_acc']):>10}  "
            f"{row['lift_pp']:>+7.1f}  {row['mcnemar_b_raw_only']:>7}{row['mcnemar_c_himmy_only']:>7}  "
            f"{row['mcnemar_p']:>11.4f}  {leader_short}"
        )
    ov = report.get("overall")
    if ov:
        print("-" * 108)
        leader = ov["mcnemar_leader"]
        leader_short = (
            "HIMMY" if leader == him else "RAW" if leader == raw else "tie"
        )
        print(
            f"{'POOLED':<20}{ov['n']:>4}  "
            f"{_fmt_pct(ov['raw_acc']):>9}  {_fmt_pct(ov['himmy_acc']):>10}  "
            f"{ov['lift_pp']:>+7.1f}  {ov['mcnemar_b_raw_only']:>7}{ov['mcnemar_c_himmy_only']:>7}  "
            f"{ov['mcnemar_p']:>11.4f}  {leader_short}"
        )
    print("-" * 108)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="RAW arm registry name")
    ap.add_argument("--himmy", required=True, help="HIMMY arm registry name")
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--json", default=None, help="optional path to write JSON summary")
    args = ap.parse_args()

    report = analyze(args.raw, args.himmy, args.categories)
    print_table(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote JSON summary -> {args.json}")


if __name__ == "__main__":
    main()
