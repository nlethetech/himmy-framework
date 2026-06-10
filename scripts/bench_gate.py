#!/usr/bin/env python3
"""PR benchmark gate: run a small, honest benchmark and compare it to the baseline.

The nightly ``himmy bench`` run (15 tasks x >= 10 trials) tracks quality over time but
is too slow to gate every PR. This script is the PR-sized version: it runs the task
subset + trial count declared in ``benchmarks/baseline.json`` (the ``gate`` block)
against the real model and FAILS (exit 1) if any metric drops below the checked-in
floors — so "did this change make agents worse?" blocks the merge, not the postmortem.

The gate/compare logic lives in :mod:`himmy.benchmark.baseline` (tested); this is the
thin CLI around it.

Usage:
    python scripts/bench_gate.py run                       # gate against the baseline
    python scripts/bench_gate.py run --json out.json       # also persist full results
    python scripts/bench_gate.py run --rebaseline --sha $(git rev-parse --short=12 HEAD)
    python scripts/bench_gate.py check --results out.json  # re-compare a saved run

Re-baselining (after an intentional quality change) is `make bench-rebaseline`; review
the diff of benchmarks/baseline.json like any other code change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASELINE = _REPO_ROOT / "benchmarks" / "baseline.json"


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _parse_specs(
    raw_models: str,
    *,
    judge_model: str | None = None,
    judge_provider: str | None = None,
) -> list[Any]:
    """``provider:model[,provider:model...]`` -> ModelSpecs (model may contain ':').

    ``judge_model``/``judge_provider`` configure the LLM-judge tier per candidate (the
    judge must differ from each candidate; unset ⇒ judge-tier trials recorded ungraded).
    """
    from himmy.benchmark import ModelSpec

    specs = []
    for raw in raw_models.split(","):
        raw = raw.strip()
        if not raw:
            continue
        provider, _, model = raw.partition(":")
        if not model:
            raise SystemExit(f"error: model spec {raw!r} must be provider:model")
        specs.append(
            ModelSpec(
                provider=provider,
                model=model,
                judge_provider=judge_provider or "",
                judge_model=judge_model or "",
            )
        )
    return specs


def _gate_or_die(baseline: dict[str, Any]) -> dict[str, Any]:
    gate = baseline.get("gate") or {}
    if not gate.get("tasks"):
        raise SystemExit(
            "error: baseline has no gate.tasks — nothing to run; see "
            "docs/architecture/benchmark-gate.md"
        )
    return gate


def _check(results: dict[str, Any], baseline: dict[str, Any]) -> int:
    from himmy.benchmark import compare_to_baseline

    failures = compare_to_baseline(results, baseline)
    if failures:
        at = baseline.get("baselined_at") or {}
        for msg in failures:
            _eprint(f"FAIL: {msg}")
        _eprint(
            f"(baseline from {at.get('sha', '?')} on {at.get('date', '?')}; "
            "if the change is an intentional quality shift, run "
            "`make bench-rebaseline` and commit the diff)"
        )
        return 1
    _eprint("OK: all metrics clear the baseline floors")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the gate subset, then compare (default) or rewrite the baseline."""
    from himmy.benchmark import (
        BenchmarkRunner,
        build_baseline,
        default_suite,
        load_baseline,
        render_markdown,
        subset_suite,
        to_json,
    )

    baseline = load_baseline(args.baseline)
    gate = _gate_or_die(baseline)
    suite = subset_suite(default_suite(), [str(t) for t in gate["tasks"]])
    trials = int(args.trials or gate.get("trials", 5))
    raw_models = args.models or ",".join(baseline.get("models", {}))
    specs = _parse_specs(
        raw_models,
        judge_model=getattr(args, "judge_model", None),
        judge_provider=getattr(args, "judge_provider", None),
    )
    if not specs:
        raise SystemExit("error: no models (baseline 'models' empty and no --models)")

    def _progress(spec: Any, task: Any, i: int, n: int) -> None:
        _eprint(f"  [{spec.name}] {task.id}  trial {i}/{n}")

    _eprint(
        f"gate: {len(specs)} model(s) on '{suite.name}' "
        f"({len(suite.tasks)} tasks x {trials} trials)…"
    )
    # The PR gate is a freshness check (does this change clear the baseline floor?), not a
    # tracked time-series run — like the Studio probe, it must NOT append to the committed
    # benchmarks/history.jsonl (which would dirty the working tree on every local gate run
    # and write machine-specific noise into the repo). The nightly `himmy bench` owns the
    # history series. (Finding 17.)
    runner = BenchmarkRunner(trials=trials, on_progress=_progress, append_history=False)
    cards = asyncio.run(runner.run(suite, specs))
    print(render_markdown(cards, suite_name=suite.name))
    results = to_json(cards, suite_name=suite.name)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        _eprint(f"wrote full results to {args.json}")

    if args.rebaseline:
        gate.setdefault("min_trials", trials)
        gate["trials"] = trials
        fresh = build_baseline(results, sha=args.sha, margin=args.margin, gate=gate)
        Path(args.baseline).write_text(
            json.dumps(fresh, indent=2) + "\n", encoding="utf-8"
        )
        _eprint(f"rebaselined {args.baseline} (sha {args.sha}) — review + commit it")
        return 0
    return _check(results, baseline)


def cmd_check(args: argparse.Namespace) -> int:
    """Compare an already-saved results JSON against the baseline (no model run)."""
    from himmy.benchmark import load_baseline

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    return _check(results, load_baseline(args.baseline))


def cmd_history(args: argparse.Namespace) -> int:
    """Show per-model trend deltas (latest vs previous run) and flag regressions."""
    from himmy.benchmark import compute_trends, load_history

    records = load_history()
    if not records:
        _eprint("no benchmark history yet (benchmarks/history.jsonl is empty)")
        return 0
    trends = compute_trends(records, threshold=args.threshold)
    regressed = False
    for trend in trends:
        suite = f"/{trend.suite}" if trend.suite else ""
        if trend.previous_when is None:
            print(f"{trend.model}{suite}: first run ({trend.latest_when}) — no trend")
            continue
        print(f"{trend.model}{suite}: {trend.previous_when} -> {trend.latest_when}")
        for mt in trend.trends:
            if mt.delta is None:
                continue
            flag = "  REGRESSED" if mt.regressed else ""
            print(
                f"    {mt.metric:18s} {mt.previous:.3f} -> {mt.latest:.3f} "
                f"(Δ {mt.delta:+.3f}){flag}"
            )
        if trend.regressed:
            regressed = True
    if regressed and args.fail_on_regression:
        _eprint("FAIL: one or more metrics regressed beyond the threshold")
        return 1
    return 0


def cmd_judge_agreement(args: argparse.Namespace) -> int:
    """Run the judge over the hand-labelled validation set and report agreement.

    Trust the judge tier ONLY after this agreement is high — a judge that disagrees with
    obviously-correct human labels can't be trusted to grade open-ended benchmark tasks.
    Extend ``benchmarks/judge_validation.jsonl`` with real model outputs you hand-label.
    """
    from himmy.benchmark.judge import (
        build_judge,
        load_judge_validation,
        run_judge_agreement,
    )
    from himmy.benchmark.models import ModelSpec

    provider, _, model = (args.judge or "").partition(":")
    if not model:
        raise SystemExit(
            "error: --judge must be provider:model (e.g. ollama:qwen2.5:3b)"
        )
    # The judge spec is standalone here (no candidate), so point judge_* at itself and
    # build the metric against a deterministic-or-real judge inference.
    spec = ModelSpec(
        provider=provider,
        model=f"__candidate__:{model}",  # distinct from the judge id (judge != candidate)
        judge_provider=provider,
        judge_model=model,
    )
    metric, judge_id = build_judge(spec)
    rows = load_judge_validation(args.validation)
    result = asyncio.run(run_judge_agreement(rows, metric=metric, judge_model=judge_id))
    for row in result.rows:
        mark = "OK " if row["matched"] else ("?? " if row["ungraded"] else "XX ")
        print(
            f"  {mark} label={row['human_label']:4s} "
            f"judge={row['judge_passed']} score={row['judge_score']:.2f}  "
            f"{str(row['task'])[:60]}"
        )
    print(
        f"\njudge={judge_id}  agreement={result.agreement:.0%} "
        f"({result.matches}/{result.graded} graded matched; "
        f"{result.ungraded} ungraded of {result.n})"
    )
    if args.min_agreement is not None and result.agreement < args.min_agreement:
        _eprint(
            f"FAIL: judge agreement {result.agreement:.0%} < "
            f"required {args.min_agreement:.0%}"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the gate benchmark and compare/rebaseline")
    p_run.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    p_run.add_argument(
        "--models", help="comma list of provider:model (default: the baseline's)"
    )
    p_run.add_argument(
        "--trials", type=int, default=None, help="override the gate's trials per task"
    )
    p_run.add_argument(
        "--judge-model",
        help="judge model for any LLM-judge-tier tasks (must differ from each "
        "candidate); unset ⇒ judge-tier trials recorded ungraded",
    )
    p_run.add_argument(
        "--judge-provider",
        help="provider for --judge-model (default: each candidate's own provider)",
    )
    p_run.add_argument("--json", help="also write the full results as JSON")
    p_run.add_argument(
        "--rebaseline",
        action="store_true",
        help="rewrite the baseline from this run instead of gating against it",
    )
    p_run.add_argument(
        "--margin",
        type=float,
        default=None,
        help="floor margin below measured values when rebaselining",
    )
    p_run.add_argument(
        "--sha", default="unknown", help="git SHA to record when rebaselining"
    )
    p_run.set_defaults(func=cmd_run)

    p_check = sub.add_parser("check", help="compare saved results to the baseline")
    p_check.add_argument("--baseline", default=str(_DEFAULT_BASELINE))
    p_check.add_argument("--results", required=True, help="a bench --json output file")
    p_check.set_defaults(func=cmd_check)

    p_hist = sub.add_parser(
        "history", help="show per-model trend deltas from benchmarks/history.jsonl"
    )
    p_hist.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="regression threshold (metric drop fraction); default 0.10",
    )
    p_hist.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any tracked metric regressed beyond the threshold",
    )
    p_hist.set_defaults(func=cmd_history)

    p_judge = sub.add_parser(
        "judge-agreement",
        help="run the judge over benchmarks/judge_validation.jsonl and report agreement",
    )
    p_judge.add_argument("--judge", required=True, help="judge model as provider:model")
    p_judge.add_argument(
        "--validation",
        default=None,
        help="validation JSONL (default: benchmarks/judge_validation.jsonl)",
    )
    p_judge.add_argument(
        "--min-agreement",
        type=float,
        default=None,
        help="exit 1 if judge agreement falls below this fraction (e.g. 0.9)",
    )
    p_judge.set_defaults(func=cmd_judge_agreement)

    args = parser.parse_args(argv)
    if getattr(args, "margin", None) is None and hasattr(args, "margin"):
        from himmy.benchmark.baseline import DEFAULT_MARGIN

        args.margin = DEFAULT_MARGIN
    if getattr(args, "threshold", None) is None and hasattr(args, "threshold"):
        from himmy.benchmark.history import DEFAULT_REGRESSION_THRESHOLD

        args.threshold = DEFAULT_REGRESSION_THRESHOLD
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
