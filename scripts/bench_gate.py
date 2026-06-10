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


def _parse_specs(raw_models: str) -> list[Any]:
    """``provider:model[,provider:model...]`` -> ModelSpecs (model may contain ':')."""
    from himmy.benchmark import ModelSpec

    specs = []
    for raw in raw_models.split(","):
        raw = raw.strip()
        if not raw:
            continue
        provider, _, model = raw.partition(":")
        if not model:
            raise SystemExit(f"error: model spec {raw!r} must be provider:model")
        specs.append(ModelSpec(provider=provider, model=model))
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
    specs = (
        _parse_specs(args.models)
        if args.models
        else _parse_specs(",".join(baseline.get("models", {})))
    )
    if not specs:
        raise SystemExit("error: no models (baseline 'models' empty and no --models)")

    def _progress(spec: Any, task: Any, i: int, n: int) -> None:
        _eprint(f"  [{spec.name}] {task.id}  trial {i}/{n}")

    _eprint(
        f"gate: {len(specs)} model(s) on '{suite.name}' "
        f"({len(suite.tasks)} tasks x {trials} trials)…"
    )
    runner = BenchmarkRunner(trials=trials, on_progress=_progress)
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

    args = parser.parse_args(argv)
    if getattr(args, "margin", None) is None and hasattr(args, "margin"):
        from himmy.benchmark.baseline import DEFAULT_MARGIN

        args.margin = DEFAULT_MARGIN
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
