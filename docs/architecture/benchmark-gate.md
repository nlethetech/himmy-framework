# Benchmark PR Gate & Baseline Tracking

> A PR-sized benchmark (a subset of the core suite at enough trials to be honest)
> compared against checked-in metric floors in `benchmarks/baseline.json` — so a real
> agent-quality regression fails the PR instead of waiting for a nightly eyeball.

## Why two benchmark lanes

The [benchmark harness](benchmark.md) runs in two CI lanes
(`.github/workflows/integration.yml`):

| Lane | When | Config | Question it answers |
| --- | --- | --- | --- |
| `benchmark` (nightly + manual) | schedule / `workflow_dispatch` | full core suite (15 tasks) × **10 trials** (pooled n = 150, ~±5% CI) | "Is agent quality drifting over time?" |
| `benchmark-gate` (PR) | every `pull_request` | the `gate` subset (6 tasks) × 5 trials (pooled n = 30, ~10–15 min on a CPU runner) | "Did **this PR** make agents worse?" |

The old setup ran the full suite at 2 trials nightly only: a ±15% CI that could not
see a 5% regression, with no per-PR signal and no baseline to compare against. Now the
nightly trial count is configurable (`workflow_dispatch` input `trials`, default 10;
locally `himmy bench --trials N`), and PRs are gated against recorded floors.

## The baseline file (`benchmarks/baseline.json`)

The single source of truth for the gate. It records:

- `gate` — which core-suite task ids the PR gate runs, with `trials` and `min_trials`
  (a run with fewer trials per task than `min_trials` fails: a thin run can clear
  floors by luck).
- `models` — per benchmarked model: the `measured` values from the baselining run, the
  `floors` derived from them (`accuracy`, `tool_call_accuracy`, optional
  `by_category`), and `ceilings` (`error_rate`).
- `baselined_at` — the git `sha`, `date`, and trial count of the run that produced the
  numbers, so every floor is traceable to a measurement.

Floors are `measured − margin` (default margin `0.15`, clamped to `[0, 1]`); the
margin absorbs trial-to-trial noise so the gate trips on real breakage (a broken tool
loop tanks accuracy toward zero), not on variance. Tighten it as trial counts grow.

## How the gate runs

```
python scripts/bench_gate.py run                  # what CI runs on every PR
python scripts/bench_gate.py check --results r.json   # re-compare a saved run
make bench-gate                                   # the same gate, locally (needs Ollama)
```

`run` loads the baseline, slices the packaged core suite down to `gate.tasks`
(`himmy.benchmark.baseline.subset_suite` — an unknown id is an error, so the gate
config can't silently drift from the suite), runs the real agent loop via
`BenchmarkRunner`, prints the scorecard, and exits non-zero with `FAIL: <model>:
<metric> ... < baseline floor ...` lines on any regression. The compare logic is
`himmy.benchmark.baseline.compare_to_baseline` (pure + unit-tested in
`tests/benchmark/test_baseline.py`); a model listed in the baseline but missing from
the results is also a failure.

## Re-baselining (after an intentional quality change)

```
make bench-rebaseline
```

This re-runs the same gate config and rewrites `benchmarks/baseline.json` from the
fresh measurements (floors = measured − margin), stamped with the current git SHA.
Review the JSON diff like any other code change and commit it — the diff *is* the
documented quality shift. Never hand-edit floors without a measurement to back them.

Calibration notes:

- The checked-in baseline was measured against `ollama:qwen2.5:3b-instruct`; accuracy
  metrics are hardware-independent, but environment differences (e.g. which embedder
  `auto` resolves to for the RAG task) can shift per-category numbers. If the gate's
  first CI runs sit close to a floor, re-baseline from a CI artifact
  (`benchmark-pr-gate-results`) via `python scripts/bench_gate.py run --rebaseline`.
- The nightly lane keeps its independent absolute floor (`--fail-under 0.6`) so the
  two lanes cross-check each other.
