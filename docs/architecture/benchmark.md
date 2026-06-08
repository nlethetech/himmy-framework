# Benchmark Harness

> Run a fixed task suite against any number of models/configs over N trials and get a per-model scorecard — accuracy with a Wilson 95% CI, tool-call accuracy, latency percentiles, cost — so model and config decisions are measured, not guessed.

## Overview

`himmy/benchmark/` is the model benchmark: a declarative `BenchmarkSuite` of graded
tasks is run for each `ModelSpec` over N trials, producing `TrialResult`s that
aggregate into a `TaskScore` per task and a `ModelScorecard` per model.

It is distinct from the [evaluation service](../services/evaluation.md): evaluation
*scores agent outputs* against expectations; the benchmark *runs the real agent loop*
against fixed tasks and grades the answers with deterministic, model-free graders. It
answers questions like "is the 7B worth it?" and "did the tool router help?".

Every trial is isolated — a fresh runtime, the task's tool packs plus optional
distractor packs, and self-contained `files`/`sqlite` fixtures materialized in a
throwaway workspace — so results are reproducible and state can't leak between tasks.

## Module map

| File | Responsibility |
| --- | --- |
| `runner.py` | `BenchmarkRunner` — per `ModelSpec × task × trial`: build a fresh runtime + fixtures, run the agent loop, grade, record metrics. |
| `models.py` | `BenchmarkTask`, `BenchmarkSuite`, `ModelSpec`, `TrialResult`, `TaskScore`, `ModelScorecard` (with all aggregate properties). |
| `graders.py` | `grade(spec, answer)` — deterministic correctness graders (`contains`/`not_contains`/`regex`/`exact`/`numeric`/`all_of`/`any_of`). |
| `stats.py` | `wilson_interval`, `percentile`, `mean`, `Z_95`. |
| `report.py` | `render_markdown` (comparative scorecard table) + `to_json` (machine record). |
| `cache.py` | On-disk scorecard cache at `~/.himmy/benchmarks.json` (`cache_path`, `summarize`, `load_entries`, `save_scorecards`). |
| `suites/core.yaml` | The packaged standing suite (offline, self-contained tasks across categories). |
| `suites/skills.yaml` | A capability/skills-focused suite. |
| `__init__.py` | Re-exports + `default_suite()` (loads `suites/core.yaml`). |

## Key abstractions

### `BenchmarkTask` (frozen dataclass)

`id`, `prompt`, `grade: dict` (grader spec), `packs: list[str]` (tool packs),
`skills: list[str]` (capabilities that imply packs + know-how), `expect_tools:
list[str]`, `instructions: list[str]`, `category` (default `"general"`), and
self-contained fixtures: `files: dict[str, str]` and `sqlite: list[str]` (init SQL
statements). Materialized per trial in a temp workspace.

### `BenchmarkSuite` (frozen dataclass)

`name`, `tasks`. Built from a mapping (`from_dict`) or YAML (`from_yaml`). The
`packs` property returns the union of all packs any task needs, including those a
skill implies (resolved via `build_skill_registry` + `resolve_skills`).

### `ModelSpec` (frozen dataclass)

One column of the scorecard: `provider`, `model`, `label`, `tool_router: bool`,
`temperature` (default `0.0`), `extra_packs` (distractor tools), `max_turns` (default
6). `name` is `label` or `provider:model`.

### `TrialResult` (dataclass)

The outcome of one task run: `task_id`, `answer`, `tools_called`, `correct`, `tool_ok`
(`None` when the task expects no specific tool), `latency_s`, `turns`, `input_tokens`,
`output_tokens`, `cost`, `error`.

### `TaskScore` (dataclass)

Aggregated trials for one task: `n`, `successes`, `pass_rate`, `pass_ci` (Wilson),
`tool_call_rate`, `p50_latency`, `errors`.

### `ModelScorecard` (dataclass)

All task scores for one model with model-level aggregates: `accuracy` (pooled across
every trial), `accuracy_ci` (Wilson), `tool_call_accuracy`, `p50_latency`,
`p95_latency`, `mean_cost`, `error_rate`, `total_trials`, and `by_category()`
(accuracy per task category).

## How it works / data flow

`BenchmarkRunner(trials=, on_progress=, clock=, inference_factory=).run(suite,
specs)`:

1. For each `spec`, for each `task`, run `trials` trials via `_run_trial`, collect
   into a `TaskScore`, then bundle all task scores into a `ModelScorecard`.
2. `_run_trial(spec, task)`:
   - `_fixtures(task)` materializes `files`/`sqlite` in a `TemporaryDirectory`
     (yielding a `ToolkitConfig` with `fs_root`/`fs_allow_write`/`sqlite_path` set
     when fixtures exist; otherwise the env config unchanged).
   - `_build_runtime(spec, task, config)` builds inference (via `inference_factory(spec)`
     when injected, else `build_inference_for(spec.provider, spec.model)`), resolves
     the task's skill bundle, dedups `task.packs + skill_packs + spec.extra_packs`,
     registers them on a fresh `ToolRegistry`, and calls `build_runtime(...)`.
   - Task `instructions` are extended with the skill bundle's instruction blocks and
     formatted examples, then the agent loop runs via `run_agent_loop(...)` with
     `LLMConfig(model_key="default", temperature=spec.temperature)`,
     `max_turns=spec.max_turns`, `route_tools=spec.tool_router`.
   - Metrics captured: `correct = grade(task.grade, answer)`; `tool_ok = set(expect_tools)
     <= set(called)` (or `None` when no tools are expected); latency, turns, tokens,
     cost, error.
   - **Any exception is a recorded (not raised) trial failure** — a `TrialResult` with
     `correct=False` and the error string.
3. `render_markdown(cards)` prints a comparative table (models sorted by accuracy
   desc) plus an accuracy-by-category table; `to_json(cards)` is the persistable
   machine record.

### Graders (`graders.py`)

`grade(spec, answer) -> bool`. The defaults are **rule-based on purpose** — pure,
fast, reproducible, no model calls — so a task's pass/fail is objective run-to-run:

| `type` | Behaviour |
| --- | --- |
| `contains` / `not_contains` | Case-insensitive substring (in/out). |
| `regex` | Case-insensitive `re.search`. |
| `exact` | Normalized equality (lowercased, whitespace-collapsed). |
| `numeric` | Any extracted number within `tolerance` of `value` (handles thousands separators). |
| `all_of` / `any_of` | Combinators over a list under `of` (a missing/empty `of` raises, so a malformed grade can't vacuously pass). |

For genuinely open-ended tasks an `llm_judge` grader can be layered on top *by the
caller* — the built-in `grade()` itself is rule-based and raises on an unknown type.

### Stats (`stats.py`)

Model runs are non-deterministic, so a single pass/fail is noise. Each task runs N
times and the pass rate is reported with a **Wilson score interval** (`wilson_interval`,
`Z_95 = 1.9599…`) — a small-sample-correct 95% binomial CI that stays inside `[0, 1]`.
Latency uses linear-interpolated `percentile` (p50/p95), robust to the long tail.

### Reporting & caching

- `report.render_markdown` is the human scorecard (accuracy + CI, tool-call, p50/p95,
  cost/trial, errors, per-category breakdown). `report.to_json` is the machine record
  for tracking regressions and diffing configs.
- `cache.save_scorecards` upserts a compact summary per `(model, suite)` into
  `~/.himmy/benchmarks.json` so Studio's Doctor can answer "is this local model good
  enough?" without re-running a slow benchmark. The CLI writes this best-effort (never
  fails the run on a cache error).

## Configuration

`BenchmarkRunner` knobs: `trials` (default 3; ≥1 enforced — more trials → tighter CI),
`on_progress(spec, task, trial_index, total)`, `clock` (defaults to
`time.perf_counter`), `inference_factory(spec) -> InferenceService` (tests inject a
deterministic one).

A `ModelSpec` carries the per-model config (`tool_router`, `temperature`,
`extra_packs`, `max_turns`).

## CI integration & floor-based gating

The standing benchmark runs in `.github/workflows/integration.yml` (job `benchmark`):

- It runs **post-merge / nightly / manual, NOT on PRs** (`if: github.event_name !=
  'pull_request'`) — CPU Ollama is too slow to gate every PR.
- It installs himmy with the `[dev,nepal,knowledge]` extras, starts Ollama, pulls
  `qwen2.5:3b-instruct`, then runs:

  ```
  python -m himmy bench \
    --models ollama:qwen2.5:3b-instruct \
    --trials 2 \
    --fail-under 0.6 \
    --json bench-results.json
  ```

- `--fail-under` is a **floor, not a target**. The CLI gate (`himmy/cli/commands.py`)
  fails (exit 1) if *any* model's pooled `accuracy < floor`, printing `FAIL: <model>
  accuracy X% < floor Y%`; otherwise it prints `OK: all models ≥ Y% floor`. The intent
  is to catch a real regression (a broken tool loop tanks accuracy well below the
  floor); the per-category table is the diagnostic, not the gate.
- `bench-results.json` is uploaded as an artifact (`if: always()`).

## Extension points

- **New task:** add a YAML entry to a suite (`id`, `prompt`, `grade`, optional
  `packs`/`skills`/`expect_tools`/`instructions`/`category`/`files`/`sqlite`).
- **New suite:** write a YAML file and load via `BenchmarkSuite.from_yaml`.
- **New grader type:** extend `grade()` in `graders.py` (open-ended → caller-supplied
  `llm_judge`).
- **Custom inference:** pass `inference_factory=` to the runner.
- **Custom model column:** add a `ModelSpec` (toggle `tool_router`, add `extra_packs`
  distractors, change `temperature`/`max_turns`).

## Gotchas & invariants

- A crash inside a trial is **recorded as a failed `TrialResult`, never raised** — one
  bad task can't abort the whole benchmark.
- `tool_ok` is `None` (excluded from tool-call accuracy) when a task declares no
  `expect_tools`.
- `accuracy` on a scorecard is **pooled across every trial** (not a mean of per-task
  pass rates).
- `all_of`/`any_of` graders **must** have a non-empty `of`, or `grade()` raises.
- Fixtures are per-trial and isolated; a task without `files`/`sqlite` reuses the env
  `ToolkitConfig` unchanged.
- The CI floor gate keys on per-model **accuracy** only (not tool-call accuracy or
  latency).

## Related docs

- [Evaluation service](../services/evaluation.md) — output-scoring kernel (sibling, not
  the same harness).
- [Knowledge](../services/knowledge.md) — its `[knowledge]` extra is installed in the
  benchmark CI lane; the knowledge package has its own retrieval-quality eval.
