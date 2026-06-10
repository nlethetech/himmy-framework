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
| `graders.py` | `grade(spec, answer)` — deterministic answer graders (`contains`/`not_contains`/`regex`/`exact`/`numeric`/`all_of`/`any_of`); `grade_trajectory(spec, tool_sequence)` — trajectory graders (`first_tool`/`max_tool_calls`/`tool_called`/`tool_not_called`/`tool_sequence`, same `all_of`/`any_of` combinators). |
| `stats.py` | `wilson_interval`, `percentile`, `mean`, `Z_95`, plus the paired-comparison `mcnemar_exact` / `mcnemar_from_outcomes` / `McNemarResult` (pure functions over plain data — no benchmark-model imports, reusable elsewhere). |
| `report.py` | `render_markdown` (comparative scorecard table + pairwise McNemar section + honest per-category reporting) + `to_json` (machine record). |
| `history.py` | Append-only run history at `benchmarks/history.jsonl` (`append_run`, `load_history`, `compute_trends`, `git_sha`). |
| `cache.py` | On-disk scorecard cache at `~/.himmy/benchmarks.json` (`cache_path`, `summarize`, `load_entries`, `save_scorecards`). |
| `team.py` | `build_team_plan(spec)` — parse a task's `team:` block into a `TeamPlan` (built `AgentTeam` + orchestrator selection); fail-loud on a bad topology. |
| `judge.py` | LLM-as-judge grader tier — a thin adapter over `himmy.services.evaluation.metrics.LLMJudgeMetric` (`build_judge`, `judge_answer`, `JudgeVerdict`, `resolve_judge_model`/`SameModelJudgeError`) + the judge-agreement harness (`load_judge_validation`, `run_judge_agreement`, `JudgeAgreement`). Reported, never gated. |
| `suites/core.yaml` | The packaged standing suite (offline, self-contained tasks). 50 tasks across 10 categories — arithmetic, files, sql, multistep, rag, memory, agentic, reasoning, nepal, regression — with **≥5 tasks per category** (so the per-category breakdown above is signal, not noise), varied difficulty (most passable by a 3B–7B local model, 1–2 stretch tasks each: NULL handling, large-file excerpt, counterfactual reasoning, RAG abstention). The six baseline-gate tasks keep their exact definitions; everything else is additive. |
| `suites/skills.yaml` | A capability/skills-focused suite. |
| `suites/multiagent.yaml` | The packaged multi-agent collaboration suite (handoff routing, delegation, no-handoff control, group-chat selection). Reported, not gated. |
| `suites/nepali.yaml` | The packaged Nepali language-understanding suite (Devanagari instruction following, code-switched query, deterministic transliteration round-trip + BS-calendar reasoning, judge-graded summarization). Reported, not gated. |
| `__init__.py` | Re-exports + `default_suite()` (core) + `multiagent_suite()` + `nepali_suite()`. |

## Key abstractions

### `BenchmarkTask` (frozen dataclass)

`id`, `prompt`, `grade: dict` (grader spec), `packs: list[str]` (tool packs),
`skills: list[str]` (capabilities that imply packs + know-how), `expect_tools:
list[str]`, `instructions: list[str]`, `category` (default `"general"`), and
self-contained fixtures: `files: dict[str, str]` and `sqlite: list[str]` (init SQL
statements). Materialized per trial in a temp workspace. Three optional blocks select
non-default code paths: `trajectory: dict` (grade the tool-call sequence — see below),
`team: dict` (run as a multi-agent team instead of a single agent — see
**Multi-agent team tasks**), and `judge: dict` (`{rubric, threshold}` — grade the answer
with an LLM judge instead of a deterministic grader; see **LLM-judge grader tier**).
Empty `team` ⇒ the unchanged single-agent path; empty `judge` ⇒ deterministic grading.
For a judge-tier task `grade` is optional.

### `BenchmarkSuite` (frozen dataclass)

`name`, `tasks`. Built from a mapping (`from_dict`) or YAML (`from_yaml`). The
`packs` property returns the union of all packs any task needs, including those a
skill implies (resolved via `build_skill_registry` + `resolve_skills`).

### `ModelSpec` (frozen dataclass)

One column of the scorecard: `provider`, `model`, `label`, `tool_router: bool`,
`temperature` (default `0.0`), `extra_packs` (distractor tools), `max_turns` (default
6). `name` is `label` or `provider:model`. For judge-tier suites it also carries
`judge_provider` / `judge_model` — the judge model for this candidate (per-run
configurable). The judge model **must differ** from the candidate (`provider:model`);
the runner refuses a same-model judge up front (`SameModelJudgeError`).

### `TrialResult` (dataclass)

The outcome of one task run: `task_id`, `answer`, `tools_called` (tool calls **in call
order, with repeats** — the canonical trajectory), `correct`, `tool_ok` (set-membership
"called the expected tools?", `None` when the task expects no specific tool),
`latency_s`, `turns`, `input_tokens`, `output_tokens`, `cost`, `error`, plus
`answer_ok` / `trajectory_ok` which split `correct` into its two halves
(`correct = answer_ok and trajectory_ok is not False`) so a report can show *why* a
trial failed. `trajectory_ok` is `None` when the task declares no trajectory expectation.
For judge-tier tasks: `judged: bool`, `judge_ok: bool | None` (the verdict; `None` when
not judged or *ungraded*), `judge_score: float | None`, `judge_ungraded: bool` (the third
state — judge timeout / unparseable verdict, distinct from a graded fail), `judge_model`.

### `TaskScore` (dataclass)

Aggregated trials for one task: `n`, `successes`, `pass_rate`, `pass_ci` (Wilson),
`tool_call_rate`, `p50_latency`, `errors`.

### `ModelScorecard` (dataclass)

All task scores for one model with model-level aggregates: `accuracy` (pooled across
every trial), `accuracy_ci` (Wilson), `tool_call_accuracy`, `p50_latency`,
`p95_latency`, `mean_cost`, `error_rate`, `total_trials`, and `by_category()`
(accuracy per task category). **The deterministic aggregates (`accuracy`, `error_rate`,
`by_category`, `total_trials`, `trial_outcomes`) cover the deterministic tier ONLY** —
judge-tier task scores are split off (`deterministic_scores` / `judge_scores`) and
reported separately via `has_judge_tier`, `judge_accuracy` (over *graded* judge trials),
`judge_ungraded`, and `judge_total_trials`. This is what keeps the judge tier out of the
gate.

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

### Trajectory grading (`grade_trajectory`)

A right answer reached the wrong way is still a process failure — the agent called the
wrong first tool, looped on a tool, called a hallucinated/nonexistent tool, or skipped a
required ordering. `grade()` only sees the final text; **trajectory graders** grade the
agent's ordered tool-call sequence (`TrialResult.tools_called`, in call order with
repeats — attempted-but-unknown tool names are included, since the runner records the
model's *attempted* calls regardless of whether the tool is registered):

| `type` | Behaviour |
| --- | --- |
| `first_tool` | The first tool called must equal `value` (fails on an empty sequence). |
| `max_tool_calls` | At most `value` total tool calls (`0` ⇒ no tool may be called) — bounds a runaway loop. |
| `tool_called` | `value` appears somewhere in the sequence. |
| `tool_not_called` | `value` does **not** appear — hallucinated-/forbidden-/unknown-tool detection. |
| `tool_sequence` | `value` (a list of tool names) appears as an ordered **subsequence** (gaps allowed, order preserved). |
| `all_of` / `any_of` | Same combinators as answer graders, over trajectory sub-specs under `of`. |

A task declares a trajectory grader in a separate `trajectory:` block alongside its
`grade:` block; when present, **both** must pass for the trial to count `correct`. A
trajectory predicate placed in an answer `grade` block (or vice-versa) raises — fail
loud, no silent mis-grade. Example:

```yaml
- id: sql_schema_discovery
  category: multistep
  prompt: "Discover the table's columns, then report the value for 'bees'."
  packs: [data]
  grade: { type: numeric, value: 40000 }   # answer grader (final text)
  trajectory:                              # trajectory grader (tool-call sequence)
    type: tool_sequence
    value: [sql_schema, sql_query]         # schema lookup must precede the query
```

The runner records `answer_ok` and `trajectory_ok` separately on each `TrialResult`;
`render_markdown` adds a `Traj✗` column and a **Trajectory failures** section (which
tasks took a bad path) whenever any model has a trajectory failure, and `to_json`
carries per-model and per-task `trajectory_failures`. A trajectory failure is distinct
from a wrong answer (the answer grader) and from a crash (`error`). Two core-suite tasks
ship a trajectory expectation (`sql_schema_discovery`, `agentic_scratchpad`); the six
baseline-gate tasks are deliberately left answer-only so the pinned-model floor is
unaffected.

### Multi-agent team tasks (`team.py`, `suites/multiagent.yaml`)

Multi-agent collaboration is the framework's differentiator, so it is benchmarked as a
suite. A task declares a small 2–3 agent `team:` block instead of running as a single
agent; the runner branches on it — an empty `team` keeps the exact single-agent code
path. `build_team_plan(spec)` parses the block into an `AgentTeam` plus an orchestrator
selection, failing loud (`ValueError`, recorded as a trial error) on a bad mode, an
unknown `entry`, a handoff/delegate edge to a non-member, or an unknown group-chat
selector / speaker.

| `mode` | Orchestrator | Topology |
| --- | --- | --- |
| `handoff` (default) | `MultiAgentOrchestrator` | Peer **handoff** (`handoffs:` → `transfer_to_<peer>`) and supervisor **delegation** (`delegates:` → `ask_<worker>`). |
| `group_chat` | `GroupChatOrchestrator` | A selector-driven panel (`selector: round_robin \| llm`, `max_rounds`, optional `speakers`). |

The orchestrator is constructed **directly** (not via
`AgentEvalHarness.evaluate_team()`): the harness collapses a team run to one answer
string for the eval kernel's 5-metric scoring, but the benchmark needs the per-turn
`RunResult` list — the **ordered tool sequence**, token counts, and cost. Both
orchestrators expose `turns` as `(member, RunResult)`, so the runner reads the
trajectory the same way it does for a single agent
(`[tc.tool_name for _, r in turns for tc in r.tool_calls]`). Crucially the synthetic
collaboration tools (`transfer_to_<peer>`, `ask_<worker>`) appear in that sequence, so
the **trajectory graders** above assert directly on the routing — `tool_called:
transfer_to_math` + `tool_not_called: transfer_to_weather` is a handoff-routing check;
`tool_not_called` on both transfer tools is a no-handoff control. The runner builds a
fresh `ToolRegistry`, registers the task's pack tools, then lets the orchestrator
register its synthetic tools on the same registry at construction (so each trial is
isolated). The team path records the *same* `TrialResult` shape as a single agent —
no special-casing leaks into `report.py`, `history.py`, or the cache, which all
aggregate over `task_scores` / `category` / `trajectory_failures` generically.

```yaml
- id: handoff_routing
  category: multiagent
  prompt: "A user asks: 'What is 6 multiplied by 7?' Route to the right specialist."
  packs: [utils]
  grade: { type: numeric, value: 42 }      # the math specialist's answer
  trajectory:                              # routing assertion on the synthetic tools
    type: all_of
    of:
      - { type: tool_called, value: transfer_to_math }
      - { type: tool_not_called, value: transfer_to_weather }
  team:
    mode: handoff
    entry: router
    members:
      - name: router
        persona: { instructions: ["For math, transfer to the math specialist."] }
        handoffs: [math, weather]
      - name: math
        persona: { instructions: ["Use the calculator, then state the answer."] }
        tools: [calculator]
      - name: weather
        persona: { description: "Wrong target for math questions." }
```

The four packaged tasks are **reported, not gated** — they exercise the orchestrators
but are not in the baseline gate subset until real small-model pass rates are measured.
The default offline stub calls *every* bound tool (so it cannot model *correct*
routing); the offline tests inject a scripted inference stub (the
`tests/benchmark/test_runner.py` pattern) that drives a team turn-by-turn and asserts
graders both ways — a competent trajectory passes, a wrong-handoff trajectory fails the
trajectory grader even with a right answer.

### LLM-judge grader tier (`judge.py`, `suites/nepali.yaml`)

Some tasks have no computable ground truth (open-ended summarization, free-text
reasoning), so a deterministic grader would mis-score them. A task may instead declare a
`judge` block — a natural-language rubric + a pass threshold — graded by an LLM judge:

```yaml
- id: nepali_summarization
  category: nepali
  prompt: "तलको नेपाली अनुच्छेदलाई एक वाक्यमा संक्षेप गर्नुहोस्: …"
  judge:
    threshold: 0.6
    rubric: >
      Score 1.0 only for a single concise NEPALI (Devanagari) sentence that faithfully
      captures the source (capital Kathmandu, bordered by China and India, contains
      Everest). Penalize English answers, multiple sentences, or hallucinated facts.
```

**Adapter, not a duplicate.** `judge.py` is a thin layer over the eval kernel's
`himmy.services.evaluation.metrics.LLMJudgeMetric`: it packs the rubric + task prompt
into a synthetic `EvaluationCase` and scores it through that one metric, so the judge
prompt, structured-output schema, and verdict parsing have a single implementation in the
codebase (the deliberate first step of unifying the bench grader registry with the eval
metric registry). Offline it runs against the deterministic stub manager (or an injected
scripted stub in tests).

Hard properties (enforced in `judge.py`/`runner.py`, not the caller):

- **Judge model is configurable per run** — `ModelSpec.judge_provider`/`judge_model`.
- **Judge ≠ candidate** — `resolve_judge_model` raises `SameModelJudgeError` when the
  judge id equals the candidate `provider:model`; the runner preflights this for every
  candidate before any trial runs (a misconfig fails loud, not as opaque per-trial errors).
- **Reported, never gating** — judge-tier task scores are split off into
  `ModelScorecard.judge_scores`; the deterministic `accuracy`/`by_category`/`error_rate`
  (and therefore `compare_to_baseline`) ignore them entirely. `to_json` puts judge results
  under a separate `judge` key per model; `render_markdown` adds a clearly-labelled
  *Judge tier (LLM-graded — reported, NOT gated)* section that names the judge model.
- **Ungraded ≠ fail** — a judge timeout / unparseable verdict marks the trial *ungraded*
  (`JudgeVerdict.ungraded`); the report shows the ungraded count separately and excludes
  it from the judge pass rate, so a flaky judge never silently counts against a model.

**Judge-agreement harness.** A judge tier is only trustworthy once it agrees with humans.
`benchmarks/judge_validation.jsonl` holds ~10 hand-labelled rows (`task`, `rubric`,
`answer`, `human_label` ∈ `good`/`bad`) with obviously-correct labels.
`run_judge_agreement(rows, metric=, judge_model=)` runs the judge over them and returns a
`JudgeAgreement` (matches / graded / ungraded + an `agreement` fraction).
**Workflow:** before trusting any judge-tier benchmark number, run
`python scripts/bench_gate.py judge-agreement --judge <provider:model> [--min-agreement 0.9]`
and confirm agreement is high; extend the validation set with real model outputs you
hand-label as you go.

### Stats (`stats.py`)

Model runs are non-deterministic, so a single pass/fail is noise. Each task runs N
times and the pass rate is reported with a **Wilson score interval** (`wilson_interval`,
`Z_95 = 1.9599…`) — a small-sample-correct 95% binomial CI that stays inside `[0, 1]`.
Latency uses linear-interpolated `percentile` (p50/p95), robust to the long tail.

### Paired model comparison (McNemar)

Comparing two models on accuracy alone (`64%` vs `58%`) ignores that they may have
passed/failed the *same* trials. The honest test for "is A actually better than B?"
on the same task grid is **McNemar's test**, paired by `(task_id, trial_index)`:

- `stats.mcnemar_exact(b, c)` — the exact two-sided p-value from the discordant
  counts `b` (A passed, B failed) and `c` (A failed, B passed). It uses the
  **binomial / sign-test form**, *not* the chi-square approximation, because trial
  counts are small (chi-square is unreliable when `b + c` is small). With no
  discordant pairs the p-value is `1.0` (no signal); `b == c` is also `1.0`.
- `stats.mcnemar_from_outcomes(label_a, grid_a, label_b, grid_b)` pairs two
  `{(task_id, trial_index): passed}` grids over their **intersection** (trials one
  model ran but the other didn't are ignored) and returns a `McNemarResult`
  (`n_pairs`, `b`, `c`, `p_value`, `leader`, `discordant_task_ids`).
- `models.compare_scorecards(a, b)` is the scorecard-level adapter (extracts each
  card's `trial_outcomes()` grid and calls the above). The stats functions hold no
  benchmark-model imports, so the math is reusable outside the benchmark package.

When a run has 2+ models, `render_markdown` emits a **Pairwise comparison (McNemar,
exact)** table for every model pair — `b / c`, `n_pairs`, exact p-value, and a
plain-English verdict (significant at `0.05`, a tie, or not significant). `to_json`
carries the same data under a `pairwise` key.

### Honest per-category reporting

A per-category percentage off a handful of tasks is noise. `report.py` therefore
renders a percentage **only for categories with ≥ `MIN_CATEGORY_TASKS` (5) tasks**;
smaller categories show per-task pass counts instead (`sql_count 3/3,
sql_aggregate 0/3`). `to_json` keeps the raw `by_category` rates but also ships
`category_counts` (tasks per category) and `min_category_tasks` so any consumer
(Studio Doctor, the cache) can apply the same rule. The packaged `core` suite carries
≥5 tasks in every category precisely so each category renders as a real percentage
rather than a per-task count (`test_core_suite.py` pins this floor).

### Run history & trends

`history.append_run(cards, suite_name=…)` appends one JSON line per `(model, suite)`
to `benchmarks/history.jsonl` after every scored run: the git SHA (via
`git rev-parse`, `null` outside a repo), an ISO timestamp, the model id, suite, trials
per task, per-task pass/fail outcomes, and the aggregate metrics. The runner does this
by default; it is disableable per-runner (`BenchmarkRunner(append_history=False)`) or
globally (`HIMMY_BENCH_NO_HISTORY=1`). Appends are a single `write` to an `"a"`-mode
handle (concurrency-safe), and `load_history` skips a corrupt/truncated line with a
warning rather than crashing.

`history.compute_trends(records, threshold=0.10)` returns per `(model, suite)` the
latest-vs-previous delta of each metric and flags a regression when an accuracy-type
metric drops (or `error_rate` rises) past the threshold. The threshold is an **absolute
percentage-point delta**, not a fraction of the previous value (0.80 → 0.69 is a 0.11
drop). Critically, a flag must also clear a **sample-size-aware noise floor**: two runs
of identical quality at rate `p` over `n` trials produce a between-run delta with SD
`sqrt(2·p·(1−p)/n)`, so a drop only counts when it beats `z·SD` (one-sided 95%) as well
as the absolute threshold. This stops the small packaged suites — the gate run is 6 tasks
× 5 trials = 30 trials, where the 0.10 threshold is only ~1 sigma of noise, and the
multiagent suite is 4 tasks × 3 trials = 12 trials, where noise dominates — from firing a
spurious red REGRESSED on a run with no real change. Records written before `trials` was
recorded (no trial count) fall back to the absolute threshold alone. `python
scripts/bench_gate.py history [--fail-on-regression]` prints the trend report.

### Reporting & caching

- `report.render_markdown` is the human scorecard (accuracy + CI, tool-call, p50/p95,
  cost/trial, errors, honest per-category breakdown, and — for 2+ models — the
  pairwise McNemar section). `report.to_json` is the machine record for tracking
  regressions and diffing configs.
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
  `packs`/`skills`/`expect_tools`/`instructions`/`category`/`files`/`sqlite`/`trajectory`).
- **Grade the tool path:** add a `trajectory:` block (a trajectory grader spec) — both
  it and `grade` must pass for the trial to count.
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
