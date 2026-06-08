# Evaluation Service

> Scores agent outputs against an `EvaluationSuite` with weighted, veto-aware, calibration-aware metrics — offline by default, judge/embedding-capable when configured.

## Overview

The evaluation kernel (`himmy/services/evaluation/`) takes a declarative suite of
input → expected-behaviour cases plus a dict of actual agent outputs, scores every
metric named by each case, and rolls them up into per-case and per-run verdicts.

Two design properties shape everything:

- **Async-capable metrics.** A metric's `score()` may return a `MetricScore`
  synchronously OR return an awaitable that resolves to one. Deterministic baselines
  return synchronously; an LLM-judge or embedding-similarity metric returns a
  coroutine. The service awaits coroutine results and fans case/metric scoring out
  with a bounded `asyncio.gather`, so a judge-backed suite is not O(N) serial.
- **Honest offline baselines.** The five built-ins are deterministic stubs that run
  with no provider and no network. They will mis-score real free text / numbers (the
  source comments say so explicitly); production deployments layer judge/embedding
  metrics on top via the same protocol.

A companion `AgentEvalHarness` (`agent_harness.py`) actually *runs* an agent or team
over each case's input to produce the outputs dict, then delegates to the service.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `EvaluationService.run_suite` — the scoring kernel: weighted aggregate, configurable `case_pass_threshold`, veto/must-pass metrics, suite-level ECE, optional persistence. |
| `metrics.py` | `MetricEvaluator` protocol, the five deterministic baselines, `EmbeddingSimilarityMetric` + `LLMJudgeMetric` (async), `bucketed_ece`, `EvaluationMetricRegistry`, `build_registry`, `default_metric_registry`. |
| `models.py` | Pydantic shapes: `EvaluationCase`, `EvaluationSuite`, `MetricScore`, `EvaluationCaseResult`, `EvaluationRun`. |
| `agent_harness.py` | `AgentEvalHarness` — runs a persona (`evaluate_agent`) or a team (`evaluate_team`) over a suite, then scores the outputs. |
| `__init__.py` | Public re-exports. |

## Key abstractions

### Data shapes (`models.py`)

- **`EvaluationCase`** — `case_id`, `input: dict`, `expected_output: dict`,
  `metric_weights: dict[str, float]`, `metadata: dict`. The expected-output dict
  drives every baseline (e.g. `keywords`, `must_cite_evidence_ids`,
  `disallowed_patterns`, `reference_text`/`answer`, `confidence`).
- **`EvaluationSuite`** — `suite_id`, `name`, `cases: list[EvaluationCase]`.
- **`MetricScore`** — one metric's verdict for one case: `metric`, `score`
  (0.0..1.0), `passed: bool`, `detail: str`.
- **`EvaluationCaseResult`** — per-case roll-up: `metric_scores`, `aggregate`,
  `passed`, `actual_output`.
- **`EvaluationRun`** — `run_id`, `suite_id`, `suite_name`, `aggregate_score`,
  `case_results`, `metadata` (carries derived `calibration_ece` and a `persisted:
  False` flag on storage failure), `created_at`.

### Metric protocol + registry (`metrics.py`)

- **`MetricEvaluator`** (`runtime_checkable` Protocol) — `score(case, actual) ->
  MetricScore | Awaitable[MetricScore]`. The union return type is `ScoreResult`.
- **`EvaluationMetricRegistry`** — name → evaluator lookup with `register` / `get` /
  `names` / `default`. `default()` returns a fresh registry preloaded with all five
  baselines. `default_metric_registry` is the shared module-level default.
- **`build_registry(inference_service=, embedder=)`** — baselines + optional
  `LLMJudgeMetric` (when an inference service is given) + `EmbeddingSimilarityMetric`
  (when an embedder is given).

### Built-in metrics

All five are deterministic and synchronous. The default pass threshold for a single
metric is `0.5` (`_PASS_THRESHOLD`).

| Metric | Name | What it scores |
| --- | --- | --- |
| `AccuracyMetric` | `accuracy` | Fraction of `expected_output` keys (excluding `must_*` keys) that match the actual output exactly. |
| `RelevanceMetric` | `relevance` | Bag-of-words keyword recall: target tokens from `expected_output['keywords']` (or the flattened expected output) present in the actual output. |
| `GroundednessMetric` | `groundedness` | Cited evidence refs (`evidence_refs`/`evidence_ids`) must all be in the allowed set (`must_cite_evidence_ids` or `allowed_evidence_ids`). Any out-of-set ref → 0.0 (hallucination). With `must_cite_evidence_ids`, partial credit = fraction of required refs cited. |
| `SafetyMetric` | `safety` | Pattern check against `expected_output['disallowed_patterns']` or a small default set (`ssn`, `password`, `credit card`). Any match → 0.0. |
| `CalibrationMetric` | `calibration` | Pointwise: `1 - |confidence - correct|`, where `correct` is the accuracy score and `confidence` is `actual['confidence']`. Its `detail` carries `confidence=<f> accuracy=<f>` — parsed later for the suite ECE. |

### Calibration: pointwise + bucketed ECE

`CalibrationMetric` is pointwise (a single sample), so the service additionally
computes a **suite-level bucketed Expected Calibration Error**. `bucketed_ece(samples,
bins=10)` buckets `(confidence, correctness)` pairs into equal-width confidence bins
and returns the sample-weighted average gap between mean confidence and mean accuracy
per bin (ECE in `[0, 1]`, lower is better). The service pairs each case's stated
`confidence` with its `accuracy` (both parsed from the calibration metric's `detail`)
and folds the result into `run.metadata['calibration_ece']`.

### Pluggable judge / embedder metrics (async)

- **`LLMJudgeMetric`** (`llm_judge`) — prompts a model via `STRUCTURED_OUTPUT`
  (schema: `{score: 0..1, rationale?}`) over an injected `InferenceService` and maps
  the verdict to a `MetricScore`. A non-`SUCCESS` response scores 0.0. Offline it
  runs against the deterministic stub manager. Registered only when an inference
  service is supplied.
- **`EmbeddingSimilarityMetric`** (`embedding_similarity`) — cosine similarity of the
  expected reference text vs the actual output, both embedded via an injected
  `EmbedderProtocol` (offline default: `DeterministicEmbedder`). Returns a coroutine.

Both plug in behind the same `MetricEvaluator` protocol, so a suite "upgrades"
scoring simply by naming them in `metric_weights`.

## How it works / data flow

`EvaluationService.run_suite(suite=, actual_outputs=)`:

1. A `Semaphore(concurrency)` (default 8) bounds concurrent case scoring.
2. For each case (`_run_case`):
   - `_score_case` selects the metric names from `case.metric_weights.keys()`, or
     falls back to `["accuracy", "relevance", "groundedness", "safety",
     "calibration"]` when no weights are given. Each metric is looked up in the
     registry; an unknown metric yields a failed `MetricScore("unknown metric")`. The
     metric's result is awaited if it is awaitable. A metric that raises is caught and
     **fails closed** (score 0.0, `passed=False`, a `metric error:` detail). Metrics
     within a case run concurrently.
   - `_weighted_aggregate` combines the scores by `case.metric_weights`. With weights:
     weighted mean over the named metrics (falls back to plain mean if total weight
     ≤ 0). Without weights: plain mean.
   - `_case_threshold` is the per-case pass threshold —
     `case.metadata['pass_threshold']` overrides the service default
     (`case_pass_threshold`, default `0.5`).
   - `_vetoed` is true when any veto/must-pass metric failed. The veto set is the
     service default (`("safety",)`) unioned with `case.metadata['veto_metrics']`.
   - **Case verdict:** `passed = (aggregate >= threshold) and not vetoed`. A failed
     veto metric forces the case to fail regardless of the weighted aggregate.
3. **Run verdict:** `aggregate_score` is the mean of every case's `aggregate` (0.0 for
   an empty suite). The suite-level bucketed ECE is computed and stored in
   `run.metadata['calibration_ece']` when calibration samples exist.
4. **Persistence:** when a `storage_service` is supplied, `save_evaluation_run(run)`
   is called. A failure is **logged, not swallowed**, and stamps
   `run.metadata['persisted'] = False`.

### Harness flow (`AgentEvalHarness`)

- `evaluate_agent(suite, persona, llm_config=, input_key="prompt")` — runs the
  persona on each case's `input[input_key]` via `runtime.run_task_detailed`,
  collecting `result.output_text` into the outputs dict, then scores the suite.
- `evaluate_team(suite, team, tool_registry, ...)` — routes each case through a
  `MultiAgentOrchestrator` and scores the final answers.

## Configuration

`EvaluationService(...)` keyword args:

| Arg | Default | Effect |
| --- | --- | --- |
| `metric_registry` | `None` | Explicit registry. If omitted and `inference_service`/`embedder` is given, `build_registry` is used; else `default_metric_registry`. |
| `storage_service` | `None` | When set, runs are persisted (`save_evaluation_run`). |
| `inference_service` | `None` | Adds `LLMJudgeMetric` to the auto-built registry. |
| `embedder` | `None` | Adds `EmbeddingSimilarityMetric` to the auto-built registry. |
| `concurrency` | `8` | Max concurrent case scoring (floored at 1). |
| `case_pass_threshold` | `0.5` | Default per-case pass threshold for the weighted aggregate. |
| `veto_metrics` | `("safety",)` | Default veto/must-pass set. |

Per-case overrides via `case.metadata`: `pass_threshold` (float) and `veto_metrics`
(list).

## Extension points

- **New metric:** implement the `MetricEvaluator` protocol (sync or async `score`)
  and `registry.register(name, evaluator)`; reference it from `metric_weights`.
- **Judge / embedder:** pass `inference_service=` / `embedder=` to the service (or to
  `build_registry`) to enable the async metrics without touching baselines.
- **Custom registry:** build your own `EvaluationMetricRegistry` and pass it as
  `metric_registry=`.

## Gotchas & invariants

- A failed **veto metric forces a case to fail** even with a high aggregate. `safety`
  is a veto metric by default.
- A **broken or unknown metric fails closed** (0.0) — it never crashes the run.
- The deterministic baselines are honest stubs: exact-key accuracy and bag-of-words
  relevance will mis-score real free text. Layer judge/embedding metrics for prod.
- `accuracy` excludes `expected_output` keys starting with `must_` from comparison
  (those describe constraints, not expected values).
- Persistence failures do not raise; they log and set `metadata['persisted'] =
  False`.
- The suite ECE is only present in `run.metadata` when at least one case carries a
  parseable `calibration` detail (i.e. the case scored the `calibration` metric and
  the actual output had a stated `confidence`).

## Related docs

- [Benchmark](../architecture/benchmark.md) — the model-vs-model harness (a different
  surface from this output-scoring kernel; it has its own deterministic graders).
- [Knowledge](knowledge.md) — `EmbeddingSimilarityMetric` reuses the knowledge
  `EmbedderProtocol`; the knowledge package also ships its own retrieval-quality eval.
- [Context](context.md) — `groundedness` checks evidence refs of the kind produced by
  the context/knowledge layers.
