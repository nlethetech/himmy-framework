# BFCL same-model lift harness (OpenRouter)

This directory holds a **reproducible** integration of the Berkeley Function
Calling Leaderboard (`bfcl-eval`) with **OpenRouter**, used to measure himmy's
**same-model tool-calling lift**: run ONE model TWICE on identical tasks/seed —
once RAW (the model on its own) and once routed through himmy — and report the
delta. himmy is a *framework, not a model*; every number is reported as
"model X RAW" vs "model X under himmy (same-model lift)", never a bare himmy
leaderboard row.

This README documents the **RAW (baseline) arm** and the **HIMMY (treatment)
arm**, both built and verified here. The lift-stats glue is layered on top and
reuses the same registration + workdir machinery.

## Layout

| file | role |
|------|------|
| `handlers.py` | `OpenRouterRawHandler` — RAW arm. Thin subclass of BFCL's stock `OpenAICompletionsHandler` pointed at `https://openrouter.ai/api/v1`. **Imports no himmy code** (this is the apples-to-apples control). |
| `handlers_himmy.py` | `HimmyOpenRouterHandler` — HIMMY arm. SAME OpenRouter model + transport, but tool manifest render + AST/execute decode + multi-turn result threading all routed through himmy's tool-call format registry (`himmy.services.inference.tool_formats`). Logs `[himmy-arm]` markers + asserts the format came from himmy. |
| `register.py` | Registers the RAW OpenRouter models into BFCL's `MODEL_CONFIG_MAPPING` at runtime (no site-packages edits). Idempotent. |
| `register_himmy.py` | Registers the HIMMY OpenRouter models (same models, himmy handler, prompting mode). Idempotent. |
| `sitecustomize.py` | Interpreter-startup hook: loads `OPENROUTER_API_KEY` from the repo `.env` (never printed), then calls `register.register()` (and `register_himmy.register()` if present). Auto-imported by Python when `scripts/bfcl/` is on `PYTHONPATH`. |
| `run.sh` | Driver: pins the bfcl venv, puts `scripts/bfcl/` on `PYTHONPATH`, sets `BFCL_PROJECT_ROOT=workdir`, copies the OpenRouter key into the workdir `.env`, then `exec`s `bfcl "$@"`. |
| `run_matrix.sh` | Orchestrates ONE model's full paired run: `generate`+`evaluate` BOTH arms over the bounded subset, then prints the lift table. `scripts/bfcl/run_matrix.sh <raw_model> <himmy_model> [categories...]`. |
| `analyze_lift.py` | Reads the OFFICIAL BFCL per-task verdicts for a RAW arm + a HIMMY arm, pairs by task id, and reports per-category RAW/HIMMY accuracy (Wilson 95% CI), LIFT in pp, and an exact McNemar p — all via `himmy/benchmark/stats.py`. Writes a JSON summary. |
| `workdir/` | `BFCL_PROJECT_ROOT` — all `result/`, `score/`, `.env`, and `test_case_ids_to_generate.json` live in-repo here for reproducibility. |

## Why no site-packages edits

`bfcl-eval` has **no plugin / entry-point mechanism**. Handlers are resolved
from the module-level dict `bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING`,
which every BFCL module imports *by reference*. So we register custom models by
**mutating that dict before the CLI reads it**. Python auto-imports a module
named `sitecustomize` from `sys.path` at interpreter startup; `run.sh` puts
`scripts/bfcl/` on `PYTHONPATH`, so `sitecustomize.py` runs the registration
before `bfcl_eval.__main__` is imported. No site-packages file is touched, so the
integration is fully recoverable from the repo.

## One-time setup

1. **bfcl venv** (already provisioned): `/Users/samriddhagc/.bfcl-venv`
   (Python 3.12, `bfcl-eval` v2026.3.23). Verify:
   ```
   /Users/samriddhagc/.bfcl-venv/bin/bfcl version
   ```
2. **himmy importable in the bfcl venv** (needed only by the HIMMY arm, not RAW).
   himmy's core deps (pydantic>=2, pyyaml>=6, httpx>=0.27) are already satisfied
   in the bfcl venv. The HIMMY-arm handler puts the repo root on `sys.path` so
   `import himmy` resolves without disturbing the bfcl venv. (RAW does not import
   himmy at all.)
3. **OpenRouter key**: must be present as `OPENROUTER_API_KEY=...` in the repo
   `.env` (`/Users/samriddhagc/LocalProjects/himmy-agent-test/.env`). `run.sh`
   copies just that line into `workdir/.env`; `sitecustomize.py` loads it into the
   environment. The value is never echoed or logged.

## Models registered (RAW arm)

| registry name | OpenRouter model | mode |
|---|---|---|
| `or-raw-gpt-4o-mini` | `openai/gpt-4o-mini` | native FC |
| `or-raw-gpt-4o-mini-prompting` | `openai/gpt-4o-mini` | prompting |
| `or-raw-qwen2.5-7b` | `qwen/qwen-2.5-7b-instruct` | native FC |
| `or-raw-qwen2.5-7b-prompting` | `qwen/qwen-2.5-7b-instruct` | prompting |
| `or-raw-llama3.2-3b` | `meta-llama/llama-3.2-3b-instruct` | native FC |
| `or-raw-llama3.2-3b-prompting` | `meta-llama/llama-3.2-3b-instruct` | prompting |
| `or-raw-llama3.2-1b-prompting` | `meta-llama/llama-3.2-1b-instruct` | prompting |

The `-prompting` variants are the apples-to-apples controls for the HIMMY arm.
`llama-3.2-3b`/`1b` are the small open models where himmy lift is expected (they
stringify args / truncate text in messy ways that BFCL's default prompting decoder
handles less robustly than himmy's tolerant parse + schema-aware coercion).

A transport robustness note: OpenRouter occasionally returns a `200` with
`usage = None`; the stock BFCL OpenAI handler raises on `usage.prompt_tokens` and
records a permanent `Error during inference: ...` for that task. `OpenRouterRawHandler`
overrides the response-parse to tolerate a missing usage block (token counts default
to 0), so a transport artifact never corrupts a graded reply. Both arms inherit this
identical behavior (the HIMMY handler subclasses the RAW handler) — apples-to-apples.

`underscore_to_dot` is set to `is_fc_model` (mirrors BFCL's official OpenAI
configs): native-FC tool names have `.` replaced by `_` over the wire (the OpenAI
tools schema forbids dots), so the checker must convert back. Getting this wrong
yields spurious `Function name 'math.factorial' not found` failures.

Prove registration:
```
scripts/bfcl/run.sh models | grep or-raw
```

## Bounded subset (cost control)

`workdir/test_case_ids_to_generate.json` maps `{category: [task_id, ...]}`.
`--run-ids` makes `generate` load only those ids (empty lists are skipped). The
**same ids** must be used for both arms so the lift comparison is paired by id.

## Smoke test (VERIFIED, real OpenRouter calls)

```
# generate 5 simple_python tasks for gpt-4o-mini over OpenRouter
scripts/bfcl/run.sh generate --model or-raw-gpt-4o-mini \
    --test-category simple_python --run-ids --temperature 0 --num-threads 1

# grade with the OFFICIAL BFCL evaluator (subset => --partial-eval is REQUIRED)
scripts/bfcl/run.sh evaluate --model or-raw-gpt-4o-mini \
    --test-category simple_python --partial-eval
```

Result observed (gpt-4o-mini, simple_python, 5 tasks): **accuracy 1.0 (5/5)**.
The `result.json` carries real per-task `latency` (~0.67–1.59 s) and live
`input_token_count` / `output_token_count`, confirming genuine OpenRouter calls.
Score file: `workdir/score/or-raw-gpt-4o-mini/non_live/BFCL_v4_simple_python_score.json`
→ `{"accuracy": 1.0, "correct_count": 5, "total_count": 5}`.

The small open model (`or-raw-qwen2.5-7b`) was smoke-tested the same way and
also hits OpenRouter (real ~5–6 s latencies); it scores low on RAW native FC
(emits malformed/partial argument JSON, e.g. `{"base": 5}` missing `height`,
`{"number": ` truncated) — the expected messy-tool-call regime where himmy's
tolerant parser + repair are designed to add lift.

## HIMMY arm (treatment)

The HIMMY arm runs the **same OpenRouter model in the same prompting mode** as the
RAW prompting baseline, so the comparison is apples-to-apples. The **only two
variables** are himmy in the loop:

1. **Manifest** — RAW injects BFCL's stock `system_prompt_pre_processing_chat_model`
   manifest; HIMMY injects `himmy.format_for(model, override).render_system_manifest`
   (the model-family-native tool-call grammar).
2. **Parse** — RAW decodes with BFCL's `default_decode_ast_prompting` /
   `default_decode_execute_prompting` (which throw on a `<tool_call>{json}</tool_call>`
   envelope); HIMMY decodes with the format's tolerant `parse` (native grammar pass
   OR'd with himmy's fail-open text/python recovery + name repair).

For `multi_turn_base`, HIMMY additionally threads tool results back with the
format's `render_tool_results` (Hermes `<tool_response>` / Llama ipython / Mistral
`[TOOL_RESULTS]`) instead of BFCL's `format_execution_results_prompting`.

| registry name | OpenRouter model | himmy format | mode |
|---|---|---|---|
| `or-himmy-gpt-4o-mini` | `openai/gpt-4o-mini` | `generic` | prompting |
| `or-himmy-qwen2.5-7b` | `qwen/qwen-2.5-7b-instruct` | `hermes_chatml_xml` | prompting |
| `or-himmy-llama3.2-3b` | `meta-llama/llama-3.2-3b-instruct` | `llama3_json` | prompting |
| `or-himmy-llama3.2-1b` | `meta-llama/llama-3.2-1b-instruct` | `llama3_json` | prompting |

### What the HIMMY arm actually does (three genuine himmy capabilities)

The HIMMY arm exercises three shipped himmy mechanisms, all from real himmy modules:

1. **Native-format manifest** (`himmy.services.inference.tool_formats.format_for(...).render_system_manifest`) — advertises the model-family-native tool-call grammar (Llama3-JSON / Hermes-Qwen XML / Mistral-v3) instead of BFCL's stock prompt.
2. **Native-first / tolerant-text decode** — mirrors himmy production (`himmy/services/inference/local.py`): prefer the provider's structured `tool_calls`; else fall back to the format's tolerant `.parse` (which OR-s in himmy's fail-open text/python recovery). Every decode logs `[himmy-arm] decode ...`.
3. **Schema-aware lenient arg coercion** (`himmy.services.tools.service._coerce_lenient_args`) — losslessly coerces a stringified scalar (`"20"`→`20`, `"true"`→`True`) **only** when the tool's JSON schema declares that scalar type and the conversion round-trips exactly. Logs `[himmy-arm] coerce ...`.

### Name-guard: the decode is guarded by the REAL bound tool names (not an empty set)

himmy's parser takes a `known` set of bound tool names and uses it to reject an
ambiguous-source name that does not match any bound tool (`_name_is_plausible`,
`himmy/services/inference/tool_protocol.py`) — this is exactly how himmy runs in
production (the parser is *guarded by the set of known tool names*). BFCL's evaluator
calls `decode_ast` / `decode_execute` with **only the raw model text** (no function
list), so to apply the production name guard the HIMMY handler writes a sidecar at
GENERATION time mapping `sha256(model_output_text) -> [bound tool names]`
(`workdir/himmy_known/<model>.jsonl`, one line per generated task) and loads it back at
DECODE time, keyed by the exact stored `result` string (BFCL passes that string verbatim
to `decode_*`). On an exact-hash miss it falls back to the per-model union of all tool
names seen (conservative — it never invents a name), and only to the empty set as a last
resort, which is logged. Every decode logs `[himmy-arm] decode ... known=N ...` so the
guard size is visible. (An earlier draft passed `set()` — i.e. NO guard — which let the
parser accept hallucinated names and collapsed the irrelevance category to 0; that was a
harness bug, now fixed.)

Honest note on `<tool_call>` envelopes: when a model emits an EXPLICIT tool-call envelope
(e.g. Hermes `<tool_call>{...}</tool_call>`), himmy's parser TRUSTS it and decodes it even
if the name is not a bound tool — the model unambiguously requested a call, and in himmy
production the *tools kernel* (not the parser) rejects an unregistered tool at execution
time. BFCL's irrelevance metric is purely parse-based, so a model that hallucinates a tool
inside an explicit envelope on an irrelevant query will be scored as a (wrong) call. This
is a real, interpretable behavioral difference driven by the MANIFEST (see run results),
not a parser defect — and we deliberately do not "fix" it by adding benchmark-specific
anti-hallucination text to himmy's manifest.

### Important honest caveat — provider-side native parsing under a native manifest

When himmy advertises a model's NATIVE tool-call grammar, the OpenRouter **upstream
provider's own chat template** frequently recognizes the emitted call and parses it
into the OpenAI `message.tool_calls` field, leaving `message.content = null`. Two
consequences, both handled and documented so the delta stays interpretable:

* himmy re-renders those structured calls back into the format's native text envelope
  (so the single himmy `.parse` decode site still does the round-trip) — and applies
  the schema-aware coercion to repair the provider's stringification. This is exactly
  himmy's production native-first path.
* The provider's native parser **stringifies scalars** (`duration: "20"`, `area:
  "0.01"`). himmy's coercion repairs the cases the schema can prove lossless; it
  deliberately does **not** widen ambiguous values (e.g. `"6"` against a `number`/float
  schema, since `repr(6.0)="6.0" != "6"`), so a few stringified-float args remain and
  can cost the HIMMY arm a task that the RAW pure-text path got for free. Conversely the
  native path captures COMPLETE calls where the RAW text path TRUNCATED mid-generation.

So on a small model the lift is genuinely **bidirectional** per task, and the reported
per-category delta is the honest net. The comparison framing is therefore precisely:
"model X RAW (BFCL default prompting decode)" vs "model X under himmy (himmy native
manifest + native-first/tolerant decode + schema-aware coercion)". The arms differ in
exactly that himmy stack and nothing else (same model, same task ids, temperature 0).

### multi_turn note

For `multi_turn_base`, the HIMMY arm threads tool results back with the format's
`render_tool_results` and records each assistant turn as a **string** in himmy's native
grammar (never the provider's `content=null` object, which makes the provider reject the
next turn with HTTP 400). RAW uses BFCL's default `format_execution_results_prompting`.

`gpt-4o-mini` is pinned to the `generic` format (no open-weight grammar) and is
expected to show **~0 lift** — the honest frontier-ceiling result. `qwen2.5-7b` is
pinned to `hermes_chatml_xml` (the Qwen2.5/Hermes ChatML `<tools>` grammar), where
himmy's native render + tolerant parse should add measurable lift (the small open
model emits messy/prose tool-calls that BFCL's default prompting decoder misses).

### Proof himmy is genuinely in the loop

* Every decode logs `[himmy-arm] decode format=<name> parsed N call(s): [...]` via the
  `himmy.bfcl` logger; the RAW arm emits no such markers.
* `HimmyOpenRouterHandler.__init__` asserts the resolved format's renderer lives in a
  `himmy.*` module, so the arm cannot silently fall back to a himmy-free path.
* Decoded calls carry himmy-minted UUID `tool_call_id`s (from himmy's parser), never
  an OpenRouter provider id — distinct provenance from RAW.
* `grep -nE '^\s*(import|from)\s+himmy' scripts/bfcl/handlers.py` returns nothing
  (RAW is himmy-free); the same grep on `handlers_himmy.py` shows the himmy imports.

### HIMMY-arm smoke test (VERIFIED, real OpenRouter calls)

Same tiny slice as the RAW arm (gpt-4o-mini, `simple_python`, 5 tasks):

```
scripts/bfcl/run.sh generate --model or-himmy-gpt-4o-mini \
    --test-category simple_python --run-ids --temperature 0 --num-threads 1
scripts/bfcl/run.sh evaluate --model or-himmy-gpt-4o-mini \
    --test-category simple_python --partial-eval
```

Result observed: **accuracy 1.0 (5/5)** — ties the RAW prompting baseline (also 5/5),
the expected frontier-ceiling ~0 lift. The generation logs show `[himmy-arm] init …
format=generic` and `[himmy-arm] manifest …` per task; the evaluation logs show
`[himmy-arm] decode format=generic parsed 1 call(s): [...]` per task (including
correct decode of dotted names like `math.factorial`), proving every tool-call
went through `himmy.format.parse`. The `result.json` carries real OpenRouter
latencies (0.50–3.98 s) and live token counts. Score file:
`workdir/score/or-himmy-gpt-4o-mini/non_live/BFCL_v4_simple_python_score.json`
→ `{"accuracy": 1.0, "correct_count": 5, "total_count": 5}`.

## Determinism / honesty notes

* `--temperature 0`. OpenRouter passes temperature through but does not guarantee
  bit-reproducibility across upstream providers/routing; treat exact replay as
  best-effort. Pair arms by `(task_id, trial_index)` (McNemar handles single-trial
  pairing).
* Grading is done by the **official BFCL evaluator** for credible numbers; lift
  stats (Wilson CI + exact McNemar) are computed from those per-task verdicts via
  `himmy/benchmark/stats.py`.
* All numbers in this harness are real runs. If a run errors, the error is
  reported — no numbers are fabricated.

## Run results (real OpenRouter runs, 2026-06-22)

Bounded subset: 6 categories x 30 tasks (multi_turn_base 20) = **170 paired tasks per
arm**, temperature 0, prompting mode for BOTH arms. Grading = official BFCL evaluator;
lift stats (Wilson CI + exact McNemar) = `himmy/benchmark/stats.py`. Approx OpenRouter
cost: gpt-4o-mini matrix ~$0.39, qwen2.5-7b matrix ~$0.16 (computed from real per-task
token counts in the result files). Run via `scripts/bfcl/run_matrix.sh <raw> <himmy>`;
per-model machine-readable summaries in `workdir/lift_<himmy_model>.json`.

### gpt-4o-mini (frontier) — RAW `or-raw-gpt-4o-mini-prompting` vs HIMMY `or-himmy-gpt-4o-mini` (generic format)

| category | n | RAW | HIMMY | lift pp | McNemar p | leader |
|---|---|---|---|---|---|---|
| simple_python | 30 | 93.3% | 90.0% | -3.3 | 1.000 | tie |
| multiple | 30 | 90.0% | 80.0% | -10.0 | 0.508 | ns |
| parallel | 30 | 90.0% | 53.3% | -36.7 | 0.0010 | RAW |
| parallel_multiple | 30 | 96.7% | 36.7% | -60.0 | 7.6e-06 | RAW |
| irrelevance | 30 | 93.3% | 90.0% | -3.3 | 1.000 | tie |
| multi_turn_base | 20 | 5.0% | 25.0% | **+20.0** | 0.219 | HIMMY (ns) |
| **POOLED** | 170 | 82.4% | 64.7% | -17.6 | 1.5e-05 | RAW |

### qwen2.5-7b (small open) — RAW `or-raw-qwen2.5-7b-prompting` vs HIMMY `or-himmy-qwen2.5-7b` (hermes_chatml_xml)

| category | n | RAW | HIMMY | lift pp | McNemar p | leader |
|---|---|---|---|---|---|---|
| simple_python | 30 | 96.7% | 90.0% | -6.7 | 0.500 | ns |
| multiple | 30 | 83.3% | 90.0% | **+6.7** | 0.625 | HIMMY (ns) |
| parallel | 30 | 83.3% | 76.7% | -6.7 | 0.625 | ns |
| parallel_multiple | 30 | 73.3% | 70.0% | -3.3 | 1.000 | ns |
| irrelevance | 30 | 76.7% | 53.3% | -23.3 | 0.119 | RAW (ns) |
| multi_turn_base | 20 | 10.0% | 5.0% | -5.0 | 1.000 | ns |
| **POOLED** | 170 | 74.1% | 67.6% | -6.5 | 0.071 | RAW (ns) |

### Interpretation (honest)

The headline is the **opposite of a marketing number**: on this BFCL subset, routing a
model's tool-calls through himmy's *generic/open-weight manifest + tolerant parse* did
**not** beat BFCL's own heavily benchmark-tuned prompting baseline, and net-hurt the
frontier model. The reason is interpretable and matters:

* **BFCL's stock prompt is benchmark-optimized.** It instructs the model to (a) emit ALL
  calls as a single Python list `[f(...), g(...)]` (great for `parallel`/`parallel_multiple`)
  and (b) *"If none of the functions can be used, point it out"* (great for `irrelevance`).
  himmy's manifests are clean PRODUCTION tool-prompts that do neither, so the same model
  under himmy emits sequential "let me do step 1 first" single calls on parallel tasks and
  hallucinates an eager `<tool_call>` on irrelevant queries. That is where the negative pp
  come from — a MANIFEST behavioral difference, confirmed by reading the raw outputs
  (e.g. gpt-4o-mini parallel_multiple b=18/c=0; qwen irrelevance RAW says "None of the
  provided functions can be used" while HIMMY hallucinates `calculate_triangle_area`).
* **Where himmy's distinctive machinery is exercised, it helps:** multi-turn result
  threading gave gpt-4o-mini **+20pp** on `multi_turn_base` (himmy won 5 discordant tasks,
  lost 1), and qwen `multiple` was **+6.7pp**. These are the himmy-native capabilities
  (native result-grammar threading, tolerant native-envelope decode); neither reached
  significance at this subset size.
* We deliberately did **not** edit himmy's manifests to add BFCL's parallel/irrelevance
  instructions — that would game the benchmark and misrepresent the shipped artifact.

So the credible, framing-correct claims are: "gpt-4o-mini under himmy (generic manifest)"
vs "gpt-4o-mini RAW (BFCL prompting)" — net **-17.6pp pooled, driven by parallel
categories, with a +20pp multi-turn gain**; and "qwen2.5-7b under himmy (hermes manifest)"
vs RAW — **-6.5pp pooled, not significant (p=0.071)**. himmy is a framework; every number
here is same-model lift, never a bare himmy leaderboard row. The actionable takeaway is
that himmy's *generic* tool manifests should adopt the parallel-batch + decline-if-irrelevant
guidance that BFCL's prompt has, independent of this harness.
