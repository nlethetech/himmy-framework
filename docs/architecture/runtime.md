# Runtime

> The single-agent execution kernel: turns a persona + task into an answered, audited thread.

## Overview

`himmy/runtime/` is the per-task conductor of one agent run. Its core,
`SingleAgentRuntime` (`himmy/runtime/single_agent.py`), runs a fixed pipeline —
resolve context, render prompts, call inference, replay tool exchanges, append the
assistant turn, register entities, and emit the full `RunEvent` sequence. The runtime
is **stateless and per-task**; multi-turn behavior (`run_agent_loop`) and multi-agent
behavior ([orchestrators](./orchestrators.md)) compose multiple runtime calls.

Only `inference_service` is required; every other collaborator (memory, tools, context,
prompts, registry, checkpoints, guardrails) is optional and the runtime degrades cleanly
when one is absent.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/runtime/single_agent.py` | `SingleAgentRuntime` — the run pipeline, agent loop, streaming, HITL resume, the `RunResult`/`AgentLoopResult` views. |
| `himmy/runtime/builder.py` | `build_runtime()` / `build_inference()` / `build_storage()` — one-call offline-first wiring (defaults to the stub manager). |
| `himmy/runtime/from_spec.py` | `build_runtime_for_spec()` — wire a runtime from a declarative `AgentSpec` (tools, packs, MCP, knowledge, memory, guardrails, spawn, skill dispatch). Shared by the CLI and Studio. |
| `himmy/runtime/checkpoint.py` | `AgentCheckpoint` + `CheckpointStore` (in-memory / sqlite) for HITL pause/resume; `GraphCheckpoint` + store for graph resume. |
| `himmy/runtime/compaction.py` | `ContextCompactor` — the pure planner that decides whether/what to summarize. |
| `himmy/runtime/termination.py` | The `final_answer` tool + `is_no_progress` no-progress detection. |
| `himmy/runtime/tool_router.py` | `select_tools` — narrow a large toolset to the relevant few for small models. |
| `himmy/runtime/session.py` | `SqliteSessionStore` — durable chat threads keyed by session id (for `himmy chat --session`). |
| `himmy/runtime/diagnostics.py` | `collect_doctor_report` — environment diagnostics for `himmy doctor` / Studio. |

## Key abstractions

### `SingleAgentRuntime`

Constructed with `inference_service` and a set of optional collaborators
(`memory_store`, `tool_service`, `context_service`, `prompt_manager`,
`context_prompt_mapper`, `entity_registry`, `checkpoint_store`, `input_guardrail`,
`output_guardrail`), plus knobs: `default_model_key`, `save_threads`,
`default_deadline_seconds`, `strict_snapshot`, `on_event`, `capture_io`. When omitted,
a `PromptManager` and `ContextPromptMapper` are auto-created.

Key methods:

- `run_task(persona, task, thread=None, *, llm_config=None, snapshot_id=None,
  deadline_seconds=None) -> ChatThread` — the back-compat entry point; returns the
  appended thread (terminal status/cost/tokens/error/structured live on the assistant
  message metadata).
- `run_task_detailed(...) -> RunResult` — the same pipeline returning a typed
  `RunResult` (status, output text/structured, real `ToolCallRecord`/`ToolReturnRecord`
  lists, tokens, cost, latency, model path). Use this to detect FAILED runs without
  scraping thread rows.
- `run_agent_loop(persona, task, thread=None, *, max_turns=6, cost_budget=None,
  llm_config=None, hitl=False, stop_on_no_progress=False, synthesize_empty=True,
  route_tools=False, route_max_tools=4) -> AgentLoopResult` — the bounded
  act→observe→re-invoke loop (below).
- `continue_turn(persona, thread, *, task_context=None, llm_config=None) -> RunResult`
  — one more inference turn on an existing thread with no new user prompt (the public
  seam orchestrators use to switch tools/model per turn).
- `reinject_system_prompt(persona, thread, *, task_context=None) -> str` — re-render and
  replace the leading SYSTEM message for a new persona on a shared thread (the
  multi-agent handoff fix; a no-op when the prompt already matches).
- `stream_task(...) -> AsyncIterator[StreamDelta]` — single-turn streamed reply; the
  final `done` delta carries the materialized response and the assistant message is
  appended before it is yielded.
- `resume_agent_loop(checkpoint_id, *, approved, llm_config=None, hitl=True)` — resume a
  paused HITL run (below).

### `RunResult` and `AgentLoopResult`

`RunResult` (a dataclass) is the typed view of one turn. Notable fields:
`status`, `output_text`, `output_structured`, `tool_calls`, `tool_returns`, `error`,
`error_code`, token counts, `cost`, `latency_ms`, `model_path`, `provider_name`,
`trace_id`, `workflow`/`workflow_complete`, and `round_trip_complete` (True when the
provider ran the whole tool round-trip internally — e.g. pydantic-ai/OpenAI — so the
loop must not continue). `succeeded` is `status == SUCCESS`.

`AgentLoopResult` aggregates a loop: `turns` (each `RunResult`), `final`,
`stopped_reason` (`final` | `final_answer` | `max_turns` | `budget` | `no_progress` |
`awaiting_approval` | `synthesized` | `error`), `checkpoint_id`, and summed
cost/token properties.

## How it works / data flow

### The run pipeline (`_run_task_body`)

```
run_task_detailed
  └─(optional deadline via asyncio.timeout)→ _run_task_body
       1. _resolve_snapshot      → CONTEXT_SNAPSHOT_BUILT (or snapshot_error)
       2. _render_prompts        (system + task; project snapshot keys; system_prefix)
       3. append SYSTEM (first turn) + USER   (USER passed through input guardrail)
       4. _register_entity(persona), _register_entity(task)
       5. AGENT_RUN_STARTED
       6. _build_request         (bind tools, llm_config-over-context precedence)
       7. INFERENCE_REQUESTED → inference_service.run → INFERENCE_SUCCEEDED|FAILED
       8. _append_tool_messages  → TOOL_CALLED, then TOOL_COMPLETED|TOOL_FAILED per pair
       9. append ASSISTANT       (output passed through output guardrail)
      10. register message + bump thread version + _link_lineage
      11. AGENT_RUN_FINISHED + _maybe_save_thread
```

`InferenceService.run` never raises for provider/manager errors — a failed turn comes
back as a `RunResult` with a non-SUCCESS status and an `error`/`error_code`. Only
`CancelledError`/deadline expiry unwinds the run; in that case the runtime still emits a
terminal `AGENT_RUN_FINISHED(error='cancelled')` and saves the partial thread before
re-raising.

### Building the request (`_build_request`)

`llm_config` takes precedence over `task.context` for model knobs. The runtime maps the
thread's messages into `InferenceMessage`s (carrying `tool_call_id`/`name` for tool
rows), resolves the effective `model_key` (llm_config > context > default), and binds
tools via `tool_service.bound_tools(tool_names)` plus a `tool_executor()`. A
`ResponseFormat.WORKFLOW` run forces exactly one step tool to be bound and fails fast if
it cannot be.

### Structured output

Structured output is requested by setting `output_json_schema` (on `llm_config`) or
`output_schema` (in `task.context`), optionally with a `response_format`. When the
response has `output_structured` but no text, the runtime serializes it to JSON for the
assistant message content; the structured payload is also stamped on the assistant
message metadata and on `RunResult.output_structured`.

### The agent loop (`run_agent_loop` → `_drive_loop`)

The first turn is a normal `run_task_detailed`. While a turn calls tools, the runtime
feeds the updated thread back for another `continue_turn`. The loop stops when:

- the model answers with **no tool calls** (`final`), or returns `round_trip_complete`;
- the model calls `final_answer` (`final_answer`, see termination below);
- `stop_on_no_progress` and the last two turns made identical tool calls (`no_progress`);
- `max_turns` or `cost_budget` is reached;
- a turn FAILS (`error`).

With `route_tools=True` the loop first narrows the bound tools via the tool router
(below). With `synthesize_empty=True`, if a tool-using loop ends with an empty answer,
the runtime runs one extra turn with tools unbound and a synthesis nudge
(`_maybe_synthesize`, stop reason `synthesized`).

### Approval gates + checkpointing (HITL pause/resume)

`himmy/runtime/checkpoint.py` defines `AgentCheckpoint` — a fully serialized suspended
run (persona/task/thread/ctx/llm_config + loop limits + the exact `PendingToolCall`s)
with a lifecycle `awaiting_approval → approved | rejected` (resolved exactly once).
`CheckpointStore` is a Protocol; `InMemoryCheckpointStore` (default, volatile) and
`SqliteCheckpointStore` (durable, survives restarts) implement it.

With `hitl=True` (requires a `checkpoint_store`), when a turn calls a tool that requires
approval the tool is *denied* with `POLICY_BLOCKED`; `_drive_loop` detects those denied
calls (`_pending_approvals`), persists a checkpoint, emits `APPROVAL_REQUIRED`, and
returns with `stopped_reason='awaiting_approval'` and a `checkpoint_id`.

`resume_agent_loop(checkpoint_id, approved=…)` rehydrates the checkpoint, applies the
decision to each pending call — executing it via `tool_service.execute(...)` and
recording the real result (`APPROVAL_GRANTED`), or recording a rejection
(`APPROVAL_REJECTED`) — writes the outcome onto the thread as a TOOL message, then drives
one continuation turn and resumes the loop. Idempotency is enforced: a resolved
checkpoint cannot be resumed twice.

### Context compaction

A multi-turn agent sends its **entire** `thread.messages` to the model each turn, so
long runs overflow the window. Compaction is opt-in via `task.context['compaction_spec']`
(`{max_tokens, keep_recent}`); `make_task` plumbs the `AgentSpec.compact_*` fields into
it. `_maybe_compact` runs before each continuation turn.

`ContextCompactor.plan(messages)` (`himmy/runtime/compaction.py`) is a pure planner
returning a `CompactionPlan`. Invariants it holds:

- **never touch the leading system message(s)** (persona/instructions);
- **keep the most recent `keep_recent` messages verbatim**;
- **never split a `tool_call` from its `tool_return`** — the tail boundary is snapped
  back so the kept tail never starts on an orphaned `tool` message;
- no-op when under budget or when there's too little safe to summarize.

When a plan fires, the runtime summarizes the middle span with one model call
(`SUMMARY_INSTRUCTION`), and only applies it if the summary is actually smaller than the
span it replaces. It replaces the span with one `[Summary of earlier conversation]`
SYSTEM message, bumps the thread version, and emits `CONTEXT_COMPACTED`. Token estimation
is a deliberately cheap ~4 chars/token heuristic (`estimate_tokens`).

### Termination (the `final_answer` tool)

`himmy/runtime/termination.py` adds an explicit terminal signal so a model that keeps
calling tools can still end cleanly. `register_final_answer_tool(registry)` binds a
synthetic `final_answer(answer)` tool; `final_answer_text(result)` returns its argument
when a turn called it. `is_no_progress(turns)` returns True when the last two turns made
the identical (non-empty) tool calls (via `calls_signature`).

### Tool routing for small models

`select_tools(inference, query, candidates, *, max_tools=4, model_key)`
(`himmy/runtime/tool_router.py`) runs one cheap structured-output call that, given the
request and a `[(name, description), …]` catalog, returns the relevant subset of tool
names. It is safe by construction: a no-op when there are already `<= max_tools`, never
narrows below the model's own pick, and returns **all** tools on any failure. The agent
loop invokes it (`route_tools=True`) via `_route_tools`, which also folds in skill
"use this when …" routing hints when present.

### Record-and-replay (cassettes)

Deterministic re-run is built on the `ClientManager` seam, not the runtime: a recording
manager appends `(cache_key, response)` to an ordered cassette, and a replay manager
returns the next recorded response for a request's content-hash with the provider off.
Tools are **not** re-executed on replay (the recorded response already carries
`tool_calls`/`tool_returns`), so replay is side-effect-free. Full details (managers,
cache key, CLI `--record`/`--replay`) are in
[`../design/record_replay_and_compaction.md`](../design/record_replay_and_compaction.md).

### Raw-I/O capture (debug inspector)

Opt-in via `capture_io=True` or `HIMMY_CAPTURE_IO`. When on, `INFERENCE_*` events carry a
bounded `io` snapshot (messages, bound tool names, response text, parsed tool calls),
size-capped per field/message (`build_io_capture`). Off by default — zero cost/exposure.

## Configuration

- Runtime knobs: `default_model_key`, `save_threads`, `default_deadline_seconds`,
  `strict_snapshot`, `capture_io`, `on_event`.
- Per-call: `llm_config`, `snapshot_id`, `deadline_seconds`; loop knobs on
  `run_agent_loop` (above).
- `task.context` knobs the runtime recognizes: `tool_names`, `model_key`,
  `output_schema`/`response_format`, `compaction_spec`, `context_build_spec`,
  `context_prompt_map_spec`, `skill_routing_hints`, `objectives`, `role`,
  `system_prefix`, `datetime`, `skills`.
- `build_inference()` chooses a real provider only when `pydantic_ai` is installed AND a
  provider key is present AND `HIMMY_EXAMPLES_MODEL` is set; otherwise the stub.

## Extension points

- Swap inference behavior by passing a different `InferenceService` (or a different
  `ClientManager` behind it) — record/replay uses exactly this.
- Add a durable `CheckpointStore` (sqlite) for cross-process HITL resume.
- Add `on_event` callbacks (or `add_event_listener`) to stream progress to a UI.
- Build from YAML via `build_runtime_for_spec` to get tools/packs/MCP/knowledge/memory/
  guardrails/spawn/skill-dispatch wired automatically.

## Gotchas & invariants

- Provider/manager failures are returned as a non-SUCCESS `RunResult`, never raised;
  only cancellation/deadline unwinds the run (and still emits a terminal event).
- `from_spec` always adds a `grounding` **output** guardrail to every spec-built agent
  (an anti-hallucination default that cannot be silently disabled).
- The thread version bump on the 2nd+ turn happens regardless of whether a registry is
  wired, so persisted thread versions stay correct without lineage.
- Compaction must never split a tool_call/tool_return pair and never compacts the system
  head — those invariants keep the message list provider-valid.
- `_emit` sinks are isolated and best-effort; a broken listener can't break a run, but
  `CancelledError` is always honored.

## Related docs

- [overview](./overview.md) — where the runtime sits in the layered architecture.
- [orchestrators](./orchestrators.md) — patterns composed over the runtime.
- [skills](./skills.md) — skill dispatch runs sub-agents through this runtime.
- [config](./config.md) — the `AgentSpec` that `from_spec` wires.
- [`../design/record_replay_and_compaction.md`](../design/record_replay_and_compaction.md) — record/replay + compaction design.
