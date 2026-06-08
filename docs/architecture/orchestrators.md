# Orchestrators

> Multi-agent and workflow patterns composed over the single-agent runtime — all audited via `RunEvent`s.

## Overview

`himmy/orchestrators/` builds higher-order collaboration patterns on top of
`SingleAgentRuntime`. None of them is a second runtime: each composes `run_task_detailed`
/ `continue_turn` / `run_agent_loop` calls and routes its own event family through the
same best-effort, isolated `_emit` fan-out the runtime uses (durable storage store →
entity registry → observability span → caller `on_event` callbacks). So every speaker
selection, handoff, delegated subtask, fan-out join, graph node, and workflow step lands
on the audit spine and is replayable.

## Module map

| File | Pattern |
| --- | --- |
| `himmy/orchestrators/group_chat.py` | `GroupChatOrchestrator` (selector-driven shared-thread chat) + standalone parallel `fan_out` + typed `SubtaskSpec`/`SubtaskResult`. |
| `himmy/orchestrators/multi_agent.py` | `MultiAgentOrchestrator` — handoff (peer transfer) + delegation (call a worker as a tool); `AgentTeam`/`TeamMember`. |
| `himmy/orchestrators/planner.py` | `PlannerOrchestrator` — plan-and-execute (decompose → work steps → synthesize). |
| `himmy/orchestrators/reflection.py` | `reflect()` — a one-turn critique-and-revise pass. |
| `himmy/orchestrators/state_graph.py` | `StateGraph` / `CompiledStateGraph` — a LangGraph-style directed graph over typed shared state. |
| `himmy/orchestrators/workflow.py` | `WorkflowOrchestrator` — a linear, declarative, state-threaded multi-step workflow. |

## Key abstractions

### Teams (`multi_agent.py`)

- `TeamMember` — a `name`, a `Persona`, its `tools`, a `model_key`, and two edge sets:
  `handoffs` (peers it may transfer control to) and `delegates` (workers it may call as
  a tool).
- `AgentTeam` — `members` + an `entry` member, with `get()`/`require()`.

### Speaker selection (`group_chat.py`)

- `SpeakerSelector` (ABC) with `select(SelectionContext) -> str`. Implementations:
  `RoundRobinSelector` (deterministic cycle), `CallableSelector` (wrap any sync/async
  function), `LLMSelector` (a "manager" LLM that reads the transcript and picks via an
  enum-constrained structured-output schema, falling back to round-robin on any failure
  or out-of-set pick).
- `SelectionContext` — `candidates`, `last_speaker`, `round_index`, `transcript`.

### Typed subtask contracts (`group_chat.py`)

- `SubtaskSpec` — `worker`, `objective`, `inputs` (the request half of fan-out).
- `SubtaskResult` — `worker`, `answer`, `data`, `ok`, `error` (the result half; a failed
  worker still yields a well-formed result so the join is total).

## How it works / data flow

### GroupChat — selector-driven shared thread

`GroupChatOrchestrator.run(prompt)` runs many members over **one shared `ChatThread`**.
Each round a `SpeakerSelector` picks the next speaker; the first turn seeds the thread
with the prompt, and later turns **re-inject the speaker's persona system prompt**
(`runtime.reinject_system_prompt`) before `continue_turn` so a speaker never inherits the
previous speaker's persona. `_ctx` always sets `tool_names` (so a speaker only sees its
own tools, never others' synthetic tools). The chat ends on `final_answer`, a
`terminate_when` predicate, or `max_rounds`.

```
round 0: select → reinject persona → speak → final_answer? terminate? max_rounds?
round 1: select → reinject persona → speak → …
```

Events: `GROUP_CHAT_STARTED`, `GROUP_SPEAKER_SELECTED`, `GROUP_SPEAKER_SPOKE`,
`GROUP_CHAT_FINISHED`.

### Parallel fan-out — deterministic typed join

`fan_out(runtime, team, specs, …)` (also exposed as `GroupChatOrchestrator.fan_out`)
runs N workers **concurrently**, each in its own isolated sub-thread, requesting a
`SubtaskResult` via the runtime's structured-output seam (`run_agent_loop` with a
`STRUCTURED_OUTPUT` schema). `asyncio.gather` preserves **submission order** in its
result list, so results zip back to specs deterministically regardless of completion
order. `concurrency` optionally caps simultaneous workers via a semaphore. A worker that
raises is captured as `SubtaskResult(ok=False, error=…)` so one failure never loses the
others' work. Spend is summed.

Events: `FANOUT_STARTED`, `FANOUT_WORKER_COMPLETED`, `FANOUT_JOINED`.

### MultiAgent — handoff + delegation

`MultiAgentOrchestrator` registers synthetic tools at construction
(`_register_synthetic_tools`): `transfer_to_<peer>` for each handoff target,
`ask_<worker>` for each delegate target, plus `final_answer`.

- **Handoff** (`transfer_to_<peer>`) — the tool only signals intent; the orchestrator
  detects it between turns (`_detect_handoff` over `RunResult.tool_calls`), switches the
  active member, re-injects the target persona onto the **same** thread, and emits
  `AGENT_HANDOFF`. Swarm-style routing.
- **Delegation** (`ask_<worker>`) — resolves inside the normal tool pipeline: the handler
  runs the named worker to completion in its own sub-thread via `run_agent_loop`, returns
  its answer, and emits `AGENT_DELEGATED`. Control stays with the manager
  (supervisor / manager-worker).

The run loop (`run`) starts at the entry member and alternates handoff-detection with
`continue_turn` until `final_answer`, a plain text answer (no tool calls), `no_progress`,
or `max_turns`. `_ctx` always sets `tool_names` (member tools + its handoff/delegate
tools + `final_answer` when it has any), and `_cfg` uses `AUTO_TOOLS` when the member has
any callable edges/tools. (The `default-team.yaml` coordinator drives this with a text
`TOOL_CALL`-style delegation protocol; see `RECIPES.md` for the model-size caveat.)

### Planner — plan-and-execute

`PlannerOrchestrator.run(goal, …)` asks the model for an ordered plan (structured output,
with a numbered/bulleted text fallback parsed by `_steps_from_text` for providers like
Ollama that don't honor `output_json_schema`), executes each step over **one shared
thread** (so later steps see earlier results, optionally with bound `tool_names`), then
runs a final synthesis turn. Returns a `PlanResult` (`plan`, `step_results`,
`output_text`, `total_cost`). Capped at `max_steps`.

### Reflection — critique-and-revise

`reflect(runtime, draft, *, criteria=…)` is a single model turn that critiques a draft
against criteria (default: accuracy, clarity, completeness) and returns the improved
version (or the original on empty). The cheapest quality lift; usable standalone or as a
plan's final step.

### StateGraph — directed graph over typed shared state

`StateGraph` is the LangGraph capability expressed offline-first. Build declaratively,
then `compile()` (which validates the topology — entry point, dangling edge targets — so
a malformed graph fails fast) into a `CompiledStateGraph`, then `await invoke(state)`.

- **Nodes** — `async|sync (state) -> delta` functions registered with `add_node(name, fn,
  max_visits=…)`. A node may wrap an agent, a tool, an LLM call, or pure Python.
- **Edges** — `add_edge(a, b)` (static) and `add_conditional_edges(a, router)` where
  `router(state)` returns a node name, a sequence of names (fan-out), or `END`.
- **`START` / `END`** — reserved sentinel node names; routing a branch to `END` finishes
  it.
- **Reducers** — `set_reducer(key, fn)` merges each node's delta into shared state
  per-key; `add_reducer` gives LangGraph `add` semantics (list concat / numeric add), so
  parallel branches append to the same key without clobbering. Unreduced keys are
  last-write-wins.
- **Execution** — a BSP **superstep** loop (`_run_loop`): every node in the current
  frontier runs concurrently against the same pre-superstep state snapshot (BSP
  isolation), deltas merge, then the next frontier is computed from the edges (routers
  see the merged state). Multiple successors fan out; multiple predecessors join on the
  shared state next superstep (a join target reached twice runs once per superstep).
- **Loop guards** — per-node `max_visits` and a global `recursion_limit` (max supersteps)
  raise a clean `GraphRecursionError`.
- **Durable resume** — after every superstep a `GraphCheckpoint` (state + frontier +
  visit counts) is persisted via `GraphCheckpointStore`, so an interrupted run (timeout,
  crash, pause) resumes from the last completed superstep; `invoke(resume=…)` continues
  it. A run timeout checkpoints as `interrupted` rather than losing progress.

Events: `GRAPH_STARTED`, `GRAPH_NODE_STARTED`, `GRAPH_NODE_COMPLETED`,
`GRAPH_EDGE_TAKEN`, `GRAPH_CHECKPOINTED`, `GRAPH_FINISHED`. `GraphRunResult` carries
`status` (`completed` | `interrupted` | `failed`), `final_state`, `supersteps`,
`node_sequence`, `checkpoint_id`.

### Workflow — linear, state-threaded steps

`WorkflowOrchestrator.run(workflow, persona, …)` drives a `Workflow` (an ordered list of
`WorkflowStep`s) over one shared thread and an accumulating `state` dict. Each step's
`subtask` is `.format(**state)`-substituted (a missing key becomes a cleanly failed step,
not a `KeyError`); when a step has an `output_key`, its structured/text output threads
into state for later steps.

Per-step knobs: `tool_names`, `response_format`, `output_json_schema`, `output_key`,
`sequential_tools` (force one tool per inference call across the step's tool list via
`ResponseFormat.WORKFLOW`, preserving call order), `temperature`, `max_tokens`,
`step_timeout_seconds`. Production hardening: an overall/per-step timeout with clean
cancellation, resume-from-failed-step (`run(resume=result)` re-drives from
`next_index`), per-step retry with exponential backoff (`max_step_retries`), and an
in-process idempotency short-circuit (`idempotency_key`). Step results are read straight
from the typed `RunResult` (real `ToolCallRecord`/`ToolReturnRecord` objects), never
re-parsed from thread rows.

Events: `WORKFLOW_STARTED`, `WORKFLOW_STEP_COMPLETED`, `WORKFLOW_FINISHED`.
`WorkflowResult.status` is `completed` | `failed` | `partial` | `cancelled`.

> Note: the `OnEvent` type exported from `orchestrators/__init__.py` is the
> `WorkflowOrchestrator`'s caller-facing event callback (`(RunEvent) -> Awaitable`); it
> is **not** an event-trigger/`OnEvent`-step mechanism. The workflow is linear, not
> event-triggered.

## Configuration

- These orchestrators are wired in code (a runtime + a team/graph/workflow). The
  declarative `team.yaml` / `workflow.yaml` loaders that build them are documented in
  [config](./config.md) (`TeamSpec.build_team`, `load_workflow_spec`).
- Common constructor knobs: `max_turns`/`max_rounds`/`max_steps`, `default_temperature`,
  `on_event`, and pattern-specific limits (`delegate_max_turns`, `recursion_limit`,
  retry/timeout policy).

## Extension points

- Custom speaker policy → subclass `SpeakerSelector` or wrap a function in
  `CallableSelector`.
- New graph behavior → add nodes/edges/reducers; nodes are plain functions, so any
  agent/tool/LLM/Python step composes.
- New collaboration shape → compose runtime calls and route events through the shared
  `_emit` helpers so the run stays auditable.

## Gotchas & invariants

- On a shared thread, a persona switch **must** re-inject the system prompt
  (`reinject_system_prompt`) or the next turn runs under the wrong persona.
- `_ctx` always sets `tool_names` (even to `[]`) so a member never accidentally binds
  every registered tool, including other members' synthetic handoff/delegate tools.
- Fan-out join order is **submission** order (via `asyncio.gather`), not completion
  order — zip specs to results by index.
- StateGraph nodes must return a mapping delta or `None`; anything else raises
  `GraphError`. Loop guards (`max_visits`, `recursion_limit`) raise
  `GraphRecursionError`.
- A worker/step failure is captured as a typed failed result, not raised, so partial
  progress and the other results survive.

## Related docs

- [runtime](./runtime.md) — the `run_task` / `continue_turn` / `run_agent_loop` /
  `reinject_system_prompt` seams these compose, and `GraphCheckpoint`.
- [config](./config.md) — `TeamSpec` / `WorkflowSpec` loaders + multi-provider teams.
- [overview](./overview.md) — where orchestration sits in the stack.
