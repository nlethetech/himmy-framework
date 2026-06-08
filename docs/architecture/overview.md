# Architecture Overview

> Top-level map of himmy-agent: a local-first, self-contained, entity-backed Python agent framework.

## Overview

Himmy is an **offline-first** agent framework. The bare install (`pydantic`, `pyyaml`,
`httpx` — see `pyproject.toml`) needs no network and no API keys: agents run against a
deterministic stub out of the box. Real models (Ollama, the Claude CLI, or cloud
providers via `pydantic-ai`) and heavier capabilities (Postgres, pgvector, FastAPI,
observability, Nepal data connectors) are **opt-in extras** declared under
`[project.optional-dependencies]`, never required.

The package re-exports its most common primitives from `himmy/__init__.py`
(`Persona`, `Task`, `Agent`, plus `build_runtime`, orchestrators, skills, typed
agents). To keep `import himmy` cheap and fully offline, the heavier kernels
(runtime/inference/config/toolkit/orchestrators) are loaded **lazily** via PEP 562
`__getattr__` — see `_LAZY*` sets in `himmy/__init__.py`.

Two ideas run through the whole codebase:

- **Provenance-native.** Personas, tasks, threads, messages, tools, skills, context
  snapshots, and run events project to versioned, content-addressed `EntityRecord`s,
  and every run emits an ordered stream of `RunEvent`s (`himmy/core/events.py`). This
  is the audit + replay spine.
- **Additive seams.** New capabilities (record/replay, compaction, HITL, graphs,
  residency, managed secrets) are layered onto existing seams (the `ClientManager`
  protocol, the runtime's `_emit` fan-out, `task.context` knobs) so the default
  behavior is unchanged when a feature is not configured.

## Module map

| Package | Responsibility |
| --- | --- |
| `himmy/agents/` | Composition primitives — `Persona`, `Task`, `ChatThread`/`Message`, `Agent`. |
| `himmy/core/` | Cross-cutting kernel: `RunEvent`/`EventType` (`events.py`), `HimmyError` (`errors.py`), id helpers (`ids.py`), typed metadata, sqlite hardening. |
| `himmy/entities/` | Append-only entity registry: `EntityRecord`/`EntityLink`, projection, lineage, sqlite/postgres backends. |
| `himmy/runtime/` | Single-agent execution loop (`SingleAgentRuntime`), full-stack builder, spec-driven builder, checkpoints, compaction, termination, tool routing, sessions, diagnostics. See [runtime](./runtime.md). |
| `himmy/orchestrators/` | Multi-agent + workflow patterns over the runtime: group chat, fan-out, handoff/delegation, planner, reflection, state graph, linear workflow. See [orchestrators](./orchestrators.md). |
| `himmy/services/` | Domain services, each its own concern: `inference/`, `storage/`, `knowledge/`, `context/`, `prompts/`, `evaluation/`, `tools/`, `mcp/`, `memory/`, `guardrails/`, `governance/`, `audit/`, `sandbox/`, `observability/`. |
| `himmy/skills/` | First-class typed capabilities (know-how + tools + guardrails): model, loader, registry, resolve, dispatch, builtin catalog. See [skills](./skills.md). |
| `himmy/toolkit/` | Named bundles of built-in tools (`web`, `files`, `data`, `code`, `google`, `memory`, `notes`, `tasks`, `nepal`, …) resolved by name via `register_packs`. |
| `himmy/config/` | Spec-driven declarative config: `AgentSpec`, `TeamSpec`, `WorkflowSpec`, `EvalSpec`, `HttpToolSpec`, `MCPServerConfig`, secrets, residency, `himmy.toml`. See [config](./config.md). |
| `himmy/application/` | Application services composing domain services for a use case. |
| `himmy/api/` | FastAPI app + routers (Himmy Studio BFF, served locally; an opt-in `api`/`studio` extra). |
| `himmy/cli/` | The `himmy` console script (`himmy.cli.__main__:main`): `init`, `run`, `chat`, `team`, `eval`, `doctor`, `tools`, `skills`, record/replay, … |
| `himmy/connectors/` | Nepal-flavored data connectors (news RSS, NRB forex) exposed as tools / an MCP server. |
| `himmy/nepal/` | Nepal domain primitives: Bikram Sambat calendar, language helpers. |
| `himmy/benchmark/` | Standing benchmark harness (runs in CI against a real local model). |

(`himmy/typed_agent.py` at the package root provides the `TypedAgent` structured-output
surface re-exported from `himmy`.)

## Key abstractions

- **`Persona`** (`himmy/agents/personas/persona.py`) — an agent's identity: name,
  description, `instructions`, tags, metadata (with a `role` property). Versioned
  entity (`kind="persona"`).
- **`Task`** (`himmy/agents/base_agent/task.py`) — one unit of work: a prompt plus a
  `context` dict carrying run knobs (`tool_names`, `model_key`, `output_schema`,
  `compaction_spec`, `context_build_spec`, …) the runtime recognizes.
- **`ChatThread` / `Message`** (`himmy/agents/base_agent/thread.py`) — the ordered
  message list for one run (system / user / assistant / tool roles), versioned.
- **`SingleAgentRuntime`** (`himmy/runtime/single_agent.py`) — the per-task conductor;
  the one public entry point that turns a persona + task into an answered thread plus a
  full audit trail.
- **`build_runtime()`** (`himmy/runtime/builder.py`) — wires the whole offline-first
  stack in one call, defaulting to the deterministic `StubClientManager`.
- **`AgentSpec`** (`himmy/config/agent_spec.py`) — the declarative `agent.yaml` façade
  with `to_persona()` / `to_llm_config()` / `make_task()` projections.
- **`RunEvent` / `EventType`** (`himmy/core/events.py`) — the audited event stream that
  every runtime and orchestrator emits.

## How it works / data flow

A single agent run (`SingleAgentRuntime.run_task`) walks a fixed pipeline. Each stage
emits `RunEvent`s onto the audit spine.

```
agent.yaml ──(load_agent_spec / apply_skills)──▶ AgentSpec
                                                   │ to_persona() / make_task() / to_llm_config()
                                                   ▼
                          Persona + Task + LLMConfig
                                                   │
        build_runtime() / build_runtime_for_spec()│  (inference, storage, registry,
                                                   ▼   context, prompts, tools wired)
┌──────────────────────────  SingleAgentRuntime.run_task  ─────────────────────────┐
│ 1. resolve/build context snapshot   ── CONTEXT_SNAPSHOT_BUILT                     │
│ 2. render system + task prompts (+ project snapshot keys)                         │
│ 3. append SYSTEM (first turn) + USER  ── (input guardrail)                        │
│ 4. register persona/task entities     ── EntityRecord rows                        │
│ 5. emit AGENT_RUN_STARTED                                                         │
│ 6. build InferenceRequest (bind tools) ── INFERENCE_REQUESTED                     │
│ 7. call InferenceService.run ─────────── INFERENCE_SUCCEEDED / INFERENCE_FAILED   │
│ 8. replay tool call/return pairs ─────── TOOL_CALLED / TOOL_COMPLETED|FAILED      │
│ 9. append ASSISTANT (+ output guardrail)                                          │
│10. register message + thread version + lineage links                             │
│11. emit AGENT_RUN_FINISHED + save thread                                          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                          RunResult / ChatThread ──┘
```

Every emitted event fans out best-effort (and isolated) to: the durable storage store →
the entity registry → an observability span → caller-supplied `on_event` callbacks
(`SingleAgentRuntime._emit`). For multi-turn agentic behavior, `run_agent_loop` drives
act→observe→re-invoke turns on top of this pipeline; orchestrators compose multiple
`run_task` / `continue_turn` calls and emit their own event families.

## Configuration

- **Provider selection** is offline-first by default. `build_inference()` picks the real
  `PydanticAIClientManager` only when `pydantic_ai` is installed **and** a provider key
  is present **and** `HIMMY_EXAMPLES_MODEL` names a model; otherwise the deterministic
  `StubClientManager` (`himmy/runtime/builder.py`).
- **Project defaults** live in `himmy.toml` (`[defaults]`, `[toolkit]`), loaded by
  `himmy/config/project.py`. Precedence: CLI flag > env > `himmy.toml` > built-in.
- **Common env vars**: `HIMMY_EXAMPLES_MODEL`, `HIMMY_EXAMPLE_PROVIDER`/`_MODEL`
  (recipes), `HIMMY_SECRETS` (secret backend), `HIMMY_REGION`/`HIMMY_ALLOWED_REGIONS`
  (residency), `HIMMY_SKILLS_PATH`, `HIMMY_CAPTURE_IO`. See [config](./config.md).

## Extension points

- **New capability bundle** → author a `Skill` YAML (see [skills](./skills.md)).
- **New tool** → register on a `ToolRegistry` (a toolkit pack, a declarative
  `HttpToolSpec`, an MCP server, or a `tools_module` registrar).
- **New provider/model** → swap the `ClientManager` behind `InferenceService`
  (record/replay reuses this exact seam).
- **New orchestration** → compose `SingleAgentRuntime` calls; emit `RunEvent`s through
  the shared `_emit` helpers so the run stays auditable.
- **New backend** (storage, secrets, checkpoints) → implement the relevant Protocol
  (`CheckpointStore`, `SecretProvider`, …).

## Gotchas & invariants

- `import himmy` is deliberately light; heavy kernels resolve lazily. Don't assume a
  symbol is eagerly importable — check the `_LAZY*` maps in `himmy/__init__.py`.
- The deterministic stub is the default. Tests run against it; recipes (`RECIPES.md`)
  show the same framework against real models.
- Events and entities are best-effort sinks: a failing sink never breaks a run, and
  `CancelledError` always propagates so a cancelled run unwinds cleanly.
- Entity rows are append-only; new versions are new rows, never mutations.

## Related docs

- [runtime](./runtime.md) — the single-agent loop, checkpointing, compaction, replay.
- [orchestrators](./orchestrators.md) — group chat, fan-out, handoff, planner, graphs, workflows.
- [skills](./skills.md) — the capability layer and dispatch.
- [config](./config.md) — declarative specs, secrets, residency, project defaults.
- Design notes: `../design/record_replay_and_compaction.md`, `../design/skills_system.md`.
