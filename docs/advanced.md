# Himmy — advanced guide

The [README](../README.md) covers the three things you need to get an agent running:
define it, run it, extend it. This page is everything beyond that — reach for a
section when you need it, not before.

- [Production capabilities](#production-capabilities)
- [Multi-agent teams](#multi-agent-teams)
- [MCP servers](#mcp-servers)
- [Recursive sub-agents](#recursive-sub-agents)
- [Live on Telegram](#live-on-telegram)
- [Embeddings (knowledge + memory)](#embeddings)
- [Choosing a provider](#choosing-a-provider)
- [Record & replay](#record--replay)
- [Optional extras](#optional-extras)

## Production capabilities

Himmy has been through a full production-hardening pass. On top of the
offline-first core, it offers:

- **Failure handling** — `InferenceService.run` never raises for provider/manager
  errors; every exception is normalized to a typed `InferenceError`, retries fire
  only on retryable codes, batches are failure-isolated, and runs record the real
  status/error (a failed inference is no longer logged as a success).
- **Streaming** — `InferenceService.run_stream` yields typed `StreamDelta` chunks
  ending in a `done` frame that carries the materialized response. The stub streams
  deterministic offline deltas; the pydantic-ai path streams real provider tokens
  via `agent.run_stream` (see `examples/07_streaming.py`).
- **Caching** — a pluggable, TTL-bounded inference response cache
  (`InMemoryTTLCache`) honored when a request opts in via `use_cache`; a no-op by
  default so behavior is unchanged unless wired.
- **Real providers** — `PydanticAIClientManager` honors the full request envelope
  (system prompt, message history, tools, generation params, timeout) and reads
  token usage/cost; `GatewayClientManager` does real Pydantic-AI-Gateway routing
  when a key + the `providers` extra are present.
- **Postgres + pgvector** — durable `PostgresStorageService` and
  `PostgresEntityRegistry` (JSONB codecs, optimistic-concurrency, idempotent
  `create_run`, pool teardown/timeouts) plus a pgvector knowledge backend with
  enforced embedding-dimension and freshness checks.
- **Security** — HTTP tools are SSRF-hardened (path/host pinning, no traversal or
  query/host injection), auth secrets are redacted from logs/events, and incoming
  tool args are validated against `args_json_schema`.
- **Observability** — optional Logfire instrumentation emits real spans for runs,
  inference, and tools via `configure_observability()` (no-op unless enabled).
- **Evaluation** — metric suites over agent outputs with honest per-metric `passed`
  verdicts (no false-success rollups).
- **Pagination & tenancy** — list endpoints support pagination/ordering/caps and
  enforce `workspace_id` isolation on read paths.

## Multi-agent teams

Agents can collaborate two ways: **handoff** (control transfers to a peer — swarm-style)
and **delegation** (a manager calls a worker as a tool and gets its result back —
supervisor / manager-worker). Define a team in a `team.yaml` and run it:

```bash
himmy init --team myteam            # scaffold myteam/team.yaml
himmy team -f myteam/team.yaml -p "Research and summarize permaculture."
```

```yaml
entry: triage
members:
  - name: triage
    description: Decide who should handle this, then hand off.
    handoffs: [researcher, writer]
  - name: researcher
    description: Gather facts from the web.
    tool_packs: [web]
    tools: [web_search, web_fetch]
    handoffs: [writer]
    # delegates: [factchecker]   # call a worker as a tool instead of transferring
  - name: writer
    description: Write the final answer.
```

`himmy team` prints the routing trail (`triage → researcher → writer`) and the final
answer (`--json` for the full transcript). In Python:

```python
from himmy import build_runtime, MultiAgentOrchestrator
from himmy.config import load_team_spec, build_team

team, registry = build_team(load_team_spec("myteam/team.yaml"))
runtime, _inf, _tools = build_runtime(tool_registry=registry)
result = await MultiAgentOrchestrator(runtime, team, registry).run("...")
print(result.handoff_chain, result.output_text)
```

## MCP servers

**The whole ecosystem as tools.** Beyond the built-in packs, point an agent at any
stdio **Model Context Protocol** server and its tools become native himmy tools
(arg-validated, approval-gated, traced). List them in `agent.yaml` (or `team.yaml`):

```yaml
mcp_servers:
  - command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"]
    prefix: fs_                 # namespace tool names (fs_read_file, …)
  - command: uvx
    args: ["mcp-server-git", "--repository", "."]
    requires_approval: true     # gate this server's tools behind approval
    tools: [git_log, git_diff]  # bind a subset (empty = all)
```

The CLI launches each server, registers its tools, runs, then tears them down — in
`himmy run`, `chat`, `team`, and `eval`. Built on a transport-direct stdio client (no SDK).

## Recursive sub-agents

Set `allow_spawn: true` and the agent gains a `spawn_agent` tool — it can hand a
sub-task to a fresh single-agent (its own instructions, optionally its own
`tool_packs`) and use the answer. No `team.yaml` needed; the parent decides at run
time. The spawned worker can't itself spawn, so recursion is capped at one level.

## Live on Telegram

Turn any `agent.yaml` into a Telegram bot — `himmy telegram -f agent.yaml`
long-polls for messages, runs the agent (one thread per chat), and replies. Set
`HIMMY_TELEGRAM_BOT_TOKEN` (or `--token`). The `telegram` tool pack also gives an
agent `send_telegram` so it can message a chat proactively.

## Embeddings

Knowledge and memory recall default to the offline `DeterministicEmbedder`
(exact-overlap). For real semantic recall, select a model via env —
`HIMMY_EMBEDDER=ollama` (local Ollama `nomic-embed-text`, keyless, no new deps),
`HIMMY_EMBEDDER=fastembed` (ONNX, `pip install 'himmy[embeddings]'`), or
`HIMMY_EMBEDDER=openai`. The dimension threads through automatically
(`HIMMY_EMBEDDER_DIM` to override).

`web_search` defaults to a keyless DuckDuckGo backend (no API key, httpx only);
set `HIMMY_SEARCH_BACKEND=tavily` + `HIMMY_SEARCH_API_KEY=…` for higher-quality
results. Toolkit settings come from `HIMMY_*` env vars (see
`himmy.toolkit.ToolkitConfig`). Use the packs from Python too:

```python
from himmy.toolkit import register_packs, ToolkitConfig
from himmy.services.tools.registry import ToolRegistry

registry = ToolRegistry()
register_packs(registry, ["web", "utils"], ToolkitConfig.from_env())
```

## Choosing a provider

Pick a provider per run with `--provider`/`--model` (e.g. `--provider claude-cli
--model haiku` to use a local Claude Max session, `--provider ollama --model
llama3.2`, or `--provider openrouter` to route through OpenRouter — set
`OPENROUTER_API_KEY`, default model `mistralai/mistral-small-3.2-24b-instruct`).
With no flag, Himmy uses a real pydantic-ai provider when a key + the
`providers` extra are present, otherwise the offline stub (`OPENROUTER_API_KEY`
alone does not auto-route to OpenRouter — pass `--provider openrouter`). Set the
default in `agent.yaml`:

```yaml
model: default
# provider: claude-cli   # stub | claude-cli | ollama | pydantic-ai | openrouter (default: auto)
```

`himmy doctor` reports which providers, keys, and extras are available, and ends
with the single next action for your machine.

## Record & replay

`himmy run --record FILE` captures every model response to a portable cassette;
`himmy run --replay FILE` re-runs the agent exactly from it — no provider, no
network, no tool side effects — so debugging a failure becomes "step through the
exact trace" instead of "rerun and hope." Matching is by the request's content
hash, so a re-run replays identically. Long multi-turn runs can also **compact**
their oldest turns in place once they cross a token budget (`compact_context: true`
in `agent.yaml`), keeping the system head and recent tail verbatim.

## Optional extras

| Extra | Adds | Unlocks |
|---|---|---|
| `api` | fastapi, uvicorn | The HTTP API (`himmy.api.create_app`). |
| `providers` | pydantic-ai | Real model providers via `PydanticAIClientManager`. |
| `postgres` | asyncpg | `PostgresEntityRegistry`, `PostgresStorageService`. |
| `knowledge` | pgvector, openai, pypdf | pgvector knowledge backend, PDF reading + OpenAI-compatible embeddings. |
| `observability` | logfire | Real run/inference/tool spans via `configure_observability()`. |
| `validation` | jsonschema | Full-fidelity tool arg/output validation (offline subset used otherwise). |
| `dev` | pytest, pytest-asyncio, jsonschema | Test tooling. |
| `all` | all of the above | Everything. |

## Architecture overview

Himmy is organized as a set of independent **kernels** that compose through small
typed contracts. Every runtime dependency except inference is optional — drop the
entity registry and you lose lineage but keep inference; drop the tool service and
tools simply aren't bound.

| Kernel | Path | Responsibility |
|---|---|---|
| Core | `himmy/core/` | Ids, errors, run events, the `EventSink` protocol. |
| Entities | `himmy/entities/` | The lineage backbone: records, links, the in-memory registry, a Postgres scaffold. |
| Agents | `himmy/agents/` | The four primitives: `Persona`/`RolePersona`, `Task`, `ChatThread`/`Message`, `Agent`. |
| Inference | `himmy/services/inference/` | Stub + gateway + pydantic-ai client managers and the retrying `InferenceService`. |
| Prompts | `himmy/services/prompts/` | YAML-templated system/task prompts and the context→prompt mapper. |
| Context | `himmy/services/context/` | Snapshot building from storage + adapters with evidence. |
| Storage | `himmy/services/storage/` | In-memory persistence + a Postgres scaffold. |
| Knowledge | `himmy/services/knowledge/` | Chunking, embedding, retrieval, and a context adapter. |
| Tools | `himmy/services/tools/` | Local/HTTP tool registry, execution, and inference binding. |
| Evaluation | `himmy/services/evaluation/` | Metric suites over agent outputs. |
| Observability | `himmy/services/observability/` | Optional Logfire instrumentation. |
| Runtime | `himmy/runtime/` | `SingleAgentRuntime.run_task` — the orchestrating loop. |
| Application | `himmy/application/` | App-level services (runs, recommendations, dashboard). |
| Orchestrators | `himmy/orchestrators/` | Multi-step `Workflow` execution. |
| API | `himmy/api/` | FastAPI app factory and routers. |
