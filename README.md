# Himmy Agent Framework

**Himmy** is an **offline-first, entity-backed Python agent framework**. It composes
LLM personas, tasks, threads, tools, context, and knowledge into auditable agent
runs — and it does all of this with zero network access by default, using a
deterministic stub inference path. Optional extras layer in real model providers,
Postgres lineage storage, pgvector knowledge bases, and Logfire observability.

The headline idea: **everything that matters is an entity.** Personas, prompts,
tools, agents, threads, messages, context snapshots, and run events are all
immutable, versioned `EntityRecord`s with typed links — so any recommendation can
be traced back to the exact persona version, prompt, evidence, and thread that
produced it.

## Production capabilities

Himmy has been through a full production-hardening pass. On top of the
offline-first core, it now offers:

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

## Install

```bash
# Core (offline) install — pydantic, pyyaml, httpx only.
pip install -e .

# With the local API server (FastAPI + uvicorn).
pip install -e ".[api]"

# Everything (providers, postgres, knowledge, observability, dev tooling).
pip install -e ".[all]"
```

Requires Python **3.12+**.

## Offline quickstart

Himmy runs end-to-end with **no API keys and no network** using the deterministic
`StubClientManager`. The smallest possible run:

```python
import asyncio

from himmy import Persona, Task
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference import InferenceService, StubClientManager


async def main() -> None:
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
    )
    persona = Persona(name="analyst", description="A concise research analyst.")
    task = Task(title="ping", prompt="Say hello in one sentence.")
    thread = await runtime.run_task(persona, task)
    print(thread.last_message.content)


asyncio.run(main())
```

## Quickstart (CLI)

Prefer not to write wiring code? Define an agent in a file and drive it from the
`himmy` command. Everything below runs offline against the stub — no keys needed.

```bash
# Scaffold an agent.yaml + an example tools.py in ./my-agent
himmy init my-agent

# One-shot: run a prompt and print the answer
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence."

# Interactive chat that keeps a single thread
himmy chat -f my-agent/agent.yaml

# Check which optional extras / local providers / keys are available
himmy doctor

# List the built-in tools an agent can use
himmy tools

# Serve the FastAPI BFF (needs the `api` extra)
himmy serve
```

### Built-in tools

Agents get a batteries-included **toolkit** — five named packs you switch on per
agent. `himmy tools` lists them:

| Pack | Tools |
|------|-------|
| `web` | `web_search`, `web_fetch`, `http_request` |
| `files` | `read_file`, `write_file`, `list_dir` (jailed to a sandbox root) |
| `data` | `sql_query` (read-only; SQLite or Postgres) |
| `code` | `run_python` (resource-limited sandbox, approval-gated) |
| `utils` | `calculator`, `current_time` |
| `knowledge` | `kb_ingest`, `kb_search` (the agent's own RAG memory) |
| `documents` | `read_document` (PDF / text / Markdown → text) |
| `comms` | `send_email`, `send_webhook` (outbound; approval-gated) |
| `data-sources` | `weather`, `geocode`, `wikipedia` (keyless public APIs) |

Enable them declaratively in `agent.yaml`:

```yaml
name: researcher
description: Researches a topic and summarizes it.
tool_packs: [web, utils]            # register these built-in packs
tools: [web_search, web_fetch]      # bind a subset to the model (omit = all)
```

`web_search` defaults to a keyless DuckDuckGo backend (no API key, httpx only);
set `HIMMY_SEARCH_BACKEND=tavily` + `HIMMY_SEARCH_API_KEY=…` for higher-quality
results. Web tools are SSRF-guarded (no private/loopback hosts), files are
root-jailed, `sql_query` is read-only, and `write_file`/`run_python` are
approval-gated. Toolkit settings come from `HIMMY_*` env vars (see
`himmy.toolkit.ToolkitConfig`). Use the packs from Python too:

```python
from himmy.toolkit import register_packs, ToolkitConfig
from himmy.services.tools.registry import ToolRegistry

registry = ToolRegistry()
register_packs(registry, ["web", "utils"], ToolkitConfig.from_env())
```

An `agent.yaml` is a thin declarative façade over `Persona` + `Task` + `LLMConfig`:

```yaml
name: market-analyst
description: A market research analyst specializing in tech.
role: Research Analyst
instructions:
  - Provide actionable insights backed by clear reasoning.
model: default
# provider: claude-cli   # stub | claude-cli | ollama | pydantic-ai (default: auto)
# tools_module: tools:register   # wire custom tools from tools.py
# output_schema: schema.json     # path to a JSON Schema for structured output
```

Pick a provider per run with `--provider`/`--model` (e.g. `--provider claude-cli
--model haiku` to use a local Claude Max session, or `--provider ollama --model
llama3.2`). With no flag, Himmy uses a real pydantic-ai provider when a key + the
`providers` extra are present, otherwise the offline stub. The same spec loads in
Python via `from himmy import load_agent_spec`.

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

## Running the examples

Every example runs offline against the stub:

```bash
python examples/01_basic_chat.py
python examples/02_tool_calling.py
python examples/03_structured_output.py
python examples/04_orchestration_team.py
python examples/05_workflow.py
python examples/06_postgres_storage.py   # skips cleanly unless HIMMY_TEST_POSTGRES_DSN is set
python examples/07_streaming.py          # token-delta streaming via InferenceService.run_stream
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use `asyncio.run` inside synchronous test functions, so `pytest-asyncio` is
**not** required. The full suite passes offline with the stub. Real-provider,
Postgres/pgvector, and Logfire tests **skip cleanly** when their deps/DB are absent
(set `HIMMY_TEST_POSTGRES_DSN` and install the relevant extra to exercise them);
run `pytest -q -rs` to see why each skip fired. Per-kernel `*_hardening.py` suites
cover the production-hardening contracts (failure normalization, streaming, caching,
SSRF, tenancy, pagination, observability spans, and so on).

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

## Project layout

```
himmy/
  core/            ids, errors, events
  entities/        records, registry, postgres scaffold
  agents/          personas, base_agent (task, thread, agent)
  services/        inference, context, prompts, tools, storage,
                   knowledge, evaluation, observability
  runtime/         single_agent runtime
  application/     app-level services + models
  orchestrators/   workflow engine
  api/             FastAPI app + routers
examples/          runnable, offline-first demos
tests/             pytest suite (asyncio.run based)
docker/            pgvector docker-compose for local Postgres
```
