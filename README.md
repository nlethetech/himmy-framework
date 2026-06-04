# OpenSims

OpenSims is an **offline-first, entity-backed Python agent framework**. It composes
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

OpenSims has been through a full production-hardening pass. On top of the
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

OpenSims runs end-to-end with **no API keys and no network** using the deterministic
`StubClientManager`. The smallest possible run:

```python
import asyncio

from opensims import Persona, Task
from opensims.runtime import SingleAgentRuntime
from opensims.services.inference import InferenceService, StubClientManager


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

## Architecture overview

OpenSims is organized as a set of independent **kernels** that compose through small
typed contracts. Every runtime dependency except inference is optional — drop the
entity registry and you lose lineage but keep inference; drop the tool service and
tools simply aren't bound.

| Kernel | Path | Responsibility |
|---|---|---|
| Core | `opensims/core/` | Ids, errors, run events, the `EventSink` protocol. |
| Entities | `opensims/entities/` | The lineage backbone: records, links, the in-memory registry, a Postgres scaffold. |
| Agents | `opensims/agents/` | The four primitives: `Persona`/`RolePersona`, `Task`, `ChatThread`/`Message`, `Agent`. |
| Inference | `opensims/services/inference/` | Stub + gateway + pydantic-ai client managers and the retrying `InferenceService`. |
| Prompts | `opensims/services/prompts/` | YAML-templated system/task prompts and the context→prompt mapper. |
| Context | `opensims/services/context/` | Snapshot building from storage + adapters with evidence. |
| Storage | `opensims/services/storage/` | In-memory persistence + a Postgres scaffold. |
| Knowledge | `opensims/services/knowledge/` | Chunking, embedding, retrieval, and a context adapter. |
| Tools | `opensims/services/tools/` | Local/HTTP tool registry, execution, and inference binding. |
| Evaluation | `opensims/services/evaluation/` | Metric suites over agent outputs. |
| Observability | `opensims/services/observability/` | Optional Logfire instrumentation. |
| Runtime | `opensims/runtime/` | `SingleAgentRuntime.run_task` — the orchestrating loop. |
| Application | `opensims/application/` | App-level services (runs, recommendations, dashboard). |
| Orchestrators | `opensims/orchestrators/` | Multi-step `Workflow` execution. |
| API | `opensims/api/` | FastAPI app factory and routers. |

## Running the examples

Every example runs offline against the stub:

```bash
python examples/01_basic_chat.py
python examples/02_tool_calling.py
python examples/03_structured_output.py
python examples/04_orchestration_team.py
python examples/05_workflow.py
python examples/06_postgres_storage.py   # skips cleanly unless OPENSIMS_TEST_POSTGRES_DSN is set
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
(set `OPENSIMS_TEST_POSTGRES_DSN` and install the relevant extra to exercise them);
run `pytest -q -rs` to see why each skip fired. Per-kernel `*_hardening.py` suites
cover the production-hardening contracts (failure normalization, streaming, caching,
SSRF, tenancy, pagination, observability spans, and so on).

## Optional extras

| Extra | Adds | Unlocks |
|---|---|---|
| `api` | fastapi, uvicorn | The HTTP API (`opensims.api.create_app`). |
| `providers` | pydantic-ai | Real model providers via `PydanticAIClientManager`. |
| `postgres` | asyncpg | `PostgresEntityRegistry`, `PostgresStorageService`. |
| `knowledge` | pgvector, openai, pypdf | pgvector knowledge backend, PDF reading + OpenAI-compatible embeddings. |
| `observability` | logfire | Real run/inference/tool spans via `configure_observability()`. |
| `validation` | jsonschema | Full-fidelity tool arg/output validation (offline subset used otherwise). |
| `dev` | pytest, pytest-asyncio, jsonschema | Test tooling. |
| `all` | all of the above | Everything. |

## Project layout

```
opensims/
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
