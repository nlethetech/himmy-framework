# OpenSims — BUILD SPEC (authoritative contract)

This file is the **single source of truth** for building the OpenSims scaffold.
Every builder agent MUST read this file first and conform to the symbol names,
signatures, and module paths defined here. Downstream layers conform to upstream
public APIs. When in doubt, match the documentation's wording and these contracts
exactly.

Source of truth for behavior: `/Users/samriddhagc/Downloads/opensims_agent_documentation.html`
(embedded markdown in `<script type="text/markdown" id="md-*">` blocks). Builder
agents may grep that file for any kernel they own.

Repo root: `/Users/samriddhagc/LocalProjects/himmy-agent-test`
Python: 3.12. Style: pydantic v2 `BaseModel` for all data types, `async` services,
type hints everywhere, concise docstrings on every public class/function.

---

## 0. PRIME DESIGN DECISIONS (do not deviate)

1. **Offline-first.** `pydantic_ai` is NOT installed in this environment. The
   framework MUST import, run all examples, and pass all tests with only:
   `pydantic`, `pyyaml`, `httpx`, `fastapi`, `uvicorn`, `pytest`. The default
   inference path is a deterministic **`StubClientManager`** (no network, no keys).
2. **pydantic-ai is an OPTIONAL extra.** `PydanticAIClientManager` and any
   pydantic-ai imports live behind lazy/guarded imports (`try/except ImportError`
   with a clear runtime error only when actually used). Never import `pydantic_ai`
   at module top-level on the core path.
3. **No circular imports.** Use `from __future__ import annotations` everywhere.
   Cross-kernel type hints that would create a cycle (e.g. storage <-> context)
   go under `if TYPE_CHECKING:`. The in-memory backends store passed objects in
   dicts keyed by id; they do not need upstream classes at import time.
4. **Clean degradation.** Every runtime dependency except `inference_service` is
   optional. Drop `entity_registry` -> lose lineage, keep inference. Drop
   `tool_service` -> tools not bound. Etc.
5. **Everything that matters is an entity.** Domain types expose
   `to_record(version=1, metadata=None) -> EntityRecord` using the id helpers.
6. **Postgres / pgvector / logfire are SCAFFOLD extras.** Provide real, importable
   code (DDL, classes) but gate behind optional imports; they are not exercised by
   the default test run. Mark clearly with module docstrings.

---

## 1. REPO LAYOUT (create exactly these paths)

```
himmy-agent-test/
  pyproject.toml
  README.md
  .env.example
  .gitignore
  docker/docker-compose.yml
  opensims/
    __init__.py                      # version + top-level re-exports
    core/
      __init__.py
      events.py                      # EventType, RunEvent, EventSink
      errors.py                      # OpenSimsError + shared enums
      ids.py                         # new_uuid(), utc_now_iso() (NO direct datetime.now in scripts; helper ok in lib)
    entities/
      __init__.py
      records.py                     # EntityRecord, EntityLink, EntityQuery, stable_id_for, record_id_for
      registry.py                    # EntityRegistry (in-memory)
      postgres.py                    # PostgresEntityRegistry scaffold (DDL + repo iface)
    agents/
      __init__.py
      personas/__init__.py
      personas/persona.py            # Persona, RolePersona
      base_agent/__init__.py
      base_agent/task.py             # Task, TaskConstraints (optional)
      base_agent/thread.py           # ChatThread, Message, MessageRole
      base_agent/agent.py            # Agent
    services/
      __init__.py
      inference/
        __init__.py                  # re-export public surface
        models.py
        client_manager.py
        pydantic_ai_manager.py
        service.py
      context/
        __init__.py
        models.py
        adapters.py
        service.py
      prompts/
        __init__.py
        manager.py
        mapper.py
        configs/prompts/default_prompt.yaml
      tools/
        __init__.py
        models.py
        registry.py
        service.py
        runtime_adapter.py           # ToolServiceToolset (pydantic-ai binding) — guarded import
      storage/
        __init__.py
        models.py
        service.py                   # StorageService (in-memory) + MemoryStore protocol
        postgres.py                  # PostgresStorageService scaffold (DDL)
      knowledge/
        __init__.py
        models.py
        chunker.py
        readers.py
        embedder.py
        service.py                   # KnowledgeBase + KnowledgeBaseAdapter
      evaluation/
        __init__.py
        models.py
        metrics.py
        service.py
      observability/
        __init__.py                  # configure_observability, emit_event_span, instrument_*
    runtime/
      __init__.py
      single_agent.py                # SingleAgentRuntime
    application/
      __init__.py
      models.py                      # Recommendation, RecommendationEnvelope
      services.py                    # ContextAppService, RunAppService, RecommendationAppService, DashboardQueryService
    orchestrators/
      __init__.py
      workflow.py                    # Workflow, WorkflowStep, WorkflowOrchestrator, WorkflowStepResult, WorkflowResult
    api/
      __init__.py
      app.py                         # create_app(...) factory
      deps.py                        # ApiContainer
      routers/__init__.py
      routers/context.py
      routers/runs.py
      routers/recommendations.py
      routers/dashboard.py
  examples/
    _runtime.py
    01_basic_chat.py
    02_tool_calling.py
    03_structured_output.py
    04_orchestration_team.py
    05_workflow.py
    06_postgres_storage.py
  tests/
    __init__.py
    conftest.py
    test_entities.py
    test_agents.py
    inference/__init__.py  inference/test_inference_service.py
    storage/__init__.py    storage/test_storage_service.py
    context/__init__.py    context/test_context_service.py
    prompts/__init__.py    prompts/test_prompts.py
    tools/__init__.py      tools/test_tool_service.py
    knowledge/__init__.py  knowledge/test_knowledge.py
    evaluation/__init__.py evaluation/test_evaluation.py
    runtime/__init__.py    runtime/test_runtime.py
    application/__init__.py application/test_application.py
    orchestrators/__init__.py orchestrators/test_workflow.py
    api/__init__.py        api/test_api.py
    services/__init__.py   services/test_observability.py
```

---

## 2. FOUNDATION CONTRACTS (owned by Foundation agent — everyone imports these)

### 2.1 `opensims/core/ids.py`
```python
def new_uuid() -> str               # str(uuid.uuid4())
def utc_now_iso() -> str            # datetime.now(timezone.utc).isoformat()
```

### 2.2 `opensims/core/errors.py`
- `class OpenSimsError(Exception)` base.
- Re-export nothing else; kernel-specific enums live in their kernels.

### 2.3 `opensims/core/events.py`
- `class EventType(str, Enum)` with members (exact names):
  `AGENT_RUN_STARTED, AGENT_RUN_FINISHED, INFERENCE_REQUESTED, INFERENCE_SUCCEEDED,
   INFERENCE_FAILED, TOOL_CALLED, TOOL_COMPLETED, TOOL_FAILED,
   CONTEXT_SNAPSHOT_BUILT, WORKFLOW_STARTED, WORKFLOW_STEP_COMPLETED, WORKFLOW_FINISHED`.
- `class RunEvent(BaseModel)` fields:
  `event_id: str = Field(default_factory=new_uuid)`,
  `event_type: EventType`, `trace_id: str | None = None`, `thread_id: str | None = None`,
  `agent_id: str | None = None`, `request_id: str | None = None`,
  `tool_call_id: str | None = None`, `latency_ms: float | None = None`,
  `cost: float | None = None`, `payload: dict[str, Any] = {}`,
  `error: str | None = None`, `timestamp: str = Field(default_factory=utc_now_iso)`.
  Method: `to_record(version=1, metadata=None) -> EntityRecord` (kind="run_event",
  stable id namespace "run_event" keyed by event_id). Import EntityRecord lazily
  inside the method to avoid a cycle.
- `class EventSink(Protocol)`: `async def append_event(self, event: RunEvent) -> None`.

### 2.4 `opensims/entities/records.py`
- `def stable_id_for(value: str, *, namespace: str, fallback_key: str | None = None) -> str`
  UUID5 of (namespace, value). If `value` already parses as a UUID, return it
  unchanged. If `value` falsy, use `fallback_key`.
- `def record_id_for(*, stable_id: str, version: int, kind: str) -> str`
  UUID5 of (kind, stable_id, version) -> deterministic.
- `class EntityRecord(BaseModel)`:
  `record_id: str` (set via record_id_for at construction if not provided — use a
  validator/factory helper or compute in a classmethod), `stable_id: str`,
  `version: int = 1`, `kind: str`, `payload: dict[str, Any] = {}`,
  `metadata: dict[str, Any] = {}`, `created_at: str = Field(default_factory=utc_now_iso)`.
  Provide `@classmethod def create(cls, *, stable_id, version, kind, payload, metadata)`
  that computes record_id deterministically.
- `class EntityLink(BaseModel)`:
  `link_id: str = Field(default_factory=new_uuid)`, `from_record_id: str`,
  `to_record_id: str`, `relation: str`, `metadata: dict = {}`,
  `created_at: str = Field(default_factory=utc_now_iso)`.
- `class EntityQuery(BaseModel)`:
  `kind: str | None = None`, `metadata_filters: dict[str, Any] = {}`,
  `stable_id: str | None = None`.

### 2.5 `opensims/entities/registry.py`
- `class EntityRegistry` (in-memory). Methods:
  - `register(self, record: EntityRecord) -> EntityRecord` — idempotent on record_id.
  - `new_version(self, *, stable_id, kind, payload, metadata=None, expected_version=None) -> EntityRecord`
    — optimistic concurrency: if `expected_version` given and != current latest
    version, raise `OpenSimsError`. New version = latest+1.
  - `link(self, *, from_record_id, to_record_id, relation, metadata=None) -> EntityLink`.
  - `get(self, record_id) -> EntityRecord | None`.
  - `get_latest(self, stable_id) -> EntityRecord | None`.
  - `get_history(self, stable_id) -> list[EntityRecord]` (ascending version).
  - `list_by_kind(self, kind) -> list[EntityRecord]`.
  - `query(self, q: EntityQuery) -> list[EntityRecord]`.
  - `links_from(self, record_id) -> list[EntityLink]` (helper).
- `__init__.py` re-exports: EntityRecord, EntityLink, EntityQuery, EntityRegistry,
  stable_id_for, record_id_for.

### 2.6 `opensims/agents/personas/persona.py`
- `class Persona(BaseModel)`:
  `agent_id: str = Field(default_factory=new_uuid)`, `name: str`,
  `description: str = ""`, `instructions: list[str] = []`, `tags: list[str] = []`,
  `metadata: dict[str, Any] = {}`.
  Method `to_record(version=1, metadata=None)` -> kind="persona",
  stable id `stable_id_for(agent_id, namespace="persona", fallback_key=name)`.
  Convenience: `@property role` -> `metadata.get("role") or name`.
- `class RolePersona(Persona)`: adds `objectives: list[str] = []`,
  `required_tools: list[str] = []`, `required_skills: list[str] = []`.

### 2.7 `opensims/agents/base_agent/task.py`
- `class Task(BaseModel)`:
  `task_id: str = Field(default_factory=new_uuid)`, `title: str`, `prompt: str`,
  `context: dict[str, Any] = {}`, `constraints: dict[str, Any] = {}`,
  `metadata: dict[str, Any] = {}`.
  `to_record()` -> kind="prompt", stable id namespace "prompt" keyed by task_id.

### 2.8 `opensims/agents/base_agent/thread.py`
- `class MessageRole(str, Enum)`: SYSTEM, USER, ASSISTANT, TOOL.
- `class Message(BaseModel)`:
  `message_id: str = Field(default_factory=new_uuid)`, `role: MessageRole`,
  `content: str = ""`, `metadata: dict[str, Any] = {}`,
  `created_at: str = Field(default_factory=utc_now_iso)`.
  `to_record()` -> kind="message" keyed by message_id.
- `class ChatThread(BaseModel)`:
  `thread_id: str = Field(default_factory=new_uuid)`, `agent_id: str | None = None`,
  `messages: list[Message] = []`, `version: int = 1`, `metadata: dict = {}`.
  Methods: `append_message(self, message: Message) -> Message`;
  `@property last_message -> Message | None`;
  `to_record(version=None,...)` -> kind="chat_thread" keyed by thread_id (uses
  self.version when version arg None).

### 2.9 `opensims/agents/base_agent/agent.py`
- `class Agent(BaseModel)`:
  `agent_id: str = Field(default_factory=new_uuid)`, `name: str`,
  `persona: Persona`, `instructions: list[str] = []`, `tools: list[str] = []`,
  `skills: list[str] = []`, `metadata: dict = {}`.
  `@classmethod from_persona(cls, persona, *, name, tools=None, skills=None, instructions=None) -> Agent`.
  `to_record()` -> kind="agent" keyed by agent_id.

### 2.10 `opensims/agents/__init__.py`
Re-export Persona, RolePersona, Task, ChatThread, Message, MessageRole, Agent.

---

## 3. INFERENCE KERNEL (owned by Inference agent) — `opensims/services/inference/`

### 3.1 `models.py`
- `class InferenceStatus(str, Enum)`: SUCCESS, FAILED.
- `class ResponseFormat(str, Enum)`: TEXT, JSON_OBJECT, STRUCTURED_OUTPUT,
  AUTO_TOOLS, WORKFLOW, TOOL.
- `class InferenceErrorCode(str, Enum)`: AUTH, QUOTA, RATE_LIMITED,
  PROVIDER_UNAVAILABLE, INVALID_REQUEST, TIMEOUT, UNKNOWN.
- `class InferenceError(BaseModel)`: `code: InferenceErrorCode`, `message: str`,
  `retryable: bool = False`.
- `class InferenceMessage(BaseModel)`: `role: str` (system|user|assistant|tool),
  `content: str = ""`, `metadata: dict = {}`, `tool_call_id: str | None = None`,
  `name: str | None = None`. Method `to_model_message()` -> placeholder for
  pydantic-ai rebuild (return dict; document that real path uses
  pydantic_ai.messages). Keep import-safe.
- `class ToolCallRecord(BaseModel)`: `tool_call_id: str`, `tool_name: str`,
  `args: dict[str, Any] = {}`.
- `class ToolReturnRecord(BaseModel)`: `tool_call_id: str`, `tool_name: str`,
  `content: Any = None`, `outcome: str = "success"`, `metadata: dict = {}`.
- `class BoundTool(BaseModel)`: internal binding used by the offline stub path.
  `name: str`, `description: str = ""`, `args_json_schema: dict = {}`,
  `output_json_schema: dict | None = None`,
  `handler: Callable[[dict], Awaitable[ToolReturnRecord]] | None = None`
  (use `model_config = ConfigDict(arbitrary_types_allowed=True)`).
- `class WorkflowDefinition(BaseModel)`: `steps: list[str] = []` (tool names per step).
- `class WorkflowState(BaseModel)`: `definition: WorkflowDefinition`,
  `current_step: int = 0`.
  `@property is_complete -> bool` (current_step >= len(steps)).
  `@property current_tool_name -> str | None`.
  `def advance(self) -> WorkflowState` (returns new state current_step+1).
- `class InferenceRequest(BaseModel)`:
  `request_id: str = Field(default_factory=new_uuid)`, `model_key: str = "default"`,
  `messages: list[InferenceMessage] = []`, `response_format: ResponseFormat | None = None`,
  `output_json_schema: dict | None = None`, `workflow: WorkflowState | None = None`,
  `generation_params: dict[str, Any] = {}`, `timeout_seconds: float = 30.0`,
  `route_override: str | None = None`, `metadata: dict = {}`, `toolsets: list = []`,
  `bound_tools: list[BoundTool] = []`,
  `tool_names_override: list[str] | None = None`.
  Validator: if `response_format is None` and `output_json_schema` set ->
  STRUCTURED_OUTPUT; if `workflow` set -> WORKFLOW.
  `model_config = ConfigDict(arbitrary_types_allowed=True)`.
- `class InferenceResponse(BaseModel)`:
  `request_id: str`, `status: InferenceStatus`, `output_text: str | None = None`,
  `output_structured: Any = None`, `tool_calls: list[ToolCallRecord] = []`,
  `tool_returns: list[ToolReturnRecord] = []`, `workflow: WorkflowState | None = None`,
  `input_tokens: int = 0`, `output_tokens: int = 0`, `cost: float = 0.0`,
  `latency_ms: float = 0.0`, `model_path: str = ""`, `provider_name: str = ""`,
  `error: InferenceError | None = None`.
- `class LLMConfig(BaseModel)`:
  `model_key: str = "default"`, `response_format: ResponseFormat | None = None`,
  `output_json_schema: dict | None = None`, `temperature: float | None = None`,
  `max_tokens: int | None = None`, `top_p: float | None = None`,
  `timeout_seconds: float | None = None`, `use_cache: bool | None = None`,
  `route_override: str | None = None`, `workflow: WorkflowState | None = None`,
  `extra_params: dict[str, Any] = {}`.
  Validators auto-derive response_format from output_json_schema/workflow; conflicts
  raise ValueError. `model_config = ConfigDict(arbitrary_types_allowed=True)`.
- `class BatchInferenceRequest(BaseModel)`: `requests: list[InferenceRequest]`,
  `max_concurrency: int = 8`.
- `class BatchInferenceResponse(BaseModel)`: `responses: list[InferenceResponse]`,
  `success_count: int`, `failure_count: int`, `elapsed_ms: float`.
- Gateway config types (used by GatewayClientManager):
  `class GatewayModelConfig(BaseModel)`: `api_format: str`, `model_name: str`.
  `class GatewayRuntimeConfig(BaseModel)`: `region: str = "us"`,
  `base_url: str | None = None`, `model_registry: dict[str, GatewayModelConfig] = {}`.
- Helper: `def synthesize_from_schema(schema: dict, *, seed_text: str = "") -> Any`
  — produce a minimal VALID instance of a JSON schema (object/array/string/number/
  integer/boolean/enum/required/default). For string fields whose name hints at
  prose (title/summary/rationale/content/text/headline/key_takeaway), inject
  `seed_text` (truncated). This makes STRUCTURED_OUTPUT work offline. EXPORT it.

### 3.2 `client_manager.py`
- `class ClientManager(Protocol)`:
  `def resolve(self, model_key: str) -> str` (returns a model_path string), and
  `async def generate(self, request: InferenceRequest) -> InferenceResponse`.
  (We keep generation in the client manager so the stub can fully simulate.)
- `class StubClientManager:` DEFAULT offline manager. Constructor:
  `__init__(self, *, default_model_path="stub:echo", latency_ms=1.0)`.
  `generate()` behavior:
  - Resolve `model_path` from model_key (e.g. f"stub:{model_key}").
  - Compute a deterministic `output_text` echo summarizing the last user message +
    any system role, e.g. a short templated answer. Token counts ~ len-based.
  - response_format handling:
    - TEXT / JSON_OBJECT / None / AUTO_TOOLS: if `bound_tools` present AND
      response_format in (AUTO_TOOLS, None) -> call EACH bound tool once with args
      synthesized from its args_json_schema (via synthesize_from_schema), record
      ToolCallRecord + ToolReturnRecord (run handler if present), then produce a
      final text that references the tool outputs. Otherwise plain echo text.
    - STRUCTURED_OUTPUT: require output_json_schema; set output_structured =
      synthesize_from_schema(schema, seed_text=<user prompt>); output_text =
      json.dumps(output_structured).
    - WORKFLOW: require request.workflow; tool name = workflow.current_tool_name;
      find that bound tool; synthesize args; produce ONE ToolCallRecord +
      ToolReturnRecord; set response.workflow = request.workflow (unchanged — the
      CALLER advances). output_text minimal.
    - TOOL: raise NotImplementedError (matches doc).
  - Always status=SUCCESS unless an internal error -> FAILED with InferenceError.
- `class GatewayClientManager:` constructor `(self, runtime_config: GatewayRuntimeConfig)`.
  For the scaffold, if `PYDANTIC_AI_GATEWAY_API_KEY` / pydantic-ai unavailable, it
  may delegate to a StubClientManager internally OR raise a clear error on
  `generate()` explaining gateway requires the `[providers]` extra. Resolve()
  returns f"{region}:{model_name}". Document it as production routing.

### 3.3 `pydantic_ai_manager.py`
- `class PydanticAIClientManager:` constructor takes a `model_registry: dict[str,str]`
  (model_key -> pydantic-ai model string) plus optional default.
  Top of file: NO module-level `import pydantic_ai`. Inside `generate()` (and a
  `_require_pydantic_ai()` helper) do `try: import pydantic_ai except ImportError:
  raise OpenSimsError("pydantic-ai not installed; pip install 'opensims[providers]'")`.
  Implement a best-effort real path (build an Agent via infer_model, run, map to
  InferenceResponse) but it's fine for it to be a documented thin adapter since it
  cannot be tested here. Keep it import-safe (the FILE must import even when
  pydantic_ai is missing).

### 3.4 `service.py`
- `class InferenceService:`
  `__init__(self, client_manager: ClientManager, *, max_retries=2,
   default_timeout_seconds=90.0, retry_base_delay_seconds=0.2,
   retry_jitter_seconds=0.1, event_sink: EventSink | None = None)`.
  - `async def run(self, request) -> InferenceResponse`: validate, apply timeout,
    bounded retries on retryable error codes only (RATE_LIMITED, TIMEOUT,
    PROVIDER_UNAVAILABLE). AUTH/QUOTA/INVALID_REQUEST never retried. Wrap each
    attempt in `asyncio.wait_for(..., timeout + 1.0)`. Delegates to
    `client_manager.generate`. Stamp latency_ms.
  - `async def run_batch(self, batch: BatchInferenceRequest) -> BatchInferenceResponse`:
    bounded concurrency via `asyncio.Semaphore(max_concurrency)`, preserve order,
    count successes/failures, measure elapsed.
  - Keep a `_run_once` seam (for parity with doc).
- `__init__.py` re-exports the full public surface listed across 3.1–3.4:
  InferenceService, InferenceRequest, InferenceResponse, InferenceMessage,
  InferenceStatus, ResponseFormat, LLMConfig, InferenceError, InferenceErrorCode,
  ToolCallRecord, ToolReturnRecord, BoundTool, BatchInferenceRequest,
  BatchInferenceResponse, WorkflowDefinition, WorkflowState, StubClientManager,
  GatewayClientManager, GatewayModelConfig, GatewayRuntimeConfig,
  PydanticAIClientManager, ClientManager, synthesize_from_schema.

---

## 4. DATA PLANE (owned by Data agent): storage + context + knowledge

These three are co-owned by ONE agent to avoid cross-agent seams (context writes
through to storage; knowledge persists to storage; KnowledgeBaseAdapter feeds
context). Use TYPE_CHECKING to avoid import cycles.

### 4.1 STORAGE — `opensims/services/storage/`
`models.py`:
- `class RunStatus(str, Enum)`: QUEUED, RUNNING, SUCCEEDED, FAILED.
- `class RecommendationStatus(str, Enum)`: PROPOSED, ACCEPTED, DISMISSED, SCHEDULED.
- `class RunRecord(BaseModel)`: `run_id: str = Field(default_factory=new_uuid)`,
  `workspace_id: str`, `subject_id: str`, `task_id: str | None=None`,
  `thread_id: str | None=None`, `snapshot_id: str | None=None`,
  `persona_name: str | None=None`, `model_key: str | None=None`,
  `idempotency_key: str | None=None`, `status: RunStatus = RunStatus.QUEUED`,
  `output_text: str | None=None`, `output_structured: Any=None`,
  `error: str | None=None`, `trace_id: str | None=None`, `metadata: dict={}`,
  `created_at/updated_at: str = factory utc_now_iso`.
- `class RecommendationItem(BaseModel)`: `recommendation_id: str = factory`,
  `run_id: str`, `workspace_id: str`, `subject_id: str`, `kind: str`,
  `title: str`, `summary: str = ""`, `rationale: str = ""`,
  `confidence: float = 0.0`, `evidence_refs: list[str] = []`,
  `status: RecommendationStatus = PROPOSED`, `notes: str | None=None`,
  `metadata: dict={}`, `created_at: str = factory`.
- Also define minimal: `MemoryObject`, `EpisodicMemoryObject`, `AgentStateRecord`,
  `ActionRecord`, `EnvironmentStateRecord`, `ContextEvidenceRecord`
  (each a small BaseModel with an id + payload + metadata — enough to satisfy the
  documented "what it persists" table; do not over-engineer).
`service.py`:
- `class MemoryStore(Protocol)`: the conversation/event subset
  (`save_thread`, `load_thread`, `append_event`, `list_events`).
- `class StorageService:` in-memory, async. Methods (all `async`):
  - Threads: `save_thread(thread)`, `load_thread(thread_id) -> ChatThread | None`.
  - Events: `append_event(event: RunEvent)`, `list_events(thread_id=None, trace_id=None) -> list[RunEvent]`.
  - Context: `save_context_field(field)`, `get_context_field(subject_id, key) -> ContextField|None`,
    `list_context_fields(subject_id) -> list[ContextField]`,
    `save_snapshot(snapshot)`, `load_snapshot(snapshot_id) -> ContextSnapshot|None`,
    `save_context_evidence(record)`.
  - Runs: `save_run(run)`, `get_run(run_id) -> RunRecord|None`,
    `list_runs(workspace_id=None, subject_id=None, status=None) -> list[RunRecord]`,
    `load_run_by_idempotency(workspace_id, idempotency_key) -> RunRecord|None`.
  - Recommendations: `save_recommendation(item)`, `get_recommendation(id)`,
    `list_recommendations(workspace_id=None, subject_id=None, run_id=None, kind=None, status=None)`,
    `update_recommendation(id, *, status=None, notes=None) -> RecommendationItem|None`.
  - Evaluation: `save_evaluation_run(run)`, `get_evaluation_run(id)`,
    `list_evaluation_runs(...)`.
  - Memory/orchestration: simple save/get/list for the records above (can be thin).
  - implements `append_event`/`save_thread`/etc. so it satisfies EventSink + MemoryStore.
  Type hints for ContextField/ContextSnapshot/EvaluationRun under TYPE_CHECKING;
  parameters typed as `Any` at runtime is acceptable.
`postgres.py`:
- `class PostgresStorageService:` SCAFFOLD. `@classmethod async def connect(cls, dsn, *, max_size=10)`,
  `async def create_schema(self)` (idempotent CREATE TABLE IF NOT EXISTS — write the
  ~14-table DDL as a string; do not require a live DB to import), plus the same
  method surface as StorageService raising `OpenSimsError("requires [postgres] extra
  + running DB")` if asyncpg/conn missing. Guard `import asyncpg` lazily. Include an
  `ai_call_log` view in the DDL string (documented). The MODULE must import without
  asyncpg/DB present.
`__init__.py` re-exports: StorageService, MemoryStore, RunRecord, RunStatus,
RecommendationItem, RecommendationStatus, PostgresStorageService, + the memory/orch records.

### 4.2 CONTEXT — `opensims/services/context/`
`models.py`:
- `class ContextSourcePreference(str, Enum)`: STORAGE_FIRST, TOOL_FIRST, TOOL_ONLY.
- `class EvidenceRef(BaseModel)`: `evidence_id: str = factory`, `source_type: str`,
  `source_id: str | None=None`, `row_id: str | None=None`,
  `account_scope: dict={}`, `metadata: dict={}`.
- `class ContextField(BaseModel)`: `key: str`, `value: Any=None`,
  `confidence: float = 1.0`, `freshness_seconds: float | None=None`,
  `source: str = "storage"`, `evidence_refs: list[EvidenceRef] = []`,
  `metadata: dict={}`.
- `class ContextSpecKey(BaseModel)`: `key: str`, `required: bool=False`,
  `source_preference: ContextSourcePreference = STORAGE_FIRST`,
  `adapter_name: str | None=None`, `metadata: dict={}`.
- `class ContextBuildSpec(BaseModel)`: `spec_id: str = factory`,
  `keys: list[ContextSpecKey] = []`, `metadata: dict={}`.
- `class ContextSnapshot(BaseModel)`: `snapshot_id: str = factory`,
  `subject_id: str`, `task_id: str | None=None`,
  `fields: dict[str, ContextField] = {}`, `missing_required_keys: list[str] = []`,
  `metadata: dict={}`, `created_at: str = factory`.
  `to_record()` -> kind="context_snapshot" keyed by snapshot_id.
`adapters.py`:
- `class ContextAdapter(ABC)`: class attr `name: str`; `async def fetch(self, key,
  scope: dict) -> ContextField | None`.
`service.py`:
- `class ContextService:` `__init__(self, *, storage_service, adapters=None,
  entity_registry=None)`. Build `self._adapters = {a.name: a for a in adapters}`.
  - `async def build_snapshot(self, *, subject_id, task_id=None, build_spec,
     metadata=None) -> ContextSnapshot`. For each ContextSpecKey honor
     source_preference (STORAGE_FIRST/TOOL_FIRST/TOOL_ONLY) using storage
     get_context_field + the named adapter's fetch. Adapter results not from
     storage are written through (`save_context_field`) EXCEPT TOOL_ONLY. Populate
     missing_required_keys for required keys with no value. Persist snapshot +
     each EvidenceRef (save_context_evidence) and register entities when registry
     present. Accept build_spec as ContextBuildSpec OR dict (model_validate).
`__init__.py` re-exports all of the above.

### 4.3 KNOWLEDGE — `opensims/services/knowledge/`
`embedder.py`:
- `class EmbedderProtocol(Protocol)`: `async def embed_documents(self, texts:
  list[str]) -> list[list[float]]`; `async def embed_query(self, text) -> list[float]`.
- `class DeterministicEmbedder:` offline, no network. Hash-based fixed-dim vectors
  (default dim 64) — same text -> same vector, normalized. DEFAULT for tests.
- `def build_openai_compatible_embedder(...)`: SCAFFOLD — returns a thin wrapper that
  raises a clear error unless `openai`/pydantic-ai available; document env vars
  `OPENAI_COMPATIBLE_*`. Import-safe.
- `class OpenAIMultimodalEmbeddingModel:` SCAFFOLD stub class (documented), import-safe.
`chunker.py`:
- `class SemanticChunker:` `__init__(self, *, max_chars=800, overlap=100)`;
  `def chunk(self, text: str) -> list[tuple[int,int,str]]` (start,end,text).
`readers.py`:
- `class DocumentReader(ABC)`: `extensions: tuple[str,...]`; `def read(self, path) -> str`.
- `class TextReader(DocumentReader)`: `.txt/.md`.
- `class PDFReader(DocumentReader)`: `.pdf` — lazy `import pypdf`; clear error if missing.
- `class DocumentReaderFactory:` register/readers-by-ext; `def read(self, path) -> str`.
`models.py`:
- `KnowledgeBaseRecord` (kb_id, workspace_id, client_id, name, vector_dim, metadata),
  `KnowledgeDocument` (document_id, kb_id, title, source_uri, text|None, metadata),
  `KnowledgeChunk` (chunk_id, document_id, kb_id, text|None, start_pos, end_pos,
   embedding: list[float], chunk_kind="text", image_uri|None, caption|None, metadata),
  `RetrievedChunk` (text|None, similarity, context_window|None, document_id,
   source_uri|None, chunk_kind, metadata),
  `DocumentInput` (text|None, file|None, title|None, source_uri|None, metadata;
   validator: exactly one of text/file).
`service.py`:
- `class KnowledgeBase:` `__init__(self, *, storage, embedder=None,
  chunker=None, reader_factory=None, max_embed_batch=512, max_concurrent_embeds=4)`.
  In-memory store (dicts) by default; storage param accepted for parity. Methods
  (async): `create_kb(*, workspace_id, client_id, name, vector_dim=64) -> KnowledgeBaseRecord`
  (unique on (workspace_id,client_id,name)); `ingest_text(kb_id, text, *, title=None,
  source_uri=None, metadata=None)`; `ingest_file(kb_id, path, *, metadata=None)`;
  `ingest_documents(kb_id, docs: list[DocumentInput])` (single batched embed call);
  `ingest_directory(kb_id, path, *, glob="**/*", metadata=None)`;
  `ingest_image(...)` (raise if embedder not multimodal); `search(kb_id, query, *,
  top_k=5, similarity_threshold=0.0, metadata_filters=None) -> list[RetrievedChunk]`
  (cosine top-k, with context window from parent doc); `delete_document(...)`,
  `delete_kb(...)`.
- `class KnowledgeBaseAdapter(ContextAdapter):` `name = "knowledge_base"`.
  `__init__(self, kb_service)`. `fetch(key, scope)` -> resolves kb by
  (workspace_id, client_id|subject_id, kb_name from spec metadata), runs search,
  returns a ContextField shaped per the doc (value={chunks, rendered_text},
  confidence=max similarity, evidence_refs from chunks). This delivers the
  "next-PR" piece from the doc — implement it for the in-memory backend.
`__init__.py` re-exports the public surface incl. KnowledgeBase, KnowledgeBaseAdapter,
DeterministicEmbedder, DocumentInput, RetrievedChunk, build_openai_compatible_embedder,
OpenAIMultimodalEmbeddingModel.

---

## 5. PROMPTS (owned by Foundation agent — it consumes Persona/Task) — `opensims/services/prompts/`
`configs/prompts/default_prompt.yaml`: exactly the two-section template from the doc
(persona: role/background/objectives/skills ; task: task/output/schema).
`manager.py`:
- `class SystemPromptVariables(BaseModel)`: `role: str = ""`, `persona: str = ""`,
  `objectives: list[str] = []`, `skills: list[str] = []`, `datetime: str = ""`.
- `class TaskPromptVariables(BaseModel)`: `task: str = ""`, `output_format: str = ""`,
  `output_schema: dict | None = None`.
- `class PromptTemplate:` loads + merges YAML files (later files win per section-key).
- `class PromptManager:` `__init__(self, *, template_paths: list[Path] | None = None)`
  (default = bundled default_prompt.yaml). Methods:
  `get_system_prompt(vars: SystemPromptVariables) -> str`,
  `get_task_prompt(vars: TaskPromptVariables) -> str`. Render: include a section's
  text only when its variable(s) are non-empty; safe-substitute missing keys.
`mapper.py`:
- `class ContextPromptKey(BaseModel)`: `key: str`, `required: bool=False`.
- `class ContextPromptMapSpec(BaseModel)`: `system_keys: list[ContextPromptKey] = []`,
  `task_keys: list[ContextPromptKey] = []`. (Accept raw dicts/strings -> coerce.)
- `class ContextPromptMapper:` `def project(self, snapshot, spec) -> tuple[str, str,
  list[str]]` returning (system_block, task_block, missing_required_keys). Each
  selected key rendered as a labelled block (`### key\n<value>`).
`__init__.py` re-exports all of the above.

---

## 6. TOOLS + EVAL + OBSERVABILITY (owned by ToolsEval agent)

### 6.1 TOOLS — `opensims/services/tools/`
`models.py`:
- `class ToolBackendKind(str, Enum)`: LOCAL, HTTP.
- `class HttpAuthMode(str, Enum)`: NONE, BEARER, HEADER, BASIC.
- `class HttpAuthConfig(BaseModel)`: `mode: HttpAuthMode = NONE`,
  `env_var: str | None=None`, `header_name: str | None=None`.
- `class HttpToolConfig(BaseModel)`: `base_url_env_var: str`, `method: str = "GET"`,
  `path_template: str = "/"`, `auth: HttpAuthConfig = HttpAuthConfig()`,
  `query_arg_names: list[str] = []`, `body_arg_names: list[str] = []`,
  `header_arg_names: list[str] = []`, `timeout_seconds: float = 15.0`.
- `class ToolDefinition(BaseModel)`: `name: str`, `kind: ToolBackendKind`,
  `description: str = ""`, `args_json_schema: dict = {}`,
  `output_json_schema: dict | None=None`, `requires_approval: bool=False`,
  `timeout_seconds: float | None=None`, `retry_hints: dict={}`,
  `sequential: bool=False`, `http_config: HttpToolConfig | None=None`,
  `metadata: dict={}`. `model_config = ConfigDict(arbitrary_types_allowed=True)`.
  Store the local handler OUT of the pydantic model (registry keeps a name->handler
  map) to keep ToolDefinition serializable. `to_record()` kind="tool_definition"
  keyed by name.
- `class ToolInvocation(BaseModel)`: `tool_call_id: str = factory`, `tool_name: str`,
  `args: dict = {}`, `metadata: dict = {}`.
- `class ToolPolicyDecision(BaseModel)`: `allow: bool = True`, `reason: str | None=None`,
  `transformed_args: dict | None=None`.
- `class ToolErrorCode(str, Enum)`: NOT_FOUND, POLICY_BLOCKED, TIMEOUT,
  RATE_LIMITED, PROVIDER_UNAVAILABLE, INVALID_REQUEST, OUTPUT_VALIDATION,
  EXECUTION_ERROR, UNKNOWN.
- `class ToolExecutionResult(BaseModel)`: `tool_call_id: str`, `tool_name: str`,
  `outcome: str` (success|failed|denied), `result: Any=None`,
  `error_code: ToolErrorCode | None=None`, `error_message: str | None=None`,
  `latency_ms: float=0.0`, `metadata: dict={}`.
`registry.py`:
- `class ToolRegistry:` `__init__(self, *, entity_registry=None)`. Holds
  definitions + local handler map. `register(definition, handler=None)`;
  `get(name) -> ToolDefinition|None`; `list() -> list[ToolDefinition]`;
  `handler_for(name)`. When entity_registry present, register tool_definition record.
- `def register_local_tool(registry, *, name, handler, description="",
  args_json_schema=None, output_json_schema=None, requires_approval=False,
  sequential=False, metadata=None) -> ToolDefinition`.
- `def register_http_tool(registry, *, name, http_config, description="",
  args_json_schema=None, output_json_schema=None, requires_approval=False,
  metadata=None) -> ToolDefinition`.
`service.py`:
- `class ToolService:` `__init__(self, registry, *, pre_execution_hook=None,
  post_execution_hook=None, event_sink=None)`. Hooks are async callables
  `(invocation, definition) -> ToolPolicyDecision` (pre) and `(result, definition)
  -> Any` (post). Method `async def execute(self, invocation: ToolInvocation) ->
  ToolExecutionResult`: lookup def (NOT_FOUND), run pre-hook (deny -> outcome
  "denied", POLICY_BLOCKED), dispatch LOCAL (call handler, allow sync or async) or
  HTTP (build request via httpx, lazy import; env-backed base url + auth; on
  missing httpx/env raise EXECUTION_ERROR/INVALID_REQUEST gracefully), validate
  output against output_json_schema (OUTPUT_VALIDATION on failure), emit
  TOOL_CALLED/TOOL_COMPLETED/TOOL_FAILED events. Provide
  `def bound_tools(self, names: list[str] | None = None) -> list[BoundTool]`
  returning inference BoundTool objects whose handler wraps `execute` (this is the
  offline binding the runtime feeds to InferenceRequest.bound_tools).
`runtime_adapter.py`:
- `class ToolServiceToolset:` SCAFFOLD pydantic-ai binding. Guarded import; converts
  ToolDefinitions to pydantic-ai Tools routing back through ToolService.execute.
  Import-safe without pydantic_ai. Document it.
`__init__.py` re-exports: ToolRegistry, ToolService, ToolDefinition, ToolBackendKind,
ToolInvocation, ToolPolicyDecision, ToolErrorCode, ToolExecutionResult, HttpToolConfig,
HttpAuthConfig, HttpAuthMode, register_local_tool, register_http_tool.

### 6.2 EVALUATION — `opensims/services/evaluation/`
`models.py`: `EvaluationCase` (case_id, input: dict, expected_output: dict,
metric_weights: dict[str,float], metadata), `EvaluationSuite` (suite_id, name,
cases: list[EvaluationCase]), `MetricScore` (metric: str, score: float (0..1),
passed: bool, detail: str=""), `EvaluationCaseResult` (case_id, metric_scores:
list[MetricScore], aggregate: float, passed: bool, actual_output: Any),
`EvaluationRun` (run_id, suite_id, suite_name, aggregate_score: float,
case_results: list[EvaluationCaseResult], created_at).
`metrics.py`: `class MetricEvaluator(Protocol)`: `def score(self, case, actual) ->
MetricScore`. Built-ins (functions/classes) for `accuracy, relevance, groundedness,
safety, calibration` per the doc descriptions. `class EvaluationMetricRegistry:`
register/get/`default()`. `default_metric_registry` instance with all five.
`service.py`: `class EvaluationService:` `__init__(self, *, metric_registry=None,
storage_service=None)`. `async def run_suite(self, *, suite, actual_outputs:
dict[str, Any]) -> EvaluationRun` — weighted aggregate per case, overall aggregate,
persist via storage.save_evaluation_run when present.
`__init__.py` re-exports the surface incl. default_metric_registry.

### 6.3 OBSERVABILITY — `opensims/services/observability/__init__.py`
- `def configure_observability() -> None`: idempotent. No-op when
  `OPENSIMS_LOGFIRE_ENABLED` unset/false. If enabled but `logfire` missing -> raise
  RuntimeError. NEVER import logfire unless enabled.
- `def emit_event_span(event) -> None`: safe no-op when off.
- `def instrument_fastapi(app) -> None` / `def instrument_asyncpg() -> None`: soft
  no-op (print one-line warning) when sub-extra missing.
- Respect `OPENSIMS_LOGFIRE_INCLUDE_CONTENT`, `OPENSIMS_LOGFIRE_SERVICE_NAME`.

---

## 7. INTEGRATION (owned by Integration agent): runtime + application + api + orchestrators

### 7.1 RUNTIME — `opensims/runtime/single_agent.py`
- `class SingleAgentRuntime:` constructor exactly:
  `__init__(self, *, inference_service, memory_store=None, tool_service=None,
   context_service=None, prompt_manager=None, context_prompt_mapper=None,
   entity_registry=None, default_model_key="default", save_threads=True)`.
  Auto-create PromptManager()/ContextPromptMapper() if None and available.
- `async def run_task(self, persona: Persona, task: Task, thread: ChatThread | None
  = None, *, llm_config: LLMConfig | None = None, snapshot_id: str | None = None)
  -> ChatThread`. Steps (mirror the doc sequence diagram):
  1. Resolve/build snapshot: snapshot_id arg, else task.context["snapshot_id"],
     else task.context["context_build_spec"] via context_service.build_snapshot
     (subject = task.context.get("context_subject_id") or persona.agent_id).
     Emit CONTEXT_SNAPSHOT_BUILT.
  2. Render prompts: SystemPromptVariables from persona (+ task.context overrides:
     role/objectives/skills/system_prefix/datetime). Project snapshot keys via
     context_prompt_mapper + task.context["context_prompt_map_spec"] -> append
     blocks to system/task. TaskPromptVariables from task.prompt + output_format +
     output_schema.
  3. Create thread if None (agent_id=persona.agent_id). On FIRST run append SYSTEM
     message. Append USER task prompt.
  4. Register persona/agent/prompt entities (when registry present).
  5. Append AGENT_RUN_STARTED event (model_key, snapshot_id, trace_id =
     f"{thread_id}:{task_id}").
  6. Resolve effective model/response_format/schema from llm_config (precedence)
     else task.context (model_key/output_schema/output_format/tool_names). Build
     InferenceRequest with messages (map thread -> InferenceMessage list),
     response_format, output_json_schema, workflow, generation_params (temp/max
     tokens/top_p from llm_config), bound_tools from tool_service.bound_tools(
     tool_names) (respect tool_names_override for WORKFLOW = [current_tool_name]).
     Emit INFERENCE_REQUESTED (request_id, rendered_prompt, retrieval_ctx).
  7. Call inference_service.run. On success emit INFERENCE_SUCCEEDED (tokens/cost);
     on failure INFERENCE_FAILED.
  8. For each tool_call/tool_return pair -> append a TOOL Message with the full
     metadata schema (tool_call_id, tool_name, tool_outcome, tool_args, request_id,
     trace_id, timestamp). Register message entities.
  9. Append ASSISTANT message: output_text (or json.dumps(output_structured)).
     Stamp latency/model metadata on the message.
  10. Register message + chat_thread entity version (bump thread.version on 2nd+
     turn). Link relations (uses_persona, in_thread, thread_for_agent, built_from,
     observed_in_run) when registry present.
  11. AGENT_RUN_FINISHED (total latency+cost). save_thread when save_threads.
     Return thread. response.workflow (if any) is reachable; for WORKFLOW the
     caller advances state.
  - Be defensive: every optional dep guarded. Never crash if registry/context/tool
    services are None.
- `opensims/runtime/__init__.py` re-exports SingleAgentRuntime.

### 7.2 APPLICATION — `opensims/application/`
`models.py`:
- `class Recommendation(BaseModel)`: `kind: str`, `title: str`, `summary: str=""`,
  `rationale: str=""`, `confidence: float=0.0`, `evidence_refs: list[str]=[]`,
  `metadata: dict={}`.
- `class RecommendationEnvelope(BaseModel)`: `recommendations: list[Recommendation]
  = []`, `summary: str | None=None`. (This is the schema example 03 + guides use.)
`services.py`:
- `class ContextAppService:` wraps ContextService + StorageService. Methods:
  `upsert_fields(workspace_id, subject_id, fields: list[ContextField])`,
  `list_fields(subject_id)`, `build_snapshot(...)`, `get_snapshot(snapshot_id)`.
- `class RunAppService:` `__init__(self, *, runtime, storage, entity_registry=None)`.
  `async def create_run(self, *, workspace_id, subject_id, persona, task,
   idempotency_key=None, llm_config=None) -> RunRecord` — idempotent via
   load_run_by_idempotency; saves QUEUED RunRecord; launches background
   `_execute_run` via `asyncio.create_task`; returns immediately.
  `_execute_run` sets RUNNING -> runtime.run_task -> SUCCEEDED/ FAILED, stores
  output_text/output_structured, extracts recommendations if output matches
  RecommendationEnvelope (delegates to RecommendationAppService.extract).
  `get_run(run_id)`, `list_runs(...)`, `get_run_events(run_id)`,
  `get_run_thread(run_id)`. Provide an `await_run(run_id, timeout)` test helper that
  polls until terminal (handy for examples/tests since execution is background).
- `class RecommendationAppService:` `extract_from_run(run) -> list[RecommendationItem]`
  (parse output_structured as RecommendationEnvelope; create items),
  `list(...)`, `update_status(recommendation_id, *, status, notes=None)`.
- `class DashboardQueryService:` `async def summary(self, *, subject_id,
  workspace_id) -> dict` — context field count + avg confidence + freshness; run
  counts by status; recommendation counts by status.
`__init__.py` re-exports models + services.

### 7.3 ORCHESTRATORS — `opensims/orchestrators/workflow.py`
- `class WorkflowStep(BaseModel)`: `name: str`, `subtask: str`,
  `tool_names: list[str] = []`, `response_format: ResponseFormat | None=None`,
  `output_json_schema: dict | None=None`, `output_key: str | None=None`,
  `sequential_tools: bool=False`, `temperature: float | None=None`,
  `max_tokens: int | None=None`, `metadata: dict={}`. `to_record()`
  kind="workflow_step" keyed by name.
- `class Workflow(BaseModel)`: `workflow_id: str = factory`, `name: str`,
  `description: str=""`, `steps: list[WorkflowStep] = []`, `metadata: dict={}`.
  `to_record()` kind="workflow" keyed by workflow_id.
- `class WorkflowStepResult(BaseModel)`: `step_name: str`, `status: str`
  (completed|failed), `output_text: str | None=None`, `output_structured: Any=None`,
  `tool_calls: list=[]`, `tool_returns: list=[]`, `error: str | None=None`,
  `metadata: dict={}`.
- `class WorkflowResult(BaseModel)`: `workflow_name: str`, `status: str`
  (completed|failed|partial), `final_state: dict={}`, `thread_id: str | None=None`,
  `step_results: list[WorkflowStepResult] = []`.
- `class WorkflowOrchestrator:` `__init__(self, runtime, *, default_temperature=0.1)`.
  `async def run(self, workflow, persona, *, initial_state=None,
   stop_on_step_failure=True) -> WorkflowResult`. For each step:
   `subtask.format(**state)` (missing key -> failed step with the documented error
   message, not KeyError), build Task with tool_names/output_schema/response_format,
   reuse ONE shared ChatThread across steps. If `sequential_tools and
   len(tool_names) >= 2`: drive forced one-tool-per-call via WorkflowState loop
   (LLMConfig response_format=WORKFLOW), aggregating tool_calls/returns. Else single
   run_task. After step, if output_key set: `state[output_key] = output_structured
   ?? output_text`. Respect stop_on_step_failure.
- `__init__.py` re-exports.

### 7.4 API — `opensims/api/`
`deps.py`:
- `class ApiContainer:` holds storage, entity_registry, inference, runtime,
  context_app, run_app, recommendation_app, dashboard. `@classmethod
  build_default(cls) -> ApiContainer` — in-memory storage + EntityRegistry +
  InferenceService(StubClientManager()) (or GatewayClientManager when
  PYDANTIC_AI_GATEWAY_API_KEY set) + SingleAgentRuntime + app services.
`app.py`:
- `def create_app(container: ApiContainer | None = None) -> FastAPI`. Calls
  configure_observability(); instrument_fastapi(app). Optional internal-key guard
  dependency when `OPENSIMS_INTERNAL_API_KEY` set (header from
  `OPENSIMS_INTERNAL_HEADER`, default `x-opensims-internal-key`). Mount routers.
  Store container on app.state.
`routers/context.py`: POST `/v1/context/fields:upsert`, GET `/v1/context/fields`,
  POST `/v1/context/snapshots:build`, GET `/v1/context/snapshots/{snapshot_id}`.
`routers/runs.py`: POST `/v1/runs` (body: workspace_id, subject_id, persona,
  task/prompt+title, idempotency_key; returns run record; background exec),
  GET `/v1/runs/{run_id}`, GET `/v1/runs`, GET `/v1/runs/{run_id}/events`,
  GET `/v1/runs/{run_id}/thread`.
`routers/recommendations.py`: GET `/v1/recommendations` (filters), PATCH
  `/v1/recommendations/{recommendation_id}` (status/notes).
`routers/dashboard.py`: GET `/v1/dashboard/summary?subject_id=&workspace_id=`.
Use pydantic request/response models. Background runs are fine (in-memory). Make the
`POST /v1/runs` accept a simplified persona (name/description/instructions) + task
(title/prompt/context) so the API is usable from curl.
`__init__.py` re-exports create_app, ApiContainer.

---

## 8. EXAMPLES — `examples/`
`_runtime.py`:
- `def build_runtime(**overrides) -> tuple[SingleAgentRuntime, InferenceService, ToolService]`.
  Chooses client manager: if `OPENPSIMS`... actually: if `pydantic_ai` importable AND
  a provider key present -> PydanticAIClientManager via `OPENSIMS_EXAMPLES_MODEL`
  env; else StubClientManager (DEFAULT, offline). Wire in-memory storage,
  EntityRegistry, ContextService, PromptManager, ContextPromptMapper, ToolService.
  Also `build_storage()`, `build_inference()` helpers as needed. Calls
  configure_observability().
- Each example is runnable with `python examples/0N_*.py`, prints readable output,
  and WORKS OFFLINE with the stub. Use `asyncio.run(main())`.
- `01_basic_chat.py`: persona + task -> print thread.last_message.content + latency/
  model metadata.
- `02_tool_calling.py`: register two local tools, task with tool_names, show TOOL
  rows appear on the thread.
- `03_structured_output.py`: LLMConfig STRUCTURED_OUTPUT with RecommendationEnvelope
  -> parse + print recommendations.
- `04_orchestration_team.py`: 3 specialists via run_batch + manager synthesis.
- `05_workflow.py`: WorkflowOrchestrator with a forced sequential-tools step + a
  structured report step (count_words/score_sentiment/extract_topics tools).
- `06_postgres_storage.py`: gated — if `OPENSIMS_TEST_POSTGRES_DSN` unset, print a
  skip notice and exit 0. Otherwise wire PostgresStorageService.

## 9. TESTS — `tests/` (pytest + pytest-asyncio style via asyncio.run or anyio)
- Use plain `def test_*` that call `asyncio.run(...)` OR add `pytest.ini`/markers.
  Prefer `asyncio.run` inside sync tests to avoid needing pytest-asyncio (NOT
  installed). conftest.py provides fixtures (build a runtime, storage, etc.).
- Cover, at minimum, the happy path of every kernel + runtime + orchestrator + api
  (use `fastapi.testclient.TestClient`) + application (idempotency, recommendation
  extraction) + entities (versioning, lineage) + observability (no-op when off).
- ALL tests MUST pass offline with the stub. Target: green `pytest -q`.

## 10. PACKAGING
`pyproject.toml` (PEP 621, hatchling or setuptools build backend):
- name "opensims", version "0.1.0", requires-python ">=3.12".
- dependencies: `pydantic>=2`, `pyyaml>=6`, `httpx>=0.27`.
- optional-dependencies:
  - `api = ["fastapi>=0.110", "uvicorn>=0.29"]`
  - `providers = ["pydantic-ai>=0.0.1"]`
  - `postgres = ["asyncpg>=0.29"]`
  - `knowledge = ["pgvector>=0.2", "openai>=1.0", "pypdf>=4.0"]`
  - `observability = ["logfire>=0.30"]`
  - `dev = ["pytest>=8", "pytest-asyncio>=0.23"]`
  - `all = [...]`
- `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` IF using pytest-asyncio;
  but since dev deps may be absent, tests should not REQUIRE pytest-asyncio — use
  asyncio.run. Keep `testpaths = ["tests"]`.
NOTE: fastapi IS installed here, so `[api]` can be assumed available for tests.
`.env.example`: OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENSIMS_EXAMPLES_MODEL,
PYDANTIC_AI_GATEWAY_API_KEY, OPENSIMS_GATEWAY_REGION, OPENSIMS_INTERNAL_API_KEY,
OPENSIMS_LOGFIRE_ENABLED, OPENSIMS_TEST_POSTGRES_DSN, OPENAI_COMPATIBLE_* — all
commented with guidance.
`.gitignore`: venv, __pycache__, .env, *.egg-info, .pytest_cache, dist/build.
`docker/docker-compose.yml`: pgvector/pgvector:pg16 on host port 5433, volume
opensims_pgdata, db/user/pass opensims.
`README.md`: what it is, install, quickstart (offline stub), architecture overview
(link kernels), running examples, running tests, optional extras, project layout.

## 11. CONVENTIONS RECAP
- `from __future__ import annotations` at top of every module.
- Public surface re-exported via each package `__init__.py`.
- Docstrings on public classes/functions; module docstring naming the kernel.
- No `datetime.now`/`random` inside example/test top-level that would break
  determinism unnecessarily (helpers in lib are fine).
- Every file must import cleanly with ONLY pydantic+pyyaml+httpx+fastapi present.
- Prefer composition + protocols over inheritance; keep services stateless except
  for in-memory stores.
