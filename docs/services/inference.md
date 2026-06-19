# Inference Service

> The single async entry point for every model call: provider-agnostic typed envelopes, retries, caching, streaming, cost, and pluggable backends (offline stub by default).

## Overview

`himmy.services.inference` is the inference kernel. Everything in the framework
that talks to a model goes through one class — `InferenceService` (`service.py`) —
which wraps a `ClientManager` and adds the things production needs on top of a raw
model call: a per-attempt timeout ceiling, bounded retries on retryable codes only,
ordered batching with bounded concurrency, an optional response cache, a streaming
entry point, and run-lifecycle event emission.

The design is **local-first and offline-by-default**: the default backend is the
deterministic `StubClientManager`, which fully simulates every response format
(text / JSON / structured output / tool calls / workflow) with no network and no
keys. pydantic-ai, the gateway, Ollama, the Claude CLI, and OpenRouter are all
optional and live behind lazy imports.

A hard service-level contract runs through the whole module: `InferenceService.run`
**never raises** for provider/manager errors. Any exception a manager throws is
normalized into a `FAILED` `InferenceResponse` carrying a typed `InferenceError`, so
retries, latency stamping, and the `INFERENCE_FAILED` event always fire and one bad
request can never kill a batch.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `InferenceService` (the entry point), `StreamDelta`, retry/timeout/cache/batch logic, exception normalization |
| `models.py` | Typed envelopes: `InferenceRequest`/`InferenceResponse`, `InferenceMessage`, `ResponseFormat`, `InferenceError`/`InferenceErrorCode`, `BoundTool`, `ToolExecutor`, `ToolCallRecord`/`ToolReturnRecord`, `WorkflowDefinition`/`WorkflowState`, `LLMConfig`, `ModelPrice`, `GatewayRuntimeConfig`, `synthesize_from_schema` |
| `client_manager.py` | The `ClientManager` protocol, `StubClientManager` (offline default), `GatewayClientManager`, and `normalize_forced_tool` |
| `local.py` | Local/self-hosted managers: `OllamaClientManager`, `ClaudeCliClientManager`, `HimalayaGptClientManager` |
| `pydantic_ai_manager.py` | `PydanticAIClientManager` — the real provider path (pydantic-ai Agent), optional `[providers]` extra |
| `routing.py` | `RoutingClientManager` + `Route` + `DEFAULT_FAILOVER_CODES` — ordered failover across managers |
| `multi_provider.py` | `MultiProviderClientManager` — dispatch each `model_key` to its own backend |
| `cache.py` | `InferenceCache` protocol, `NoopInferenceCache` (default), `InMemoryTTLCache`, `compute_cache_key` |
| `replay.py` | `RecordingClientManager` / `ReplayClientManager` + `InferenceCassette` — record-and-replay cassettes |
| `pricing.py` | Layered USD price table (bundled snapshot ← LiteLLM sync file ← env override), `price_for`, `sync_prices` |
| `tool_protocol.py` | `parse_text_tool_calls` — tolerant parsing of tool calls from a text-only model reply |
| `__init__.py` | Public re-exports |

## Key abstractions

### `InferenceService` (`service.py`)

The single async surface. Constructed with a `ClientManager` plus policy knobs:

```python
InferenceService(
    client_manager,
    *,
    max_retries=2,
    default_timeout_seconds=90.0,
    retry_base_delay_seconds=0.2,
    retry_jitter_seconds=0.1,
    event_sink=None,
    timeout_grace_seconds=0.25,
    timeout_grace_factor=1.05,
    cache=None,
)
```

Methods:

- `run(request) -> InferenceResponse` — single call with timeout ceiling + bounded retries.
- `run_batch(batch) -> BatchInferenceResponse` — bounded-concurrency, order-preserving, failure-isolated.
- `run_stream(request, *, chunk_size=24) -> AsyncIterator[StreamDelta]` — streaming (see below).

The outer hard ceiling is **proportional**: `ceiling = timeout * timeout_grace_factor
+ timeout_grace_seconds`, so a sub-second per-request timeout is not dominated by a
fixed pad.

### `ClientManager` protocol (`client_manager.py`)

A `runtime_checkable` Protocol with two members:

```python
def resolve(self, model_key: str) -> str: ...
async def generate(self, request: InferenceRequest) -> InferenceResponse: ...
```

Generation is part of the protocol (not just resolution) precisely so the stub can
fully simulate provider behavior offline. Implementations *should* self-normalize
(return a `FAILED` response rather than raise), but `InferenceService` wraps every
`generate` in a normalization boundary regardless. A manager may optionally implement
`generate_stream` (an async generator yielding `str` deltas then a final
`InferenceResponse`) to back real streaming.

### The request/response envelopes (`models.py`)

- `InferenceRequest` carries `model_key`, `messages` (list of `InferenceMessage`),
  `response_format`, `output_json_schema`, `workflow`, `generation_params`,
  `timeout_seconds`, `bound_tools`, `tool_executor`, `tool_names_override`,
  `toolsets`, `metadata`, and a random `request_id`. A model validator derives
  `response_format` when omitted (workflow → `WORKFLOW`, schema → `STRUCTURED_OUTPUT`)
  and rejects contradictions at construction time.
- `InferenceResponse` carries `status`, `output_text`, `output_structured`,
  `tool_calls`/`tool_returns`, `workflow`, token counts, `cost`, `latency_ms`,
  `model_path`, `provider_name`, `error`, and free-form `metadata`.
- `ResponseFormat`: `TEXT`, `JSON_OBJECT`, `STRUCTURED_OUTPUT`, `AUTO_TOOLS`,
  `WORKFLOW`, `TOOL`.
- `InferenceErrorCode`: `AUTH`, `QUOTA`, `RATE_LIMITED`, `PROVIDER_UNAVAILABLE`,
  `INVALID_REQUEST`, `TIMEOUT`, `UNKNOWN`. Only `RATE_LIMITED`/`TIMEOUT`/
  `PROVIDER_UNAVAILABLE` are in `RETRYABLE_ERROR_CODES`.
- `LLMConfig` is a typed alternative to stuffing knobs into `generation_params`
  (temperature/max_tokens/top_p/use_cache/etc.); the runtime maps each field onto the
  request.

### The tool seam: `BoundTool` + `ToolExecutor` (`models.py`)

This is the load-bearing decoupling between the inference layer and the tool layer:

- `BoundTool` is **pure data** — `name`, `description`, `args_json_schema`,
  `output_json_schema`, `read_only`. It carries no Python callable, so the inference
  layer can advertise a tool's schema to the model and synthesize arguments without
  holding tool-layer code.
- `ToolExecutor = Callable[[str, dict], Awaitable[ToolReturnRecord]]` — a single
  execution callback set on the request alongside `bound_tools`. A manager that emits
  tool calls (stub, Ollama, Claude CLI, pydantic-ai) invokes execution through this
  one explicit seam and records the resulting `ToolCallRecord`/`ToolReturnRecord`
  pairs. `None` means no execution capability (managers then synthesize stub returns).

> Note: despite its name, `tool_protocol.py` is **not** where `BoundTool`/
> `ToolExecutor` live — those are in `models.py`. `tool_protocol.py` is the tolerant
> *text* tool-call parser (`parse_text_tool_calls`) used by the text-protocol
> backends (Claude CLI, prose-answering Ollama models).

### Forced-tool normalization (`normalize_forced_tool`)

`ResponseFormat.TOOL` (force one named tool) is rewritten uniformly for *every*
provider: the service turns it into an `AUTO_TOOLS` call that binds only the forced
tool and appends a system nudge instructing the model to call it. The forced tool is
`tool_names_override[0]` if set, else the first bound tool; a `TOOL` request with no
bound tools is a caller error (`HimmyError` → `INVALID_REQUEST`). This runs in
`InferenceService._run_once` before dispatch, so no manager special-cases it.

## How it works / data flow

End-to-end of a single `await service.run(request)`:

1. Emit `INFERENCE_REQUESTED` to the event sink (best-effort).
2. **Cache lookup** (only if `generation_params["use_cache"]`): compute
   `cache.key_for(request)` and try `cache.get`. A hit is deep-copied, re-stamped
   with this `request_id` and latency, marked `metadata["cache_hit"]=True`, emits
   `INFERENCE_SUCCEEDED` (cost 0), and returns immediately.
3. Compute the timeout ceiling and enter the retry loop (`max_retries + 1` attempts):
   - `_run_once` calls `normalize_forced_tool(request)` then `client_manager.generate`
     inside an `asyncio.wait_for(ceiling)`. `NotImplementedError` and `HimmyError`
     map to non-retryable `INVALID_REQUEST`; any other exception is normalized via
     `_normalize_exception` (a small class-name map turns transport-ish errors into
     retryable transient codes; everything else is non-retryable `UNKNOWN`).
   - On `SUCCESS`: stamp latency, write to cache if opted in, emit
     `INFERENCE_SUCCEEDED`, return.
   - On `FAILED`: retry only if attempts remain **and** `error.retryable` **and**
     `error.code in RETRYABLE_ERROR_CODES`, with exponential backoff + jitter.
     Otherwise break.
4. On exhaustion: stamp latency, emit `INFERENCE_FAILED`, return the last `FAILED`
   response.

`run_batch` runs each request through `run` under an `asyncio.Semaphore`
(`max_concurrency`), preserving input order; a belt-and-suspenders guard converts any
escaped exception to `FAILED` so one request can never abort the batch.

`run_stream` prefers the manager's `generate_stream` when present (real provider
deltas via pydantic-ai's `agent.run_stream`); otherwise it falls back to buffering via
`run()` and chunking the `output_text` deterministically. Either way the terminal
`StreamDelta` has `done=True` and carries the fully materialized `InferenceResponse`.

### Providers / client managers

| Manager | Backend | Notes |
| --- | --- | --- |
| `StubClientManager` | none (offline) | Framework default. Deterministic; simulates every `ResponseFormat`; executes bound tools via the executor; free ($0). |
| `PydanticAIClientManager` | pydantic-ai `Agent` | Real provider path. System msg → `instructions`; history → `message_history`; bound tools → `Tool.from_schema`; usage → tokens; price table → cost. WORKFLOW via `agent.iter()` breaking after the first `CallToolsNode`; streaming via `agent.run_stream`. Needs the `[providers]` extra. |
| `GatewayClientManager` | Pydantic AI Gateway / OpenAI-compatible | Resolves keys against a `GatewayRuntimeConfig` registry, builds a gateway-configured `PydanticAIClientManager` delegate. Needs `PYDANTIC_AI_GATEWAY_API_KEY` + the extra; can fall back to the stub with `HIMMY_GATEWAY_STUB_FALLBACK`, else raises a clear error. |
| `OllamaClientManager` | local Ollama `/api/chat` | Native function-tool schema; falls back to text tool-call parsing for prose replies; native structured output via `format`. |
| `ClaudeCliClientManager` | local `claude` CLI (Claude Max) | Subprocess, not HTTP. `-p --model X --output-format json`; disables the CLI's own builtin tools; drives tools via a ReAct text protocol; floors timeout at 150s; reads real token usage + `total_cost_usd`. |
| `HimalayaGptClientManager` | self-hosted HF Transformers | Blocking inference on a worker thread (`asyncio.to_thread`). |

CLI selection lives in `himmy/cli/provider.py` (`build_manager_for` /
`build_inference_for`). Supported `--provider` values: `stub`, `claude-cli`,
`ollama`, `pydantic-ai`, `openrouter` (OpenRouter routes through the pydantic-ai path
with `OPENROUTER_BASE_URL`). `provider=None` delegates to
`himmy.runtime.builder.build_inference` (pydantic-ai when a key + the extra + a model
are present, else the offline stub).

### Composition managers

- `RoutingClientManager` (`routing.py`) — *is* a `ClientManager`, so it composes over
  any of the above. Holds an ordered list of `Route`s and, on a failover-eligible
  failure (`DEFAULT_FAILOVER_CODES` = `PROVIDER_UNAVAILABLE`/`RATE_LIMITED`/`TIMEOUT`/
  `QUOTA`/`AUTH`; `INVALID_REQUEST`/`UNKNOWN` are excluded), re-dispatches to the next
  route. Stamps `metadata["fallback_chain"]`. `cost_ordered` tries cheapest routes
  first (local/free, then cloud). Intentionally has no `generate_stream` — streaming
  falls back to buffer-and-chunk while keeping the failover guarantee.
- `MultiProviderClientManager` (`multi_provider.py`) — maps each `model_key` to its
  own manager and delegates (resetting `model_key` to `"default"` so each backend uses
  its own model). Unknown keys hit a designated default. Lets one `InferenceService`
  fan out across heterogeneous backends (e.g. a Claude-CLI "brain" orchestrating cheap
  Ollama workers).

### Caching (`cache.py`)

`InferenceCache` is a Protocol (`key_for` / `get` / `set`). `compute_cache_key`
hashes only the caching-relevant surface — `model_key`, `route_override`,
`response_format`, `output_json_schema`, `workflow`, the messages, the bound tool
names, and the non-`use_cache` generation params — into a SHA-256. The random
`request_id` and `use_cache` itself are excluded so logically-identical calls collide.
`NoopInferenceCache` is the default (never stores). `InMemoryTTLCache` is a
process-local, TTL-bounded cache that stores only `SUCCESS` responses and evicts the
soonest-to-expire entry past `max_entries`.

### Record / replay (`replay.py`)

- `RecordingClientManager` wraps a real manager and appends every
  `(compute_cache_key(request), response)` to an ordered `InferenceCassette`
  (recorded responses already carry tool call/return pairs, so tool *outputs* are
  captured). `dump(path)` writes JSON.
- `ReplayClientManager` answers each request from the cassette by content hash (FIFO
  per key, so retries/duplicates replay in order). Tools are **not** re-executed — the
  recorded response is returned verbatim — so replays are side-effect-free and
  deterministic. A missing key raises `ReplayError` unless a `fallback` manager is
  wired. Matching by content hash (which excludes `request_id`) means a re-run replays
  exactly even though every `request_id` differs.

### Pricing / cost (`pricing.py`)

Cost = tokens × per-model price, resolved from a layered table:
`explicit override > synced file > bundled snapshot > unpriced ($0)`. The synced file
(`~/.himmy/model_prices.json`, written by `himmy prices sync` from the LiteLLM
community price JSON) keeps prices current without upgrading himmy; `HIMMY_MODEL_PRICES`
points at a custom file; `prices.json` is a small dated offline fallback. Lookup is
forgiving — a `provider:` prefix and `-YYYY-MM-DD`/`-latest` suffix are stripped on a
miss. `PydanticAIClientManager._compute_cost` prefers an explicit
`GatewayRuntimeConfig` price, then falls back to `pricing.price_for`.

## Configuration

| Mechanism | Effect |
| --- | --- |
| Constructor args on `InferenceService` | retries, timeouts, grace factor, cache, event sink |
| `generation_params["use_cache"]` | opt a request into the response cache |
| `PYDANTIC_AI_GATEWAY_API_KEY` | enables the real gateway path in `GatewayClientManager` |
| `HIMMY_GATEWAY_STUB_FALLBACK` | gateway degrades to the offline stub instead of raising |
| `OPENROUTER_API_KEY` | required by the `openrouter` CLI provider |
| `HIMMY_MODEL_PRICES` | custom price-table file path |
| `[providers]` extra | installs pydantic-ai (the gateway / pydantic-ai / openrouter paths) |

## Extension points

- **New backend**: implement the `ClientManager` protocol (`resolve` + `generate`,
  self-normalizing to `FAILED`); optionally add `generate_stream` for real deltas.
- **New cache**: implement `InferenceCache` and pass `cache=` to `InferenceService`.
- **Failover / multi-backend**: wrap managers in `RoutingClientManager` or
  `MultiProviderClientManager` — both *are* `ClientManager`s, so they slot in
  uniformly.
- **Deterministic tests / debugging**: wrap any manager in `RecordingClientManager`,
  dump the cassette, replay with `ReplayClientManager`.

## Gotchas & invariants

- `InferenceService.run` and `run_batch` **never raise** for provider/manager errors —
  always inspect `response.status` / `response.error`.
- Only `RETRYABLE_ERROR_CODES` are retried; `AUTH`/`QUOTA`/`INVALID_REQUEST`/`UNKNOWN`
  bubble straight to `FAILED`. Note `RoutingClientManager`'s failover set is broader
  (it *will* try a different provider on `QUOTA`/`AUTH`).
- Exception → error-code mapping keys on the exception **class name** so provider SDKs
  are never imported on the offline path.
- The cache only stores `SUCCESS` responses; a cache hit reports `cost=0`.
- `pydantic-ai` and `asyncpg` are never imported at module top — only inside
  `generate`/`connect`. The files import cleanly without the extras.
- `PydanticAIClientManager` stamps `metadata["round_trip_complete"]=True`: pydantic-ai
  runs the whole agent loop in one call, so the runtime must not do a continuation turn.
- The stub's token counts are length-based estimates (~4 chars/token), not real
  tokenizer counts.

## Recently added (modules post-dating this doc — 2026-06-08)

The inference kernel grew modules this doc predates; authoritative inventory comes from a periodic pass via `docs/INFERENCE_RECONCILIATION.md`.

- **Direct provider managers** — `anthropic_manager.py` (`AnthropicClientManager`, the `anthropic` SDK Messages API directly) + `openai_manager.py` (`OpenAIClientManager`, the `openai` SDK `chat.completions`) sit *alongside* `pydantic_ai_manager.py`. pydantic-ai is the optional path, not the only one.
- **Local / self-hosted** — `local.py`: `OllamaClientManager`, `ClaudeCliClientManager`, `HimalayaGptClientManager`.
- **Routing & multi-provider** — `routing.py` (`RoutingClientManager`, `DEFAULT_FAILOVER_CODES`, cost-ordered fallback, `fallback_chain` metadata) + `multi_provider.py` (per-`model_key` dispatch).
- **Prompt caching** — `prompt_cache.py`: the universal prefix-cache contract — `CacheCapability` (`NONE`/`ANTHROPIC_EXPLICIT`/`OPENAI_AUTOMATIC`/`OPENROUTER_PASSTHROUGH`), `MIN_CACHEABLE_TOKENS`, `compute_cached_cost`. Distinct from `cache.py` (the tenant-scoped *response* cache).
- **Pricing / replay / compare** — `pricing.py` (model→USD; LiteLLM + OpenRouter sources), `replay.py` (record/replay cassettes), `compare.py` (model catalog + side-by-side).

> Known drift (latest inference-reconciliation report): add Anthropic 4.x rows to `MIN_CACHEABLE_TOKENS` (Haiku-4.5 / Opus-4.5/4.6 = 4096); pin the `pydantic-ai`/`openai`/`anthropic` floors; gate the Anthropic `thinking` payload by model family.

## Related docs

- [Storage Service](storage.md) — where threads, runs, events, and recommendations persist.
- [Observability](observability.md) — the `RunEvent`/`EventSink` the service emits to.
- [Memory Service](memory.md) — uses `InferenceService` for the LLM consolidation path.
- [Entity Registry](../architecture/entities.md) — the append-only spine runs/events project onto.
