# Observability

> The run-event model (`RunEvent` + `EventType` + `EventSink`) and the sinks it flows to — a durable SQLite trace store plus opt-in, off-by-default Logfire/OpenTelemetry wiring — alongside two operational signals on the FastAPI app: a `GET /metrics` Prometheus endpoint and an optional `HIMMY_LOG_FORMAT=json` structured-log toggle.

## Three signals at a glance

himmy exposes three independent observability signals. Be precise about what is
on by default:

| Signal | Surface | Default | Toggle |
| --- | --- | --- | --- |
| **Run-event tracing** | `RunEvent` stream → `.himmy/trace.db` (`himmy trace`) and an optional Logfire/OTel trace tree | trace.db on with `--trace`; Logfire bridge **off** | `himmy run --trace`; `HIMMY_LOGFIRE_ENABLED` for the bridge |
| **Prometheus metrics** | `GET /metrics` text exposition on the FastAPI app | **on** when the API/Studio app is served (in-process, dependency-free) | none — always exposed by the served app |
| **Structured JSON logs** | one JSON object per log line on the root logger | **off** (human-readable output unchanged) | `HIMMY_LOG_FORMAT=json` |

The Prometheus endpoint and JSON logging live in
`himmy/services/observability/metrics.py` and
`himmy/services/observability/logging.py` and are wired onto the app in
`himmy/api/app.py` (via `install_metrics(app)` and `configure_logging()`). Both
are **no-ops on the zero-config/offline path** — collecting a few in-process
counters costs nothing, and JSON logging stays off until the env toggle is set.

## Overview

Observability in himmy is built on one primitive: the `RunEvent`. Every observable
moment in a run's lifecycle — turns, inference, tool calls, handoffs, workflow steps,
memory ops, graph nodes — is emitted as a `RunEvent` to an `EventSink`. Sinks are
interchangeable: the in-memory/Postgres `StorageService` is a sink, a stdlib-SQLite
trace store (`SqliteEventStore`) is a sink, and an optional Logfire bridge mirrors the
same stream into a real OpenTelemetry trace tree.

The event model lives in **`himmy/core/events.py`** (kernel-level, so every layer can
emit). The `himmy.services.observability` package holds the rendering/trace store
(`trace.py`) and the Logfire bridge (`__init__.py`).

**Logfire/OTel wiring is real but thin and opt-in.** It is a complete no-op (and never
imports `logfire`) unless `HIMMY_LOGFIRE_ENABLED` is truthy. The persistent,
always-available trace surface is the SQLite store + the `format_timeline` renderer
used by the `himmy trace` CLI.

## Module map

| File | Responsibility |
| --- | --- |
| `../../himmy/core/events.py` | `EventType` (the closed event enum), `RunEvent` (the event model, with `to_record()` spine projection), `EventSink` protocol |
| `observability/trace.py` | `format_timeline` (human-readable indented timeline + cost/token footer), `SqliteEventStore` (durable stdlib-sqlite3 event log / trace.db) |
| `observability/__init__.py` | Opt-in Logfire bridge: `configure_observability`, `emit_event_span`, `instrument_fastapi`, `instrument_asyncpg`, `reset_spans` |

## Key abstractions

### `EventType` (`core/events.py`)

A closed `str` enum of every run-lifecycle event emitted across the framework. The set
spans:

- Run/turn: `AGENT_RUN_STARTED/FINISHED`, `AGENT_TURN_STARTED/COMPLETED`.
- Approvals: `APPROVAL_REQUIRED/GRANTED/REJECTED`.
- Inference: `INFERENCE_REQUESTED/SUCCEEDED/FAILED`.
- Tools: `TOOL_CALLED/COMPLETED/FAILED`.
- Context/guardrails: `CONTEXT_SNAPSHOT_BUILT`, `CONTEXT_COMPACTED`, `GUARDRAIL_APPLIED`.
- Workflow: `WORKFLOW_STARTED/STEP_COMPLETED/FINISHED`.
- Multi-agent: `AGENT_HANDOFF`, `AGENT_DELEGATED`, the `GROUP_*` and `FANOUT_*` families.
- Memory: `MEMORY_REMEMBERED/RECALLED/CONSOLIDATED`.
- Graphs: `GRAPH_STARTED/NODE_STARTED/NODE_COMPLETED/EDGE_TAKEN/CHECKPOINTED/FINISHED`.
- Typed output: `TYPED_OUTPUT_VALIDATED/REPAIRED`.

### `RunEvent` (`core/events.py`)

A single observable event:

```python
event_id: str           # random uuid
event_type: EventType
trace_id / thread_id / agent_id / request_id / tool_call_id: str | None
latency_ms / cost: float | None
payload: dict
error: str | None
timestamp: str          # ISO-8601
```

`RunEvent.to_record()` projects the event onto the entity spine as kind `run_event`
(lazy import to avoid a `core <-> entities` cycle) — so events are first-class lineage
nodes a run can be replayed and audited from.

### `EventSink` protocol (`core/events.py`)

A minimal `runtime_checkable` Protocol — anything that can durably accept events:

```python
async def append_event(self, event: RunEvent) -> None: ...
```

## How it works / data flow

1. **Emission.** Runtime, orchestrators, and services build a `RunEvent` and `await
   sink.append_event(event)`. The inference service, for example, emits
   `INFERENCE_REQUESTED` then `INFERENCE_SUCCEEDED`/`INFERENCE_FAILED` per call
   (always best-effort: emission is wrapped so observability never breaks the run).
2. **Sinks.** The sink is injected. Implementations of `EventSink`:
   - `StorageService` / `PostgresStorageService` (`append_event` + `list_events`) — the
     canonical audit stream. The Postgres `run_events` table backs the `ai_call_log`
     view (one row per LLM call, joining the request/response pair by `request_id`).
   - `SqliteEventStore` (`trace.py`) — a durable stdlib-`sqlite3` event log usable as a
     runtime sink. `append_event` is `async` (to satisfy the protocol); `list_events`
     and `recent_threads` read it back. This is the **`.himmy` trace.db** the CLI uses
     to inspect a *past* run, not just one streaming live. (The CLI passes a file path;
     `:memory:` is the default for transient use.)
3. **Rendering.** `format_timeline(events)` sorts events by timestamp and renders an
   ordered, indented timeline (icons + indent depth per event type, tool/handoff/turn
   details, per-event `(Nms)` timing) with a footer summarizing event count, tool-call
   count, and total cost.
4. **Logfire bridge (opt-in).** When enabled, `emit_event_span(event)` mirrors the same
   stream into a Logfire/OTel trace *tree* (see below).

### Logfire / OpenTelemetry (opt-in, off by default)

`observability/__init__.py` is a full no-op — and never imports `logfire` — unless
`HIMMY_LOGFIRE_ENABLED` is truthy (`1/true/yes/on`). The only hard error is explicit
misconfiguration: the switch is on but the `logfire` package is missing
(`RuntimeError`).

- `configure_observability()` — idempotent; when enabled, calls `logfire.configure(...)`
  and `logfire.instrument_pydantic_ai()`. Service name from
  `HIMMY_LOGFIRE_SERVICE_NAME` (default `himmy`).
- `emit_event_span(event)` — builds a real span *tree* rather than flat logs:
  `AGENT_RUN_STARTED` / `TOOL_CALLED` / `WORKFLOW_STARTED` **open** a context-managed
  `logfire.span` recorded by a correlation key; the paired `*_FINISHED` / `TOOL_COMPLETED`
  / `TOOL_FAILED` / `WORKFLOW_FINISHED` event **closes** it (so tool spans nest under
  the run span and carry real timing). Other events become point-in-time logs on the
  active span. Never raises.
- `instrument_fastapi(app)` / `instrument_asyncpg()` — soft-degrade with a one-line
  warning if the corresponding `logfire[...]` transport is missing; full no-op when off.
- `reset_spans()` — test/teardown helper that closes any open spans.

**PII safety:** "content" payload keys (`rendered_prompt`, `prompt`, `completion`,
`output`/`output_text`/`output_structured`, `content`, `tool_args`, `messages`,
`retrieval_ctx`) are dropped from span attributes unless
`HIMMY_LOGFIRE_INCLUDE_CONTENT` is truthy.

### Prometheus metrics — `GET /metrics` (`metrics.py`)

`himmy/services/observability/metrics.py` exposes the served FastAPI app
(`himmy serve` / `himmy studio`) in the **Prometheus text exposition format** at
`GET /metrics`. `install_metrics(app)` (called from `himmy/api/app.py`) adds a
lightweight ASGI middleware plus the endpoint. It is **dependency-free** — the
`Counter`/`Histogram`/`Gauge` primitives are hand-rolled and accumulate in
plain in-process dicts, so there is **no new hard dependency and no network**;
if `prometheus_client` (from the `observability` extra) happens to be installed,
nothing changes, since himmy keeps its own in-process registry.

Instruments (process-wide, cumulative across requests, as a scrape target
expects):

| Metric | Type | Labels |
| --- | --- | --- |
| `http_requests_total` | counter | `method`, `route`, `status` |
| `http_request_duration_seconds` | histogram | `method`, `route` |
| `http_requests_in_flight` | gauge | (none) |

**Cardinality is bounded by construction:** `method` is clamped to a fixed verb
allow-list (anything else → `OTHER`); `route` is the matched **route template**
(`/v1/runs/{run_id}`, never the filled-in path — unmatched paths collapse to
`<unmatched>`); `status` is the status *class* (`2xx`/`4xx`/…). No secret,
header, query string, or raw path parameter is ever used as a label. The
endpoint is excluded from the OpenAPI schema and exposes no secrets. The
middleware is registered **after** request-context so it is the outermost layer
and observes every request (including ones short-circuited by inner guards).

### Structured JSON logs — `HIMMY_LOG_FORMAT=json` (`logging.py`)

`himmy/services/observability/logging.py` adds an **opt-in** JSON log format,
wired via `configure_logging()` in `himmy/api/app.py`. Default behavior is
unchanged: with `HIMMY_LOG_FORMAT` unset (or anything other than `json`), this
installs nothing and the existing human-readable output is preserved
byte-for-byte. When `HIMMY_LOG_FORMAT=json`, the root logger's handlers are
switched to `JsonLogFormatter`, which emits **one JSON object per line** — no new
dependency, friendly to log shippers (Loki / CloudWatch / ELK).

Each line carries at least `timestamp` (ISO-8601, UTC), `level`, `logger`, and
`message`. When a request/trace id is bound to the current context (set by the
API's request-context middleware), it is included as `request_id` — the **same
id echoed to clients in the `X-Request-ID` response header** — so a log line
correlates to a request without inventing a new mechanism. `exc_info`/`stack_info`
become `exception`/`stack`, and JSON-serializable `extra=` fields are surfaced
without clobbering the core keys. `configure_logging()` is idempotent and a no-op
when the toggle is unset.

## Configuration

| Mechanism | Effect |
| --- | --- |
| Default (no env, no Logfire) | Events flow to whatever `EventSink` is injected (storage / SQLite trace); Logfire bridge is a no-op |
| `HIMMY_LOGFIRE_ENABLED` | Master switch for the Logfire/OTel bridge |
| `HIMMY_LOGFIRE_SERVICE_NAME` | OTel service name (default `himmy`) |
| `HIMMY_LOGFIRE_INCLUDE_CONTENT` | Allow prompt/completion content onto spans (default off) |
| `SqliteEventStore(path)` | Durable trace.db for `himmy trace` |
| `GET /metrics` (served app) | Prometheus text exposition — always on for `himmy serve`/`himmy studio`, in-process and dependency-free |
| `HIMMY_LOG_FORMAT=json` | Switch root-logger output to one-JSON-object-per-line structured logs (default: human-readable) |
| `[observability]` extra | Installs `logfire` (and its `[fastapi]`/`[asyncpg]` transports) for the OTel bridge — **not** required for `/metrics` or JSON logs |

## Extension points

- **New sink**: implement `EventSink.append_event` (e.g. ship to an external bus).
  Storage and the SQLite trace store are the reference implementations.
- **Span mapping**: extend `_SPAN_OPEN` / `_SPAN_CLOSE` in `observability/__init__.py`
  to make additional event pairs render as nested spans.
- **Timeline rendering**: add entries to `_RENDER` in `trace.py` for new event types.
- **Spine audit**: rely on `RunEvent.to_record()` to put events on the entity registry.

## Gotchas & invariants

- Event emission is **best-effort** — sinks are wrapped so a telemetry failure never
  breaks inference or a run.
- The Logfire layer is genuinely **thin and opt-in**: with `HIMMY_LOGFIRE_ENABLED`
  unset, `logfire` is never imported. If you need always-on tracing, use the
  `SqliteEventStore` sink — that's the persistent surface.
- `SqliteEventStore.append_event` is `async` only to satisfy the protocol; the read
  methods (`list_events`, `recent_threads`, `close`) are sync.
- Span content keys are stripped by default (PII safety) — turn on
  `HIMMY_LOGFIRE_INCLUDE_CONTENT` explicitly if you need prompt/output text in traces.
- `EventType` is a **closed** enum — adding an event type means adding to the enum (and,
  if it should nest, to the span open/close maps and the timeline renderer).

## Related docs

- [Storage Service](storage.md) — `StorageService`/`PostgresStorageService` are `EventSink`s; the `ai_call_log` view flattens the inference event pair.
- [Inference Service](inference.md) — emits `INFERENCE_REQUESTED/SUCCEEDED/FAILED`.
- [Memory Service](memory.md) — emits `MEMORY_REMEMBERED/RECALLED/CONSOLIDATED`.
- [Entity Registry](../architecture/entities.md) — `RunEvent.to_record()` projects events onto the spine.
