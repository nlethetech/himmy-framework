# Storage Service

> Local-first async persistence for threads, run events, runs, recommendations, evaluations, context, and multi-agent/world-model records — in-memory by default, durable Postgres opt-in.

## Overview

`himmy.services.storage` is the persistence kernel. It is split into **focused,
single-responsibility stores** (threads, events, context, runs, recommendations,
evaluations, orchestration) behind a single backward-compatible facade. Two backends
implement the same per-concern protocols:

- `StorageService` (`service.py`, `inmemory.py`) — the **default**: process-local
  dicts, async API, all state volatile and lost on exit. This is what tests, examples,
  and local dev get for free with no setup.
- `PostgresStorageService` (`postgres.py`) — a 1:1 durable mirror over asyncpg
  (requires the `[postgres]` extra and a running database).

The local-first stance is deliberate: durability is **opt-in**. You start in-memory
and graduate to Postgres without changing call sites, because both backends satisfy
the same protocols in `protocols.py`.

> Scope note: the storage layer ships **two** backends — in-memory and Postgres.
> There is no dedicated SQLite *StorageService*. SQLite appears elsewhere in himmy
> (the memory store, the entity registry, and the observability trace store), and
> `himmy/core/sqlite_util.py` is the shared connection-hardening helper those use.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `StorageService` — in-memory facade composing the focused in-memory stores; satisfies `EventSink` + `ThreadEventStore` |
| `inmemory.py` | The focused in-memory stores (`InMemoryThreadStore`, `InMemoryEventLog`, `InMemoryContextStore`, `InMemoryRunStore`, `InMemoryRecommendationStore`, `InMemoryEvaluationStore`, `InMemoryOrchestrationStore`) |
| `postgres.py` | `PostgresStorageService` + the per-concern Postgres stores, `STORAGE_DDL`, `STORAGE_MIGRATIONS`, JSONB/timestamp codecs, the `ai_call_log` view |
| `protocols.py` | The focused store contracts: `ThreadStore`, `EventLog`, `ThreadEventStore`, `ContextStore`, `RunStore`, `RecommendationStore`, `EvaluationStore`, `OrchestrationStore` |
| `models.py` | Persisted record types: `RunRecord`/`RunStatus`, `RecommendationItem`/`RecommendationStatus`, `MemoryObject`, `EpisodicMemoryObject`, `AgentStateRecord`, `ActionRecord`, `EnvironmentStateRecord`, `ContextEvidenceRecord` |
| `encryption.py` | Optional field-level encryption at rest (`FieldEncryptor`, `RecordCipher`, `build_field_encryptor`) |
| `__init__.py` | Public re-exports |
| `../../himmy/core/sqlite_util.py` | `connect_hardened` — WAL + busy-timeout SQLite connection used by the SQLite stores elsewhere |

## Key abstractions

### The facade: `StorageService` (`service.py`)

A thin facade that composes the seven in-memory stores and delegates to them. It
exposes one flat async API covering all concerns (≈the ~15 call sites that inject
storage depend on this single surface). It satisfies both:

- `EventSink` (`append_event`) — so it can be wired as the runtime's event sink, and
- `ThreadEventStore` — the runtime-facing union of `ThreadStore` + `EventLog`.

### The focused protocols (`protocols.py`)

Each concern is a separate `runtime_checkable` Protocol so it can be implemented,
tested, and swapped independently rather than funneled through one god object:

- `ThreadStore` — `save_thread` / `load_thread` (chat threads keyed by `thread_id`).
- `EventLog` — `append_event` / `list_events` (append-only run-event audit stream; also the `EventSink` surface).
- `ThreadEventStore` — runtime-facing union of the two above.
- `ContextStore` — context fields (keyed by `(subject_id, key)`), snapshots, evidence.
- `RunStore` — run records + atomic idempotent creation.
- `RecommendationStore` — recommendation items + status/notes updates.
- `EvaluationStore` — evaluation runs.
- `OrchestrationStore` — multi-agent / world-model records (memory, episodic memory, agent states, actions, environment states).

`ThreadEventStore` (async threads + events, runtime persistence) is intentionally
distinct from `himmy.services.memory.store.MemoryStore` (sync, long-term cognitive
facts) — both were historically named `MemoryStore`; this one was renamed to reflect
what it actually stores.

### The record types (`models.py`)

| Record | Identity | Purpose |
| --- | --- | --- |
| `RunRecord` | `run_id` | The operational unit of work: an async run's lifecycle (`RunStatus` = `QUEUED`/`RUNNING`/`SUCCEEDED`/`FAILED`), output, error, lineage (`workspace_id`, `subject_id`, `task_id`, `thread_id`, `snapshot_id`, `persona_name`, `idempotency_key`, `trace_id`). Storage owns `updated_at`. |
| `RecommendationItem` | `recommendation_id` | A dashboard-facing advisory output extracted from a run (`kind`, `title`, `summary`, `rationale`, `confidence`, `evidence_refs`, `RecommendationStatus` = `PROPOSED`/`ACCEPTED`/`DISMISSED`/`SCHEDULED`). Has `to_record()` → projects onto the entity spine (excluding mutable `status`/`notes` so the content-addressed id stays stable). |
| `MemoryObject` | `memory_id` | A cognitive (long-lived) memory item scoped to a subject/agent. |
| `EpisodicMemoryObject` | `episode_id` | An episodic memory item — a recalled event/interaction trace. |
| `AgentStateRecord` | `state_id` | Snapshot of an agent's internal state at an orchestration round. |
| `ActionRecord` | `action_id` | A single action an agent took within an environment/round. |
| `EnvironmentStateRecord` | `environment_state_id` | Snapshot of a shared environment's state at a round. |
| `ContextEvidenceRecord` | `evidence_id` | A persisted pointer to where a context value originated (an `EvidenceRef` projection). |

All records use ISO-string timestamps (`created_at`, and `updated_at` on `RunRecord`)
and carry an open `metadata: dict`.

> The `MemoryObject`/`EpisodicMemoryObject` here are the *orchestration/world-model*
> memory records persisted by `OrchestrationStore`. They are distinct from the
> long-term cognitive `MemoryRecord` of the [Memory Service](memory.md).

## How it works / data flow

### In-memory backend (default)

Each `InMemory*Store` keeps plain dicts/lists and implements its protocol with simple
filtering. Highlights:

- `InMemoryRunStore` maintains a `(workspace_id, idempotency_key) -> run_id` index so
  `load_run_by_idempotency` is O(1) and `save_run_if_absent_by_idempotency` closes the
  TOCTOU race **with no `await` between read and write** — two concurrent callers with
  the same key cannot both create a run. `save_run` always refreshes `updated_at`.
- `InMemoryEventLog` is append-only; `list_events` filters by `thread_id`/`trace_id`.

### Postgres backend (opt-in)

`PostgresStorageService` mirrors the facade 1:1, delegating to the same set of focused
stores (`PostgresThreadStore`, `PostgresEventLog`, …) which share a `_PgStoreBase`
holding the bound `_require_pool` callable and generic SQL helpers.

- Construct via `await PostgresStorageService.connect(dsn, ...)` (creates an asyncpg
  pool with a JSONB codec + an ISO-string timestamp codec registered on every pooled
  connection, plus acquire/command timeouts). Supports `async with` for deterministic
  teardown; `close()` is idempotent.
- `create_schema()` applies `STORAGE_DDL` (idempotent `CREATE ... IF NOT EXISTS`: 14
  tables + the `ai_call_log` view + `schema_migrations`). `migrate()` runs the ordered
  `STORAGE_MIGRATIONS` forward list exactly once each, tracked in `schema_migrations`,
  inside transactions.
- Threads store their `Message` list inline in a JSONB `payload` (no separate messages
  table). Runs upsert via `INSERT ... ON CONFLICT`. The idempotency guarantee is a
  partial `UNIQUE` index `runs_idempotency_idx (workspace_id, idempotency_key) WHERE
  idempotency_key IS NOT NULL`, and `save_run_if_absent_by_idempotency` uses
  `ON CONFLICT ... DO NOTHING RETURNING` to let exactly one concurrent writer win.
- Timestamps that drive analytics are `TIMESTAMPTZ` with indexes; the model boundary
  stays ISO strings (the codecs translate both ways).
- The `ai_call_log` **view** flattens the `INFERENCE_REQUESTED` /
  `INFERENCE_SUCCEEDED|FAILED` event pair (joined by `request_id`) into one row per LLM
  call (prompt template/version, model, tokens, latency, cost, status). It also exposes
  a pgvector knowledge backend via `knowledge_backend()` / `create_knowledge_schema()`.

### Optional payload encryption (`encryption.py`)

Field-level encryption at rest using an **envelope** scheme: each value is encrypted
with a fresh random AES-GCM data key (DEK), which is itself wrapped by a long-lived
key-encryption key (KEK) — so rotating the KEK never requires re-encrypting data.
AES-GCM is authenticated, and an optional `aad` binds a ciphertext to its context
(record id / tenant). The KEK comes from `HIMMY_ENCRYPTION_KEY` (base64) via the secret
provider; ciphertext is a prefixed URL-safe token (`himmy:enc:v1:`) that round-trips
through JSON/JSONB. `RecordCipher` encrypts/decrypts named fields of a record dict
(idempotent). **Encryption is opt-in** — `build_field_encryptor()` returns `None` when
no key is configured and storage stays plaintext (unchanged). Requires the
`cryptography` package (the `encryption` extra; also pulled in by `auth`).

## Configuration

| Mechanism | Effect |
| --- | --- |
| Default construction (`StorageService()`) | In-memory, volatile (no setup) |
| `PostgresStorageService.connect(dsn, ...)` | Durable Postgres (needs `[postgres]` extra + DB) |
| `create_schema()` / `migrate()` | Apply DDL / run forward migrations |
| `HIMMY_ENCRYPTION_KEY` | Base64 KEK; presence enables field encryption |
| `[postgres]` / `encryption` extras | asyncpg / cryptography |

## Extension points

- **New backend / store**: implement the relevant protocol(s) from `protocols.py`
  (e.g. a Redis-backed `RunStore` replacement) without touching the other concerns.
- **New persisted record**: add a Pydantic model to `models.py` and a method to the
  appropriate store protocol + both backends.
- **At-rest encryption**: wrap field values with `RecordCipher` at the store boundary.
- **Schema evolution (Postgres)**: append a `(version, name, [statements])` entry to
  `STORAGE_MIGRATIONS`.

## Gotchas & invariants

- The default backend is **volatile** — nothing survives process exit unless you wire
  Postgres.
- Storage owns `RunRecord.updated_at`; every write refreshes it, so it can't drift.
- Idempotent run creation: in-memory closes the race via no-`await`-between-read-write;
  Postgres via the partial UNIQUE index + `ON CONFLICT DO NOTHING`.
- `postgres.py` is import-safe without asyncpg or a DB — `asyncpg` is imported lazily
  in `connect`, and `STORAGE_DDL` is a plain string you can inspect offline.
- `RunRecord.output_structured` is JSONB; scalars/None are wrapped as `{"value": ...}`
  on write and unwrapped on read so the bind type is unambiguous.
- `workspace_id` is a first-class column on `runs`/`recommendations` (multi-tenant
  scoping flows through the records).

## Recently added (modules post-dating this doc — 2026-06-08)

- **Leased run work-queue (Q2)** — `run_lane.py`: provider-lane keying for the durable run queue (enqueue → claim → lease/recover → retry). The Postgres claim is an atomic `SELECT … FOR UPDATE SKIP LOCKED` + guarded status update, with a reaper requeuing expired leases — multi-worker-safe, crash-recoverable.
- **Durable inbound dedup (Q4)** — `trigger_dedup.py`: a `trigger_dedup` table (TTL-CAS, mark-after-success) so a redelivered webhook/trigger fires the agent **at-most-once**.
- **Field encryption at rest (WS4.4)** — `at_rest.py`: wires per-field encryption into the SQLite/Postgres durable stores (the durable side of the governance sidecar-encryption work).

## Related docs

- [Observability](observability.md) — `RunEvent`/`EventType`/`EventSink`; `StorageService` *is* an `EventSink`.
- [Memory Service](memory.md) — the long-term cognitive `MemoryStore` (distinct from this layer's stores).
- [Entity Registry](../architecture/entities.md) — where `RecommendationItem.to_record()` and run/event projections land.
- [Inference Service](inference.md) — emits the events the `ai_call_log` view flattens.
