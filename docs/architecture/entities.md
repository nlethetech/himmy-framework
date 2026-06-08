# Entity Registry (the Lineage Spine)

> An append-only, content-addressed, versioned registry that every domain artefact projects onto — the single backbone for provenance, lineage traversal, and tamper-evident audit.

## Overview

`himmy.entities` is the lineage backbone. Runs, events, personas, threads, messages,
tasks/prompts, agents, skills, tool definitions, context snapshots, evidence,
recommendations, memory facts, workflow steps, and graph nodes are all projected into
one canonical shape — the `EntityRecord` — via a single projection algorithm. Records
are **immutable** and **content-addressed**: `record_id` is derived deterministically
from `(kind, stable_id, version)`, so re-registering identical content is idempotent
and a content change is a *new version*, never an in-place edit.

On top of records sit typed directed `EntityLink`s (`uses_persona`, `in_thread`,
`derived_from`, `cites`, `supersedes`, `invalidated_by`, …) forming a provenance graph
you can traverse in either direction. An integrity layer adds a tamper-evident,
signable audit bundle. Three interchangeable backends (in-memory, SQLite, Postgres)
share one API.

## Module map

| File | Responsibility |
| --- | --- |
| `records.py` | `EntityRecord` (frozen, content-addressed), `EntityLink`, `EntityQuery`; deterministic id helpers `stable_id_for` / `record_id_for`; `metadata_contains` (JSONB-`@>` semantics) |
| `registry.py` | `EntityRegistry` — the in-memory registry: versioning, idempotency, content-address-violation detection, links, query, `trace` |
| `projection.py` | `project` — the single domain-model → `EntityRecord` algorithm |
| `lineage.py` | `LineageDirection`, `LineageGraph` (typed/JSON/DOT-projectable subgraph), `DEFAULT_TRACE_DEPTH` |
| `integrity.py` | Tamper-evidence: `content_hash`, `link_hash`, `AuditBundle`, `export_audit_bundle`/`verify_audit_bundle` (HMAC) + Ed25519 variants |
| `sqlite_registry.py` | `SqliteEntityRegistry` — durable, **synchronous**, file-backed (drops into the sync runtime) |
| `postgres.py` | `PostgresEntityRegistry` — async, JSONB + GIN, `WITH RECURSIVE` traversal; `ENTITY_REGISTRY_DDL` |
| `__init__.py` | Public re-exports |
| `../../himmy/core/ids.py` | `new_uuid`, `utc_now_iso` (base id/timestamp helpers) |
| `../../himmy/core/metadata.py` | `TypedDict`s naming the framework-written `metadata` keys (`AssistantMessageMetadata`, `RouteMetadata`, `PersonaMetadata`, …) |

## Key abstractions

### `EntityRecord` (`records.py`)

An immutable, content-addressed, versioned projection of a domain artefact. It is a
**frozen** Pydantic model so `record_id` can never drift out of sync with the triple it
was derived from.

```python
record_id: str          # derived from (kind, stable_id, version) when omitted
stable_id: str          # shared across versions of the same artefact
version: int = 1
kind: str
payload: dict
metadata: dict
created_at: str
```

- `record_id_for(*, stable_id, version, kind)` → `UUID5(_HIMMY_NAMESPACE,
  "{kind}:{stable_id}:{version}")`. A fixed UUID5 namespace makes derived ids stable
  across processes and machines. `EntityRecord.create(...)` and `model_post_init`
  fill it in deterministically (`object.__setattr__` bypasses the freeze during
  construction).
- `stable_id_for(value, *, namespace, fallback_key)` derives the stable id from a
  semantic key: a value that already parses as a UUID round-trips unchanged; otherwise
  it's `UUID5(namespace, "{namespace}:{value}")`. `fallback_key` covers models whose
  primary key may be blank (e.g. a persona falling back to its name).

`EntityLink` is a typed directed relationship (`from_record_id` → `to_record_id`,
`relation`, `metadata`). `EntityQuery` filters by `kind` / `stable_id` /
`metadata_filters` (JSONB-containment via `metadata_contains`, equivalent to Postgres
`@>`); `kind` and `stable_id` are both optional.

### `project` (`projection.py`)

The single algorithm every artefact uses: derive `stable_id` from a semantic key +
namespace, then content-address it via `EntityRecord.create`. Each domain model keeps a
thin `to_record()` wrapper that **lazily** imports `project` (a deliberate choice to
avoid the historical `core <-> entities` import cycle). `payload` defaults to
`entity.model_dump(mode="json")`; callers pass an explicit payload only to capture a
curated subset (e.g. a recommendation excludes its mutable `status`/`notes` so the
content-addressed id stays stable across status transitions).

### `EntityRegistry` (`registry.py`)

The in-memory registry. Records keyed by `record_id`; versions of one artefact grouped
by `stable_id`. Key methods:

- `register(record)` — **idempotent** on byte-for-byte identical content (returns the
  stored record). Re-registering the same `record_id` with **different** payload/
  metadata is a content-address violation → raises `HimmyError`. Evolve content via
  `new_version`.
- `new_version(*, stable_id, kind, payload, metadata, expected_version=)` — appends the
  next version, with optimistic-concurrency check on `expected_version`.
- `link(...)`, `get`, `get_latest`, `get_history`, `list_by_kind`, `query`.
- `links_from` / `links_to` — forward and **reverse** edge reads (the reverse read is
  what powers "trace any output back to its persona/evidence").
- `neighbors(record_id, *, direction, relation)` and `trace(record_id, *, max_depth,
  direction, relations)` — BFS lineage traversal returning a `LineageGraph`.

### `LineageGraph` / `LineageDirection` (`lineage.py`)

`LineageDirection` = `OUT` (edges leaving a node) / `IN` (edges entering) / `BOTH`.
`trace` returns a `LineageGraph`: `nodes` (record_id → `EntityRecord`, root included),
`edges` (the typed links traversed), and `truncated` (True when a `max_depth` cutoff
stopped the walk while reachable nodes remained — so a caller can tell "whole story"
from "more, deeper"). The graph offers `filter_relations`, `relations()`,
`record_ids()`, and `to_dot()` (Graphviz). `DEFAULT_TRACE_DEPTH = 6`.

### Integrity (`integrity.py`)

Because `record_id` covers only identity (not payload), an in-place row edit in a
backend store is undetectable on its own. The integrity layer closes that **without**
changing record identity:

- `content_hash(record)` — SHA-256 over kind/stable_id/version/payload/metadata
  (excludes `created_at`, so it's stable across re-projections of identical content).
- `link_hash(link)` — SHA-256 over endpoints/relation/metadata.
- `export_audit_bundle(records, links, *, secret)` → `AuditBundle`: per-record and
  per-link content hashes, an order-independent Merkle root over them, and an
  HMAC-SHA256 signature. `verify_audit_bundle(...)` re-derives hashes from a (possibly
  tampered) live graph and reports exactly which records/links were `tampered` /
  `missing` / `added`, plus whether the signature itself is intact (`ok` requires both
  a valid signature and zero divergence).
- Ed25519 variants (`export_audit_bundle_ed25519` / `verify_audit_bundle_ed25519`) let
  an auditor verify with only the public key (stronger non-repudiation; needs
  `cryptography`).

## How it works / data flow

### How artefacts project onto the spine

Each first-class model declares only *what* identifies it (stable key, namespace, kind)
and delegates the *how* to `project`. The kinds/namespaces in the codebase include:

| Artefact | namespace / kind | Source |
| --- | --- | --- |
| Run event | `run_event` | `core/events.py` `RunEvent.to_record` |
| Persona | `persona` | `agents/personas/persona.py` |
| Task / prompt | `prompt` | `agents/base_agent/task.py` |
| Message | `message` | `agents/base_agent/thread.py` |
| Chat thread | `chat_thread` | `agents/base_agent/thread.py` |
| Agent | `agent` | `agents/base_agent/agent.py` |
| Skill | `skill` | `skills/models.py` |
| Tool definition | `tool_definition` | `services/tools/models.py` |
| Context snapshot | `context_snapshot` | `services/context/models.py` |
| Context evidence | `context_evidence` | `services/context/service.py` |
| Recommendation | `recommendation` | `services/storage/models.py` `RecommendationItem.to_record` |
| Memory fact | `memory_fact` | `services/memory/projection.py` |
| Workflow / step | `workflow` / `workflow_step` | `orchestrators/workflow.py` |
| Graph node | `graph_node` | `orchestrators/state_graph.py` |

A run's lineage is assembled by linking these. For example, the application layer (see
`himmy/application/services.py`) registers a `RecommendationItem.to_record()`, then
draws `derived_from` → the run's `chat_thread` hub record and `cites` → each evidence
record that actually exists in the registry (dangling citations stay in the payload, not
the graph). Memory consolidation draws `supersedes` / `invalidated_by` edges across
version chains. Because every artefact is content-addressed by the same algorithm, a
`trace` from a recommendation reaches the persona, prompt, snapshot, and evidence that
produced it.

### Backends

| Backend | Sync/Async | Durability | Traversal |
| --- | --- | --- | --- |
| `EntityRegistry` | sync | volatile (process-local dicts) | Python BFS |
| `SqliteEntityRegistry` | **sync** | durable stdlib `sqlite3` file | Python BFS |
| `PostgresEntityRegistry` | async | Postgres (`[postgres]` extra) | `WITH RECURSIVE` SQL |

All three are API-compatible (`register` / `new_version` / `link` / `get` /
`get_latest` / `get_history` / `list_by_kind` / `query` / `links_from` / `links_to` /
`neighbors` / `trace`) and enforce the same content-address-violation rule and
optimistic-concurrency check.

- **In-memory** is the default; **SQLite** is the middle tier — durable across restarts
  but no server, and crucially **synchronous**, so it drops straight into the runtime
  (which calls the registry synchronously) to give a run's lineage a durable home. It
  stores each record's `content_hash` alongside, ready for the audit layer.
- **Postgres** registers a JSONB codec on every connection, backs `metadata_filters`
  with a GIN index + the `@>` containment operator, uses `SELECT ... FOR UPDATE` + an
  advisory lock on the `stable_id` for race-safe `new_version`, and traverses with a
  single `WITH RECURSIVE` query. Import-safe without asyncpg; `ENTITY_REGISTRY_DDL` is
  inspectable offline.

## Configuration

| Mechanism | Effect |
| --- | --- |
| `EntityRegistry()` | In-memory, volatile (default) |
| `SqliteEntityRegistry(path)` | Durable sync registry (runtime-friendly) |
| `PostgresEntityRegistry.connect(dsn)` | Durable async registry (`[postgres]` extra) |
| `export_audit_bundle(..., secret=)` / `..._ed25519(private_pem=)` | Freeze + sign integrity |

## Extension points

- **Project a new artefact**: give the model a `to_record()` that lazily calls
  `project(...)` with its stable key, namespace, and kind.
- **New backend**: implement the registry surface (register/new_version/link/reads/
  trace) honoring the content-address rule.
- **New relation**: just `link(..., relation="your_relation")`; `trace(...,
  relations={...})` filters by it.
- **Audit**: snapshot a graph's records + links into an `AuditBundle` and verify later.

## Gotchas & invariants

- Records are **append-only and immutable** — never mutate a registered payload;
  content changes are new versions. The registry actively detects and rejects in-place
  content changes to an existing `record_id`.
- `record_id` is derived **only** from `(kind, stable_id, version)` — the payload is
  *not* in the identity, which is exactly why the integrity layer exists.
- Mutable fields (recommendation `status`/`notes`, memory `valid_to`/`tier`) are
  deliberately kept **out** of the projected payload (in metadata or links) so the
  content-addressed id stays stable and re-registration is idempotent.
- `stable_id_for` round-trips real UUIDs unchanged; only non-UUID semantic keys get
  hashed into a UUID5.
- Postgres and SQLite registries are import-safe / offline-inspectable (DDL is a plain
  string; asyncpg is lazily imported).

## Related docs

- [Memory Service](../services/memory.md) — `memory_fact` projection + `supersedes`/`invalidated_by` chains.
- [Storage Service](../services/storage.md) — `RecommendationItem.to_record()` and run/event persistence.
- [Observability](../services/observability.md) — `RunEvent.to_record()` projects events onto the spine.
- [Inference Service](../services/inference.md) — emits the run events that become spine nodes.
