# Context Service

> Assemble an immutable, evidenced, reproducible `ContextSnapshot` for a run by resolving declared keys from storage and/or pluggable adapters per a source preference.

## Overview

`himmy/services/context/` builds the context a run reasons over. A
`ContextBuildSpec` declares which keys to fetch, from where, and with what preference;
`ContextService.build_snapshot` resolves each key — reading storage, calling a
`ContextAdapter`, or both — into a `ContextField`, then freezes the result into a
`ContextSnapshot`. Every field carries `EvidenceRef`s (provenance), and the whole
snapshot is persisted (and optionally projected into the entity lineage graph). Two
runs with the same snapshot produce comparable answers — the basis for
reproducibility, evaluation, and audit.

This service is the **integration seam** between the run's inputs and downstream
layers: knowledge retrieval feeds it (via `KnowledgeBaseAdapter`), and the prompts
layer reads from it (via the context→prompt mapper). The service itself is deliberately
thin — it is an orchestration layer over storage + adapters.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `ContextService` — resolve keys, write-through caching, freshness/staleness, persist snapshot + evidence, entity projection. |
| `models.py` | `ContextSourcePreference`, `EvidenceRef`, `ContextField`, `ContextSpecKey`, `ContextBuildSpec`, `ContextSnapshot`. |
| `adapters.py` | `ContextAdapter` ABC — the plugin contract for external sources. |
| `__init__.py` | Public re-exports. |

## Key abstractions

### Models (`models.py`)

- **`ContextSourcePreference`** (enum): `STORAGE_FIRST`, `TOOL_FIRST`, `TOOL_ONLY`.
- **`EvidenceRef`** — provenance pointer: `source_type`, `source_id`, `row_id`,
  `account_scope`, `metadata`.
- **`ContextField`** — one typed key/value: `key`, `value`, `confidence` (default
  1.0), `freshness_seconds`, `source`, `evidence_refs`, `metadata`.
- **`ContextSpecKey`** — one build-spec entry: `key`, `required`, `source_preference`
  (default `STORAGE_FIRST`), `adapter_name`, `metadata` (carries adapter inputs like
  `kb_name`, `query`, `top_k`).
- **`ContextBuildSpec`** — `spec_id`, `keys: list[ContextSpecKey]`, `metadata`.
- **`ContextSnapshot`** — `snapshot_id`, `subject_id`, `task_id`, `fields`,
  `missing_required_keys`, `metadata`, `created_at`. A first-class entity
  (`kind="context_snapshot"`) via `to_record()`.

### `ContextAdapter` (`adapters.py`)

An ABC keyed by `name` (matched against `ContextSpecKey.adapter_name`). Wraps an
external source (CRM, market data, a vector store). `async fetch(key, scope) ->
ContextField | None`; returning `None` means "nothing for that key". The
`KnowledgeBaseAdapter` (in the [knowledge](knowledge.md) package) is the primary
implementation.

## How it works / data flow

`build_snapshot(subject_id, task_id=, build_spec, metadata=)`:

1. Coerce `build_spec` to a `ContextBuildSpec` (validated). Build a `scope` from
   `subject_id`, `task_id`, and run metadata.
2. For each `spec_key`, `_resolve_key` honors the source preference. The per-key scope
   adds `spec_metadata` (the key's metadata) so an adapter can read inputs like
   `kb_name`.
   - **`TOOL_ONLY`** — call the adapter only; never read storage, never write through.
   - **`STORAGE_FIRST`** — read storage; if fresh, return it. Else call the adapter,
     write the result through to storage, and return it. If the adapter produces
     nothing, fall back to the (possibly stale) cache.
   - **`TOOL_FIRST`** — call the adapter (write through on success), else fall back to
     storage.
3. A key that yields no value and is `required` lands in `missing_required_keys`.
4. Assemble the `ContextSnapshot`, then:
   - `save_snapshot(snapshot)`,
   - `_persist_evidence` — write each field's `EvidenceRef` as a
     `ContextEvidenceRecord` to the evidence stream,
   - `_register_entities` — when an `EntityRegistry` is wired, project the snapshot and
     each evidence ref into the lineage graph and link them (`relation="built_from"`).

### Write-through & freshness

- `_write_through` persists a **copy** of an adapter-sourced field with `subject_id`
  and `cached_at` stamped into metadata — never mutating the field held in the
  snapshot (it may be a shared/cached instance).
- `_is_stale` measures age from `cached_at` against `freshness_seconds`: no TTL → never
  stale; a TTL with no `cached_at` (e.g. a hand-seeded value) → treated as fresh.

### Integration with knowledge & memory

- **Knowledge retrieval** plugs in as a `ContextAdapter`: `KnowledgeBaseAdapter`
  resolves a KB from `(workspace_id, client_id|subject_id, kb_name)` in the spec
  metadata, runs `KnowledgeBase.search`, and returns a `ContextField`
  (`value={chunks, rendered_text}`, `confidence` = max similarity, one `EvidenceRef`
  per chunk). So retrieved knowledge enters context like any other source.
- **Memory** and other sources: any source can be wrapped as a `ContextAdapter` (or
  read from storage). The service does not hard-code a memory adapter; it composes
  whatever adapters are passed in plus the storage backend. (Tool *results* land in the
  run thread, not the pre-run snapshot; the snapshot is the declaratively-built context
  that precedes/augments the run.)

## Configuration

`ContextService(storage_service=, adapters=None, entity_registry=None)`. Drop any
optional dependency and it degrades cleanly: no registry → no lineage; no adapters →
storage-only resolution. The per-run behaviour is driven entirely by the
`ContextBuildSpec` passed to `build_snapshot`.

## Extension points

- **New source:** implement `ContextAdapter` (set a unique `name`, implement
  `fetch`), pass it in `adapters=`, and target it from a `ContextSpecKey.adapter_name`.
- **Per-key tuning:** put adapter inputs in `ContextSpecKey.metadata` (e.g. `kb_name`,
  `query`/`query_template`, `top_k`, `similarity_threshold`, `metadata_filters`,
  `freshness_seconds`).
- **Lineage:** wire an `EntityRegistry` to get snapshot/evidence projection for free.

## Gotchas & invariants

- A `ContextSnapshot` is meant to be **immutable and reproducible** — the audit and
  evaluation story depends on it.
- `TOOL_ONLY` never touches storage (no read, no write-through).
- Write-through always persists a *copy*; the adapter's returned field is never
  mutated.
- A required key with no resolvable value does not raise here — it surfaces in
  `missing_required_keys` (the runtime/strict-snapshot policy decides what to do with
  it).

## Related docs

- [Knowledge](knowledge.md) — `KnowledgeBaseAdapter` is the primary `ContextAdapter`;
  KB search results become `ContextField`s.
- [Prompts](prompts.md) — `ContextPromptMapper` projects snapshot fields into system /
  task prompt blocks.
- [Evaluation service](evaluation.md) — `groundedness` checks the kind of evidence refs
  this layer produces.
