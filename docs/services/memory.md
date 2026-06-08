# Memory Service

> Long-term cognitive memory: remember facts, recall them by cosine similarity, with bi-temporal validity, Letta-style tiers, audited consolidation, and optional projection onto the entity spine.

## Overview

`himmy.services.memory` is the long-term memory kernel — the agent's durable "what do
I know about this subject" store, distinct from the conversation/event persistence in
the [Storage Service](storage.md). Its core is `MemoryService` (`remember` / `recall`),
backed by a swappable `MemoryStore` (volatile in-memory or durable SQLite).

Three ideas layer on top of plain save/recall, each borrowed from a named system and
made auditable:

- **Bi-temporal validity** (Graphiti): a fact is true over `[valid_from, valid_to)`;
  facts are *invalidated*, not deleted, so point-in-time recall ("what did the user
  believe in March") stays answerable.
- **Tiered recall** (Letta): facts live in `core` / `recall` / `archival` tiers.
- **Audited consolidation** (Mem0 insight on himmy's spine): a new candidate fact is
  not blindly appended — a consolidator decides ADD / UPDATE / DELETE / NOOP and emits
  the whole decision as a replayable event.

Everything runs **fully offline** by default: the embedder is deterministic, and even
the opt-in LLM consolidation path is satisfiable by the offline inference stub.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `MemoryService` (`remember`/`recall`/`forget`/`get`/`invalidate`/`promote`), `MemoryHit`, cosine ranking, optional spine projection + audit emission |
| `store.py` | `MemoryRecord`, the `MemoryStore` protocol, `InMemoryMemoryStore`, `SqliteMemoryStore`, `MEMORY_TIERS` |
| `temporal.py` | Bi-temporal filtering helpers: `is_valid_at`, `filter_as_of` |
| `consolidation.py` | `MemoryConsolidator` — ADD/UPDATE/DELETE/NOOP decision + bi-temporal apply + `MEMORY_CONSOLIDATED` audit |
| `projection.py` | `memory_to_record` — projects a `MemoryRecord` onto an `EntityRecord` (kind `memory_fact`) |
| `adapter.py` | `MemoryContextAdapter` — injects recalled memories into a prompt via the context service |
| `__init__.py` | Public re-exports |

## Key abstractions

### `MemoryRecord` (`store.py`)

One stored fact for a subject. Beyond `memory_id` / `subject_id` / `kind` / `text` /
`metadata` / `created_at` it carries:

- `tier` — Letta tier (`core` / `recall` / `archival`; default `recall`).
- `valid_from` / `valid_to` / `superseded_by` — Graphiti bi-temporal validity.
  `valid_from` defaults to `created_at`; `valid_to=None` means currently true;
  `superseded_by` points at the replacing fact.
- `confidence` / `source` — provenance (`user` / `llm_extracted` / `tool` / `imported`).
- `stable_key` — the *semantic identity* of the fact (e.g. `"alice/home_city"`) so
  successive versions share a spine `stable_id`. `to_record(version=)` projects onto
  the entity spine.

### `MemoryStore` protocol (`store.py`)

A **synchronous** persistence contract: `save` / `list(subject_id, *, active_only,
tier)` / `get` / `delete`.

- `InMemoryMemoryStore` — volatile process-local dict.
- `SqliteMemoryStore(path)` — durable stdlib-`sqlite3` file (opened via
  `connect_hardened` from `core/sqlite_util.py`: WAL + busy timeout). Has an additive,
  idempotent in-place migration (`_MIGRATIONS`) that backfills the tier/bi-temporal/
  provenance columns onto a legacy 6-column `memories` table, so existing
  `.himmy/memory.db` files upgrade without data loss.

> This `MemoryStore` is **sync** and stores cognitive `MemoryRecord`s. It is distinct
> from the storage layer's async `ThreadEventStore` (formerly also `MemoryStore`).

### `MemoryService` (`service.py`)

Wires a (durable or in-memory) store + an embedder (default `DeterministicEmbedder`,
so recall works offline). Optional `registry` and/or `event_sink` (both default `None`)
opt the service into spine projection + audit; with neither wired it's a pure
save/recall over the store. An optional `min_similarity` is a default recall floor.

- `remember(text, *, subject_id, kind, tier, source, confidence, stable_key,
  project_version)` — persists a `MemoryRecord`; with a registry/sink wired also
  projects a `memory_fact` and emits `MEMORY_REMEMBERED`.
- `recall(query, *, subject_id, top_k, similarity_threshold, as_of, tier,
  active_only)` — see data flow below. Emits `MEMORY_RECALLED`.
- `forget` / `get` — delete (drops the cached vector) / fetch by id.
- `invalidate(memory_id, *, valid_to, superseded_by)` — Graphiti invalidate-not-delete:
  stamps `valid_to`/`superseded_by` so the fact drops out of `active_only`/`as_of`
  recall while staying queryable as of an earlier instant. The spine payload is **not**
  mutated — validity transitions are expressed via metadata + typed links.
- `promote(memory_id, tier)` — move a fact between tiers; re-projects as a new spine
  version and emits `MEMORY_CONSOLIDATED`.

### `MemoryHit`

A recalled `MemoryRecord` paired with its `similarity` to the query.

## How it works / data flow

### Recall

1. `store.list(subject_id, active_only=, tier=)` pulls candidate records.
2. If `as_of` is set, `filter_as_of` keeps only facts whose `[valid_from, valid_to)`
   window contains that instant (bi-temporal point-in-time recall;
   `temporal.is_valid_at` does the half-open interval check — ISO-8601 strings compare
   correctly, no parsing needed).
3. Missing embeddings are computed via `embedder.embed_documents` and cached per
   `memory_id` in-process; the query is embedded via `embed_query`.
4. Each candidate is scored by cosine similarity (`_cosine`) and sorted descending.
5. Thresholding:
   - `similarity_threshold` (or the service `min_similarity`) set → drop every hit
     below the floor, capped at `top_k`. An orthogonal query can correctly recall
     **nothing**.
   - Both `None` (historical default) → always return at least the single most-similar
     hit, even at similarity `0.0`.

### Consolidation (`consolidation.py`)

`MemoryConsolidator.consolidate(candidate_text, ...)` recalls the most similar
*active* existing facts, then decides one of four actions via one of two paths:

- **Offline default** (no client manager, or a stub one): a deterministic
  similarity rule — `>= dup_threshold` (0.95) → `NOOP`; `>= update_threshold` (0.80) →
  `UPDATE` the closest match; else `ADD`.
- **LLM path** (`use_llm=True` with a real `ClientManager`): a `STRUCTURED_OUTPUT`
  inference whose schema pins `action` to `NOOP`/`ADD`/`UPDATE`/`DELETE`. The enum is
  ordered so the offline stub (which picks the first enum value) yields a safe `NOOP`
  rather than a fabricated mutation.

The decision is applied **bi-temporally**: an `UPDATE` mints a new fact sharing the
old fact's `stable_key` (continuing the spine version chain as v2) and invalidates the
old one (`valid_to` stamped + `supersedes`/`invalidated_by` links drawn); a `DELETE`
invalidates the target. The whole decision is emitted as a `MEMORY_CONSOLIDATED`
`RunEvent` and projected to the registry — making the extract → decide → apply loop
replayable and auditable.

### Spine projection (`projection.py`)

`memory_to_record` runs the single `himmy.entities.projection.project` algorithm with
kind/namespace `memory_fact`. It deliberately captures only **immutable** content in
the `payload` (subject/kind/text/valid_from/source/stable_key); the mutable validity
fields (`valid_to`/`superseded_by`/`confidence`/`tier`) go to `metadata`, so the
content-addressed `record_id` stays stable when a fact is later invalidated and
re-registration remains idempotent. Successive versions of the same fact share a
`stable_id` derived from `stable_key`, so the version chain *is* the fact's
transaction-time history.

### Context injection (`adapter.py`)

`MemoryContextAdapter` is a `ContextAdapter` registered on the context service: when
the runtime builds a snapshot for a subject, it recalls the most relevant memories and
returns them as a rendered `ContextField` — so the agent sees its long-term memory with
no tool call. It supports `top_k`, a pinned `subject_id`, a `similarity_threshold`, and
tier-aware gathering: when `tiers` is set, `core` facts inject unconditionally (an
always-in-context working set) while other tiers go through the thresholded semantic
recall, so an off-topic turn still sees core facts but no irrelevant recall noise.

## Configuration

| Mechanism | Effect |
| --- | --- |
| `MemoryService(SqliteMemoryStore(path))` | Durable memory across restarts |
| `MemoryService(..., embedder=)` | Swap the recall embedder (default deterministic, offline) |
| `MemoryService(..., min_similarity=)` | Default recall floor |
| `MemoryService(..., registry=, event_sink=)` | Opt into spine projection + audit |
| `MemoryConsolidator(..., use_llm=True, client_manager=)` | LLM-driven consolidation decisions |
| `MemoryConsolidator(..., dup_threshold=, update_threshold=)` | Tune the offline similarity rule |

## Extension points

- **New store**: implement the sync `MemoryStore` protocol (`save`/`list`/`get`/`delete`).
- **New embedder**: implement `EmbedderProtocol` (from `himmy.services.knowledge.embedder`).
- **Consolidation policy**: subclass/replace `MemoryConsolidator` or flip `use_llm`.
- **Prompt injection**: register a `MemoryContextAdapter` on the context service.

## Gotchas & invariants

- Facts are **invalidated, not deleted** by default (`invalidate`); validity lives in
  metadata + links, never by editing a registered spine payload (that would trip the
  registry's content-address-violation guard).
- Recall thresholding is double-edged: with no threshold you *always* get the top hit
  (even at sim 0.0); with a threshold you can correctly get nothing.
- Embedding vectors are cached per `memory_id` within the process; `forget` drops the
  cache entry.
- Projection/audit are best-effort and isolated — a registry/sink failure never breaks
  a memory write.
- `core`-tier facts injected by the adapter bypass the similarity floor.

## Related docs

- [Entity Registry](../architecture/entities.md) — the `memory_fact` spine, version chains, and `supersedes`/`invalidated_by` links.
- [Observability](observability.md) — `MEMORY_REMEMBERED`/`MEMORY_RECALLED`/`MEMORY_CONSOLIDATED` events.
- [Inference Service](inference.md) — the `STRUCTURED_OUTPUT` path the LLM consolidator uses.
- [Storage Service](storage.md) — conversation/run persistence; its orchestration `MemoryObject` is a different record.
