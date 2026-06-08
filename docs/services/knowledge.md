# Knowledge Service

> Per-client embedded knowledge bases with evidenced retrieval: ingest documents, then retrieve by dense cosine (default) or hybrid BM25 + dense RRF with optional cross-encoder rerank and query rewrite — fully offline by default.

## Overview

`himmy/services/knowledge/` is a per-client RAG subsystem. A `KnowledgeBase` owns
named indexes scoped by `(workspace_id, client_id, name)`; documents are read,
chunked, and embedded at ingest time (one batched embed call per job), and chunks are
retrieved at decision time, each result carrying its similarity and a context window
pulled from the parent document.

Retrieval has two paths sharing one shape:

- the declarative **`KnowledgeBaseAdapter`** (a `ContextAdapter`) for pre-run context
  enrichment, and
- the in-run **`kb_search`** tool (`register_kb_search_tool`).

Both emit the same `ContextField` projection via `build_kb_context_field`
(`value={chunks, rendered_text}`, `confidence` = max similarity, one `EvidenceRef`
per chunk), so an in-run lookup is audited identically to a context-built one.

The default backend is in-memory; pass a `KnowledgeBackendProtocol` (e.g.
`PgVectorKnowledgeBackend`) to push persistence and search into Postgres + pgvector.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `KnowledgeBase` (KB CRUD, ingest, dense + hybrid search, tenancy guards), `KnowledgeBaseAdapter`, `build_kb_context_field`. |
| `models.py` | `KnowledgeBaseRecord`, `KnowledgeDocument`, `KnowledgeChunk`, `RetrievedChunk`, `DocumentInput`. |
| `tools.py` | `register_kb_search_tool` (in-run `kb_search` LOCAL tool) + `KB_SEARCH_ARGS_SCHEMA`. |
| `embedder.py` | `EmbedderProtocol`, `DeterministicEmbedder` (offline default), OpenAI-compatible + multimodal embedders, `embedder_is_multimodal`. |
| `local_embedders.py` | `OllamaEmbedder`, `FastEmbedEmbedder`, `build_embedder`, `resolve_auto_backend`, `default_dim_for`. |
| `chunker.py` | `SemanticChunker` (default), `MarkdownAwareChunker`. |
| `readers.py` | `DocumentReaderFactory` + Text/PDF/CSV/Excel readers. |
| `backend.py` | `KnowledgeBackendProtocol`, `LexicalSearchProtocol`, `PgVectorKnowledgeBackend`, schema DDL helpers. |
| `retrieval/config.py` | `RetrievalConfig` (the single pipeline knob), `RetrievalMode`, `DEFAULT_RETRIEVAL_CONFIG`. |
| `retrieval/fusion.py` | `reciprocal_rank_fusion` (pure RRF), `DEFAULT_RRF_K = 60`. |
| `retrieval/lexical.py` | `BM25Index` (pure-Python Okapi BM25), `LexicalIndex` protocol, shared `tokenize`. |
| `retrieval/hybrid.py` | `HybridRetriever` — dense + lexical → RRF → rerank → top_k. |
| `retrieval/reranker.py` | `RerankerProtocol`, `FastEmbedReranker` (local ONNX cross-encoder), `build_reranker`. |
| `retrieval/query_rewrite.py` | `QueryRewriterProtocol`, `IdentityRewriter` (default), `MultiQueryRewriter`, `HyDERewriter`. |
| `retrieval_eval.py` | Retrieval-quality eval (recall@k / precision@k / MRR / nDCG / hit-rate). |

## Key abstractions

### Record types (`models.py`)

- **`KnowledgeBaseRecord`** — `kb_id`, `workspace_id`, `client_id`, `name`,
  `vector_dim` (default 64).
- **`KnowledgeDocument`** — full original `text` (None for images), `content_hash`
  (sha256, the content identity used to dedup/replace re-ingests), `source_uri`.
- **`KnowledgeChunk`** — `text`, `start_pos`/`end_pos` (offsets into the parent doc),
  `embedding`, `chunk_kind` (`text`/`image`), `image_uri`/`caption`.
- **`RetrievedChunk`** — `text`, `similarity`, `context_window`, `document_id`,
  `source_uri`, `metadata` (includes `chunk_id`, `start_pos`, `end_pos`, and for
  hybrid, the ranking breadcrumbs).
- **`DocumentInput`** — exactly one *non-empty* `text` or `file` (validated, so an
  empty source can't ingest a silent no-op).

### The embedder seam (`EmbedderProtocol`)

`async embed_documents(texts) -> list[list[float]]` + `async embed_query(text) ->
list[float]`. Implementations:

- **`DeterministicEmbedder`** (default) — offline, network-free, hash-based unit
  vectors (`dim=64`). Cosine reflects lexical token overlap, making it a usable
  offline retrieval default. Text-only.
- **`OllamaEmbedder`** — local Ollama `/api/embeddings` over httpx (keyless,
  injectable transport for tests).
- **`FastEmbedEmbedder`** — local ONNX via the `[embeddings]` extra (no server,
  lazy model download).
- **OpenAI-compatible** (`build_openai_compatible_embedder`) and
  **`OpenAIMultimodalEmbeddingModel`** — wire-format providers; import-safe, lazy,
  read `OPENAI_COMPATIBLE_*` env vars.

`build_embedder(name)` selects by name: `auto | deterministic | ollama | fastembed |
openai | nepali`. `"auto"` (via `resolve_auto_backend`) prefers `fastembed` (if
importable), then a reachable local Ollama, else falls back to `deterministic` — so a
zero-config caller still runs with no keys, no required deps, and no network.

### Chunking (`chunker.py`)

- **`SemanticChunker`** (default) — overlapping char windows (`max_chars=800`,
  `overlap=100`) cut on natural boundaries (paragraph/sentence/whitespace), with
  `min_new_chars` to avoid near-duplicate micro-chunks. Offsets index the original
  document.
- **`MarkdownAwareChunker`** — splits on markdown headers (`header_split_levels`,
  default H1/H2/H3) FIRST so a chunk never straddles a section, then applies the inner
  `SemanticChunker` per section. Header-less docs degrade to plain `SemanticChunker`.

## How it works / data flow

### Ingestion (`ingest_*` → `ingest_documents`)

`ingest_text` / `ingest_file` / `ingest_directory` / `ingest_image` all funnel into
`ingest_documents(kb_id, docs)`:

1. Each input is read (file) or used directly (text), then chunked. Whitespace-only
   content is **skipped** (never persisted as a no-op).
2. All chunk texts across every input are embedded in a **single batched
   `embed_documents` call** (`_embed_batched` honors `max_embed_batch=512` and
   `max_concurrent_embeds=4`). Every embedding is validated to be non-empty,
   non-zero-norm, and the correct dimension before any chunk is written.
3. Ingest is **content-addressed** (in-memory path): an unchanged input (same
   `source_uri` + `content_hash`, or same raw text) is a no-op and is *not*
   re-embedded; a changed source REPLACES the prior document and its chunks; genuinely
   new content is inserted.
4. The BM25 lexical index is invalidated on any mutation so the next hybrid search
   rebuilds it in lock-step with the dense store.

`ingest_image` requires a multimodal embedder (`embedder_is_multimodal`), else it
raises rather than producing garbage image vectors.

### Retrieval pipeline (`search`)

`search(kb_id, query, top_k=5, similarity_threshold=None, metadata_filters=,
workspace_id=, client_id=)`:

- **Tenancy guard** first (`_authorize_kb`): when a scope is supplied, a `kb_id` that
  resolves to a different workspace/client is rejected.
- **Threshold semantics:** `None` (default) drops every non-positive similarity (so
  an orthogonal chunk never leaks in just because the default was `0.0`); an explicit
  float keeps `sim >= threshold`.

**Dense mode** (`DEFAULT_RETRIEVAL_CONFIG`): embed the query, cosine-score candidates
(applying `metadata_filters` exact-equality), sort, take top_k, attach a parent-doc
`context_window` (±200 chars). This reproduces the pre-hybrid path byte-for-byte.

**Hybrid mode** (`RetrievalConfig(mode="hybrid")`) via `HybridRetriever.retrieve`:

1. **Query expansion** (optional) — `query_rewrite` runs the rewriter (multi-query /
   HyDE), always keeping the original; each variant is retrieved.
2. **Candidate generation** — the dense leg (cosine) and lexical leg (BM25) each
   retrieve `candidate_pool` deep (default 50).
3. **Fusion** — `reciprocal_rank_fusion` over the per-leg rank lists. RRF fuses on
   *rank position* (`1/(k+rank)`, summed, `k=60`), sidestepping the BM25-vs-cosine
   scale mismatch; a chunk ranked highly in both legs beats one ranked highly in only
   one. Ties break by id (deterministic).
4. **Rerank** (optional) — when `rerank=True`, a cross-encoder (`FastEmbedReranker`,
   `[embeddings]` extra) re-scores the fused top candidates and re-orders.
5. **Cut + hydrate** — keep top_k, hydrate each into a `RetrievedChunk` whose
   `similarity` is the fused/reranked score and whose metadata records *how* it ranked
   (`dense_rank`, `dense_sim`, `lexical_rank`, `lexical_score`, `rrf_score`,
   `rerank_score`, `retrieval_mode`). The post-fusion threshold uses the same
   semantics as dense.

The `HybridRetriever` is decoupled from *where* the legs run: `KnowledgeBase` injects
a dense leg, a lexical leg, a hydrator, and a text-lookup over either the in-memory
store or a lexical-capable backend. A backend that can't serve a lexical leg degrades
to dense-only (RRF over a single list) — never an error.

### Two retrieval surfaces, one shape

- **`KnowledgeBaseAdapter.fetch`** resolves a KB from
  `(workspace_id, client_id|subject_id, kb_name)` in the spec metadata, runs `search`,
  and returns a `ContextField` via `build_kb_context_field`.
- **`kb_search` tool** (`register_kb_search_tool`) resolves the KB by
  `(workspace_id, client_id, kb_name)` on every call (tenancy enforced; defaults can
  pin the scope), runs `search`, and returns the same projection. An optional per-call
  `mode` (`dense`/`hybrid`) temporarily flips the KB's retrieval mode (inheriting
  rerank/rewrite stages, always restored on exit). `KB_SEARCH_ARGS_SCHEMA` drives arg
  validation.

## Configuration

- **`KnowledgeBase(...)`**: `storage`, `embedder` (default `DeterministicEmbedder`,
  asserted to satisfy `EmbedderProtocol` at construction), `chunker` (default
  `SemanticChunker`), `reader_factory`, `max_embed_batch=512`,
  `max_concurrent_embeds=4`, `backend` (None = in-memory), `retrieval`
  (`DEFAULT_RETRIEVAL_CONFIG`).
- **`RetrievalConfig`**: `mode` (`dense`/`hybrid`), `rrf_k=60`, `candidate_pool=50`,
  `rerank=False` + `reranker`, `query_rewrite=False` + `rewriter`. A turned-on stage
  with no collaborator raises at construction (no silent no-op). `is_default_dense`
  lets the KB take the zero-risk dense path verbatim.
- **Embedder/KB dim must match**: `vector_dim` is validated against every embedding;
  `default_dim_for(name)` gives the conventional dim per backend (and `"auto"` reports
  the dim it would resolve to right now).

## Extension points

- **New embedder:** implement `EmbedderProtocol`; register a name in `build_embedder`
  if you want CLI/config selection.
- **New reranker:** implement `RerankerProtocol`; `build_reranker` currently supports
  `"fastembed"`.
- **New query rewriter:** implement `QueryRewriterProtocol` (default
  `IdentityRewriter` keeps the path offline).
- **New chunker:** any object with `chunk(text) -> list[(start, end, text)]`.
- **New backend:** implement `KnowledgeBackendProtocol` (and optionally
  `LexicalSearchProtocol` for a hybrid lexical leg) — e.g. `PgVectorKnowledgeBackend`.
- **New context adapter:** subclass `ContextAdapter` (see
  [context](context.md)).

## Gotchas & invariants

- **Tenancy is physical:** a raw `kb_id` from another tenant is rejected when a scope
  is supplied; `kb_search` always resolves by `(workspace, client, kb_name)`.
- `similarity_threshold=None` drops non-positive similarities (not `0.0`).
- The dense default carries **no lexical-index cost** — the BM25 index is built lazily
  only on the first hybrid search and invalidated on every ingest/delete.
- Hybrid `rerank=True` / `query_rewrite=True` require their collaborator or
  construction raises.
- Image ingest requires a multimodal embedder.
- The offline `DeterministicEmbedder` only matches exact token overlap — for genuine
  semantic recall use `auto`/`fastembed`/`ollama`/`openai`.

## Related docs

- [Context](context.md) — `KnowledgeBaseAdapter` is a `ContextAdapter`; KB results
  become `ContextField`s in a snapshot.
- [Prompts](prompts.md) — snapshot fields (including KB chunks) project into prompt
  blocks via the context→prompt mapper.
- [Evaluation service](evaluation.md) — `EmbeddingSimilarityMetric` reuses
  `EmbedderProtocol`; `retrieval_eval.py` scores the retrieval pipeline itself.
