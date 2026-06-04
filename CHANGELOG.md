# Changelog

All notable changes to Himmy are documented here. This project keeps an
**offline-first** invariant: every release runs end-to-end with no network and no
keys via the deterministic `StubClientManager`; optional extras layer in real
providers, Postgres/pgvector, and observability.

## [Unreleased]

### Changed
- **Renamed the project to the Himmy Agent Framework.** The import package is now
  `himmy` (was `opensims`), the base exception is `HimmyError`, environment
  variables use the `HIMMY_` prefix (e.g. `HIMMY_DATABASE_URL`), and the
  distribution is `himmy`. Update imports: `from himmy import ...`.

### Added
- **Nepal data connectors (`himmy.connectors`).** Independent, fetch-direct (no
  intermediary). **News:** a curated set of ~20 Nepali outlet RSS feeds via
  `NewsFetcher`, plus a standalone **MCP server** (`python -m
  himmy.connectors.news_mcp_server`) exposing `list_sources`/`fetch_news`/
  `search_news` to any MCP client. **NRB:** `NRBClient.forex` (the public
  forex JSON API), `list_macro_reports` (monthly Current Macroeconomic & Financial
  Situation reports via the category feed, with language/period parsed), and
  `fetch_macro_workbook`/`parse_workbook` (download + parse a report's Excel into
  sheet rows). `register_nrb_tools` wires NRB into the ToolService. All behind an
  injectable `Fetcher` seam, so the whole layer is offline-testable; live-verified
  against NRB + the feeds. New `connectors` extra (feedparser + openpyxl).
- **Tamper-evident audit (`himmy.entities.integrity`).** A deterministic SHA-256
  `content_hash` over a record's full content, plus `export_audit_bundle` (a signed
  manifest: per-record/link hashes, a Merkle root, an HMAC signature) and
  `verify_audit_bundle`, which re-derives hashes from a live graph and pinpoints
  every altered/removed/added record or link — catching even an in-place row edit
  the `(kind, stable_id, version)` id alone could not. The "provably auditable"
  layer.
- **Durable SQLite registry (`SqliteEntityRegistry`).** A stdlib-`sqlite3`,
  file-backed registry, API-compatible with `EntityRegistry` and — unlike the async
  Postgres one — **synchronous**, so it drops straight into the runtime and gives a
  run's lineage a durable home that survives restarts/power cuts with no server.
- **MCP kernel (`himmy.services.mcp`).** A transport-direct Model Context
  Protocol stdio client — newline-delimited JSON-RPC 2.0 to a server subprocess,
  with a background reader task de-multiplexing responses by id. Implemented
  against the wire format (no `mcp`/pydantic-ai SDK), so the core stays
  dependency-light and the whole path is exercisable offline against a mock
  server. `MCPClient.connect/list_tools/call_tool/aclose`; `register_mcp_tools`
  bridges a server's tools into a `ToolRegistry` (MCP `inputSchema` → the tool's
  `args_json_schema`), so MCP tools flow through the same validation / approval /
  events / lineage pipeline as native ones.
- **Sandbox kernel (`himmy.services.sandbox`).** Isolated, resource-limited code
  execution. `SubprocessSandbox` is portable defense-in-depth: a child process with
  POSIX `setrlimit` caps (CPU/memory/file-size/core), a hard wall-clock timeout that
  kills the process group, a throwaway working dir, a stripped/allow-listed env,
  bounded output, and Python `-I` isolated mode — with an explicit threat model
  (safe against runaway/buggy code; for *untrusted* code compose an OS-level
  isolate). The `Sandbox` protocol is the seam for a stronger backend, and
  `register_sandbox_tool` exposes it as an approval-gated, audited agent tool.
- **Provider/model fallback routing.** `RoutingClientManager` (a `ClientManager`)
  fails over across an ordered list of `Route`s on eligible failures
  (QUOTA/AUTH/RATE_LIMITED/TIMEOUT/PROVIDER_UNAVAILABLE), stamping
  `metadata['fallback_chain']` with what was tried and which route served — closing
  the single biggest "reach for LiteLLM" gap. Composes over stub/pydantic-ai/gateway
  and slots into `InferenceService`; streaming keeps the failover guarantee via the
  buffered fallback.
- **Content-addressed ingest.** `KnowledgeDocument` gains a `content_hash`;
  `KnowledgeBase.ingest_documents` now skips unchanged sources, replaces changed
  ones, and dedups identical raw text — so re-scanning a corpus no longer doubles
  the index or re-spends embed quota. The pgvector backend replaces-by-source in
  `persist_documents` (new `content_hash` column + source index).
- **Retrieval-quality evaluation.** New `himmy.services.knowledge.retrieval_eval`
  (`RetrievalEvalCase` → `evaluate_retrieval` → `RetrievalEvalReport`) scores a
  golden set with recall@k / precision@k / MRR / nDCG / hit-rate, fully offline —
  the feedback loop that makes chunker/threshold/reranker changes measurable.
- **Public `build_runtime()` facade.** The full offline-first stack now wires in
  one call via `from himmy import build_runtime` (lazily exposed to keep
  `import himmy` light). The example bootstrap delegates to it instead of
  duplicating the wiring.
- **Recommendations as first-class lineage nodes.** `RecommendationItem.to_record`
  projects each extracted recommendation into a `recommendation` entity, and
  `RecommendationAppService` (now registry-wired, incl. `deps.py` and the
  `RunAppService` fallback) links it `derived_from` the run's thread hub and
  `cites` each cited evidence record that exists — best-effort, idempotent, and
  sync/async-registry safe. New `RecommendationAppService.get_recommendation_lineage`
  + `GET /v1/recommendations/{id}/provenance` (tenant-scoped, `?format=dot`)
  deliver the literal "trace THIS recommendation to its persona + evidence" demo.
- **Lineage read layer.** The provenance graph was write-only; it is now
  queryable. New `EntityRegistry.links_to` (reverse edges), `neighbors`
  (direction- and relation-filtered), and `trace` (bounded multi-hop BFS)
  returning a typed `LineageGraph` (nodes + edges, `truncated` flag,
  `filter_relations`, `to_dot`). `PostgresEntityRegistry` mirrors these with a
  `WITH RECURSIVE` walk over the (previously unused) `entity_links_to_idx`.
  Surfaced as `RunAppService.get_run_lineage` and `GET /v1/runs/{run_id}/lineage`
  (tenant-scoped; `?format=dot` for Graphviz) — fulfilling the documented "trace
  any run back to its persona + evidence" promise.
- **Project health & tooling.** Initialized version control, an MIT `LICENSE`,
  a blocking GitHub Actions CI gate (`ruff` lint + format check, `mypy`, and
  `pytest` with coverage), a `.pre-commit-config.yaml` mirroring that gate, and a
  `CONTRIBUTING.md`. Ruff (`E/W/F/I/UP/B`) and mypy config now live in
  `pyproject.toml`; the tree is lint-clean, format-clean, and type-clean.
- **Streaming.** `InferenceService.run_stream` yields typed `StreamDelta` chunks
  ending in a `done` frame carrying the materialized `InferenceResponse`. The stub
  streams deterministic offline deltas; the pydantic-ai path streams real provider
  tokens via `agent.run_stream`. New `examples/07_streaming.py`.
- **Inference response cache.** Pluggable `InferenceCache` protocol with a
  `NoopInferenceCache` default and an `InMemoryTTLCache`, honored when a request
  opts in via `use_cache`.
- **pgvector knowledge backend** and a `kb_search` in-run tool that returns
  evidence-shaped results.
- **Durable persistence selection** via `HIMMY_DATABASE_URL`
  (`ApiContainer.build_default_async` wires a Postgres-backed `StorageService`).
- New `CHANGELOG.md`; README "Production capabilities" section; expanded
  `.env.example` and `pyproject.toml` extras.

## [0.1.0] — Initial offline-first framework

- Entity-backed agent framework: personas, tasks, threads, tools, context,
  knowledge, evaluation, runtime, orchestrators, and a FastAPI app.
- Deterministic offline `StubClientManager` and the retrying `InferenceService`.
- In-memory storage + entity registry; Postgres/pgvector/providers/observability
  scaffolds behind optional extras.
- Six runnable offline examples and a `pytest` suite (asyncio.run based).
