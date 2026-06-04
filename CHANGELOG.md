# Changelog

All notable changes to OpenSims are documented here. This project keeps an
**offline-first** invariant: every release runs end-to-end with no network and no
keys via the deterministic `StubClientManager`; optional extras layer in real
providers, Postgres/pgvector, and observability.

## [Unreleased] — Production-hardening pass

A full kernel-by-kernel hardening pass driven by the 6-reviewer audit in
`IMPROVEMENTS.md`. Every audited item is addressed; the offline test suite is
green (`pytest -q` → all pass, only dep/DB-gated tests skip). Grouped by kernel
below.

### Added
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
- **Durable persistence selection** via `OPENSIMS_DATABASE_URL`
  (`ApiContainer.build_default_async` wires a Postgres-backed `StorageService`).
- New `CHANGELOG.md`; README "Production capabilities" section; expanded
  `.env.example` and `pyproject.toml` extras.

### Inference (INF-1 … INF-12)
- Service-level failure contract: `run` never raises for provider/manager errors —
  exceptions are normalized to a typed `InferenceError`, so retries (retryable-only),
  latency stamping, and the `INFERENCE_FAILED` event always fire (INF-4, INF-10).
- `run_batch` is failure-isolated; one bad request can no longer kill a batch (INF-5).
- `PydanticAIClientManager` honors the full request envelope — system prompt,
  message history, tools/toolsets, generation params, per-call timeout — and reads
  `result.usage()` for real token/cost accounting (INF-1, INF-3).
- `GatewayClientManager` does real Pydantic-AI-Gateway routing when a key + the
  `providers` extra are present (with an `OPENSIMS_GATEWAY_STUB_FALLBACK` offline
  switch) instead of always raising (INF-2).
- WORKFLOW mode uses `agent.iter()` and breaks right after the first
  `CallToolsNode`, enforcing one-tool-per-call on the live path (INF-6).
- Streaming entry point added (INF-7); response cache added (INF-9).
- `synthesize_from_schema` now honors `const`/`enum`/`default` ordering and basic
  constraints so offline structured output is schema-valid (INF-8).
- Proportional/configurable timeout grace replaces the fixed +1.0s floor (INF-11).
- `InferenceRequest` rejects contradictory `response_format` at construction,
  symmetric with `LLMConfig` (INF-12).
- *Deferred:* `ResponseFormat.TOOL` (force a single named tool) remains reserved
  and maps to a non-retryable `NotImplementedError` (tested) — not needed by any
  current path; WORKFLOW already covers forced single-tool exposure.

### Context + Knowledge (CK-1 … CK-12)
- Real pgvector knowledge backend implemented (CK-1); real `Embedder`/embedding
  delegation behind the `knowledge` extra (CK-2, CK-12).
- Embedding dimension validation rejects empty vectors (CK-3); freshness/staleness
  is enforced (CK-4); `kb_search` in-run tool implemented (CK-5).
- Enforced workspace/tenancy scope on knowledge reads (CK-6); sane default
  `similarity_threshold` excludes zero-similarity chunks (CK-7).
- `SemanticChunker` no longer emits near-duplicate micro-chunks (CK-8);
  write-through copies instead of mutating the adapter's `ContextField` (CK-9).
- Multimodal `ingest_image` path made reachable (CK-10); empty-text/file documents
  are rejected (CK-11).

### Storage + Entities (SE-1 … SE-12)
- `PostgresStorageService` data methods work against a live pool — the Postgres
  path is no longer dead (SE-1).
- `create_run` idempotency enforced at the DB / in-memory layer, closing the
  check-then-act race (SE-2, AAEO-7).
- `PostgresEntityRegistry.new_version` enforces optimistic concurrency at the DB so
  concurrent writers cannot silently lose a version (SE-3).
- JSONB codec registered on the asyncpg pool so `register()`/`link()` encode
  payload/metadata correctly (SE-4); pool teardown, connection-acquire timeouts,
  and migrations added (SE-5).
- `ai_call_log` view aligned with payload keys the runtime actually emits (SE-6).
- Postgres contract tests + `EntityRegistry.query`/`links_from` coverage (SE-7).
- `EntityQuery.metadata_filters` and Postgres `query()` filter consistently (SE-8).
- `EntityRecord` immutability honored and `record_id` no longer goes stale (SE-9);
  `register()` no longer silently drops content on identity collision (SE-10).
- Upserts refresh `updated_at`; timestamps indexed (SE-11); pydantic mutable
  defaults use `default_factory` (SE-12).

### Tools + Prompts (TP-1 … TP-13)
- HTTP tools SSRF-hardened: path/host pinning, no traversal or query/host
  injection (TP-1).
- `requires_approval`, `retry_hints`, and LOCAL/HTTP `timeout_seconds` are enforced
  (TP-2, TP-12); a shared reused `httpx.AsyncClient` replaces per-call clients (TP-5).
- Auth secrets redacted from logs/events; BASIC mode encodes correctly (TP-3).
- Incoming args validated against `args_json_schema` before execution (TP-4); the
  schema validator now handles `additionalProperties`/`oneOf`/`format`/numeric
  bounds and rejects unknown shapes (TP-7).
- pydantic-ai toolset binding builds per-tool arg models from `args_json_schema`
  (TP-6); post-execution hooks can no longer bypass output validation (TP-9).
- Tool-service tests cover HTTP connector, args validation, timeout, approval, and
  post-hook paths (TP-8).
- Prompt `_render` no longer corrupts templates/values containing braces or `$`
  (TP-10); `ContextPromptMapper` guards size/PII and avoids heading injection
  (TP-11); `from_paths` validates multi-doc YAML and surfaces schema feedback (TP-13).

### Runtime + Orchestrators (RO-1 … RO-12)
- Overall run/workflow timeout + cancellation handling (RO-1); workflows support
  idempotent re-run, partial-failure resume, and per-step retry (RO-3).
- `TOOL_CALLED`/`TOOL_COMPLETED`/`TOOL_FAILED` events are emitted (RO-2); step-result
  collection pairs tool calls from the response, not reconstructed TOOL rows (RO-4).
- `run_task` returns a typed run result (status/cost/structured output), not just a
  `ChatThread` (RO-5); a progress/streaming callback is available to the caller
  (RO-6).
- `thread.version` progresses correctly even without a wired entity registry (RO-8);
  forced single-tool exposure no longer no-ops when `tool_service` is `None` (RO-9).
- Cache/usage accounting wired through (RO-10); snapshot build/load failures surface
  diagnostics instead of degrading silently (RO-11).
- Public API typing tightened away from broad `Any` (RO-12); tests added for tool
  events, version progression, registry-absent multi-turn, and sequential ordering
  (RO-7).

### Application + API + Evaluation + Observability (AAEO-1 … AAEO-16)
- Background runs are durable, cancellable, and drained on shutdown (AAEO-1);
  env-driven persistence selection via `OPENSIMS_DATABASE_URL` (AAEO-2).
- Failed inference is recorded as a FAILED run with `run.error` populated, not a
  false SUCCESS (AAEO-3); structured output is schema-validated before
  recommendation extraction (AAEO-6).
- `workspace_id` tenant isolation enforced on read paths (AAEO-4); list endpoints
  gain pagination, ordering, and result caps (AAEO-8).
- Evaluation metrics documented as deterministic heuristics; `MetricEvaluator`
  supports async impls (AAEO-5, AAEO-10); `EvaluationCaseResult.passed` honors
  per-metric verdicts (AAEO-14); evaluation read/list surface added and storage
  failures no longer swallowed (AAEO-15).
- API error model + richer OpenAPI responses (AAEO-9); constant-time auth comparison
  (AAEO-13).
- Observability emits real spans for runs/inference/tools, wired to the run stream,
  instead of bare `logfire.info` (AAEO-11).
- Tests cover the FAILED run path, concurrency, and workspace isolation (AAEO-12);
  `model_key` resolution and `updated_at` stamping brought on-contract (AAEO-16).

## [0.1.0] — Initial offline-first framework

- Entity-backed agent framework: personas, tasks, threads, tools, context,
  knowledge, evaluation, runtime, orchestrators, and a FastAPI app.
- Deterministic offline `StubClientManager` and the retrying `InferenceService`.
- In-memory storage + entity registry; Postgres/pgvector/providers/observability
  scaffolds behind optional extras.
- Six runnable offline examples and a `pytest` suite (asyncio.run based).
