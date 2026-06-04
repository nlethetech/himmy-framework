# OpenSims — IMPROVEMENTS (production-hardening work-list)

> ## ✅ STATUS: production-hardening pass COMPLETE
>
> **All audited items below are implemented.** The offline test suite is fully
> green (`python3 -m pytest -q` → all pass, with only dependency/DB-gated tests
> skipping; run `pytest -q -rs` to confirm each skip is intentional). All seven
> examples (`01`–`07`) exit 0 offline (`06` skips without a DSN), import safety is
> verified (no optional dep — `pydantic_ai`/`asyncpg`/`pgvector`/`openai`/`pypdf`/
> `logfire` — is imported on the core path), and per-kernel `*_hardening.py` /
> `*_aaeo_extra.py` test suites cover the hardening contracts. See `CHANGELOG.md`
> for the per-kernel summary of what changed.
>
> **Coverage by kernel:** INF-1…12 ✅ · CK-1…12 ✅ · SE-1…12 ✅ · TP-1…13 ✅ ·
> RO-1…12 ✅ · AAEO-1…16 ✅.
>
> **Genuinely-deferred items (explicit, by design):**
> - **`ResponseFormat.TOOL`** (force a single named tool) — intentionally reserved.
>   It maps to a non-retryable `NotImplementedError` and is tested as such
>   (`tests/inference/test_inference_service.py`). *Why:* no current code path needs
>   it; WORKFLOW mode already provides forced single-tool exposure
>   (`tool_names_override` + break-after-first-`CallToolsNode`). This is the only
>   audited capability left unimplemented, and it is a clean, documented stub rather
>   than a latent bug.
>
> Items whose live behavior requires real providers/Postgres/pgvector/Logfire are
> implemented behind their optional extras; their tests skip cleanly offline and run
> against the `docker/docker-compose.yml` pgvector instance when
> `OPENSIMS_TEST_POSTGRES_DSN` is set.

---

Generated from the 6-reviewer kernel audit. Each item has a stable ID, impact, effort,
the affected files, the current behavior, and the concrete fix. Implementation agents own
the items under their assigned area. **Offline-first is a hard invariant**: every change must
keep `pytest` green with only pydantic/pyyaml/httpx/fastapi installed; real provider/DB code
stays lazily guarded and its tests are extra/docker-gated (skip when deps absent).


---

## INF — INFERENCE kernel (opensims/services/inference/)

**Assessment:** The offline kernel is well-structured and faithful to the BUILD_SPEC: typed envelopes, the StubClientManager simulating every ResponseFormat, retry-on-retryable-only, and ordered batching all work and are tested. The biggest weakness is that error normalization — a headline promise of the doc ("Errors are normalized to a stable enum... the service uses these to decide whether to retry") — is NOT enforced at the service layer: InferenceService.run only catches asyncio.TimeoutError, so any other exception a client manager raises escapes uncaught, bypassing retries, latency stamping, the INFERENCE_FAILED event, and the entire normalization contract, and it kills whole batches. The single biggest lever is to wrap _run_once in a normalize-to-InferenceError boundary (and make run_batch failure-isolated). Secondary high-value gaps: the real pydantic-ai path is a thin adapter that drops messages/tools/usage, and synthesize_from_schema silently emits invalid instances for several common JSON-schema constructs.

### INF-1 · PydanticAIClientManager real path drops messages, system prompt, tools, workflow, and generation params
- **category:** fidelity-gap · **impact:** high · **effort:** L
- **files:** opensims/services/inference/pydantic_ai_manager.py, opensims/services/inference/models.py
- **current:** pydantic_ai_manager.py:61-112. generate() builds the user prompt by concatenating only non-system messages (_user_prompt, lines 94-97) — the system prompt is discarded entirely (it should become the Agent's system_prompt/instructions). bound_tools/toolsets are never attached to the Agent, so AUTO_TOOLS and the doc's per-step tool-exchange capture (tool_calls/tool_returns) never populate. workflow/WORKFLOW is ignored. generation_params (temperature/max_tokens/top_p) and route_override are never forwarded. There is no message_history reconstruction (the to_model_message placeholder in models.py:78-91 is unused). The doc (md-inference, md-architecture provider-neutrality) describes this as the live local-dev path that produces real tool_calls/tool_returns and structured output.
- **fix:** Build the Agent with system_prompt from the system message, pass message_history reconstructed from request.messages, attach toolsets/bound tools, forward model_settings (temperature/max_tokens/top_p) and per-call timeout, and populate response.tool_calls/tool_returns from result.all_messages(). Keep it behind the optional extra, but make the adapter honor the full request envelope so it is a usable provider path, not just an echo of result.output.

### INF-2 · GatewayClientManager is a dead-end for production routing (raises even with key + extra installed)
- **category:** production · **impact:** high · **effort:** L
- **files:** opensims/services/inference/client_manager.py, opensims/services/inference/models.py
- **current:** client_manager.py:287-315. Even when PYDANTIC_AI_GATEWAY_API_KEY is set AND pydantic-ai is importable, generate() deliberately raises OpenSimsError ('GatewayClientManager production routing is a scaffold; use PydanticAIClientManager...'). The doc (md-inference 'Provider routing') presents GatewayClientManager as THE production manager for centralized billing/secrets/routing, and md-architecture lists it as the production default. There is no real gateway code path at all; api_format on GatewayModelConfig is never used.
- **fix:** Implement real gateway routing (construct a pydantic-ai Gateway/OpenAI-compatible provider from base_url/region + key, select model by registry entry's api_format+model_name) so the documented production path actually works. Until implemented, the honest error is acceptable, but it should be flagged as a known gap, not presented as the prod default.

### INF-3 · No token usage or cost accounting on the real provider path (cost is always 0.0)
- **category:** production · **impact:** high · **effort:** M
- **files:** opensims/services/inference/pydantic_ai_manager.py, opensims/services/inference/models.py
- **current:** pydantic_ai_manager.py:_map_result (lines 114-138) never reads result.usage(); input_tokens/output_tokens/cost default to 0. The StubClientManager hardcodes cost=0.0 (client_manager.py:161) and only estimates tokens for the final text (client_manager.py:157-160) — tool args/returns and structured payloads are excluded. The doc/response envelope advertises input_tokens/output_tokens/cost and the INFERENCE_SUCCEEDED event carries cost (service.py:90), which downstream dashboards/budgets are meant to consume. Production billing/quota enforcement is impossible without this.
- **fix:** In _map_result, read result.usage() (request_tokens/response_tokens) into input/output_tokens and compute cost via a per-model price table (the GatewayRuntimeConfig is the natural home for $/token). Emit those on INFERENCE_SUCCEEDED so budget/rate-limit logic has real numbers.

### INF-4 · InferenceService.run does not normalize non-timeout exceptions from the client manager
- **category:** robustness · **impact:** high · **effort:** S
- **files:** opensims/services/inference/service.py
- **current:** service.py:68-108 wraps each attempt in asyncio.wait_for and only catches asyncio.TimeoutError. _run_once (service.py:123-125) calls client_manager.generate directly. If a manager raises any other exception (a real HTTP/provider error, a pydantic-ai failure, a bug), it propagates uncaught out of run(): I confirmed `await svc.run(InferenceRequest())` raises ValueError when the manager raises, bypassing retries, latency stamping, and the INFERENCE_FAILED event. The doc (md-inference) promises 'Errors are normalized to a stable enum ... the service uses these to decide whether to retry' — that normalization currently only happens inside StubClientManager.generate, not at the service layer. The ClientManager Protocol (client_manager.py:29-43) places no obligation on generate to never raise.
- **fix:** In _run_once (or run's attempt loop), wrap client_manager.generate in try/except: map known provider exception types and the generic Exception to a FAILED InferenceResponse with a normalized InferenceError (UNKNOWN/PROVIDER_UNAVAILABLE, retryable as appropriate) so retries, latency, and the INFERENCE_FAILED event all still fire. This is the service-level contract the doc describes.

### INF-5 · run_batch is not failure-isolated — one raising request kills the whole batch
- **category:** robustness · **impact:** high · **effort:** S
- **files:** opensims/services/inference/service.py
- **current:** service.py:138-140 uses `await asyncio.gather(*(_guarded(req) ...))` without return_exceptions=True. Combined with the unguarded _run_once above, I confirmed a single manager that raises makes run_batch raise RuntimeError instead of returning a BatchInferenceResponse with failure_count incremented. The doc's batching contract (md-inference, examples/04_orchestration_team.py) implies per-request isolation with success_count/failure_count tallies.
- **fix:** Either fix the service-level normalization (finding above, which makes run() never raise) or pass return_exceptions=True to gather and convert any returned exception into a FAILED InferenceResponse before tallying. Prefer the former so the guarantee holds for run() too.

### INF-6 · WORKFLOW uses agent.run-equivalent, not the doc's agent.iter()/CallToolsNode single-step mechanic
- **category:** fidelity-gap · **impact:** medium · **effort:** L
- **files:** opensims/services/inference/pydantic_ai_manager.py, opensims/services/inference/client_manager.py
- **current:** The doc (md-inference 'Workflow mode') is explicit: 'The inference service uses agent.iter() instead of agent.run() and breaks right after the first CallToolsNode executes, so the model never gets re-prompted with the tool result.' The actual implementation has no agent.iter() anywhere — the stub's _run_workflow (client_manager.py:205-229) directly invokes the one bound tool and the PydanticAIClientManager always calls agent.run() (pydantic_ai_manager.py:74) regardless of response_format. The stub contract is fine offline, but the real provider path will run the full agent loop (re-prompting after the tool result) instead of breaking after the first CallToolsNode, so forced one-tool-per-call is NOT actually enforced against a live model.
- **fix:** Implement a workflow branch in PydanticAIClientManager that uses agent.iter(), filters the toolset to request.tool_names_override (already produced by the runtime, single_agent.py:517), and breaks after the first CallToolsNode — matching the documented mechanic. Until then, document clearly that WORKFLOW enforcement is stub-only.

### INF-7 · No streaming support anywhere in the envelope or service
- **category:** capability · **impact:** medium · **effort:** L
- **files:** opensims/services/inference/service.py, opensims/services/inference/pydantic_ai_manager.py, opensims/services/inference/client_manager.py
- **current:** Neither InferenceRequest/InferenceResponse nor InferenceService expose any streaming/token-delta path; run() returns a single fully-materialized InferenceResponse. pydantic-ai supports agent.run_stream()/iter() token streaming. For interactive/chat surfaces (the framework targets agent runs and a FastAPI BFF per md-architecture) the lack of streaming is a real capability gap for latency-sensitive UX.
- **fix:** Add an async-generator entry point (e.g. InferenceService.run_stream → AsyncIterator of typed deltas) backed by agent.run_stream() in the pydantic-ai manager, with the stub emitting deterministic chunked deltas of its echo so the path is testable offline. Keep run() as the buffered convenience wrapper.

### INF-8 · synthesize_from_schema silently produces schema-INVALID or empty instances for common constructs
- **category:** robustness · **impact:** medium · **effort:** M
- **files:** opensims/services/inference/models.py
- **current:** models.py:325-418. I verified several gaps that power the offline STRUCTURED_OUTPUT path: (1) `default` short-circuits at line 342 BEFORE enum/required checks, so {enum:[a,b], default:z} returns 'z' (enum-violating). (2) `const` is unhandled → returns None (a Literal('X')/const schema yields None, invalid). (3) Dict-typed objects (type:object with additionalProperties but no properties) return {} — if minProperties>0 this is invalid; for required Dict[str,X] fields it yields an empty map. (4) Numeric/string constraints minimum/maximum/minLength/maxLength/pattern/format are ignored: {type:string,minLength:5} returns 'value', {type:integer,minimum:10} returns 0 — both violate the schema a real model would satisfy. (5) Non-prose required string fields get the literal field name as their value (e.g. {'description':'description'}), which is technically valid but semantically useless seed data.
- **fix:** Handle `const` (return its value); check enum/required BEFORE blindly returning `default` (or validate default against the local constraints); honor minLength/minimum/minItems so generated instances clear basic JSON-schema validators; for object schemas with additionalProperties and minProperties>0 synthesize one key. Optionally validate the synthesized instance with jsonschema in tests to catch regressions.

### INF-9 · LLMConfig.use_cache is plumbed but no caching layer exists
- **category:** production · **impact:** medium · **effort:** M
- **files:** opensims/services/inference/service.py, opensims/services/inference/models.py
- **current:** LLMConfig.use_cache (models.py:225) is merged into generation_params by the runtime (single_agent.py) but nothing consumes it — there is no response cache in InferenceService or any client manager. The doc table (md-inference LLMConfig) lists use_cache as a real lever. For production cost control, identical requests re-hit the provider every time.
- **fix:** Add an optional pluggable cache (keyed by a hash of model_key + messages + response_format + output_json_schema + generation_params) at the InferenceService layer, honored when generation_params['use_cache'] is True, with a clear TTL and a no-op default. Stamp cache hits on the response (e.g. metadata or a cached flag) for observability.

### INF-10 · ClientManager Protocol lets failures hide as SUCCESS; no contract that generate normalizes its own errors
- **category:** dx · **impact:** medium · **effort:** M
- **files:** opensims/services/inference/client_manager.py, opensims/services/inference/service.py, tests/inference/test_inference_service.py
- **current:** The ClientManager Protocol (client_manager.py:29-43) only declares resolve/generate signatures. There is no documented invariant about whether generate may raise vs must return a normalized FAILED response. The StubClientManager and PydanticAIClientManager self-normalize, but the service relies on this implicitly (see the unguarded _run_once finding). Test coverage also has gaps: there is no test for a manager that RAISES (only ones that return FAILED, test_inference_service.py:153-213), no test for the +1.0 timeout ceiling actually firing, and no test asserting structured-output schema validity.
- **fix:** Either document and enforce 'generate must not raise; normalize to InferenceError' in the Protocol docstring, or (preferred) add the service-level normalization boundary so the contract is robust regardless. Add tests for: a raising manager (run + run_batch isolation), the timeout path firing, and jsonschema-validating the synthesize_from_schema output for enum/const/default/constraint cases.

### INF-11 · Hard timeout floor of +1.0s makes small per-request timeouts ineffective
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/inference/service.py
- **current:** service.py:70-71 enforces asyncio.wait_for(..., timeout + 1.0). With request.timeout_seconds=0.1 the real ceiling is 1.1s — I measured a slow manager being cancelled at ~1.1s, not ~0.1s. The TIMEOUT error message (service.py:79) even reports `timeout + 1.0`. The doc (md-inference 'Retries and timeouts') says 'default_timeout_seconds: per-attempt timeout, overridable per request' and 'hard asyncio.wait_for ceiling at timeout + 1.0s' — but the +1.0 fixed floor means any timeout below ~1s is dominated by the constant, which is surprising for sub-second SLAs and batch fan-out.
- **fix:** Make the grace margin proportional or configurable (e.g. ceiling = timeout * 1.05 + small_const, or a constructor-level grace_seconds) and ensure the per-request timeout is actually the value passed to the provider call, not just the outer wait_for. Document that the inner provider also needs the timeout so cancellation is cooperative.

### INF-12 · InferenceRequest validator derives response_format but never rejects contradictions (asymmetric with LLMConfig)
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/inference/models.py
- **current:** models.py:178-186 (_derive_response_format) only fills response_format when it is None; it never validates an explicit value against related fields. LLMConfig (models.py:230-253) DOES reject conflicts. I confirmed InferenceRequest(response_format=JSON_OBJECT, output_json_schema={...}) is accepted silently and the schema is then ignored by the stub (JSON_OBJECT branch, client_manager.py:141-144). Similarly TEXT + output_json_schema, or WORKFLOW + no workflow, pass construction. This is an easy footgun for anyone building requests directly (the doc says batch/eval authors do).
- **fix:** Mirror LLMConfig's conflict checks in InferenceRequest's model_validator: if an explicit response_format contradicts output_json_schema/workflow, raise ValueError at construction. At minimum reject WORKFLOW with workflow=None and STRUCTURED_OUTPUT with output_json_schema=None so the stub's runtime OpenSimsError (client_manager.py:180,211) becomes a construction-time error.


---

## CK — CONTEXT + KNOWLEDGE kernels (opensims/services/context/*, opensims/services/knowledge/*)

**Assessment:** The implemented surface is solid and faithful to the BUILD_SPEC: snapshot assembly correctly honors STORAGE_FIRST/TOOL_FIRST/TOOL_ONLY with write-through and missing-required-key tracking, the in-memory KnowledgeBase + KnowledgeBaseAdapter deliver the doc's "next-PR" projection (correct ContextField shape, max-similarity confidence, evidence refs), and the deterministic embedder/chunker/readers all run offline. The biggest fidelity gap is that the doc's md-knowledge block repeatedly states the Postgres/pgvector backend (vector/halfvec, ivfflat/hnsw, create_knowledge_schema) is "shipped" while postgres.py has zero knowledge methods — so the documented production retrieval path does not exist. The single highest-leverage robustness fix is closing two silent-corruption paths in the KnowledgeBase (empty-vector dim-check evasion and the >=0.0 threshold returning zero-similarity chunks), since both let irrelevant or unretrievable data flow into evidenced context snapshots without error.

### CK-1 · Documented pgvector knowledge backend is entirely absent from postgres.py
- **category:** fidelity-gap · **impact:** high · **effort:** L
- **files:** opensims/services/storage/postgres.py, opensims/services/knowledge/service.py
- **current:** md-knowledge states (twice, in a Status banner and the Backends table) that 'Postgres + pgvector ... is shipped' with create_knowledge_schema(vector_dim, embedding_column_type, index_method), vector(N)/halfvec(N) columns, ivfflat/hnsw, auto-detect from pg_attribute, and cascade-delete. opensims/services/storage/postgres.py (413 lines) has NO knowledge_bases/knowledge_documents/knowledge_chunks methods and no create_knowledge_schema at all; every data method just raises self._no_conn(). KnowledgeBase.__init__ accepts `storage` but ignores it entirely, keeping everything in process-local dicts (service.py:72-76).
- **fix:** Either implement create_knowledge_schema + the knowledge_* persistence methods on PostgresStorageService (the doc's column-type/index-method/dim-ceiling logic and pg_attribute auto-detect), or correct md-knowledge to mark the pgvector backend as scaffold/next-PR like the adapter note already does. As-is the doc materially overclaims the shipped surface.

### CK-2 · Real Embedder / multimodal model are stubs; provider delegation is unimplemented
- **category:** production · **impact:** high · **effort:** L
- **files:** opensims/services/knowledge/embedder.py
- **current:** embedder.py: _OpenAICompatibleEmbedder.embed_* call _require_backend() then unconditionally raise OpenSimsError (lines 113-127) even when openai IS installed — so build_openai_compatible_embedder() can never actually embed. OpenAIMultimodalEmbeddingModel.embed_* always raise (lines 161-176). The doc's core 'embedding delegated to pydantic-ai Embedder' value prop (asymmetric query/doc, provider batching/retry/OTel, one EmbeddingModel contract across OpenAI/Cohere/Voyage/Bedrock/ST) is entirely absent.
- **fix:** Implement at least the OpenAI-compatible path against the openai SDK (batched embeddings.create, asymmetric embed_query/embed_documents, retry) gated behind the [knowledge] extra, and the pydantic-ai EmbeddingModel subclass for multimodal. Keep DeterministicEmbedder as the offline default. Add a docker-gated integration test.

### CK-3 · _validate_dims silently accepts empty-vector embeddings, corrupting the index
- **category:** robustness · **impact:** high · **effort:** S
- **files:** opensims/services/knowledge/service.py
- **current:** service.py:366-374 `_validate_dims` guards with `if vec and len(vec) != kb.vector_dim` — so an embedder returning [] (a broken provider call, a multimodal stub, a partial response) passes the check. Verified empirically: ingesting with an embedder returning [] succeeds, stores chunks with empty embeddings, and search returns them with similarity 0.0 forever. This is a silent data-corruption path into evidenced context.
- **fix:** Reject empty/zero-length vectors explicitly: treat `len(vec) == 0` (or a vector whose norm is 0) as a hard error during ingest, not a skip. Validate every chunk got a usable embedding before persisting.

### CK-4 · Freshness / staleness is modeled but never enforced anywhere
- **category:** fidelity-gap · **impact:** medium · **effort:** M
- **files:** opensims/services/context/service.py, opensims/services/knowledge/service.py
- **current:** ContextField carries freshness_seconds (models.py:41) and the md-context doc highlights it ('build context once ... freshness'), but no code reads it. STORAGE_FIRST in _resolve_key (service.py:122-131) returns any stored field unconditionally — a stale cached value is always preferred over a fresh adapter fetch, with no TTL/expiry check. The KnowledgeBaseAdapter never sets freshness_seconds at all.
- **fix:** Honor freshness in STORAGE_FIRST: if a stored field has freshness_seconds and is older than that (computed from a stored timestamp), fall through to the adapter and re-cache. Have adapters stamp freshness_seconds. This is the doc's 'caching ... tomorrow's run starts warm' promise made correct.

### CK-5 · kb_search in-run tool (TOOL_CALLED with evidence shape) is unimplemented
- **category:** fidelity-gap · **impact:** medium · **effort:** M
- **files:** opensims/services/knowledge/service.py
- **current:** md-knowledge's 'In-run lookups via the tool surface' section describes a planned kb_search tool that registers on the tool registry so ad-hoc mid-run retrievals land as TOOL_CALLED events with the same evidence shape as context-built lookups. grep finds no kb_search / register_kb_search_tool anywhere in opensims/ or examples/. Only the declarative ContextBuildSpec path exists; the 'ad-hoc 20%' path is missing.
- **fix:** Add register_kb_search_tool(registry, kb_service) that wraps KnowledgeBase.search, emits TOOL_CALLED with EvidenceRefs identical to KnowledgeBaseAdapter, and enforces tenancy (workspace/client scope) on every call. This closes the doc's 'two retrieval paths, one shape' claim.

### CK-6 · Tenancy isolation depends on caller-supplied scope with no enforced workspace check
- **category:** production · **impact:** medium · **effort:** M
- **files:** opensims/services/knowledge/service.py
- **current:** The doc promises 'per-client isolation is physical, not advisory ... that ID reachable only through a parent workspace check.' In practice KnowledgeBase.search/delete_document/get_kb take a raw kb_id and never verify it belongs to the caller's workspace/client (service.py:269-336). The adapter resolves by (workspace_id, client_id, kb_name) but search itself trusts any kb_id. No cross-tenant guard exists at the service boundary.
- **fix:** Add an authorization seam: have search/delete accept (and verify) workspace_id/client_id against the resolved KB record, or require resolution through resolve_kb so a raw kb_id alone cannot reach another tenant's chunks. This makes the 'no cross-tenant leak path' claim true in code.

### CK-7 · Default similarity_threshold=0.0 returns zero-similarity (irrelevant) chunks
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/knowledge/service.py
- **current:** service.py:293 drops a chunk only when `sim < similarity_threshold`; with the default threshold 0.0, chunks scoring exactly 0.0 (orthogonal/no lexical overlap with the deterministic embedder) are retained. Verified: a query with no token overlap still returns a chunk at similarity 0.0. The adapter then reports confidence=max similarity=0.0 and renders 'Source 1 (sim=0.00): ...' as evidence (service.py:444-448), feeding noise into the snapshot.
- **fix:** Either default the threshold to a small positive epsilon, or change the filter to drop non-positive similarities (`sim <= 0.0`) when the caller hasn't set a threshold. At minimum document that 0.0 admits orthogonal results so callers set a real threshold (the doc example uses 0.7).

### CK-8 · SemanticChunker emits near-duplicate micro-chunks when overlap is large relative to boundary cuts
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/knowledge/chunker.py
- **current:** chunker.py:42-44 sets next_start = end - overlap; when _find_boundary cuts well before start+max_chars, the next window can be almost entirely overlap. Verified with max_chars=4, overlap=3 on 'a b c d e f g h' it yields pairs like (0,4,'a b ') then (1,4,' b ') — a 3-char chunk that is ~all overlap and adds no new content, doubling chunk count and polluting top-k with near-duplicates (and doubling embed cost). Forward-progress is guaranteed but quality/cost is not.
- **fix:** Clamp overlap relative to the realized chunk length (e.g. effective_overlap = min(overlap, (end-start)//2)) or skip emitting a chunk whose new (non-overlapping) span is below a minimum. Add a test asserting no chunk is >X% overlap with its predecessor.

### CK-9 · Write-through mutates the adapter's returned ContextField in place
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/context/service.py, opensims/services/storage/service.py
- **current:** service.py:141-146 `_write_through` reassigns field.metadata on the exact ContextField object the adapter returned, which is also the object stored in snapshot.fields. An adapter that returns a cached/shared field instance (a reasonable pattern) would see its metadata mutated, and the snapshot's stored field metadata is silently rewritten with subject_id. Coupled with save_context_field deriving the storage key only from field.metadata['subject_id'] (storage/service.py:114), the storage key contract is fragile and leaks an internal subject_id into snapshot field metadata.
- **fix:** Persist a copy (field.model_copy(update={'metadata': {**field.metadata, 'subject_id': subject_id}})) instead of mutating, or pass subject_id to save_context_field as an explicit argument rather than smuggling it through metadata.

### CK-10 · ingest_image is dead code — multimodal ingest path can never succeed
- **category:** robustness · **impact:** low · **effort:** M
- **files:** opensims/services/knowledge/service.py, opensims/services/knowledge/embedder.py
- **current:** service.py:250-266 ingest_image checks `hasattr(self._embedder, 'embed_documents')` (always true for any embedder, including the deterministic one) then unconditionally raises 'requires a multimodal embedder'. There is no capability flag distinguishing multimodal embedders, so even wiring the OpenAIMultimodalEmbeddingModel would still hit the raise. KnowledgeChunk supports image_uri/caption but nothing populates them.
- **fix:** Introduce an explicit multimodal capability check (e.g. an `is_multimodal`/`supports_images` attribute or a separate protocol) and route image ingest through it, populating chunk_kind='image', image_uri, caption. Remove the always-true hasattr guard.

### CK-11 · DocumentInput accepts empty-string text/file, ingesting silent no-op documents
- **category:** robustness · **impact:** low · **effort:** S
- **files:** opensims/services/knowledge/models.py, opensims/services/knowledge/service.py
- **current:** models.py:74-81 validates 'exactly one of text/file' via `is not None`. Verified DocumentInput(text='') passes (empty string is not None); ingest_documents then chunks '' to [] (chunker.py:29-30) and creates a KnowledgeDocument with zero chunks — a silent no-op that looks ingested but is never retrievable. Likewise file='' combined with text is caught only incidentally.
- **fix:** Treat empty/whitespace-only text and empty file paths as invalid in the validator, and in ingest_documents warn or skip when a document produces zero chunks rather than persisting an empty document.

### CK-12 · EmbedderProtocol typing not enforced; KnowledgeBase.storage typed as Any
- **category:** dx · **impact:** low · **effort:** S
- **files:** opensims/services/knowledge/service.py, opensims/services/knowledge/embedder.py
- **current:** service.py:58 `storage: Any` and :59 embedder typed via TYPE_CHECKING-only import — no runtime or static guarantee the injected embedder satisfies EmbedderProtocol (which is @runtime_checkable, embedder.py:20, but never used in an isinstance check). The KB silently accepts any object; a missing embed_query surfaces only at search time. metadata_filters only supports exact top-level equality (service.py:288-289) with no typed contract.
- **fix:** Type storage against a StorageProtocol (or the StorageService union), optionally assert isinstance(embedder, EmbedderProtocol) at construction for a clear early error, and document metadata_filters as exact-match-only (or extend to nested/operator filters).


---

## SE — STORAGE + ENTITIES kernels

**Assessment:** The in-memory StorageService and EntityRegistry are clean, complete, and faithful to BUILD_SPEC and the docs: 14-table DDL matches, the ai_call_log view is real, deterministic id derivation is correct, and the in-memory surface satisfies MemoryStore/EventSink with sensible filtering. The biggest issue is a fidelity/consistency split in the Postgres layer: PostgresEntityRegistry has real SQL bodies, but PostgresStorageService's data methods unconditionally raise even when a pool is wired, so the documented example 06 "live path" (and the doc's "reads back every persisted artifact" claim) cannot run. The single biggest lever is to finish PostgresStorageService's SQL bodies (with a jsonb codec, transactions, and a DB-enforced idempotency upsert), since persistence is the kernel everything else durably leans on. Secondary high-value fixes: enforce the documented "immutable/content-addressed" record invariants (EntityRecord is mutable; same-version-different-payload silently drops data) and close the check-then-act idempotency/versioning races.

### SE-1 · PostgresStorageService data methods raise even with a live pool — the documented Postgres path is dead
- **category:** fidelity-gap · **impact:** high · **effort:** M
- **files:** opensims/services/storage/postgres.py, examples/06_postgres_storage.py
- **current:** Every data method on PostgresStorageService (save_thread, append_event, save_run, save_recommendation, etc.) calls `raise self._no_conn()` unconditionally and never touches `self._pool` (postgres.py lines 302-410). Confirmed by introspection: with a non-None pool injected, `_require_pool`/`_pool` is absent from the method bodies and `_no_conn` is always raised. md-storage promises a 'full mirror of the in-memory surface' and says `examples/06_postgres_storage.py` 'wires SingleAgentRuntime to PostgresStorageService, runs a stub-inference task, and reads back every persisted artifact'. But 06_postgres_storage.py line 60 calls `runtime.run_task(...)` which calls memory_store.save_thread/append_event (single_agent.py lines 214/665), so the live path crashes on the first persist. By contrast PostgresEntityRegistry.postgres.py has real, working SQL bodies — an internal inconsistency.
- **fix:** Implement the SQL bodies for PostgresStorageService mirroring the entity repo (INSERT ... ON CONFLICT (pk) DO UPDATE for upserts, SELECT for reads, WHERE filters for lists), gated by `_require_pool()`. Until then, the docstring and md-storage should be corrected to say the data surface is scaffold-only and example 06's live path is not exercised, so the doc does not overstate shipped behavior.

### SE-2 · create_run idempotency is a check-then-act race (TOCTOU) with no DB/in-memory unique enforcement
- **category:** robustness · **impact:** high · **effort:** M
- **files:** opensims/application/services.py, opensims/services/storage/service.py, opensims/services/storage/postgres.py
- **current:** RunAppService.create_run does `await load_run_by_idempotency(...)` then, on miss, `await save_run(run)` (application/services.py lines 214-231) with an await between the check and the write. Two concurrent requests with the same key both miss and both create runs. The in-memory StorageService.save_run is a blind dict upsert keyed only on run_id (service.py line 149) with no idempotency uniqueness, and load_run_by_idempotency is an O(n) scan (service.py lines 175-181). The Postgres DDL DOES have `runs_idempotency_idx` UNIQUE (workspace_id, idempotency_key) (postgres.py lines 109-111), but the Postgres save_run is unimplemented so the constraint never runs. md-storage sells idempotent create_run as a guarantee.
- **fix:** Make save_run idempotency-aware: in-memory, index by (workspace_id, idempotency_key) and raise/return-existing on conflict atomically (no await between read and write); in Postgres, implement save_run with INSERT ... ON CONFLICT (workspace_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING + RETURNING so the unique index is the source of truth and the race is closed at the DB.

### SE-3 · PostgresEntityRegistry.new_version optimistic-concurrency is not enforced at the DB; concurrent writers silently lose a version
- **category:** robustness · **impact:** high · **effort:** M
- **files:** opensims/entities/postgres.py
- **current:** new_version does `await get_latest()` then `await register()` with no transaction or row lock (postgres.py lines 133-147). register() uses `ON CONFLICT (record_id) DO NOTHING` (lines 111). Two concurrent new_version calls both read version N, both compute version N+1 → identical record_id, the second INSERT is swallowed by DO NOTHING, and the caller receives a record object indicating success even though nothing was written and one update was lost. The UNIQUE (stable_id, kind, version) constraint (lines 31) would catch it, but DO NOTHING suppresses the error rather than surfacing the conflict.
- **fix:** Run get_latest + insert inside a single transaction with `SELECT ... FOR UPDATE` (or an advisory lock on stable_id), and have new_version's INSERT omit ON CONFLICT (or use `RETURNING` and verify a row was actually inserted) so a lost version raises OpenSimsError instead of a false success. Mirror the in-memory error semantics.

### SE-4 · No JSONB codec registered on the asyncpg pool — entity register()/link() will mis-encode payload/metadata
- **category:** production · **impact:** high · **effort:** S
- **files:** opensims/entities/postgres.py, opensims/services/storage/postgres.py
- **current:** PostgresEntityRegistry.register/link bind JSONB columns with `json.dumps(...)` producing a Python str (postgres.py lines 117-118, 178). Without a connection-level `set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')`, asyncpg binds that str as a quoted JSON string literal (a JSON string, not an object) or errors — so payload round-trips wrong. The read side `_row_to_record` only json.loads `if isinstance(value, str)` (lines 258-262), which is brittle: with a proper codec asyncpg returns parsed dicts (the branch never fires), without one it returns a double-encoded string. connect() (lines 77-84) sets no codec and no pool `init` callback. STORAGE_DDL/entity DDL define JSONB columns but nothing wires the codec.
- **fix:** Pass an `init` coroutine to asyncpg.create_pool that calls `conn.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')`, then bind dicts directly (drop the manual json.dumps) and drop the str-guard in the row mappers. Add a docker-gated integration test that round-trips a record with a nested-dict payload to lock the contract.

### SE-5 · No pool teardown, migrations, or connection-acquire timeouts on either Postgres scaffold
- **category:** production · **impact:** medium · **effort:** M
- **files:** opensims/services/storage/postgres.py, opensims/entities/postgres.py
- **current:** Both connect() classmethods create an asyncpg pool with only `max_size` (postgres.py storage lines 268-275, entities lines 77-84). There is no `close()`/`disconnect()`/`__aexit__`, no `min_size`/`timeout`/`command_timeout`/`max_inactive_connection_lifetime`, and no migration tooling — schema evolution is a single idempotent CREATE TABLE IF NOT EXISTS DDL string (so adding a column or index to an existing table is impossible without manual ALTERs). md-storage leans on 'model fields can evolve without column migrations' which is true only for JSONB blobs, not for the indexed filter columns or new tables.
- **fix:** Add `async def close(self)` (await self._pool.close()) and async-context-manager support; expose min_size/timeout/command_timeout on connect(); and introduce a lightweight versioned migration runner (a schema_migrations table + ordered SQL files, or alembic-on-asyncpg) instead of relying solely on the idempotent DDL string for forward changes.

### SE-6 · ai_call_log view depends on payload keys the runtime is never proven to emit
- **category:** fidelity-gap · **impact:** medium · **effort:** M
- **files:** opensims/services/storage/postgres.py, opensims/core/events.py
- **current:** The ai_call_log view extracts `payload ->> 'prompt_template'`, `'prompt_version'`, `'rendered_prompt'`, `'retrieval_ctx'`, `'snapshot_id'`, and `(payload ->> 'input_tokens')::int` from run_events (postgres.py lines 214-240), and joins INFERENCE_REQUESTED to INFERENCE_SUCCEEDED/FAILED on request_id. md-storage documents these exact columns as the 'SQL-native AI trace surface'. But there is no test or example confirming the inference/runtime layer actually writes those keys into RunEvent.payload, nor that request_id is populated on both legs — so the view could silently return all-NULL columns against real data. RunEvent has request_id as a top-level column (events.py line 45), good, but token/prompt fields live only in the free-form payload dict.
- **fix:** Add a docker-gated test that emits a realistic INFERENCE_REQUESTED/SUCCEEDED pair through append_event and asserts ai_call_log returns non-null prompt_template/model_name/input_tokens/cost_usd/status for that request_id — locking the payload-key contract between the inference layer and the view. Document the required payload schema next to EventType.INFERENCE_*.

### SE-7 · Storage test depth lags the doc/spec: only the in-memory path is exercised; the Postgres scaffold has no contract test, and EntityRegistry.query/links_from have no Postgres coverage
- **category:** dx · **impact:** medium · **effort:** M
- **files:** tests/storage/test_storage_service.py, opensims/services/storage/postgres.py, opensims/entities/postgres.py
- **current:** md-storage claims `tests/storage/test_postgres_storage_service.py` (14 tests) 'exercises every method against a real Postgres', gated on OPENSIMS_TEST_POSTGRES_DSN — that file does not exist (only test_storage_service.py and tests/test_entities.py exist; no tests/storage/test_storage_runs_recommendations.py either, also cited in the doc). The only Postgres test is test_postgres_scaffold_imports_and_exposes_ddl (test_storage_service.py lines 102-109) which just asserts the DDL string contains substrings. PostgresEntityRegistry has working SQL but zero tests. No test covers in-memory list filters with multiple simultaneous predicates, or the evidence/memory/orchestration record CRUD.
- **fix:** Either ship the docker-gated tests the docs reference (a parametrized contract suite that runs the SAME assertions against StorageService and PostgresStorageService) or correct md-storage to stop citing non-existent test files. Add the missing in-memory coverage (multi-predicate list_runs/list_recommendations, evidence/memory/orchestration CRUD) so the spec'd surface is actually pinned.

### SE-8 · EntityQuery.metadata_filters only supports flat equality and Postgres query() filters metadata client-side
- **category:** capability · **impact:** medium · **effort:** M
- **files:** opensims/entities/registry.py, opensims/entities/postgres.py
- **current:** EntityRegistry.query and PostgresEntityRegistry.query both implement metadata_filters as flat top-level `record.metadata.get(k) != v` equality (registry.py lines 114-116, postgres.py lines 237-239). The Postgres path additionally fetches ALL records of a kind via list_by_kind and filters in Python (postgres.py lines 232-242) — O(n) over the whole kind, defeating the JSONB columns. There is no support for nested keys, ranges, or stable_id-only queries without a kind (query returns [] when q.kind is None in the Postgres path, line 232).
- **fix:** Push metadata_filters into SQL via the JSONB containment operator (`metadata @> $1::jsonb`) and add a GIN index on entity_records.metadata; support stable_id-only and metadata-only queries (don't require kind). Keep the in-memory path equivalent. This turns query() from a full-kind scan into an indexed lookup and unlocks the lineage queries md-entities advertises.

### SE-9 · EntityRecord is documented 'immutable' but is a plain mutable pydantic model; record_id goes stale on mutation
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/entities/records.py, tests/test_entities.py
- **current:** records.py docstrings call EntityRecord 'immutable' (lines 46, 16-21 of registry.py rely on it being content-addressed), and model_post_init uses `object.__setattr__` (line 62) as if frozen — but the model is not `frozen=True`. Verified: `r.version = 99` succeeds and leaves `record_id` pointing at the old (kind, stable_id, version), silently breaking the deterministic-id invariant that register()/get()/idempotency all depend on.
- **fix:** Set `model_config = ConfigDict(frozen=True)` on EntityRecord so the immutability/content-addressing claim is enforced. Frozen models still allow the `object.__setattr__` in model_post_init. Add a test that mutation raises and that record_id always equals record_id_for(kind, stable_id, version).

### SE-10 · register() silently drops new content on (kind, stable_id, version) collision — violates the documented identity rule
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/entities/records.py, opensims/entities/registry.py
- **current:** record_id_for derives the physical id from (kind, stable_id, version) ONLY (records.py line 42), so two records with the same triple but different payload/metadata get the same record_id. EntityRegistry.register treats an existing record_id as a no-op and returns the stored one (registry.py lines 31-33). Verified: registering payload {'v':'TAMPERED'} after {'v':'original'} returns the original and the tamper is silently lost. md-entities states 'Same stable_id, same kind, same payload, same metadata → same record' and 'Same stable_id, different payload → new version', but same-version-different-payload is neither detected nor versioned — it is silent data divergence between in-memory state and what a caller thinks was written.
- **fix:** On register(), when an existing record_id is found whose payload/metadata differ from the incoming record, raise OpenSimsError (true content-address violation) rather than silently returning the stale record; callers who want a new version must use new_version(). Document that record identity is (kind, stable_id, version), not content, so the 'different payload → new version' rule is the caller's responsibility.

### SE-11 · save_run/save_thread upserts do not refresh updated_at; created_at/updated_at are unindexed string timestamps
- **category:** robustness · **impact:** low · **effort:** M
- **files:** opensims/services/storage/service.py, opensims/services/storage/postgres.py, opensims/core/ids.py
- **current:** StorageService.save_run is a blind dict put (service.py line 149) and never touches updated_at — the field is only bumped by the caller in RunAppService._execute_run (application/services.py lines 258/268/281). Any other writer (or the future Postgres save_run) must remember to set it manually, so updated_at can silently lie. All timestamps are ISO strings via utc_now_iso (ids.py line 16) stored in TEXT columns (e.g. runs.created_at/updated_at TEXT, postgres.py lines 103-104) with no index, so the doc's 'last 24h' style ai_call_log/event queries do a full scan and string-compare on timestamps.
- **fix:** Have save_run/save_thread stamp updated_at at write time (storage owns the field) so it cannot drift, and switch persisted timestamps to TIMESTAMPTZ columns (keep the ISO string at the model boundary) with indexes on the columns actually filtered by time (run_events.timestamp, runs.updated_at). This makes the documented time-windowed analytics queries indexable.

### SE-12 · Pydantic mutable-default fields ({}/[]) are shared-class defaults rather than default_factory
- **category:** dx · **impact:** low · **effort:** S
- **files:** opensims/services/storage/models.py, opensims/entities/records.py
- **current:** Many models declare mutable defaults as literals: RunRecord.metadata={} (models.py line 48), RecommendationItem.evidence_refs=[] / metadata={} (lines 65,68), EntityRecord.payload={}/metadata={} (records.py lines 56-57), EntityQuery.metadata_filters={} (line 107), and the memory/orchestration records. Pydantic v2 deep-copies these per-instance so it is currently safe, but it relies on that implementation detail and is inconsistent with the codebase's own use of `Field(default_factory=...)` for ids/timestamps. It is a latent footgun if any of these ever become plain dataclasses or are copied via `object.__setattr__`.
- **fix:** Use `Field(default_factory=dict)` / `Field(default_factory=list)` for all mutable defaults for consistency and to remove reliance on pydantic's copy-on-default behavior. Low risk, improves maintainability and matches the rest of the file's style.


---

## TP — TOOLS + PROMPTS kernels (opensims/services/tools/*, opensims/services/prompts/*)

**Assessment:** The two kernels are clean, well-typed, import-safe, and faithful to the BUILD_SPEC data shapes; the offline LOCAL/HTTP dispatch, policy-hook seam, output validation, event emission, and bound_tools wiring all work and are tested. The biggest lever is hardening the HTTP connector path and closing the gap between the doc's promised tool-execution guarantees and what the service actually enforces: argument validation, approval gating, timeouts, and retries are all named in the doc/model but are no-ops in ToolService.execute, and the HTTP path has real SSRF/path-traversal and secret-handling exposure. The prompt _render brace-rewriting trick is also subtly incorrect for templates/values that legitimately contain braces or dollar signs.

### TP-1 · HTTP path_template is vulnerable to path traversal / query & host injection (SSRF)
- **category:** robustness · **impact:** high · **effort:** M
- **files:** opensims/services/tools/service.py
- **current:** service.py:268 builds the URL with cfg.path_template.format(**args) then service.py:280 does base_url.rstrip('/') + '/' + path.lstrip('/'). Args flow into the path unescaped: customer_id='../../admin/secrets' yields /customers/../../admin/secrets/orders, and customer_id='123?admin=true' injects a query string. No URL-encoding, no rejection of '/', '..', '?', '#', or scheme/host characters. Since args originate from model-synthesized tool calls, a prompt-injected model can reach arbitrary upstream paths.
- **fix:** URL-encode each path arg with urllib.parse.quote(str(v), safe='') before formatting (or use httpx's param substitution), reject values containing path separators/control chars, and validate that the final URL's scheme+host still match the configured base_url host after resolution. Also restrict cfg.method to an allow-list and disallow following cross-host redirects (follow_redirects=False).

### TP-2 · requires_approval, retry_hints, and LOCAL-path timeout_seconds are declared but never enforced
- **category:** fidelity-gap · **impact:** high · **effort:** M
- **files:** opensims/services/tools/service.py, opensims/services/tools/models.py
- **current:** ToolDefinition carries requires_approval, timeout_seconds, retry_hints, sequential (models.py:77-80) and the doc claims 'timeouts and approval are first-class' and 'error retries use ModelRetry'. But service.py references definition.timeout_seconds only for HTTP (service.py:281); requires_approval, retry_hints, and LOCAL timeouts have zero references in service.py. There is no asyncio.wait_for around the LOCAL handler, no built-in approval gate (the doc's example hands it to a user-written hook), and no retry loop on RATE_LIMITED/TIMEOUT inside the tool service.
- **fix:** Wrap both dispatch paths in asyncio.wait_for(definition.timeout_seconds or default) -> ToolErrorCode.TIMEOUT; add a built-in approval check (if requires_approval and not invocation.metadata.get('approved') -> POLICY_BLOCKED) before the user pre-hook; and add a bounded retry loop honoring retry_hints (max_attempts/backoff) on retryable codes (RATE_LIMITED, TIMEOUT, PROVIDER_UNAVAILABLE), mirroring the inference kernel's retry behavior described in BUILD_SPEC 4.

### TP-3 · Auth secrets risk leaking into logs/events; BASIC mode passes the raw value through unencoded
- **category:** production · **impact:** high · **effort:** M
- **files:** opensims/services/tools/service.py, opensims/services/tools/models.py
- **current:** _build_auth_headers (service.py:315-330) reads the secret from env and for BASIC returns {'Authorization': f'Basic {secret}'} expecting the env var to already be base64(user:pass) -- undocumented and a footgun. Auth headers are merged into the request but never redacted; while the success event payload only includes tool_name, args (service.py:123) ARE emitted on TOOL_CALLED and may contain header_arg_names values that are themselves secrets. There is no secret-name allow-list or redaction of args/headers in events.
- **fix:** Document/encode BASIC (accept env as user:pass and base64-encode here, or rename to a PREENCODED_BASIC mode); add redaction of known-sensitive arg/header keys before emitting TOOL_CALLED args; and ensure header values are never echoed back in error_message (service.py:308 already truncates body text -- apply similar care). Consider a per-tool 'sensitive_arg_names' list.

### TP-4 · ToolService.execute never validates incoming args against args_json_schema
- **category:** robustness · **impact:** high · **effort:** S
- **files:** opensims/services/tools/service.py
- **current:** execute (service.py:114-224) validates only the OUTPUT (service.py:185 _validate_against_schema on output_json_schema). Incoming invocation.args / policy-hook transformed_args are dispatched unchecked. The doc (md-tools) states 'args_json_schema (object schema -> pydantic-ai validates arguments)' but in the offline path that validation only happens inside pydantic-ai; client_manager.py:235 merely synthesizes args, it does not validate them. A hand-built ToolInvocation or a transformed_args from a pre-hook reaches the handler/HTTP call with no validation, and required path args are only caught as a KeyError at format time.
- **fix:** Run _validate_against_schema(args, definition.args_json_schema) (when present and an object schema) before dispatch and return ToolErrorCode.INVALID_REQUEST on failure, so the LOCAL/HTTP path has the same input-contract guarantee as the pydantic-ai path. Re-validate transformed_args from the pre-hook too.

### TP-5 · A fresh httpx.AsyncClient is created and torn down on every HTTP tool call
- **category:** production · **impact:** medium · **effort:** M
- **files:** opensims/services/tools/service.py
- **current:** service.py:283 does `async with httpx.AsyncClient(timeout=timeout) as client:` inside each _dispatch_http, so there is no connection pooling, no keep-alive reuse, no shared limits, and no retry/transport config across calls. Under load this means a new TCP+TLS handshake per tool call.
- **fix:** Hold a lazily-constructed, shared httpx.AsyncClient on the ToolService (configurable httpx.Limits, follow_redirects=False, optional httpx transport with retries) and close it on service shutdown; key per-call timeout via the request-level timeout arg rather than per-client construction.

### TP-6 · ToolServiceToolset pydantic-ai binding generates argument-less Tools (no per-tool arg model from args_json_schema)
- **category:** fidelity-gap · **impact:** medium · **effort:** M
- **files:** opensims/services/tools/runtime_adapter.py
- **current:** runtime_adapter.py:69-88 builds each pydantic-ai Tool from a generic `async def _proxy(**kwargs)` with only name+description; it never derives a typed argument model from definition.args_json_schema. The doc (md-tools 'Exposing tools to the agent') explicitly promises 'converts each ToolDefinition into a pydantic-ai Tool with a generated argument model', which is the mechanism that gives the model a real function signature and lets pydantic-ai validate arguments. The current stub would expose tools with no parameter schema to the model.
- **fix:** When the [providers] extra is present, build a pydantic model (or pydantic-ai's json-schema-driven Tool) from args_json_schema so the generated Tool advertises typed parameters and pydantic-ai performs argument validation + ModelRetry, matching the documented behavior. Add a providers-gated test.

### TP-7 · _validate_against_schema is a permissive subset and silently accepts unknown keywords (additionalProperties, oneOf, format, minimum, etc.)
- **category:** capability · **impact:** medium · **effort:** M
- **files:** opensims/services/tools/service.py
- **current:** _validate_against_schema (service.py:32-75) handles type/required/properties/items/enum and correctly special-cases bool-vs-int/number (service.py:49-52), but ignores additionalProperties (the doc's own example schema uses additionalProperties:false), oneOf/anyOf/allOf, numeric ranges, string formats/patterns, and array constraints. Output that the author believes is constrained (e.g. extra unexpected keys) passes validation. The same limited validator is the one proposed for input validation.
- **fix:** For production fidelity, optionally delegate to the `jsonschema` library when available (lazy import, like httpx) and fall back to the built-in subset offline; at minimum honor additionalProperties:false since it appears in the documented examples. Document explicitly which keywords the offline validator supports.

### TP-8 · Tool-service tests omit the HTTP connector, args validation, timeout, approval, and post-hook paths
- **category:** dx · **impact:** medium · **effort:** M
- **files:** tests/tools/test_tool_service.py, opensims/services/tools/service.py
- **current:** tests/tools/test_tool_service.py covers local dispatch, async handler, NOT_FOUND, pre-hook deny/transform, output validation, events, and bound_tools -- but there is no test for _dispatch_http at all (auth header building, path/query/body assembly, 429->RATE_LIMITED, 5xx->PROVIDER_UNAVAILABLE, 4xx->INVALID_REQUEST, non-JSON->text fallback, missing-env, missing-httpx), no test for post-execution hook, and no test for the path-traversal/escaping behavior. The HTTP connector is the most security-sensitive surface and is entirely untested.
- **fix:** Add tests using httpx.MockTransport (or monkeypatched client) to cover the full _dispatch_http matrix incl. auth modes and error-code mapping, plus tests asserting path args are escaped and traversal is rejected once the SSRF fix lands. Add a post-hook transform/exception test.

### TP-9 · Post-execution hook can replace the result AFTER output validation, bypassing the schema check
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/tools/service.py
- **current:** Output is validated at service.py:185-193, then the post-hook runs at service.py:205-211 and overwrites result.result with its return value if non-None. A post-hook (e.g. PII redaction that reshapes the payload) can therefore produce output that violates output_json_schema yet is returned as outcome='success'. The post-hook is also fully swallowed on exception (service.py:210-211), silently keeping the un-transformed result with no event/log.
- **fix:** Either run output validation after the post-hook, or re-validate result.result when the post-hook returns a new value. Surface post-hook exceptions (at minimum a warning event / metadata flag) rather than swallowing them silently so a broken redaction hook is observable.

### TP-10 · Prompt _render brace-rewriting corrupts templates/values that legitimately contain braces or dollar signs
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/services/prompts/manager.py
- **current:** manager.py:42-46 rewrites every '{' to '${' then uses Template.safe_substitute. Verified: a template containing literal non-placeholder braces, e.g. 'Use set {a, b, c}', renders as 'Use set ${a, b, c}' (the stray ${...} leaks into the final prompt). A template literal containing '$word' (e.g. 'Budget is $total') is parsed as a placeholder and only survives because the key is absent -- if a values key happened to match, it would be wrongly substituted. There is no way to escape a literal brace in a business-owned YAML template, which is exactly the audience the doc targets.
- **fix:** Replace the global brace->dollar rewrite with a real {name}-only formatter: use a regex that substitutes only valid identifier placeholders ({\w+}) present in values and leaves all other braces/'$' untouched (e.g. string.Formatter with a default-missing vformat, or str.format_map with a defaultdict that re-emits {key} for misses). Add tests for literal braces and dollar signs in both templates and values.

### TP-11 · ContextPromptMapper renders structured values inline with no size/PII guard and emits ### headings the model may misread as instructions
- **category:** capability · **impact:** low · **effort:** M
- **files:** opensims/services/prompts/mapper.py
- **current:** mapper.py:52-61 _render_value json.dumps the full field value with no truncation, and _render_keys (mapper.py:115) wraps each as '### key\n<value>'. Large snapshot fields (e.g. a big metrics blob) are injected verbatim into the prompt with no length cap, and markdown ### headings derived from arbitrary key names can collide with prompt structure. There is no redaction hook for sensitive snapshot fields flowing into the model context.
- **fix:** Add an optional max-length/truncation policy and a per-key formatter/redactor on ContextPromptKey (or the spec), and consider a less injection-prone delimiter than raw markdown headings. This also helps token-budget control in the real-inference path.

### TP-12 · register_http_tool cannot set definition-level timeout_seconds / retry_hints / sequential, so the HTTP timeout override is unreachable
- **category:** dx · **impact:** low · **effort:** S
- **files:** opensims/services/tools/registry.py
- **current:** register_http_tool (registry.py:88-114) accepts only name/http_config/description/schemas/requires_approval/metadata. ToolDefinition.timeout_seconds is consulted first in service.py:281 (definition.timeout_seconds or cfg.timeout_seconds), but the helper never sets it, so the only way to set an HTTP timeout is via HttpToolConfig.timeout_seconds, and definition-level timeout/retry_hints are silently inaccessible for HTTP tools. register_local_tool likewise omits timeout_seconds and retry_hints.
- **fix:** Add timeout_seconds and retry_hints (and, for symmetry, allow sequential on HTTP if meaningful) to both register_* helpers so all enforced/declared metadata is reachable through the public API.

### TP-13 · PromptTemplate.from_paths uses yaml.safe_load (single-doc) and silently drops non-dict sections; no schema/validation feedback
- **category:** dx · **impact:** low · **effort:** S
- **files:** opensims/services/prompts/manager.py
- **current:** manager.py:69 reads each YAML with yaml.safe_load and manager.py:71-73 silently skips any section whose value is not a dict. A malformed or multi-document template, or a section authored as a list, vanishes with no error -- the very 'business team owns YAML' workflow the doc promises has no validation or actionable failure. There is also no caching: PromptManager re-reads and re-parses files on every construction.
- **fix:** Validate the loaded structure (dict-of-dict-of-str), raise/log a clear error on malformed sections, and consider a lightweight schema check. Cache parsed templates keyed by (path, mtime) so repeated PromptManager() construction in the runtime doesn't re-parse YAML each run.


---

## RO — Runtime + Orchestrators (opensims/runtime/single_agent.py, opensims/orchestrators/workflow.py)

**Assessment:** The runtime and orchestrator faithfully implement the BUILD_SPEC's documented run_task sequence and workflow stepping: snapshot resolve/build, prompt+context projection, the SYSTEM/USER/TOOL/ASSISTANT append order with the full TOOL metadata schema, llm_config-over-task.context precedence, thread-version bumping, and lineage links are all present and exercised by 12 passing tests. Defensive degradation against missing optional deps is consistently applied. The two biggest gaps are (1) a documented-but-unimplemented event contract — the doc's per-run sequence promises TOOL_CALLED / TOOL_COMPLETED|FAILED events (and the EventType enum defines them) but run_task never emits them, breaking the audit-trail/ai_call_log fidelity claim; and (2) production-readiness around orchestration: no overall run/workflow timeout, no cancellation handling, no partial-failure recovery or idempotent re-run, no streaming/event callback to the caller, and a couple of correctness edges in multi-turn step-result collection and the thread.version bump path. The single biggest lever is to make the runtime emit the full documented event set and expose an event/stream callback so callers get incremental progress and the promised lineage.

### RO-1 · No overall run/workflow timeout or cancellation handling
- **category:** production · **impact:** high · **effort:** M
- **files:** opensims/runtime/single_agent.py, opensims/orchestrators/workflow.py
- **current:** InferenceService.run enforces request.timeout_seconds via asyncio.wait_for (service.py:65-70), so a single inference call is bounded. But run_task has no wall-clock budget across snapshot build + prompt render + inference + persistence, and the WorkflowOrchestrator.run loop (workflow.py:152-188) can run an unbounded number of steps (and the sequential_tools while-loop at 289-309 runs once per tool) with no aggregate deadline. Neither run_task nor orchestrator.run handles asyncio.CancelledError specially — a cancelled run_task will partially mutate the thread (USER appended, no ASSISTANT) and skip AGENT_RUN_FINISHED/save, and CancelledError is swallowed by the broad `except Exception` in _run_step (workflow.py:237) only if it subclassed Exception (it doesn't in py3.8+, so it propagates and aborts the workflow mid-step with no WORKFLOW_FINISHED emitted).
- **fix:** Add an optional run-level deadline (e.g. run_task(..., deadline_seconds) and orchestrator(default_step_timeout / total_timeout)) wrapping the inference call and step loop in asyncio.timeout. Explicitly catch asyncio.CancelledError in run_task to emit AGENT_RUN_FINISHED(error='cancelled') + save before re-raising, and in orchestrator.run to emit WORKFLOW_FINISHED(status='cancelled'). Document that _run_step's `except Exception` deliberately does not trap cancellation.

### RO-2 · TOOL_CALLED / TOOL_COMPLETED / TOOL_FAILED events are documented and defined but never emitted
- **category:** fidelity-gap · **impact:** high · **effort:** S
- **files:** opensims/runtime/single_agent.py, opensims/core/events.py
- **current:** md-runtime's 'Events and state changes per run' block lists `TOOL_CALLED` and `TOOL_COMPLETED | FAILED` as per-tool-invocation events, and opensims/core/events.py:24-26 defines all three EventType members. But run_task's tool replay (_append_tool_messages, single_agent.py:538-580) only writes TOOL Message rows onto the thread — it emits no RunEvents. The full _emit sequence in run_task (lines 150-304) only covers AGENT_RUN_STARTED, INFERENCE_REQUESTED, INFERENCE_SUCCEEDED/FAILED, AGENT_RUN_FINISHED (plus CONTEXT_SNAPSHOT_BUILT in _resolve_snapshot). The doc's promise that 'every tool exchange ... flows to StorageService.append_event' and powers the ai_call_log/lineage view is unmet for tool steps.
- **fix:** In _append_tool_messages, await self._emit a TOOL_CALLED event per ToolCallRecord (with tool_call_id, tool_name, tool_args, request_id, trace_id) and a TOOL_COMPLETED or TOOL_FAILED event per paired ToolReturnRecord keyed on ret.outcome. Thread trace_id/thread_id/agent_id through so the events link to the run like the other emissions. Add a test asserting these appear in the sink for the tool-calling path.

### RO-3 · Workflow has no idempotent re-run, partial-failure resume, or per-step retry
- **category:** production · **impact:** medium · **effort:** L
- **files:** opensims/orchestrators/workflow.py
- **current:** WorkflowOrchestrator.run (workflow.py:119-212) always creates a fresh ChatThread (line 137) and runs from step 0; there is no way to resume a partially-completed workflow from its last failed step, no retry on a transient INFERENCE_FAILED, and no idempotency key. stop_on_step_failure=True simply breaks the loop (line 168/188) leaving a 'partial'/'failed' WorkflowResult with no continuation handle. The doc frames workflows as 'one record of the whole run' but offers no recovery story.
- **fix:** Add optional resume support: accept a prior WorkflowResult/final_state + start_index (or persist step results keyed by workflow_id) so a failed workflow can be re-driven from the failing step with the accumulated state. Add an optional per-step max_retries with backoff for INFERENCE_FAILED-class errors. Expose an idempotency_key that short-circuits if a completed result for that key exists.

### RO-4 · Step-result collection mis-pairs tool_calls when args were lost, and reconstructs calls from TOOL rows rather than the response
- **category:** robustness · **impact:** medium · **effort:** M
- **files:** opensims/orchestrators/workflow.py, opensims/runtime/single_agent.py
- **current:** _collect_step_result (workflow.py:320-373) and _extract_last_tool_exchange (375-408) rebuild tool_calls/tool_returns by walking thread TOOL rows and reading metadata (tool_call_id, tool_name, tool_args). This loses the real ToolCallRecord/ToolReturnRecord objects (the doc's 'Result shape' says step.tool_calls is list[ToolCallRecord], but the code produces plain dicts) and depends on every TOOL row carrying complete metadata. It also walks back only to the most recent USER row — correct per turn, but in _run_single after a multi-turn shared thread, if a prior step left an ASSISTANT with content and the current step's assistant content is empty (output_text becomes None), the `output_text is None` guard at line 335 will keep scanning and could capture a stale earlier assistant message before hitting the USER boundary only if ordering differs. The reconstruction also can't distinguish two calls to the same tool in one turn because pairing relies solely on row order.
- **fix:** Have run_task stash the InferenceResponse (or its tool_calls/tool_returns and output) on the appended assistant message metadata (or return a small RunResult alongside the thread), and have the orchestrator read those typed records directly instead of re-parsing thread rows. This makes step.tool_calls genuinely list[ToolCallRecord] per the doc and removes the fragile USER-boundary walk.

### RO-5 · run_task returns only a ChatThread — no typed run result with status/cost/structured output for the caller
- **category:** capability · **impact:** medium · **effort:** M
- **files:** opensims/runtime/single_agent.py, opensims/orchestrators/workflow.py
- **current:** run_task returns `thread` (single_agent.py:313); callers must scrape thread.last_message.content (and json.loads it) to recover structured output, and read assistant metadata to learn status/cost/tokens. The doc's multi-agent pipeline example (md-guides-multi-agent) literally does `json.loads(discovered.last_message.content)`. There is no single object exposing response.status, output_structured, cost, latency, tool exchanges, or whether the workflow completed (workflow_complete is buried in assistant_message.metadata at line 310-312).
- **fix:** Add an optional richer return path, e.g. a RunResult dataclass (thread, status, output_text, output_structured, tool_calls, tool_returns, cost, latency_ms, workflow_complete) returned either directly or via a run_task_detailed() method, while keeping run_task->thread for back-compat. This also feeds the orchestrator's typed-record need above.

### RO-6 · No streaming / progress callback to the caller during a run or workflow
- **category:** production · **impact:** medium · **effort:** M
- **files:** opensims/runtime/single_agent.py, opensims/orchestrators/workflow.py
- **current:** Events are emitted only to memory_store.append_event, the entity registry, and observability (single_agent._emit, lines 662-681; orchestrator._emit, 410-425). There is no caller-facing event sink/callback parameter on run_task or WorkflowOrchestrator.run, so a UI driving a long run cannot receive incremental AGENT_RUN_STARTED/INFERENCE_*/WORKFLOW_STEP_COMPLETED events except by polling storage afterward. md-runtime advertises 'Polling for run status from a UI' via RunApplicationService but no push/stream path exists at the runtime layer.
- **fix:** Accept an optional `on_event: Callable[[RunEvent], Awaitable[None]] | None` (or an EventSink list) on the runtime and orchestrator, invoked best-effort inside _emit alongside the existing sinks. This unlocks SSE/websocket streaming and per-step progress without changing the storage spine.

### RO-7 · Test depth gaps: no assertions for tool-event emission, thread.version progression, registry-absent multi-turn, or sequential_tools ordering correctness
- **category:** dx · **impact:** medium · **effort:** M
- **files:** tests/runtime/test_runtime.py, tests/orchestrators/test_workflow.py
- **current:** tests/runtime/test_runtime.py covers the happy path, full top-level event set (test_run_emits_full_event_sequence asserts only START/INFERENCE/FINISH — not TOOL_* events), lineage, tool rows, structured output, minimal runtime, and second-turn system-skip. It never asserts thread.version increments, never runs multi-turn without a registry, and never checks inference-failure event flow. tests/orchestrators/test_workflow.py checks sequential_tools by set-equality of called tool names (line 113-114) but not call ORDER, which is the load-bearing guarantee of the WORKFLOW mode per md-guides-workflows.
- **fix:** Add tests: (1) TOOL_CALLED/TOOL_COMPLETED emitted to the sink on the tool path; (2) thread.version==2 after a second turn with and without an entity_registry; (3) INFERENCE_FAILED -> AGENT_RUN_FINISHED(error) flow with a failing stub; (4) sequential_tools preserves tool ORDER (assert the call sequence equals tool_names); (5) cancellation of run_task still saves/emits a terminal event.

### RO-8 · thread.version is only bumped when an entity_registry is wired, so persisted thread versions are wrong without lineage
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/runtime/single_agent.py, opensims/agents/base_agent/thread.py
- **current:** _register_thread_version (single_agent.py:600-610) bumps thread.version (`thread.version += 1` on 2nd+ turn) only inside the branch reached after the `if self.entity_registry is None: return None` guard at line 602-603. ChatThread docstring (thread.py:51-54) states 'Each turn that appends messages produces a new chat_thread record version'. With save_threads=True but entity_registry=None (a documented valid config: 'Drop entity_registry and you lose lineage but keep inference'), every saved version of the thread is stamped version=1 even across many turns, so memory_store.save_thread overwrites/versions incorrectly.
- **fix:** Move the version bump out of the registry-gated helper into run_task's turn logic (e.g. bump when not is_new_thread regardless of registry), then have _register_thread_version only project the record. Add a test that runs two turns with registry=None and asserts thread.version == 2 before save.

### RO-9 · Forced single-tool exposure for WORKFLOW is silently a no-op when tool_service is None
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** opensims/runtime/single_agent.py
- **current:** In _build_request (single_agent.py:507-520) the WORKFLOW/forced-tool branch that sets tool_names_override=[workflow.current_tool_name] is nested inside `if self.tool_service is not None`. If a caller drives ResponseFormat.WORKFLOW (per the doc's 'Driving the inference layer directly' example) without a tool_service, no override is set and tool_names returned to the event payload is the raw ctx tool_names. The request still carries response_format=WORKFLOW with empty bound_tools, and the stub then raises OpenSimsError('workflow step tool ... not found in bound_tools') (client_manager.py:220-223) — surfaced only as a generic INFERENCE_FAILED with no hint that the real cause is a missing tool_service.
- **fix:** When response_format==WORKFLOW and workflow.current_tool_name is set but tool_service is None (or the named tool isn't bound), fail fast with a clear error before calling inference (e.g. ValueError 'WORKFLOW response_format requires a tool_service with the named step tool bound'). Keep the tool_names_override computation outside the tool_service guard so the event payload reflects the intended single tool.

### RO-10 · LLMConfig.use_cache and generation top_p/extra_params are passed through but no caching exists; cost/token accounting relies on provider fields
- **category:** production · **impact:** low · **effort:** M
- **files:** opensims/runtime/single_agent.py, opensims/services/inference/models.py
- **current:** LLMConfig has use_cache (models.py:225) but _build_request (single_agent.py:482-494) never reads it — only response_format, schema, workflow, route_override, timeout, temperature, max_tokens, top_p, and extra_params are mapped; use_cache is silently dropped. There is no request-level caching/dedup in the runtime, so identical re-runs (common with idempotent retries) always hit the provider. Cost/token RunEvent payloads depend entirely on provider_name/cost/input_tokens being populated by the real client manager.
- **fix:** Either wire use_cache into a runtime/inference cache (keyed on model_key + messages hash + generation_params) or remove the field to avoid a misleading knob. If keeping it, document the cache backend and TTL. At minimum, map use_cache into generation_params or request metadata so a real provider adapter can honor it.

### RO-11 · Snapshot build/load failures degrade silently with no diagnostic
- **category:** robustness · **impact:** low · **effort:** S
- **files:** opensims/runtime/single_agent.py
- **current:** _resolve_snapshot (single_agent.py:316-371) swallows every exception from load_snapshot and build_snapshot into `snapshot = None` (lines 336-337, 352-353) with a `# pragma: no cover - defensive` and no event/log. A caller that supplied a snapshot_id or context_build_spec expecting evidence-backed context gets a silent prompt without it; missing_required_keys is only surfaced when a snapshot was actually built. There is no signal distinguishing 'no snapshot requested' from 'snapshot requested but failed'.
- **fix:** On a caught build/load exception, emit a diagnostic (e.g. CONTEXT_SNAPSHOT_BUILT with an error field, or a dedicated payload key 'snapshot_error') and/or log via observability so the run's audit trail records that requested context was unavailable. Optionally add a strict mode that raises when a snapshot was explicitly requested but could not be resolved.

### RO-12 · Heavy use of `Any` typing for thread/persona/task and tool_service erodes static safety on the public API
- **category:** dx · **impact:** low · **effort:** S
- **files:** opensims/runtime/single_agent.py
- **current:** run_task is typed `thread: Any | None`, returns `Any` (single_agent.py:95,99); tool_service is `Any = None` (line 54); all the helper methods take `thread: Any`. The BUILD_SPEC signature (7.1) documents `thread: ChatThread | None -> ChatThread`. The TYPE_CHECKING block already imports the real types, and ChatThread/Message are concrete pydantic models, so the Any erasure is broader than needed and hides mismatches (e.g. callers relying on thread.last_message get no IDE/type help).
- **fix:** Type run_task as `thread: ChatThread | None = None -> ChatThread` (string forward-ref under TYPE_CHECKING, already imported), and give tool_service a Protocol (e.g. ToolServiceProtocol with bound_tools(tool_names) -> list[BoundTool]) instead of Any. Keep snapshot as Any only where the optional context types genuinely vary.


---

## AAEO — Application + API + Evaluation + Observability

**Assessment:** The application/API/eval/observability layers are clean, well-documented, and faithful to the BUILD_SPEC's structural contract — services are storage-backed, degrade without a registry, and the FastAPI BFF is a genuinely thin transport layer with idempotency, background runs, and an opt-in auth guard. The biggest correctness gap is the run-failure path: the runtime never raises on a FAILED inference response, so RunAppService._execute_run marks every run SUCCEEDED and never records the model/schema error the doc explicitly promises to surface in the run record. The biggest production lever is that there is no env-driven persistence selection — build_default() always uses in-memory storage (background runs and idempotency vanish on restart and don't survive multiple workers), even though a Postgres schema/stub already exists. Evaluation metrics are honest, deterministic heuristics (token-overlap relevance, exact-key accuracy) — fine as a scaffold but naive; they should be documented as such and made pluggable toward judge/embedding-based scoring.

### AAEO-1 · Background runs are not durable, cancellable, or drained on shutdown
- **category:** production · **impact:** high · **effort:** L
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/app.py
- **current:** create_run launches asyncio.create_task(self._execute_run(...)) and tracks tasks in self._tasks for GC-safety (services.py:233-242), but there is no shutdown drain, no cancellation API, no timeout/wall-clock cap on a run, and no recovery for runs left in QUEUED/RUNNING when the process dies. On uvicorn reload or crash, in-flight runs are lost and stuck records never reach a terminal state. The doc (md-api lines 2805-2806) markets async runs + polling as a core reason for the BFF, implying durability.
- **fix:** Add a FastAPI lifespan/shutdown hook that cancels+awaits self._tasks (or a graceful drain with timeout); add a per-run execution timeout that transitions to FAILED with a timeout error; on startup, sweep runs in non-terminal states older than a TTL and mark them FAILED (or requeue). For real multi-worker durability this is the seam to introduce a task queue (e.g. asyncpg LISTEN/NOTIFY, Redis, or an external worker) — note it explicitly as the production path.

### AAEO-2 · No env-driven persistence selection — build_default always uses in-memory storage
- **category:** production · **impact:** high · **effort:** L
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/deps.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/storage/postgres.py
- **current:** ApiContainer.build_default (deps.py:46-99) hardcodes StorageService() (in-memory). _build_inference (deps.py:101-118) DOES switch on PYDANTIC_AI_GATEWAY_API_KEY, but there is no analogous _build_storage that picks PostgresStorageService when a DSN is configured. The doc (md-api line 2751) claims 'Production deployments inject real backends through ApiContainer.build_default()' and md-api lines 2810-2818 say to wire a Postgres-backed StorageService — but no such wiring path exists, and PostgresStorageService raises on every data method anyway.
- **fix:** Add a _build_storage() helper that constructs PostgresStorageService.connect()/create_schema() when e.g. OPENSIMS_DATABASE_URL is set (else in-memory), and make build_default async or provide an async build_default_async for pool setup. Pair this with finishing the Postgres backend (currently a schema + raises stub). At minimum, make the gap explicit in build_default's docstring so it doesn't read as production-ready.

### AAEO-3 · Failed inference is recorded as a SUCCEEDED run — run.error never populated
- **category:** robustness · **impact:** high · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/runtime/single_agent.py
- **current:** RunAppService._execute_run (opensims/application/services.py:245-286) only transitions to FAILED inside an `except Exception`. But SingleAgentRuntime.run_task (opensims/runtime/single_agent.py:217-305) does NOT raise on a FAILED inference response — it emits INFERENCE_FAILED, appends an assistant message with metadata['status']='FAILED', and returns the thread normally. So when the model fails (e.g. schema validation failure, gateway error), _execute_run sees a thread, sets status=SUCCEEDED, stores an empty/garbage output, and leaves run.error=None. The md-guides-structured-outputs doc (lines 3103-3109) explicitly promises: status=FAILED with error surfaced in the run record ('the application service does this already').
- **fix:** In _execute_run, inspect the terminal assistant message metadata (or have run_task return the InferenceResponse/status). If last.metadata.get('status') != 'SUCCESS', set run.status=FAILED, run.error = response error message/code (INVALID_REQUEST etc.), and skip recommendation extraction. Add a test that feeds a FAILED stub response and asserts the run record is FAILED with a populated error.

### AAEO-4 · No tenant isolation across workspace_id on read paths
- **category:** robustness · **impact:** high · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/runs.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/context.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/storage/service.py
- **current:** Several read paths ignore workspace scoping. get_run (services.py:303-305) and the GET /v1/runs/{run_id} route (routers/runs.py:86-92) return any run by id with no workspace check; get_run_events/get_run_thread likewise. list_fields / GET /v1/context/fields (routers/context.py:50-53) filter only by subject_id with no workspace_id at all — StorageService.list_context_fields (service.py:124-130) keys solely on subject_id, so two workspaces sharing a subject_id see each other's fields. update_recommendation (services.py:164-174) takes only the id, no workspace guard. The doc (md-api lines 2807) frames the BFF as 'one place to add ... tenancy'.
- **fix:** Thread workspace_id through the by-id reads (get_run, get_run_thread, get_run_events, update_recommendation, get_snapshot) and have the store/service enforce it (return None / 404 on mismatch). Add workspace_id to list_context_fields filtering. Make workspace_id a required query/dependency on the GET routes rather than optional.

### AAEO-5 · Evaluation metrics are naive deterministic heuristics presented without that caveat
- **category:** fidelity-gap · **impact:** medium · **effort:** L
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/metrics.py
- **current:** The doc (md-evaluation lines 2211-2221) describes accuracy/relevance/groundedness/safety/calibration as 'metric families' powering deploy gates. The implementations are simple heuristics: AccuracyMetric is exact key==value match (metrics.py:60-81) — brittle for free text and numbers; RelevanceMetric is bag-of-words token recall (metrics.py:93-118) with no stemming/synonyms/semantics; SafetyMetric is 3 hardcoded regex defults (metrics.py:182-203); CalibrationMetric is 1-|confidence-accuracy| on a single sample (metrics.py:215-242), which is point-wise, not a calibration curve. These are reasonable offline stubs but will mis-score real model outputs.
- **fix:** Document these as deterministic baselines and make the registry the upgrade seam (the protocol already supports it): add optional judge-based (LLM-as-judge via InferenceService) and embedding-similarity relevance metrics behind the same MetricEvaluator protocol, gated on provider availability. For calibration, aggregate across cases into bucketed ECE rather than per-case. For groundedness, support partial-credit when refs are a superset that includes all required ids.

### AAEO-6 · No structured-output schema validation before recommendation extraction
- **category:** robustness · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py
- **current:** _execute_run (services.py:278-286) calls _parse_structured() which only checks json.loads succeeds and returns a dict/list — it does NOT validate against the requested output_json_schema. extract_from_run (services.py:111-144) then best-effort coerces; if the dict has a 'recommendations' key but wrong item shapes it silently returns [] (the `except Exception: return None` at line 142-143). A model that returns malformed JSON-but-valid-text, or valid JSON not matching the requested schema, is recorded as a clean SUCCEEDED run with no signal that extraction failed.
- **fix:** Validate output_structured against the run's requested schema when one was supplied (carry the schema onto the RunRecord or re-derive from task.context). On envelope coercion failure, record the validation error into run.metadata (e.g. metadata['extraction_error']) instead of silently swallowing it, so the dashboard/operator can see schema-failure rate (the doc calls this 'a metric worth tracking per model', line 3109).

### AAEO-7 · Idempotency has a check-then-act race and no DB-level guarantee in the live path
- **category:** robustness · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/storage/service.py
- **current:** create_run (services.py:214-231) does load_run_by_idempotency then save_run as two awaited steps with no lock. Two concurrent POSTs with the same (workspace_id, idempotency_key) both pass the load (returns None) before either saves, creating two runs and two background tasks. The in-memory StorageService.load_run_by_idempotency (service.py:171-181) is a linear scan with no uniqueness enforcement. The Postgres schema DOES define a partial UNIQUE index (postgres.py:109-111) but that backend raises on every method (it's a stub) and is never wired into build_default, so the safe path is unreachable.
- **fix:** Make idempotency atomic: in the in-memory store add a save_run_if_absent_by_idempotency (or insert-then-catch-conflict) primitive, and in create_run rely on it rather than load-then-save. For the live path, the Postgres save_run should use INSERT ... ON CONFLICT (workspace_id, idempotency_key) DO NOTHING RETURNING and return the existing row on conflict. Add a concurrency test firing N create_run calls with one key.

### AAEO-8 · List endpoints have no pagination, ordering, or result caps
- **category:** production · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/runs.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/recommendations.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py
- **current:** GET /v1/runs (routers/runs.py:73-83), GET /v1/recommendations (routers/recommendations.py:27-43), and GET /v1/context/fields return whole filtered lists. The store returns dict.values() with no limit/offset and no stable sort (services.py list_runs / storage service.py:156-169, 197-214). At scale a single workspace's run list is unbounded and the JSON response grows without limit; ordering is dict-insertion order (non-deterministic across backends).
- **fix:** Add limit/offset (or cursor) + a deterministic order (created_at desc, run_id tiebreak) to list_runs/list_recommendations at the service and route level, with a sane default cap (e.g. 100) and max. Return a small envelope ({items, next_cursor/total}) or use pagination headers so the dashboard can page.

### AAEO-9 · API error shapes and OpenAPI quality are thin (raw dict/Any responses, no error model)
- **category:** dx · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/dashboard.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/routers/runs.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/app.py
- **current:** Routes return bare dict[str, Any] / Any (dashboard.py:18, runs.py get_run_events returns list[Any], context build returns ContextSnapshot but dashboard/summary is untyped dict). 404s use FastAPI's default {'detail': ...}; there's no consistent error envelope or documented error responses. The internal-key guard (app.py:37-43) raises 401 with detail string but adds no WWW-Authenticate header and isn't surfaced in OpenAPI as a security scheme. POST /v1/runs returns RunRecord with output_structured: Any, which yields a weak schema.
- **fix:** Define typed response models (DashboardSummary, a paginated RunList, an ErrorResponse) and declare responses={404: ...} on routes for accurate OpenAPI. Register the internal key as an APIKeyHeader security scheme so it shows in docs. Add a global exception handler mapping OpenSimsError -> structured error body. Consider a response_model on POST /v1/runs that hides internal fields.

### AAEO-10 · MetricEvaluator protocol is sync but the natural production impls (LLM judge) are async
- **category:** capability · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/metrics.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/service.py
- **current:** MetricEvaluator.score (metrics.py:14-20) and EvaluationService._score_case (service.py:83-106) are synchronous. Every realistic 'hardened' metric (LLM-as-judge, embedding similarity, an HTTP safety classifier) is I/O-bound and async. The current contract forces such metrics to block the event loop or do sync HTTP, and run_suite scores cases serially (service.py:50-62) with no concurrency.
- **fix:** Either make score() optionally awaitable (support both sync and async evaluators, awaiting when a coroutine is returned) or add an async score_async path; then fan out case/metric scoring with asyncio.gather (bounded by a semaphore) so a suite of N cases over a judge model isn't O(N) serial latency.

### AAEO-11 · Observability span emission is decoupled from the run stream and only logfire.info, not real spans
- **category:** fidelity-gap · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/observability/__init__.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/runtime/single_agent.py
- **current:** emit_event_span (observability/__init__.py:69-93) calls logfire.info(span_name, ...) — that emits a log record, not a parent/child span with timing, so the documented trace tree (md-observability lines 2329-2341: agent_run_started as a marker nested under the pydantic-ai 'agent run' span) is not actually produced as spans. configure_observability (lines 57-61) calls instrument_pydantic_ai but emit_event_span is never invoked from the runtime (no call site exists in single_agent.py), and OPENSIMS_LOGFIRE_INCLUDE_CONTENT is stashed into os.environ (lines 63-65) but no code reads it to gate content on spans. So the 'mirror RunEvents into the timeline' capability is effectively unwired.
- **fix:** Wire emit_event_span into the runtime's _emit so AGENT_RUN_STARTED/FINISHED and TOOL_COMPLETED/FAILED actually reach Logfire; use logfire.span(...) (context-managed) for run/tool lifecycle rather than logfire.info so they nest correctly; and have emit_event_span read OPENSIMS_LOGFIRE_INCLUDE_CONTENT to drop prompt/completion payload keys when content is disabled (currently it forwards all payload values, a PII risk the doc's default-false setting is meant to prevent).

### AAEO-12 · Test depth: no coverage of the FAILED run path, concurrency, or workspace isolation
- **category:** dx · **impact:** medium · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/tests/application/test_application.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/tests/api/test_api.py
- **current:** tests/application/test_application.py exercises QUEUED->SUCCEEDED, idempotency (happy path), extraction, status transition, dashboard counts, and event/thread replay — all on the stub success path. There is no test that an inference failure yields RunStatus.FAILED with run.error set, no concurrent-idempotency test, no cross-workspace leakage test, and no test that get_run/get_run_thread enforce scoping. Because the stub always succeeds, the entire FAILED branch (services.py:265-270) and the doc-promised error surfacing are unverified.
- **fix:** Add a stub/failing client manager fixture that returns an InferenceStatus.FAILED response and assert the run record ends FAILED with a populated error and no recommendations. Add a concurrency test (asyncio.gather of N create_run with one key -> exactly one run). Add a workspace-isolation test once scoping is added.

### AAEO-13 · Auth guard uses non-constant-time comparison and is the only access control
- **category:** production · **impact:** medium · **effort:** S
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/api/app.py
- **current:** _internal_key_dependency._guard (app.py:37-42) compares `provided != expected` with ==, which is not timing-safe for a secret. It's also a single shared static key with no per-caller identity, no rate limiting, and no per-workspace authorization — anyone with the key reaches every workspace's data (compounding the tenant-isolation gap above). The doc frames it as a trusted-boundary header behind a proxy, which is reasonable, but the comparison itself is a real (if minor) weakness.
- **fix:** Use hmac.compare_digest for the key check. Document clearly that this is a coarse trusted-boundary gate, not user authz, and that tenant scoping must come from an upstream identity (e.g. a workspace claim injected as a dependency). Add a basic rate-limit hook point. Optionally support multiple keys / key rotation via a set.

### AAEO-14 · EvaluationCaseResult.passed verdict ignores per-metric passed flags
- **category:** robustness · **impact:** medium · **effort:** S
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/service.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/metrics.py
- **current:** A case passes when its weighted aggregate >= 0.5 (service.py:59, _CASE_PASS_THRESHOLD). This means a case can be marked passed while a critical metric (e.g. safety=0.0, a hard fail) is outvoted by high accuracy/relevance. For a deploy gate (the doc's stated use, md-evaluation line 2259), a safety failure should be a hard veto regardless of aggregate.
- **fix:** Support 'must-pass' / veto metrics (e.g. safety, groundedness) so any failed required metric forces case.passed=False independent of the weighted aggregate; expose the veto set via EvaluationCase/metric metadata. Also make the pass threshold configurable per suite/case rather than a module constant.

### AAEO-15 · EvaluationService swallows storage failures silently and has no read/list surface in the API
- **category:** dx · **impact:** low · **effort:** M
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/services/evaluation/service.py, /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py
- **current:** run_suite wraps save_evaluation_run in a bare try/except: pass (service.py:76-81), so a persistence failure is invisible — the doc (md-evaluation line 2257) and the dashboard pitch (line 2261, 'pipe results into the dashboard') assume eval runs are durable and queryable, but there is no /v1/evaluation route and DashboardQueryService.summary (services.py:362-410) does not include eval scores. The storage layer has save/get/list_evaluation_runs but they're unreachable via HTTP.
- **fix:** Log (don't silently swallow) eval-run persistence failures, and either re-raise or record a flag on the returned run. Add an evaluation router (POST run-suite / GET runs by suite) and fold latest aggregate eval scores into the dashboard summary so the documented 'scorecards on a dashboard' story is real.

### AAEO-16 · model_key resolution and updated_at stamping are slightly off-contract
- **category:** robustness · **impact:** low · **effort:** S
- **files:** /Users/samriddhagc/LocalProjects/himmy-agent-test/opensims/application/services.py
- **current:** create_run resolves model_key as (llm_config.model_key) or task.context.get('model_key') (services.py:226-228) — but LLMConfig.model_key defaults to 'default' (non-None), so a task that intends a different model_key in context is shadowed by the config default whenever any llm_config is passed. Separately, RunRecord.updated_at is set via _now() on each transition (services.py:258,268,281) but created_at/updated_at are ISO strings (storage/models.py:49-50); list_runs ordering and any 'latest' logic that relies on these would need lexical-ISO sort, which works only because utc_now_iso is zero-padded — fragile and undocumented.
- **fix:** Resolve model_key with explicit precedence (llm_config.model_key only when caller set it non-default, else task.context, else None) or document that llm_config always wins. Keep timestamps ISO-8601 UTC (already the case) and add an explicit created_at-desc sort key in list_runs so ordering is intentional rather than incidental.
