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
- **Recipe gallery + real-model proof (`RECIPES.md`, `examples/10–12`).** Provider-
  selectable examples (offline stub by default; `HIMMY_EXAMPLE_PROVIDER=ollama` for a real
  model) — a tool-using agent, a live web-research agent, and durable semantic memory.
  Verified end-to-end on Ollama `qwen2.5`: the agent calls `calculator`/`web_search`, the
  runtime executes it against the real tool/web, and the model answers. `RECIPES.md`
  documents how to run them and the findings (incl. that small models reliably call tools
  but may not hand off in a team — a model-capability limit, not a framework one).
- **Real tool-calling on the local providers.** `OllamaClientManager` now sends bound
  tools via Ollama's native `/api/chat` `tools` schema and parses `message.tool_calls`;
  `ClaudeCliClientManager` (text-only) drives a best-effort ReAct protocol (appends a tool
  manifest, parses a `TOOL_CALL <name> <json>` line). Both also **execute** the bound
  handlers and populate `tool_returns` — the runtime only replays call/return pairs onto
  the thread, so without execution the model never saw a result and looped. Previously
  these managers returned only text and never populated `tool_calls`, so the toolkit /
  teams / memory / guardrail tool-hook silently no-op'd on local models; now an agent
  actually calls a tool, gets the result, and answers off-cloud (verified end-to-end on a
  real Ollama qwen2.5 model). No new dependencies; offline-tested via injected
  transport/runner.
- **Guardrails across three surfaces (`himmy.services.guardrails`).** Composable,
  dependency-free `Guardrail`s — `PIIGuardrail` (redacts emails/phones/cards/SSNs/keys),
  `InjectionGuardrail` (flags/blocks prompt-injection phrasings), `BlocklistGuardrail` —
  combine in a `GuardrailPipeline`. They guard all three surfaces: **tool arguments**
  (via `build_guardrail_pre_hook` → the existing `ToolService` pre-hook: redacts or
  blocks before a tool runs), the **input prompt**, and the **model's output** (optional
  `input_guardrail`/`output_guardrail` on `SingleAgentRuntime`, forwarded by
  `build_runtime`; default `None` → no behavior change). Enable per agent with
  `guardrails: [pii, injection]` in `agent.yaml`; `himmy doctor` lists them. No new deps.
- **Agent evaluation harness (`himmy.services.evaluation.agent_harness`).** An
  `AgentEvalHarness` runs an agent or a team over an `EvaluationSuite` (executing each
  case's input through `run_task_detailed` / `MultiAgentOrchestrator.run`) and scores the
  outputs via the existing `EvaluationService` — deterministic metrics offline, LLM-judge
  + embedding-similarity when a provider/embedder is configured. Suites are declarative
  (`suite.yaml` → `himmy.config.load_eval_suite`) and runnable via `himmy eval -f
  suite.yaml --agent agent.yaml` (or `--team team.yaml`), printing a per-case PASS/FAIL
  scorecard. No new dependencies.
- **Token streaming + crisp agent-loop termination.** `SingleAgentRuntime.stream_task`
  streams a reply as `StreamDelta` chunks (over `InferenceService.run_stream`); `himmy
  chat` now streams to stdout token-by-token and `himmy run --stream` opts in. A
  registerable `final_answer` tool lets an agent end its loop explicitly (the
  single-agent loop and the multi-agent orchestrator stop with
  `stopped_reason="final_answer"`), and a no-progress guard halts a loop that repeats the
  same tool call (`stopped_reason="no_progress"`) instead of spinning to `max_turns`.
  Team members that hold tools auto-get `final_answer`. No new dependencies.
- **Real local embeddings (selectable; stub stays the default).** New `OllamaEmbedder`
  (local Ollama `/api/embeddings` over httpx — zero new deps, keyless) and
  `FastEmbedEmbedder` (self-contained ONNX via the new `embeddings` extra), plus a
  `build_embedder(name, …)` factory covering `deterministic|ollama|fastembed|openai`. The
  `knowledge` and `memory` packs pick the embedder from `ToolkitConfig`
  (`HIMMY_EMBEDDER`/`HIMMY_EMBEDDER_MODEL`/`HIMMY_EMBEDDER_DIM`/`HIMMY_OLLAMA_URL`) and
  thread its real dimension into `create_kb(vector_dim=…)`, so semantic recall no longer
  needs exact word overlap. The offline `DeterministicEmbedder` remains the default.
- **CSV/Excel document readers.** `CsvReader` (`.csv`, stdlib) and `ExcelReader`
  (`.xlsx`/`.xlsm`, via the existing `connectors` openpyxl) join the
  `DocumentReaderFactory`, so `read_document` and `kb_ingest` now cover spreadsheets
  alongside PDF/text/Markdown.
- **Durable knowledge via pgvector (toolkit).** Setting `HIMMY_KB_DSN` makes the
  `knowledge` pack's `kb_ingest`/`kb_search` persist across processes on Postgres +
  pgvector (resolved by a fixed KB name); the in-process KB remains the default. Uses
  the existing `postgres`/`knowledge` extras.
- **Long-term memory module (`himmy.services.memory`) + `memory` toolkit pack.** A
  `MemoryService` persists facts (`remember`) and recalls them semantically (`recall`,
  cosine over the offline embedder). Storage is pluggable: `InMemoryMemoryStore` or a
  durable `SqliteMemoryStore` (stdlib sqlite3) so memory survives across processes. A
  `MemoryContextAdapter` auto-injects recalled memories into prompts when registered on
  a `ContextService`. The `memory` toolkit pack exposes `remember`/`recall` to agents
  (subject + sqlite path from `HIMMY_MEMORY_SUBJECT`/`HIMMY_MEMORY_PATH`); the catalog is
  now ten packs. No new dependencies.
- **Multi-agent teams: handoff + supervisor delegation (`himmy.orchestrators`).** A
  `MultiAgentOrchestrator` routes a task across an `AgentTeam` of `TeamMember`s over a
  shared thread. Two collaboration edges: **handoff** (`handoffs`) binds a synthetic
  `transfer_to_<peer>` tool and the orchestrator transfers control when the model calls
  it (detected from `RunResult.tool_calls`); **delegation** (`delegates`) binds an
  `ask_<worker>` tool whose handler runs the worker to completion in its own sub-thread
  and returns its answer (control stays with the manager). New `AGENT_HANDOFF` /
  `AGENT_DELEGATED` events; a public `SingleAgentRuntime.continue_turn` exposes the
  runtime's own continuation step (no new user prompt) that the orchestrator drives.
  Teams are declarative (`team.yaml` → `himmy.config.TeamSpec`/`build_team`) and runnable
  via `himmy team -f team.yaml -p ...` (`himmy init --team` scaffolds one); the CLI prints
  the routing trail + final answer. No new dependencies.
- **Toolkit wave 2: +8 tools in 4 packs (`himmy.toolkit`).** Closes the recall/act/
  source gaps on top of the original five packs. `knowledge` (`kb_ingest`, `kb_search`)
  wraps the existing `KnowledgeBase`/RAG so an agent can build and semantically search
  its own memory (in-process `DeterministicEmbedder`). `documents` (`read_document`)
  extracts text from PDF/text/Markdown via the existing `DocumentReaderFactory`,
  root-jailed, feeding straight into `kb_ingest`. `comms` (`send_email` via stdlib
  `smtplib`, `send_webhook` SSRF-guarded) is the outbound "act" capability —
  approval-gated by default unless `HIMMY_COMMS_ALLOW_SEND=1`, creds read only from env.
  `data-sources` (`weather` via Open-Meteo, `geocode` via Nominatim, `wikipedia`) are
  keyless public connectors. All network behind injectable seams (offline tests). The
  catalog is now nine packs / 18 tools (`himmy tools`).
- **Built-in toolkit: 10 general-purpose tools in 5 packs (`himmy.toolkit`).** Agents
  now ship with batteries included — `web` (`web_search` keyless-DuckDuckGo by default
  / Tavily / Brave, `web_fetch`, `http_request`), `files` (`read_file`, `write_file`,
  `list_dir`, jailed to a sandbox root), `data` (`sql_query`, read-only over SQLite or
  Postgres), `code` (`run_python` via the existing sandbox, approval-gated), and `utils`
  (`calculator`, `current_time`). A `ToolPack` catalog (`register_packs`/`BUILTIN_PACKS`)
  resolves packs by name so an `agent.yaml` can say `tool_packs: [web, utils]`; `himmy
  tools` lists them. Security: model-supplied URLs pass an SSRF guard (no private/
  loopback hosts, http/https only, no embedded creds, redirects off), files are
  root-jailed (traversal + symlink-escape rejected), SQL is read-only-enforced (SQLite
  authorizer + single-statement), and write/exec tools are approval-gated. Network I/O
  is behind injectable seams so the suite runs fully offline. New optional extra
  `toolkit` (`beautifulsoup4`) improves `web_fetch` extraction; defaults need no new
  core deps. Config via `HIMMY_*` env vars (`himmy.toolkit.ToolkitConfig`).
- **`himmy` command-line interface + declarative agents.** A console-script entry
  point (`himmy`, also `python -m himmy`) with `run`, `chat`, `init`, `serve`, and
  `doctor` subcommands, so an agent can go from install to running without writing
  wiring code. Agents are described in an `agent.yaml` (`himmy.config.AgentSpec` /
  `load_agent_spec`) that maps onto `Persona` + `Task` + `LLMConfig`; `himmy init`
  scaffolds the spec plus an example `tools.py`. Provider is selectable per run
  (`--provider stub|claude-cli|ollama|pydantic-ai`, `--model ...`), defaulting to the
  existing pydantic-ai→stub auto-select. No new core dependencies (stdlib `argparse`
  + the existing `pyyaml`).
- **Nepali-language RAG (`himmy.nepal.language`).** Cross-script retrieval:
  `transliterate` (Devanagari→Roman), `normalize_nepali` (script-folds so
  `नेपाल`/`Nepal`/`nepal` all become `nepal`), and `NepaliEmbedder` (wraps any
  embedder, embedding the folded form) — so a query in one script finds documents
  in another (verified both directions over the offline DeterministicEmbedder).
- **Local-model client managers + cost-aware routing.** `OllamaClientManager`
  (local Ollama `/api/chat`), `ClaudeCliClientManager` (the local `claude` CLI —
  not HTTP), and `HimalayaGptClientManager` (self-hosted Transformers) — each with
  an injectable transport/runner/generate-fn (offline-testable) and `$0` cost.
  `RoutingClientManager.cost_ordered` tries cheapest routes first (free local
  before paid cloud, cloud only on failover). (OpenAI/Claude/Gemini/OpenAI-compatible
  already work via `PydanticAIClientManager`.)
- **Bikram Sambat calendar (`himmy.nepal`).** A typed BS calendar on the
  authoritative `nepali-datetime` library (`nepal` extra): `BikramDate` (AD<->BS,
  today, month/weekday names in English or Devanagari) and the Nepal fiscal year
  (starts Shrawan 1).
- **Human-in-the-loop pause/resume.** Tool approval is no longer a terminal deny:
  `run_agent_loop(..., hitl=True)` PAUSES when the model calls a tool that requires
  approval — persisting a durable `AgentCheckpoint` (full thread + persona/task/
  context + the pending tool call + loop limits) via a `CheckpointStore`
  (`InMemoryCheckpointStore` or durable `SqliteCheckpointStore`), emitting
  `APPROVAL_REQUIRED`, and returning `stopped_reason="awaiting_approval"` + a
  `checkpoint_id`. `resume_agent_loop(checkpoint_id, approved=…)` rehydrates it and
  either executes the approved tool (recording the real result) or records the
  rejection, then continues the loop. Idempotent (a resolved checkpoint can't be
  resumed twice) and **survives a process restart** (a fresh runtime resumes a
  SQLite checkpoint). New `APPROVAL_REQUIRED`/`GRANTED`/`REJECTED` events.
- **Runtime-owned agentic loop (`SingleAgentRuntime.run_agent_loop`).** A bounded
  act→observe→re-invoke loop: while a turn calls tools, the runtime feeds the
  updated thread back for another model turn — until the model answers with no tool
  calls (`final`), or `max_turns` / `cost_budget` is hit, or a turn FAILS. Returns a
  typed `AgentLoopResult` (every turn, totals, `stopped_reason`) and emits
  `AGENT_TURN_STARTED`/`AGENT_TURN_COMPLETED` per turn. Additive — the single-turn
  `run_task` path is unchanged.
- **Nepal data connectors (`himmy.connectors`).** Independent, fetch-direct (no
  intermediary). **News:** a curated set of ~20 Nepali outlet RSS feeds via
  `NewsFetcher`, plus a standalone **MCP server** (`python -m
  himmy.connectors.news_mcp_server`) exposing `list_sources`/`fetch_news`/
  `search_news` to any MCP client. **NRB:** `NRBClient.forex` (the public
  forex JSON API), `list_macro_reports` (monthly Current Macroeconomic & Financial
  Situation reports via the category feed, with language/period parsed), and
  `fetch_latest_macro_workbook`/`fetch_macro_workbook`/`parse_workbook` — the macro
  Excel is auto-discovered + fetched (NRB's 'Tables' report URLs serve the .xlsx
  directly; no URL needed for the latest), then every sheet parsed with openpyxl
  (live: ~93 sheets of CPI/WPI/GDP data). `register_nrb_tools` wires NRB into the
  ToolService. All behind an
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
