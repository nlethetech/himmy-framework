# Changelog

All notable changes to Himmy are documented here. This project keeps an
**offline-first** invariant: every release runs end-to-end with no network and no
keys via the deterministic `StubClientManager`; optional extras layer in real
providers, Postgres/pgvector, and observability.

## [Unreleased]

### Added
- **RAG quality, measured on real data (not toy queries).** Two complementary evals:
  a fast **regression gate** (`tests/integration/test_rag_eval.py`, `-m integration`) on a
  hand-built confusable corpus with floors derived from live qwen3-embedding numbers; and
  a re-runnable **real-benchmark** script (`scripts/rag_nfcorpus_benchmark.py`) that
  downloads/caches the **BEIR NFCorpus** medical-IR dataset (real queries + human qrels)
  and reports recall@k / MRR / nDCG@k / hit-rate for dense vs hybrid. Measured (20 queries,
  439-doc subsample, top_k=10): qwen3-embedding dense **nDCG@10 0.279** (hybrid 0.289) vs the
  offline deterministic embedder's **0.082** (hybrid 0.174) — the real embedder is ~3.4x
  better on a real benchmark, and the hybrid BM25 leg's lift is largest exactly when the
  embedder is weak (+0.092 nDCG) and small when it is strong (+0.010). nDCG@10 ≈ 0.28 is in
  line with the published BEIR literature for NFCorpus. The benchmark also measures the
  opt-in **cross-encoder reranker** (`hybrid+rerank`, the `[embeddings]` extra) when
  installed: on NFCorpus it is the biggest precision lever (nDCG@10 0.285 → 0.373, +31%),
  but the *default* general MS-MARCO MiniLM reranker trades away recall/coverage on
  out-of-domain medical text (recall@10 0.234 → 0.095, hit-rate 0.70 → 0.40) — a measured
  "use a domain-appropriate reranker and verify on your corpus before enabling it" finding.
- **Institutional-grade upgrade pass (7 workstreams).** Each landed in its own
  file-ownership cluster, was adversarially verified, and ships with deterministic offline
  tests; the offline-first invariant holds throughout.
  - **Real relational / graph memory.** New `MemoryLink` (typed relations) +
    `MemoryService.traverse_graph(seed_id, max_depth=…)` deterministic bounded BFS honouring
    `as_of` / tier / subject scoping, returning a `MemoryGraph`. Links persist in both the
    in-memory and SQLite stores (additive, idempotent migration). `recall` /
    `MemoryContextAdapter` gain an optional `max_hops` (off by default — single-hop output is
    byte-for-byte unchanged). The "graph memory" claim is now genuinely true.
  - **Local semantic embedding by default, via Ollama.** The toolkit embedder now defaults
    to `"auto"`: `fastembed` → a local **Ollama with the embed model actually pulled**
    (`HIMMY_OLLAMA_EMBED_MODEL`, default `qwen3-embedding`, 4096-d; override dim with
    `HIMMY_OLLAMA_EMBED_DIM`) → the offline `DeterministicEmbedder`. The Ollama leg is gated
    on `ollama_embed_model_available()` (probes `/api/tags`) so a reachable-but-embed-less
    server degrades to deterministic instead of 404-ing. `himmy doctor` reports the active
    backend + how to enable a better one. New live integration tests
    (`tests/integration/test_ollama_live.py`, `-m integration`) exercise real
    `qwen3-embedding` semantic ranking, end-to-end knowledge + memory recall, and a live
    `qwen2.5` LLM agent turn. The unit suite stays hermetic via an autouse fixture that pins
    non-integration tests to the deterministic backend.
  - **Durable storage by default for servers.** New `StoreFactory` + `SqliteStorageService`
    (WAL, `asyncio.to_thread`). `himmy serve` / `himmy studio` / the BFF default to durable
    SQLite at `HIMMY_STORE_PATH` (default `.himmy/storage.db`), or Postgres when
    `HIMMY_DATABASE_URL` is a `postgres://` DSN; one-shot `himmy run` / `himmy chat` stay
    zero-setup in-memory. `ApiContainer.build_default()` (sync) remains in-memory for offline
    programmatic use.
  - **Token streaming through the multi-turn tool loop.** New
    `SingleAgentRuntime.stream_agent_loop(...)` async generator interleaves text deltas with
    `tool_call` / `tool_result` / `turn_end` events across every turn; `StreamDelta` gains
    optional `event_type` / `event_payload`. `run_agent_loop()` is unchanged.
  - **Direct Anthropic + OpenAI client managers.** `AnthropicClientManager` /
    `OpenAIClientManager` (lazy SDK imports via the new `[anthropic]` / `[openai]` extras)
    implement the `ClientManager` never-raises contract directly (no pydantic-ai
    indirection); `build_inference()` auto-prefers them when a matching key + SDK is present.
    Fixed the pydantic-ai `OpenAIModel` → `OpenAIChatModel` deprecation (version-guarded).
  - **Cloud KMS for field encryption.** Pluggable `KekProvider` abstraction
    (`LocalKekProvider` default, no new dep) + `AwsKmsKekProvider` (the `[kms-aws]` extra)
    + key rotation (`FieldEncryptor.rotate_kek`, `SubjectKeyVault.rotate_subject_key`) that
    re-wraps DEKs without touching ciphertext; GCP/Azure are documented seams. The token
    gains a key-version segment and **legacy version-less ciphertext still decrypts
    transparently** (verified against a real pre-change token — no migration required).
  - **Load / concurrency profiling harness** (`tests/load/`, `-m slow`, `make test-load`):
    concurrent BFF run/lineage/list load, cProfile hot-path profiling, and an optional
    pgvector latency bench (skipped without Postgres) — all deterministic and offline.
- **Himmy Studio — a local web GUI (`himmy studio`).** A no-code front door served by
  the same FastAPI BFF (loopback-bound): **Chat** with an agent (live SSE token
  streaming, multi-turn, tool calls), an **Agent builder** that reads/validates/saves
  an `agent.yaml` from a form (skill/tool-pack/guardrail pickers; saves merge so
  advanced fields like `http_tools`/`mcp_servers` survive an edit; a 409 guard stops a
  new agent silently overwriting an existing file), a **Runs** browser (every run
  persisted to `.himmy/studio.db` with transcript + step-by-step timeline), and a JSON
  **Doctor**. The React/Vite frontend lives in `studio/` and builds into the package
  (`himmy/api/_studio_static`, shipped as package data); `pip install 'himmy[studio]'`.
  The spec→runtime wiring is now shared between the CLI and the API via
  `himmy.runtime.from_spec`, so both wire agents identically. Verified live: real token
  streaming on Ollama, and the builder create/edit/overwrite-guard round-trip in a browser.
- **Automatic model pricing (OpenAI + Anthropic), kept current via the community source.**
  Token usage was always captured; now dollar cost is too, with no hand-configured price
  table. `himmy.services.inference.pricing` resolves a per-model price from a layered
  table — explicit override > `himmy prices sync` file > bundled offline snapshot >
  unpriced ($0, never a guessed number). `himmy prices sync` downloads the
  ecosystem-standard, continuously-maintained LiteLLM price JSON (2000+ models incl. brand
  new ones) to `~/.himmy/model_prices.json`, so prices stay current without upgrading
  himmy; `HIMMY_MODEL_PRICES` points at a custom file (LiteLLM flat shape drops straight
  in). `himmy prices show <model>` / `himmy prices` inspect the table; name lookup strips
  a `provider:` prefix and `-YYYY-MM-DD`/`-latest` suffix. The pydantic-ai manager
  (OpenAI/Anthropic API) now fills `InferenceResponse.cost` from this table automatically.
- **Deterministic record-and-replay of agent runs.** `himmy run --record FILE` captures
  every model response to a portable cassette; `himmy run --replay FILE` re-runs the agent
  exactly from it — no provider, no network, no tool side effects — so debugging a failure
  becomes "step through the exact trace" instead of "rerun and hope." Matching is by the
  request's content hash (`compute_cache_key`, which excludes the random `request_id`), so
  a re-run replays identically; duplicate/retry calls replay FIFO; a miss is a strict
  `ReplayError`. Verified live: a recorded multi-turn Ollama run replays byte-identically
  with the provider unreachable.
- **Automatic context compaction.** Long multi-turn runs that would overflow the context
  window now summarize their oldest turns in place once the thread crosses a token budget
  (`compact_context: true`, `compact_after_tokens`, `compact_keep_recent` in `agent.yaml`).
  Invariants: never touch the system head, always keep the recent tail verbatim, and never
  split a tool_call from its tool_return (the boundary snaps back); the summary is only
  applied if it actually shrinks the context, and a `CONTEXT_COMPACTED` event records what
  was condensed. Verified live on Ollama: a budgeted multi-turn run compacts mid-loop and
  still answers correctly.
- **Skills — first-class agent capabilities.** A skill bundles the tools a job needs with
  the know-how to use them; declaring `skills: [data_analysis]` in `agent.yaml` binds the
  tools *and* injects the guidance, no separate `tool_packs`/instructions. Skills are
  Pydantic-validated (`extra="forbid"`), versioned entities (`kind="skill"`), compose via
  `requires_skills` (cycle-guarded), and resolve with a did-you-mean on typos. Author them
  as one-file YAML in `skills/` (auto-discovered; project files shadow built-ins), list
  them with `himmy skills`, scaffold one with `himmy init`. A skill can carry few-shot
  `examples` (rendered into the prompt) and a `when_to_use` hint (rendered as guidance and
  fed to the tool router). Built-ins: `web_research`, `data_analysis`, `file_ops`,
  `python_compute`, `knowledge_base`, `nepal_brief`, `summarize`, and the composite
  `research_writer` (requires web_research + summarize). Verified live on Ollama: a
  composite, skills-only agent binds its prerequisites' packs and a skill's worked example
  drives the small model straight to the correct query.
- **Skill dispatch, detail view, and skill-level benchmarking.** `dispatch_skill`
  (enabled by `allow_skill_dispatch: true`) runs a named capability as an isolated
  sub-agent scoped to just that skill's tools + know-how — a one-level recursion cap, like
  `spawn_agent` but capability-bound (`SkillDispatcher` is the programmatic API).
  `himmy skills <name>` shows a skill in full (version, when-to-use, tools, instructions,
  examples). Benchmark tasks can be declared by `skills:` (not just raw `packs:`), so a
  standing suite measures that a capability binds its tools and that its know-how yields
  correct answers — new built-in `skills` suite. Verified live on Ollama: a parent with no
  data tools dispatches `data_analysis` and answers correctly; the `skills` suite scores
  100% on `qwen2.5:3b-instruct`.

### Fixed
- **Tool-using agents now actually answer on real providers (`himmy run`/`chat`/`telegram`).**
  These surfaces called single-turn `run_task_detailed`, so with a real model (Ollama,
  Claude CLI) the *first* turn is the tool **call** and the run ended before the model
  saw the result — returning an empty answer. They now use the runtime-owned
  `run_agent_loop` (act → observe → answer) whenever the agent has tools, so the model
  gets the tool result and replies. The offline stub was unaffected, which masked this.
  Verified live: Ollama `qwen2.5:3b-instruct` + a real MCP filesystem server reads a file
  and answers correctly.
- **`todo_write` works on small local models.** Its schema was a nested array-of-objects
  (`[{content, status}]`), which Ollama's tool grammar / small models silently choke on
  (empty reply, no tool call). It's now a flat array of strings, with a new
  `todo_complete` tool tracking status — verified live on `qwen2.5:3b-instruct`.
- **`claude-cli` provider works from inside a Claude Code session.** The subprocess now
  strips the parent session's `CLAUDECODE`/`CLAUDE_CODE_*` env markers so a nested
  `claude -p` starts a clean session instead of erroring.
- **pydantic-ai toolset binding works on pydantic-ai ≥ 1.x.** `ToolServiceToolset.as_pydantic_ai_toolset`
  built each proxy with an inline `args: model` annotation; under PEP 563 (`from __future__
  import annotations`) that stringified to `"model"`, which pydantic-ai 1.106's `get_type_hints()`
  could no longer resolve (`NameError` at bind time). The generated arg-model class is now
  assigned directly to the proxy's `__annotations__`, so no string eval runs.
- **Postgres entity-registry metadata-filter queries actually match now.** `PostgresEntityRegistry.query()`
  pre-serialized `metadata_filters` with `json.dumps` AND the registered jsonb codec re-serialized
  it, double-encoding the parameter into a jsonb *string* (`"{...}"`) that `@>` could never match a
  jsonb *object* — so every `metadata @> $1::jsonb` query silently returned nothing. It now binds
  the dict directly (the codec encodes once). Verified end-to-end against a live `pgvector/pgvector:pg16`.
- **pgvector knowledge + Postgres-entity integration tests are re-runnable.** They drove a persistent
  asyncpg pool across multiple `asyncio.run` loops (a pool is bound to its creating loop → "operation
  in progress"), and reused a shared, un-truncated table across tests/runs. The knowledge tests now
  run each scenario in a single event loop, and `_fresh_registry()` truncates so count/containment
  assertions are isolated.

### Internal — production hardening (architecture)
- **Decomposed the `StorageService` god object into focused, single-responsibility stores —
  both backends.** The in-memory backend is a thin facade composing `InMemory{Thread,Event,
  Context,Run,Recommendation,Evaluation,Orchestration}Store`, and `PostgresStorageService` is
  likewise a facade composing `Postgres{…}Store` classes (a shared `_PgStoreBase` holds the pool
  accessor + generic SQL helpers; row mappers are module-level). Each store satisfies a focused
  protocol in `himmy.services.storage.protocols` (`ThreadStore`, `EventLog`, `ContextStore`,
  `RunStore`, `RecommendationStore`, `EvaluationStore`, `OrchestrationStore`, plus the
  runtime-facing `ThreadEventStore`). The public `StorageService`/`PostgresStorageService` APIs
  are unchanged, so the ~15 call sites are untouched. A DB-free conformance test asserts both
  facades satisfy every protocol; the Postgres split is verified against a live database.
- **Disambiguated the two `MemoryStore` protocols.** The storage-layer one (async threads +
  events) is renamed `ThreadEventStore`; the cognitive-memory one
  (`himmy.services.memory.store.MemoryStore`, sync `MemoryRecord` CRUD) keeps the name it
  actually describes.
- **Centralized entity projection.** All 12 `to_record()` methods (plus an inline projection
  in the context service) now delegate to one algorithm, `himmy.entities.project()`. Domain
  models declare only *what identifies them* (stable key, namespace, kind); the projection
  mechanism (stable-id derivation + `EntityRecord.create`) lives in one place. Adds dedicated
  projection tests (previously there were none).
- **Decoupled tool execution from the inference layer.** `BoundTool` is now pure data
  (name + schemas); execution flows through a single `ToolExecutor` callback on the
  `InferenceRequest`, provided by `ToolService.tool_executor()`. The inference layer no longer
  carries tool-layer Python callables.
- **Documented the `metadata` key vocabulary.** `himmy.core.metadata` adds `total=False`
  `TypedDict`s naming the framework-written keys (assistant message, routing, persona, context,
  knowledge-adapter, tool-event). The model fields stay open `dict[str, Any]` (extensibility is
  a feature); the TypedDicts type the write sites that opt in. (Typing the routing site surfaced
  and fixed a latent `route_label` nullability bug.)
- **Tightened mypy to near-strict.** Enabled `disallow_untyped_defs`/`_incomplete_defs`/
  `_untyped_calls`/`_untyped_decorators`, `disallow_any_generics`, `warn_return_any`,
  `strict_equality`, `extra_checks`, `no_implicit_reexport`, and `warn_unused_configs`, and
  cleared the resulting ~86 errors. `warn_unused_ignores` is intentionally left off: many
  `# type: ignore`s guard optional-extra imports and are required in the no-extras base install.

### Changed
- **Renamed the project to the Himmy Agent Framework.** The import package is now
  `himmy` (was `opensims`), the base exception is `HimmyError`, environment
  variables use the `HIMMY_` prefix (e.g. `HIMMY_DATABASE_URL`), and the
  distribution is `himmy`. Update imports: `from himmy import ...`.

### Changed
- **The offline stub now terminates faithfully.** Under `AUTO_TOOLS` the
  `StubClientManager` used to call a bound tool on *every* turn, so stub-driven agent
  loops never produced a final answer (they spun to `max_turns`) — which masked real bugs.
  It now calls a tool once and, on the next turn (seeing a `tool`-role result on the
  thread), answers with text — exactly like a real model. Single-turn runs are unchanged;
  a tool-using agent now ends with `stopped_reason="final"` in two turns on both the stub
  and Ollama. (One internal test that relied on the old "always loops" behavior was moved
  to an explicit repeat-manager.)

### Changed
- **Hardened the regex PII guardrails (still zero-dep, offline).** Detection moved to a
  validated `PIIRule` model: credit cards are only redacted when they pass the **Luhn
  checksum**, and IPv4s are octet-validated — cutting the regex's biggest weakness, false
  positives. Coverage expanded to API keys (OpenAI/GitHub/Google/Slack/AWS), JWTs, URLs
  with embedded credentials, IBANs, IPv4/IPv6, and MAC addresses. The Nepal `nepal_pii`
  guardrail's over-greedy PAN rule (which redacted *any* 9-digit number) now requires a
  `PAN` label, and a domestic 10-digit mobile pattern was added. `PIIGuardrail(rules=…)`
  accepts custom rules. (A heavier ML detector, e.g. openai/privacy-filter or Presidio,
  remains a clean opt-in via the `Guardrail` protocol when more is needed.)

### Added
- **Telegram — himmy agents live in Telegram.** A new `telegram` tool pack adds
  `send_telegram` (message a chat via a bot, approval-gated like `comms`), and a new
  `himmy telegram` command runs an `agent.yaml` as a live bot: long-poll for messages,
  run the agent (one conversation thread per chat), reply. Token + default chat from
  `HIMMY_TELEGRAM_BOT_TOKEN` / `HIMMY_TELEGRAM_CHAT_ID` (or `--token`), never the model.
  New `himmy.toolkit.telegram` (`TelegramClient`, `TelegramBot`, `register_telegram_pack`);
  the bot loop is injectable so it's fully tested offline. `httpx` only, no SDK. Works
  with `mcp_servers`/`allow_spawn` too (they stay connected for the session).
- **`spawn_agent` — ad-hoc recursive sub-agents.** Set `allow_spawn: true` in `agent.yaml`
  and the agent gets a `spawn_agent` tool: hand a sub-task to a fresh single-agent (its own
  `instructions`/`prompt`, optionally its own `tool_packs`) that runs to completion and
  returns its answer. Unlike team delegation it needs no `team.yaml` — the parent decides
  at run time. The sub-agent shares the parent's inference but runs in a fresh runtime with
  no `spawn_agent` tool, so recursion is capped at one level; unknown `tool_packs` are
  reported, not fatal. New `himmy.toolkit.spawn.register_spawn_tool`.
- **`agentic` pack — the tools that make an agent act like one.** `ask_human` (pause
  mid-run and ask the operator; returns `answered: false` when non-interactive instead of
  hanging), `scratchpad_set`/`scratchpad_get` (a keyed working-memory notepad the agent
  manages itself, distinct from durable `memory`), and `todo_write`/`todo_read` (a
  self-managed task list — write the whole list with per-item status, read it back).
  Working state lives for the process (persists across `himmy chat` turns, resets between
  `himmy run` calls). `set_human_responder()` lets you embed himmy in another UI or test.
- **MCP servers, declaratively (the whole MCP ecosystem as agent tools).** An
  `agent.yaml`/`team.yaml` can now list `mcp_servers:` — any stdio Model Context Protocol
  server (GitHub, Slack, filesystem, a Playwright browser, …) is launched and its tools are
  registered as native himmy tools (arg validation, approval gating, events, lineage). Each
  server supports `prefix` (namespacing), `requires_approval`, a `tools` subset, and
  `env`/`cwd`. New `himmy.config.mcp_spec` (`MCPServerConfig`, `attach_mcp_servers`,
  `close_mcp_clients`); the CLI connects them inside the run loop and tears them down after.
  Wired into `himmy run` (incl. `--plan`/`--stream`), `himmy chat`, `himmy team`, and
  `himmy eval`. Builds on the existing transport-direct MCP stdio client (no SDK).
- **Mixed-provider teams (strong brain + cheap local workers).** Each `team.yaml` member
  can declare its own `provider` + `model`, so a Claude-CLI (Opus) brain can orchestrate
  free local Ollama workers with a cloud model wherever you want one. A new
  `MultiProviderClientManager` dispatches each member's `model_key` to its own backend
  (unknown keys fall back to a default); the CLI builds it automatically via
  `build_team_inference`. Members without a provider use the CLI `--provider`/`--model`.
- **Nepal pack + localization.** A `nepal` toolkit pack makes the framework's Nepal
  modules agent-facing: `nepali_date` (AD↔Bikram Sambat + fiscal year), `nepali_format`
  (NPR with Nepali lakh-grouping + Devanagari numerals — `रू १२,३४,५६७.५०`),
  `nepali_transliterate` (Devanagari↔Roman), and NRB forex tools. A `nepal_pii` guardrail
  redacts `+977` phones, citizenship, and PAN numbers; `language: ne` on an `agent.yaml`
  instructs the agent to answer in Nepali (Devanagari); and `build_embedder("nepali")`
  selects the cross-script `NepaliEmbedder` for memory/RAG. Calendar/NRB tools lazy-import
  their extras (`nepal`/`connectors`).
- **Auto-memory into context.** Set `memory: true` on an `agent.yaml` and the agent
  automatically recalls its most relevant long-term memories and injects them into the
  system prompt **every run, with no tool call** (`AgentSpec.memory`/`memory_top_k` wire a
  `context_build_spec` + `context_prompt_map_spec`; the CLI builds a `MemoryService` +
  `MemoryContextAdapter` over a `ContextService`). `MemoryContextAdapter` gained a pinned
  `subject_id` so it reads the same subject facts were remembered under.
- **Plan-and-execute + reflection (`himmy.orchestrators`).** `PlannerOrchestrator`
  decomposes a goal into an ordered plan (structured output, with a numbered-text
  fallback for providers that lack JSON mode), executes each step over a shared thread,
  and synthesizes a final answer — the agentic counterpart to the reactive tool loop.
  `reflect(runtime, draft, …)` is a one-turn critique-and-revise pass. CLI: `himmy run
  --plan`. Verified on Ollama qwen2.5: 4 real plan steps → synthesized answer.
- **Structured output on Ollama.** `OllamaClientManager` now sends `output_json_schema`
  via Ollama's native `format` and parses the JSON reply into `output_structured`, so
  structured output (and the planner) work on the local model, not just stub/pydantic-ai.
- **Tool-error visibility.** When a tool fails or is denied, the runtime now writes an
  `ERROR: <code>: <message>` `TOOL` message (instead of a bare `null`), so the model can
  see what went wrong and adapt on the next turn.
- **Project config (`himmy.toml`) + chat session persistence.** A `himmy.toml`
  (`himmy.config.load_project_config`; cwd or `~/.himmy/config.toml`) sets per-project
  `[defaults]` (provider, model, tool_packs, guardrails) and `[toolkit]` (embedder,
  memory_path…) so you don't pass flags or export many `HIMMY_*` vars — precedence is
  **CLI flag > env > himmy.toml > built-in** (`ToolkitConfig.from_sources` overlays the
  toml under the env). `himmy init` scaffolds a starter `himmy.toml` and `himmy doctor`
  shows the resolved source. `himmy chat --session <id>` persists the conversation to
  `.himmy/sessions.db` (`SqliteSessionStore`) and resumes it next invocation. No new deps
  (stdlib `tomllib`/`sqlite3`).
- **`himmy trace` run inspector (`himmy.services.observability.trace`).** Real agent runs
  are now debuggable: `himmy run --trace` prints and saves a chronological, indented event
  timeline (run → inference → tool call/return → handoff/delegate, with latencies + a
  cost/tool-count footer), and `himmy trace [thread_id]` lists recent runs or replays a
  saved one. `format_timeline` renders any `RunEvent` list; `SqliteEventStore` (stdlib
  sqlite3) is a durable event log (`.himmy/trace.db`) usable as a runtime event sink.
  `build_runtime` now forwards an `on_event` override. No new dependencies.
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
