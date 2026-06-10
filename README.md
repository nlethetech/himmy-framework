# Himmy

**Build AI agents that run entirely on your own machine — no API keys, no cloud, every
run auditable and replayable.**

Himmy is an **offline-first** Python agent framework. The default install pulls only three
packages (`pydantic`, `pyyaml`, `httpx`) and needs zero network and zero keys: agents run
against a deterministic stub out of the box. Real local models run through **Ollama** or the
**Claude CLI**; cloud providers (OpenAI / Anthropic / gateways, via `pydantic-ai`) are an
opt-in extra, never a requirement.

Under the hood, personas, prompts, tools, threads, messages, context snapshots, and run
events can be projected to immutable, versioned `EntityRecord`s with typed links — an
**append-only audit log + versioned `EntityRecord` snapshots + content-hashed record/replay**
spine. State is reconstructed from the latest versioned snapshot (not by folding events).

> **Pre-1.0, source-available, single-maintainer.** Version `0.1.0`. The offline build is
> reproducible and the bulk of the framework runs with no keys/network. There are **no**
> third-party certifications (no SOC2, etc.). This is **not** an autonomous coding agent —
> it does not edit your repo or run SWE-agent-style loops.

Requires Python **3.12+**.

---

## See it in 30 seconds — fully offline, zero keys

The bare install runs against a deterministic stub, so this works with no provider and no
network:

```bash
pip install -e .
himmy init my-agent
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence."
```

The stub produces canned, deterministic output — useful for wiring things up and for tests.
When stdout is a TTY, the CLI prints a hint telling you how to add a real model (Ollama /
Claude CLI / a cloud key). Add one and re-run:

```bash
ollama pull qwen2.5:3b-instruct
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence." \
  --provider ollama --model qwen2.5:3b-instruct
```

Want to talk to your own documents offline? See the runnable
[`examples/local-doc-chat/`](examples/local-doc-chat/) (local hashing embedder + Ollama).

---

## Why himmy

Each claim below distinguishes the **zero-config offline default** from what needs an
optional extra or explicit config — because the default really is offline, and we'd rather
be precise than impressive.

- **Offline-first by default.** A bare install runs every core capability — agents, tools,
  skills, memory, RAG, orchestration, typed agents, multi-agent — against the deterministic
  `StubClientManager` and a deterministic embedder, with no keys and no network. Real model
  quality, real semantic search, and durable storage are explicit opt-ins (see below).
- **Auditable + replayable.** Record a run to a cassette, then replay it from the **recorded
  model responses with the provider turned off**. Replay returns recorded responses verbatim
  by content-hash; tools are **not** re-executed, so replay has no side effects and is fully
  deterministic. Tamper-evidence is available via content-hashed audit bundles (HMAC, or
  Ed25519 with the `cryptography` library).
- **Batteries included.** Declarative agents in YAML, a skills layer, **17 built-in tool
  packs**, MCP servers, temporal/graph memory, hybrid RAG, graph orchestration, typed
  agents, and multi-agent teams — all with an offline default path.
- **Honest about quality.** A standing benchmark runs in CI against a real local model
  (Ollama `qwen2.5:3b-instruct`) post-merge/nightly, gated by a `--fail-under` *floor*
  (not a claimed score), so "did my change make agents better or worse?" is answerable.

### What's offline-default vs opt-in (read this)

| Capability | Zero-config offline default | Opt-in (extra / config / keys) |
|---|---|---|
| Inference | `StubClientManager` (deterministic, $0) | Ollama, Claude CLI (local); `pydantic-ai` cloud/gateway (`providers` extra + key) |
| Embeddings / recall | `DeterministicEmbedder` (sha256 hashing-trick, dim 64 — reflects *lexical* overlap, not real semantics) | `fastembed` (`embeddings` extra), Ollama, or OpenAI (`knowledge` extra) embedders |
| RAG retrieval | Dense cosine (reproduces the pre-hybrid path byte-for-byte) | Hybrid BM25+dense RRF (offline, no deps); cross-encoder rerank (`embeddings` extra); LLM query-rewrite (needs a model) |
| Audit/lineage spine | In-memory registry + storage (**volatile, lost on restart**) | SQLite or Postgres registry/storage (passed explicitly) |
| Memory store | In-memory (volatile) | `SqliteMemoryStore(path)` for durability across restarts |
| Memory spine projection + events | **Off** (plain save/recall, no spine records/events) | Wire a registry + event sink to project facts and emit `MEMORY_*` events |
| Auth / RBAC / tenant isolation / rate limit / security audit | **Off** (anonymous, all-tenants) | OIDC (`auth` extra) or API keys; RBAC/rate-limit/audit enforce only once auth is on |
| Code sandbox | `subprocess` (resource isolation, **not** a hostile-code boundary) | `container` mode (Docker/Podman, `HIMMY_CODE_EXEC=container`) |
| Field encryption at rest | **Off** (plaintext) | `HIMMY_ENCRYPTION_KEY` + `cryptography` (envelope AES-GCM) |
| Web search | Keyless DuckDuckGo HTML scrape | Tavily / Brave (`HIMMY_SEARCH_API_KEY`) |
| FastAPI BFF / Studio GUI | Not installed | `api` / `studio` extras |

---

# Three concepts, start to finish

You need three things to ship an agent: **define it**, **run it**, and **extend it** with
tools, knowledge, and skills.

## 1. Define it — `agent.yaml`

An agent is a small YAML file. Scaffold one (with an example `tools.py`, `himmy.toml`, and a
starter skill):

```bash
himmy init my-agent
```

`himmy init` (no flags) writes four files: `agent.yaml`, `tools.py`, `himmy.toml`, and
`skills/my_skill.yaml`. Use `--force` to overwrite, or start from a working template:

```bash
himmy init my-agent --template helpdesk    # docs-grounded Q&A (needs the [knowledge] extra)
himmy init my-agent --template analyst     # live REST lookups via http_tools (frankfurter.dev)
himmy init my-agent --template researcher  # web research (skills: [web_research])
himmy init my-team   --team               # scaffold a team.yaml only
```

```yaml
# my-agent/agent.yaml
name: market-analyst
description: A market research analyst specializing in tech.
role: Research Analyst
instructions:
  - Provide actionable insights backed by clear reasoning.
model: default
# provider: ollama   # stub | claude-cli | ollama | pydantic-ai | openrouter (default: auto)
```

`agent.yaml` is a thin declarative façade over `Persona` + `Task` + `LLMConfig`.

## 2. Run it

Everything below runs **offline against the deterministic stub** by default — no keys
needed. Add a real model with `--provider ollama` (or `claude-cli`, or a cloud key).

```bash
# One-shot: run a prompt and print the answer
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence."

# Same, as JSON (status/output/cost/tokens/model_path/provider/latency/error)
himmy run -f my-agent/agent.yaml -p "…" --json

# Stream tokens, or write an audited trace to .himmy/trace.db
himmy run -f my-agent/agent.yaml -p "…" --stream
himmy run -f my-agent/agent.yaml -p "…" --trace

# Interactive chat that keeps a single thread; --session persists/resumes it
himmy chat -f my-agent/agent.yaml --session my-session

# What providers/keys/extras are available, and my next step?
himmy doctor
```

The smallest possible run, in pure Python:

```python
import asyncio

from himmy import Persona, Task
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference import InferenceService, StubClientManager


async def main() -> None:
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
    )
    persona = Persona(name="analyst", description="A concise research analyst.")
    task = Task(title="ping", prompt="Say hello in one sentence.")
    thread = await runtime.run_task(persona, task)
    print(thread.last_message.content)


asyncio.run(main())
```

> The tool loop's `max_turns` defaults to 8 inside the runtime and is **not** exposed as a
> CLI flag. `himmy chat` has no `--stream` flag — no-tool agents stream in the REPL
> implicitly; tool-using agents run the full loop and print the answer. REPL commands:
> `/exit`, `/quit`, `/reset`, `/help`.

## 3. Extend it — tools, knowledge, and skills

Three words, one ladder:

```
tool        a single function the model can call          web_search
tool_pack   a named bundle of related tools               web → web_search, web_fetch, http_request
skill       a tool_pack + the know-how to use it well     web_research → the web tools + how-to + examples
```

**Rule of thumb:** reach for a **skill** first — you get the tools *and* the guidance. Drop
to **`tool_packs`** for the raw tools without the know-how. Name individual **`tools`** to
bind a subset.

```yaml
# capabilities, the easy way: tools + know-how in one line
skills: [web_research]

# or wire raw tool packs yourself
tool_packs: [web, utils]            # register these built-in packs
tools: [web_search, web_fetch]      # bind a subset to the model (omit = all)

# ground the agent in your own docs — auto-ingested into a local KB → kb_search
knowledge: [./docs]
```

List what's available: `himmy tools` (packs) and `himmy skills` (capabilities, built-in +
project-local).

### The full pack table — 17 built-in packs, 47 catalogued tools

| Pack | Tools | Notes |
|---|---|---|
| `web` | `web_search`, `web_fetch`, `http_request` | search defaults to **keyless DuckDuckGo**; Tavily/Brave need a key. `web_fetch` uses BeautifulSoup if the `toolkit` extra is present, else stdlib parsing. `http_request` is a generic GET/POST/… escape hatch (not claimed read-only; `headers` marked sensitive) |
| `files` | `read_file`, `write_file`, `list_dir` | `write_file` is **approval-gated** unless `HIMMY_FS_ALLOW_WRITE` |
| `data` | `sql_query`, `sql_schema` | SQLite path needs no extra; Postgres (`HIMMY_SQL_DSN`) needs the `postgres` extra |
| `code` | `run_python` | **always approval-gated**; runs in the sandbox (default `subprocess`) |
| `utils` | `calculator`, `current_time` | offline |
| `knowledge` | `kb_ingest`, `kb_search` | in-process per-run by default; durable KB needs `HIMMY_KB_DSN` + `postgres` extra (pgvector) |
| `documents` | `read_document` | `.txt/.md/.csv` stdlib; PDF needs `knowledge` extra (pypdf); `.xlsx` needs `connectors` extra (openpyxl) |
| `comms` | `send_email`, `send_webhook` | both **approval-gated** unless `HIMMY_COMMS_ALLOW_SEND`; email needs `HIMMY_SMTP_HOST`. `send_webhook` `headers` marked sensitive |
| `data-sources` | `weather`, `geocode`, `wikipedia` | keyless public APIs |
| `news` | `news_sources`, `news_search`, `news_fetch` | needs the `connectors` extra (feedparser); `HIMMY_NEWS_FIXTURE` enables an offline local-RSS demo (still imports feedparser) |
| `memory` | `remember`, `recall` | in-process unless `HIMMY_MEMORY_PATH` (SQLite). `recall` exposes bi-temporal args (`as_of`, `active_only`, `tier`, `similarity_threshold`) |
| `notes` | `list_notes`, `read_note`, `write_note` | offline |
| `tasks` | `list_tasks`, `add_task`, `complete_task` | offline |
| `google` | `gmail_inbox`, `gmail_send`, `gcal_events`, `gcal_create` | use the connected Google account; return a friendly hint (never crash) if none connected |
| `nepal` | `nepali_date`, `nepali_format`, `nepali_transliterate`, `nrb_forex`, `nrb_macro_reports`, `nrb_macro_workbook` | `nepali_date` needs the `nepal` extra (Bikram Sambat); `nrb_macro_workbook` reads Excel (`connectors` extra) |
| `agentic` | `ask_human`, `scratchpad_set`, `scratchpad_get`, `todo_write`, `todo_complete`, `todo_read` | human-in-the-loop + self-managed task list |
| `telegram` | `send_telegram` | **approval-gated** unless `HIMMY_COMMS_ALLOW_SEND`; needs a bot token + chat id |

> **Honesty notes on the table.** The `google` pack's docstring says sending is
> "approval-gated by default," but the registrar does not pass `requires_approval` for
> `gmail_send`/`gcal_create` at the toolkit layer — don't rely on those being gated unless a
> runtime policy enforces it. A few tools live **outside** the catalog and aren't shown by
> `himmy tools`: the memory pack's `consolidate` (only when `HIMMY_MEMORY_CONSOLIDATE` is
> set), and `spawn_agent` / `dispatch_skill` (wired by the CLI/runtime, each capped to one
> level of recursion). The 47 count is the catalog total; the registered-tool count per
> agent varies with which packs and conditional tools are wired.

### Built-in skills (9)

`clarify`, `data_analysis` (binds `data`), `file_ops` (binds `files`), `knowledge_base`
(binds `knowledge`), `nepal_brief` (binds `nepal`), `python_compute` (binds `code`),
`research_writer` (composes `web_research` + `summarize`), `summarize`, `web_research`
(binds `web`).

Author your own as a one-file YAML in `skills/` (auto-discovered, non-recursive; a
project skill whose name matches a built-in **shadows** it and logs that it did):

```yaml
# skills/exact_math.yaml
name: exact_math
description: Compute exact arithmetic by running Python.
tool_packs: [code]
instructions:
  - Never estimate arithmetic — run Python and print() the result.
```

Skills are discovered from `./skills`, every entry of `HIMMY_SKILLS_PATH`, and any extra
dirs passed in code (later wins). A skill may declare `requires_skills` (depth-first
expansion, cycle-guarded), and the `Skill` model uses `extra='forbid'` so a typo'd field
fails loudly at load. Skills project to versioned `EntityRecord(kind='skill')`.

### Connect your own API, no Python

Declare REST endpoints right in `agent.yaml` and they become arg-validated, traced tools:

```yaml
http_tools:
  - name: get_order
    base_url_env_var: MYAPI_URL
    path: /orders/{order_id}
    auth: { type: bearer, env_var: MYAPI_KEY }
```

---

# Capabilities

The five capabilities below are merged on `main`. Each has a genuine **offline default**
(runs on the stub + deterministic embedder + in-memory stores, no keys/network) and clearly
labelled **opt-in** upgrades. The offline path is meant for tests and local dev; it is
**not** a high-quality semantic/LLM path.

## Temporal / graph memory

Import from `himmy.services.memory` (only `MemoryService` is re-exported at top-level
`himmy`, lazily).

- **Bi-temporal facts.** `MemoryRecord` carries a half-open validity interval
  `[valid_from, valid_to)` plus `superseded_by`, `tier`, `confidence`, `source`, and an
  optional `stable_key`. `valid_to=None` means "currently true." Temporal helpers
  `is_valid_at(record, as_of)` and `filter_as_of(records, as_of)` compare ISO-8601 strings
  lexicographically. The `recall` tool exposes `as_of` / `active_only` / `tier` /
  `similarity_threshold`.
- **Tiers.** `MEMORY_TIERS = ('core', 'recall', 'archival')` (Letta-style hot→cold);
  `promote(memory_id, tier)` moves a fact between them.
- **Invalidate-not-delete.** `invalidate(...)` stamps `valid_to`/`superseded_by` rather than
  removing rows, so history is preserved.
- **Relational / graph memory (multi-hop).** Facts can be linked with typed `MemoryLink`
  relations (`relates_to`, `caused_by`, `contradicts`, `about_entity`, `supersedes`,
  `part_of`, `depends_on`). `MemoryService.traverse_graph(seed_id, max_depth=N)` runs a
  deterministic bounded BFS that honours bi-temporal validity (`as_of`), tier, and subject
  scoping and returns a `MemoryGraph` (with a `truncated` flag). Links persist in **both**
  `InMemoryMemoryStore` and `SqliteMemoryStore` (additive, idempotent migration). `recall`
  and `MemoryContextAdapter` accept an optional `max_hops` to auto-expand hits with related
  memories — **off by default**, so single-hop behaviour is byte-for-byte unchanged.

**Offline default:** `MemoryService()` uses an **in-memory** store (volatile, lost on
restart) and the `DeterministicEmbedder` (lexical-overlap cosine, not real semantics).
Recall ranking is cosine; with **no** similarity threshold, recall **always returns the top
hit even at similarity 0.0** (a guaranteed-non-empty default) — returning *zero* hits on an
off-topic query is opt-in (set `similarity_threshold` or service `min_similarity`).

**Opt-in:**
- **Durability:** pass `SqliteMemoryStore(path)`. (It additively migrates legacy 6-column
  DBs up to the bi-temporal 7-extra-column schema, idempotently.)
- **Real semantic recall:** the embedder now defaults to **`"auto"`** —
  `fastembed` (local ONNX, the `[embeddings]` extra) → a local **Ollama** *with the
  configured embed model pulled* (`HIMMY_OLLAMA_EMBED_MODEL`, default `qwen3-embedding`,
  4096-d) → the offline `DeterministicEmbedder`. The Ollama leg is gated on the embed model
  being *present* (not just the server being reachable), so a half-ready Ollama degrades to
  deterministic instead of 404-ing at embed time. Resolution is network-free unless an
  Ollama is actually up. `himmy doctor` shows the active backend (and how to enable a better
  one); override with `HIMMY_EMBEDDER=deterministic|ollama|fastembed|openai|auto`.
- **Spine projection + audit:** with the **default** constructor (`registry=None`,
  `event_sink=None`) there are **no** spine records and **no** events. Wire a registry +
  event sink to project facts to `EntityRecord(kind='memory_fact')` and emit
  `MEMORY_REMEMBERED` / `MEMORY_RECALLED` / `MEMORY_CONSOLIDATED`.
- **Consolidation:** `MemoryConsolidator` reconciles a new fact via `ADD`/`UPDATE`/`DELETE`/
  `NOOP`. The **default is an offline deterministic similarity rule** (`>=0.95` → NOOP,
  `>=0.80` → UPDATE/supersede, else ADD). The LLM decision path is opt-in (`use_llm=True` +
  a real `ClientManager`); on the offline stub it deterministically resolves to the safe
  `NOOP`, so "LLM consolidation offline" does not make intelligent decisions. `UPDATE`/
  `DELETE` are bi-temporal (mint a new versioned fact sharing the `stable_key`, stamp the
  old, draw `supersedes` / `invalidated_by` links) — never a physical delete.
- **Auto-inject:** `MemoryContextAdapter` injects recalled memories into prompt context with
  no tool call; with tiers set, `core` is injected unconditionally and other tiers go
  through the threshold.

## Hybrid RAG

Import from `himmy.services.knowledge` (none of these are re-exported at top-level `himmy`).

- **Default is dense.** `KnowledgeBase` with no `retrieval` kwarg uses
  `DEFAULT_RETRIEVAL_CONFIG` (`mode='dense'`), reproducing the pre-hybrid cosine path
  byte-for-byte. The default chunker is the offline `SemanticChunker` (max 800 chars,
  100 overlap).
- **Hybrid is opt-in (and offline).** `RetrievalConfig(mode='hybrid')` fuses a dense leg and
  a pure-Python in-memory **Okapi BM25** index (`BM25Index`, no third-party deps) via
  **Reciprocal Rank Fusion** (`reciprocal_rank_fusion`, a pure deterministic function).
  Result chunks carry breadcrumb metadata (`dense_rank`, `lexical_rank`, `rrf_score`, …).
  The in-memory KB builds its BM25 index automatically when hybrid is selected; a pgvector
  backend needs to implement `LexicalSearchProtocol` (and an additive `tsvector`+GIN DDL),
  else hybrid degrades silently to dense-only.
- **Reranking is opt-in and needs an extra.** `FastEmbedReranker` wraps a local ONNX
  cross-encoder and requires the **`embeddings` extra** (`fastembed`); it downloads its
  model on first use. `build_reranker('fastembed')`; `fastembed_rerank_available()` probes
  importability with no side effects.
- **Query rewriting is opt-in and needs a model.** `IdentityRewriter` is the offline default
  (returns `[query]`); `MultiQueryRewriter` and `HyDERewriter` need a real `ClientManager`
  and degrade to `[query]` on failure.
- **Structure-aware chunking (opt-in):** `MarkdownAwareChunker` splits on ATX headers first
  and degrades to `SemanticChunker` on header-less docs.
- **Retrieval eval.** `evaluate_retrieval` / `compare_retrieval` run **fully offline** on the
  deterministic embedder (recall@k, precision@k, MRR, nDCG@k). Treat the numbers as
  **relative** (dense vs hybrid) and as a CI regression guard — not absolute quality
  benchmarks.

## Graph orchestration (StateGraph)

Import from `himmy.orchestrators` (not re-exported at top-level `himmy`).

- A LangGraph-style `StateGraph` with `add_node`, `add_edge`, `add_conditional_edges`,
  `set_entry_point`, `set_reducer`, and `compile(...)`. `START='__start__'`,
  `END='__end__'`. `compile()` validates topology and fails fast.
- **BSP superstep model.** Each superstep runs the current frontier of nodes concurrently
  (`asyncio.gather`) over the same pre-superstep snapshot; per-key reducers merge deltas
  (`add_reducer` concatenates lists / adds numbers; unreduced keys are last-write-wins).
  Parallel fan-out via multiple successors; join at the next superstep. Loop guards:
  per-node `max_visits` and a global `recursion_limit` (default **50 supersteps**, clamped
  to ≥1) raise `GraphRecursionError`.
- **Offline-first.** Zero new dependencies, no network, no LLM required — a graph of
  pure-Python nodes runs with no keys. Inference (if a node uses it) goes through the
  existing `InferenceService`.
- **Durable resume (opt-in store).** After each superstep a checkpoint (state + frontier +
  visit counts + status) is persisted. The **default** `InMemoryGraphCheckpointStore` is
  **volatile**; cross-process resume needs `SqliteGraphCheckpointStore`. `invoke(...)`
  returns a `GraphRunResult` with status `completed` / `interrupted` / `failed`.
  `timeout_seconds` requires **Python 3.11+** (on expiry the run is checkpointed as
  interrupted and is resumable). Audited via `GRAPH_*` events.

## Typed agents

`TypedAgent`, `TypedAgentRunResult`, and `RunContext` **are** re-exported at top-level
`himmy` (lazily).

- `TypedAgent[DepsT, OutputT]` is **additive** — a façade over the existing
  `SingleAgentRuntime.run_agent_loop`, not a separate engine. `output_type` **must** be a
  pydantic `BaseModel`.
- **Schema-from-signature.** The `@agent.tool` decorator compiles a tool's argument JSON
  Schema from the Python function signature/type hints (`compile_tool_schema`) — no
  hand-written schemas. An optional first `RunContext[DepsT]` param receives typed deps at
  call time and is hidden from the model.
- **Validated output with bounded repair.** Output is validated against `OutputT`; on failure
  the same loop retries with a corrective nudge naming the failing fields (bounded by
  `output_retries`, default 2), emitting audited `TYPED_OUTPUT_VALIDATED` /
  `TYPED_OUTPUT_REPAIRED` events.
- **Offline-first.** The stub returns a schema-valid structured result, so a typed agent runs
  end-to-end with no keys/network — though the *content* is only meaningful with a real
  provider. Tools are registered as **local read-only** on the shared `ToolService` and
  unbound in a `finally`; concurrent runs sharing one registry could in principle collide on
  tool names.

## Multi-agent: teams, group chat, fan-out

`AgentTeam`, `TeamMember`, `MultiAgentOrchestrator`, `GroupChatOrchestrator`,
`RoundRobinSelector`, `LLMSelector`, `CallableSelector`, `SubtaskSpec`, `SubtaskResult`,
`fan_out`, and `PlannerOrchestrator` **are** re-exported at top-level `himmy` (lazily);
`MultiAgentResult` / `GraphRunResult` are not.

- **Handoff vs delegation** (`MultiAgentOrchestrator`). A synthetic `transfer_to_<peer>` tool
  hands control to a peer on the **same shared thread** (re-injects the peer persona, emits
  `AGENT_HANDOFF`); a synthetic `ask_<worker>` tool **delegates** to a worker that runs to
  completion in its own sub-thread and returns its answer while control stays with the
  manager (emits `AGENT_DELEGATED`). `himmy team -f team.yaml -p "…"` runs this; non-JSON
  output prints the `route: a → b (reason)` to stderr.
- **Group chat** (`GroupChatOrchestrator`). Many members share **one audited thread**; each
  round a `SpeakerSelector` picks the next speaker (one turn each), ending on `final_answer`,
  a `terminate_when` predicate, or `max_rounds`. Selectors: `RoundRobinSelector` (default,
  deterministic), `CallableSelector(fn)`, and `LLMSelector` (a manager LLM picks via
  enum-constrained structured output — **offline it deterministically picks the first
  candidate**, and falls back to round-robin on any failure). Emits `GROUP_*` events.
- **Parallel fan-out.** `fan_out(...)` runs N typed `SubtaskSpec` workers **concurrently**
  (`asyncio.gather`), each in its own sub-thread, returning `SubtaskResult`s in **submission
  order** (deterministic join). A failing worker yields `SubtaskResult(ok=False, error=…)` —
  the join is total and never raises. `concurrency` caps simultaneous workers. Emits
  `FANOUT_*` events. Contracts are pydantic, not free text.
- **Offline caveat.** "Runs offline" here means the stub returns schema-valid deterministic
  answers; real speaker selection, multi-query/HyDE, and typed-output *quality* all need a
  real provider.

---

# Providers & choosing one

The `--provider` choices everywhere are `stub`, `claude-cli`, `ollama`, `pydantic-ai`,
`openrouter`. When unspecified, the CLI **auto-selects**: `pydantic-ai` when a key **and**
the `providers` extra **and** a model are all present, otherwise the offline deterministic
stub. (Setting only `OPENROUTER_API_KEY` does **not** auto-route to OpenRouter — pass
`--provider openrouter` explicitly.)

| Provider | What it is | Keys / deps | Cost reported |
|---|---|---|---|
| `stub` | Deterministic offline `StubClientManager`; simulates every response format, executes bound tools by synthesizing args from JSON schema | none | `$0.0` |
| `ollama` | Local Ollama over HTTP (`/api/chat`); native tool schema + structured output | local Ollama (default model `llama3.2`) | `$0.0` |
| `claude-cli` | Drives the local `claude` CLI via subprocess (not HTTP); disables the CLI's own built-in tools so it acts as a pure text model | local `claude` CLI (default model `haiku`) | real `total_cost_usd` from the CLI |
| `pydantic-ai` | Cloud / gateway via `pydantic-ai` (OpenAI, Anthropic, Gemini, OpenAI-compatible base_url) | `providers` extra + a provider key | from a pricing table |
| `openrouter` | OpenAI-compatible [OpenRouter](https://openrouter.ai) via `pydantic-ai` (`base_url=https://openrouter.ai/api/v1`); default model `mistralai/mistral-small-3.2-24b-instruct` | `providers` extra + `OPENROUTER_API_KEY` | from a pricing table |

```bash
# One-off run through OpenRouter (override the default model with --model):
export OPENROUTER_API_KEY=sk-or-...
himmy run --provider openrouter --model anthropic/claude-3.5-sonnet -p "hello"
```

There's also a self-hosted `HimalayaGptClientManager` (HF Transformers,
`HimalayaAI/HimalayaGPT-0.5B`, `trust_remote_code=True`) — real but not exercised in CI.

**Failover vs retry.** `InferenceService`'s own retry loop (`max_retries` default 2,
retryable codes only) re-hits the **same** model. Cross-provider, cost-aware failover exists
only if you explicitly wrap managers in `RoutingClientManager` (failover codes:
`PROVIDER_UNAVAILABLE`, `RATE_LIMITED`, `TIMEOUT`, `QUOTA`, `AUTH`; `cost_ordered()` tries
cheapest/local-free routes first and stamps a `fallback_chain`). It is not the default
manager. Unknown models price to `$0`.

`himmy doctor` reports your Python version, installed extras, local providers on PATH,
provider keys in env, the guardrails list, and a next-step suggestion.

---

# Audit, record/replay & tamper-evidence

The audit/lineage spine is the `EntityRecord` model plus a registry of typed links.

- **Identity vs integrity.** `EntityRecord` is immutable (`frozen=True`); its `record_id` is
  a deterministic UUID5 over **`(kind, stable_id, version)` only** — the payload/metadata are
  **not** part of identity. So the registry **alone cannot detect an in-place edited row**.
  Tamper-evidence comes separately from a **content hash** (SHA-256 over canonical JSON of
  `kind/stable_id/version/payload/metadata`, excluding `created_at`).
- **Durability is opt-in.** The zero-config registry is **in-memory and volatile** (lost on
  restart). `SqliteEntityRegistry` (durable file, stdlib `sqlite3`) and
  `PostgresEntityRegistry` (async, `postgres` extra) are API-compatible but are **not**
  auto-wired — pass them explicitly via the `registry=` override. (The CLI's
  `.himmy/trace.db` event trace is a separate concern from the entity/lineage registry; the
  default CLI lineage path stays in-memory.)
- **Audit bundles.** `export_audit_bundle(records, links, secret)` builds a Merkle root over
  content hashes and signs it with **HMAC-SHA256** (stdlib, dependency-free);
  `verify_audit_bundle` reports tampered/missing/added ids. An asymmetric **Ed25519** path
  also exists (verifier needs only the public key) and requires the `cryptography` library.
- **Record / replay.** `RecordingClientManager` appends each `(content-hash cache key,
  response)` to a JSON cassette. `ReplayClientManager` answers from the cassette by cache key
  (FIFO), returning recorded responses **verbatim with the provider off** — **tools are not
  re-executed**, so replay is side-effect-free and deterministic. Matching excludes the
  `request_id`. CLI: `himmy run --record FILE` / `--replay FILE` (mutually exclusive).
- **Trace.** `himmy run --trace` writes events to `.himmy/trace.db`; `himmy trace [thread]`
  shows a thread's timeline or lists recent runs.

---

# Himmy Studio (optional GUI)

A local web app for everything above — no terminal required. Studio needs the `studio` extra
**and** a pre-built SPA.

```bash
pip install -e ".[studio]"
cd studio && npm install && npm run build   # build the GUI once (Node 18+)
himmy studio                                 # opens http://127.0.0.1:8765
```

If the SPA isn't built, `himmy studio` prints the exact build instructions and exits 1. In
this repo the SPA is already built and committed under `himmy/api/_studio_static/`, so
`himmy studio` runs here; a fresh source checkout must build it first. For hot-reload dev:
`himmy serve` (API on `:8000`) + `cd studio && npm run dev` (UI on `:5173`, proxying
`/api` → `:8000`).

- Served by the same FastAPI BFF, **bound to loopback** (`127.0.0.1`) by default; opens a
  browser ~1s after start unless `--no-browser`.
- Screens (React Router) are grouped **WORKSPACE** (Home, Chat, Chats, Research, Approvals,
  Activity), **APPS** (Calendar, Email, Tasks, Notes, Brain, Tools, Library, Theme),
  **BUILD** (Agents, Cookbook, Models, Compare, Connections), and a collapsible **ADVANCED**
  (Teams, Workflows, Knowledge, Memory, Evaluation, Lineage, Doctor).
- Frontend: React 18, react-router-dom 6, react-markdown, built with Vite + TypeScript into
  `himmy/api/_studio_static`.

> `studio` and `api` are dependency-identical (both `fastapi` / `uvicorn` / `starlette`);
> the difference is the CLI command and the pre-built SPA. `himmy serve` defaults to port
> 8000, `himmy studio` to 8765, the Vite dev server to 5173 — don't conflate the three.
> `starlette` is pinned `>=0.49.1` for a CVE fix.

## Deploying

Past `pip install` + `himmy studio` on one machine, there are durable, multi-user shapes —
all offline-capable, secrets file-delivered, single-writer-honest:

- **Docker Compose** — [`deploy/compose/docker-compose.yml`](deploy/compose/docker-compose.yml):
  studio + Postgres (+ an optional bundled Ollama behind the `ollama` profile). `make
  compose-up` (or `make compose-up-ollama`).
- **Kubernetes** — the minimal [`deploy/helm/himmy-studio/`](deploy/helm/himmy-studio/) chart
  (single-replica by design — the `.himmy` SQLite stores are single-writer; external Postgres
  only). `make helm-lint` to validate.
- **Air-gapped** — `scripts/airgap_bundle.py` builds a no-network install bundle (images +
  wheelhouse + Ollama models); see [`docs/enterprise/airgap.md`](docs/enterprise/airgap.md).

The full runbook — configuration reference, reverse-proxy/TLS and the loopback-guard
interplay, WAL-safe backup/restore, and the upgrade procedure — is in
[`docs/enterprise/deployment.md`](docs/enterprise/deployment.md).

---

# Install & extras

```bash
# Core (offline) install — pydantic, pyyaml, httpx only.
pip install -e .

# Local API server (FastAPI + uvicorn).
pip install -e ".[api]"

# A broad bundle (see the caveat below — it is NOT literally everything).
pip install -e ".[all]"
```

There are **19 optional extras** (counting the three `secrets-*` backends separately). The
base install pulls only `pydantic`, `pyyaml`, `httpx`.

| Extra | Adds | For |
|---|---|---|
| `api` | fastapi, uvicorn, starlette | the FastAPI BFF (`himmy serve`) |
| `studio` | fastapi, uvicorn, starlette | the local GUI (`himmy studio`) |
| `auth` | pyjwt[crypto] | OIDC/JWT bearer verification |
| `dlp` | presidio-analyzer/anonymizer | ML-based PII detection beyond regex |
| `encryption` | cryptography | field-level envelope AES-GCM at rest |
| `secrets-aws` / `secrets-gcp` / `secrets-azure` | boto3 / google-cloud-secret-manager / azure-* | cloud secret managers |
| `providers` | pydantic-ai | cloud/gateway model providers |
| `postgres` | asyncpg | Postgres registry / storage / SQL |
| `knowledge` | pgvector, openai, pypdf | pgvector KB + OpenAI embedder + PDF reading |
| `observability` | logfire | tracing/metrics |
| `connectors` | feedparser, openpyxl | Nepali-news RSS + NRB Excel |
| `nepal` | nepali-datetime | Bikram Sambat calendar |
| `validation` | jsonschema | full JSON-Schema (offline falls back to a built-in subset) |
| `toolkit` | beautifulsoup4 | better `web_fetch` readability (stdlib fallback works) |
| `embeddings` | fastembed | local ONNX embeddings + the cross-encoder reranker |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff, mypy, jsonschema, types-PyYAML, pyjwt[crypto] | the quality gate |
| `all` | a broad bundle (see caveat) | |

> **`all` is not literally everything.** It includes fastapi/uvicorn/starlette,
> pydantic-ai, asyncpg, pgvector/openai/pypdf, logfire, feedparser/openpyxl,
> nepali-datetime, jsonschema, beautifulsoup4, and some dev deps — but it **excludes**
> `auth`, `dlp`, `encryption`, all three `secrets-*` backends, and `embeddings`. Install
> those explicitly if you need them.

---

# Security & governance posture (honest)

Himmy is offline-first, so the zero-config posture is **open** — designed for a single
local user, not a hardened multi-tenant deployment. The controls below are real, but be
precise about **what enforces vs what advises**, and what's opt-in.

**Default-on / zero-config:** stub inference, in-memory `EntityRegistry` lineage (volatile),
the `subprocess` sandbox for the `code` pack, `EnvSecrets`, and — **only on agents built
from a spec** (`from_spec`, including Studio-created agents) — the `grounding` output
guardrail (prepended, can't be removed via spec). The bare `build_runtime()` path does **not**
auto-add grounding.

**Default-off / opt-in:** all auth (default `None` ⇒ anonymous, all-tenants principal),
RBAC (bypassed entirely when no auth is configured), tenant isolation (meaningful only once
auth binds a principal to tenants), rate limiting (no `HIMMY_RATE_LIMIT` ⇒ none), the
security audit log (a no-op without auth), DLP, input/output guardrails on the bare
`build_runtime` path, field encryption at rest, durable registries, the `container` sandbox,
and record/replay.

**What actually enforces vs advises:**

- **Tool-arg guardrails genuinely deny.** The tool-arg pre-hook can return
  `allow=False` and **block the tool call** (and apply redacted/transformed args). This is
  real enforcement.
- **Input/output guardrails on the runtime path only redact/replace — they do not halt the
  run.** An `allowed=False` verdict emits a `GUARDRAIL_APPLIED` event and may redact text,
  but the (possibly unredacted) text still flows. So a plain blocklist on input/output
  **advises and flags**, it does not stop generation.
- **Effective enforcement comes from:** tool-arg blocking; PII / Nepal-PII / DLP
  **redaction-or-tokenization**; and the `grounding` guardrail, which **replaces** an
  ungrounded answer with a fixed refusal string (regex-only, conservative, no model call).
- **Sandbox.** The **default `subprocess`** sandbox provides resource/fault isolation
  (separate process, `setrlimit`, wall-clock timeout, throwaway tmpdir, stripped env) but its
  own docstring states it is **not** a security boundary against hostile code (network not
  enforced; `RLIMIT_AS` silently unenforced on macOS). For model-authored/untrusted code use
  `HIMMY_CODE_EXEC=container` (Docker/Podman: `--network none`, read-only rootfs,
  `--cap-drop ALL`, non-root, pid/mem/cpu limits) — the image must be pre-pulled.
- **Secrets & encryption.** Credentials are read via `get_secret()` (default `EnvSecrets`;
  file/Vault/AWS/GCP/Azure backends opt-in via `HIMMY_SECRETS`). Field encryption at rest is
  opt-in (`HIMMY_ENCRYPTION_KEY` + `cryptography`); with no key, storage stays plaintext.
- **Auth.** OIDC (JWKS-verified Bearer JWT, `auth` extra) or shared/mapped API keys
  (constant-time compare). RBAC has built-in `viewer`/`operator`/`auditor`/`admin` roles
  (deny-by-default) — but only once an authenticator is configured.

---

# Architecture & project layout

```
himmy/
  __init__.py        lazy top-level re-exports
  __main__.py        CLI parser + dispatch (14 subcommands)
  typed_agent.py     TypedAgent (additive façade over the runtime)
  core/              ids, errors, events (EventType enum)
  entities/          EntityRecord spine: records, registry, sqlite_registry,
                     postgres, integrity (hashes/Merkle/HMAC/Ed25519), lineage, projection
  agents/            personas, base_agent (task, thread, agent)
  services/          14 subpackages: audit, context, evaluation, governance,
                     guardrails, inference, knowledge (incl. retrieval/ hybrid RAG),
                     mcp, memory (temporal/graph), observability, prompts, sandbox,
                     storage, tools
  orchestrators/     state_graph, multi_agent, group_chat, planner, reflection, workflow
  runtime/           SingleAgentRuntime, builder, diagnostics
  api/               FastAPI BFF + auth (oidc/apikey/rbac) + Studio static
  application/       app-level services + models
  benchmark/         standing eval (suites/, graders, runner, stats, report)
  cli/               __main__, commands, provider
  config/            secrets
  connectors/        news + NRB
  nepal/             Bikram Sambat + NPR helpers
  skills/            built-in skills (code), loader, resolve, dispatch
  toolkit/           the 17 built-in tool packs
examples/            runnable, offline-first demos
studio/              the Vite + React SPA source
tests/               pytest suite (asyncio.run based)
```

Top-level `himmy` re-exports `MemoryService`, `TypedAgent`/`TypedAgentRunResult`/
`RunContext`, the multi-agent symbols, and `PlannerOrchestrator` (lazily). `StateGraph`,
`MemoryConsolidator`, `SemanticChunker`, `RetrievalConfig`, `HybridRetriever`, `BM25Index`,
the rerankers/rewriters, etc. live in their subpackages — import them from
`himmy.orchestrators` / `himmy.services.memory` / `himmy.services.knowledge`.

---

# Running the examples

Standalone `python examples/0N_*.py` scripts (note the numbering gap — no 08/09):

```bash
python examples/01_basic_chat.py          # offline stub
python examples/02_tool_calling.py        # offline stub
python examples/03_structured_output.py   # offline stub
python examples/04_orchestration_team.py  # offline stub
python examples/05_workflow.py            # offline stub
python examples/07_streaming.py           # token-delta streaming, offline stub
python examples/10_real_tool_agent.py     # utils pack; HIMMY_EXAMPLE_PROVIDER/MODEL to use Ollama
python examples/11_web_research.py        # keyless DuckDuckGo on the real path
python examples/12_memory_recall.py       # deterministic embedder by default; HIMMY_EMBEDDER=ollama for real recall
python examples/06_postgres_storage.py    # skips cleanly unless HIMMY_TEST_POSTGRES_DSN is set
```

YAML/config-driven example projects: `examples/connectors-demo/` (news + nepal on Ollama),
`examples/http-tool/` (a REST tool with no code), `examples/local-doc-chat/` (offline doc
chat), `examples/offline-helpdesk/` (fully offline helpdesk with goldens), and
`examples/farm-studio/` (a Studio team + custom tools — undocumented, no README).

---

# Testing & the standing benchmark

```bash
pip install -e ".[dev]"
make gate        # ruff check . → ruff format --check . → mypy himmy → pytest -q
```

- **~1410 tests collected** in this checkout (a current, moving count — not a frozen
  constant, and not a "passing" count). Of these, **2** are real-provider integration tests
  (Ollama, marked `integration`) and the rest are the offline suite. Provider, Postgres/
  pgvector, and observability tests **skip cleanly** when their backend/extra is absent.
  Tests use `asyncio.run` inside sync functions; run `pytest -q -rs` to see why each skip
  fired.
- **Lint/types.** Ruff selects `E,W,F,I,UP,B,S` (`S` = flake8-bandit SAST). Mypy is
  **near-strict** — every `--strict` flag *except* `warn_unused_ignores` (intentionally off
  so `type: ignore` can guard optional-extra imports).
- **CI lanes** (GitHub Actions): `ci.yml` (lint · types · tests with coverage),
  `integration.yml` (Ollama integration + the standing benchmark), `security.yml`
  (pip-audit — the blocking CVE gate — plus CycloneDX SBOM, gitleaks, and a report-only
  Trivy scan).
- **The standing benchmark** is a *separate thing from the pytest suite* — an LLM eval run
  via `himmy bench`. In CI it runs post-merge/nightly (not per-PR; CPU Ollama is too slow):
  ```bash
  python -m himmy bench --models ollama:qwen2.5:3b-instruct \
    --trials 2 --fail-under 0.6 --json bench-results.json
  ```
  `--fail-under` is a regression **floor, not a target**; no accuracy score is asserted
  anywhere. The default suite is `core.yaml` (15 tasks across arithmetic/files/sql/reasoning/
  nepal/multistep/rag/memory/agentic/regression); `skills.yaml` (3 tasks) is opt-in via
  `--suite`. Scorecards report accuracy + Wilson CI, tool-call accuracy, p50/p95 latency,
  cost, tokens, and per-category breakdown.

---

# CLI reference

```bash
himmy run      -f agent.yaml -p "…"     # one-shot prompt → answer (--json --stream --trace --plan --record/--replay)
himmy chat     -f agent.yaml            # interactive single-thread chat (--message --session)
himmy telegram -f agent.yaml            # run an agent.yaml as a Telegram bot (--token / HIMMY_TELEGRAM_BOT_TOKEN)
himmy team     -f team.yaml -p "…"      # multi-agent handoff/delegation
himmy eval     -f suite.yaml --agent agent.yaml   # eval cases (--team to eval a team)
himmy bench    --models ollama:qwen2.5:3b-instruct  # standing LLM benchmark (--fail-under is a CI floor)
himmy init     my-agent                 # scaffold agent.yaml + tools.py + himmy.toml + skill (--template, --team)
himmy serve                             # FastAPI BFF (needs the `api` extra)  [127.0.0.1:8000]
himmy studio                            # local GUI (needs the `studio` extra + built SPA)  [127.0.0.1:8765]
himmy doctor                            # extras / providers / keys + next step
himmy tools                             # list the 17 built-in tool packs
himmy skills [name]                     # list skills (built-in + project-local), or show one
himmy prices [sync|show|list]           # token pricing table (USD per 1M tokens)
himmy trace  [thread]                   # read .himmy/trace.db: timeline or recent runs
```

14 subcommands total. Note: `team` and `eval` do **not** accept the shared `--name`/
`--instruction` flags (only `run`/`chat`/`telegram` do). Several arguments are enforced at
runtime (exit 2) rather than by argparse — e.g. `run`/`team` `--prompt`, `telegram` token,
`prices show <model>`.

---

# Maturity & status

- **Version `0.1.0`, pre-1.0, single-maintainer, source-available** (MIT). APIs may change.
- The **offline build is reproducible**: a bare install (3 deps) runs the core framework with
  no keys/network.
- **No third-party certifications** (no SOC2 or equivalent). The security controls above are
  real but the zero-config posture is open by design; harden explicitly for any shared
  deployment.
- This is **not** an autonomous coding agent and makes no SWE-agent claims.

# License

MIT.
