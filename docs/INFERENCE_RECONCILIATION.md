# Inference Reconciliation — Methodology

**Purpose**: Periodically reconcile `himmy/services/inference/` against the moving sources of truth it wraps — **pydantic-ai**, the **OpenAI** API + Python SDK, the **Anthropic** API + Python SDK, **OpenRouter**, and **Ollama** — to catch feature drift, behavior drift, and contract drift before they reach a deployment.

This document is the canonical reference. The runnable workflow is at `.agents/skills/inference-reconciliation/SKILL.md`. Reports go to `docs/inference-reconciliation-reports/YYYY-MM-DD.md`.

> **Why himmy fans out wider than a typical pydantic-ai app.** himmy does **not** route everything through pydantic-ai. It ships **direct** provider managers — `AnthropicClientManager` talks to the `anthropic` SDK's Messages API; `OpenAIClientManager` talks to the `openai` SDK's `chat.completions`; `PydanticAIClientManager` is only the *optional* path. It also serves open-weight routes via **OpenRouter** and self-hosted models via **Ollama** / Claude CLI / HimalayaGPT (`local.py`). Each of those is an independently-evolving surface. So reconciliation surveys **five external sources**, and — crucially — tracks the **SDK** surface (`anthropic`, `openai`), not only the HTTP API docs, because the direct managers bind to the SDK.

## Paper trail (differs from a memory-backed repo)

himmy has no per-source memory store, so the **reports are the durable paper trail** — they are **checked in** under `docs/inference-reconciliation-reports/` (not gitignored). Each run also applies:

1. Targeted updates to `docs/services/inference.md` when a finding shifts a documented behavior (e.g. the dedicated managers / prompt-cache mechanics the doc currently lags).
2. Updates to the repo's agent guidance (`CLAUDE.md` / `AGENTS.md`) when a finding changes an architectural assumption (e.g. a cache-minimum bump, an SDK breaking change).
3. The triaged P0/P1/P2 actions, which become issues the team actions over time.

## Why this exists

The inference kernel is the single provider-neutral doorway between himmy's runtime and several independently-evolving systems:

- **pydantic-ai** ships new capabilities, event types, and `ModelSettings` fields fast (used by `PydanticAIClientManager`).
- **OpenAI SDK + API** changes `chat.completions` / Responses shapes, prompt-cache minimums, `tool_choice`, `response_format`, and `usage` detail fields — and the **`openai` Python SDK** changes method signatures `OpenAIClientManager` binds to.
- **Anthropic SDK + API** evolves `cache_control` mechanics, per-model minimums (Haiku 4.5 1024 → 4096), tool-use shapes, the token-counting endpoint — and the **`anthropic` Python SDK** surface `AnthropicClientManager` binds to.
- **OpenRouter** changes routing, model availability, and (notably) reports **unreliable cache telemetry** — `prompt_cache`/`cached_tokens` overhead that looks constant on free routes.
- **Ollama** changes its local API shape, model tags, and options (`OllamaClientManager`).

In-house contracts (`HimalayaGptClientManager`, `ClaudeCliClientManager`) reconcile against **their own** definitions, not an external doc — note them in the snapshot but they are not external survey targets.

If we don't reconcile periodically, the kernel *thinks* it's provider-neutral but isn't.

## When to run

Run a reconciliation pass when **any** of these is true:

- A new pydantic-ai, `openai`, or `anthropic` version is pinned in `pyproject.toml` / lockfile.
- OpenAI, Anthropic, OpenRouter, or Ollama announces a substantive change to caching, tool-use, structured-output, streaming, or routing.
- The inference kernel gets a new manager, a new `ResponseFormat` mode, or a new field on `InferenceRequest` / `InferenceResponse`.
- Quarterly, regardless. Drift is invisible until you look for it.

## The four phases

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. SNAPSHOT │ →  │  2. SURVEY   │ →  │   3. DIFF    │ →  │  4. TRIAGE   │
│  what we have│    │   5 sources  │    │ line up rows │    │ classify +   │
│              │    │  (parallel)  │    │              │    │  prioritize  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Phase 1 — Snapshot the inference kernel ("we have")

Walk `himmy/services/inference/` and produce a structured inventory. The skill spawns an `Explore` agent; manually you'd inventory:

1. **Public exports** from `inference/__init__.py`.
2. **Every `ClientManager`** and what it binds to — `AnthropicClientManager` (`anthropic` SDK), `OpenAIClientManager` (`openai` SDK), `PydanticAIClientManager` (pydantic-ai), `GatewayClientManager`, `StubClientManager`, and the locals in `local.py` (`OllamaClientManager`, `ClaudeCliClientManager`, `HimalayaGptClientManager`). For each: provider class/SDK call, env vars, base URL, `extra_body`/`extra_headers`.
3. **Routing + multi-provider** — `routing.py` (fallback/routing across model keys) and `multi_provider.py` (per-`model_key` dispatch): which error codes trigger failover, the route table shape.
4. **Every `ResponseFormat` mode** (text / JSON / tools / structured / workflow) and how each manager simulates/serves it (the `StubClientManager` is the reference implementation).
5. **Every `InferenceRequest` / `InferenceResponse` field** with type + default; **every `InferenceErrorCode`** and what raises it.
6. **Prompt-cache mechanics** in `prompt_cache.py` (the universal prefix-cache contract) vs the **response cache** in `cache.py` — what's wired per manager vs TODO. `grep -rn "TODO" himmy/services/inference/`.
7. **Pricing** (`pricing.py`) — the model→USD table and which usage fields feed it.
8. **Replay** (`replay.py`) — record/replay managers and what they capture.
9. **Streaming event kinds** and the raw provider/pydantic-ai events they translate.
10. **Test coverage** — `tests/.../inference/` — what's exercised vs not.

Output: a markdown "we have" inventory. Also capture pinned versions of `pydantic-ai`, `openai`, `anthropic` (lockfile vs installed) and the git HEAD short hash.

### Phase 2 — Survey the five sources ("what's available / current")

Run all five surveys **in parallel** — they're independent and latency is the bottleneck. Each produces a flat list of facts: `feature / symbol / parameter → one-line description → import path or URL anchor`.

#### Survey A: pydantic-ai
Canonical: https://ai.pydantic.dev/ . Pages: `agents/`, `output/`, `tools/` + advanced tools, `message-history/`, `models/overview/`, `api/models/`, `api/settings/` (`ModelSettings`), capabilities, `api/usage/`. Pin the version.

#### Survey B: OpenAI (API **and** `openai` Python SDK)
API: https://platform.openai.com/docs/ — Chat Completions, Responses, prompt caching (minimum 1024; `prompt_cache_key`), function calling (`tool_choice` shapes, strict mode), structured outputs, streaming, `usage.prompt_tokens_details`. SDK: the `openai` Python package — `chat.completions.create` signature, `AsyncOpenAI` client params, exception classes (what `OpenAIClientManager` maps to `InferenceErrorCode`). **Both**, because the manager binds the SDK directly.

#### Survey C: Anthropic (API **and** `anthropic` Python SDK)
API: https://platform.claude.com/docs/en/ (migrated from `docs.anthropic.com`; follow redirects) — Messages API, prompt caching (`cache_control`, ≤4 breakpoints, per-model minimums, `ttl` 5m/1h), tool use (`tool_choice` `auto`/`any`/`{type:tool,name}`), streaming, extended thinking, **count-tokens** endpoint, model overview. SDK: the `anthropic` Python package — `messages.create` signature, `AsyncAnthropic` params, exception classes.

#### Survey D: OpenRouter
https://openrouter.ai/docs — the unified request shape, model routing / provider preferences, `usage` accounting, and the **cache-telemetry caveat** (free/some routes report constant `cached_tokens` overhead — flag wherever himmy reads cache hits through OpenRouter).

#### Survey E: Ollama
https://github.com/ollama/ollama/blob/main/docs/api.md — `/api/chat` + `/api/generate` request/response, `options` (temperature, num_ctx, etc.), streaming shape, tool-calling support per model, and the model-tag surface `OllamaClientManager` targets.

### Phase 3 — Diff (line up the rows)

Per surface area, build a table. Columns: `Feature axis | pydantic-ai | OpenAI (API/SDK) | Anthropic (API/SDK) | OpenRouter | Ollama | himmy`. A blank cell is itself a finding (that source has nothing for the axis — common for the local/Ollama column on caching, for example).

Surface areas (each its own table):

1. Request shape (system / messages / segments)
2. Response shape (output, usage, tool exchanges)
3. Tool registration + invocation
4. Tool choice and forcing
5. Output-type modes (text / JSON / structured / tool / workflow)
6. Streaming events
7. Multi-turn / message history
8. Multi-modal content
9. Prompt-caching mechanics (per provider — the big asymmetry)
10. Token-usage telemetry + cost (incl. OpenRouter cache caveat, `pricing.py`)
11. Errors + retries + timeouts (per-SDK exception → `InferenceErrorCode`)
12. Routing / failover (`routing.py`, `multi_provider.py`) — himmy-specific; no external analogue
13. Local / self-hosted (Ollama options, Claude CLI, HimalayaGPT) — himmy-specific
14. Workflow / deferred tool execution
15. Observability / record-replay (`replay.py`)

### Phase 4 — Triage

Classify each row into one bucket + a one-line rationale:

| Bucket | Meaning |
|---|---|
| **ADOPT** | A source exposes this; we should wrap it (e.g. a pydantic-ai capability, an SDK feature) |
| **EXPOSE** | A provider has a behavior we're hiding from callers (e.g. OpenAI `service_tier`) |
| **NORMALIZE** | Providers diverge; we need a single neutral shape (e.g. cache-hit accounting across SDKs + OpenRouter) |
| **ALIGN** | Same idea, different name on our side |
| **IGNORE** | Our abstraction is correctly above this (e.g. routing/failover — himmy owns it) |
| **DOC** | No code change; record it in `docs/services/inference.md` / agent guidance |

For ADOPT / EXPOSE / ALIGN, assign a priority:

- **P0** — silent correctness risk (cache invalidation, wrong error code, an **SDK breaking change** the direct manager hasn't tracked)
- **P1** — measurable cost / latency / reliability gain left on the table
- **P2** — naming / hygiene / future-proofing

> **himmy-specific P0 class: direct-SDK drift.** Because `AnthropicClientManager` / `OpenAIClientManager` bind the SDKs directly (not via pydantic-ai), an SDK method-signature or exception change can break a manager silently. A pydantic-ai-only app never sees this — himmy must.

## Output: the report

A single markdown file at `docs/inference-reconciliation-reports/YYYY-MM-DD.md` (**checked in** — it is himmy's paper trail):

```markdown
# Inference Reconciliation — YYYY-MM-DD

**Versions surveyed**
- pydantic-ai: <pinned vs installed>
- openai (SDK): <version> · OpenAI API docs: <fetch date>
- anthropic (SDK): <version> · Anthropic API docs: <fetch date>
- OpenRouter docs: <fetch date> · Ollama docs: <fetch date>
- himmy HEAD: <git rev-parse --short HEAD>

**Findings summary** — ADOPT (P0/P1/P2) · EXPOSE · NORMALIZE · ALIGN · IGNORE · DOC

## Tables (one per surface area)
## Findings (per row): # | Surface | Row | Bucket | Priority | Rationale | Next step
## Recommended next actions (priority order)
## Notes for the docs
- docs/services/inference.md: <one-line>
- CLAUDE.md / AGENTS.md: <one-line>
```

## Anti-patterns this guards against

- **Quiet wheel-reinvention** — we wrote a thing pydantic-ai now ships (ADOPT).
- **Stale direct-SDK assumptions** — `openai`/`anthropic` changed a signature; a direct manager still calls the old one (P0).
- **Asymmetric provider coverage** — a behavior added to the Anthropic manager but not OpenAI (Phase 3 rows make it visible).
- **OpenRouter cache illusion** — reading `cached_tokens` through OpenRouter as if it were real provider telemetry.
- **Doc drift** — `docs/services/inference.md` describing behavior that's no longer current (it already lags the dedicated managers).

## Tools handoff

The Tools kernel has its own reconciliation pair at `docs/TOOLS_RECONCILIATION.md` + `.agents/skills/tools-reconciliation/` (the MCP / Toolset / tool-input-schema surface). The two overlap on only two axes — both treat **inference reconciliation** as primary:

- `tool_choice` request-envelope serialization per provider (Phase 3 row 4 here)
- Structured-output wire format when an output type is set (Phase 3 row 5 here)

A tools report flagging drift on either is a signal to run an inference reconciliation too.

## What this does NOT do

- It does not run code changes. ADOPT/EXPOSE/ALIGN findings become issues, not commits.
- It does not validate behavior empirically — that's the examples / `StubClientManager` + record-replay layer.
- It does not re-survey the Toolset / MCP / tool-input-schema surface — that's the tools reconciliation.

## Sources surveyed (canonical URLs)

- pydantic-ai — https://ai.pydantic.dev/
- OpenAI — https://platform.openai.com/docs/ + the `openai` Python SDK
- Anthropic — https://platform.claude.com/docs/en/ + the `anthropic` Python SDK
- OpenRouter — https://openrouter.ai/docs
- Ollama — https://github.com/ollama/ollama/blob/main/docs/api.md
