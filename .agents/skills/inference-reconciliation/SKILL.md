---
name: "inference-reconciliation"
description: "Reconcile himmy/services/inference/ against pydantic-ai, OpenAI (API+SDK), Anthropic (API+SDK), OpenRouter, and Ollama; produce a drift report by fanning out one subagent per source"
---

# inference-reconciliation

Use this skill when the user asks to reconcile himmy's inference kernel against its upstream sources of truth.

**Trigger phrases**: "reconcile inference", "reconcile inference service", "inference drift report", "check inference against docs".

**Methodology**: see `docs/INFERENCE_RECONCILIATION.md` — read it once at the start so the four phases, the five sources, and the triage rubric are in context.

## Hard requirements

1. **Run five subagents — one per external source.** Non-negotiable. The orchestrator (you) does **not** survey the docs directly; you spawn one subagent each for **pydantic-ai**, **OpenAI (API + `openai` SDK)**, **Anthropic (API + `anthropic` SDK)**, **OpenRouter**, and **Ollama**, fired in the **same tool-use block** so they run concurrently. The orchestrator owns Phase 1 (Snapshot via `Explore`), Phase 3 (Diff synthesis), and Phase 4 (Triage). *(himmy talks to the OpenAI/Anthropic SDKs directly, so those surveys cover the SDK surface, not just the HTTP API.)*

2. **Reports are the paper trail — check them in.** himmy has no per-source memory store, so unlike a memory-backed repo the report is **not** ephemeral: write it to `docs/inference-reconciliation-reports/YYYY-MM-DD.md` and commit it. The other persistent artifacts are updates to `docs/services/inference.md` and the repo's agent guidance (`CLAUDE.md` / `AGENTS.md`).

3. **Fail loudly, don't silently substitute.** If a subagent reports `WebFetch` permission denied, surface it and ask the user to grant the permission once. Do not fall back to surveying from the orchestrator — that defeats the fan-out and produces thinner inventories.

## Phases

### Phase 1 — Snapshot (1 orchestrator-spawned Explore agent)

Spawn an `Explore` agent to inventory `himmy/services/inference/`. Ask for:

1. Public exports from `__init__.py`.
2. Every `ClientManager` and what it binds to — `AnthropicClientManager` (`anthropic` SDK), `OpenAIClientManager` (`openai` SDK), `PydanticAIClientManager` (pydantic-ai), `GatewayClientManager`, `StubClientManager`, and the locals in `local.py` (`Ollama` / `ClaudeCli` / `HimalayaGpt`). Provider class/SDK call, env vars, base URL, extras.
3. `routing.py` + `multi_provider.py` — failover trigger codes + the route table shape.
4. Every `ResponseFormat` mode + how each manager serves it (`StubClientManager` is the reference).
5. Every `InferenceRequest`/`InferenceResponse` field + every `InferenceErrorCode` and what raises it.
6. Prompt-cache mechanics in `prompt_cache.py` vs the response cache in `cache.py`; wired-vs-TODO (grep TODOs).
7. `pricing.py` (model→USD) and `replay.py` (record/replay) surfaces.
8. Streaming event kinds; test coverage under `tests/.../inference/`.

Also capture: pinned vs installed versions of `pydantic-ai`, `openai`, `anthropic`; git HEAD short hash.

### Phase 2 — Survey five sources (five subagents, parallel)

Fire five `general-purpose` subagents in one tool-use block (use `run_in_background: true`). Each surveys exactly one source against `docs/INFERENCE_RECONCILIATION.md` § "Phase 2":

- **A — pydantic-ai** (`ai.pydantic.dev`).
- **B — OpenAI**: API (`platform.openai.com/docs/`) **and** the `openai` Python SDK surface (`chat.completions.create` signature, exception classes).
- **C — Anthropic**: API (`platform.claude.com/docs/en/`, follow redirects from `docs.anthropic.com`) **and** the `anthropic` Python SDK surface.
- **D — OpenRouter** (`openrouter.ai/docs`) — routing, provider prefs, usage accounting, **cache-telemetry caveat**.
- **E — Ollama** (`github.com/ollama/ollama/blob/main/docs/api.md`) — `/api/chat`, `options`, streaming, per-model tool support.

Each returns a flat inventory across the 15 surface-area sections. Format: `feature → one-line → URL/symbol`. No commentary.

### Phase 3 — Diff (orchestrator)

One table per surface area (15). Columns: `Feature axis | pydantic-ai | OpenAI (API/SDK) | Anthropic (API/SDK) | OpenRouter | Ollama | himmy`. A blank cell is a finding.

### Phase 4 — Triage (orchestrator)

Classify each row — ADOPT / EXPOSE / NORMALIZE / ALIGN / IGNORE / DOC — and assign P0/P1/P2 for ADOPT/EXPOSE/ALIGN. **Watch the himmy-specific P0 class: direct-SDK drift** (an `openai`/`anthropic` signature or exception change a direct manager hasn't tracked). Rubric in the methodology doc.

## Output

Write the report to `docs/inference-reconciliation-reports/YYYY-MM-DD.md` and **commit it**. Header MUST cite: pydantic-ai / `openai` / `anthropic` pinned-vs-installed versions, OpenAI/Anthropic/OpenRouter/Ollama doc fetch dates, himmy HEAD, and outstanding inference TODOs.

Then: findings summary, the 15 surface-area tables, per-row findings table, prioritized next-actions, and a "Notes for the docs" section (one-line updates for `docs/services/inference.md` + agent guidance).

## After writing the report

1. **Surface the top P0 findings inline** (so they're seen without opening the report) — lead with any direct-SDK drift.
2. **Ask whether to apply the "Notes for the docs" updates now** — `docs/services/inference.md` notably lags the dedicated managers + prompt-cache mechanics, so this is usually real work.
3. **Do NOT auto-implement ADOPT / EXPOSE / ALIGN findings.** Those are issues, not commits.

## Common gotchas

- **Five sources, not three** — a pydantic-ai-centric reconciliation misses himmy's direct SDK managers, OpenRouter, and Ollama. All five run.
- **SDK ≠ API** — survey the `openai` / `anthropic` **Python SDK** surface (the managers bind it), not just the HTTP API docs.
- **OpenRouter cache telemetry is unreliable** — free/some routes report constant `cached_tokens`; never read it as real provider cache accounting.
- **Anthropic doc host** — migrated `docs.anthropic.com` → `platform.claude.com`; follow redirects.
- **Version drift** — installed `pydantic-ai`/`openai`/`anthropic` older than the lockfile pin is a P0 NORMALIZE finding; surface first.
