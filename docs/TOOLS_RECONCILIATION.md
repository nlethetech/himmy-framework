# Tools Reconciliation — Methodology

**Purpose**: Periodically reconcile himmy's tool surface — `himmy/services/tools/` **and** `himmy/services/mcp/` — against three moving sources of truth: **pydantic-ai** (Toolset / MCP API), the **MCP specification**, and **provider tool-schema constraints** (OpenAI function-calling + Anthropic tool-use). Catches feature drift, MCP lifecycle bugs, and cross-provider schema-rejection regressions before they reach an agent run.

This document is the canonical reference. The runnable workflow is at `.agents/skills/tools-reconciliation/SKILL.md`. Reports go to `docs/tools-reconciliation-reports/YYYY-MM-DD.md`.

> **himmy specifics to keep in view.** (1) **MCP is its own service** (`himmy/services/mcp/`, e.g. `connector.py` + `config/mcp_spec.py` + `api/routers/studio_mcp.py`), not a backend under `tools/` — the snapshot covers both. (2) himmy carries tool surfaces a cloud-only harness doesn't, because **Nepal teams run small local models** (Ollama / Claude CLI / HimalayaGPT): `repair.py` (recover a small model's near-miss tool call), `access.py` (read-vs-change intent to catch wrong-tool selection), `security.py` (SSRF / path-traversal / secret redaction on the HTTP connector), `validation.py` (input **and** output schema validation). These get their own diff rows and their own "is there an upstream equivalent?" question.

## Paper trail

Reports are **ephemeral** — `docs/tools-reconciliation-reports/` is **gitignored**; a report is one run's working output, not a committed artifact. himmy has no per-source memory store, so the **persistent** artifacts of each run are:

1. Targeted updates to `docs/services/tools.md` and `docs/services/mcp.md` when a finding shifts documented behavior.
2. Updates to agent guidance (`CLAUDE.md` / `AGENTS.md`) when a finding changes an architectural assumption (e.g. a provider rejecting a previously-accepted JSON Schema shape, or pydantic-ai shipping a lifecycle hook we should adopt and delete our copy of).
3. The triaged P0/P1/P2 actions, which become issues the team actions over time.

Apply the report's "Notes for the docs" to (1)/(2) before discarding the report — the durable signal lives there, not in the gitignored file.

## Why this is separate from inference reconciliation

The tool surface sits *above* the inference managers: it composes toolsets, owns MCP lifecycle, validates/repairs tool I/O, and applies policy hooks. pydantic-ai's `Toolset` / `MCPServer*` surface is a different contract from its `Model` adapter surface, and the MCP spec has no analogue in the inference stack.

| Concern | Owner |
|---|---|
| pydantic-ai `Model` adapter behavior, message + structured-output wire format | **Inference reconciliation** |
| `tool_choice` request-envelope serialization per provider | Inference (primary), cross-referenced here |
| pydantic-ai `Toolset` / `AbstractToolset` API, `MCPServer*` lifecycle | **Tools reconciliation** (this doc) |
| MCP-discovered tool input-schema normalization (cross-provider) | Tools reconciliation |
| `output_type` ↔ `toolsets` interaction | Tools reconciliation (primary) |
| himmy tool-call repair / access-intent / HTTP security / I-O validation | Tools reconciliation (himmy-specific) |

## When to run

- A new pydantic-ai or `mcp` SDK version is pinned.
- The MCP spec ships a new revision (transports, capability shapes, notifications, auth).
- OpenAI/Anthropic change tool-input-schema validation (strict-mode tightening, nullable rules, etc.).
- himmy's tools/mcp kernel gains a new `ToolBackendKind`, policy-hook shape, MCP transport, or `ToolDefinition` field.
- A new MCP server is mounted exposing capabilities not reconciled before (resources / prompts / sampling / elicitation).
- An incident traces to a tool-schema rejection, MCP lifecycle bug, or a repair/validation miss.
- Quarterly, regardless.

## The four phases

```
1. SNAPSHOT  →  2. SURVEY (3 sources, parallel)  →  3. DIFF  →  4. TRIAGE
```

### Phase 1 — Snapshot ("we have")

Inventory **both** `himmy/services/tools/` and `himmy/services/mcp/`. The skill spawns an `Explore` agent; capture:

1. **Public exports** from `tools/__init__.py` and `mcp/__init__.py`.
2. **Every `ToolBackendKind`** + how each binds to a pydantic-ai primitive (`runtime_adapter.py`: `ToolServiceToolset`, `build_arg_model`).
3. **Every `ToolDefinition` / `ToolInvocation` / `ToolExecutionResult` / `ToolPolicyDecision` field** + every `ToolErrorCode` and what raises it.
4. **`service.py` dispatch** — policy-hook ordering, the event emission it performs (the `TOOL_*` events), guarantees enforced.
5. **MCP implementation** (`services/mcp/`) — `connector.py` client/lifecycle, `config/mcp_spec.py`, transports, session/reconnect, per-server locking, keepalive, capability gating, header propagation for tenancy. **Does it wrap pydantic-ai `MCPServer*` or its own client?** Where is MCP-discovered **tool-input-schema normalization** done (if at all)?
6. **`repair.py`** — what near-misses it recovers (tool-name fuzzy match, arg coercion) and the recovery contract.
7. **`access.py`** — the read-vs-change intent model and where the runtime consumes it.
8. **`security.py`** — SSRF / path-traversal guards + secret redaction on the HTTP connector.
9. **`validation.py`** — input + output JSON-schema validation; what it rejects.
10. **`registry.py`** — catalog mechanics, local/HTTP registration, shadowing/override rules.
11. **Test coverage** (`tests/.../tools/`, `tests/.../mcp/`, `tests/skills/`) — lifecycle, schema fixtures, repair cases, security cases, policy-block paths.

Also capture: pinned vs installed `pydantic-ai` + `mcp` SDK versions; git HEAD; `grep -rn "TODO" himmy/services/tools himmy/services/mcp`.

### Phase 2 — Survey three sources (three subagents, parallel)

`feature / symbol → one-line → import path or URL anchor`.

**Survey A: pydantic-ai (Toolset / MCP)** — `ai.pydantic.dev`. Pages: advanced tools (`Tool.from_schema`, `ToolReturn`, `prepare`/`prepare_tools`, `tool_choice`, `parallel_tool_call_execution_mode`, `defer_loading`/`ToolSearch`, `ModelRetry`, `tool_timeout`, `args_validator`); toolsets (`AbstractToolset`, `FunctionToolset`, `Combined`/`Filtered`/`Renamed`/`Prepared`, `get_tools`/`call_tool`); MCP (`MCPServerStdio` / `MCPServerStreamableHTTP` / `MCPServerSSE`, transports, `tool_prefix`, lifecycle, sampling/elicitation handlers); deferred tools (`DeferredToolRequests`/`Results`, `ToolApproved`/`Denied`); `output_type`↔`toolsets`; capabilities; API pages for tools/toolsets/mcp. **Do not** re-survey model-adapter/message pages — that's inference.

**Survey B: MCP specification** — `modelcontextprotocol.io/specification/` (capture the dated revision). Sections: architecture, initialize + capability negotiation, transports (stdio / Streamable HTTP / legacy SSE, session-id, reconnection), server features (tools incl. `inputSchema` constraints + `CallToolResult.content`/`structuredContent`/`isError`; resources; prompts), client features (sampling, elicitation), notifications (`tools/list_changed`, `progress`, `cancelled`), authorization, versioning. Capture the changelog if present.

**Survey C: provider tool-schema constraints** — narrow: only the **tool input schema** (`parameters`/`input_schema`), not the full request envelope (inference owns that). OpenAI strict + non-strict mode schema rules, structured-outputs schema rules; Anthropic `input_schema` shape + nullable handling (no `null`-in-union) + `tool_choice` interaction + tool-def `cache_control`; cross-provider quirks (Gemini undefined-`required`, Kimi/Moonshot `#/$defs/`, Bedrock). Record: which provider, which shape rejected, which accepted, minimal reproducer.

### Phase 3 — Diff

One table per surface area. Columns: `Feature axis | pydantic-ai | MCP spec | Provider constraints | himmy`. Blank cell = a finding. Cross-reference the latest inference report for `tool_choice` serialization rather than duplicating.

Surface areas (each its own table) — the universal set **plus himmy-specific rows**:

1. `ToolDefinition` shape & lifecycle
2. Backend kinds → pydantic-ai primitives (`ToolServiceToolset`, `build_arg_model`)
3. Tool registry mechanics (registration, shadowing, generation)
4. MCP server lifecycle (connect / initialize / shutdown / reconnect) — *in `services/mcp/`*
5. MCP transport selection (stdio / Streamable HTTP / SSE; session-id; per-request headers)
6. MCP tool discovery + `tools/list_changed` reactive refresh
7. MCP capability gating (resources / prompts only if advertised)
8. MCP keepalive & idle-timeout
9. Per-server RPC serialization
10. Per-request context / tenancy header propagation to server-side RLS
11. Tool input-schema normalization (cross-provider repairs) — **where, or absent?**
12. Tool name namespacing (`mcp_{server}_{tool}`, collisions)
13. Tool output envelope (MCP `content` blocks, `structuredContent`, `isError`)
14. Policy hooks (pre/post, ordering, blocked-result shape) — `service.py`
15. `output_type` ↔ `toolsets` interaction
16. Deferred / human-in-the-loop tools (`DeferredToolRequests`, `ToolApproved`/`Denied`)
17. Observability — the `TOOL_*` events `service.py` emits
18. **(himmy) Tool-call repair** — near-miss recovery for small local models; is there a pydantic-ai analogue (`ModelRetry`, fuzzy tool match)? `repair.py`
19. **(himmy) Access intent** — read-vs-change classification; relation to any upstream idempotency taxonomy; `access.py`
20. **(himmy) HTTP connector security** — SSRF / path-traversal / secret redaction on model-synthesized args; `security.py`
21. **(himmy) I/O validation** — input **and** output schema validation vs pydantic-ai's `args_validator` / `output_validator`; `validation.py`

### Phase 4 — Triage

Buckets: ADOPT / EXPOSE / NORMALIZE / ALIGN / IGNORE / DOC. Priority P0/P1/P2 for ADOPT/EXPOSE/ALIGN.

**P0 classes for himmy tools:** cross-tenant leak via missing MCP header propagation; tool-call wedged by a missing per-server RPC lock; a schema-repair gap that makes a provider drop the whole request; an SSRF/path-traversal hole in `security.py`; a circuit-breaker / reconnect regression under a pydantic-ai bump; **a `repair.py` recovery that silently turns a wrong tool call into a confidently-wrong execution** (repair must fail safe, not fabricate).

> **himmy-specific lens — small-model correctness.** `repair.py` / `access.py` exist because local models (Ollama, etc.) misfire more than frontier models. When pydantic-ai or a provider ships a feature that subsumes one of these (e.g. native fuzzy tool-matching, an output validator), that's an ADOPT (delete our copy). When it doesn't, that's an IGNORE (we correctly own a gap upstream leaves open) — document *why* so it isn't mistaken for drift.

## Output: the report

`docs/tools-reconciliation-reports/YYYY-MM-DD.md` (**gitignored — ephemeral**; persist findings via "Notes for the docs"). Header MUST cite: pydantic-ai + `mcp` SDK pinned-vs-installed; MCP spec revision date; OpenAI/Anthropic doc fetch dates; himmy HEAD; implementation state; outstanding tools/mcp TODOs. Then: findings summary, cross-references to the latest inference report, the surface-area tables, per-row findings table, prioritized next-actions, and "Notes for the docs" (one-liners for `docs/services/tools.md`, `docs/services/mcp.md`, agent guidance).

## Anti-patterns this guards against

- **Quiet wheel-reinvention** — pydantic-ai ships an MCP lifecycle wrapper / fuzzy tool-match we hand-rolled (ADOPT, delete code).
- **Stale schema-repair assumptions** — a provider relaxes a rule `validation.py`/normalization still enforces.
- **MCP spec drift** — a new transport / elicitation our `services/mcp/` ignores; the next server using it fails opaquely.
- **Asymmetric provider coverage** — a repair added for one provider, not the analogous case in another.
- **Repair that fabricates** — `repair.py` "recovers" a call the model never sensibly intended (a correctness P0).
- **Doc drift** — `docs/services/tools.md` / `mcp.md` describing behavior that's no longer current.

## What this does NOT do

- No code changes — ADOPT/EXPOSE/ALIGN become issues, not commits.
- No empirical validation — live MCP + cross-provider tool-call smoke tests are the empirical layer.
- No inference re-survey — cross-reference the latest inference report for `tool_choice` / structured-output wire format; if none is recent, flag (DOC) that inference reconciliation should run first.

## Sources surveyed (canonical URLs)

- pydantic-ai (Toolset / MCP) — **https://pydantic.dev/docs/ai/** (301 from `ai.pydantic.dev`)
- MCP specification — https://modelcontextprotocol.io/specification/ — **current revision `2025-11-25`** (capture the dated revision each run; it supersedes `2025-06-18`/`2025-03-26`)
- OpenAI (tool-schema constraints only) — **https://developers.openai.com/api/docs/** (301 from `platform.openai.com/docs/`)
- Anthropic (tool-schema constraints only) — https://platform.claude.com/docs/en/

**WebFetch permission (operational):** per-domain, and background survey subagents can't answer interactive prompts — pre-allow the survey hosts (`WebFetch(domain:platform.openai.com)`, `WebFetch(domain:developers.openai.com)`, `WebFetch(domain:modelcontextprotocol.io)`) before a run, or they hard-deny.
