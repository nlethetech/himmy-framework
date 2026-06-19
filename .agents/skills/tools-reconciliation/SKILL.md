---
name: "tools-reconciliation"
description: "Reconcile himmy/services/tools/ + himmy/services/mcp/ against pydantic-ai (Toolset/MCP), the MCP spec, and provider tool-schema constraints; produce a drift report by fanning out three subagents"
---

# tools-reconciliation

Use this skill when the user asks to reconcile himmy's tool surface against its upstream sources of truth — pydantic-ai's Toolset/MCP API, the MCP specification, and provider tool-schema constraints.

**Trigger phrases**: "reconcile tools", "reconcile tools service", "tools drift report", "check tools against docs", "reconcile MCP".

**Methodology**: see `docs/TOOLS_RECONCILIATION.md` — read it once at the start so the four phases, the surface areas, and the triage rubric are in context.

**Cross-reference**: sibling of `inference-reconciliation`. The two share a small overlap (`tool_choice` request-envelope serialization, structured-output wire format) that **inference** owns as primary — do NOT re-survey those here; cross-reference the latest `docs/inference-reconciliation-reports/` in the relevant cells.

## Hard requirements

1. **Run three subagents — one per source.** Non-negotiable. The orchestrator (you) does not survey docs directly; spawn one subagent each for **pydantic-ai (Toolset/MCP)**, the **MCP spec**, and **provider tool-schema constraints**, fired in the **same tool-use block**. The orchestrator owns Phase 1 (Snapshot via `Explore`), Phase 3 (Diff), Phase 4 (Triage).

2. **Snapshot BOTH `himmy/services/tools/` AND `himmy/services/mcp/`.** MCP is its own service in himmy (not a backend under tools) — a snapshot that misses `services/mcp/` misses the entire lifecycle/transport surface.

3. **Reports are ephemeral — do NOT commit them.** `docs/tools-reconciliation-reports/` is gitignored; the report is one run's working output. The persistent artifacts (no per-source memory store) are the "Notes for the docs" applied to `docs/services/tools.md`, `docs/services/mcp.md`, and agent guidance (`CLAUDE.md` / `AGENTS.md`). Apply those before discarding the report.

4. **Fail loudly, don't silently substitute.** If a subagent reports `WebFetch` denied, surface it and ask for the permission once. Don't fall back to surveying from the orchestrator.

## Phases

### Phase 1 — Snapshot (1 orchestrator-spawned Explore agent)

Inventory `himmy/services/tools/` **and** `himmy/services/mcp/`. Ask for:

1. Public exports from both `__init__.py`s.
2. Every `ToolBackendKind` + how each binds to pydantic-ai (`runtime_adapter.py`: `ToolServiceToolset`, `build_arg_model`).
3. `ToolDefinition` / `ToolInvocation` / `ToolExecutionResult` / `ToolPolicyDecision` fields; every `ToolErrorCode` and what raises it.
4. `service.py` dispatch — policy-hook ordering + the `TOOL_*` events it emits.
5. MCP (`services/mcp/`) — `connector.py` lifecycle, `config/mcp_spec.py`, transports, session/reconnect, per-server lock, keepalive, capability gating, tenancy header propagation. **Does it wrap pydantic-ai `MCPServer*` or its own client? Where (if anywhere) is MCP tool-input-schema normalization done?**
6. `repair.py` (near-miss tool-call recovery), `access.py` (read-vs-change intent), `security.py` (SSRF / path-traversal / secret redaction), `validation.py` (input + output schema validation), `registry.py` (catalog/shadowing).
7. Test coverage (`tests/.../tools/`, `tests/.../mcp/`).

Also capture: pinned vs installed `pydantic-ai` + `mcp` SDK; git HEAD; TODO grep in both dirs.

### Phase 2 — Survey three sources (three subagents, parallel; `run_in_background: true`)

- **A — pydantic-ai (Toolset/MCP)** (`ai.pydantic.dev`): advanced tools, toolsets, MCP, deferred tools, `output_type`↔`toolsets`, capabilities, API pages for tools/toolsets/mcp. Not model-adapter/message pages.
- **B — MCP spec** (`modelcontextprotocol.io/specification/`, capture the dated revision): architecture, initialize + capabilities, transports, server features (tools/resources/prompts), client features (sampling/elicitation), notifications, authorization, versioning + changelog.
- **C — provider tool-schema constraints** (narrow — input schema only): OpenAI strict/non-strict + structured-output schema rules; Anthropic `input_schema` + nullable + `tool_choice` interaction + tool-def `cache_control`; Gemini/Kimi/Bedrock quirks.

Each returns a flat inventory across the methodology's surface-area sections. `feature → one-line → anchor`.

### Phase 3 — Diff (orchestrator)

One table per surface area. Columns: `Feature axis | pydantic-ai | MCP spec | Provider constraints | himmy`. Blank cell = finding. **Include the himmy-specific rows** (repair, access intent, HTTP security, I/O validation) and ask, for each, whether an upstream equivalent exists (→ ADOPT) or himmy correctly owns a gap (→ IGNORE, documented). Cross-reference the inference report for `tool_choice`.

### Phase 4 — Triage (orchestrator)

ADOPT / EXPOSE / NORMALIZE / ALIGN / IGNORE / DOC + P0/P1/P2. himmy P0 classes: cross-tenant MCP header leak; wedged tool-call from a missing RPC lock; a schema-repair gap dropping a whole request; an SSRF/path-traversal hole in `security.py`; reconnect/circuit regression under a pydantic-ai bump; **`repair.py` fabricating a tool call the model never sensibly intended** (repair must fail safe).

## Output

Write the report to `docs/tools-reconciliation-reports/YYYY-MM-DD.md` (**gitignored — do not commit**). Header MUST cite: pydantic-ai + `mcp` SDK pinned-vs-installed; MCP spec revision date; OpenAI/Anthropic fetch dates; himmy HEAD; implementation state; outstanding tools/mcp TODOs. Then findings summary, inference cross-references, surface-area tables, per-row findings, prioritized next-actions, and "Notes for the docs" (one-liners for `docs/services/tools.md`, `docs/services/mcp.md`, agent guidance).

## After writing the report

1. **Surface the top P0 findings inline** — lead with any repair-fabrication or SSRF/tenancy risk.
2. **Ask whether to apply the "Notes for the docs" updates now.**
3. **Do NOT auto-implement ADOPT / EXPOSE / ALIGN findings.** Issues, not commits.

## Common gotchas

- **MCP is its own service** — snapshot `services/mcp/`, not just `services/tools/`.
- **MCP spec is versioned** — `modelcontextprotocol.io/specification/` redirects to a dated revision (**current: `2025-11-25`**); capture it; a new revision since last report is itself a finding.
- **WebFetch is per-domain + background subagents can't be prompted** — a denied host hard-fails (`WEBFETCH_DENIED`); pre-allow the hosts before the run: `WebFetch(domain:platform.openai.com)`, `WebFetch(domain:developers.openai.com)`, `WebFetch(domain:modelcontextprotocol.io)`. Surface a denial and re-fire after the grant — don't survey from memory. (`pydantic.dev`, `platform.claude.com`, `github.com` were already allowed in the 2026-06-19 run.)
- **Transport name drift** — "HTTP+SSE" → "Streamable HTTP"; reconcile which pydantic-ai supports vs what himmy mounts.
- **Small-model surfaces are himmy-owned for a reason** — `repair.py`/`access.py` exist because local models misfire; only ADOPT-away when upstream genuinely subsumes them, else IGNORE with a documented rationale.
- **Overlap with inference** — cross-reference, don't re-survey, `tool_choice` + structured-output wire format. If no recent inference report exists, flag (DOC) that inference reconciliation should run first.
- **Version drift** — installed `pydantic-ai`/`mcp` older than the pin is a P0 NORMALIZE finding; surface first.
