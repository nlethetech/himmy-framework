# himmy-agent documentation

> Developer documentation for the himmy-agent harness — a **local-first, self-contained
> autonomous AI agent framework** that runs fully offline by default and scales up to a
> hardened, multi-backend deployment through opt-in extras.

These docs describe the harness as it exists in the code. Each page follows the same
shape: **Overview → Module map → Key abstractions → How it works → Configuration →
Extension points → Gotchas → Related docs.** Every claim is grounded in a real file path
under `himmy/`.

## Start here

- **[Architecture overview](./architecture/overview.md)** — the layered package map, the
  offline-first philosophy, and the end-to-end shape of a single agent run. Read this first.
- **[Local-first & hardening](./architecture/local-first.md)** — what runs with zero config,
  the full table of opt-in extras, and how each enterprise control degrades to a safe default.

## Execution model

| Doc | What it covers |
| --- | --- |
| [Runtime](./architecture/runtime.md) | The single-agent loop, structured output, approval/checkpoint pause-resume, context compaction, termination, tool routing, record/replay. |
| [Orchestrators](./architecture/orchestrators.md) | Multi-agent patterns: group chat, fan-out, handoff/delegation, planner, reflection, state graph, workflow. |
| [Skills](./architecture/skills.md) | First-class skills — definition, discovery, DFS resolution, merge into agents, `dispatch_skill`. |
| [Config](./architecture/config.md) | Spec-driven YAML: `AgentSpec`/`TeamSpec`/`WorkflowSpec`, secret providers, residency, project defaults. |
| [Entities](./architecture/entities.md) | The append-only entity registry, deterministic ids, lineage, integrity, projection. |

## Capabilities

| Doc | What it covers |
| --- | --- |
| [Toolkit](./architecture/toolkit.md) | The 17 built-in tool packs, the `ToolPack` mechanism, declarative HTTP tools, sub-agent spawn. |
| [Tools service](./services/tools.md) | `ToolService` dispatch pipeline, registry, SSRF/security, validation, the `BoundTool`/`ToolExecutor` seam. |
| [MCP](./services/mcp.md) | SDK-free stdio JSON-RPC client and how MCP servers become tool sources. |
| [Connectors & Nepal](./services/connectors.md) | News (RSS/Atom), NRB forex/macro, and Nepal (Bikram Sambat / Devanagari) localization. |

## State & inference plane

| Doc | What it covers |
| --- | --- |
| [Inference](./services/inference.md) | `InferenceService`, the client-manager pool, providers (stub/Ollama/Claude CLI/gateway), routing, caching, streaming, pricing. |
| [Storage](./services/storage.md) | `StorageService` facade, per-concern stores, in-memory + Postgres backends, optional encryption. |
| [Memory](./services/memory.md) | Long-term memory: remember/recall, bi-temporal semantics, consolidation, context adapter. |
| [Observability](./services/observability.md) | `RunEvent`/`EventType`/`EventSink`, the SQLite trace store, the opt-in Logfire/OTel bridge. |

## Quality & knowledge plane

| Doc | What it covers |
| --- | --- |
| [Evaluation](./services/evaluation.md) | `EvaluationService`, deterministic metrics, veto gates, LLM-judge/embedding signals, ECE calibration. |
| [Benchmark](./architecture/benchmark.md) | The benchmark harness, graders, Wilson-interval stats, scorecards, and CI floor-gating. |
| [Knowledge](./services/knowledge.md) | Ingestion + hybrid RAG (BM25 + dense + RRF + optional rerank), the embedder seam. |
| [Context](./services/context.md) | How a run's prompt context snapshot is assembled and where evidence comes from. |
| [Prompts](./services/prompts.md) | `PromptManager`, versioned templates, the context→prompt mapping. |

## Enterprise controls

| Doc | What it covers |
| --- | --- |
| [Sandbox](./services/sandbox.md) | `off`/`subprocess`/`container` code execution, container hardening, gVisor/Kata seam. |
| [Guardrails](./services/guardrails.md) | Guardrail pipeline, PII/injection/blocklist built-ins, DLP + reversible tokenization. |
| [Governance](./services/governance.md) | Retention, crypto-shredding for right-to-erasure, erasure tombstones. |
| [Audit](./services/audit.md) | `SecurityEvent` records, signed bundles / SIEM export, tamper-evidence. |
| [CLI & Studio](./architecture/cli-and-studio.md) | The `himmy` CLI subcommands and the local Studio web GUI. |

## Existing design notes

These pre-date this doc set and remain canonical for their topics:

- [Record, replay & compaction](./design/record_replay_and_compaction.md)
- [Skills system design](./design/skills_system.md)
- [Enterprise hardening plan (WS1–WS6)](./enterprise/HARDENING_PLAN.md)
- [Sandbox backends](./enterprise/sandbox_backends.md)
- [Advanced usage](./advanced.md)
