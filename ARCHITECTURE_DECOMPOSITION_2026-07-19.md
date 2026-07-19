# Himmy Architecture Decomposition — 2026-07-19 (P3)

Design + first-increment record for the P3 "architecture" tier of the engineering-excellence
audit. **Prime directive: behavior preservation** — every step keeps public class names, public
method signatures, construction, event ordering, prompt bytes, and error taxonomy byte-identical;
only internal structure changes. Each step must leave the full offline suite green.

## Status: BOTH god-object decompositions COMPLETE (6232 tests green, mypy + ruff clean)

The full staged plan (Blueprints A & B below) has been **executed to completion**, one collaborator at a
time, suite-green after each. Result: **`single_agent.py` 4157 → 2281 LOC** and **`services.py` 4052 → 2411 LOC**
(both now thin facades that hold the public API + delegating shims), with the logic relocated into 14 focused
collaborator modules. Public class names, method signatures, construction, event order, prompt bytes, and
error taxonomy are byte-identical throughout.

**Extracted collaborators:**
- Runtime (`himmy/runtime/`): `snapshot.py` (SnapshotResolver), `tool_exchange.py` (ToolExchange), `prompt_assembly.py`
  (RequestBuilder), `compaction.py` (CompactionRunner, extended), `audit.py` (AuditEmitter — absorbed the earlier
  `_EntityRegistrar`, which was deleted), `loop.py` (LoopDriver), `resume.py` (ResumeCoordinator), `streaming.py` (StreamDriver).
- Application (`himmy/application/`): `run_context.py` (_RunContext, one shared `_tasks` set), `run_reads.py` (RunReadService),
  `run_side_effects.py` (RunSideEffects), `run_retry.py` (RetryPolicyEngine), `run_recovery.py` (RunRecovery),
  `run_enqueue.py` (RunEnqueuer), `run_drive.py` (RunDriveEngine + HITL + orchestration core), plus the earlier `workspace_quota.py`.

**Also landed earlier in P3:** removed the stale `build/lib/himmy` copy (422 untracked files); storage aux-store
`ConversationStoreProtocol` + conformance test + the `PostgresConversationStore.prune` drift-fix; fixed the
entities→services layering leak (`spine_factory.py`).

**Landmines respected (verified):** `_apply_guardrail` never moved; every test-poked private kept a byte-identical class-level
shim (re-exported names carry `# noqa: F401`); one `_tasks` set on `_RunContext` (drain); `orchestration_runner` stayed a
function-local import (cycle); stream delegation uses `async for … yield` (early-close); the tool-authorizer confused-deputy
gate moved intact into `run_drive.py` (the gate-coverage AST guard was re-pointed there).

## Deferred by design (with reasons)

- **web_search unification:** the two `web_search` providers have **divergent public contracts** (toolkit: `HIMMY_SEARCH_API_KEY`,
  arg `max_results`, keyless-DDG-always-available; connector: `HIMMY_TAVILY/BRAVE_API_KEY`, arg `count`). Unifying them would change a
  public tool contract, violating behavior-preservation. Needs a deliberate contract-reconciliation decision — not a silent refactor.
- **Remaining aux-store protocols** (Checkpoint/GraphCheckpoint/Binding/Routines/Calendar/Cookbook/Notes/Tasks/Memory) — additive,
  all twins already name-match; add per Blueprint C when convenient.

---

## Blueprint A — SingleAgentRuntime (`himmy/runtime/single_agent.py`, ~4157 LOC, 66 methods)

Extract cohesive method-groups into collaborators the runtime constructs and delegates to. `_EntityRegistrar` is done (folds into AuditEmitter later). Order (each step suite-green):

| # | Collaborator (module) | Owns | Risk |
|---|---|---|---|
| 0 | (pure-helper moves) | cache-key fns (:405-476)→prompt_assembly; compaction scrub/`_auto_compact_default_spec`→compaction.py | low |
| 1 | `SnapshotResolver` (`snapshot.py`) | `_resolve_snapshot`, `_snapshot_grounding` | **lowest — do first, not test-poked** |
| 2 | `ToolExchange` (`tool_exchange.py`) | tool replay/retry/result-capping (`_append_tool_messages`†, `_wrap_executor_with_retry`, `_cap_tool_result_for_model`) | med (1 poke) |
| 3 | `RequestBuilder` (`prompt_assembly.py`) | prompt+tools+request assembly (`_build_request`†, `_prompt_cache_policy`†, `_render_*`, `_effective_model_key`) | med (13 pokes) |
| 4 | `CompactionRunner` (`compaction.py`) | `_maybe_compact`† body | med (19 pokes, 5-arg sig) |
| 5 | `AuditEmitter` (`audit.py`) | `_emit` fan-out + registration/lineage/save (absorbs `_EntityRegistrar`) | **high fanout — full suite** |
| 6 | `LoopDriver` (`loop.py`) | `_drive_loop` stop-ladder, `_maybe_synthesize`, `_route_*`, `_drain_steer_queue` | med |
| 7 | `ResumeCoordinator` (`resume.py`) | HITL resume/checkpoint (`_resume_agent_loop_locked`, `_save_checkpoint`) | med |
| 8 | `StreamDriver` (`streaming.py`) | streaming helpers + `stream_task/stream_agent_loop` bodies | **extract LAST — async early-close** |

† = keep a byte-identical class-level delegating shim (tests bind/read it).

**Non-negotiable landmines:**
- `_apply_guardrail` must **NOT** move — `test_guardrail_offload.py` rebinds `SingleAgentRuntime._apply_guardrail.__get__(stub)`; `_emit` must stay a real class method.
- Keep on the runtime: `_pending_approvals` (called as a **staticmethod**), and attributes tests read directly (`_consent_decider`, `_input_guardrail`, `_output_guardrail`, `_resume_locks`). Collaborators back-reference them; they do not migrate.
- **Async-generator early-close:** stream delegation must be `async for d in inner: yield d` so `GeneratorExit`/`CancelledError` propagates into the inner `finally → aclose()`.
- **Event ordering + `_CURRENT_SUBJECT` contextvar scope** are load-bearing (`AGENT_RUN_STARTED → tool-replay → lineage → AGENT_RUN_FINISHED`).
- The stop/cost-budget ladder is **duplicated** in `_drive_loop` and `_stream_drive_loop`; if unified, assert both still emit identical `stopped_reason` strings.

## Blueprint B — RunAppService (`himmy/application/services.py`, ~4052 LOC, ~50 lifecycle methods)

`WorkspaceQuota`/`WorkspaceAdmission` is done. Order (each step suite-green):

| # | Collaborator (module) | Owns | Risk |
|---|---|---|---|
| 0 | (import hoist) | 34 cycle-safe function-local imports → module top (keep `orchestration_runner` local) | low |
| 1 | `_RunContext` | shared handles + **the single `_tasks` set** + live-read tunables | med (foundation) |
| 2 | `RunReadService` (`run_reads.py`) | tenant-scoped reads (`get_run`, `list_runs`, `count_runs`, `get_run_events`, …) | **safe leaf — first** |
| 3 | `RunSideEffects` (`run_side_effects.py`) | authz/subject-scope/sink/lineage helpers | safe |
| 4 | `RetryPolicyEngine` (`run_retry.py`) | `_apply_retry_policy` + `_is_transient_run_error` + backoff constants | safe |
| 5 | `WorkspaceAdmission` (`workspace_admission.py`) | **sole owner** of quota state (extends the shipped `WorkspaceQuota`) | med (public surface) |
| 6 | `RunRecovery` (`run_recovery.py`) | `drain`, `sweep_stuck_runs`, `reconcile_resolving_runs` (via shared `_tasks`) | med |
| 7 | `RunEnqueuer` (`run_enqueue.py`) | `create_run`, `continue_thread`, `_stamp_queue_fields`, `_launch_or_enqueue` | med |
| 8 | `RunDriveEngine` + `HitlResumeCoordinator` + `OrchestrationExecutor` | the coupled execution/HITL/orchestration core — extract **together, last**; defer any subset not green | **risky** |

**Non-negotiable landmines:**
- **One `_tasks` set** on `_RunContext` — `drain()` must cancel resume/execute/orchestration tasks; a collaborator owning its own set → shutdown hang + lost cancel→FAILED.
- `enable_dispatch()` mutates `_dispatch_enabled/_run_timeout_seconds/_default_max_attempts/_dispatch_fairness` **at runtime** — collaborators must read these **live** from the context, never snapshot at construction.
- `himmy.application.orchestration_runner` stays **function-local** even inside `application.*` collaborators (re-entering `application/__init__` → partial-import `ImportError`).
- The **ambient tool-authorizer contextvar** (`set → reset` in try/finally) must wrap the **entire** drive when moved — a misplaced reset re-opens the confused-deputy gate on the approved write path.
- Public byte-identity: dispatcher + 7 routers + CLI + `application/__init__` re-exports depend on exact signatures + exception types (`WorkspaceRunQuotaExceeded`, `RunNotApprovableError`, `HitlNotSupportedError`, `HitlRequiresAgentError`) + the OpenAPI snapshot.

## Blueprint C — Storage aux-store shared contract (`protocols.py` + conformance test)

Aux stores are a **2-way** SQLite↔Postgres mirror (the SQLite twins live in `api/*`, `runtime/checkpoint.py`, `services/memory/store.py`, `conversations.py` — not `sqlite.py`). `ConversationStoreProtocol` + conformance test + the `PostgresConversationStore.prune` drift-fix landed this pass. Remaining protocols to add (all twins already name-match, so additive + immediately green): Checkpoint, GraphCheckpoint, Binding (Teams+Workflows), Routines, Calendar, Cookbook, Notes, Tasks, Memory.

**Caveats:** the contract is **method-presence only** (`Any` signatures, `issubclass`) — it catches a missing method, **not** a signature/semantic drift (e.g. `delete(team_id)` vs `(record_id)`). `NotifyStore` is one-sided (no SQLite class twin — module functions in `studio_notify.py`), so its parity is unenforceable until that side is class-ified (separate change). Postgres prune SQL is verified via a fake-pool harness here; real JSONB/rowcount semantics still need the live-PG CI lane.
