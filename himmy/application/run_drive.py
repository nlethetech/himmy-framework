"""The coupled run execution / HITL-resume / orchestration CORE for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the FINAL, riskiest slice of the staged
decomposition of ``RunAppService`` (LANE runapp, step 8). It fuses three tightly-coupled
responsibilities that share the same drive primitives, terminal finalizers, and the ambient
tool-authorizer scope, so they are extracted TOGETHER rather than as three separable objects:

- **RunDriveEngine** — the run-drive core: ``_execute_run`` (the background worker),
  ``_execute_on_runtime`` (+``_inner``, the ambient-authorizer-wrapped drive body),
  ``_finalize_succeeded_run``, ``_resolve_runtime`` (shared vs per-run tool-bearing runtime),
  ``_parse_structured``, ``dispatch_claimed_run`` (the Q3 leased-claim executor with retry),
  and ``_dispatch_orchestration_run``.
- **HitlResumeCoordinator** — the HITL approve/reject lifecycle: ``pending_approvals``,
  ``resume_run`` (the atomic AWAITING_APPROVAL -> RESOLVING claim), ``_resume_in_background``,
  ``_drive_hitl_run`` / ``_apply_loop_outcome`` (the pause-again / succeed / fail projection),
  and the orchestration-resume band (``_resume_orchestration`` /
  ``_resume_orchestration_in_background`` / ``_apply_orchestration_outcome`` /
  ``_resolve_graph_checkpoint_store``).
- **OrchestrationExecutor** — the team/workflow path: ``create_orchestration_run`` and
  ``_execute_orchestration_run``.

Behaviour is BYTE-IDENTICAL to the former inline methods: the dispatch ordering, the
lease/idempotency/HITL exactly-once semantics (the run-level ``claim_run_for_resume`` CAS + the
member checkpoint ``claim()`` backstop), the error taxonomy
(``WorkspaceRunQuotaExceeded`` / ``RunNotApprovableError`` / ``HitlNotSupportedError``), the
ambient tool-authorizer ``set``/``reset`` lifetime wrapping the ENTIRE drive, and the event
ordering are all preserved. Every store / runtime / dispatch-tunable / timeout handle is read
LIVE through the shared :class:`_RunContext` (never a construction-time snapshot, so
``enable_dispatch`` and a re-pointed store are observed at once); the per-workspace outstanding
cap + execution semaphores are read LIVE from the shared :class:`WorkspaceQuota`; and the
cross-cutting side effects (tool-authz gate, subject scope, lineage/conversation projection),
the leased-retry policy, and the tenant-scoped reads are delegated to the SAME collaborators the
service holds. ``himmy.application.orchestration_runner`` stays FUNCTION-LOCAL (re-entering
``application/__init__`` at import time would partial-import ``ImportError``).

``RunAppService``'s former methods delegate here as thin shims (the public + test-poked ones);
the purely-internal drive/HITL/orchestration methods now live ONLY here, so every router,
dispatcher, CLI path, and test caller stays byte-identical.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from himmy.application.run_side_effects import RunSideEffects
from himmy.application.services import (
    HitlNotSupportedError,
    RunNotApprovableError,
    _is_resume_claim_loss,
    _maybe_await,
    _now,
    _requested_schema,
    _validate_structured,
    logger,
)
from himmy.application.workspace_quota import WorkspaceRunQuotaExceeded
from himmy.config.spec_sanitizer import sanitize_tenant_spec
from himmy.core.errors import HimmyError
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    AgentDefRecord,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.run_input import RunInputError, decode_run_input
from himmy.services.storage.run_lane import LANE_DEFAULT

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.application.run_context import _RunContext
    from himmy.application.run_reads import RunReadService
    from himmy.application.run_retry import RetryPolicyEngine
    from himmy.application.workspace_quota import WorkspaceQuota
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.models import LLMConfig
    from himmy.services.storage.service import StorageService


class RunDriveEngine:
    """The coupled run-drive + HITL-resume + orchestration executor core.

    Holds no lifecycle state of its own: every handle is read LIVE through the shared
    :class:`_RunContext` + :class:`WorkspaceQuota`, and the cross-cutting side effects /
    retry policy / tenant-scoped reads are delegated to the SAME collaborators the service
    constructs. The backward-compatible ``self._X`` facade below (mirroring the service's own
    context/quota views) lets every extracted method body stay BYTE-IDENTICAL to its former
    inline form.
    """

    def __init__(
        self,
        *,
        context: _RunContext,
        ws_quota: WorkspaceQuota,
        side_effects: RunSideEffects,
        retry: RetryPolicyEngine,
        reads: RunReadService,
    ) -> None:
        """Wire the shared context + quota (live handles) and the peer collaborators."""
        self._ctx = context
        self._ws_quota = ws_quota
        self._side_effects = side_effects
        self._retry = retry
        self._reads = reads

    # -- Live views over the shared context/quota, mirroring the service's facade so the
    # extracted method bodies below read exactly the same names they always did.
    @property
    def _runtime(self) -> SingleAgentRuntime:
        return self._ctx.runtime

    @property
    def _storage(self) -> StorageService:
        return self._ctx.storage

    @property
    def _recommendations(self) -> Any:
        return self._ctx.recommendations

    @property
    def _checkpoint_store(self) -> Any:
        return self._ctx.checkpoint_store

    @property
    def _agent_resolver(self) -> Any:
        return self._ctx.agent_resolver

    @property
    def _graph_checkpoint_store_provider(self) -> Any:
        return self._ctx.graph_checkpoint_store_provider

    @property
    def _tasks(self) -> Any:
        return self._ctx.tasks

    @property
    def _run_timeout_seconds(self) -> float:
        return self._ctx.run_timeout_seconds

    @property
    def _default_max_attempts(self) -> int:
        return self._ctx.default_max_attempts

    @property
    def _dispatch_enabled(self) -> bool:
        return self._ctx.dispatch_enabled

    @property
    def _workspace_max_outstanding(self) -> int:
        return self._ws_quota.max_outstanding

    # -- Delegating mirrors of the service helpers the extracted bodies call.
    def _workspace_semaphore(self, workspace_id: str) -> asyncio.Semaphore:
        return self._ws_quota.semaphore(workspace_id)

    def _release_workspace_run(self, workspace_id: str) -> None:
        self._ws_quota.release(workspace_id)

    def _admit_workspace_run(self, workspace_id: str) -> None:
        self._ws_quota.admit(workspace_id)

    def _build_tool_authorizer(self, actor: dict[str, Any] | None) -> Any:
        return self._side_effects.build_tool_authorizer(actor)

    @staticmethod
    def _subject_scope_from_actor(actor: dict[str, Any] | None) -> str | None:
        return RunSideEffects.subject_scope_from_actor(actor)

    async def _project_run_agent_link(
        self, run: RunRecord, agent_def: AgentDefRecord
    ) -> None:
        await self._side_effects.project_run_agent_link(run, agent_def)

    async def _notify_conversation_sink(self, run: RunRecord) -> None:
        await self._side_effects.notify_conversation_sink(run)

    async def _apply_retry_policy(self, run_id: str) -> None:
        await self._retry.apply_retry_policy(run_id)

    async def get_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> RunRecord | None:
        return await self._reads.get_run(run_id, workspace_id=workspace_id)

    async def _execute_run(
        self,
        run_id: str,
        *,
        workspace_id: str,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None = None,
        agent_def: AgentDefRecord | None = None,
        hitl: bool = False,
        plan: bool = False,
        thread: ChatThread | None = None,
    ) -> None:
        """Background worker: RUNNING -> run_task -> SUCCEEDED/FAILED + extraction.

        Reads the terminal :class:`RunResult` status (invariant #4 / AAEO-3): a
        FAILED inference response is recorded as a FAILED run with ``run.error``
        populated and recommendation extraction skipped, instead of being marked
        SUCCEEDED with garbage output. The whole run is bounded by
        ``run_timeout_seconds`` (AAEO-1).

        Runtime selection (T0.2): with ``agent_spec`` set, the run executes on a
        PER-RUN tool-bearing runtime built from the spec (so the agent's tools fire);
        otherwise it stays on the shared tool-less runtime (inline-persona
        back-compat). Execution holds the per-workspace concurrency semaphore (T0.4)
        and the outstanding-run reservation taken in :meth:`create_run` is released
        in ``finally`` so a failed/cancelled run frees its slot. ``workspace_id`` is
        passed in (not re-read from the record) so the slot is always released for the
        right workspace even on the defensive ``run is None`` path.
        """
        semaphore = self._workspace_semaphore(workspace_id)
        try:
            async with semaphore:
                run = await self._storage.get_run(run_id)
                if run is None:  # pragma: no cover - defensive
                    return
                await self._execute_on_runtime(
                    run,
                    persona=persona,
                    task=task,
                    llm_config=llm_config,
                    agent_spec=agent_spec,
                    agent_def=agent_def,
                    hitl=hitl,
                    plan=plan,
                    thread=thread,
                )
        finally:
            self._release_workspace_run(workspace_id)


    async def dispatch_claimed_run(self, run: RunRecord) -> None:
        """Execute a leased-queue run the dispatcher just CLAIMED, with retry/backoff (Q3).

        The dispatcher hands this a run already flipped to RUNNING with a fresh lease (the Q2
        ``claim_next_queued_run`` CAS) and ``attempt`` incremented. This:

        1. REHYDRATES the recoverable launch input from ``run.input_blob`` (the Q0 blob the
           enqueue persisted). A run with no blob (legacy / non-recoverable) cannot be
           re-executed from a fresh process, so it is failed with a clear reason rather than
           silently dropped.
        2. RE-RESOLVES the stored ``agent_def`` from ``metadata['agent_id']`` (for the
           run<->agent lineage edge) via the same resolver the resume path uses.
        3. Drives :meth:`_execute_on_runtime` under the per-workspace concurrency semaphore.
        4. On a TRANSIENT failure (provider blip, timeout, model-not-loaded) RE-QUEUES the run
           with exponential backoff while attempts + age remain; once the budget is exhausted
           (or the failure is PERMANENT) it leaves the terminal FAILED set by the runtime,
           or PARKS it so an operator can ``redrive``. A SUCCEEDED / AWAITING_APPROVAL /
           RESOLVING outcome is left as-is (a paused HITL run is NOT a dispatcher failure).

        The lease-renewal heartbeat is run by the dispatcher as a sibling sub-task, not here.
        """
        run_id = run.run_id
        workspace_id = run.workspace_id
        # Orchestration (team/workflow/graph) runs reconstruct from member_agent_ids + the
        # persisted prompt, not a single-agent input_blob — route them to their own driver.
        if (run.metadata or {}).get("orchestration"):
            await self._dispatch_orchestration_run(run)
            await self._apply_retry_policy(run_id)
            return
        # 1. rehydrate the recoverable input.
        if not run.input_blob:
            run.status = RunStatus.FAILED
            run.error = (
                "run has no recoverable input (input_blob missing); cannot be "
                "re-executed by the dispatcher"
            )
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        try:
            rinput = decode_run_input(run.input_blob, run_id=run_id)
        except RunInputError as exc:
            run.status = RunStatus.FAILED
            run.error = f"run input could not be rehydrated: {exc}"
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        # 2. re-resolve the stored agent_def (lineage edge) — best-effort; a removed agent
        # just means no run<->agent link, not a failure.
        agent_def: AgentDefRecord | None = None
        agent_id = (run.metadata or {}).get("agent_id")
        if agent_id and self._agent_resolver is not None:
            try:
                agent_def = await _maybe_await(
                    self._agent_resolver(agent_id, workspace_id=workspace_id)
                )
            except Exception:  # noqa: BLE001 - resolver failure must not crash the worker
                agent_def = None

        # 3. execute under the per-workspace concurrency semaphore (same back-pressure as the
        # inline path). A continuation carries thread_id == its conversation id, so reload the
        # prior thread to continue it; a fresh run starts a new thread.
        thread: ChatThread | None = None
        if run.thread_id:
            try:
                thread = await self._storage.load_thread(run.thread_id)
            except Exception:  # noqa: BLE001 - a missing thread just starts fresh
                thread = None

        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            await self._execute_on_runtime(
                run,
                persona=rinput.persona,
                task=rinput.task,
                llm_config=rinput.llm_config,
                agent_spec=rinput.agent_spec,
                agent_def=agent_def,
                hitl=rinput.hitl,
                plan=rinput.plan,
                thread=thread,
            )

        # 4. retry/backoff/PARK on a transient failure.
        await self._apply_retry_policy(run_id)


    async def _dispatch_orchestration_run(self, run: RunRecord) -> None:
        """Reconstruct + drive a CLAIMED orchestration run from a fresh process (Q3).

        The team/workflow/graph run carries its member ids + the persisted prompt in metadata
        (not a single-agent input_blob), so recovery re-resolves the members via the same
        resolver the resume path uses and rebuilds the graph checkpoint store from the
        provider. A member that has since been removed fails the run with a clear reason
        (the retry policy then classifies it permanent). Delegates the actual orchestration to
        :meth:`_execute_orchestration_run`, which sets the terminal/AWAITING state.
        """
        meta = run.metadata or {}
        member_agent_ids = list(meta.get("member_agent_ids") or [])
        prompt = meta.get("orchestration_prompt", "")
        kind = meta.get("orchestration_kind", "graph")
        resource_kind = meta.get("orchestration", "workflow")
        operator_provisioned = bool(meta.get("operator_provisioned", False))
        graph_resume_id = meta.get("graph_resume_id")

        if not member_agent_ids or self._agent_resolver is None:
            run.status = RunStatus.FAILED
            run.error = "orchestration run cannot be recovered: no resolvable members"
            run.last_error = run.error
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        members: list[AgentDefRecord] = []
        for agent_id in member_agent_ids:
            rec = await _maybe_await(
                self._agent_resolver(agent_id, workspace_id=run.workspace_id)
            )
            if rec is None:
                run.status = RunStatus.FAILED
                run.error = f"orchestration recovery failed: member agent {agent_id} removed"
                run.last_error = run.error
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            members.append(rec)

        await self._execute_orchestration_run(
            run.run_id,
            workspace_id=run.workspace_id,
            kind=kind,
            members=members,
            prompt=prompt,
            resource_kind=resource_kind,
            operator_provisioned=operator_provisioned,
            graph_checkpoint_store=self._resolve_graph_checkpoint_store(),
            graph_resume_id=graph_resume_id,
        )


    async def _execute_on_runtime(
        self,
        run: RunRecord,
        *,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        agent_def: AgentDefRecord | None = None,
        hitl: bool = False,
        plan: bool = False,
        thread: ChatThread | None = None,
    ) -> None:
        """Drive one run to a terminal state on the resolved runtime (T0.2 core).

        With ``hitl=True`` (T2f) the run drives :meth:`SingleAgentRuntime.run_agent_loop`
        with ``hitl=True`` on a per-run runtime carrying /v1's checkpoint store; if it
        pauses on an approval-gated tool the run goes AWAITING_APPROVAL (carrying the
        checkpoint id) and STOPS — never swept, resumable via :meth:`resume_run`. The
        non-HITL path is byte-identical to before (single ``run_task_detailed`` turn).

        ``plan=True`` (T2g) is an agentic, pausable run too: the per-run runtime carries
        the gated ``update_plan`` tool so the loop pauses at PLAN-READY through the same
        approval machinery. ``thread`` (T2g) is the continuation seam — a prior
        :class:`ChatThread` the loop continues so the model sees earlier turns; ``None``
        starts a fresh thread (every pre-T2g call site).
        """
        # centralize-tool-gate: bind this run's tool-capability gate AMBIENTLY for the whole
        # drive (set BEFORE _resolve_runtime so it propagates into the off-loop
        # build_runtime_for_spec and the background run-task — contextvars copy into
        # asyncio.to_thread/create_task). Even a runtime built by a sub-path that forgot to
        # thread the authorizer then consults the gate at the tool chokepoint. The explicit
        # threading into build_runtime_for_spec stays (authoritative for cross-process
        # recovery + sub-agent attenuation); this is the belt-and-braces ambient layer. A
        # None authorizer (no RBAC policy wired / offline) binds an inert "no ambient gate"
        # scope, byte-unchanged.
        from himmy.services.tools.ambient import use_tool_authorizer

        ambient_authorizer = self._build_tool_authorizer(
            (run.metadata or {}).get("actor")
        )
        with use_tool_authorizer(ambient_authorizer):
            await self._execute_on_runtime_inner(
                run,
                persona=persona,
                task=task,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                hitl=hitl,
                plan=plan,
                thread=thread,
            )


    async def _execute_on_runtime_inner(
        self,
        run: RunRecord,
        *,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        agent_def: AgentDefRecord | None = None,
        hitl: bool = False,
        plan: bool = False,
        thread: ChatThread | None = None,
    ) -> None:
        """The run-drive body, executed inside the ambient-authorizer scope.

        Split out of :meth:`_execute_on_runtime` so the ambient tool-capability binding
        wraps the ENTIRE drive (runtime build + loop) with a single ``with`` block; all
        the original logic is unchanged below.
        """
        # HITL/plan runs pause into /v1's OWN checkpoint store; the plain single-turn path
        # passes None so the per-run runtime stays exactly as the T0.2 build wired it.
        agentic = hitl or plan
        checkpoint_store = self._checkpoint_store if agentic else None
        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        # T2g: a plan-first run prepends the plan nudge + binds ``update_plan`` so the
        # agent publishes (and pauses on) its plan first.
        if plan:
            from himmy.runtime.plan_mode import apply_plan_mode_to_task

            apply_plan_mode_to_task(task)

        try:
            runtime = await self._resolve_runtime(
                agent_spec,
                checkpoint_store=checkpoint_store,
                plan_mode=plan,
                workspace_id=run.workspace_id,
                actor=(run.metadata or {}).get("actor"),
            )
        except Exception as exc:  # noqa: BLE001 - spec wiring failure is terminal
            run.status = RunStatus.FAILED
            run.error = f"agent runtime build failed: {exc}"
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        # T2f/T2g: the agentic path drives the loop (which can PAUSE on an approval-gated
        # tool — the HITL gate OR the plan gate); the plain path is the single-turn fast
        # path, optionally continuing a prior thread (T2g).
        if agentic:
            await self._drive_hitl_run(
                run,
                persona,
                task,
                runtime,
                llm_config=llm_config,
                agent_def=agent_def,
                thread=thread,
                plan=plan,
            )
            return

        try:
            # Pass ``thread`` ONLY when continuing a prior conversation (T2g); the
            # fresh-run path omits it so the call is byte-identical to the pre-T2g signature
            # (``run_task_detailed(persona, task, llm_config=...)``) — back-compat for every
            # existing call site and test double.
            coro = (
                runtime.run_task_detailed(persona, task, thread, llm_config=llm_config)
                if thread is not None
                else runtime.run_task_detailed(persona, task, llm_config=llm_config)
            )
            result = await asyncio.wait_for(coro, timeout=self._run_timeout_seconds)
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = (
                f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
            )
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        except asyncio.CancelledError:
            # Shutdown drain / explicit cancel: record FAILED then re-raise so the
            # task unwinds as a cancellation (AAEO-1).
            run.status = RunStatus.FAILED
            run.error = "run cancelled"
            run.updated_at = _now()
            try:
                await self._storage.save_run(run)
            except Exception:  # pragma: no cover - best-effort during cancel
                pass
            raise
        except Exception as exc:  # noqa: BLE001 - terminal failure transition
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        thread = result.thread
        run.thread_id = thread.thread_id
        run.trace_id = result.trace_id

        # T2e: link the run's chat_thread hub -> the stored agent node so a run launched
        # by ``agent_id`` carries a durable run<->agent lineage edge (best-effort; the
        # run never fails because lineage projection did).
        if agent_def is not None:
            await self._project_run_agent_link(run, agent_def)

        # AAEO-3: honour the FAILED inference path. ``RunResult.succeeded`` is the
        # typed status surface; a failed run records the error and skips extraction.
        if not result.succeeded:
            run.status = RunStatus.FAILED
            run.error = result.error or (result.error_code or "inference failed")
            run.output_text = result.output_text or None
            # Q3: carry the structured error_code so the retry classifier can branch on the
            # authoritative inference taxonomy instead of substring-matching the free text.
            if result.error_code:
                run.metadata = {**(run.metadata or {}), "error_code": result.error_code}
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        await self._finalize_succeeded_run(run, result, task=task, llm_config=llm_config)
        await self._notify_conversation_sink(run)


    async def _finalize_succeeded_run(
        self,
        run: RunRecord,
        result: Any,
        *,
        task: Task,
        llm_config: LLMConfig | None,
    ) -> None:
        """Record a terminal-SUCCEEDED run + extract recommendations (shared path).

        Factored out so both the single-turn path and the HITL loop's terminal turn
        finish identically (structured-output parse, AAEO-6 schema validation, the
        SUCCEEDED transition, and recommendation extraction).
        """
        run.output_text = result.output_text or None
        # Prefer the typed structured output; fall back to parsing the text.
        structured = result.output_structured
        if structured is None:
            structured = self._parse_structured(result.output_text)
        run.output_structured = structured

        # AAEO-6: validate the structured output against the requested schema
        # before extraction, recording any failure on the run.
        schema = _requested_schema(llm_config, task)
        if structured is not None and schema is not None:
            error = _validate_structured(structured, schema)
            if error is not None:
                run.metadata = {
                    **(run.metadata or {}),
                    "extraction_error": f"schema validation failed: {error}",
                }

        run.status = RunStatus.SUCCEEDED
        run.updated_at = _now()
        await self._storage.save_run(run)

        # Auto-extract recommendations when the output matches the envelope.
        if run.output_structured is not None:
            await self._recommendations.extract_from_run(run)


    async def _drive_hitl_run(
        self,
        run: RunRecord,
        persona: Persona,
        task: Task,
        runtime: SingleAgentRuntime,
        *,
        llm_config: LLMConfig | None,
        agent_def: AgentDefRecord | None,
        thread: ChatThread | None = None,
        plan: bool = False,
    ) -> None:
        """Drive a hitl/plan run's agentic loop; pause to AWAITING_APPROVAL on a gate.

        Runs :meth:`SingleAgentRuntime.run_agent_loop` (``hitl=True``) bounded by the
        per-run timeout. The loop either:

        * pauses on an approval-gated tool — ``stopped_reason == 'awaiting_approval'`` —
          in which case the run is stamped AWAITING_APPROVAL + ``metadata['checkpoint_id']``
          and STOPS (never swept, resumable via approve/reject), or
        * completes — handled exactly like the single-turn success/failure paths via the
          shared finalizers.

        ``thread`` (T2g) continues a prior conversation; ``plan`` marks a plan-first run
        so :meth:`_apply_loop_outcome` extracts the published plan into
        ``metadata['plan']`` when the run pauses at PLAN-READY.
        """
        try:
            # Pass ``thread`` only on a continuation (T2g) so a fresh agentic run's call is
            # byte-identical to the pre-T2g signature (back-compat for test doubles).
            loop_coro = (
                runtime.run_agent_loop(
                    persona, task, thread, llm_config=llm_config, hitl=True
                )
                if thread is not None
                else runtime.run_agent_loop(
                    persona, task, llm_config=llm_config, hitl=True
                )
            )
            loop = await asyncio.wait_for(
                loop_coro, timeout=self._run_timeout_seconds
            )
        except TimeoutError:
            run.status = RunStatus.FAILED
            run.error = (
                f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
            )
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        except asyncio.CancelledError:
            run.status = RunStatus.FAILED
            run.error = "run cancelled"
            run.updated_at = _now()
            try:
                await self._storage.save_run(run)
            except Exception:  # pragma: no cover - best-effort during cancel
                pass
            raise
        except Exception as exc:  # noqa: BLE001 - terminal failure transition
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.updated_at = _now()
            await self._storage.save_run(run)
            return

        await self._apply_loop_outcome(
            run, loop, task=task, llm_config=llm_config, agent_def=agent_def, plan=plan
        )


    async def _apply_loop_outcome(
        self,
        run: RunRecord,
        loop: Any,
        *,
        task: Task,
        llm_config: LLMConfig | None,
        agent_def: AgentDefRecord | None,
        plan: bool = False,
    ) -> None:
        """Project a finished/paused :class:`AgentLoopResult` onto the run record (T2f).

        Shared by the initial HITL drive and the resume path so a run pauses-again,
        succeeds, or fails identically regardless of which entry produced the loop.

        ``plan`` (T2g) makes a PLAN-READY pause stamp the published plan steps into
        ``metadata['plan']`` so a caller can read the proposed plan before approving.
        """
        thread = loop.thread
        run.thread_id = thread.thread_id
        # The loop's trace id is derived as ``{thread_id}:{task_id}``; record it so the
        # canonical event replay (``get_run_events``) finds the turn/tool events.
        run.trace_id = f"{thread.thread_id}:{task.task_id}"

        # T2e: link the run's chat_thread hub -> the stored agent node (best-effort).
        if agent_def is not None:
            await self._project_run_agent_link(run, agent_def)

        if loop.stopped_reason == "awaiting_approval":
            # The headline T2f transition: the run paused on an approval-gated tool.
            # Stamp the checkpoint id so approve/reject can re-claim it, and STOP — the
            # sweeper explicitly skips AWAITING_APPROVAL, so this run is never reaped.
            run.status = RunStatus.AWAITING_APPROVAL
            metadata = {
                **(run.metadata or {}),
                "checkpoint_id": loop.checkpoint_id,
            }
            # T2g: a plan-first run paused on its (gated) ``update_plan`` call — surface
            # the proposed plan in metadata so a caller reads it before approving.
            if plan:
                plan_steps = self._extract_plan_from_checkpoint(loop.checkpoint_id)
                if plan_steps:
                    metadata["plan"] = plan_steps
            run.metadata = metadata
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        final = loop.final
        if not final.succeeded:
            run.status = RunStatus.FAILED
            run.error = final.error or (final.error_code or "inference failed")
            run.output_text = final.output_text or None
            # Q3: carry the structured error_code for the retry classifier (see above).
            if final.error_code:
                run.metadata = {**(run.metadata or {}), "error_code": final.error_code}
            run.updated_at = _now()
            await self._storage.save_run(run)
            await self._notify_conversation_sink(run)
            return

        await self._finalize_succeeded_run(run, final, task=task, llm_config=llm_config)
        await self._notify_conversation_sink(run)


    async def _resolve_runtime(
        self,
        agent_spec: AgentSpec | None,
        *,
        checkpoint_store: Any = None,
        plan_mode: bool = False,
        workspace_id: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> SingleAgentRuntime:
        """Pick the runtime for a run: shared tool-less, or a per-run tool-bearing one.

        With ``agent_spec is None`` the existing shared (tool-less) runtime is
        returned — the inline-persona fast path stays byte-identical (back-compat).

        With a spec present (T0.2) a PER-RUN runtime is built via
        :func:`himmy.runtime.from_spec.build_runtime_for_spec`, which wires the spec's
        tool packs / tools / guardrails / knowledge / connectors / MCP + a tool
        service, so the run can finally CALL the agent's tools (impossible on the
        shared runtime, which carries no tool_service). It is built off-loop in a
        worker thread because ``build_runtime_for_spec`` may run an inner
        ``asyncio.run`` (knowledge ingest) that cannot nest in the running loop.

        Wiring choices that preserve the zero-config offline default: the per-run
        runtime SHARES this service's storage (so its thread/events/memory land in the
        one store the app layer reads) and REUSES the shared runtime's inference
        service when the spec pins no provider — so an offline deployment keeps the
        stub and a configured deployment keeps its gateway, with no surprise provider
        switch. When the spec names a provider explicitly, ``build_runtime_for_spec``
        honors it. ``checkpoint_store`` is threaded so a HITL run can pause (T2f).

        ``plan_mode`` (T2g) registers the APPROVAL-GATED ``update_plan`` tool into the
        per-run registry so a plan-first run pauses at PLAN-READY through the SAME
        approval machinery (it MUST be registered on the resume runtime too, hence this
        flag is threaded both on the initial drive and on resume).

        ``workspace_id`` (P1 tenancy) is the run's owning tenant, threaded into
        ``build_runtime_for_spec(subject=...)`` so a ``self_learning`` agent's tool-
        reputation mining is scoped to this tenant on the SHARED ``/v1`` event store
        instead of aggregating every tenant's tool failures.
        """
        if agent_spec is None:
            return self._runtime
        # Reuse the shared runtime's inference only when the spec does not pin its own
        # provider; otherwise let from_spec build the provider-specific service.
        shared_inference = (
            getattr(self._runtime, "inference_service", None)
            if not agent_spec.provider
            else None
        )
        from himmy.runtime.from_spec import build_runtime_for_spec

        # P0: rebuild the run principal's tool-capability gate from the persisted actor and
        # thread it into the per-run runtime (no-op offline / when no RBAC policy is wired).
        tool_authorizer = self._build_tool_authorizer(actor)

        # P1 tenancy (subject axis): under a subject_scoped per-user actor, namespace this
        # run's memory/KB/tasks/notes tool stores by the user so two users of ONE tenant never
        # read each other's facts/docs/tasks. The flag is persisted in ``actor`` by
        # ``Principal.actor_metadata`` (stamped only when actually subject_scoped + not a
        # tenant_admin), so a non-subject-scoped / offline run leaves the scope tenant-only
        # (``None``) — byte-for-byte unchanged.
        subject_scope = (
            actor.get("subject")
            if actor and actor.get("subject_scoped")
            else None
        )

        runtime, registry = await asyncio.to_thread(
            build_runtime_for_spec,
            agent_spec,
            inference=shared_inference,
            storage=self._storage,
            checkpoint_store=checkpoint_store,
            subject=workspace_id,
            subject_scope=subject_scope,
            tool_authorizer=tool_authorizer,
        )
        runtime = cast("SingleAgentRuntime", runtime)
        if plan_mode:
            # The plan tool must live on a tool registry; a spec with no tools builds no
            # registry, so resolve the runtime's own tool_service registry as the target.
            target = registry
            if target is None:
                target = getattr(
                    getattr(runtime, "tool_service", None), "registry", None
                )
            if target is not None:
                from himmy.runtime.plan_mode import register_plan_tool

                register_plan_tool(target)
        return runtime


    @staticmethod
    def _parse_structured(content: str | None) -> Any:
        """Parse JSON content into a structure, returning None on non-JSON text."""
        if not content:
            return None
        import json

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
        return None


    async def pending_approvals(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> list[dict[str, Any]] | None:
        """The redacted pending tool call(s) a HITL-paused run awaits (T2f).

        Tenant-scoped: a run outside ``workspace_id`` reads as None (404). Returns the
        list of ``{tool_name, args}`` for the checkpoint the run paused on, with secret-
        looking arg values masked (the same redaction Studio's approvals inbox uses), so
        a reviewer can see WHAT will run before approving without leaking a credential.
        None when the run is unknown/out-of-workspace; an empty list when the run carries
        no checkpoint (e.g. not actually paused) or the checkpoint has been resolved.
        """
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            return None
        if self._checkpoint_store is None:
            return []
        checkpoint_id = (run.metadata or {}).get("checkpoint_id")
        if not checkpoint_id:
            return []
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            return []
        from himmy.runtime.checkpoint import redact_tool_args

        return [
            {"tool_name": p.tool_name, "args": redact_tool_args(p.args)}
            for p in checkpoint.pending_tool_calls
        ]


    async def resume_run(
        self,
        run_id: str,
        *,
        approved: bool,
        workspace_id: str | None = None,
        actor: str = "human",
    ) -> RunRecord:
        """Approve/reject a HITL-paused run; resume it on a tracked bg task (T2f).

        Loads the run tenant-scoped (a 404 for unknown/out-of-workspace, a 409 for a
        terminal/non-paused run), then ATOMICALLY claims ``AWAITING_APPROVAL`` ->
        ``RESOLVING`` via :meth:`StorageService.claim_run_for_resume` — the run-level
        compare-and-set that mirrors the member checkpoint ``claim()``. A SECOND concurrent
        approve (a double-clicked Approve, two tabs, two workers) loses this CAS and is
        refused with :class:`RunNotApprovableError` (409) BEFORE launching any resume, so
        for an ORCHESTRATION run the graph advance — which has no claim of its own and
        could otherwise double-fire DOWNSTREAM members' tools — only ever happens once.

        The winner then REBUILDS its OWN per-run tool-bearing runtime FROM THE STORED DB
        ``AgentSpec`` (resolved by ``agent_id`` — /v1 has no filesystem ``agent_path`` to
        rebuild from, and ``resume_agent_loop`` HARD-requires a ``tool_service``) and
        launches :meth:`SingleAgentRuntime.resume_agent_loop` (or the orchestration graph
        resume) on a fresh tracked background task. The background task drives the run to
        SUCCEEDED / FAILED / AWAITING_APPROVAL-again. A resume that crashes mid-flight
        leaves the run at ``RESOLVING`` so startup recovery can re-drive it exactly-once
        (the member checkpoint ``claim()`` + idempotency ledger), distinct from this
        rejected "concurrent second click". Returns the in-progress record (fire-and-
        forget, mirroring :meth:`create_run`).
        """
        run = await self.get_run(run_id, workspace_id=workspace_id)
        if run is None:
            raise RunNotApprovableError(run_id, status="unknown")
        if run.status != RunStatus.AWAITING_APPROVAL:
            raise RunNotApprovableError(run_id, status=run.status.value)
        if self._checkpoint_store is None:  # pragma: no cover - guarded at create
            raise HitlNotSupportedError("no checkpoint store wired; cannot resume")

        # Validate that the resume CAN proceed BEFORE claiming, so a config error
        # (missing checkpoint / unresolvable agent) 409s without first stranding the run
        # in RESOLVING. Orchestration runs validate their member ids/graph checkpoint
        # inside ``_resume_orchestration`` (also pre-claim, below).
        is_orchestration = (run.metadata or {}).get("hitl_kind") == "orchestration"
        checkpoint_id = ""
        agent_def: AgentDefRecord | None = None
        if not is_orchestration:
            checkpoint_id = (run.metadata or {}).get("checkpoint_id") or ""
            if not checkpoint_id:  # pragma: no cover - an AWAITING run always has one
                raise RunNotApprovableError(run_id, status="no checkpoint")
            agent_id = (run.metadata or {}).get("agent_id")
            if not agent_id or self._agent_resolver is None:
                raise RunNotApprovableError(run_id, status="no resolvable agent")
            agent_def = await _maybe_await(
                self._agent_resolver(agent_id, workspace_id=run.workspace_id)
            )
            if agent_def is None:
                raise RunNotApprovableError(run_id, status="agent removed")

        # ATOMIC run-level claim: flip AWAITING_APPROVAL -> RESOLVING exactly once. The
        # non-atomic check above is only for a clean 404/409 message; THIS compare-and-set
        # is the authoritative gate. The loser of a concurrent double-approve fails the CAS
        # here and 409s without launching a resume — so an orchestration graph advance can
        # never double-fire downstream members' tools. (For a single-agent run the member
        # checkpoint claim() is a second backstop; for orchestration this is the ONLY gate
        # on the post-member graph advance.)
        claimed = await self._storage.claim_run_for_resume(
            run_id, workspace_id=run.workspace_id
        )
        if not claimed:
            raise RunNotApprovableError(run_id, status=RunStatus.RESOLVING.value)
        # Reflect the won claim on the local record (the background task flips RESOLVING ->
        # RUNNING when it actually starts; the returned record shows the in-progress state).
        # Persist the approve/reject decision + actor onto the run so a resume that CRASHES
        # mid-flight (leaving the run at RESOLVING) can be re-driven by startup recovery
        # with the SAME decision — exactly-once is preserved by the member checkpoint
        # claim() + idempotency ledger; only the decision (run the tool, or not) is needed.
        run.status = RunStatus.RESOLVING
        run.metadata = {
            **(run.metadata or {}),
            "resume_decision": "approved" if approved else "rejected",
            "resume_actor": actor,
        }
        run.updated_at = _now()
        await self._storage.save_run(run)

        # HITL ORCHESTRATION pause (a team/workflow member paused): a graph/workflow run
        # carries a member-agent-id LIST (not a single ``agent_id``) and resumes via the
        # durable graph splice, not the single-agent resume.
        if is_orchestration:
            return await self._resume_orchestration(run, approved=approved, actor=actor)

        # Flip RESOLVING -> RUNNING up front so the inbox reflects "in progress". The
        # authoritative exactly-once gates are the run-level claim above + the runtime
        # ``claim()`` (crash-retry replays the per-tool idempotency ledger).
        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        assert agent_def is not None  # noqa: S101 - narrowed: validated pre-claim above
        bg = asyncio.create_task(
            self._resume_in_background(
                run_id,
                checkpoint_id=checkpoint_id,
                approved=approved,
                actor=actor,
                agent_def=agent_def,
                workspace_id=run.workspace_id,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return run


    async def _resume_in_background(
        self,
        run_id: str,
        *,
        checkpoint_id: str,
        approved: bool,
        actor: str,
        agent_def: AgentDefRecord,
        workspace_id: str,
    ) -> None:
        """Background worker: rebuild the runtime from the DB spec + resume the loop (T2f).

        Holds the per-workspace concurrency semaphore (T0.4) for the duration, rebuilds a
        tool-bearing runtime FROM THE STORED SPEC carrying /v1's checkpoint store, and
        drives :meth:`SingleAgentRuntime.resume_agent_loop`. A loser of the exactly-once
        ``claim()`` race (``HimmyError('already resolved')``) is a NO-OP — the run is left
        at whatever the winner set it to (never re-failed, never re-run).
        """
        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            run = await self._storage.get_run(run_id)
            if run is None:  # pragma: no cover - defensive
                return
            # Re-sanitize the stored spec under the SAME operator status the run was
            # created with, so the approval-gated tool the run paused on is still present
            # when it executes (a tenant spec would have had its tools stripped at create,
            # so a paused run could only exist for an operator-provisioned/clean spec). If
            # the operator has since REVOKED the opt-in (env var unset between pause and
            # resume), the re-sanitize fail-closes — the resume becomes a clean FAILED,
            # never a crashed task and never a privileged-tool execution without the opt-in.
            operator_provisioned = bool(
                (run.metadata or {}).get("operator_provisioned", False)
            )
            # T2g: a plan-first run paused on its gated ``update_plan`` tool — the rebuilt
            # resume runtime MUST re-register that synthetic tool, else the now-approved
            # plan call has no handler to execute against.
            plan_mode = bool((run.metadata or {}).get("plan_mode", False))
            try:
                spec = sanitize_tenant_spec(
                    agent_def.agent_spec(),
                    operator_provisioned=operator_provisioned,
                ).spec
                runtime = await self._resolve_runtime(
                    spec,
                    checkpoint_store=self._checkpoint_store,
                    plan_mode=plan_mode,
                    workspace_id=run.workspace_id,
                    # Carry the PERSISTED launch actor (stamped at create into
                    # run.metadata['actor']) into the rebuilt runtime, mirroring the drive
                    # path above. Without it the tool-capability gate would rebuild
                    # NON-enforcing and the now-approved, side-effecting tool would execute
                    # with the gate DISABLED — re-opening the confused-deputy hole on the
                    # highest-value path (the approved write).
                    actor=(run.metadata or {}).get("actor"),
                )
            except Exception as exc:  # noqa: BLE001 - spec rebuild failure is terminal
                run.status = RunStatus.FAILED
                run.error = f"resume runtime build failed: {exc}"
                run.updated_at = _now()
                await self._storage.save_run(run)
                return

            try:
                loop = await asyncio.wait_for(
                    runtime.resume_agent_loop(
                        checkpoint_id, approved=approved, actor=actor
                    ),
                    timeout=self._run_timeout_seconds,
                )
            except HimmyError as exc:
                # The exactly-once loser: the checkpoint was already resolved by a
                # concurrent/earlier resume. This is a NO-OP — do NOT touch the run (the
                # winner owns its terminal state). Leaving it as-is means a double-approve
                # neither re-runs the tool nor flips a SUCCEEDED run to FAILED.
                logger.info(
                    "resume of run %s was a no-op (%s)", run_id, exc
                )
                return
            except TimeoutError:
                run.status = RunStatus.FAILED
                run.error = (
                    f"resume exceeded {self._run_timeout_seconds:.0f}s timeout"
                )
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            except asyncio.CancelledError:
                run.status = RunStatus.FAILED
                run.error = "resume cancelled"
                run.updated_at = _now()
                try:
                    await self._storage.save_run(run)
                except Exception:  # pragma: no cover - best-effort during cancel
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 - terminal failure transition
                run.status = RunStatus.FAILED
                run.error = str(exc)
                run.updated_at = _now()
                await self._storage.save_run(run)
                return

            # Reconstruct the task so the shared finalizers compute the right trace id +
            # validate structured output against the originally-requested schema.
            from himmy.agents.base_agent.task import Task as _Task

            task = _Task(title=run.persona_name or "resume", prompt="", context={})
            if run.task_id:
                task.task_id = run.task_id
            await self._apply_loop_outcome(
                run, loop, task=task, llm_config=None, agent_def=agent_def
            )


    async def _resume_orchestration(
        self, run: RunRecord, *, approved: bool, actor: str
    ) -> RunRecord:
        """Approve/reject a HITL-paused TEAM/WORKFLOW run; resume the graph (WI-6).

        By the time this runs the caller (``resume_run``) has already won the atomic
        run-level ``claim_run_for_resume`` CAS, flipping AWAITING_APPROVAL -> RESOLVING;
        that run-level claim is now the PRIMARY exactly-once gate — a concurrent second
        approve loses the CAS and 409s before ever reaching here, so the graph advance
        happens exactly once. This method flips RESOLVING -> RUNNING and launches the
        durable graph resume on a tracked background task. The MEMBER checkpoint
        ``claim()`` inside ``resume_agent_loop`` remains a defence-in-depth backstop (a
        crash re-drive that re-enters here finds the member already resolved -> no-op).
        Returns the RUNNING record.
        """
        graph_resume_id = (run.metadata or {}).get("orchestration_checkpoint_id")
        if not graph_resume_id:  # pragma: no cover - an orchestration pause always has one
            raise RunNotApprovableError(run.run_id, status="no orchestration checkpoint")
        member_agent_ids = list((run.metadata or {}).get("member_agent_ids") or [])
        if not member_agent_ids or self._agent_resolver is None:
            raise RunNotApprovableError(run.run_id, status="no resolvable members")

        run.status = RunStatus.RUNNING
        run.updated_at = _now()
        await self._storage.save_run(run)

        bg = asyncio.create_task(
            self._resume_orchestration_in_background(
                run.run_id,
                approved=approved,
                actor=actor,
                workspace_id=run.workspace_id,
                graph_resume_id=graph_resume_id,
                member_agent_ids=member_agent_ids,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return run


    def _resolve_graph_checkpoint_store(self) -> Any:
        """The durable graph checkpoint store to resume an orchestration from.

        Uses the surface-provided getter (file-backed in a server, so the SAME db the run
        paused into is reopened) and falls back to an in-memory store offline/in tests.
        """
        if self._graph_checkpoint_store_provider is not None:
            return self._graph_checkpoint_store_provider()
        from himmy.runtime.checkpoint import InMemoryGraphCheckpointStore

        return InMemoryGraphCheckpointStore()


    async def _resume_orchestration_in_background(
        self,
        run_id: str,
        *,
        approved: bool,
        actor: str,
        workspace_id: str,
        graph_resume_id: str,
        member_agent_ids: list[str],
    ) -> None:
        """Background worker: re-resolve members + drive the durable graph resume (WI-6).

        Holds the per-workspace concurrency semaphore for the duration, re-resolves and
        (via ``run_orchestration``) re-sanitizes the members under the run's operator
        status — so a revoked opt-in fails closed — rebuilds the graph member runtime with
        BOTH the member and graph checkpoint stores wired, and resumes the graph. The
        outcome is projected exactly like the execute path, INCLUDING pausing again at a
        later member. A claim loser (the member checkpoint already resolved) is a clean
        no-op — the run is left at the winner's terminal state.
        """
        from himmy.application.orchestration_runner import run_orchestration

        semaphore = self._workspace_semaphore(workspace_id)
        async with semaphore:
            run = await self._storage.get_run(run_id)
            if run is None:  # pragma: no cover - defensive
                return
            operator_provisioned = bool(
                (run.metadata or {}).get("operator_provisioned", False)
            )
            kind = (run.metadata or {}).get("orchestration_kind", "graph")
            resource_kind = (run.metadata or {}).get("orchestration", "workflow")

            members: list[AgentDefRecord] = []
            for agent_id in member_agent_ids:
                rec = await _maybe_await(
                    self._agent_resolver(agent_id, workspace_id=workspace_id)
                    if self._agent_resolver is not None
                    else None
                )
                if rec is None:
                    run.status = RunStatus.FAILED
                    run.error = f"resume failed: member agent {agent_id} removed"
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                members.append(rec)

            # centralize-tool-gate: bind the launcher's gate ambiently across the resume
            # drive too (belt-and-braces; the explicit arg below stays authoritative). Use
            # the contextvar set/reset directly (not a ``with``) so the large call block
            # below keeps its indentation; reset in ``finally`` so the binding is scoped.
            from himmy.services.tools.ambient import _active_authorizer

            resume_authorizer = self._build_tool_authorizer(
                (run.metadata or {}).get("actor")
            )
            _resume_authz_token = _active_authorizer.set(resume_authorizer)
            try:
                outcome = await asyncio.wait_for(
                    run_orchestration(
                        kind=kind,
                        members=members,
                        prompt="",
                        resource_kind=resource_kind,
                        storage=self._storage,
                        shared_inference=getattr(
                            self._runtime, "inference_service", None
                        ),
                        operator_provisioned=operator_provisioned,
                        graph_checkpoint_store=self._resolve_graph_checkpoint_store(),
                        graph_resume_id=graph_resume_id,
                        checkpoint_store=self._checkpoint_store,
                        approve_member=approved,
                        actor=actor,
                        # P0 confused-deputy fix: re-thread the launching principal's
                        # tool-capability gate from the run's persisted actor on resume too,
                        # so a HITL resume cannot regain tool reach the launcher lacked.
                        tool_authorizer=resume_authorizer,
                        # P1 tenancy: re-thread the run's tenant + (within-tenant) subject so the
                        # resumed members' memory/KB packs stay namespaced to the owner — a HITL
                        # resume cannot collapse onto the shared static namespace. None/None
                        # offline / all_tenants is byte-unchanged.
                        owner_workspace_id=run.workspace_id,
                        owner_subject_scope=self._subject_scope_from_actor(
                            (run.metadata or {}).get("actor")
                        ),
                    ),
                    timeout=self._run_timeout_seconds,
                )
            except TimeoutError:
                run.status = RunStatus.FAILED
                run.error = f"resume exceeded {self._run_timeout_seconds:.0f}s timeout"
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            except asyncio.CancelledError:
                run.status = RunStatus.FAILED
                run.error = "resume cancelled"
                run.updated_at = _now()
                try:
                    await self._storage.save_run(run)
                except Exception:  # pragma: no cover - best-effort during cancel
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 - terminal failure transition
                # A claim-loser raises HimmyError('already resolved') from the member
                # resume — that is a clean NO-OP (the winner owns the terminal state),
                # NOT a failure. Distinguish it so a double-approve never flips a
                # SUCCEEDED run to FAILED.
                if _is_resume_claim_loss(exc):
                    logger.info(
                        "orchestration resume of run %s was a no-op (%s)", run_id, exc
                    )
                    return
                run.status = RunStatus.FAILED
                run.error = str(exc)
                run.updated_at = _now()
                await self._storage.save_run(run)
                return
            finally:
                _active_authorizer.reset(_resume_authz_token)

            await self._apply_orchestration_outcome(run, outcome)


    async def _apply_orchestration_outcome(
        self, run: RunRecord, outcome: Any
    ) -> None:
        """Project a resumed orchestration outcome onto the run (shared with execute).

        Pauses AGAIN at a later member (AWAITING_APPROVAL with restamped ids), or lands
        terminal SUCCEEDED/FAILED — mirroring the initial execute-path projection.
        """
        run.thread_id = outcome.thread_id
        run.output_text = outcome.output_text or None
        run.metadata = {
            **(run.metadata or {}),
            "stopped_reason": outcome.stopped_reason,
            "route": outcome.route,
        }
        if outcome.graph_checkpoint_id:
            run.metadata["graph_checkpoint_id"] = outcome.graph_checkpoint_id
        if outcome.awaiting_approval:
            run.status = RunStatus.AWAITING_APPROVAL
            run.metadata["checkpoint_id"] = outcome.member_checkpoint_id
            run.metadata["orchestration_checkpoint_id"] = (
                outcome.orchestration_checkpoint_id
            )
            run.metadata["awaiting_member"] = outcome.awaiting_member
            run.metadata["hitl_kind"] = "orchestration"
            run.updated_at = _now()
            await self._storage.save_run(run)
            return
        run.status = RunStatus.FAILED if outcome.failed else RunStatus.SUCCEEDED
        if outcome.failed:
            run.error = outcome.error or "orchestration failed"
        run.updated_at = _now()
        await self._storage.save_run(run)

    # --------------------------------------------------------------------- reads
    # Tenant-scoped NO-MUTATION reads live on the :class:`RunReadService` collaborator; the
    # methods below are thin delegating shims preserving the exact public signatures. Internal

    async def create_orchestration_run(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        kind: str,
        members: list[AgentDefRecord],
        prompt: str,
        resource_kind: str,
        resource_id: str,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
        graph_checkpoint_store: Any = None,
        graph_resume_id: str | None = None,
    ) -> RunRecord:
        """Launch a team/workflow orchestration on the EXISTING run machinery (T3b).

        A team/workflow run is NOT a second executor: it creates a canonical
        :class:`RunRecord`, admits it against the SAME T0.4 per-workspace quota, and
        executes on a tracked background task under the per-workspace concurrency semaphore
        — exactly like :meth:`create_run`. The difference is the body: instead of one
        per-run agent runtime it builds a TEAM runtime from the ordered member
        :class:`AgentDefRecord`s (resolved + sanitized) and drives the matching orchestrator
        (``multi_agent`` | ``group_chat`` for a team; an ordered pipeline for a workflow).

        ``kind`` selects the orchestrator. ``members`` is the ordered, pre-resolved member
        list (the router validated each exists in the workspace + drew the same-workspace
        membership check). ``resource_kind``/``resource_id`` (``team``/``workflow`` + its id)
        are stamped into the run metadata so a run is traceable to the team/workflow that
        launched it. ``graph_checkpoint_store``/``graph_resume_id`` (the ``graph`` kind)
        thread the durable :class:`SqliteGraphCheckpointStore` so a long graph run resumes
        after a restart.

        Returns the QUEUED :class:`RunRecord` immediately; poll ``get_run`` for the outcome.
        Raises :class:`WorkspaceRunQuotaExceeded` (429) when the workspace is at its cap.
        """
        metadata: dict[str, Any] = {
            "orchestration": resource_kind,
            "orchestration_kind": kind,
            f"{resource_kind}_id": resource_id,
            "member_agent_ids": [m.agent_id for m in members],
            # Persisted so a HITL resume re-sanitizes the members under the SAME operator
            # status (a revoked opt-in between pause and resume then fails closed).
            "operator_provisioned": bool(operator_provisioned),
        }
        if actor:
            metadata["actor"] = actor
        if graph_resume_id:
            metadata["graph_resume_id"] = graph_resume_id
        # Q3: persist the launch PROMPT so the dispatcher can reconstruct the orchestration
        # run from a fresh process (the members are re-resolved from ``member_agent_ids`` and
        # the graph checkpoint store is rebuilt from the provider — both already crash-safe).
        if self._dispatch_enabled:
            metadata["orchestration_prompt"] = prompt
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=None,
            persona_name=members[0].name if members else resource_kind,
            model_key="default",
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            metadata=metadata,
        )
        if self._dispatch_enabled:
            # The orchestration run has no single-agent input_blob; its lane is the neutral
            # default (members may target mixed providers) + its retry ceiling is stamped so
            # the dispatcher claims + drives it via the orchestration reconstruction path.
            # It MUST be the literal LANE_DEFAULT, not NULL: the claim filter is
            # ``lane_key IN (...)`` and SQL NULL never matches an IN/ANY list, so a NULL lane
            # would sit QUEUED forever whenever the local probe gates out the local lane —
            # exactly the laptop-transient the health gate is meant to drain through.
            run.lane_key = LANE_DEFAULT
            run.max_attempts = self._default_max_attempts
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): an N-member
        # orchestration writes EXACTLY ONE run row (this single PARENT; members execute
        # in-process under the parent's per-workspace semaphore and never persist their own
        # RunRecords), so the parent IS the single admission unit and gating it is the whole
        # logical-unit gate. In dispatch mode the old code short-circuited to ``return stored``
        # WITHOUT ever admitting against the outstanding cap — the cap was simply NOT enforced
        # at create time (deferred to the dispatcher's runtime concurrency cap). Route it
        # through the SAME atomic op as create_run so a concurrent burst lands EXACTLY at the
        # cap. ``admitted=False`` => at/over cap and NOTHING written (no parent row, no member
        # state => zero partial/orphaned orchestration): raise the same 429. An idempotent
        # re-submit returns the prior row and does NOT consume a slot. A fresh admit leaves the
        # parent QUEUED for the dispatcher (recoverable on crash) — the SAME dispatch tail as
        # before, just now atomically capped. No soft ``_admit_workspace_run_durable`` is added
        # (the dispatch orchestration path never called it; the atomic op is the enforcer).
        if (
            self._dispatch_enabled
            and self._workspace_max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._storage.save_run_if_under_quota(
                run, cap=self._workspace_max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._workspace_max_outstanding,
                    outstanding=self._workspace_max_outstanding,
                )
            if stored.run_id != run.run_id:
                # idempotent re-submit: do not relaunch, do not consume a slot.
                return stored
            # Fresh admit: leave QUEUED for the dispatcher (recoverable on crash).
            return stored

        stored, created = await self._storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        # Q3 dispatch mode: leave QUEUED for the dispatcher to claim (recoverable on crash).
        if self._dispatch_enabled:
            return stored

        try:
            self._admit_workspace_run(workspace_id)
        except WorkspaceRunQuotaExceeded:
            stored.status = RunStatus.FAILED
            stored.error = "rejected: workspace run-concurrency quota exceeded"
            stored.updated_at = _now()
            try:
                await self._storage.save_run(stored)
            except Exception:  # pragma: no cover - best-effort terminal mark
                logger.warning("failed to mark quota-rejected run %s", stored.run_id)
            raise

        bg = asyncio.create_task(
            self._execute_orchestration_run(
                stored.run_id,
                workspace_id=workspace_id,
                kind=kind,
                members=members,
                prompt=prompt,
                resource_kind=resource_kind,
                operator_provisioned=operator_provisioned,
                graph_checkpoint_store=graph_checkpoint_store,
                graph_resume_id=graph_resume_id,
            )
        )
        self._tasks.add(bg)
        bg.add_done_callback(self._tasks.discard)
        return stored


    async def _execute_orchestration_run(
        self,
        run_id: str,
        *,
        workspace_id: str,
        kind: str,
        members: list[AgentDefRecord],
        prompt: str,
        resource_kind: str,
        operator_provisioned: bool,
        graph_checkpoint_store: Any,
        graph_resume_id: str | None,
    ) -> None:
        """Background worker: build the team runtime + drive the orchestrator (T3b).

        Holds the per-workspace concurrency semaphore (T0.4) and releases the outstanding
        reservation in ``finally`` (mirroring :meth:`_execute_run`), so a team/workflow run
        is bounded and accounted exactly like a single-agent run.
        """
        from himmy.application.orchestration_runner import run_orchestration

        semaphore = self._workspace_semaphore(workspace_id)
        try:
            async with semaphore:
                run = await self._storage.get_run(run_id)
                if run is None:  # pragma: no cover - defensive
                    return
                run.status = RunStatus.RUNNING
                run.updated_at = _now()
                await self._storage.save_run(run)
                # centralize-tool-gate: also bind the launching principal's gate AMBIENTLY
                # for the whole orchestration drive, so every member runtime — including any
                # built by an orchestrator sub-path that forgot to thread the authorizer —
                # consults the chokepoint. The explicit ``tool_authorizer=`` below stays
                # authoritative (it attenuates per member); this is the belt-and-braces
                # ambient layer. Inert (None) offline / when no RBAC policy is wired. Use the
                # contextvar set/reset directly (not ``with``) so the large call block keeps
                # its indentation; reset in ``finally`` so the binding is scoped.
                from himmy.services.tools.ambient import _active_authorizer

                team_authorizer = self._build_tool_authorizer(
                    (run.metadata or {}).get("actor")
                )
                _team_authz_token = _active_authorizer.set(team_authorizer)
                try:
                    outcome = await asyncio.wait_for(
                        run_orchestration(
                            kind=kind,
                            members=members,
                            prompt=prompt,
                            resource_kind=resource_kind,
                            storage=self._storage,
                            shared_inference=getattr(
                                self._runtime, "inference_service", None
                            ),
                            operator_provisioned=operator_provisioned,
                            graph_checkpoint_store=graph_checkpoint_store,
                            graph_resume_id=graph_resume_id,
                            # HITL: thread the surface-owned AgentCheckpoint store so a
                            # graph/workflow member calling an approval-gated tool pauses
                            # to a durable member checkpoint (None disables nested HITL).
                            checkpoint_store=self._checkpoint_store,
                            # P0 confused-deputy fix: rebuild the LAUNCHING principal's
                            # tool-capability gate from the run's persisted actor and thread
                            # it into every member runtime, so a team/workflow can only
                            # invoke tools the launcher's own role was granted (no-op
                            # offline / when no RBAC policy is wired). Mirrors the
                            # single-agent path's _resolve_runtime/_build_tool_authorizer.
                            tool_authorizer=team_authorizer,
                            # P1 tenancy: namespace the members' memory/KB (and tasks/notes)
                            # packs to THIS run's tenant + (within-tenant) subject so two
                            # tenants' — or two users of one tenant's — orchestration runs never
                            # share the durable memory/KB namespace (cross-tenant confused-deputy
                            # DATA leak). Mirrors the single-agent + Studio team paths; None/None
                            # offline / all_tenants is byte-unchanged.
                            owner_workspace_id=run.workspace_id,
                            owner_subject_scope=self._subject_scope_from_actor(
                                (run.metadata or {}).get("actor")
                            ),
                        ),
                        timeout=self._run_timeout_seconds,
                    )
                except TimeoutError:
                    run.status = RunStatus.FAILED
                    run.error = (
                        f"run exceeded {self._run_timeout_seconds:.0f}s execution timeout"
                    )
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                except asyncio.CancelledError:
                    run.status = RunStatus.FAILED
                    run.error = "run cancelled"
                    run.updated_at = _now()
                    try:
                        await self._storage.save_run(run)
                    except Exception:  # pragma: no cover - best-effort during cancel
                        pass
                    raise
                except Exception as exc:  # noqa: BLE001 - terminal failure transition
                    run.status = RunStatus.FAILED
                    run.error = str(exc)
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                finally:
                    _active_authorizer.reset(_team_authz_token)

                run.thread_id = outcome.thread_id
                run.output_text = outcome.output_text or None
                run.metadata = {
                    **(run.metadata or {}),
                    "stopped_reason": outcome.stopped_reason,
                    "route": outcome.route,
                }
                if outcome.graph_checkpoint_id:
                    run.metadata["graph_checkpoint_id"] = outcome.graph_checkpoint_id
                # HITL pause: a member called an approval-gated tool. Stamp the MEMBER
                # checkpoint id under the ``checkpoint_id`` key so the unchanged
                # pending-approvals path reads it, plus the orchestration (graph)
                # checkpoint id + awaiting member + a hitl_kind discriminator so
                # ``resume_run`` routes to the orchestration resume. STOP here — the
                # sweeper skips AWAITING_APPROVAL, so the paused run is never reaped.
                if outcome.awaiting_approval:
                    run.status = RunStatus.AWAITING_APPROVAL
                    run.metadata["checkpoint_id"] = outcome.member_checkpoint_id
                    run.metadata["orchestration_checkpoint_id"] = (
                        outcome.orchestration_checkpoint_id
                    )
                    run.metadata["awaiting_member"] = outcome.awaiting_member
                    run.metadata["hitl_kind"] = "orchestration"
                    run.updated_at = _now()
                    await self._storage.save_run(run)
                    return
                run.status = (
                    RunStatus.FAILED if outcome.failed else RunStatus.SUCCEEDED
                )
                if outcome.failed:
                    run.error = outcome.error or "orchestration failed"
                run.updated_at = _now()
                await self._storage.save_run(run)
        finally:
            self._release_workspace_run(workspace_id)

    def _extract_plan_from_checkpoint(
        self, checkpoint_id: str | None
    ) -> list[dict[str, str]]:
        """Read the bounded plan steps out of a PLAN-READY checkpoint (T2g).

        The plan-first run pauses on its gated ``update_plan`` call; that call's args
        carry the proposed steps. Returns the normalized, bounded steps (an empty list
        when no checkpoint / no plan call is pending), so a caller can read the plan
        before approving it.
        """
        if not checkpoint_id or self._checkpoint_store is None:
            return []
        checkpoint = self._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:  # pragma: no cover - the pause just wrote it
            return []
        from himmy.runtime.plan_mode import PLAN_TOOL, normalize_plan_steps

        for pending in checkpoint.pending_tool_calls:
            if pending.tool_name == PLAN_TOOL:
                return normalize_plan_steps((pending.args or {}).get("steps"))
        return []
