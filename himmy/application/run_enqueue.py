"""Idempotent run creation + Q3 queue-field stamping for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the ``RunEnqueuer`` collaborator in the
staged decomposition of ``RunAppService`` (LANE runapp, step 7 — enqueue). It owns the
cohesive band that turns a caller's request into a persisted, launchable run:

- :meth:`create_run` — the idempotent create-or-return-existing entry point for a fresh
  inline/agent run (the HITL/plan precondition gate, the tenant-spec sanitize, the actor +
  agent lineage metadata stamp, and the atomic admission ordering),
- :meth:`continue_thread` — the same admission machinery for APPENDING an agentic turn to a
  stored conversation (thread_id pinned to conversation_id),
- :meth:`_stamp_queue_fields` — the Q3 leased-queue field stamp (lane / retry ceiling /
  recoverable input blob), a no-op on the inline path,
- :meth:`_launch_or_enqueue` — the admission-then-drive fork: INLINE mode admits against the
  in-RAM per-workspace cap and launches the fire-and-forget background task; DISPATCH mode
  leaves the run QUEUED (recoverable) for the leased dispatcher after the durable count-cap.

Behaviour is BYTE-IDENTICAL to the former inline methods. The idempotency primitive
(``save_run_if_absent_by_idempotency`` on the soft path / ``save_run_if_under_quota`` on the
atomic dispatch path), the lane-routing, and the admission ORDERING are preserved exactly —
including that an idempotent re-submit NEVER relaunches and NEVER consumes a slot, and that a
newly-created run in dispatch mode is left QUEUED rather than fired inline.

The collaborator reads its ``storage`` / ``checkpoint_store`` / ``dispatch_enabled`` /
``default_max_attempts`` / ``tasks`` handles LIVE through the shared :class:`_RunContext`
(never a construction-time snapshot), reads the outstanding cap LIVE from the shared
:class:`WorkspaceQuota`, and delegates the durable count-cap + the drive path back to the
service (``_admit_workspace_run_durable`` / ``_execute_run``), so ``enable_dispatch`` and a
re-pointed store are observed at once.

``RunAppService``'s former methods (``create_run`` / ``continue_thread`` /
``_stamp_queue_fields`` / ``_launch_or_enqueue``) delegate here as thin shims, so every
router, CLI path, and test caller stays byte-identical.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from himmy.application.services import (
    HitlNotSupportedError,
    HitlRequiresAgentError,
    _now,
    logger,
)
from himmy.application.workspace_quota import WorkspaceRunQuotaExceeded
from himmy.config.spec_sanitizer import sanitize_tenant_spec
from himmy.services.storage.models import (
    LOCAL_WORKSPACE,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.run_input import encode_run_input
from himmy.services.storage.run_lane import lane_for_model_key

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from collections.abc import Awaitable, Callable, Coroutine

    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.application.run_context import _RunContext
    from himmy.application.workspace_quota import WorkspaceQuota
    from himmy.config.agent_spec import AgentSpec
    from himmy.services.inference.models import LLMConfig
    from himmy.services.storage.models import AgentDefRecord


class RunEnqueuer:
    """Idempotent create + Q3 queue-field stamping + admission-then-drive fork.

    Holds no state of its own beyond the shared context handle, the shared quota, and
    back-references to the durable count-cap + drive path (both of which stay on the
    service). Reads ``storage`` / ``checkpoint_store`` / ``dispatch_enabled`` /
    ``default_max_attempts`` / ``tasks`` live from the context and the outstanding cap live
    from the quota, so behaviour matches the former inline implementation byte-for-byte.
    """

    def __init__(
        self,
        *,
        context: _RunContext,
        ws_quota: WorkspaceQuota,
        admit_workspace_run_durable: Callable[[str, RunRecord], Awaitable[None]],
        execute_run: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Wire the shared context + quota (live handles) and the service back-references."""
        self._ctx = context
        self._ws_quota = ws_quota
        self._admit_workspace_run_durable = admit_workspace_run_durable
        self._execute_run = execute_run

    async def create_run(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        persona: Persona,
        task: Task,
        idempotency_key: str | None = None,
        llm_config: LLMConfig | None = None,
        actor: dict[str, Any] | None = None,
        agent_spec: AgentSpec | None = None,
        agent_def: AgentDefRecord | None = None,
        operator_provisioned: bool = False,
        hitl: bool = False,
        plan: bool = False,
    ) -> RunRecord:
        """Create (or return the existing) run and launch background execution.

        Idempotent on ``(workspace_id, idempotency_key)``: re-submitting a key
        returns the prior run rather than creating a duplicate. Returns
        immediately with a ``QUEUED`` record; execution runs in the background.

        The create+check is atomic via
        :meth:`StorageService.save_run_if_absent_by_idempotency` (no ``await``
        between read and write in-memory; ``ON CONFLICT DO NOTHING`` in Postgres),
        so two concurrent requests with the same key cannot both create a run.
        Background execution is only launched for the run this call actually
        created.

        ``agent_spec`` (T0.2) is the load-bearing seam: when a run resolves to a
        stored/declarative :class:`AgentSpec` the run executes on a PER-RUN
        tool-bearing runtime built via
        :func:`himmy.runtime.from_spec.build_runtime_for_spec` — so a ``/v1`` run can
        finally call the agent's TOOLS, which the shared (tool-less) runtime cannot.
        When ``agent_spec`` is ``None`` the existing inline-persona fast path runs on
        the shared tool-less runtime, byte-unchanged (back-compat). ``agent_spec`` is
        sanitized against the tenant RCE/SSRF surface (T0.3) unless
        ``operator_provisioned`` is set AND the operator opted in — a tenant spec
        carrying ``tools_module``/``http_tools``/``mcp_servers`` is rejected here.

        Per-workspace admission (T0.4): a workspace already holding
        ``workspace_max_outstanding`` in-flight runs is rejected with
        :class:`WorkspaceRunQuotaExceeded` BEFORE a record is created, so a runaway
        fan-out cannot pin unbounded tasks. An idempotent re-submit of an existing
        run never counts against the cap.
        """
        # HITL admission (T2f): a ``hitl=True`` run pauses on an approval-gated tool, so
        # it REQUIRES a per-run tool-bearing runtime (the shared inline runtime carries
        # no tool_service and cannot call — let alone gate — a tool) AND a checkpoint
        # store to pause into. Reject early (before any record is written) when either is
        # missing, so a caller never gets a silently tool-less "HITL" run that can never
        # pause. The store is resolved on resume from the stored ``agent_id``.
        #
        # Plan mode (T2g) pauses at PLAN-READY through the SAME approval gate (a gated
        # synthetic ``update_plan`` tool), so it has the identical preconditions: a stored
        # agent (so a per-run registry exists to register the plan tool onto) + a
        # checkpoint store. A run may set ``plan`` and ``hitl`` together — the plan gate is
        # simply the first gate the loop hits.
        if hitl or plan:
            if self._ctx.checkpoint_store is None:
                raise HitlNotSupportedError(
                    "this deployment has no checkpoint store; hitl/plan runs are "
                    "unavailable"
                )
            if agent_spec is None or agent_def is None:
                raise HitlRequiresAgentError(
                    "hitl/plan runs require a stored agent (agent_id); an inline persona "
                    "carries no tools to gate"
                )
            # Plan mode synthesizes a gated ``update_plan`` tool and registers it onto the
            # per-run tool registry to force the PLAN-READY pause. A bare persona spec
            # (e.g. {"name": "x"}) builds NO registry (from_spec returns registry=None and
            # the runtime gets no tool_service), so there is nowhere to register the plan
            # tool and the run would silently complete WITHOUT pausing. Reject up front
            # (422) — consistent with the inline-persona rejection — rather than letting a
            # legitimate-but-tool-less stored agent slip through to a silently wrong
            # "plan=true never paused" outcome.
            if plan and not agent_spec.builds_tool_registry():
                raise HitlRequiresAgentError(
                    "plan mode requires a tool-bearing stored agent (one declaring "
                    "tool_packs/tools_module/http_tools/knowledge/mcp_servers/connectors/"
                    "allow_spawn/allow_skill_dispatch); a tool-less agent builds no "
                    "registry to host the gated update_plan tool, so the run could never "
                    "pause at PLAN-READY"
                )

        # Fail-closed sanitize a per-run spec against the tenant attack surface
        # BEFORE anything is persisted or executed (T0.3). Operator-provisioned specs
        # may keep their tools when the operator opted in; everyone else is stripped
        # or (default) rejected by ``sanitize_tenant_spec`` raising.
        if agent_spec is not None:
            agent_spec = sanitize_tenant_spec(
                agent_spec, operator_provisioned=operator_provisioned
            ).spec

        # Stamp the authenticated actor ("who launched this") into the durable
        # metadata JSONB (round-trips on both in-memory and Postgres) so every run
        # records its initiator — the operational half of "who did what" (WS1.3).
        metadata: dict[str, Any] = {}
        if actor:
            metadata["actor"] = actor
        if agent_spec is not None:
            metadata["agent_name"] = agent_spec.name
        # T2e: a run launched by a stored agent records its agent_id so the run<->agent
        # lineage is reconstructable and the Studio/CLI can show "run of agent X".
        if agent_def is not None:
            metadata["agent_id"] = agent_def.agent_id
        # T2f: mark the run HITL-driven so the resume path (which re-resolves the spec
        # from agent_id) knows this run is approvable and rebuilds the same gated runtime.
        # T2g: plan mode is also an agentic, pausable run — mark it so the resume path
        # re-registers the gated ``update_plan`` tool when rebuilding the runtime.
        if hitl or plan:
            metadata["hitl"] = True
            # ``operator_provisioned`` governs whether the stored spec's privileged tools
            # survive the run-time re-sanitize; the resume must honor the SAME status so
            # the gated tool the run paused on is still present when it is approved.
            metadata["operator_provisioned"] = bool(operator_provisioned)
        if plan:
            metadata["plan_mode"] = True
        model_key = _resolve_model_key(llm_config, task)
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=task.task_id,
            persona_name=persona.name,
            model_key=model_key,
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            metadata=metadata,
        )
        # Q3: in leased-dispatch mode the run carries its lane (provider health gate) + retry
        # ceiling + recoverable input so the dispatcher (possibly in a FRESH process after a
        # crash) can claim and re-execute it. A no-op on the inline path (fields stay default).
        self._stamp_queue_fields(
            run,
            model_key=model_key,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
        )
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): when the cap
        # is ON and the workspace is not the exempt ``__local__``, the create goes through the
        # ATOMIC store op that FUSES the idempotency check, the active-run COUNT, and the
        # INSERT into ONE advisory-locked transaction (Postgres) / ``BEGIN IMMEDIATE``
        # (SQLite) — so concurrent same-workspace creates serialise and land EXACTLY at the
        # cap, never the unlocked count-then-insert overshoot. ``admitted=False`` means the
        # tenant was at/over cap and NOTHING was written (cleaner than the old
        # insert-then-mark-FAILED): raise the same 429 surface. An idempotent re-submit
        # returns the prior run with ``admitted=True`` and does NOT consume a slot — it must
        # NOT relaunch, so it is distinguished via ``created``.
        if (
            self._ctx.dispatch_enabled
            and self._ws_quota.max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._ctx.storage.save_run_if_under_quota(
                run, cap=self._ws_quota.max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._ws_quota.max_outstanding,
                    outstanding=self._ws_quota.max_outstanding,
                )
            # A re-submit of an existing run (same run object identity is NOT guaranteed;
            # detect it by idempotency key resolving to a prior row) must not relaunch. The
            # atomic op returns the stored row; ``run.run_id != stored.run_id`` (or a
            # mismatched created_at) signals a re-submit. The op returns the PRIOR run on a
            # key hit, so compare ids: a fresh insert returns the same object we passed in.
            if stored.run_id != run.run_id:
                return stored
            return await self._launch_or_enqueue(
                stored,
                workspace_id=workspace_id,
                persona=persona,
                task=task,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                hitl=hitl,
                plan=plan,
                quota_already_admitted=True,
            )

        # Atomic idempotent insert FIRST (the race-safe primitive), so an idempotent
        # re-submit (created=False) returns the prior run without ever touching the
        # T0.4 cap — a duplicate spawns no new task. Only a NEWLY-created run is
        # admitted against the per-workspace outstanding cap below. (Used by the INLINE
        # path, the exempt ``__local__`` workspace, and the cap-disabled case.)
        stored, created = await self._ctx.storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        return await self._launch_or_enqueue(
            stored,
            workspace_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
            hitl=hitl,
            plan=plan,
        )

    async def continue_thread(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        conversation_id: str,
        thread: ChatThread,
        prompt: str,
        agent_spec: AgentSpec,
        agent_def: AgentDefRecord,
        llm_config: LLMConfig | None = None,
        idempotency_key: str | None = None,
        actor: dict[str, Any] | None = None,
        operator_provisioned: bool = False,
        hitl: bool = False,
        plan: bool = False,
    ) -> RunRecord:
        """Continue a stored conversation with a new user turn on the per-run runtime (T2g).

        Unlike :meth:`create_run` (a fresh inline/agent run), this APPENDS a turn to an
        existing :class:`ChatThread` and drives the AGENTIC loop on the per-run
        tool-bearing runtime (T0.2) so a ``/v1`` conversation is as capable as a Studio one
        — the model sees prior turns AND can call the agent's tools / pause for approval
        (reviewer must_fix: NOT a tool-less single-shot ``run_task_detailed`` masquerading
        as chat). A continuation always resolves a stored ``agent_id`` (a per-run runtime
        needs a spec), runs through the SAME admission + sanitizer + quota path as a fresh
        run, and links the new :class:`RunRecord` to the conversation via
        ``metadata['conversation_id']`` (the run's ``thread_id`` IS the conversation id) so
        the thread router (and the ConversationStore projection sink) can find it.

        With ``hitl``/``plan`` the run can pause at ``AWAITING_APPROVAL`` exactly like a
        fresh agentic run; the SAME ``approve``/``reject`` machinery resumes it.
        """
        # A continuation is always agentic + needs a checkpoint store when it can pause.
        if (hitl or plan) and self._ctx.checkpoint_store is None:
            raise HitlNotSupportedError(
                "this deployment has no checkpoint store; hitl/plan runs are unavailable"
            )
        # Plan mode hosts its gated ``update_plan`` tool on the per-run registry; a
        # tool-less stored agent builds none, so the run could never pause at PLAN-READY.
        # Reject up front (422) — same gate as ``create_run`` — rather than silently
        # running plan=true to completion without a pause.
        if plan and not agent_spec.builds_tool_registry():
            raise HitlRequiresAgentError(
                "plan mode requires a tool-bearing stored agent; a tool-less agent "
                "builds no registry to host the gated update_plan tool, so the run "
                "could never pause at PLAN-READY"
            )

        # Fail-closed sanitize the spec under the request's operator status BEFORE it
        # runs (defense-in-depth: the stored spec was sanitized at write, this honors the
        # caller's operator status so a tenant continuation can never reach privileged
        # tools the stored spec might carry).
        agent_spec = sanitize_tenant_spec(
            agent_spec, operator_provisioned=operator_provisioned
        ).spec

        # Pin the thread id to the conversation id so the run's thread_id IS the
        # conversation id (the run<->conversation join the router/sink rely on). The new
        # user turn is NOT appended here — the runtime's ``run_agent_loop`` /
        # ``run_task_detailed`` appends it from ``task.prompt`` (appending it twice would
        # duplicate the message), so the loaded thread carries only the PRIOR turns.
        thread.thread_id = conversation_id

        persona = agent_spec.to_persona()
        if llm_config is None:
            llm_config = agent_spec.to_llm_config()
        task = agent_spec.make_task(prompt)

        metadata: dict[str, Any] = {"conversation_id": conversation_id}
        if actor:
            metadata["actor"] = actor
        metadata["agent_name"] = agent_spec.name
        metadata["agent_id"] = agent_def.agent_id
        if hitl or plan:
            metadata["hitl"] = True
            metadata["operator_provisioned"] = bool(operator_provisioned)
        if plan:
            metadata["plan_mode"] = True

        model_key = _resolve_model_key(llm_config, task)
        run = RunRecord(
            workspace_id=workspace_id,
            subject_id=subject_id,
            task_id=task.task_id,
            persona_name=persona.name,
            model_key=model_key,
            idempotency_key=idempotency_key,
            status=RunStatus.QUEUED,
            thread_id=conversation_id,
            metadata=metadata,
        )
        # Q3: stamp lane/retry/recoverable-input so a continuation can be claimed + re-run by
        # the dispatcher (a no-op on the inline path). The recovered run carries thread_id =
        # conversation_id, so the rebuilt task continues the SAME conversation.
        self._stamp_queue_fields(
            run,
            model_key=model_key,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
        )
        # T3 HARD per-tenant outstanding-run cap (dispatch / multi-node path): a continuation
        # is a real run and MUST land under the SAME atomic gate as a fresh ``create_run`` —
        # the prior soft ``save_run_if_absent`` + count-then-mark-FAILED ``_admit`` had a
        # TOCTOU window letting concurrent same-conversation/same-workspace continuations
        # overshoot the cap. Mirror create_run (count+insert fused in ONE advisory-locked
        # xact): ``admitted=False`` means the tenant was at/over cap and NOTHING was written
        # (raise the 429); an idempotent re-submit returns the prior row WITHOUT relaunching
        # and WITHOUT consuming a slot; a fresh admit launches with the soft re-check skipped.
        if (
            self._ctx.dispatch_enabled
            and self._ws_quota.max_outstanding > 0
            and workspace_id != LOCAL_WORKSPACE
        ):
            stored, admitted = await self._ctx.storage.save_run_if_under_quota(
                run, cap=self._ws_quota.max_outstanding
            )
            if not admitted:
                raise WorkspaceRunQuotaExceeded(
                    workspace_id,
                    cap=self._ws_quota.max_outstanding,
                    outstanding=self._ws_quota.max_outstanding,
                )
            if stored.run_id != run.run_id:
                # idempotent re-submit: do not relaunch, do not consume a slot.
                return stored
            return await self._launch_or_enqueue(
                stored,
                workspace_id=workspace_id,
                persona=persona,
                task=task,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                hitl=hitl,
                plan=plan,
                thread=thread,
                quota_already_admitted=True,
            )

        # OFF / __local__ / cap<=0 / inline — byte-identical legacy soft path.
        stored, created = await self._ctx.storage.save_run_if_absent_by_idempotency(run)
        if not created:
            return stored

        return await self._launch_or_enqueue(
            stored,
            workspace_id=workspace_id,
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            agent_def=agent_def,
            hitl=hitl,
            plan=plan,
            thread=thread,
        )

    # --------------------------------------------------------- Q3 enqueue/dispatch
    def _stamp_queue_fields(
        self,
        run: RunRecord,
        *,
        model_key: str | None,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        hitl: bool,
        plan: bool,
    ) -> None:
        """Populate the leased-queue fields on a single-agent run when dispatch is on (Q3).

        A no-op on the inline path (the fields keep their RunRecord defaults). In dispatch
        mode it stamps: ``lane_key`` (the provider health-gate lane derived from the model
        key), ``max_attempts`` (the retry ceiling), and ``input_blob`` — the Q0 recoverable
        launch input serialized so a dispatcher in a FRESH process can rehydrate the exact
        persona/task/spec and re-execute. The blob is stored PLAINTEXT here; the durable run
        store encrypts it at rest (bound to run_id) on write, exactly like chat content.
        """
        if not self._ctx.dispatch_enabled:
            return
        run.lane_key = lane_for_model_key(model_key)
        run.max_attempts = self._ctx.default_max_attempts
        run.input_blob = encode_run_input(
            persona=persona,
            task=task,
            llm_config=llm_config,
            agent_spec=agent_spec,
            hitl=hitl,
            plan=plan,
            run_id=run.run_id,
        )

    async def _launch_or_enqueue(
        self,
        stored: RunRecord,
        *,
        workspace_id: str,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None,
        agent_spec: AgentSpec | None,
        agent_def: AgentDefRecord | None,
        hitl: bool,
        plan: bool,
        thread: ChatThread | None = None,
        quota_already_admitted: bool = False,
    ) -> RunRecord:
        """Admit + (inline) launch OR (dispatch) leave QUEUED for the dispatcher (Q3).

        INLINE mode (default, offline single-box + bare TestClient): admits the run against
        the per-workspace outstanding cap and launches today's fire-and-forget background
        task — byte-identical to the pre-Q3 behaviour. DISPATCH mode (durable store + the
        lifespan dispatcher): the run is left QUEUED for the dispatcher to claim, so a crash
        between enqueue and execution leaves it recoverable (not FAILED). Admission/concurrency
        in dispatch mode is the dispatcher's job (its bounded claim loop + the per-workspace
        execution semaphore at run time), so the in-memory outstanding counter — which a
        cross-process claim could never release — is NOT taken here.

        ``quota_already_admitted`` is set by the dispatch caller when the per-tenant
        outstanding cap was already enforced ATOMICALLY (count+insert fused) via
        :meth:`StorageService.save_run_if_under_quota` — the HARD path. In that case the
        durable count-then-mark-FAILED admit below is SKIPPED (it would double-count and the
        atomic op already rejected an over-cap create before any row was written).
        """
        if self._ctx.dispatch_enabled:
            # T3 per-tenant QUOTA at enqueue (dispatch / multi-node path): the in-RAM
            # outstanding counter cannot span worker processes, so the cap is enforced by
            # COUNTING the workspace's non-terminal runs in the SHARED store before leaving the
            # run QUEUED. When the caller already enforced the cap atomically
            # (``quota_already_admitted``), this soft re-check is skipped — the atomic op is
            # the HARD enforcer and a second count here would only double-work. Otherwise (the
            # legacy soft fallback) a burst beyond the cap is rejected (the record is marked
            # FAILED, not orphaned QUEUED) and the 429 propagates. ``0`` disables the cap
            # (single-tenant default unaffected, since the LOCAL workspace is exempted).
            if not quota_already_admitted:
                await self._admit_workspace_run_durable(workspace_id, stored)
            # Recoverable QUEUED state; the dispatcher claims + executes it.
            return stored

        # T0.4 admission (inline only): a workspace already at its outstanding-run cap cannot
        # launch another background task. The record exists (atomicity preserved), so it is
        # marked FAILED rather than orphaned QUEUED, and the quota error propagates (HTTP 429).
        try:
            self._ws_quota.admit(workspace_id)
        except WorkspaceRunQuotaExceeded:
            stored.status = RunStatus.FAILED
            stored.error = "rejected: workspace run-concurrency quota exceeded"
            stored.updated_at = _now()
            try:
                await self._ctx.storage.save_run(stored)
            except Exception:  # pragma: no cover - best-effort terminal mark
                logger.warning("failed to mark quota-rejected run %s", stored.run_id)
            raise

        bg: asyncio.Task[Any] = asyncio.create_task(
            self._execute_run(
                stored.run_id,
                workspace_id=workspace_id,
                persona=persona,
                task=task,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                hitl=hitl,
                plan=plan,
                thread=thread,
            )
        )
        self._ctx.tasks.add(bg)
        bg.add_done_callback(self._ctx.tasks.discard)
        return stored


def _resolve_model_key(llm_config: LLMConfig | None, task: Task) -> str | None:
    """Resolve the effective model key for a run (delegates to the service helper).

    Imported lazily from :mod:`himmy.application.services` because that helper is defined
    AFTER this module is imported at the bottom of ``services`` — a module-top import would
    bind before the name exists.
    """
    from himmy.application.services import _resolve_model_key as _impl

    return _impl(llm_config, task)
