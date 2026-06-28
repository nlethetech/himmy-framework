"""API kernel: the /v1/runs router (async run create + read + replay).

Reads are tenant-scoped on ``workspace_id`` (AAEO-4), list is paginated with a
deterministic order + cap (AAEO-8), and routes declare typed responses + 404
shapes for accurate OpenAPI (AAEO-9).
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from himmy.api.auth import (
    authorize_object,
    enforce_subject_write,
    get_principal,
    narrow_subject,
    require_permission,
    require_workspace,
    resolve_workspace,
)
from himmy.api.models import (
    NOT_FOUND_RESPONSE,
    RunListResponse,
)
from himmy.api.security_audit import audit_event
from himmy.application.services import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    HitlNotSupportedError,
    HitlRequiresAgentError,
    RunNotApprovableError,
)
from himmy.entities.lineage import DEFAULT_TRACE_DEPTH
from himmy.services.storage.models import RunRecord, RunStatus

router = APIRouter(prefix="/v1/runs", tags=["runs"])

_READ = [Depends(require_permission("run", "read"))]
_WRITE = [Depends(require_permission("run", "write"))]


class PersonaInput(BaseModel):
    """A simplified persona payload usable from curl."""

    name: str
    description: str = ""
    instructions: list[str] = []
    metadata: dict[str, Any] = {}


class TaskInput(BaseModel):
    """A simplified task payload usable from curl."""

    title: str
    prompt: str
    context: dict[str, Any] = {}


class ApprovalDecisionRequest(BaseModel):
    """Optional body for approve/reject: an explicit workspace scope (else inferred)."""

    workspace_id: str | None = None


class PendingToolCallView(BaseModel):
    """A pending, approval-gated tool call (args redacted) a paused run awaits (T2f)."""

    tool_name: str
    args: dict[str, Any] = {}


class PendingApprovalsResponse(BaseModel):
    """The redacted pending tool call(s) a HITL-paused run is blocked on (T2f)."""

    run_id: str
    status: RunStatus
    pending_tool_calls: list[PendingToolCallView] = []


class CreateRunRequest(BaseModel):
    """The POST /v1/runs body: subject scope + persona/agent + task + idempotency.

    Exactly one identity source must be supplied: either an inline ``persona`` (the
    legacy fast path, which runs tool-less on the shared runtime, byte-unchanged) OR an
    ``agent_id`` referencing a stored :class:`AgentSpec` (T2e) — which runs WITH ITS
    TOOLS on a per-run tool-bearing runtime (T0.2). They are mutually exclusive.
    """

    workspace_id: str
    subject_id: str
    persona: PersonaInput | None = None
    agent_id: str | None = None
    task: TaskInput
    idempotency_key: str | None = None
    hitl: bool = False
    plan: bool = False


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


async def _bola_blocked(request: Request, run_id: str, workspace_id: str | None) -> bool:
    """Whether object-level (BOLA, WS-bola) narrowing hides this run from the caller.

    Returns True ONLY for an opt-in ``subject_scoped`` principal reading a run that belongs
    to ANOTHER data subject (so the caller-facing route renders a clean 404). For every
    other caller — offline / ``all_tenants`` / the historical multi-user-workspace default
    / a ``tenant_admin`` — :meth:`~himmy.api.auth.principal.Principal.subject_scoped` is
    falsy, so this short-circuits WITHOUT touching the run-resolution path and the legacy
    behavior (e.g. ``/events`` returning ``[]`` for a foreign workspace) is byte-unchanged.
    """
    if not get_principal(request).subject_scoped:
        return False
    run = await _container(request).run_app.get_run(run_id, workspace_id=workspace_id)
    if run is None:
        return False  # unknown/out-of-workspace is handled by the route's own lookup
    return not authorize_object(request, run.subject_id)


@router.post("", response_model=RunRecord, dependencies=_WRITE)
async def create_run(body: CreateRunRequest, request: Request) -> RunRecord:
    """Create a run (idempotent) and execute it in the background; returns the record.

    Two mutually-exclusive identity sources (T2e): an inline ``persona`` (tool-less,
    byte-unchanged back-compat) OR an ``agent_id`` that resolves a stored ``AgentSpec``
    and executes WITH ITS TOOLS on the per-run runtime, recording an agent lineage edge.

    With ``hitl=true`` (T2f) the run drives the agentic loop and PAUSES at
    ``AWAITING_APPROVAL`` when it hits an approval-gated tool — resumable via
    ``POST /v1/runs/{id}/approve|reject``. HITL requires a stored ``agent_id`` (an inline
    persona carries no tools to gate → 422) and a deployment with a checkpoint store
    (→ 400 when absent).

    With ``plan=true`` (T2g) the run pauses at PLAN-READY: the agent publishes its plan
    (carried in ``metadata['plan']``) and the run stops at ``AWAITING_APPROVAL``,
    resumable via the SAME ``approve`` endpoint. Plan mode has the same preconditions as
    HITL (a stored ``agent_id`` + a checkpoint store).
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    if (body.persona is None) == (body.agent_id is None):
        raise HTTPException(
            status_code=422,
            detail="exactly one of 'persona' or 'agent_id' is required",
        )

    workspace_id = require_workspace(request, body.workspace_id)
    # BOLA write gate (WS-bola): a subject_scoped principal may only create a run under its
    # OWN subject — else it could stamp a run (and its lineage / erasure linkage) under a
    # foreign data subject. A no-op for offline / all_tenants / tenant_admin callers.
    enforce_subject_write(request, body.subject_id)
    task = Task(
        title=body.task.title,
        prompt=body.task.prompt,
        context=body.task.context,
    )

    persona: Persona
    agent_spec = None
    agent_def = None
    llm_config = None
    operator_provisioned = False

    if body.agent_id is not None:
        # T2e: resolve the stored, tenant-scoped agent and run it WITH ITS TOOLS.
        agent_def = await _container(request).agent_app.get_agent_def(
            body.agent_id, workspace_id=workspace_id
        )
        if agent_def is None:
            raise HTTPException(status_code=404, detail="agent not found")
        agent_spec = agent_def.agent_spec()
        persona = agent_spec.to_persona()
        llm_config = agent_spec.to_llm_config()
        # The stored spec was already sanitized at WRITE time; the run-time sanitizer
        # (re-applied inside create_run, defense-in-depth) must honor the same operator
        # status, else an operator-provisioned spec's tools would be stripped at run.
        operator_provisioned = bool(
            getattr(get_principal(request), "all_tenants", False)
        )
    else:
        # The inline-persona fast path: tool-less shared runtime, byte-unchanged.
        assert body.persona is not None  # guaranteed by the XOR check above
        persona = Persona(
            name=body.persona.name,
            description=body.persona.description,
            instructions=body.persona.instructions,
            metadata=body.persona.metadata,
        )

    try:
        run: RunRecord = cast(
            RunRecord,
            await _container(request).run_app.create_run(
                workspace_id=workspace_id,
                subject_id=body.subject_id,
                persona=persona,
                task=task,
                idempotency_key=body.idempotency_key,
                llm_config=llm_config,
                agent_spec=agent_spec,
                agent_def=agent_def,
                operator_provisioned=operator_provisioned,
                hitl=body.hitl,
                plan=body.plan,
                actor=get_principal(request).actor_metadata(),
            ),
        )
    except HitlRequiresAgentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HitlNotSupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"run {run.run_id} created",
    )
    return run


async def _resume_run(
    run_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    *,
    approved: bool,
) -> RunRecord:
    """Shared approve/reject handler: resume a HITL-paused run (T2f).

    Tenant-scoped (``resolve_workspace`` + a service-side workspace check); a 409 when the
    run is not AWAITING_APPROVAL (so a double-approve of an already-resumed run is a clean
    no-op, never a second firing of the gated tool). Stamps the approver actor and audits
    the decision. The actual gated-tool execution + continuation happen on a fresh tracked
    background task; the response is the RUNNING record (fire-and-forget, like create).

    Object-level (BOLA, WS-bola): these are the WRITE siblings of the gated read surfaces
    (``get_run`` / ``/events`` / ``/thread`` / ``/pending-approvals``), and resuming another
    data subject's HITL-paused run is a strictly worse hole than a read leak — it would FIRE
    (approve) or permanently BLOCK (reject) that subject's gated tool. So a ``subject_scoped``
    principal targeting a run that belongs to ANOTHER subject is collapsed into the SAME clean
    404 as an unknown/out-of-workspace run (never a 409 that would confirm the foreign run
    exists or leak its lifecycle state). The check runs BEFORE ``resume_run`` so the gated
    tool is never touched. Non-subject-scoped / ``all_tenants`` / offline callers are
    unaffected (:func:`_bola_blocked` short-circuits to ``False``).
    """
    workspace_id = resolve_workspace(request, body.workspace_id)
    if await _bola_blocked(request, run_id, workspace_id):
        raise HTTPException(status_code=404, detail="run not found")
    principal = get_principal(request)
    try:
        run = cast(
            RunRecord,
            await _container(request).run_app.resume_run(
                run_id,
                approved=approved,
                workspace_id=workspace_id,
                actor=principal.subject,
            ),
        )
    except RunNotApprovableError as exc:
        # An unknown / out-of-workspace run is a 404; a known-but-not-paused run is a 409.
        if exc.status == "unknown":
            raise HTTPException(status_code=404, detail="run not found") from exc
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HitlNotSupportedError as exc:  # pragma: no cover - guarded at create
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="approve" if approved else "reject",
        workspace_id=run.workspace_id,
        detail=f"run {run_id} {'approved' if approved else 'rejected'}",
    )
    return run


@router.post(
    "/{run_id}/approve",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def approve_run(
    run_id: str,
    request: Request,
    body: ApprovalDecisionRequest = ApprovalDecisionRequest(),
) -> RunRecord:
    """Approve a HITL-paused run: fire the gated tool EXACTLY ONCE and continue (T2f).

    The run must be AWAITING_APPROVAL (else 409). Resumes via the runtime's atomic
    ``claim()`` so a double-approve is a no-op (the gated tool never fires twice).
    """
    return await _resume_run(run_id, body, request, approved=True)


@router.post(
    "/{run_id}/reject",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def reject_run(
    run_id: str,
    request: Request,
    body: ApprovalDecisionRequest = ApprovalDecisionRequest(),
) -> RunRecord:
    """Reject a HITL-paused run: record the rejection WITHOUT running the gated tool (T2f).

    The run must be AWAITING_APPROVAL (else 409). The model sees the rejection and the
    loop continues to a terminal state; the gated tool is never executed.
    """
    return await _resume_run(run_id, body, request, approved=False)


@router.get(
    "/{run_id}/pending-approvals",
    response_model=PendingApprovalsResponse,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_pending_approvals(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> PendingApprovalsResponse:
    """The redacted pending tool call(s) a HITL-paused run awaits (T2f).

    Tenant-scoped (404 when the run is unknown/out-of-workspace). Secret-looking arg
    values are masked (the same redaction the Studio approvals inbox uses).
    """
    workspace_id = resolve_workspace(request, workspace_id)
    run = await _container(request).run_app.get_run(run_id, workspace_id=workspace_id)
    if run is None or not authorize_object(request, run.subject_id):
        raise HTTPException(status_code=404, detail="run not found")
    pending = await _container(request).run_app.pending_approvals(
        run_id, workspace_id=workspace_id
    )
    return PendingApprovalsResponse(
        run_id=run_id,
        status=run.status,
        pending_tool_calls=[
            PendingToolCallView(tool_name=p["tool_name"], args=p["args"])
            for p in (pending or [])
        ],
    )


@router.get("", response_model=RunListResponse, dependencies=_READ)
async def list_runs(
    request: Request,
    workspace_id: str | None = None,
    subject_id: str | None = None,
    status: RunStatus | None = None,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    """List runs (created_at desc), paginated. Returns a paged envelope (AAEO-8).

    Object-level (BOLA, WS-bola): a ``subject_scoped`` principal's list is pinned to its
    OWN subject via :func:`narrow_subject` (a ``tenant_admin`` and every non-subject-scoped
    / offline caller keep the caller-supplied filter), so a subject-scoped tenant can never
    enumerate another data subject's runs within a shared workspace.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    subject_id = narrow_subject(request, subject_id)
    run_app = _container(request).run_app
    items = await run_app.list_runs(
        workspace_id=workspace_id,
        subject_id=subject_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    total = await run_app.count_runs(
        workspace_id=workspace_id, subject_id=subject_id, status=status
    )
    return RunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if (offset + len(items)) < total else None,
    )


@router.get(
    "/{run_id}",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_run(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> RunRecord:
    """Read one run record by id (404 when unknown/out-of-workspace, AAEO-4).

    Object-level (BOLA, WS-bola) narrowing rides on top of the workspace scope: a
    ``subject_scoped`` principal that resolves a run belonging to ANOTHER data subject
    within its own tenant gets a clean 404 (not-found, never a 403 that would confirm the
    run exists). Non-subject-scoped / ``all_tenants`` / offline callers are unaffected —
    :func:`authorize_object` is a no-op for them, so a shared multi-user workspace and the
    zero-config path keep reading every run byte-for-byte.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    run = await _container(request).run_app.get_run(run_id, workspace_id=workspace_id)
    if run is None or not authorize_object(request, run.subject_id):
        raise HTTPException(status_code=404, detail="run not found")
    return cast(RunRecord, run)


@router.get("/{run_id}/events", dependencies=_READ)
async def get_run_events(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> list[Any]:
    """Replay the canonical event stream for one run (tenant-scoped, AAEO-4).

    BOLA (WS-bola): a ``subject_scoped`` principal reading another data subject's run gets
    a clean 404 (:func:`_bola_blocked`). For every other caller that guard short-circuits
    without touching the legacy path — a foreign workspace still returns ``[]`` (AAEO-4).
    """
    workspace_id = resolve_workspace(request, workspace_id)
    if await _bola_blocked(request, run_id, workspace_id):
        raise HTTPException(status_code=404, detail="run not found")
    return cast(
        list[Any],
        await _container(request).run_app.get_run_events(
            run_id, workspace_id=workspace_id
        ),
    )


@router.get("/{run_id}/thread", responses=NOT_FOUND_RESPONSE, dependencies=_READ)
async def get_run_thread(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> Any:
    """Replay the full conversation thread for one run (404 when absent, AAEO-4).

    BOLA (WS-bola): a ``subject_scoped`` principal cannot replay another data subject's
    thread (clean 404 via :func:`_bola_blocked`); other callers take the legacy path.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    if await _bola_blocked(request, run_id, workspace_id):
        raise HTTPException(status_code=404, detail="thread not found")
    thread = await _container(request).run_app.get_run_thread(
        run_id, workspace_id=workspace_id
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return thread


@router.get("/{run_id}/lineage", responses=NOT_FOUND_RESPONSE, dependencies=_READ)
async def get_run_lineage(
    run_id: str,
    request: Request,
    workspace_id: str | None = None,
    max_depth: int = Query(DEFAULT_TRACE_DEPTH, ge=0, le=64),
    relations: str | None = Query(
        None,
        description="comma-separated relation allow-list (e.g. uses_persona,in_thread)",
    ),
    fmt: str = Query("json", alias="format", pattern="^(json|dot)$"),
) -> Any:
    """Trace a run's provenance subgraph: the persona, prompt, and evidence snapshot.

    Returns the typed lineage graph as JSON, or Graphviz DOT with ``?format=dot``.
    404 when the run is unknown / out-of-workspace or has no projected lineage.

    BOLA (WS-bola): lineage exposes a run's persona, prompt text, and context/evidence
    snapshot, so it is a read sibling of ``/events`` and ``/thread`` and is gated the same
    way — a ``subject_scoped`` principal cannot read another data subject's run lineage
    within a shared tenant (clean 404 via :func:`_bola_blocked`). Other callers take the
    legacy path byte-unchanged.
    """
    rel = [r.strip() for r in relations.split(",") if r.strip()] if relations else None
    workspace_id = resolve_workspace(request, workspace_id)
    if await _bola_blocked(request, run_id, workspace_id):
        raise HTTPException(status_code=404, detail="run lineage not found")
    graph = await _container(request).run_app.get_run_lineage(
        run_id, workspace_id=workspace_id, max_depth=max_depth, relations=rel
    )
    if graph is None:
        raise HTTPException(status_code=404, detail="run lineage not found")
    if fmt == "dot":
        return PlainTextResponse(graph.to_dot())
    return graph


__all__ = ["router"]
