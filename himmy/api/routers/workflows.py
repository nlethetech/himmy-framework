"""API kernel: the ``/v1/workflows`` router — workspace-scoped agent pipelines (T3b).

A workflow is an ORDERED pipeline of STORED agent ids (the ``/v1/agents`` resource, T2e):
the members run in sequence and each member's text output is threaded as context into the
next member's prompt. It is a durable, workspace-scoped resource stored in its own
``.himmy/workflows.db`` (the routines-store pattern), distinct from a team's dynamic
handoff/selection.

The same load-bearing properties as ``/v1/teams`` apply:

* every member ``agent_id`` must EXIST and live in the SAME workspace (404 otherwise) — and
  the same reference set powers the T2e delete-409 path (deleting an agent still referenced
  by a workflow is refused);
* ``POST /{id}/run`` launches via ``RunAppService.create_orchestration_run`` on the EXISTING
  run machinery under the T0.4 quota — a workflow always runs as the durable linear
  state-graph pipeline, so a long run checkpoints each completed step to the
  :class:`SqliteGraphCheckpointStore` and resumes after a restart. The result is a
  canonical :class:`RunRecord` polled via ``GET /v1/runs/{id}``.

Security mirrors ``/v1/teams``: ``run:read``/``run:write`` (a workflow is a run container),
tenant-scoped on every route, inert under the zero-config no-auth default.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from himmy.api import teams_store as svc
from himmy.api.auth import (
    enforce_subject_write,
    get_principal,
    require_permission,
    require_workspace,
    resolve_workspace,
    scoped_read,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.api.security_audit import audit_event
from himmy.application.services import WorkspaceRunQuotaExceeded
from himmy.services.storage.models import RunRecord

router = APIRouter(
    prefix="/v1/workflows", tags=["workflows"], dependencies=[Depends(scoped_read)]
)

_READ = [Depends(require_permission("run", "read"))]
_WRITE = [Depends(require_permission("run", "write"))]


# ---- request/response models -------------------------------------------------


def _validate_distinct_members(value: list[str]) -> list[str]:
    """Reject duplicate member agent_ids at request-parse time (clean 422)."""
    if len(set(value)) != len(value):
        raise ValueError("workflow member agent_ids must be distinct")
    return value


class CreateWorkflowRequest(BaseModel):
    """The POST /v1/workflows body: a workspace-scoped ordered agent pipeline."""

    workspace_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    members: list[str] = Field(min_length=1, max_length=svc.MAX_MEMBERS)
    idempotency_key: str | None = None

    _distinct = field_validator("members")(_validate_distinct_members)


class UpdateWorkflowRequest(BaseModel):
    """The PUT /v1/workflows/{id} body: replace the pipeline's members / name."""

    workspace_id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    members: list[str] = Field(min_length=1, max_length=svc.MAX_MEMBERS)

    _distinct = field_validator("members")(_validate_distinct_members)


class RunWorkflowRequest(BaseModel):
    """The POST /v1/workflows/{id}/run body: the prompt + run knobs."""

    workspace_id: str | None = None
    subject_id: str = "default"
    prompt: str = Field(min_length=1, max_length=100_000)
    idempotency_key: str | None = None


class WorkflowView(BaseModel):
    """The API view of a stored workflow."""

    id: str
    workspace_id: str
    name: str
    description: str
    members: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, workflow: svc.WorkflowRecord) -> WorkflowView:
        """Project a stored :class:`WorkflowRecord` into the API view."""
        return cls(
            id=workflow.id,
            workspace_id=workflow.workspace_id,
            name=workflow.name,
            description=workflow.description,
            members=list(workflow.members),
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
        )


class WorkflowListResponse(BaseModel):
    """The GET /v1/workflows envelope: the workspace's workflows."""

    items: list[WorkflowView]
    total: int


def _agent_app(request: Request) -> Any:
    """Pull the wired :class:`AgentDefAppService` off the app container."""
    return request.app.state.container.agent_app


def _run_app(request: Request) -> Any:
    """Pull the wired :class:`RunAppService` off the app container."""
    return request.app.state.container.run_app


def _operator_provisioned(request: Request) -> bool:
    """Whether THIS request may carry operator-only spec fields (mirrors agents.py)."""
    return bool(getattr(get_principal(request), "all_tenants", False))


async def _resolve_members(
    request: Request, member_ids: list[str], workspace_id: str
) -> list[Any]:
    """Resolve each member ``agent_id`` to a stored agent in ``workspace_id`` (404 otherwise)."""
    agent_app = _agent_app(request)
    records: list[Any] = []
    for agent_id in member_ids:
        record = await agent_app.get_agent_def(agent_id, workspace_id=workspace_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"workflow member agent {agent_id!r} not found in this workspace",
            )
        records.append(record)
    return records


# ---- CRUD ---------------------------------------------------------------------


@router.post("", response_model=WorkflowView, dependencies=_WRITE, status_code=201)
async def create_workflow(
    body: CreateWorkflowRequest, request: Request
) -> WorkflowView:
    """Create a workspace-scoped workflow (idempotent); every member is validated to exist."""
    workspace_id = require_workspace(request, body.workspace_id)
    store = svc.get_workflows_store()
    if body.idempotency_key is not None:
        existing = store.get_by_idempotency_key(
            body.idempotency_key, workspace_id=workspace_id
        )
        if existing is not None:
            return WorkflowView.of(existing)
    await _resolve_members(request, body.members, workspace_id)
    workflow = svc.WorkflowRecord(
        workspace_id=workspace_id,
        name=body.name,
        description=body.description,
        members=list(body.members),
        idempotency_key=body.idempotency_key,
    )
    stored = store.upsert(workflow)
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"workflow {stored.id} created",
    )
    return WorkflowView.of(stored)


@router.get("", response_model=WorkflowListResponse, dependencies=_READ)
async def list_workflows(
    request: Request, workspace_id: str | None = None
) -> WorkflowListResponse:
    """List the workflows in a workspace (tenant-scoped), newest first."""
    workspace_id = resolve_workspace(request, workspace_id)
    workflows = svc.get_workflows_store().list(workspace_id=workspace_id)
    items = [WorkflowView.of(w) for w in workflows]
    return WorkflowListResponse(items=items, total=len(items))


@router.get(
    "/{workflow_id}",
    response_model=WorkflowView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_workflow(
    workflow_id: str, request: Request, workspace_id: str | None = None
) -> WorkflowView:
    """Read one workflow by id (404 when unknown / out-of-workspace)."""
    workspace_id = resolve_workspace(request, workspace_id)
    workflow = svc.get_workflows_store().get(workflow_id, workspace_id=workspace_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return WorkflowView.of(workflow)


@router.put(
    "/{workflow_id}",
    response_model=WorkflowView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def update_workflow(
    workflow_id: str, body: UpdateWorkflowRequest, request: Request
) -> WorkflowView:
    """Replace a workflow's members / name (tenant-scoped). 404 when absent.

    The replacement members are re-validated to exist in the workspace.
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    store = svc.get_workflows_store()
    workflow = store.get(workflow_id, workspace_id=workspace_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    await _resolve_members(request, body.members, workspace_id)
    updated = workflow.model_copy(
        update={
            "name": body.name,
            "description": body.description,
            "members": list(body.members),
        }
    )
    stored = store.upsert(updated)
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="update",
        workspace_id=workspace_id,
        detail=f"workflow {workflow_id} updated",
    )
    return WorkflowView.of(stored)


@router.delete(
    "/{workflow_id}", responses=NOT_FOUND_RESPONSE, dependencies=_WRITE
)
async def delete_workflow(
    workflow_id: str, request: Request, workspace_id: str | None = None
) -> dict[str, str]:
    """Delete a workflow (tenant-scoped). 404 when absent / out-of-workspace."""
    workspace_id = resolve_workspace(request, workspace_id)
    if not svc.get_workflows_store().delete(workflow_id, workspace_id=workspace_id):
        raise HTTPException(status_code=404, detail="workflow not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="delete",
        workspace_id=workspace_id,
        detail=f"workflow {workflow_id} deleted",
    )
    return {"workflow_id": workflow_id, "status": "deleted"}


# ---- run ----------------------------------------------------------------------


@router.post(
    "/{workflow_id}/run",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def run_workflow(
    workflow_id: str, body: RunWorkflowRequest, request: Request
) -> RunRecord:
    """Launch the workflow pipeline as a canonical run (T3b).

    Re-resolves every member, then dispatches via
    ``RunAppService.create_orchestration_run`` (the durable linear graph pipeline) under the
    T0.4 quota. Returns the QUEUED :class:`RunRecord`; a quota breach is a 429.
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    # BOLA write gate (WS-bola): the orchestration RunRecord is stamped with the body's
    # subject_id, so a subject_scoped principal must only launch under its OWN subject —
    # else it could attribute a run (and its lineage / erasure linkage) to a foreign data
    # subject, exactly as runs.py:create_run guards. No-op offline / all_tenants /
    # tenant_admin.
    enforce_subject_write(request, body.subject_id)
    workflow = svc.get_workflows_store().get(workflow_id, workspace_id=workspace_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    members = await _resolve_members(request, workflow.members, workspace_id)
    try:
        run = cast(
            RunRecord,
            await _run_app(request).create_orchestration_run(
                workspace_id=workspace_id,
                subject_id=body.subject_id,
                kind="graph",
                members=members,
                prompt=body.prompt,
                resource_kind="workflow",
                resource_id=workflow.id,
                idempotency_key=body.idempotency_key,
                actor=get_principal(request).actor_metadata(),
                operator_provisioned=_operator_provisioned(request),
                graph_checkpoint_store=svc.get_graph_checkpoint_store(),
            ),
        )
    except WorkspaceRunQuotaExceeded:
        raise
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"workflow {workflow_id} run {run.run_id} launched",
    )
    return run


__all__ = ["router"]
