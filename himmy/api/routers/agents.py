"""API kernel: the ``/v1/agents`` router — durable, workspace-scoped stored agents (T2e).

A tenant stores a reusable :class:`~himmy.config.agent_spec.AgentSpec` once, then
launches runs by ``agent_id`` (``POST /v1/runs`` with ``agent_id``) instead of
re-sending an inline persona each time. The stored spec is the SAME declarative shape
the CLI writes to ``agent.yaml``, so the surfaces interoperate.

Security (the load-bearing part of this resource):

* every write goes through the T0.3 sanitizer — a tenant-submitted spec carrying the
  operator-only ``tools_module`` (import+call → RCE), ``http_tools`` (server-side HTTP
  → SSRF), or ``mcp_servers`` (stdio subprocess spawn) fields is REJECTED (or stripped
  with a warning when the deployment opts into strip mode), unless the spec is
  operator-provisioned AND the operator opted in;
* every path is RBAC-gated (``agent:read`` / ``agent:write``) and tenant-scoped via
  :func:`resolve_workspace` / :func:`require_workspace`, so one tenant's stored agent is
  invisible (404) to another in an authenticated deployment;
* create is idempotent on ``(workspace_id, idempotency_key)`` — a re-POST returns the
  prior record (201→200) instead of duplicating;
* DELETE refuses (409) an agent still referenced by a team/routine unless ``?cascade``.

Offline note: with no authenticator configured (the zero-config default) the caller is
the unrestricted operator, so RBAC + workspace scoping are inert and the sanitizer's
operator path applies — exactly the standalone-CLI posture.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from himmy.api.auth import (
    get_principal,
    require_permission,
    require_workspace,
    resolve_workspace,
    scoped_read,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.api.security_audit import audit_event
from himmy.application.services import AgentDefReferencedError
from himmy.config.agent_spec import AgentSpec
from himmy.config.spec_sanitizer import SpecSanitizationError
from himmy.services.storage.models import AgentDefRecord

router = APIRouter(
    prefix="/v1/agents", tags=["agents"], dependencies=[Depends(scoped_read)]
)

_READ = [Depends(require_permission("agent", "read"))]
_WRITE = [Depends(require_permission("agent", "write"))]


class CreateAgentRequest(BaseModel):
    """The POST /v1/agents body: the target workspace + the agent spec + idempotency.

    ``spec`` is a full :class:`AgentSpec` (the same declarative shape as ``agent.yaml``).
    Its operator-only fields are sanitized server-side before persistence.
    """

    workspace_id: str
    spec: AgentSpec
    idempotency_key: str | None = None


class UpdateAgentRequest(BaseModel):
    """The PUT /v1/agents/{id} body: the replacement spec for an existing stored agent."""

    workspace_id: str | None = None
    spec: AgentSpec


class AgentDefView(BaseModel):
    """The API view of a stored agent definition (the durable record + its spec)."""

    agent_id: str
    workspace_id: str
    name: str
    description: str = ""
    spec: dict[str, Any]
    idempotency_key: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, record: AgentDefRecord) -> AgentDefView:
        """Project a stored :class:`AgentDefRecord` into the API view."""
        return cls(
            agent_id=record.agent_id,
            workspace_id=record.workspace_id,
            name=record.name,
            description=record.description,
            spec=record.spec,
            idempotency_key=record.idempotency_key,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class AgentListResponse(BaseModel):
    """The GET /v1/agents envelope: the workspace's stored agents."""

    items: list[AgentDefView]
    total: int


def _agent_app(request: Request) -> Any:
    """Pull the wired :class:`AgentDefAppService` off the app container."""
    return request.app.state.container.agent_app


def _operator_provisioned(request: Request) -> bool:
    """Whether THIS request may carry operator-only spec fields.

    Only an unrestricted (offline / all-tenants) principal is treated as the operator;
    a tenant-bound principal is never operator-provisioned, so the sanitizer always
    bites for a real multi-tenant caller. The sanitizer additionally gates on the
    explicit ``HIMMY_ALLOW_OPERATOR_SPEC_TOOLS`` env flag, so even an operator cannot
    reach the dangerous fields by accident.
    """
    return bool(getattr(get_principal(request), "all_tenants", False))


@router.post(
    "",
    response_model=AgentDefView,
    dependencies=_WRITE,
    status_code=201,
)
async def create_agent(
    body: CreateAgentRequest, request: Request, response: Response
) -> AgentDefView:
    """Store a workspace-scoped agent definition (idempotent). 201 created / 200 existing.

    The spec is sanitized (T0.3): a tenant spec carrying ``tools_module``/``http_tools``/
    ``mcp_servers`` is rejected with 422 unless operator-provisioned (offline default).
    """
    workspace_id = require_workspace(request, body.workspace_id)
    try:
        record, created = await _agent_app(request).save_agent_def(
            body.spec,
            workspace_id=workspace_id,
            idempotency_key=body.idempotency_key,
            actor=get_principal(request).actor_metadata(),
            operator_provisioned=_operator_provisioned(request),
        )
    except SpecSanitizationError as exc:
        audit_event(
            request,
            event_type="access",
            outcome="deny",
            resource="agent",
            action="create",
            workspace_id=workspace_id,
            detail=f"rejected tenant spec privileged fields: {', '.join(exc.fields)}",
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response.status_code = 201 if created else 200
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="agent",
        action="create",
        workspace_id=workspace_id,
        detail=f"agent {record.agent_id} {'created' if created else 'idempotent-hit'}",
    )
    return AgentDefView.of(record)


@router.get("", response_model=AgentListResponse, dependencies=_READ)
async def list_agents(
    request: Request, workspace_id: str | None = None
) -> AgentListResponse:
    """List the stored agents in a workspace (tenant-scoped)."""
    workspace_id = resolve_workspace(request, workspace_id)
    records = await _agent_app(request).list_agent_defs(workspace_id=workspace_id)
    items = [AgentDefView.of(r) for r in records]
    return AgentListResponse(items=items, total=len(items))


@router.get(
    "/{agent_id}",
    response_model=AgentDefView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_agent(
    agent_id: str, request: Request, workspace_id: str | None = None
) -> AgentDefView:
    """Read one stored agent by id (404 when unknown / out-of-workspace)."""
    workspace_id = resolve_workspace(request, workspace_id)
    record = await _agent_app(request).get_agent_def(
        agent_id, workspace_id=workspace_id
    )
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return AgentDefView.of(record)


@router.put(
    "/{agent_id}",
    response_model=AgentDefView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def update_agent(
    agent_id: str, body: UpdateAgentRequest, request: Request
) -> AgentDefView:
    """Replace a stored agent's spec (tenant-scoped, re-sanitized). 404 when absent."""
    workspace_id = require_workspace(request, body.workspace_id or "")
    try:
        record = await _agent_app(request).update_agent_def(
            agent_id,
            body.spec,
            workspace_id=workspace_id,
            actor=get_principal(request).actor_metadata(),
            operator_provisioned=_operator_provisioned(request),
        )
    except SpecSanitizationError as exc:
        audit_event(
            request,
            event_type="access",
            outcome="deny",
            resource="agent",
            action="update",
            workspace_id=workspace_id,
            detail=f"rejected tenant spec privileged fields: {', '.join(exc.fields)}",
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="agent",
        action="update",
        workspace_id=workspace_id,
        detail=f"agent {agent_id} updated",
    )
    return AgentDefView.of(record)


@router.delete(
    "/{agent_id}", responses=NOT_FOUND_RESPONSE, dependencies=_WRITE
)
async def delete_agent(
    agent_id: str,
    request: Request,
    workspace_id: str | None = None,
    cascade: bool = Query(
        False, description="remove the agent even if referenced (caller owns cleanup)"
    ),
) -> dict[str, str]:
    """Delete a stored agent. 404 when absent; 409 when referenced (unless ``?cascade``)."""
    workspace_id = resolve_workspace(request, workspace_id)
    try:
        removed = await _agent_app(request).delete_agent_def(
            agent_id, workspace_id=workspace_id, cascade=cascade
        )
    except AgentDefReferencedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="agent not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="agent",
        action="delete",
        workspace_id=workspace_id,
        detail=f"agent {agent_id} deleted",
    )
    return {"agent_id": agent_id, "status": "deleted"}


__all__ = ["router"]
