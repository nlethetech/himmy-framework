"""API kernel: the ``/v1/teams`` router — workspace-scoped multi-agent teams (T3b).

A team is an ORDERED list of STORED agent ids (the ``/v1/agents`` resource, T2e) plus an
orchestration ``kind`` (``multi_agent`` | ``group_chat`` | ``graph``). It is a durable,
workspace-scoped resource stored in its own ``.himmy/teams.db`` (the routines-store
pattern).

Two load-bearing properties:

* **Referential integrity at create.** Every member ``agent_id`` must EXIST and be in the
  SAME workspace; an unknown / cross-workspace member is a 404 at create time. The same
  reference set powers the T2e delete-409 path: deleting an agent still referenced by a
  team is refused (the production reference_finder wired in ``deps.py``).
* **Runs on the EXISTING run machinery.** ``POST /{id}/run`` launches the multi-agent
  orchestration via ``RunAppService.create_orchestration_run`` — the SAME background-task +
  T0.4 per-workspace quota machinery as a single-agent run, NOT a second executor. The
  ``graph`` kind threads the durable :class:`SqliteGraphCheckpointStore` so a long run
  resumes after a restart. The result is a canonical :class:`RunRecord` polled via
  ``GET /v1/runs/{id}``.

Security: ``run:read`` for GET, ``run:write`` for create/run (a team is a run container, so
it rides the ``run`` RBAC resource). Every route is tenant-scoped via ``resolve_workspace``
/ ``require_workspace`` (a cross-workspace team is a clean 404). With no authenticator (the
zero-config default) the caller is the unrestricted operator, so RBAC + scoping are inert —
the standalone posture.
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
    prefix="/v1/teams", tags=["teams"], dependencies=[Depends(scoped_read)]
)

_READ = [Depends(require_permission("run", "read"))]
_WRITE = [Depends(require_permission("run", "write"))]


# ---- request/response models -------------------------------------------------


def _validate_distinct_members(value: list[str]) -> list[str]:
    """Reject duplicate member agent_ids at request-parse time (clean 422)."""
    if len(set(value)) != len(value):
        raise ValueError("team member agent_ids must be distinct")
    return value


class CreateTeamRequest(BaseModel):
    """The POST /v1/teams body: a workspace-scoped team of ordered stored agents."""

    workspace_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    kind: svc.OrchestrationKind = "multi_agent"
    members: list[str] = Field(min_length=1, max_length=svc.MAX_MEMBERS)
    idempotency_key: str | None = None

    _distinct = field_validator("members")(_validate_distinct_members)


class UpdateTeamRequest(BaseModel):
    """The PUT /v1/teams/{id} body: replace the team's members / kind / name."""

    workspace_id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    kind: svc.OrchestrationKind = "multi_agent"
    members: list[str] = Field(min_length=1, max_length=svc.MAX_MEMBERS)

    _distinct = field_validator("members")(_validate_distinct_members)


class RunTeamRequest(BaseModel):
    """The POST /v1/teams/{id}/run body: the prompt + run knobs."""

    workspace_id: str | None = None
    subject_id: str = "default"
    prompt: str = Field(min_length=1, max_length=100_000)
    idempotency_key: str | None = None


class TeamView(BaseModel):
    """The API view of a stored team."""

    id: str
    workspace_id: str
    name: str
    description: str
    kind: svc.OrchestrationKind
    members: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, team: svc.TeamRecord) -> TeamView:
        """Project a stored :class:`TeamRecord` into the API view."""
        return cls(
            id=team.id,
            workspace_id=team.workspace_id,
            name=team.name,
            description=team.description,
            kind=team.kind,
            members=list(team.members),
            created_at=team.created_at,
            updated_at=team.updated_at,
        )


class TeamListResponse(BaseModel):
    """The GET /v1/teams envelope: the workspace's teams."""

    items: list[TeamView]
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
    """Resolve each member ``agent_id`` to a stored agent in ``workspace_id`` (T3b ref-check).

    Returns the ordered :class:`AgentDefRecord`s. Raises 404 on the FIRST member that does
    not exist in the workspace — referential integrity: a team may only bind agents that
    exist AND live in the same workspace (no cross-workspace membership, no FS-leak path).
    """
    agent_app = _agent_app(request)
    records: list[Any] = []
    for agent_id in member_ids:
        record = await agent_app.get_agent_def(agent_id, workspace_id=workspace_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"team member agent {agent_id!r} not found in this workspace",
            )
        records.append(record)
    return records


# ---- CRUD ---------------------------------------------------------------------


@router.post("", response_model=TeamView, dependencies=_WRITE, status_code=201)
async def create_team(body: CreateTeamRequest, request: Request) -> TeamView:
    """Create a workspace-scoped team (idempotent); every member is validated to exist.

    Each member ``agent_id`` must resolve to a stored agent in the same workspace (404
    otherwise). A re-POST of an existing ``idempotency_key`` returns the prior team.
    """
    workspace_id = require_workspace(request, body.workspace_id)
    store = svc.get_teams_store()
    if body.idempotency_key is not None:
        existing = store.get_by_idempotency_key(
            body.idempotency_key, workspace_id=workspace_id
        )
        if existing is not None:
            return TeamView.of(existing)
    await _resolve_members(request, body.members, workspace_id)
    team = svc.TeamRecord(
        workspace_id=workspace_id,
        name=body.name,
        description=body.description,
        kind=body.kind,
        members=list(body.members),
        idempotency_key=body.idempotency_key,
    )
    stored = store.upsert(team)
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"team {stored.id} created",
    )
    return TeamView.of(stored)


@router.get("", response_model=TeamListResponse, dependencies=_READ)
async def list_teams(
    request: Request, workspace_id: str | None = None
) -> TeamListResponse:
    """List the teams in a workspace (tenant-scoped), newest first."""
    workspace_id = resolve_workspace(request, workspace_id)
    teams = svc.get_teams_store().list(workspace_id=workspace_id)
    items = [TeamView.of(t) for t in teams]
    return TeamListResponse(items=items, total=len(items))


@router.get(
    "/{team_id}",
    response_model=TeamView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_team(
    team_id: str, request: Request, workspace_id: str | None = None
) -> TeamView:
    """Read one team by id (404 when unknown / out-of-workspace)."""
    workspace_id = resolve_workspace(request, workspace_id)
    team = svc.get_teams_store().get(team_id, workspace_id=workspace_id)
    if team is None:
        raise HTTPException(status_code=404, detail="team not found")
    return TeamView.of(team)


@router.put(
    "/{team_id}",
    response_model=TeamView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def update_team(
    team_id: str, body: UpdateTeamRequest, request: Request
) -> TeamView:
    """Replace a team's members / kind / name (tenant-scoped). 404 when absent.

    The replacement members are re-validated to exist in the workspace (referential
    integrity preserved on edit).
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    store = svc.get_teams_store()
    team = store.get(team_id, workspace_id=workspace_id)
    if team is None:
        raise HTTPException(status_code=404, detail="team not found")
    await _resolve_members(request, body.members, workspace_id)
    updated = team.model_copy(
        update={
            "name": body.name,
            "description": body.description,
            "kind": body.kind,
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
        detail=f"team {team_id} updated",
    )
    return TeamView.of(stored)


@router.delete("/{team_id}", responses=NOT_FOUND_RESPONSE, dependencies=_WRITE)
async def delete_team(
    team_id: str, request: Request, workspace_id: str | None = None
) -> dict[str, str]:
    """Delete a team (tenant-scoped). 404 when absent / out-of-workspace."""
    workspace_id = resolve_workspace(request, workspace_id)
    if not svc.get_teams_store().delete(team_id, workspace_id=workspace_id):
        raise HTTPException(status_code=404, detail="team not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="delete",
        workspace_id=workspace_id,
        detail=f"team {team_id} deleted",
    )
    return {"team_id": team_id, "status": "deleted"}


# ---- run ----------------------------------------------------------------------


@router.post(
    "/{team_id}/run",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def run_team(
    team_id: str, body: RunTeamRequest, request: Request
) -> RunRecord:
    """Launch the team's orchestration as a canonical run (T3b).

    Re-resolves every member to a stored agent in the workspace (404 if one has been
    removed since create), then dispatches via ``RunAppService.create_orchestration_run``
    under the T0.4 per-workspace quota. Returns the QUEUED :class:`RunRecord` (poll
    ``GET /v1/runs/{id}`` for the outcome); a quota breach is a 429.
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    # BOLA write gate (WS-bola): the orchestration RunRecord is stamped with the body's
    # subject_id, so a subject_scoped principal must only launch under its OWN subject —
    # else it could attribute a run (and its lineage / erasure linkage) to a foreign data
    # subject, exactly as runs.py:create_run guards. No-op offline / all_tenants /
    # tenant_admin.
    enforce_subject_write(request, body.subject_id)
    team = svc.get_teams_store().get(team_id, workspace_id=workspace_id)
    if team is None:
        raise HTTPException(status_code=404, detail="team not found")
    members = await _resolve_members(request, team.members, workspace_id)
    graph_store = (
        svc.get_graph_checkpoint_store() if team.kind == "graph" else None
    )
    try:
        run = cast(
            RunRecord,
            await _run_app(request).create_orchestration_run(
                workspace_id=workspace_id,
                subject_id=body.subject_id,
                kind=team.kind,
                members=members,
                prompt=body.prompt,
                resource_kind="team",
                resource_id=team.id,
                idempotency_key=body.idempotency_key,
                actor=get_principal(request).actor_metadata(),
                operator_provisioned=_operator_provisioned(request),
                graph_checkpoint_store=graph_store,
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
        detail=f"team {team_id} run {run.run_id} launched",
    )
    return run


__all__ = ["router"]
