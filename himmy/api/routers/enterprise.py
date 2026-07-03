"""API kernel: the /v1/enterprise router — the Enterprise Edition governance console.

A single read-only governance / observability / cost rollup that makes Himmy's enterprise
value VISIBLE in one call:

* ``GET /v1/enterprise/overview`` — a windowed rollup for the caller's tenant: an agents
  summary, a runs rollup (count by status), COST attributed per agent and per model (from
  the run cost fields + the inference-event cost/token metrics), recent security/audit
  events, an RBAC/policy summary (roles + permission counts), and health/readiness.

The whole surface is an ENTERPRISE feature: it is gated behind
:func:`~himmy.licensing.require_entitlement` for :data:`FEATURE_ENTERPRISE_CONSOLE`, so a
community install gets a **402** with the actionable upgrade message rather than data — the
OSS / offline / single-tenant core is unaffected. On top of the edition gate the caller
must also hold the ``dashboard:read`` RBAC permission (the same read grant the operator
overview uses), so an entitled-but-unauthorized principal is a clean **403**.

Tenant scoping: the rollup is scoped to the caller's authorized workspace via
:func:`~himmy.api.auth.resolve_workspace`, so a tenant-bound principal never sees another
tenant's agents, runs, cost or security events. An ``all_tenants`` (offline / admin)
principal resolves to ``None`` and sees the unscoped, cross-tenant rollup.

Efficiency: the rollup issues a bounded, fixed number of store reads (one ``list_runs``,
one ``list_agent_defs``, one ``list_events``, one security-log ``recent``) and aggregates
in memory — there is no per-run / per-agent follow-up query (no N+1).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from himmy.api.auth import require_permission, resolve_workspace, scoped_read
from himmy.api.auth.rbac import DEFAULT_POLICY
from himmy.core.events import EventType
from himmy.licensing import (
    FEATURE_ENTERPRISE_CONSOLE,
    EnterpriseFeatureError,
    require_entitlement,
)

router = APIRouter(
    prefix="/v1/enterprise", tags=["enterprise"], dependencies=[Depends(scoped_read)]
)

#: The console reads the same governance data the operator overview does; ``dashboard:read``
#: is held by the built-in viewer / operator / auditor / admin roles.
_READ = [Depends(require_permission("dashboard", "read"))]

#: The inference lifecycle events that carry a per-call ``cost`` (the terminal ones).
_COST_EVENTS = frozenset(
    {EventType.INFERENCE_SUCCEEDED, EventType.INFERENCE_FAILED}
)


class CostBucket(BaseModel):
    """A cost + call-count rollup for one attribution key (an agent or a model)."""

    key: str
    cost: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AgentsSummary(BaseModel):
    """The tenant's registered agents (count + a light listing)."""

    total: int = 0
    agents: list[dict[str, Any]] = Field(default_factory=list)


class RunsSummary(BaseModel):
    """A runs rollup: total + count by lifecycle status."""

    total: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)


class RbacSummary(BaseModel):
    """The active RBAC policy shape: roles and their permission counts."""

    roles: list[str] = Field(default_factory=list)
    permission_counts: dict[str, int] = Field(default_factory=dict)


class HealthSummary(BaseModel):
    """Health / readiness truth for the console header."""

    edition: str
    backend: str
    durable: bool
    ready: bool


class EnterpriseOverview(BaseModel):
    """The one-call governance / observability / cost rollup (tenant-scoped)."""

    workspace_id: str | None = None
    agents: AgentsSummary
    runs: RunsSummary
    cost_total: float = 0.0
    cost_by_agent: list[CostBucket] = Field(default_factory=list)
    cost_by_model: list[CostBucket] = Field(default_factory=list)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)
    rbac: RbacSummary
    health: HealthSummary


def _container(request: Request) -> Any:
    """Pull the wired :class:`~himmy.api.deps.ApiContainer` off the app state."""
    return request.app.state.container


def _rbac_summary(request: Request) -> RbacSummary:
    """Summarize the active RBAC policy: roles + per-role permission counts.

    Reads the process's resolved :class:`~himmy.api.auth.rbac.AccessPolicy` off app state
    (falling back to the built-in :data:`DEFAULT_POLICY`), so the console reflects the
    SAME policy the request pipeline actually enforces — a custom ``HIMMY_RBAC_FILE`` is
    surfaced, not the defaults.
    """
    policy = getattr(request.app.state, "access_policy", None) or DEFAULT_POLICY
    perms = policy.role_permissions
    return RbacSummary(
        roles=sorted(perms.keys()),
        permission_counts={role: len(p) for role, p in perms.items()},
    )


def _health_summary(request: Request) -> HealthSummary:
    """Health / readiness truth (edition + storage durability), read from app state.

    Mirrors the ``/readyz`` durability truth stamped by ``_bind_state`` (``active_backend``
    / ``built_durable``); a non-durable in-memory backend is still ``ready`` (the readiness
    contract only *requires* durability when it was explicitly requested).
    """
    from himmy.licensing import current_edition

    state = request.app.state
    backend = str(getattr(state, "active_backend", "memory"))
    durable = backend not in ("memory", "")
    requested = bool(getattr(state, "durable_requested", False))
    built = bool(getattr(state, "built_durable", durable))
    return HealthSummary(
        edition=current_edition().value,
        backend=backend,
        durable=durable,
        ready=(built or not requested),
    )


@router.get("/overview", response_model=EnterpriseOverview, dependencies=_READ)
async def enterprise_overview(
    request: Request,
    workspace_id: str | None = None,
    events_limit: int = Query(
        20, ge=1, le=200, description="recent security-event rows to include"
    ),
    agents_limit: int = Query(
        50, ge=1, le=500, description="agents listed in the summary (count is unbounded)"
    ),
) -> EnterpriseOverview:
    """Return the governance / cost rollup for the caller's tenant (Enterprise Edition).

    Edition-gated: a community install is refused **402** with the upgrade message (the
    OSS core is untouched). RBAC-gated: the caller must hold ``dashboard:read`` (**403**
    otherwise). Tenant-scoped: :func:`resolve_workspace` pins the rollup to the caller's
    authorized workspace, so a tenant-bound principal never aggregates across tenants; an
    ``all_tenants`` principal sees the unscoped rollup.

    The rollup issues a fixed, bounded set of store reads and aggregates in memory (no
    N+1). Cost is attributed from the terminal inference events' ``cost`` field — per agent
    (the event ``agent_id``) and per model (the run's ``model_key``, joined by ``trace_id``).
    """
    # 1) Edition gate (fail-closed to community with an actionable 402) — BEFORE any read.
    try:
        require_entitlement(FEATURE_ENTERPRISE_CONSOLE)
    except EnterpriseFeatureError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc

    workspace_id = resolve_workspace(request, workspace_id)
    container = _container(request)
    storage = container.storage

    # 2) Bounded store reads (no per-row follow-up query).
    runs = await storage.list_runs(workspace_id=workspace_id)
    agent_defs = await storage.list_agent_defs(workspace_id=workspace_id)
    events = await storage.list_events(workspace_id=workspace_id)

    # 3) Runs rollup: count by status.
    by_status: dict[str, int] = {}
    for run in runs:
        by_status[run.status.value] = by_status.get(run.status.value, 0) + 1

    # 4) Cost attribution. ``trace_id -> model_key`` from the run records lets us attribute
    #    each inference event's cost to a model without a second query per event.
    trace_model: dict[str, str] = {
        run.trace_id: (run.model_key or "unknown")
        for run in runs
        if run.trace_id
    }
    by_agent: dict[str, CostBucket] = {}
    by_model: dict[str, CostBucket] = {}
    cost_total = 0.0
    for event in events:
        if event.event_type not in _COST_EVENTS:
            continue
        cost = float(event.cost or 0.0)
        cost_total += cost
        payload = event.payload or {}
        in_tok = int(payload.get("input_tokens") or 0)
        out_tok = int(payload.get("output_tokens") or 0)

        agent_key = event.agent_id or "unknown"
        bucket = by_agent.setdefault(agent_key, CostBucket(key=agent_key))
        bucket.cost += cost
        bucket.calls += 1
        bucket.input_tokens += in_tok
        bucket.output_tokens += out_tok

        model_key = trace_model.get(event.trace_id or "", "unknown")
        mbucket = by_model.setdefault(model_key, CostBucket(key=model_key))
        mbucket.cost += cost
        mbucket.calls += 1
        mbucket.input_tokens += in_tok
        mbucket.output_tokens += out_tok

    cost_by_agent = sorted(by_agent.values(), key=lambda b: b.cost, reverse=True)
    cost_by_model = sorted(by_model.values(), key=lambda b: b.cost, reverse=True)

    # 5) Recent security / audit events, tenant-scoped to the SAME workspace. ``None`` (an
    #    all-tenants principal) keeps the unscoped recent window.
    audit = getattr(request.app.state, "security_audit", None)
    recent_events: list[dict[str, Any]] = []
    if audit is not None:
        for e in audit.recent(limit=events_limit, workspace_id=workspace_id):
            recent_events.append(
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "outcome": e.outcome,
                    "created_at": e.created_at,
                    "actor": (e.actor or {}).get("subject")
                    or (e.actor or {}).get("id"),
                    "resource": e.resource,
                    "action": e.action,
                    "detail": e.detail,
                }
            )

    # 6) Agents summary (unbounded count, bounded listing).
    agents = AgentsSummary(
        total=len(agent_defs),
        agents=[
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "created_at": a.created_at,
            }
            for a in agent_defs[:agents_limit]
        ],
    )

    return EnterpriseOverview(
        workspace_id=workspace_id,
        agents=agents,
        runs=RunsSummary(total=len(runs), by_status=by_status),
        cost_total=cost_total,
        cost_by_agent=cost_by_agent,
        cost_by_model=cost_by_model,
        recent_events=recent_events,
        rbac=_rbac_summary(request),
        health=_health_summary(request),
    )


__all__ = ["router"]
