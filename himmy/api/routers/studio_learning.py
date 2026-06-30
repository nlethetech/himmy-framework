"""API kernel: Studio Learning router — the "what Himmy learned" panel + operator overrides.

The GUI sibling of the CLI's ``himmy learned``. It reads the container's durable event
store (the same stream the self-learning reputation miner reads) and returns a worst-first
view of per-tool reputation plus the recent ``LEARNING_APPLIED`` reorders, so an operator
can SEE which tools the agent demoted and why.

Sprint 2 ("teach & correct") makes the GET override-aware and adds two best-effort POST
write paths — ``POST .../overrides`` (pin / mute / reset / clear a tool) and
``POST .../trust-feedback`` (gate the 👍/👎 outcome blend). Both write through
:class:`~himmy.services.learning.overrides.OverrideStore` (``actor=human``, ``source=studio``)
onto the SAME ``.himmy/spine.db`` the runtime reads, so an override the operator sets in the
panel is the override the runtime honours (and the table can never disagree with
``bound_tools``). The GET and each write return the refreshed :class:`LearningResponse`.

Multi-tenant: the effective workspace is resolved from the request's verified principal
via :func:`~himmy.api.auth.context.resolve_workspace` and threaded into BOTH the GET
``load`` / ``build_learning_report`` read path AND the POST write paths
(``set_override`` / ``set_trust_feedback``), so one tenant can never read or mutate
another tenant's learning data. With no authenticator configured (the offline
single-operator default) the principal is unrestricted and the resolved workspace is
``None`` — byte-for-byte the legacy ``default``-subject behaviour.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.auth import resolve_workspace, scoped_read
from himmy.api.routers.studio_common import build_studio_router, studio_permission
from himmy.services.learning.overrides import (
    OverrideKind,
    OverrideSet,
    OverrideStore,
)
from himmy.services.learning.report import LearningReport, build_learning_report

router = build_studio_router("learning", tag="studio-learning")
# The learning report read is tenant-scoped: ``_panel_workspace`` resolves the effective
# ``workspace_id`` from the verified principal (a tenant-bound principal is pinned to its
# own workspace) and threads it into ``build_learning_report`` so a tenant only ever reads
# its own learned reputation + overrides. Stamp the router as a scoped reader for the
# whole-app tenant-scope coverage gate. A no-op marker; the real scoping is the handler.
router.dependencies.append(Depends(scoped_read))

#: Writing learning overrides and trust feedback steers how the agent picks tools — a
#: privileged mutation gated by ``studio.learning:write`` (admin-only by default),
#: additively on top of the router's ``studio.learning:read`` baseline so a read-only
#: role can inspect the learning report but never edit overrides.
_learning_write = Depends(studio_permission("studio.learning", "write"))


def _panel_workspace(request: Request) -> str | None:
    """Resolve the effective ``workspace_id`` for this request from its principal.

    Delegates to the BFF's single tenant-isolation choke point
    (:func:`~himmy.api.auth.context.resolve_workspace`): an unrestricted principal
    (offline default / trusted shared key) yields ``None`` (legacy ``default`` subject),
    while a tenant-bound principal is pinned to its own workspace (and a multi-tenant
    principal with no single default gets a 400) — so a tenant only ever reads/writes
    its own learning data.
    """
    return resolve_workspace(request, None)


class LearningToolModel(BaseModel):
    """One tool's learned reputation, as the table renders it."""

    tool_name: str
    score: float
    operational_score: float
    completed: int
    failed: int
    has_min_samples: bool
    flaky: bool
    outcome_score: float | None = None
    outcome_samples: int = 0
    #: The active operator override verdict — ``"pinned"`` / ``"ignored"`` / ``"reset"``,
    #: or ``None`` for pure auto-learning. Display label; the behaviour already lives in
    #: ``score`` / ``flaky`` (the runtime acts on the same post-override reputation).
    override: str | None = None


class ReorderModel(BaseModel):
    """A recorded ``LEARNING_APPLIED`` reorder."""

    when: str | None = None
    tools_reordered: int
    deprioritised_tools: list[str]


class LearningResponse(BaseModel):
    """The full learning panel payload."""

    floor: float
    outcome_weight: float
    #: Whether the per-workspace 👍/👎 trust gate is ON (blending recorded outcomes).
    trust_feedback: bool
    flaky_count: int
    has_data: bool
    tools: list[LearningToolModel]
    recent_reorders: list[ReorderModel]


class OverrideRequest(BaseModel):
    """``POST .../overrides`` body — one pin / mute / reset / clear for one tool."""

    tool_name: str
    action: str  # pin | mute | reset | clear


class TrustFeedbackRequest(BaseModel):
    """``POST .../trust-feedback`` body — flip the per-workspace outcome-blend gate."""

    enabled: bool


def _event_log(request: Request) -> Any:
    """The container's durable storage (event log), or a clear 404 when unwired."""
    container = getattr(request.app.state, "container", None)
    storage = getattr(container, "storage", None)
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail="no storage is wired on this server — learning is unavailable",
        )
    return storage


def _to_response(report: LearningReport) -> LearningResponse:
    """Project the dataclass report into the API model."""
    return LearningResponse(
        floor=report.floor,
        outcome_weight=report.outcome_weight,
        trust_feedback=report.trust_feedback,
        flaky_count=report.flaky_count,
        has_data=report.has_data,
        tools=[
            LearningToolModel(
                tool_name=t.tool_name,
                score=t.score,
                operational_score=t.operational_score,
                completed=t.completed,
                failed=t.failed,
                has_min_samples=t.has_min_samples,
                flaky=t.flaky,
                outcome_score=t.outcome_score,
                outcome_samples=t.outcome_samples,
                override=t.override,
            )
            for t in report.tools
        ],
        recent_reorders=[
            ReorderModel(
                when=r.when,
                tools_reordered=r.tools_reordered,
                deprioritised_tools=r.deprioritised_tools,
            )
            for r in report.recent_reorders
        ],
    )


def _load_override_set(workspace_id: str | None) -> OverrideSet:
    """Load the operator overrides for ``workspace_id`` (best-effort).

    Reaches the SAME ``.himmy/spine.db`` the runtime reads via
    :meth:`OverrideStore.for_context` (``server=True`` — this is the server entrypoint), so
    the panel and ``bound_tools`` share one store. Any spine/registry error degrades to an
    empty :class:`OverrideSet` (trust off) → the GET falls back to pure Sprint-1 auto-learning
    rather than 500-ing.
    """
    try:
        return OverrideStore.for_context(server=True).load(workspace_id)
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        return OverrideSet()


async def _build_response(
    request: Request, *, outcome_weight: float = 0.0
) -> LearningResponse:
    """Build the override-aware learning response (shared by GET + both POST writes).

    Scoped to the request principal's workspace so a tenant only ever sees its own
    learned reputation + overrides.
    """
    workspace_id = _panel_workspace(request)
    report = await build_learning_report(
        _event_log(request),
        workspace_id=workspace_id,
        outcome_weight=outcome_weight,
        override_set=_load_override_set(workspace_id),
    )
    return _to_response(report)


@router.get("", response_model=LearningResponse)
async def learning(
    request: Request,
    outcome_weight: float = Query(0.0, ge=0.0, le=1.0),
) -> LearningResponse:
    """Return the override-aware learning report (per-tool reputation + recent reorders)."""
    return await _build_response(request, outcome_weight=outcome_weight)


@router.post(
    "/overrides", response_model=LearningResponse, dependencies=[_learning_write]
)
async def set_override(
    request: Request,
    body: OverrideRequest,
) -> LearningResponse:
    """Pin / mute / reset / clear a tool, then return the refreshed learning report.

    Writes through :class:`OverrideStore` (``actor=human``, ``source=studio``) onto the
    shared spine the runtime reads, scoped to the request principal's workspace so a tenant
    can only mutate its own overrides. Best-effort: a swallowed store error still returns
    the (unchanged) refreshed report.
    """
    workspace_id = _panel_workspace(request)
    try:
        kind = OverrideKind(body.action.strip().lower())
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"unknown override action {body.action!r} (expected pin|mute|reset|clear)",
        ) from exc
    tool = body.tool_name.strip()
    if not tool:
        raise HTTPException(status_code=422, detail="tool_name must not be empty")
    try:
        store = OverrideStore.for_context(server=True)
        if kind is OverrideKind.CLEAR:
            store.clear_override(
                tool, workspace_id=workspace_id, actor="human", source="studio"
            )
        else:
            store.set_override(
                tool, kind, workspace_id=workspace_id, actor="human", source="studio"
            )
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        pass
    return await _build_response(request)


@router.post(
    "/trust-feedback", response_model=LearningResponse, dependencies=[_learning_write]
)
async def set_trust_feedback(
    request: Request,
    body: TrustFeedbackRequest,
) -> LearningResponse:
    """Flip the per-workspace 👍/👎 trust gate, then return the refreshed learning report.

    Best-effort write through :class:`OverrideStore` (``actor=human``, ``source=studio``),
    scoped to the request principal's workspace so a tenant can only flip its own gate.
    """
    workspace_id = _panel_workspace(request)
    try:
        OverrideStore.for_context(server=True).set_trust_feedback(
            bool(body.enabled),
            workspace_id=workspace_id,
            actor="human",
            source="studio",
        )
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        pass
    return await _build_response(request)


__all__ = ["router"]
