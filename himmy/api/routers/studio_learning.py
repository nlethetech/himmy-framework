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

Single-operator-local: like the seclog viewer the panel does not filter by workspace (the
``ws=None`` default scopes to the operator's own learning). TODO(multi-tenant): a multi-tenant
deployment MUST thread the request's resolved ``workspace_id`` into BOTH the GET ``load`` /
``build_learning_report`` AND the POST write paths (``set_override`` / ``set_trust_feedback``)
— otherwise one tenant could read or set another tenant's overrides.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.routers.studio_common import build_studio_router
from himmy.services.learning.overrides import (
    OverrideKind,
    OverrideSet,
    OverrideStore,
)
from himmy.services.learning.report import LearningReport, build_learning_report

router = build_studio_router("learning", tag="studio-learning")

#: The workspace the single-operator-local panel scopes to. ``None`` -> the ``default``
#: subject on the spine (parity with the CLI / Sprint-1 read path). TODO(multi-tenant):
#: resolve from the request principal and thread into every load / build / write below.
_PANEL_WORKSPACE: str | None = None


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


def _load_override_set() -> OverrideSet:
    """Load the operator overrides for the panel workspace (best-effort).

    Reaches the SAME ``.himmy/spine.db`` the runtime reads via
    :meth:`OverrideStore.for_context` (``server=True`` — this is the server entrypoint), so
    the panel and ``bound_tools`` share one store. Any spine/registry error degrades to an
    empty :class:`OverrideSet` (trust off) → the GET falls back to pure Sprint-1 auto-learning
    rather than 500-ing.
    """
    try:
        return OverrideStore.for_context(server=True).load(_PANEL_WORKSPACE)
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        return OverrideSet()


async def _build_response(
    request: Request, *, outcome_weight: float = 0.0
) -> LearningResponse:
    """Build the override-aware learning response (shared by GET + both POST writes)."""
    report = await build_learning_report(
        _event_log(request),
        workspace_id=_PANEL_WORKSPACE,
        outcome_weight=outcome_weight,
        override_set=_load_override_set(),
    )
    return _to_response(report)


@router.get("", response_model=LearningResponse)
async def learning(
    request: Request,
    outcome_weight: float = Query(0.0, ge=0.0, le=1.0),
) -> LearningResponse:
    """Return the override-aware learning report (per-tool reputation + recent reorders)."""
    return await _build_response(request, outcome_weight=outcome_weight)


@router.post("/overrides", response_model=LearningResponse)
async def set_override(
    request: Request,
    body: OverrideRequest,
) -> LearningResponse:
    """Pin / mute / reset / clear a tool, then return the refreshed learning report.

    Writes through :class:`OverrideStore` (``actor=human``, ``source=studio``) onto the
    shared spine the runtime reads. Best-effort: a swallowed store error still returns the
    (unchanged) refreshed report. TODO(multi-tenant): scope the write to the request's
    resolved ``workspace_id`` — today every write lands on the single-operator ``default``.
    """
    try:
        kind = OverrideKind(body.action.strip().lower())
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=f"unknown override action {body.action!r} (expected pin|mute|reset|clear)",
        )
    tool = body.tool_name.strip()
    if not tool:
        raise HTTPException(status_code=422, detail="tool_name must not be empty")
    try:
        store = OverrideStore.for_context(server=True)
        if kind is OverrideKind.CLEAR:
            store.clear_override(
                tool, workspace_id=_PANEL_WORKSPACE, actor="human", source="studio"
            )
        else:
            store.set_override(
                tool, kind, workspace_id=_PANEL_WORKSPACE, actor="human", source="studio"
            )
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        pass
    return await _build_response(request)


@router.post("/trust-feedback", response_model=LearningResponse)
async def set_trust_feedback(
    request: Request,
    body: TrustFeedbackRequest,
) -> LearningResponse:
    """Flip the per-workspace 👍/👎 trust gate, then return the refreshed learning report.

    Best-effort write through :class:`OverrideStore` (``actor=human``, ``source=studio``).
    TODO(multi-tenant): scope the write to the request's resolved ``workspace_id``.
    """
    try:
        OverrideStore.for_context(server=True).set_trust_feedback(
            bool(body.enabled),
            workspace_id=_PANEL_WORKSPACE,
            actor="human",
            source="studio",
        )
    except Exception:  # pragma: no cover - defensive: overrides never break the panel
        pass
    return await _build_response(request)


__all__ = ["router"]
