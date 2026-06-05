"""API kernel: the /v1/audit router (read the security audit trail).

Read-only and gated by the ``audit:read`` permission (the ``auditor``/``admin``
roles), so only authorized callers can review who did what. Events are tenant-scoped
via the principal, like the rest of the BFF.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from himmy.api.auth import require_permission, resolve_workspace
from himmy.services.audit.models import SecurityEvent

router = APIRouter(prefix="/v1/audit", tags=["audit"])

_AUDIT_READ = [Depends(require_permission("audit", "read"))]


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.get("/events", response_model=list[SecurityEvent], dependencies=_AUDIT_READ)
async def list_security_events(
    request: Request,
    workspace_id: str | None = None,
    event_type: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[SecurityEvent]:
    """List recent security events (newest first), tenant-scoped (auditor/admin)."""
    workspace_id = resolve_workspace(request, workspace_id)
    log = getattr(request.app.state, "security_audit", None)
    if log is None:
        return []
    return log.recent(limit=limit, workspace_id=workspace_id, event_type=event_type)


__all__ = ["router"]
