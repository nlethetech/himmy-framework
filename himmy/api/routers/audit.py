"""API kernel: the /v1/audit router (read the security audit trail).

Read-only and gated by the ``audit:read`` permission (the ``auditor``/``admin``
roles), so only authorized callers can review who did what. Events are tenant-scoped
via the principal, like the rest of the BFF.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from himmy.api.auth import require_permission, resolve_workspace
from himmy.entities.integrity import AuditBundle
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


@router.get("/bundle", response_model=AuditBundle, dependencies=_AUDIT_READ)
async def export_audit_bundle_route(request: Request) -> AuditBundle:
    """Export a signed, tamper-evident bundle over the security-audit trail (WS4.5).

    Signs with Ed25519 when ``HIMMY_AUDIT_PRIVATE_KEY`` (PEM) is configured — an
    auditor then verifies offline with the public key alone — else HMAC with
    ``HIMMY_AUDIT_SECRET``. The bundle commits to every ``security_event`` entity, so
    tampering with the log after the fact is detectable.
    """
    from himmy.config.secrets import get_secret
    from himmy.entities.integrity import (
        export_audit_bundle,
        export_audit_bundle_ed25519,
    )
    from himmy.services.audit.log import SECURITY_EVENT_KIND

    registry = _container(request).entity_registry
    records = registry.list_by_kind(SECURITY_EVENT_KIND)
    private_pem = get_secret("HIMMY_AUDIT_PRIVATE_KEY")
    if private_pem:
        return export_audit_bundle_ed25519(records, [], private_pem=private_pem)
    secret = get_secret("HIMMY_AUDIT_SECRET")
    if secret:
        return export_audit_bundle(records, [], secret=secret)
    raise HTTPException(
        status_code=503,
        detail="audit signing key not configured "
        "(set HIMMY_AUDIT_PRIVATE_KEY or HIMMY_AUDIT_SECRET)",
    )


__all__ = ["router"]
