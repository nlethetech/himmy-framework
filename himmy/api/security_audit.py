"""Request-aware emit helper for the security audit log (WS1.4).

:func:`audit_event` builds a :class:`SecurityEvent` from the request + its principal
and records it into ``app.state.security_audit``. It is a **no-op when no authenticator
is configured**, so the offline/zero-config path records nothing and is unchanged; in a
configured deployment it captures auth failures, authz denials, and data access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.services.audit.models import SecurityEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request


def audit_event(
    request: Request,
    *,
    event_type: str,
    outcome: str,
    resource: str | None = None,
    action: str | None = None,
    workspace_id: str | None = None,
    detail: str = "",
) -> None:
    """Record a security event for this request (no-op when auth is not configured)."""
    state = request.app.state
    if getattr(state, "authenticator", None) is None:
        return  # offline / no-auth: security audit is off (zero-config unchanged)
    log = getattr(state, "security_audit", None)
    if log is None:  # pragma: no cover - always wired alongside the authenticator
        return
    from himmy.api.auth.context import get_principal

    log.record(
        SecurityEvent(
            event_type=event_type,
            outcome=outcome,
            actor=get_principal(request).actor_metadata(),
            resource=resource,
            action=action,
            workspace_id=workspace_id,
            method=request.method,
            path=str(request.url.path),
            detail=detail,
        )
    )


__all__ = ["audit_event"]
