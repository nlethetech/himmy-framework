"""API kernel: Studio security-log router — the read-only seclog viewer (T3f).

Mounted under ``/api/studio/seclog`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). It is the GUI sibling of the CLI's
``himmy seclog`` command: a read-only table of the recent
:class:`~himmy.services.audit.models.SecurityEvent` records — denied tools,
approvals, policy blocks, auth/authz decisions — that the Privacy screen renders
beside its consent/erasure/audit-bundle ledger. One endpoint:

* ``GET /``  — recent security events (newest first) with the SAME ``limit`` and
  ``type`` filters :func:`himmy.cli.security_audit_cmd.render_seclog` exposes.

DATA SOURCE (reviewer must_fix): the events are read from
``app.state.security_audit`` — a :class:`~himmy.services.audit.log.SecurityAuditLog`
over the container's entity registry, which since T2.1 (``SpineFactory``) IS the
durable shared ``.himmy/spine.db`` the CLI ``himmy seclog`` also reads. So a deny
recorded by ``himmy run`` and a deny recorded by the server land in ONE database,
and this viewer's rows are IDENTICAL to ``himmy seclog --limit N --type T`` for the
same project — NOT the old per-process in-memory registry that would have shown a
different event set and falsely claimed parity.

Single-user-local: no workspace filter (the seclog is the operator's own audit
trail). Offline-first: a clean trail simply returns ``[]`` (the table renders its
empty state); reading is never the access gate, so this endpoint never 500s on an
empty/odd spine.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("seclog", tag="studio-seclog")

#: Row cap bounds — mirror the CLI default (20) but allow a deeper browse.
_LIMIT_DEFAULT = 50
_LIMIT_MAX = 500


class SeclogEvent(BaseModel):
    """One security event row (the flat shape the table renders).

    A projection of :class:`~himmy.services.audit.models.SecurityEvent` chosen to
    match the columns ``himmy seclog`` prints (time · actor · action · resource ·
    outcome) plus the fields the GUI needs to filter/deep-link.
    """

    event_id: str
    event_type: str
    outcome: str
    created_at: str
    actor: str  # the principal subject (the CLI's ``actor`` column), "-" when unknown
    resource: str | None = None
    action: str | None = None
    method: str | None = None
    path: str | None = None
    detail: str = ""


class SeclogResponse(BaseModel):
    """Recent security events + the distinct event types present (for the filter)."""

    items: list[SeclogEvent]
    total: int  # rows returned (after the limit/type filter)
    types: list[str]  # distinct event_types in the recent window (filter options)


def _security_audit(request: Request) -> Any:
    """The wired :class:`SecurityAuditLog`, or a clear 404 when none exists.

    ``app.state.security_audit`` is set in :func:`himmy.api.app._bind_state` /
    ``create_app`` over the container's entity registry — the durable shared spine
    since T2.1. It is always present on a fully-built app, so the 404 only guards a
    partially-wired test app, never a real deployment.
    """
    audit = getattr(request.app.state, "security_audit", None)
    if audit is None:
        raise HTTPException(
            status_code=404,
            detail="no security audit log is wired on this server",
        )
    return audit


def _actor_label(actor: dict[str, Any]) -> str:
    """The principal that acted — the actor ``subject`` (mirrors the CLI's column).

    Matches :func:`himmy.cli.security_audit_cmd._actor_label`: ``subject`` →
    ``id`` → ``name``, falling back to ``-`` so the column is never blank.
    """
    subject = actor.get("subject") or actor.get("id") or actor.get("name")
    return str(subject) if subject else "-"


@router.get("", response_model=SeclogResponse)
async def seclog(
    request: Request,
    limit: int = Query(_LIMIT_DEFAULT, ge=1, le=_LIMIT_MAX),
    type: str | None = Query(None, max_length=64),
) -> SeclogResponse:
    """Recent security events from the durable shared spine (same as ``himmy seclog``).

    ``limit`` bounds the rows (newest first); ``type`` filters by ``event_type``
    (``auth_failure`` / ``authz_denied`` / ``access`` / ``admin``) — the SAME two
    filters :func:`himmy.cli.security_audit_cmd.render_seclog` exposes. The rows are
    read from ``app.state.security_audit`` over ``.himmy/spine.db``, so they are
    byte-for-byte the rows the CLI prints for the same project.
    """
    audit = _security_audit(request)
    # The CLI's render_seclog passes event_type/limit straight to ``recent`` with no
    # workspace filter for the local viewer; mirror that exactly (single-user-local).
    events = audit.recent(limit=limit, event_type=type, workspace_id=None)
    items = [
        SeclogEvent(
            event_id=e.event_id,
            event_type=e.event_type,
            outcome=e.outcome,
            created_at=e.created_at,
            actor=_actor_label(e.actor or {}),
            resource=e.resource,
            action=e.action,
            method=e.method,
            path=e.path,
            detail=e.detail,
        )
        for e in events
    ]
    # Distinct types over the SAME recent window (unfiltered) so the GUI filter
    # offers the event types actually present, not a hardcoded list.
    window = audit.recent(limit=limit, event_type=None, workspace_id=None)
    types = sorted({e.event_type for e in window if e.event_type})
    return SeclogResponse(items=items, total=len(items), types=types)


__all__ = ["router"]
