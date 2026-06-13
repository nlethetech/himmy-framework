"""API kernel: the read-only ``/v1/connectors`` catalog (coherence Tier-1, T1.3).

A read-only view of the connector catalog the CLI ``himmy connectors`` and the Studio
Connections screen drive: for each built-in connector, its kind, the secret NAMES it
needs (with presence flags only), its allow-list field, and whether it is configured /
enabled / live-available. This endpoint NEVER returns a secret value — only whether each
named secret is present — and it is read-only by design: configuring a connector stays on
the CLI/Studio path (``/v1`` connector CONFIG is explicitly out of scope per the plan).

The catalog is process-global (connectors are an operator resource, not a per-tenant
one — like ``/v1/models``), so there is no workspace scoping; it is RBAC-gated on
``connector:read`` (bypassed offline, enforced once an authenticator is configured).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from himmy.api.auth import require_permission
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.connectors.manage import ConnectorService

_READ = [Depends(require_permission("connector", "read"))]

router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


def _service() -> ConnectorService:
    """Build the connector service (built-ins ensured-registered on construction)."""
    return ConnectorService()


@router.get("", dependencies=_READ)
async def list_connectors() -> dict[str, Any]:
    """The connector catalog with per-connector configured/enabled status (no secrets)."""
    catalog = _service().catalog()
    return {"connectors": [status.to_dict() for status in catalog]}


@router.get("/{name}", dependencies=_READ, responses=NOT_FOUND_RESPONSE)
async def get_connector(name: str) -> dict[str, Any]:
    """One connector's catalog row (no secret values); 404 for an unknown name."""
    try:
        status = _service().status(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no connector named {name!r}"
        ) from exc
    return status.to_dict()


__all__ = ["router"]
