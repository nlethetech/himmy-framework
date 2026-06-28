"""API kernel: shared plumbing for the ``/api/studio/...`` router family.

Studio is a ~140-route operator console. Historically every route was gated by a
single coarse ``studio:use`` permission — a read-only auditor and a key-mutating
admin were indistinguishable to RBAC. This module replaces that with PER-SURFACE,
PER-ACTION permissions mirroring the ``/v1`` pattern (``studio.runs:read``,
``studio.mcp:manage``, ``studio.connections:write``, …):

* :func:`studio_permission` is a dependency FACTORY: ``studio_permission(resource,
  action)`` returns a route/router dependency that requires that one permission.
* :func:`build_studio_router` stamps each sub-router with a **read** default (so
  every GET on, say, ``/api/studio/files`` requires ``studio.files:read``); a
  mutating route additionally declares its write/manage permission via
  ``dependencies=[Depends(studio_permission(WRITE, "manage"))]``.

Both ultimately route through :func:`himmy.api.auth.rbac.require_permission`, so the
two load-bearing invariants are preserved BY CONSTRUCTION:

* **Offline / zero-config is byte-unchanged.** ``require_permission`` returns early
  when no authenticator is configured, so a granular Studio permission is a NO-OP on
  the single-box loopback default exactly as the old coarse guard was.
* **The ``HIMMY_STUDIO_AUTH=off`` kill-switch** still short-circuits every Studio
  permission (a trusted single-user escape hatch; refused at startup under a
  multi-tenant posture — see :func:`himmy.api.app._enforce_auth_posture`).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from himmy.api.auth import require_permission

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

#: The RBAC resource prefix every Studio permission lives under, so the granular
#: surface (``studio.runs``, ``studio.mcp``, …) is visibly distinct from the legacy
#: coarse ``studio:use`` and from the tenant-facing ``/v1`` resources (``run``,
#: ``context``). A read-only Studio role is expressed as ``studio.*:read`` in a
#: policy file; the default policy grants exactly that to viewer/operator/auditor.
STUDIO_RESOURCE_PREFIX = "studio"


def _studio_auth_off() -> bool:
    """Whether the ``HIMMY_STUDIO_AUTH`` kill-switch disables every Studio guard.

    The single reader both :func:`studio_permission` and any future Studio gate share,
    so the escape hatch can never be honored in one place and forgotten in another.
    DANGEROUS (re-opens the operator console to any authenticated principal); refused
    at startup under a multi-tenant posture.
    """
    return os.environ.get("HIMMY_STUDIO_AUTH", "on").lower() in (
        "off",
        "0",
        "false",
        "no",
    )


def studio_permission(
    resource: str = STUDIO_RESOURCE_PREFIX, action: str = "use"
) -> Callable[[Request], Awaitable[None]]:
    """Build a Studio route dependency requiring one ``resource:action`` permission.

    The granular replacement for the old single ``studio:use`` guard. Called with a
    per-surface resource and action (e.g. ``studio_permission("studio.mcp",
    "manage")``); the returned dependency:

    1. No-ops when ``HIMMY_STUDIO_AUTH`` is disabled (the single-user kill-switch); then
    2. delegates to :func:`himmy.api.auth.rbac.require_permission`, which itself
       no-ops when no authenticator is configured (offline-first) and otherwise 403s a
       principal whose roles do not grant the permission.

    The default arguments reproduce the legacy ``studio:use`` guard verbatim, so a
    caller that imports the bare dependency (``Depends(studio_permission)`` — note: no
    call) still gets the historical coarse behavior. ``admin`` satisfies any Studio
    permission via its ``*:*`` wildcard.
    """

    async def _dep(request: Request) -> None:
        if _studio_auth_off():
            return
        await require_permission(resource, action)(request)

    return _dep


def build_studio_router(
    segment: str,
    *,
    tag: str,
    read_action: str = "read",
) -> APIRouter:
    """A guarded ``APIRouter`` under ``/api/studio/<segment>`` with a per-surface READ guard.

    The router-level dependency requires ``studio.<segment>:<read_action>`` (default
    ``read``), so every route on the sub-router is covered by a least-privilege read
    grant out of the box — a read-only role can browse the surface. A MUTATING route
    declares its stronger permission additively via a route-level dependency, e.g.::

        @router.post("/servers", dependencies=[Depends(studio_permission(res, "manage"))])

    FastAPI runs router-level and route-level dependencies together, so a mutator
    effectively requires ``read`` AND ``manage``; admin (``*:*``) holds both, while a
    read-only role passes the read guard and is 403'd by the manage guard.

    Same ``/api/studio`` prefix family + the DNS-rebinding guard coverage in
    :func:`himmy.api.app._install_studio_guard` (which matches the path prefix) as the
    main Studio router.
    """
    resource = f"{STUDIO_RESOURCE_PREFIX}.{segment}"
    return APIRouter(
        prefix=f"/api/studio/{segment}",
        tags=[tag],
        dependencies=[Depends(studio_permission(resource, read_action))],
    )


__all__ = [
    "STUDIO_RESOURCE_PREFIX",
    "build_studio_router",
    "studio_permission",
]
