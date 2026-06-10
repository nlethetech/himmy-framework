"""API kernel: shared plumbing for the ``/api/studio/...`` router family.

Every Studio sub-router (privacy, lineage, files, …) mounts the same auth guard
as the main Studio router in :mod:`himmy.api.routers.studio`: once an
authenticator is configured, callers must hold the ``studio:use`` permission.
Centralizing the guard here keeps the hardening rule in one place so a new
Studio surface can never forget it.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request

from himmy.api.auth import require_permission


async def studio_permission(request: Request) -> None:
    """Require ``studio:use`` on every Studio route once auth is configured.

    No authenticator configured ⇒ no-op (the zero-config loopback default is
    unchanged). With auth on, the principal must hold a role granting
    ``studio:use`` (``admin`` — including the shared ``HIMMY_INTERNAL_API_KEY``
    boundary — qualifies via its ``*:*`` wildcard; grant it to other roles via
    ``HIMMY_RBAC_FILE``). Escape hatch: ``HIMMY_STUDIO_AUTH=off`` skips this
    check — DANGEROUS, as it re-opens Studio surfaces to any authenticated
    principal; only for a trusted single-user deployment.
    """
    if os.environ.get("HIMMY_STUDIO_AUTH", "on").lower() in ("off", "0", "false", "no"):
        return
    await require_permission("studio", "use")(request)


def build_studio_router(segment: str, *, tag: str) -> APIRouter:
    """A guarded ``APIRouter`` under ``/api/studio/<segment>``.

    Same prefix family + auth dependency as the main Studio router, so the
    DNS-rebinding guard in :func:`himmy.api.app._install_studio_guard` (which
    matches on the ``/api/studio`` path prefix) covers it too.
    """
    return APIRouter(
        prefix=f"/api/studio/{segment}",
        tags=[tag],
        dependencies=[Depends(studio_permission)],
    )


__all__ = ["build_studio_router", "studio_permission"]
