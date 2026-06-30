"""API kernel: the /v1/diagnostics router — global infra health (T3d).

``GET /v1/diagnostics`` is the REST analogue of ``himmy doctor`` (and Studio's doctor):
a read-only, secrets-redacted view of what the SERVER can do and where its durable state
lives — providers/keys/extras/embedders, the storage backend, the canonical ``.himmy``
SQLite stores, and whether the routine scheduler loop is active.

It is GLOBAL, not tenant-scoped: a per-tenant doctor is not meaningful (model availability,
the storage backend, and the scheduler are infra facts, shared across tenants). It is
read-gated on ``diagnostics:read`` once auth is configured, and never echoes a secret —
provider keys are reported by presence only and any Postgres DSN is password-redacted.

red-team scope-r5: ``diagnostics:read`` is held by every tenant browse role
(viewer/operator/auditor), so a tenant-bound caller could read OPERATOR DEPLOYMENT POSTURE
(installed-binary absolute paths, the provider-key presence inventory, the storage DSN host,
the ``.himmy/*.db`` paths) — the same reconnaissance class the Studio ``/doctor`` / ``/health``
twins withhold from tenant browse roles. A tenant-bound principal now receives the POSTURE-FREE
snapshot (:meth:`~himmy.runtime.diagnostics.DiagnosticsReport.to_public_dict`); the offline /
``all_tenants`` / CLI path keeps the full report byte-unchanged via
:func:`~himmy.api.auth.context.reveal_host_posture`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from himmy.api.auth import require_permission, reveal_host_posture

router = APIRouter(prefix="/v1/diagnostics", tags=["diagnostics"])

_READ = [Depends(require_permission("diagnostics", "read"))]


def _scheduler_active(request: Request) -> bool:
    """Whether the routine scheduler tick loop is running (best-effort, never raises)."""
    try:
        from himmy.api.routines import get_scheduler

        return bool(get_scheduler().active)
    except Exception:  # noqa: BLE001 - scheduler is optional; absence ⇒ inactive
        return False


@router.get("", dependencies=_READ)
async def diagnostics(request: Request) -> dict[str, Any]:
    """Global infra/health snapshot (secrets redacted, read-only, not per-tenant).

    Host-posture fields (installed-binary paths, key-presence inventory, DSN host, DB-file
    paths) are disclosed only to an unrestricted (``all_tenants`` / offline) principal; a
    tenant-bound caller gets the posture-free view — see the module docstring.
    """
    from himmy.runtime.diagnostics import collect_diagnostics_report

    report = collect_diagnostics_report(scheduler_active=_scheduler_active(request))
    if reveal_host_posture(request):
        return report.to_dict()
    return report.to_public_dict()


__all__ = ["router"]
