"""API kernel: the /v1/diagnostics router — global infra health (T3d).

``GET /v1/diagnostics`` is the REST analogue of ``himmy doctor`` (and Studio's doctor):
a read-only, secrets-redacted view of what the SERVER can do and where its durable state
lives — providers/keys/extras/embedders, the storage backend, the canonical ``.himmy``
SQLite stores, and whether the routine scheduler loop is active.

It is GLOBAL, not tenant-scoped: a per-tenant doctor is not meaningful (model availability,
the storage backend, and the scheduler are infra facts, shared across tenants). It is
read-gated on ``diagnostics:read`` once auth is configured, and never echoes a secret —
provider keys are reported by presence only and any Postgres DSN is password-redacted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from himmy.api.auth import require_permission

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
    """Global infra/health snapshot (secrets redacted, read-only, not per-tenant)."""
    from himmy.runtime.diagnostics import collect_diagnostics_report

    report = collect_diagnostics_report(scheduler_active=_scheduler_active(request))
    return report.to_dict()


__all__ = ["router"]
