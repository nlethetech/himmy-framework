"""API kernel: the Himmy Studio router (the local web GUI's backend).

Studio is the no-code front door to himmy: chat with an agent, build/edit an
``agent.yaml``, and browse past runs/traces — all over the same FastAPI BFF. This
router is intentionally GUI-shaped (not the tenant-scoped ``/v1`` surface): it is
meant to be served on loopback by ``himmy studio`` for a single local user.

Endpoints are added per phase:
  * Phase 0 — ``GET /api/studio/doctor`` (environment diagnostics as JSON).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/studio", tags=["studio"])


@router.get("/doctor")
async def doctor() -> dict:
    """Environment diagnostics: extras, providers, keys, and the next step.

    The JSON twin of ``himmy doctor`` — same
    :func:`himmy.runtime.diagnostics.collect_doctor_report` snapshot the CLI prints.
    """
    from himmy.runtime.diagnostics import collect_doctor_report

    return collect_doctor_report().to_dict()


__all__ = ["router"]
