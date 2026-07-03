"""API kernel: the /v1/audit/privacy router (run + read the privacy audit, WS4.7 B4).

Surfaces the :class:`~himmy.services.evaluation.privacy.service.PrivacyAuditService` over
HTTP so an operator can *actively* run a Ragas-inspired privacy & compliance audit and an
auditor can read its posture trend and any one signed report:

* ``POST /v1/audit/privacy`` — run an audit and return the rolled-up scorecard
  (``audit:run``; the auditor/admin roles). The recorded-data half is tenant-scoped to the
  caller's authorized workspace, exactly like the rest of the BFF.
* ``GET /v1/audit/privacy`` — the posture trend (registered reports newest-first,
  ``audit:read``).
* ``GET /v1/audit/privacy/{report_id}`` — one report by id (``audit:read``; 404 unknown).

The report is its own ``privacy_audit_report`` :class:`~himmy.entities.records.EntityRecord`,
so ``GET /v1/audit/bundle`` (widened in ``api/routers/audit.py``) now genuinely covers it; a
per-report signed bundle is also available via ``?bundle=1`` on the run, mirroring the
503-when-no-signing-key pattern of the security-audit bundle route.

Offline-first: the router only does work when ``container.privacy_audit`` is wired (it always
is in :class:`~himmy.api.deps.ApiContainer`, but the service is inert over an empty store), so
the zero-config surface is a clean recorded-only scan that registers a passing report.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.auth import require_permission, resolve_workspace, scoped_read
from himmy.entities.integrity import AuditBundle
from himmy.services.evaluation.privacy import PrivacyAuditReport

router = APIRouter(
    prefix="/v1/audit/privacy", tags=["audit"], dependencies=[Depends(scoped_read)]
)

_AUDIT_READ = [Depends(require_permission("audit", "read"))]
_AUDIT_RUN = [Depends(require_permission("audit", "run"))]


class PrivacyAuditRunResponse(BaseModel):
    """The result of a privacy-audit run: the report plus an optional signed bundle.

    ``bundle`` is populated only when the caller asked for one (``?bundle=1``) AND a
    signing key is configured; otherwise it is ``null`` and the report alone is returned.
    """

    report: PrivacyAuditReport
    bundle: AuditBundle | None = None


def _service(request: Request) -> Any:
    """Return the wired :class:`PrivacyAuditService`, or 404 when it is absent.

    The service is always wired by :class:`~himmy.api.deps.ApiContainer`, so this is a
    defensive guard for a hand-built container that omitted it (mirrors how the consent
    router degrades when governance is off) rather than an offline gate.
    """
    service = getattr(request.app.state.container, "privacy_audit", None)
    if service is None:
        raise HTTPException(
            status_code=404, detail="privacy audit service is not available"
        )
    return service


@router.post("", response_model=PrivacyAuditRunResponse, dependencies=_AUDIT_RUN)
async def run_privacy_audit(
    request: Request,
    workspace_id: str | None = None,
    lookback: float | None = Query(
        None, description="recorded-data scan window in seconds (None ⇒ unbounded)"
    ),
    bundle: bool = Query(
        False, description="also export a signed bundle over the registered report"
    ),
) -> PrivacyAuditRunResponse:
    """Run a privacy & compliance audit (recorded-data scan, tenant-scoped) (auditor/admin).

    The recorded-data half scans only the caller's authorized workspace; the report is
    registered as its own tamper-evident ``privacy_audit_report`` record. When
    ``bundle=true`` and a signing key is configured the response also carries a signed
    bundle over that report; with no key configured it is a **503** (mirroring
    ``GET /v1/audit/bundle``).
    """
    workspace_id = resolve_workspace(request, workspace_id)
    service = _service(request)
    report = await service.run(lookback_seconds=lookback, workspace_id=workspace_id)
    signed: AuditBundle | None = None
    if bundle:
        from himmy.core.errors import HimmyError
        from himmy.licensing import (
            FEATURE_AUDIT_EXPORT,
            EnterpriseFeatureError,
            require_entitlement,
        )

        # Signed audit-bundle EXPORT is an Enterprise Edition feature (the free scan and
        # its report above are unaffected). Community gets a 402 with the upgrade message.
        try:
            require_entitlement(FEATURE_AUDIT_EXPORT)
        except EnterpriseFeatureError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        try:
            signed = service.export_signed_bundle(report)
        except HimmyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PrivacyAuditRunResponse(report=report, bundle=signed)


@router.get("", response_model=list[PrivacyAuditReport], dependencies=_AUDIT_READ)
async def list_privacy_audits(
    request: Request,
    workspace_id: str | None = None,
    limit: int = Query(20, ge=1, le=200),
) -> list[PrivacyAuditReport]:
    """Return the posture trend: registered audit reports newest-first (auditor/admin).

    ``workspace_id`` is resolved against the principal (tenant scoping) and the resolved
    workspace is pushed into ``trend()`` so a tenant-bound caller sees ONLY its own
    workspace's reports (plus global/ops reports with no ``workspace_id``) — never another
    tenant's report, its stamped workspace or the affected ``subject_refs`` (closes the
    cross-tenant IDOR). An all-tenants / offline principal resolves to ``None`` and keeps
    the full posture trend, byte-for-byte unchanged.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    return list(_service(request).trend(limit=limit, workspace_id=workspace_id))


@router.get("/{report_id}", response_model=PrivacyAuditReport, dependencies=_AUDIT_READ)
async def get_privacy_audit(
    request: Request,
    report_id: str,
    workspace_id: str | None = None,
) -> PrivacyAuditReport:
    """Return one registered audit report by id, or 404 (auditor/admin).

    The lookup is tenant-scoped: ``trend()`` is filtered to the caller's resolved
    workspace, so a foreign ``report_id`` (another tenant's report) is a clean **404** with
    no existence leak — never the foreign report's workspace or ``subject_refs``. An
    all-tenants / offline principal resolves to ``None`` and may read any report by id.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    reports: list[PrivacyAuditReport] = list(
        _service(request).trend(limit=10_000, workspace_id=workspace_id)
    )
    for report in reports:
        if report.report_id == report_id:
            return report
    raise HTTPException(status_code=404, detail=f"no audit report {report_id!r}")


__all__ = ["router"]
