"""API kernel: the /v1/consent router (record + read consent, WS4.6 A3).

Surfaces the :class:`~himmy.services.governance.consent_ledger.ConsentLedger` over HTTP so
an operator (or a data subject acting for itself) can grant/deny/withdraw consent and read
the current decision, latest record, or full version history. The router exists **only in a
governed deployment** (``HIMMY_CONSENT`` on, which is when ``app.state.consent_ledger`` is
wired); on the offline path every endpoint returns 404-style "consent governance is not
enabled" so the zero-config surface is unchanged.

Authorization mirrors the rest of the BFF: writes need ``consent:write`` and reads need
``consent:read``, both tenant-scoped. The destructive ``/withdraw`` (crypto-shred +
erasure tombstone) follows the ``:write``-for-mutation convention — it requires
``consent:write`` for any caller EXCEPT a ``data_subject`` erasing its OWN ``subject_id``
(its inherent right to erasure), so a read-only ``auditor`` cannot erase a subject. A
``data_subject`` principal is otherwise **self-scoped** — it may only read or withdraw its
OWN ``subject_id`` (``principal.subject``), never another's. Every state change emits a
tamper-evident ``security_event`` via :func:`audit_event`.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from himmy.api.auth import get_principal, require_permission, resolve_workspace
from himmy.api.security_audit import audit_event
from himmy.services.governance.consent import (
    ConsentRecord,
    ConsentState,
    Purpose,
)

router = APIRouter(prefix="/v1/consent", tags=["consent"])

_READ = [Depends(require_permission("consent", "read"))]
_WRITE = [Depends(require_permission("consent", "write"))]

#: The role that may only act on its own subject_id.
_DATA_SUBJECT_ROLE = "data_subject"


class ConsentWriteRequest(BaseModel):
    """A grant/deny payload for one ``(subject_id, purpose)``."""

    subject_id: str
    purpose: Purpose
    workspace_id: str | None = None
    basis: str | None = None
    expires_at: str | None = None


class WithdrawRequest(BaseModel):
    """A withdrawal payload; ``purpose`` omitted withdraws every recorded purpose."""

    subject_id: str
    purpose: Purpose | None = None
    reason: str = ""
    workspace_id: str | None = None


class DecisionResponse(BaseModel):
    """The PDP verdict for a ``(subject_id, purpose)``."""

    subject_id: str
    purpose: Purpose
    state: ConsentState
    effect: str
    allowed: bool
    reason: str


def _ledger(request: Request) -> Any:
    """Return the wired :class:`ConsentLedger`, or 404 when governance is off."""
    ledger = getattr(request.app.state, "consent_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=404,
            detail="consent governance is not enabled (set HIMMY_CONSENT=on)",
        )
    return ledger


def _enforce_self_scope(request: Request, subject_id: str) -> None:
    """A ``data_subject`` principal may only touch its own ``subject_id`` (else 403).

    Other roles (operator/admin/auditor) are unrestricted on the subject axis here; the
    tenant axis is enforced separately via :func:`resolve_workspace`. Offline (no auth) the
    principal is ANONYMOUS without the role, so this is a no-op — zero-config unchanged.
    """
    principal = get_principal(request)
    if not principal.has_role(_DATA_SUBJECT_ROLE):
        return
    # A self-scoped principal that ALSO holds a broader role keeps the broader role's reach.
    privileged = {"operator", "admin", "auditor"}
    if privileged & principal.roles:
        return
    if subject_id != principal.subject:
        audit_event(
            request,
            event_type="authz_denied",
            outcome="deny",
            resource="consent",
            action="self_scope",
            detail="data_subject may only act on its own subject_id",
        )
        raise HTTPException(
            status_code=403, detail="data_subject may only act on its own subject_id"
        )


@router.post("/grant", response_model=ConsentRecord, dependencies=_WRITE)
async def grant_consent(body: ConsentWriteRequest, request: Request) -> ConsentRecord:
    """Record a GRANTED consent for ``(subject_id, purpose)`` (operator/admin)."""
    _enforce_self_scope(request, body.subject_id)
    workspace_id = resolve_workspace(request, body.workspace_id)
    ledger = _ledger(request)
    record = ledger.grant(
        body.subject_id,
        body.purpose,
        workspace_id=workspace_id,
        actor=get_principal(request).subject,
        source="api",
        basis=body.basis,
        expires_at=body.expires_at,
    )
    audit_event(
        request,
        event_type="consent_granted",
        outcome="allow",
        resource="consent",
        action="grant",
        workspace_id=workspace_id,
        detail=f"{body.subject_id}|{body.purpose.value}",
    )
    return ConsentRecord.model_validate(record.payload)


@router.post("/deny", response_model=ConsentRecord, dependencies=_WRITE)
async def deny_consent(body: ConsentWriteRequest, request: Request) -> ConsentRecord:
    """Record a DENIED consent for ``(subject_id, purpose)`` (operator/admin)."""
    _enforce_self_scope(request, body.subject_id)
    workspace_id = resolve_workspace(request, body.workspace_id)
    ledger = _ledger(request)
    record = ledger.deny(
        body.subject_id,
        body.purpose,
        workspace_id=workspace_id,
        actor=get_principal(request).subject,
        source="api",
        basis=body.basis,
        expires_at=body.expires_at,
    )
    audit_event(
        request,
        event_type="consent_denied",
        outcome="deny",
        resource="consent",
        action="deny",
        workspace_id=workspace_id,
        detail=f"{body.subject_id}|{body.purpose.value}",
    )
    return ConsentRecord.model_validate(record.payload)


async def _authorize_withdraw(request: Request, subject_id: str) -> None:
    """Authorize a withdrawal: a self-erasing data_subject, else ``consent:write``.

    Withdraw is **destructive** — it crypto-shreds the subject via
    ``RetentionService.erase_subject`` and writes a signed erasure tombstone — so it
    follows the BFF's ``:write``-for-mutation convention rather than sitting behind
    ``consent:read``. Two principals may withdraw:

    * a ``data_subject`` exercising its OWN right to erasure (self-scoped to its own
      ``subject_id``, holding only ``consent:read``) — the self-scope guard already
      pinned it to its own subject; or
    * any caller holding ``consent:write`` (operator/admin), unrestricted on the
      subject axis the same way grant/deny are.

    A read-only ``auditor`` (``consent:read`` but no ``consent:write`` and not the
    subject itself) is therefore denied — closing the privilege hole where a read-only
    role could irreversibly erase any subject in its tenant.

    Offline (no authenticator) :func:`require_permission` is a no-op, so the zero-config
    path is unchanged.
    """
    principal = get_principal(request)
    privileged = {"operator", "admin"}
    is_self_subject = (
        principal.has_role(_DATA_SUBJECT_ROLE)
        and not (privileged & principal.roles)
        and subject_id == principal.subject
    )
    if is_self_subject:
        return  # a data_subject erasing its OWN data — its inherent right
    # Everyone else must hold consent:write for this state change.
    await require_permission("consent", "write")(request)


@router.post("/withdraw", response_model=list[ConsentRecord], dependencies=_READ)
async def withdraw_consent(
    body: WithdrawRequest, request: Request
) -> list[ConsentRecord]:
    """Withdraw consent and crypto-shred the subject (self-erasure or ``consent:write``).

    The route carries ``consent:read`` so a self-scoped ``data_subject`` (which holds
    read, not write) can exercise its own right to erasure. But this is a **destructive
    mutation**, so :func:`_authorize_withdraw` additionally requires that any caller NOT
    acting as its own ``data_subject`` hold ``consent:write`` — a read-only ``auditor``
    cannot erase another subject. The self-scope check still forbids touching another
    subject's consent.
    """
    _enforce_self_scope(request, body.subject_id)
    await _authorize_withdraw(request, body.subject_id)
    workspace_id = resolve_workspace(request, body.workspace_id)
    ledger = _ledger(request)
    records = ledger.withdraw(
        body.subject_id,
        body.purpose,
        reason=body.reason,
        actor=get_principal(request).subject,
        source="api",
    )
    audit_event(
        request,
        event_type="consent_withdrawn",
        outcome="allow",
        resource="consent",
        action="withdraw",
        workspace_id=workspace_id,
        detail=f"{body.subject_id}|{body.purpose.value if body.purpose else '*'}",
    )
    return [ConsentRecord.model_validate(r.payload) for r in records]


@router.get("/decision", response_model=DecisionResponse, dependencies=_READ)
async def get_decision(
    request: Request,
    subject_id: str = Query(...),
    purpose: Purpose = Query(...),
) -> DecisionResponse:
    """Resolve the PDP decision for ``(subject_id, purpose)`` (self-scoped allowed)."""
    _enforce_self_scope(request, subject_id)
    resolve_workspace(request, None)
    ledger = _ledger(request)
    decision = ledger.decision(subject_id, purpose)
    return DecisionResponse(
        subject_id=subject_id,
        purpose=purpose,
        state=ledger.state(subject_id, purpose),
        effect=decision.effect.value,
        allowed=decision.allowed,
        reason=decision.reason,
    )


@router.get("/latest", response_model=ConsentRecord | None, dependencies=_READ)
async def get_latest(
    request: Request,
    subject_id: str = Query(...),
    purpose: Purpose = Query(...),
) -> ConsentRecord | None:
    """Return the latest :class:`ConsentRecord` for the pair, or ``null``."""
    _enforce_self_scope(request, subject_id)
    resolve_workspace(request, None)
    return cast("ConsentRecord | None", _ledger(request).latest(subject_id, purpose))


@router.get("/history", response_model=list[ConsentRecord], dependencies=_READ)
async def get_history(
    request: Request,
    subject_id: str = Query(...),
    purpose: Purpose = Query(...),
) -> list[ConsentRecord]:
    """Return the full append-only version chain for ``(subject_id, purpose)``."""
    _enforce_self_scope(request, subject_id)
    resolve_workspace(request, None)
    return cast("list[ConsentRecord]", _ledger(request).history(subject_id, purpose))


__all__ = ["router"]
