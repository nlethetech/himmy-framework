"""API kernel: Studio privacy router — the governance ledger behind the Privacy screen.

Mounted under ``/api/studio/privacy`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). The endpoints are GUI-shaped reads
and actions over the machinery WS4.6/WS4.7 already shipped:

* ``GET  /subjects``      — data subjects with consent/record counts + erased flag.
* ``GET  /consents``      — consent version chains (purpose · state · effect · time).
* ``POST /erase``         — right-to-erasure: withdraw + crypto-shred (typed confirm).
* ``POST /audit/export``  — a signed, downloadable audit bundle (Ed25519 or HMAC).
* ``POST /audit/verify``  — verify an uploaded bundle against the live spine.

Offline-first: with ``HIMMY_CONSENT`` off the reads answer ``governed: false`` with
empty ledgers (the screen renders its empty state) and the destructive ``/erase``
is 404-inert, mirroring :mod:`himmy.api.routers.consent`. Audit export/verify only
need a signing key (``HIMMY_AUDIT_PRIVATE_KEY`` / ``HIMMY_AUDIT_SECRET``) — without
one they answer a clear 503, never an import error.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from himmy.api.auth import get_principal
from himmy.api.routers.studio_common import build_studio_router
from himmy.api.security_audit import audit_event
from himmy.core.ids import utc_now_iso
from himmy.entities.integrity import AuditBundle
from himmy.services.governance.consent import (
    CONSENT_KIND,
    ConsentState,
    Effect,
    Purpose,
)
from himmy.services.governance.retention import ERASURE_KIND

router = build_studio_router("privacy", tag="studio-privacy")

#: Subject-bearing spine kinds counted per subject — mirrors the consent-gated set
#: wired in :mod:`himmy.api.deps` (``_GATED_SPINE_KINDS``).
_SUBJECT_KINDS: tuple[str, ...] = (
    "run_event",
    "message",
    "chat_thread",
    "context_snapshot",
    "recommendation",
)

#: Hard caps so a huge spine can never blow up a GUI response.
_MAX_SUBJECTS = 500
_MAX_CONSENT_ROWS = 1000
_MAX_BUNDLE_ENTRIES = 250_000


def _bundle_kinds() -> tuple[str, ...]:
    """The evidence kinds a Studio audit bundle commits to.

    The ``/v1/audit/bundle`` canon (security events + privacy audit reports) plus the
    consent ledger and erasure tombstones, so the one signed artifact covers the full
    governance story the Privacy screen tells.
    """
    from himmy.services.audit.log import SECURITY_EVENT_KIND
    from himmy.services.evaluation.privacy import PRIVACY_AUDIT_REPORT_KIND

    return (SECURITY_EVENT_KIND, PRIVACY_AUDIT_REPORT_KIND, CONSENT_KIND, ERASURE_KIND)


# ------------------------------------------------------------------ shared helpers


def _registry(request: Request) -> Any:
    """The entity registry off the wired container (reads delegate when gated)."""
    return request.app.state.container.entity_registry


def _ledger(request: Request) -> Any:
    """Return the wired :class:`ConsentLedger`, or 404 when governance is off."""
    ledger = getattr(request.app.state, "consent_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=404,
            detail="consent governance is not enabled (set HIMMY_CONSENT=on)",
        )
    return ledger


def _subject_of(record: Any) -> str | None:
    """A spine record's data subject (metadata first, payload fallback)."""
    subject = record.metadata.get("subject_id") or record.payload.get("subject_id")
    return str(subject) if subject else None


def _tombstones_by_subject(registry: Any) -> dict[str, list[Any]]:
    """Erasure tombstones grouped by subject id."""
    out: dict[str, list[Any]] = {}
    for record in registry.list_by_kind(ERASURE_KIND):
        subject = _subject_of(record)
        if subject:
            out.setdefault(subject, []).append(record)
    return out


def _data_record_counts(registry: Any) -> dict[str, int]:
    """How many subject-bearing spine records each subject owns."""
    counts: dict[str, int] = {}
    for kind in _SUBJECT_KINDS:
        for record in registry.list_by_kind(kind):
            subject = _subject_of(record)
            if subject:
                counts[subject] = counts.get(subject, 0) + 1
    return counts


# ----------------------------------------------------------------------- subjects


class SubjectInfo(BaseModel):
    """One data subject's footprint on the spine."""

    subject_id: str
    consent_records: int  # consent versions across every purpose
    purposes: int  # distinct purposes with at least one record
    data_records: int  # subject-bearing spine records (run_event/message/…)
    erased: bool
    last_activity: str | None = None


class SubjectsResponse(BaseModel):
    """The subjects ledger + whether consent governance is on at all."""

    governed: bool
    subjects: list[SubjectInfo]


@router.get("/subjects", response_model=SubjectsResponse)
async def list_subjects(request: Request) -> SubjectsResponse:
    """Every known data subject with consent/record counts and the erased flag."""
    governed = getattr(request.app.state, "consent_ledger", None) is not None
    registry = _registry(request)

    consent_count: dict[str, int] = {}
    purposes: dict[str, set[str]] = {}
    last_seen: dict[str, str] = {}
    for record in registry.list_by_kind(CONSENT_KIND):
        subject = _subject_of(record)
        if not subject:
            continue
        consent_count[subject] = consent_count.get(subject, 0) + 1
        purpose = str(record.payload.get("purpose") or "")
        if purpose:
            purposes.setdefault(subject, set()).add(purpose)
        recorded = str(record.payload.get("recorded_at") or "")
        if recorded and recorded > last_seen.get(subject, ""):
            last_seen[subject] = recorded

    tombstones = _tombstones_by_subject(registry)
    for subject, stones in tombstones.items():
        for stone in stones:
            erased_at = str(stone.payload.get("erased_at") or "")
            if erased_at and erased_at > last_seen.get(subject, ""):
                last_seen[subject] = erased_at

    data_counts = _data_record_counts(registry)
    all_subjects = sorted(set(consent_count) | set(tombstones) | set(data_counts))[
        :_MAX_SUBJECTS
    ]
    return SubjectsResponse(
        governed=governed,
        subjects=[
            SubjectInfo(
                subject_id=s,
                consent_records=consent_count.get(s, 0),
                purposes=len(purposes.get(s, set())),
                data_records=data_counts.get(s, 0),
                erased=s in tombstones,
                last_activity=last_seen.get(s) or None,
            )
            for s in all_subjects
        ],
    )


# ----------------------------------------------------------------------- consents


class ConsentEntry(BaseModel):
    """One version in a subject's consent chain, with the PDP effect it yields."""

    subject_id: str
    purpose: str
    state: str
    effect: str  # what the policy does with this state (allow/ephemeral/deny)
    version: int  # position in the (subject, purpose) version chain
    actor: str = ""
    source: str = ""
    basis: str | None = None
    expires_at: str | None = None
    recorded_at: str = ""


class ConsentsResponse(BaseModel):
    """Consent ledger rows + whether governance is on."""

    governed: bool
    items: list[ConsentEntry]


@router.get("/consents", response_model=ConsentsResponse)
async def list_consents(
    request: Request,
    subject: str | None = Query(None, max_length=200),
) -> ConsentsResponse:
    """Consent records (optionally for one subject), newest first, version-chained."""
    governed = getattr(request.app.state, "consent_ledger", None) is not None
    registry = _registry(request)
    policy = getattr(request.app.state, "consent_policy", None)

    items: list[ConsentEntry] = []
    for record in registry.list_by_kind(CONSENT_KIND):
        sid = _subject_of(record)
        if not sid or (subject and sid != subject):
            continue
        try:
            purpose = Purpose(str(record.payload.get("purpose")))
            state = ConsentState(str(record.payload.get("state")))
        except ValueError:
            continue  # malformed/foreign record — never break the ledger view
        effect = (
            policy.decide(sid, purpose, state).effect
            if policy is not None
            else Effect.ALLOW
        )
        items.append(
            ConsentEntry(
                subject_id=sid,
                purpose=purpose.value,
                state=state.value,
                effect=effect.value,
                version=record.version,
                actor=str(record.payload.get("actor") or ""),
                source=str(record.payload.get("source") or ""),
                basis=record.payload.get("basis"),
                expires_at=record.payload.get("expires_at"),
                recorded_at=str(record.payload.get("recorded_at") or ""),
            )
        )
    items.sort(key=lambda e: (e.recorded_at, e.subject_id, e.purpose, e.version))
    items.reverse()
    return ConsentsResponse(governed=governed, items=items[:_MAX_CONSENT_ROWS])


# ------------------------------------------------------------------------- erase


class EraseRequest(BaseModel):
    """Right-to-erasure: ``confirm`` must re-type the subject id exactly."""

    subject_id: str = Field(..., min_length=1, max_length=200)
    confirm: str = Field(..., min_length=1, max_length=200)


class EraseResponse(BaseModel):
    """What the erasure destroyed (the proof the UI shows)."""

    subject_id: str
    consents_withdrawn: list[str]  # purposes flipped to WITHDRAWN
    data_records: int  # subject-bearing spine records now unrecoverable
    crypto_shredded: bool  # whether a per-subject key existed and was destroyed
    tombstone_id: str | None = None
    erased_at: str | None = None


@router.post("/erase", response_model=EraseResponse)
async def erase_subject(body: EraseRequest, request: Request) -> EraseResponse:
    """Withdraw every consent and crypto-shred the subject (typed confirmation).

    The real WS4.6 path — :meth:`ConsentLedger.withdraw` appends WITHDRAWN versions
    and routes through :meth:`RetentionService.erase_subject` to destroy the
    subject's key and register an immutable erasure tombstone.
    """
    ledger = _ledger(request)
    if body.confirm != body.subject_id:
        raise HTTPException(
            status_code=400,
            detail="typed confirmation does not match the subject id",
        )
    registry = _registry(request)
    subject = body.subject_id

    has_consents = any(
        _subject_of(r) == subject for r in registry.list_by_kind(CONSENT_KIND)
    )
    data_records = _data_record_counts(registry).get(subject, 0)
    already_erased = subject in _tombstones_by_subject(registry)
    if not has_consents and data_records == 0 and not already_erased:
        raise HTTPException(
            status_code=404, detail=f"no data recorded for subject {subject!r}"
        )

    withdrawn = ledger.withdraw(
        subject,
        reason="studio erasure",
        actor=get_principal(request).subject,
        source="studio",
    )

    # The freshest tombstone is the proof of THIS erasure.
    stones = _tombstones_by_subject(registry).get(subject, [])
    latest = max(
        stones, key=lambda s: str(s.payload.get("erased_at") or ""), default=None
    )
    audit_event(
        request,
        event_type="consent_withdrawn",
        outcome="allow",
        resource="consent",
        action="erase",
        detail=f"{subject}|*",
    )
    return EraseResponse(
        subject_id=subject,
        consents_withdrawn=sorted({str(r.payload.get("purpose")) for r in withdrawn}),
        data_records=data_records,
        crypto_shredded=bool(latest.payload.get("crypto_shredded"))
        if latest
        else False,
        tombstone_id=latest.record_id if latest else None,
        erased_at=str(latest.payload.get("erased_at")) if latest else None,
    )


# ------------------------------------------------------------------ audit bundle


class AuditExportResponse(BaseModel):
    """A signed audit bundle plus the context needed to verify it later."""

    exported_at: str
    algorithm: str
    record_count: int
    record_kinds: list[str]
    bundle: AuditBundle


def _evidence_records(registry: Any, kinds: tuple[str, ...]) -> list[Any]:
    """Every spine record of the evidence kinds (the set a bundle commits to)."""
    records: list[Any] = []
    for kind in kinds:
        records.extend(registry.list_by_kind(kind))
    return records


@router.post("/audit/export", response_model=AuditExportResponse)
async def export_audit(request: Request) -> AuditExportResponse:
    """Export a signed, tamper-evident bundle over the governance evidence.

    Signs with Ed25519 when ``HIMMY_AUDIT_PRIVATE_KEY`` (PEM) is configured, else
    HMAC-SHA256 with ``HIMMY_AUDIT_SECRET``; 503 with the fix when neither is set.
    """
    from himmy.config.secrets import get_secret
    from himmy.entities.integrity import (
        export_audit_bundle,
        export_audit_bundle_ed25519,
    )

    kinds = _bundle_kinds()
    records = _evidence_records(_registry(request), kinds)
    private_pem = get_secret("HIMMY_AUDIT_PRIVATE_KEY")
    if private_pem:
        bundle = export_audit_bundle_ed25519(records, [], private_pem=private_pem)
    else:
        secret = get_secret("HIMMY_AUDIT_SECRET")
        if not secret:
            raise HTTPException(
                status_code=503,
                detail="audit signing key not configured "
                "(set HIMMY_AUDIT_PRIVATE_KEY or HIMMY_AUDIT_SECRET)",
            )
        bundle = export_audit_bundle(records, [], secret=secret)
    audit_event(
        request,
        event_type="audit_bundle_exported",
        outcome="allow",
        resource="audit",
        action="export",
        detail=f"{len(records)} records|{bundle.algorithm}",
    )
    return AuditExportResponse(
        exported_at=utc_now_iso(),
        algorithm=bundle.algorithm,
        record_count=len(records),
        record_kinds=list(kinds),
        bundle=bundle,
    )


class VerifyCheck(BaseModel):
    """One verification verdict the UI renders as a ledger row."""

    name: str
    ok: bool
    detail: str = ""


class AuditVerifyRequest(BaseModel):
    """An uploaded bundle: either the export envelope or a raw :class:`AuditBundle`."""

    # Envelope form (what /audit/export produced).
    bundle: dict[str, Any] | None = None
    record_kinds: list[str] | None = Field(None, max_length=16)
    # Raw-bundle form (a bare AuditBundle JSON pasted/uploaded directly).
    bundle_version: int | None = None
    records: dict[str, str] | None = None
    links: dict[str, str] | None = None
    merkle_root: str | None = Field(None, max_length=128)
    signature: str | None = Field(None, max_length=8192)
    algorithm: str | None = Field(None, max_length=40)


class AuditVerifyResponse(BaseModel):
    """The overall verdict + the per-check breakdown."""

    ok: bool
    algorithm: str
    checks: list[VerifyCheck]


def _parse_uploaded_bundle(body: AuditVerifyRequest) -> AuditBundle:
    """Accept the export envelope or a raw bundle; 400 on anything else."""
    raw: dict[str, Any]
    if body.bundle is not None:
        raw = body.bundle
    else:
        raw = {
            k: v
            for k, v in {
                "bundle_version": body.bundle_version,
                "records": body.records,
                "links": body.links,
                "merkle_root": body.merkle_root,
                "signature": body.signature,
                "algorithm": body.algorithm,
            }.items()
            if v is not None
        }
    if not raw.get("signature") or not raw.get("merkle_root"):
        raise HTTPException(
            status_code=400,
            detail="not an audit bundle (missing signature/merkle_root)",
        )
    try:
        bundle = AuditBundle.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="not a valid audit bundle") from exc
    if len(bundle.records) + len(bundle.links) > _MAX_BUNDLE_ENTRIES:
        raise HTTPException(status_code=413, detail="bundle too large to verify")
    return bundle


def _public_pem_from_private(private_pem: str) -> str:
    """Derive the Ed25519 public key PEM from the configured private key."""
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(private_pem.encode("ascii"), password=None)
    return (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def _ids_detail(ids: list[str], verb: str) -> str:
    """A bounded human detail line for a list of diverging record ids."""
    head = ", ".join(ids[:3])
    more = f" (+{len(ids) - 3} more)" if len(ids) > 3 else ""
    return f"{len(ids)} {verb}: {head}{more}"


@router.post("/audit/verify", response_model=AuditVerifyResponse)
async def verify_audit(
    body: AuditVerifyRequest, request: Request
) -> AuditVerifyResponse:
    """Verify an uploaded bundle against the live spine, check by check."""
    from himmy.config.secrets import get_secret
    from himmy.entities.integrity import (
        verify_audit_bundle,
        verify_audit_bundle_ed25519,
    )

    bundle = _parse_uploaded_bundle(body)
    default_kinds = _bundle_kinds()
    kinds = (
        tuple(
            k for k in (body.record_kinds or list(default_kinds)) if k in default_kinds
        )
        or default_kinds
    )
    records = _evidence_records(_registry(request), kinds)

    if bundle.algorithm.lower().startswith("ed25519"):
        private_pem = get_secret("HIMMY_AUDIT_PRIVATE_KEY")
        if not private_pem:
            raise HTTPException(
                status_code=503,
                detail="cannot verify an Ed25519 bundle: "
                "HIMMY_AUDIT_PRIVATE_KEY is not configured",
            )
        try:
            public_pem = _public_pem_from_private(private_pem)
        except Exception as exc:  # bad PEM / missing extra — a clear 503, never a 500
            raise HTTPException(
                status_code=503,
                detail=f"audit key unusable for verification: {exc}",
            ) from exc
        result = verify_audit_bundle_ed25519(bundle, records, [], public_pem=public_pem)
    else:
        secret = get_secret("HIMMY_AUDIT_SECRET")
        if not secret:
            raise HTTPException(
                status_code=503,
                detail="cannot verify an HMAC bundle: "
                "HIMMY_AUDIT_SECRET is not configured",
            )
        result = verify_audit_bundle(bundle, records, [], secret=secret)

    checks = [
        VerifyCheck(
            name="signature",
            ok=result.signature_valid,
            detail=f"{bundle.algorithm} over merkle root {bundle.merkle_root[:12]}…"
            if bundle.merkle_root
            else bundle.algorithm,
        ),
        VerifyCheck(
            name="records intact",
            ok=not result.tampered_record_ids,
            detail=_ids_detail(result.tampered_record_ids, "tampered")
            if result.tampered_record_ids
            else f"{len(bundle.records)} record hashes match",
        ),
        VerifyCheck(
            name="records present",
            ok=not result.missing_record_ids,
            detail=_ids_detail(result.missing_record_ids, "missing from the spine")
            if result.missing_record_ids
            else "every bundled record is still on the spine",
        ),
        VerifyCheck(
            name="no records added",
            ok=not result.added_record_ids,
            detail=f"{len(result.added_record_ids)} records registered since export "
            "(expected on a live ledger)"
            if result.added_record_ids
            else "live spine matches the bundle exactly",
        ),
    ]
    if bundle.links:
        checks.append(
            VerifyCheck(
                name="links intact",
                ok=not (result.tampered_link_ids or result.missing_link_ids),
                detail=_ids_detail(
                    result.tampered_link_ids + result.missing_link_ids, "diverged"
                )
                if (result.tampered_link_ids or result.missing_link_ids)
                else f"{len(bundle.links)} link hashes match",
            )
        )
    return AuditVerifyResponse(ok=result.ok, algorithm=bundle.algorithm, checks=checks)


__all__ = ["router"]
