"""API kernel: the /v1/context router (field upsert + snapshot build/read).

Field listing and snapshot reads are tenant-scoped on ``workspace_id`` (AAEO-4);
routes declare typed responses + 404 shapes (AAEO-9).
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from himmy.api.auth import (
    authorize_object,
    enforce_subject_write,
    require_permission,
    require_workspace,
    resolve_workspace,
)
from himmy.api.models import NOT_FOUND_RESPONSE, UpsertResult
from himmy.services.context.models import (
    ContextBuildSpec,
    ContextField,
    ContextSnapshot,
)

router = APIRouter(prefix="/v1/context", tags=["context"])

_READ = [Depends(require_permission("context", "read"))]
_WRITE = [Depends(require_permission("context", "write"))]


class UpsertFieldsRequest(BaseModel):
    """Bulk context-field upsert payload for one subject."""

    workspace_id: str
    subject_id: str
    fields: list[ContextField]


class BuildSnapshotRequest(BaseModel):
    """Request to build a context snapshot from a declarative build spec.

    ``workspace_id`` (optional) is stamped into the snapshot metadata so the
    snapshot read path can enforce tenancy (AAEO-4).
    """

    subject_id: str
    task_id: str | None = None
    build_spec: ContextBuildSpec
    workspace_id: str | None = None
    metadata: dict[str, Any] | None = None


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


@router.post("/fields:upsert", response_model=UpsertResult, dependencies=_WRITE)
async def upsert_fields(body: UpsertFieldsRequest, request: Request) -> UpsertResult:
    """Bulk-upsert context fields for a subject; returns the saved count.

    Object-level (BOLA, WS-bola): a ``subject_scoped`` principal may only write under its
    OWN ``subject_id`` (a foreign subject is 403 via :func:`enforce_subject_write`), so a
    subject-scoped tenant can never overwrite/poison another data subject's PII fields in a
    shared workspace. The gate is a no-op for the offline / ``all_tenants`` / ``tenant_admin``
    path, so the zero-config write is byte-unchanged.
    """
    workspace_id = require_workspace(request, body.workspace_id)
    enforce_subject_write(request, body.subject_id)
    saved = await _container(request).context_app.upsert_fields(
        workspace_id, body.subject_id, body.fields
    )
    return UpsertResult(upserted=len(saved), subject_id=body.subject_id)


@router.get("/fields", response_model=list[ContextField], dependencies=_READ)
async def list_fields(
    subject_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> list[ContextField]:
    """List the stored context fields for a subject (workspace-scoped, AAEO-4).

    Object-level (BOLA, WS-bola): a ``subject_scoped`` principal reading ANOTHER data
    subject's PII fields within its own tenant gets an empty list (the same not-found-shaped
    response a foreign subject yields) — :func:`authorize_object` is a no-op for offline /
    ``all_tenants`` / ``tenant_admin`` callers, so the zero-config read is byte-unchanged.
    """
    workspace_id = resolve_workspace(request, workspace_id)
    if not authorize_object(request, subject_id):
        return []
    return cast(
        list[ContextField],
        await _container(request).context_app.list_fields(
            subject_id, workspace_id=workspace_id
        ),
    )


@router.post("/snapshots:build", response_model=ContextSnapshot, dependencies=_WRITE)
async def build_snapshot(
    body: BuildSnapshotRequest, request: Request
) -> ContextSnapshot:
    """Build, persist, and return a context snapshot (stamping workspace scope).

    Object-level (BOLA, WS-bola): a ``subject_scoped`` principal may only build a snapshot
    for its OWN ``subject_id`` (403 otherwise via :func:`enforce_subject_write`) — a no-op
    for offline / ``all_tenants`` / ``tenant_admin`` callers.
    """
    workspace_id = resolve_workspace(request, body.workspace_id)
    enforce_subject_write(request, body.subject_id)
    metadata = dict(body.metadata or {})
    if workspace_id is not None:
        metadata.setdefault("workspace_id", workspace_id)
    return cast(
        ContextSnapshot,
        await _container(request).context_app.build_snapshot(
            subject_id=body.subject_id,
            task_id=body.task_id,
            build_spec=body.build_spec,
            metadata=metadata or None,
        ),
    )


@router.get(
    "/snapshots/{snapshot_id}",
    response_model=ContextSnapshot,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_snapshot(
    snapshot_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> ContextSnapshot:
    """Load one context snapshot by id (404 when unknown/out-of-workspace, AAEO-4).

    Object-level (BOLA, WS-bola): a ``subject_scoped`` principal resolving a snapshot built
    for ANOTHER data subject within its own tenant gets a clean 404 — the workspace scope is
    checked first, then the snapshot's ``subject_id`` is gated via :func:`authorize_object`.
    A no-op for offline / ``all_tenants`` / ``tenant_admin`` callers (byte-unchanged).
    """
    workspace_id = resolve_workspace(request, workspace_id)
    snapshot = await _container(request).context_app.get_snapshot(
        snapshot_id, workspace_id=workspace_id
    )
    if snapshot is None or not authorize_object(
        request, getattr(snapshot, "subject_id", None)
    ):
        raise HTTPException(status_code=404, detail="snapshot not found")
    return cast(ContextSnapshot, snapshot)


__all__ = ["router"]
