"""API kernel: Studio memory router.

Mounted under ``/api/studio/memory`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). The browse/add/forget/recall
surface (list with ``?subject=`` filtering, ``/subjects``, ``DELETE /{id}``)
lives on the main Studio router in :mod:`himmy.api.routers.studio`; this
router carries what that surface is missing — editing a memory in place.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field

from himmy.api.auth.context import authorize_object
from himmy.api.auth.scope_marker import subject_write
from himmy.api.routers.studio_common import build_studio_router, studio_permission

router = build_studio_router("memory", tag="studio-memory")

#: Editing a memory in place is a mutation gated by ``studio.memory:write`` (admin-only
#: by default), additively on top of the router's ``studio.memory:read`` baseline so a
#: read-only role can recall memories but never rewrite one.
_memory_write = Depends(studio_permission("studio.memory", "write"))


class MemoryEditRequest(BaseModel):
    """New text for an existing memory (same bounds as the add endpoint)."""

    text: str = Field(..., min_length=1, max_length=20_000)


@router.patch("/{memory_id}", dependencies=[_memory_write, Depends(subject_write)])
async def memory_edit(
    memory_id: Annotated[str, Path(min_length=1, max_length=200)],
    body: MemoryEditRequest,
    request: Request,
) -> Any:
    """Rewrite a memory's text in place, keeping its identity and history.

    Everything else on the record — ``memory_id``, ``subject_id`` (the
    governance/crypto-shred scope, deliberately immutable here), ``kind``,
    ``tier``, ``stable_key`` and the bi-temporal validity window — is
    preserved. The cached embedding for the record is dropped so the next
    recall re-embeds the new text instead of ranking against the old one.

    TENANT axis: a tenant-bound caller may only edit a memory in its OWN
    ``t:<workspace>:`` namespace on the shared (column-less) process store; a
    record stamped under another tenant's namespace folds to a uniform 404
    (existence never leaked), mirroring the by-id ``memory_forget`` writer.
    Without this gate ``memory_edit`` was the one Studio memory write that
    skipped the tenant-prefix axis its siblings enforce, so a tenant-bound
    ``studio.memory:write`` holder could PATCH-overwrite another tenant's memory.

    SUBJECT axis: the (de-namespaced) owning ``subject_id`` is then checked
    against the principal so a ``subject_scoped`` caller may only edit its OWN
    subject's memory; a cross-subject edit folds to the same 404. The tenant
    prefix is STRIPPED before the subject comparison (mirroring ``memory_forget``)
    so a legitimately-owned namespaced memory remains editable. NO-OP offline /
    ``all_tenants`` (empty prefix → byte-unchanged single-box path).
    """
    from himmy.api import studio_memory
    from himmy.api.routers.studio import _memory_tenant_prefix, _run_subject_scope

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    service = studio_memory.get_memory_service()
    record = service.get(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")
    # TENANT(+SUBJECT) axis: a record outside this principal's ``t:<tenant>[:s:<subject>]:``
    # namespace is a uniform 404 BEFORE any further check, so existence never leaks across
    # tenants/users. When the prefix already carries the within-tenant user (``:s:<subject>:``),
    # this prefix-match IS the BOLA gate and the stripped remainder is the free-form
    # MEMORY_SUBJECT (not a data subject), so the by-id subject check below is skipped.
    prefix = _memory_tenant_prefix(request)
    if prefix and not record.subject_id.startswith(prefix):
        raise HTTPException(status_code=404, detail="memory not found")
    # SUBJECT axis (tenant-only / legacy path only): strip the tenant namespace so the bare
    # subject (the owner the principal can be scoped to) is what ``authorize_object`` compares
    # against. Off the subject-scoped path the prefix above carries no user axis.
    if _run_subject_scope(request) is None:
        bare_subject = (
            record.subject_id[len(prefix) :]
            if prefix and record.subject_id.startswith(prefix)
            else record.subject_id
        )
        if not authorize_object(request, bare_subject):
            raise HTTPException(status_code=404, detail="memory not found")
    updated = record.model_copy(update={"text": text})
    service.store.save(updated)
    # The service caches embeddings per memory_id in-process; a stale vector
    # would make recall rank the old wording. Best-effort cache drop.
    vectors = getattr(service, "_vectors", None)
    if isinstance(vectors, dict):
        vectors.pop(memory_id, None)
    return studio_memory.MemoryItem(
        memory_id=updated.memory_id,
        subject_id=updated.subject_id,
        kind=updated.kind,
        text=updated.text,
        created_at=updated.created_at,
    )


__all__ = ["router"]
