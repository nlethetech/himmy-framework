"""API kernel: Studio memory router.

Mounted under ``/api/studio/memory`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). The browse/add/forget/recall
surface (list with ``?subject=`` filtering, ``/subjects``, ``DELETE /{id}``)
lives on the main Studio router in :mod:`himmy.api.routers.studio`; this
router carries what that surface is missing — editing a memory in place.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Path
from pydantic import BaseModel, Field

from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("memory", tag="studio-memory")


class MemoryEditRequest(BaseModel):
    """New text for an existing memory (same bounds as the add endpoint)."""

    text: str = Field(..., min_length=1, max_length=20_000)


@router.patch("/{memory_id}")
async def memory_edit(
    memory_id: Annotated[str, Path(min_length=1, max_length=200)],
    body: MemoryEditRequest,
) -> Any:
    """Rewrite a memory's text in place, keeping its identity and history.

    Everything else on the record — ``memory_id``, ``subject_id`` (the
    governance/crypto-shred scope, deliberately immutable here), ``kind``,
    ``tier``, ``stable_key`` and the bi-temporal validity window — is
    preserved. The cached embedding for the record is dropped so the next
    recall re-embeds the new text instead of ranking against the old one.
    """
    from himmy.api import studio_memory

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    service = studio_memory.get_memory_service()
    record = service.get(memory_id)
    if record is None:
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
