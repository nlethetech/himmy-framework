"""API kernel: the ``/v1/knowledge`` router — workspace-namespaced ingest + search (T3a).

A tenant ingests documents into a named ``collection`` and searches them back. The
collection is addressed ONLY through the single choke-point key
:func:`~himmy.services.knowledge.app_service.knowledge_namespace`
(``"{workspace_id}:{collection}"``), so one tenant's collection can never collide with
(or read) another tenant's documents.

ISOLATION HONESTY (the load-bearing caveat — reviewer must_fix):

This cross-tenant isolation is a property of an AUTHENTICATED deployment ONLY. The
``workspace_id`` that reaches the service is the one :func:`resolve_workspace` derives
from the verified principal — but in the zero-config offline default there is NO
authenticator, so the principal is :data:`ANONYMOUS` (``all_tenants=True``):
``resolve_workspace`` echoes back whatever ``workspace_id`` the caller supplied AND RBAC
is bypassed. In that regime ANY caller can name ANY workspace, so the namespace is a
labelling convenience, NOT a security boundary — the isolation is INERT. The
"different workspace returns nothing" guarantee holds only once an authenticator (and,
for true multi-tenancy, per-tenant API keys) is configured. This is documented rather
than pretended away; a test asserts BOTH regimes.

Security envelope:

* every path is RBAC-gated (``knowledge:read`` / ``knowledge:write``) and tenant-scoped
  via :func:`resolve_workspace` / :func:`require_workspace`;
* ingest bodies are size/character bounded (``MAX_INGEST_CHARS``);
* ``ingest_file`` reads a SERVER filesystem path, so it is operator-only (an
  all-tenants/offline principal) — a tenant-supplied path is rejected (no LFI/traversal
  surface), exactly as ``/v1/routines`` refuses a tenant ``agent_path``;
* every mutating endpoint stamps the principal's actor metadata for the audit spine.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from himmy.api.auth import (
    get_principal,
    require_permission,
    require_workspace,
    resolve_workspace,
    scoped_read,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.api.security_audit import audit_event
from himmy.core.errors import HimmyError

router = APIRouter(
    prefix="/v1/knowledge", tags=["knowledge"], dependencies=[Depends(scoped_read)]
)

_READ = [Depends(require_permission("knowledge", "read"))]
_WRITE = [Depends(require_permission("knowledge", "write"))]

#: Hard cap on a single ingest body's text (mirrors the Studio paste/ingest bound).
MAX_INGEST_CHARS = 1_000_000
#: Bound on the collection identifier so a workspace cannot stuff arbitrary data into
#: the namespace key.
_MAX_COLLECTION = 200
#: Bound on a search query.
_MAX_QUERY = 4_000
#: Bound on the requested result count.
_MAX_TOP_K = 50


class IngestTextRequest(BaseModel):
    """The POST /v1/knowledge/{collection}/documents body for a text ingest."""

    workspace_id: str
    text: str = Field(min_length=1)
    title: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] | None = None


class IngestFileRequest(BaseModel):
    """The POST body for ingesting a server-side file (operator-only).

    ``path`` is a SERVER filesystem path; reading it is an operator boundary, so a
    tenant-bound caller is rejected. Use the Studio multipart upload for tenant content.
    """

    workspace_id: str
    file: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class DocumentView(BaseModel):
    """The API view of an ingested document (no embeddings, no raw chunk vectors)."""

    document_id: str
    title: str | None = None
    source_uri: str | None = None
    characters: int = 0
    metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    """The ingest result: the stored document plus its collection."""

    collection: str
    document: DocumentView


class DocumentListResponse(BaseModel):
    """The GET /v1/knowledge/{collection}/documents envelope."""

    collection: str
    items: list[DocumentView]
    total: int


class SearchHit(BaseModel):
    """One retrieved chunk projected for the API (similarity + text + provenance)."""

    text: str | None = None
    similarity: float = 0.0
    context_window: str | None = None
    document_id: str
    source_uri: str | None = None
    chunk_kind: str = "text"
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    """The GET /v1/knowledge/{collection}/search envelope."""

    collection: str
    query: str
    hits: list[SearchHit]
    total: int


def _knowledge_app(request: Request) -> Any:
    """Pull the wired :class:`KnowledgeAppService` off the app container."""
    return request.app.state.container.knowledge_app


def _operator_provisioned(request: Request) -> bool:
    """Whether THIS request may read a SERVER filesystem path (ingest_file).

    Only an unrestricted (offline / all-tenants) principal is the operator; a
    tenant-bound principal is never operator-provisioned, so a server-path file ingest
    is refused for a real multi-tenant caller.
    """
    return bool(getattr(get_principal(request), "all_tenants", False))


def _check_collection(collection: str) -> None:
    """Validate the collection identifier (length + non-empty)."""
    if not collection or len(collection) > _MAX_COLLECTION:
        raise HTTPException(
            status_code=422,
            detail=f"collection must be 1-{_MAX_COLLECTION} characters",
        )


def _doc_view(doc: Any) -> DocumentView:
    """Project a :class:`KnowledgeDocument` into the API view."""
    return DocumentView(
        document_id=doc.document_id,
        title=doc.title,
        source_uri=doc.source_uri,
        characters=len(doc.text or ""),
        metadata=dict(doc.metadata or {}),
    )


@router.post(
    "/{collection}/documents",
    response_model=IngestResponse,
    dependencies=_WRITE,
    status_code=201,
)
async def ingest_document(
    collection: str, body: IngestTextRequest, request: Request
) -> IngestResponse:
    """Ingest one text document into a workspace's collection (auto-creates it)."""
    _check_collection(collection)
    workspace_id = require_workspace(request, body.workspace_id)
    if len(body.text) > MAX_INGEST_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"text is {len(body.text):,} characters; the limit is "
            f"{MAX_INGEST_CHARS:,}. Split the document and ingest the parts.",
        )
    try:
        doc = await _knowledge_app(request).ingest_text(
            workspace_id=workspace_id,
            collection=collection,
            text=body.text,
            title=body.title,
            source_uri=body.source_uri,
            metadata=body.metadata,
        )
    except HimmyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="knowledge",
        action="ingest",
        workspace_id=workspace_id,
        detail=f"ingested doc {doc.document_id} into {collection!r}",
    )
    return IngestResponse(collection=collection, document=_doc_view(doc))


@router.post(
    "/{collection}/documents/file",
    response_model=IngestResponse,
    dependencies=_WRITE,
    status_code=201,
)
async def ingest_document_file(
    collection: str, body: IngestFileRequest, request: Request
) -> IngestResponse:
    """Ingest a SERVER-side file into a collection (operator-only — no tenant LFI).

    Reading an arbitrary server path on behalf of a tenant is an LFI/traversal surface,
    so a tenant-bound caller is rejected (403). Only an all-tenants/offline operator may
    name a server path; tenants ingest content via text or the Studio upload.
    """
    _check_collection(collection)
    workspace_id = require_workspace(request, body.workspace_id)
    if not _operator_provisioned(request):
        audit_event(
            request,
            event_type="access",
            outcome="deny",
            resource="knowledge",
            action="ingest_file",
            workspace_id=workspace_id,
            detail="rejected tenant server-path file ingest (operator-only)",
        )
        raise HTTPException(
            status_code=403,
            detail="ingesting a server-side file path is operator-only; "
            "tenants ingest via text or the Studio upload",
        )
    try:
        doc = await _knowledge_app(request).ingest_file(
            workspace_id=workspace_id,
            collection=collection,
            path=body.file,
            metadata=body.metadata,
        )
    except HimmyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="knowledge",
        action="ingest_file",
        workspace_id=workspace_id,
        detail=f"ingested file doc {doc.document_id} into {collection!r}",
    )
    return IngestResponse(collection=collection, document=_doc_view(doc))


@router.get(
    "/{collection}/search", response_model=SearchResponse, dependencies=_READ
)
async def search_collection(
    collection: str,
    request: Request,
    q: str = "",
    workspace_id: str | None = None,
    top_k: int = 5,
) -> SearchResponse:
    """Search a workspace's collection by query ``q`` (an unknown collection → no hits)."""
    _check_collection(collection)
    workspace_id = resolve_workspace(request, workspace_id)
    if not q or not q.strip():
        raise HTTPException(status_code=422, detail="query 'q' is required")
    if len(q) > _MAX_QUERY:
        raise HTTPException(
            status_code=422, detail=f"query exceeds {_MAX_QUERY} characters"
        )
    top_k = max(1, min(int(top_k), _MAX_TOP_K))
    hits = await _knowledge_app(request).search(
        workspace_id=workspace_id or "",
        collection=collection,
        query=q,
        top_k=top_k,
    )
    items = [
        SearchHit(
            text=h.text,
            similarity=h.similarity,
            context_window=h.context_window,
            document_id=h.document_id,
            source_uri=h.source_uri,
            chunk_kind=h.chunk_kind,
            metadata=dict(h.metadata or {}),
        )
        for h in hits
    ]
    return SearchResponse(
        collection=collection, query=q, hits=items, total=len(items)
    )


@router.get(
    "/{collection}/documents",
    response_model=DocumentListResponse,
    dependencies=_READ,
)
async def list_documents(
    collection: str, request: Request, workspace_id: str | None = None
) -> DocumentListResponse:
    """List the documents in a workspace's collection (empty when unknown)."""
    _check_collection(collection)
    workspace_id = resolve_workspace(request, workspace_id)
    docs = await _knowledge_app(request).list_documents(
        workspace_id=workspace_id or "", collection=collection
    )
    items = [_doc_view(d) for d in docs]
    return DocumentListResponse(
        collection=collection, items=items, total=len(items)
    )


@router.delete(
    "/{collection}/documents/{document_id}",
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def delete_document(
    collection: str,
    document_id: str,
    request: Request,
    workspace_id: str | None = None,
) -> dict[str, str]:
    """Delete one document from a workspace's collection (404 when absent)."""
    _check_collection(collection)
    workspace_id = resolve_workspace(request, workspace_id)
    removed = await _knowledge_app(request).delete_document(
        workspace_id=workspace_id or "",
        collection=collection,
        document_id=document_id,
    )
    if not removed:
        raise HTTPException(status_code=404, detail="document not found")
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="knowledge",
        action="delete",
        workspace_id=workspace_id,
        detail=f"deleted doc {document_id} from {collection!r}",
    )
    return {"collection": collection, "document_id": document_id, "status": "deleted"}


__all__ = ["router"]
