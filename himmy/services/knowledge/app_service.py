"""Knowledge application service: workspace-namespaced CRUD/ingest/search over a KB.

This is the service the ``/v1/knowledge`` router rides on (T3a). It wraps a single
:class:`~himmy.services.knowledge.service.KnowledgeBase` and threads every operation
through ONE choke-point namespace key — :func:`knowledge_namespace` — so a tenant's
``collection`` can never collide with (or read) another tenant's documents.

ISOLATION HONESTY (the load-bearing caveat — do NOT pretend otherwise):

The per-workspace namespace gives isolation ONLY in an AUTHENTICATED deployment. The
``workspace_id`` that reaches this service is the one
:func:`~himmy.api.auth.context.resolve_workspace` returns, and that function only
*derives* a trusted workspace from the verified principal when an authenticator is
configured. In the zero-config offline default there is no authenticator: the principal
is :data:`~himmy.api.auth.principal.ANONYMOUS` with ``all_tenants=True``, so
``resolve_workspace`` echoes back whatever ``workspace_id`` the caller supplied AND RBAC
is bypassed. In that regime ANY caller can name ANY workspace, so the namespace is a
*labelling* convenience, NOT a security boundary — the isolation is INERT. A
multi-tenant operator MUST configure an authenticator (and ideally per-tenant API keys)
for the boundary to hold. The router documents this on every endpoint; a test asserts
the namespace derivation and that offline isolation is inert (no false pass).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from himmy.services.knowledge.models import KnowledgeDocument, RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.knowledge.service import KnowledgeBase


def knowledge_namespace(workspace_id: str, collection: str) -> str:
    """Derive the SINGLE per-tenant KB name from ``(workspace_id, collection)``.

    This is the one and only place a workspace + collection are joined into a KB
    identity, so there is no ad-hoc per-route string-building that could drift and let
    one tenant address another tenant's collection. The shape ``"{workspace}:{collection}"``
    embeds the (trusted, in an authenticated deployment) workspace into the KB name, so
    even a collection name that collides across tenants resolves to a different KB.

    SECURITY NOTE: this is only a boundary when ``workspace_id`` is the
    authenticated/derived workspace (see this module's docstring). Offline (no auth) the
    caller chooses ``workspace_id`` freely, so the namespace is inert — a label, not a
    wall.
    """
    return f"{workspace_id}:{collection}"


class KnowledgeAppService:
    """Workspace-namespaced ingest/search/list/delete over one :class:`KnowledgeBase`.

    A KB is resolved-or-created lazily per ``(workspace_id, collection)`` via the single
    :func:`knowledge_namespace` choke point. The ``client_id`` is fixed (a himmy
    deployment is one logical client; the *workspace* is the tenant axis), and the KB's
    own native ``(workspace_id, client_id, name)`` scoping is fed the resolved workspace
    as a second, defense-in-depth check — so even a forged raw ``kb_id`` could not reach
    another workspace's chunks at the KB layer.

    Durability follows the wrapped KB: with the default in-memory backend documents live
    for the process lifetime (the offline contract); with a pgvector-backed
    :class:`~himmy.services.knowledge.backend.KnowledgeBackendProtocol` they persist. The
    container reuses its durable storage handle here so the surface is consistent across
    interfaces.
    """

    #: The fixed logical client for a himmy deployment (the tenant axis is the workspace).
    CLIENT_ID = "v1"

    def __init__(self, knowledge_base: KnowledgeBase, *, vector_dim: int = 64) -> None:
        """Wire the service to a :class:`KnowledgeBase`.

        ``vector_dim`` MUST match the dimension of the KB's embedder — a newly created
        collection is stamped with it, and the KB rejects an ingest whose embeddings do
        not match the collection's ``vector_dim``. The container threads the embedder's
        resolved dimension here so an ``embedder="auto"`` (or any non-default) backend
        does not silently mismatch the 64-dim default.
        """
        self._kb = knowledge_base
        self._vector_dim = vector_dim

    async def _resolve_collection_kb_id(
        self, workspace_id: str, collection: str, *, create: bool
    ) -> str | None:
        """Resolve the kb_id backing ``(workspace_id, collection)``; optionally create it.

        Returns ``None`` when the collection does not exist and ``create`` is False, so a
        read against an unknown collection is an empty result rather than an error.
        """
        name = knowledge_namespace(workspace_id, collection)
        existing = await self._kb.resolve_kb(
            workspace_id=workspace_id, client_id=self.CLIENT_ID, name=name
        )
        if existing is not None:
            return existing.kb_id
        if not create:
            return None
        record = await self._kb.create_kb(
            workspace_id=workspace_id,
            client_id=self.CLIENT_ID,
            name=name,
            vector_dim=self._vector_dim,
        )
        return record.kb_id

    async def ingest_text(
        self,
        *,
        workspace_id: str,
        collection: str,
        text: str,
        title: str | None = None,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Ingest one raw-text document into a workspace's collection (auto-creating it)."""
        kb_id = await self._resolve_collection_kb_id(
            workspace_id, collection, create=True
        )
        assert kb_id is not None  # noqa: S101 - create=True always returns an id
        return await self._kb.ingest_text(
            kb_id,
            text,
            title=title,
            source_uri=source_uri,
            metadata=metadata,
        )

    async def ingest_file(
        self,
        *,
        workspace_id: str,
        collection: str,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        """Ingest one file (read via the reader factory) into a workspace's collection."""
        kb_id = await self._resolve_collection_kb_id(
            workspace_id, collection, create=True
        )
        assert kb_id is not None  # noqa: S101 - create=True always returns an id
        return await self._kb.ingest_file(kb_id, path, metadata=metadata)

    async def search(
        self,
        *,
        workspace_id: str,
        collection: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search a workspace's collection; an unknown collection returns ``[]``.

        The KB's native ``(workspace_id, client_id)`` guard is passed the resolved
        workspace so a chunk can only ever be returned for the requesting workspace.
        """
        kb_id = await self._resolve_collection_kb_id(
            workspace_id, collection, create=False
        )
        if kb_id is None:
            return []
        return await self._kb.search(
            kb_id,
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters,
            workspace_id=workspace_id,
            client_id=self.CLIENT_ID,
        )

    async def list_documents(
        self, *, workspace_id: str, collection: str
    ) -> list[KnowledgeDocument]:
        """List the documents in a workspace's collection (empty when unknown).

        Reads the in-memory document store directly (the same surface the Studio KB
        document listing uses); a pgvector-backed KB degrades to an empty list, matching
        the documented backend behaviour.
        """
        kb_id = await self._resolve_collection_kb_id(
            workspace_id, collection, create=False
        )
        if kb_id is None:
            return []
        docs_map = getattr(self._kb, "_documents", {}).get(kb_id, {})
        return list(docs_map.values())

    async def delete_document(
        self, *, workspace_id: str, collection: str, document_id: str
    ) -> bool:
        """Delete one document from a workspace's collection; ``False`` when absent.

        The KB's native scoping receives the resolved workspace so a raw ``kb_id`` from a
        different tenant could not delete this workspace's documents at the KB layer.
        """
        kb_id = await self._resolve_collection_kb_id(
            workspace_id, collection, create=False
        )
        if kb_id is None:
            return False
        return await self._kb.delete_document(
            kb_id,
            document_id,
            workspace_id=workspace_id,
            client_id=self.CLIENT_ID,
        )


__all__ = ["KnowledgeAppService", "knowledge_namespace"]
