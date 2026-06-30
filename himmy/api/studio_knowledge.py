"""Studio Knowledge: create a knowledge base, ingest text, and test retrieval.

Wraps :class:`~himmy.services.knowledge.service.KnowledgeBase` as a process-wide
singleton. The default store is in-memory (persists for the session); set ``HIMMY_KB_DSN``
to back it with Postgres+pgvector for durability. The same ``kb_search`` an agent uses reads
it.

TENANCY: the KB ``(workspace_id, client_id)`` scope is no longer a fixed
``("studio", "local")`` — it is derived PER REQUEST from the verified principal via
:func:`scope_keys`, mirroring the memory surface's ``t:<workspace>:`` namespacing and the
tool-pack :meth:`ToolkitConfig.scoped_kb_keys`. An offline / ``all_tenants`` principal keeps
the historical ``("studio", "local")`` scope (byte-for-byte unchanged); a tenant-bound
principal's KBs live under its own ``t:<token>`` scope so two tenants sharing one process
(or one shared ``HIMMY_KB_DSN`` backend) can neither list nor read each other's KBs. The
underlying :meth:`KnowledgeBase.search`/:meth:`delete_kb` are additionally handed the scope so
``_authorize_kb`` rejects a raw foreign ``kb_id`` by id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import Request

#: The scope used when no principal narrows it (offline / ``all_tenants``) — byte-unchanged.
_WORKSPACE = "studio"
_CLIENT = "local"

_KB: Any = None
#: (workspace_id) → {name → kb_id} for THIS session, namespaced by tenant scope so a
#: tenant-bound ``list_kbs`` never surfaces another tenant's KB names/ids.
_NAMES: dict[str, dict[str, str]] = {}
#: kb_id → ingested doc count (kb_id is globally unique, so this stays flat).
_DOCS: dict[str, int] = {}


class KbInfo(BaseModel):
    kb_id: str
    name: str
    documents: int = 0


class IngestResult(BaseModel):
    document_id: str
    title: str | None = None


class SearchHit(BaseModel):
    text: str
    similarity: float
    source_uri: str | None = None


def scope_keys(request: Request | None) -> tuple[str, str]:
    """The ``(workspace_id, client_id)`` a Studio knowledge op runs under for ``request``.

    Returns the historical ``("studio", "local")`` for an offline / ``all_tenants`` principal
    (or when ``request`` is ``None``, e.g. the upload twin's count sync) — byte-for-byte
    unchanged — and otherwise ``("t:<token>", "t:<token>")`` where ``<token>`` is the
    principal's tenant(+within-tenant subject) namespace, the SAME scheme the tool-pack KB
    (:meth:`ToolkitConfig.scoped_kb_keys`) uses, so a tenant-bound Studio console and that
    tenant's agents share one KB scope and no other tenant's.
    """
    if request is None:
        return _WORKSPACE, _CLIENT
    from himmy.api.routers.studio import _run_owner, _run_subject_scope
    from himmy.toolkit.config import ToolkitConfig

    owner_workspace, _ = _run_owner(request)
    subject_scope = _run_subject_scope(request)
    if owner_workspace is None and subject_scope is None:
        return _WORKSPACE, _CLIENT
    cfg = ToolkitConfig.from_env()
    cfg.tenant_scope = owner_workspace
    cfg.subject_scope = subject_scope
    return cfg.scoped_kb_keys()


def _service() -> Any:
    global _KB
    if _KB is None:
        from himmy.runtime.builder import build_storage
        from himmy.services.knowledge.service import KnowledgeBase
        from himmy.toolkit.config import ToolkitConfig

        embedder, _dim = ToolkitConfig.from_env().build_embedder_and_dim()
        _KB = KnowledgeBase(storage=build_storage(), embedder=embedder)
    return _KB


def reset_kb_service() -> None:
    global _KB, _NAMES, _DOCS
    _KB = None
    _NAMES = {}
    _DOCS = {}


def list_kbs(*, workspace_id: str = _WORKSPACE) -> list[KbInfo]:
    names = _NAMES.get(workspace_id, {})
    return [
        KbInfo(kb_id=kb_id, name=name, documents=_DOCS.get(kb_id, 0))
        for name, kb_id in names.items()
    ]


async def create_kb(
    name: str, *, workspace_id: str = _WORKSPACE, client_id: str = _CLIENT
) -> KbInfo:
    names = _NAMES.setdefault(workspace_id, {})
    if name in names:
        return KbInfo(
            kb_id=names[name], name=name, documents=_DOCS.get(names[name], 0)
        )
    rec = await _service().create_kb(
        workspace_id=workspace_id, client_id=client_id, name=name
    )
    names[name] = rec.kb_id
    _DOCS[rec.kb_id] = 0
    return KbInfo(kb_id=rec.kb_id, name=name, documents=0)


def _kb_in_scope(kb_id: str, workspace_id: str) -> bool:
    """True when ``kb_id`` was created within ``workspace_id``'s namespace this session."""
    return kb_id in _NAMES.get(workspace_id, {}).values()


async def ingest_text(
    kb_id: str,
    text: str,
    *,
    title: str | None = None,
    workspace_id: str = _WORKSPACE,
    client_id: str = _CLIENT,
) -> IngestResult:
    # Tenancy: a raw foreign ``kb_id`` must not be writable by id. ``_authorize_kb`` on
    # the service verifies ownership; we also gate by the session name map for the
    # in-memory store (whose records carry their creating workspace).
    await _authorize_kb_scope(kb_id, workspace_id, client_id)
    doc = await _service().ingest_text(kb_id, text, title=title)
    _DOCS[kb_id] = _DOCS.get(kb_id, 0) + 1
    return IngestResult(document_id=doc.document_id, title=doc.title)


async def search(
    kb_id: str,
    query: str,
    *,
    top_k: int = 5,
    workspace_id: str = _WORKSPACE,
    client_id: str = _CLIENT,
) -> list[SearchHit]:
    # Tenancy guard: ``_authorize_kb`` raises if ``kb_id`` is not owned by this scope, so a
    # raw foreign kb_id from another tenant cannot read chunks here.
    chunks = await _service().search(
        kb_id, query, top_k=top_k, workspace_id=workspace_id, client_id=client_id
    )
    return [
        SearchHit(
            text=c.text or c.context_window or "",
            similarity=c.similarity,
            source_uri=c.source_uri,
        )
        for c in chunks
    ]


async def delete_kb(
    kb_id: str, *, workspace_id: str = _WORKSPACE, client_id: str = _CLIENT
) -> bool:
    # ``_authorize_kb`` (missing_ok) verifies ownership so a foreign kb_id cannot be deleted.
    await _service().delete_kb(kb_id, workspace_id=workspace_id, client_id=client_id)
    names = _NAMES.get(workspace_id, {})
    for name, kid in list(names.items()):
        if kid == kb_id:
            del names[name]
    _DOCS.pop(kb_id, None)
    return True


async def _authorize_kb_scope(kb_id: str, workspace_id: str, client_id: str) -> None:
    """Raise 404-style if ``kb_id`` is not owned by ``(workspace_id, client_id)``.

    Delegates to the service's own tenancy guard (``_authorize_kb``) which compares the
    resolved KB record's scope. A KB created outside this scope (e.g. another tenant's)
    folds to a ``HimmyError`` the router maps to 404, so existence never leaks.
    """
    svc = _service()
    authorize = getattr(svc, "_authorize_kb", None)
    if authorize is not None:
        await authorize(kb_id, workspace_id, client_id, missing_ok=True)


__all__ = [
    "KbInfo",
    "IngestResult",
    "SearchHit",
    "scope_keys",
    "reset_kb_service",
    "list_kbs",
    "create_kb",
    "ingest_text",
    "search",
    "delete_kb",
]
