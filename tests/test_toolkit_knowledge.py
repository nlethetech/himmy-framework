"""Tests for the knowledge pack: kb_ingest + kb_search (offline, deterministic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.knowledge import register_knowledge_pack
from tests.conftest import run_async


def _registry(root: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    register_knowledge_pack(registry, ToolkitConfig(fs_root=root or Path.cwd()))
    return registry


def test_ingest_then_search_finds_text() -> None:
    reg = _registry()
    ingest = reg.handler_for("kb_ingest")
    search = reg.handler_for("kb_search")
    out = run_async(
        ingest(
            {
                "text": "Permaculture food forests layer canopy shrubs and roots",
                "title": "pc",
            }
        )
    )
    assert out["ingested"] == 1
    assert out["document_ids"]
    # The offline DeterministicEmbedder matches on exact token overlap, so the query
    # shares words with the ingested text.
    found = run_async(search({"query": "food forests layer canopy", "top_k": 3}))
    assert found["query"] == "food forests layer canopy"
    assert len(found["results"]) >= 1
    assert any("food forests" in (r["text"] or "").lower() for r in found["results"])


def test_ingest_requires_text_or_path() -> None:
    with pytest.raises(ValueError):
        run_async(_registry().handler_for("kb_ingest")({}))


def test_ingest_from_file(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("bees pollinate the orchard each spring")
    reg = _registry(tmp_path)
    out = run_async(reg.handler_for("kb_ingest")({"path": "doc.txt"}))
    assert out["ingested"] == 1
    found = run_async(
        reg.handler_for("kb_search")({"query": "bees pollinate the orchard"})
    )
    assert len(found["results"]) >= 1


def test_search_empty_kb_returns_no_results() -> None:
    found = run_async(_registry().handler_for("kb_search")({"query": "anything"}))
    assert found["results"] == []


def test_scoped_kb_keys_namespace_by_tenant() -> None:
    """Round-8 regression: the durable-KB scope is tenant-namespaced, not a fixed singleton.

    The knowledge pack pinned its KB to a static ``("local", "local")`` scope, so a SHARED
    durable (pgvector) backend handed EVERY tenant the same KB — t1's ``kb_ingest`` chunks
    came back from t2's ``kb_search``. With ``tenant_scope`` set the scope keys are the
    tenant's own (distinct per tenant); ``None`` keeps the historical shared
    ``("local", "local")`` so the offline default is byte-for-byte unchanged.
    """
    none_keys = ToolkitConfig().scoped_kb_keys()
    assert none_keys == ("local", "local")  # offline invariant: unchanged

    t1 = ToolkitConfig(tenant_scope="tenant-1").scoped_kb_keys()
    t2 = ToolkitConfig(tenant_scope="tenant-2").scoped_kb_keys()
    assert t1 != t2, "two tenants must resolve to distinct KB scopes"
    assert t1 != none_keys and t2 != none_keys


def test_kb_isolated_across_tenants_on_shared_durable_backend(tmp_path: Path) -> None:
    """t2 cannot resolve t1's KB on a SHARED DURABLE backend (the durable-leak fix).

    Reproduces the ``HIMMY_KB_DSN`` deployment offline with the persistent
    :class:`SqliteKnowledgeBackend` (a single shared file, like one shared pgvector DB):
    tenant t1 creates+ingests its KB under its OWN scope; a SECOND ``KnowledgeBase`` on the
    SAME backend resolving under t2's scope must return ``None`` — t1's chunks are
    unreachable from t2. Before the fix both packs pinned the fixed ``(local, local,
    default)`` scope → t2 resolved t1's exact KB and read its chunks. The per-tenant
    ``scoped_kb_keys`` makes t2's resolve miss. FAILS before, passes after.
    """
    from himmy.services.knowledge import DocumentInput, KnowledgeBase
    from himmy.services.knowledge.sqlite_backend import SqliteKnowledgeBackend
    from himmy.services.storage.service import StorageService

    db = str(tmp_path / "shared_kb.db")
    embedder, dim = ToolkitConfig().build_embedder_and_dim()

    async def scenario() -> tuple[str | None, str]:
        # Tenant 1: create + ingest under ITS scope on the shared durable backend.
        ws1, c1 = ToolkitConfig(tenant_scope="tenant-1").scoped_kb_keys()
        kb1 = KnowledgeBase(
            storage=StorageService(),
            embedder=embedder,
            backend=SqliteKnowledgeBackend(db),
        )
        rec1 = await kb1.create_kb(
            workspace_id=ws1, client_id=c1, name="default", vector_dim=dim
        )
        await kb1.ingest_documents(
            rec1.kb_id, [DocumentInput(text="tenant one private merger plan secret")]
        )
        # Tenant 2: a FRESH KB on the SAME file resolves under ITS scope — must miss t1.
        ws2, c2 = ToolkitConfig(tenant_scope="tenant-2").scoped_kb_keys()
        kb2 = KnowledgeBase(
            storage=StorageService(),
            embedder=embedder,
            backend=SqliteKnowledgeBackend(db),
        )
        t2_kb = await kb2.resolve_kb(workspace_id=ws2, client_id=c2, name="default")
        return (t2_kb.kb_id if t2_kb else None), rec1.kb_id

    t2_resolved, t1_kb_id = run_async(scenario())
    assert t2_resolved != t1_kb_id, "tenant-2 resolved tenant-1's KB on shared backend"
    assert t2_resolved is None, "tenant-2 must not see any KB under its own scope"
