"""Tests for content-addressed ingest: a re-scan replaces, it does not double.

Without content identity, re-ingesting a corpus minted a fresh document_id each
time and stacked duplicate chunks (polluting top-k and wasting embed quota). These
tests pin the corrected behaviour on the in-memory path.
"""

from __future__ import annotations

from himmy.services.knowledge import DeterministicEmbedder, KnowledgeBase
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _kb() -> tuple[KnowledgeBase, str]:
    service = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    record = run_async(service.create_kb(workspace_id="w", client_id="c", name="kb"))
    return service, record.kb_id


def _doc_count(service: KnowledgeBase, kb_id: str) -> int:
    return len(service._documents[kb_id])


def test_reingest_identical_source_is_a_noop() -> None:
    """Re-ingesting the same source + content reuses the doc, never duplicates."""
    kb, kb_id = _kb()
    first = run_async(kb.ingest_text(kb_id, "ACME revenue grew.", source_uri="doc://1"))
    again = run_async(kb.ingest_text(kb_id, "ACME revenue grew.", source_uri="doc://1"))
    assert again.document_id == first.document_id
    assert _doc_count(kb, kb_id) == 1
    assert first.content_hash  # the identity is populated


def test_changed_source_replaces_rather_than_doubles() -> None:
    """Editing a source's content replaces the prior document and its chunks."""
    kb, kb_id = _kb()
    run_async(kb.ingest_text(kb_id, "old widget content alpha", source_uri="doc://1"))
    run_async(kb.ingest_text(kb_id, "new gadget content beta", source_uri="doc://1"))
    assert _doc_count(kb, kb_id) == 1  # replaced, not stacked

    fresh = run_async(kb.search(kb_id, "gadget beta", top_k=5))
    assert fresh and "gadget" in (fresh[0].text or "")
    # The superseded chunk is gone — nothing still surfaces the old text.
    stale = run_async(kb.search(kb_id, "widget alpha", top_k=5))
    assert all("widget" not in (r.text or "") for r in stale)


def test_directory_rescan_does_not_double_and_picks_up_edits(tmp_path) -> None:
    """Re-scanning a directory keeps the doc count stable and reflects edits."""
    (tmp_path / "a.txt").write_text("alpha apple orchard", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta banana grove", encoding="utf-8")
    kb, kb_id = _kb()

    run_async(kb.ingest_directory(kb_id, str(tmp_path)))
    assert _doc_count(kb, kb_id) == 2
    run_async(kb.ingest_directory(kb_id, str(tmp_path)))  # unchanged re-scan
    assert _doc_count(kb, kb_id) == 2

    (tmp_path / "a.txt").write_text("alpha cherry orchard", encoding="utf-8")
    run_async(kb.ingest_directory(kb_id, str(tmp_path)))
    assert _doc_count(kb, kb_id) == 2  # replaced in place
    hit = run_async(kb.search(kb_id, "cherry", top_k=5))
    assert hit and "cherry" in (hit[0].text or "")


def test_unchanged_reingest_skips_embedding() -> None:
    """An unchanged re-ingest does not spend an embed call."""

    class CountingEmbedder(DeterministicEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=64)
            self.doc_calls = 0

        async def embed_documents(self, texts):  # type: ignore[override]
            self.doc_calls += 1
            return await super().embed_documents(texts)

    emb = CountingEmbedder()
    kb = KnowledgeBase(storage=StorageService(), embedder=emb)
    kb_id = run_async(kb.create_kb(workspace_id="w", client_id="c", name="kb")).kb_id

    run_async(kb.ingest_text(kb_id, "stable content here", source_uri="s://1"))
    assert emb.doc_calls == 1
    run_async(kb.ingest_text(kb_id, "stable content here", source_uri="s://1"))
    assert emb.doc_calls == 1  # no re-embed for unchanged content


def test_identical_raw_text_without_uri_dedups_distinct_still_adds() -> None:
    """Raw text dedups on content; genuinely different text still inserts."""
    kb, kb_id = _kb()
    a = run_async(kb.ingest_text(kb_id, "no uri content"))
    b = run_async(kb.ingest_text(kb_id, "no uri content"))
    assert a.document_id == b.document_id
    assert _doc_count(kb, kb_id) == 1

    c = run_async(kb.ingest_text(kb_id, "entirely different content"))
    assert c.document_id != a.document_id
    assert _doc_count(kb, kb_id) == 2
