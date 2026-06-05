"""Tests for Nepali language support: transliteration, folding, cross-script RAG."""

from __future__ import annotations

from himmy.nepal import (
    NepaliEmbedder,
    is_devanagari,
    normalize_nepali,
    transliterate,
)
from himmy.services.knowledge import KnowledgeBase
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def test_is_devanagari() -> None:
    """Devanagari detection distinguishes scripts."""
    assert is_devanagari("नेपाल") is True
    assert is_devanagari("Nepal") is False
    assert is_devanagari("nepal 123") is False


def test_transliterate_handles_consonant_matra_halant() -> None:
    """A consonant takes its matra; a halant drops the inherent vowel."""
    assert transliterate("क") == "ka"
    assert transliterate("के") == "ke"
    assert "nepa" in transliterate("नेपाल")  # ne-paa-la


def test_normalize_folds_scripts_together() -> None:
    """Devanagari, Romanized, and English forms fold to the same token."""
    assert normalize_nepali("नेपाल") == normalize_nepali("Nepal") == "nepal"
    assert normalize_nepali("NEPAL") == "nepal"
    # A phrase shares the key token across scripts.
    ne = set(normalize_nepali("नेपाल को अर्थतन्त्र").split())
    en = set(normalize_nepali("Nepal economy").split())
    assert "nepal" in ne & en


def _kb() -> tuple[KnowledgeBase, str]:
    kb = KnowledgeBase(storage=StorageService(), embedder=NepaliEmbedder())
    kb_id = run_async(kb.create_kb(workspace_id="w", client_id="c", name="ne")).kb_id
    return kb, kb_id


def test_cross_script_retrieval_devanagari_doc_roman_query() -> None:
    """A Devanagari document is retrievable by a Romanized query."""
    kb, kb_id = _kb()
    run_async(kb.ingest_text(kb_id, "नेपाल को अर्थतन्त्र बलियो छ", title="ne"))
    run_async(kb.ingest_text(kb_id, "gardening tomatoes in spring", title="garden"))
    results = run_async(kb.search(kb_id, "nepal", top_k=1))
    assert results
    assert "नेपाल" in (results[0].text or "")
    assert results[0].similarity > 0.0


def test_cross_script_retrieval_roman_doc_devanagari_query() -> None:
    """A Romanized document is retrievable by a Devanagari query."""
    kb, kb_id = _kb()
    run_async(kb.ingest_text(kb_id, "Nepal economy outlook report", title="en"))
    run_async(kb.ingest_text(kb_id, "unrelated weather note", title="weather"))
    results = run_async(kb.search(kb_id, "नेपाल", top_k=1))
    assert results
    assert "Nepal" in (results[0].text or "")
