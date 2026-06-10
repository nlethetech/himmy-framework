"""Golden-trace RAG snapshots: query -> expected doc -> evidence assertions, offline.

``golden_traces.json`` checks in a set of golden retrieval traces over the same
human-judged corpus the live eval uses (:mod:`tests.integration.rag_corpus`),
but runs them against the OFFLINE deterministic hashing embedder — the same
backend the hermetic unit suite uses — so the traces are exactly reproducible
on any machine with no model, no network, and no flake.

Where :mod:`tests.integration.test_rag_eval` gates aggregate recall/nDCG floors
on a live model, each golden case here asserts EVIDENCE quality per query:

* the expected document is retrieved FIRST (not merely somewhere in top-k);
* the retrieved chunk's own text contains the case's evidence terms, so the
  hit genuinely supports an answer rather than being topically adjacent;
* the result carries usable attribution (document id, the ingest metadata key,
  a chunk id, and a non-empty context window containing the chunk);
* every returned similarity is positive (the no-orthogonal-noise contract) and
  the list is sorted descending;
* the named decoy documents (the corpus's confusable neighbours) never outrank
  the target.

NOT marked ``integration``: this is the offline regression gate for the
retrieval pipeline (chunker, cosine path, threshold semantics, metadata
plumbing) and runs in the default lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from himmy.services.knowledge.embedder import DeterministicEmbedder
from himmy.services.knowledge.models import RetrievedChunk
from himmy.services.knowledge.service import KnowledgeBase
from himmy.services.storage.service import StorageService
from tests.conftest import run_async
from tests.integration.rag_corpus import DOCS

_WS, _CLIENT = "golden-traces", "golden-traces"
_FIXTURE_PATH = Path(__file__).with_name("golden_traces.json")
_FIXTURE: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_CASES: list[dict[str, Any]] = _FIXTURE["cases"]
_TOP_K: int = _FIXTURE["top_k"]


def _build_kb() -> tuple[KnowledgeBase, str, dict[str, str]]:
    """Ingest the corpus on the deterministic embedder; return (kb, kb_id, id->key)."""
    embedder = DeterministicEmbedder(dim=_FIXTURE["embedder"]["dim"])
    kb = KnowledgeBase(storage=StorageService(), embedder=embedder)
    rec = run_async(
        kb.create_kb(
            workspace_id=_WS, client_id=_CLIENT, name="corpus", vector_dim=embedder.dim
        )
    )
    id_to_key: dict[str, str] = {}
    for key, title, text in DOCS:
        doc = run_async(
            kb.ingest_text(rec.kb_id, text, title=title, metadata={"key": key})
        )
        id_to_key[doc.document_id] = key
    return kb, rec.kb_id, id_to_key


def _search(case: dict[str, Any]) -> tuple[list[RetrievedChunk], dict[str, str]]:
    kb, kb_id, id_to_key = _build_kb()
    results = run_async(
        kb.search(
            kb_id,
            case["query"],
            top_k=_TOP_K,
            workspace_id=_WS,
            client_id=_CLIENT,
        )
    )
    return results, id_to_key


def test_golden_fixture_shape_is_sound() -> None:
    """The checked-in fixture stays well-formed and keyed to real corpus docs."""
    corpus_keys = {key for key, _title, _text in DOCS}
    assert len(_CASES) >= 5
    names = [c["name"] for c in _CASES]
    assert len(names) == len(set(names)), "case names must be unique"
    for case in _CASES:
        assert case["query"].strip()
        assert case["expected_top_key"] in corpus_keys
        assert case["evidence_terms"], case["name"]
        for decoy in case["decoy_keys"]:
            assert decoy in corpus_keys, (case["name"], decoy)
            assert decoy != case["expected_top_key"], case["name"]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_golden_trace_evidence(case: dict[str, Any]) -> None:
    """Each golden query retrieves its document FIRST with real supporting evidence."""
    results, id_to_key = _search(case)
    assert results, f"{case['name']}: retrieval returned nothing"

    # Structural contract: positive similarities only, sorted descending.
    similarities = [r.similarity for r in results]
    assert all(s > 0.0 for s in similarities), case["name"]
    assert similarities == sorted(similarities, reverse=True), case["name"]

    # Precision@1: the judged-relevant document is the TOP hit, not just present.
    top = results[0]
    assert id_to_key[top.document_id] == case["expected_top_key"], (
        f"{case['name']}: expected {case['expected_top_key']!r} first, "
        f"got {id_to_key.get(top.document_id)!r}"
    )

    # Evidence quality: the retrieved chunk ITSELF contains the supporting terms
    # (a topically-adjacent chunk that lacks them would pass recall but fail here).
    text = (top.text or "").lower()
    for term in case["evidence_terms"]:
        assert term.lower() in text, f"{case['name']}: evidence {term!r} not in hit"

    # Attribution: the hit is traceable back to its source document.
    assert top.metadata.get("key") == case["expected_top_key"], case["name"]
    assert top.metadata.get("chunk_id"), case["name"]
    assert top.context_window, case["name"]
    assert (top.text or "") in top.context_window, case["name"]

    # The confusable neighbours never outrank the target (strictly weaker evidence).
    decoys = set(case["decoy_keys"])
    for other in results[1:]:
        if id_to_key.get(other.document_id) in decoys:
            assert other.similarity < top.similarity, (
                f"{case['name']}: decoy {id_to_key[other.document_id]!r} "
                f"tied/outranked the target"
            )
