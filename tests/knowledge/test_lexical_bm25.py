"""Behavioral tests for the pure-Python BM25 lexical index."""

from __future__ import annotations

from himmy.services.knowledge.retrieval.lexical import BM25Index, tokenize


def test_rare_exact_token_retrieved_above_common_chunk() -> None:
    """A rare exact token surfaces its chunk; a token-absent chunk scores nothing."""
    index = BM25Index()
    index.add("c_rare", "The configuration uses identifier XQ7Z9 for routing.")
    index.add("c_other", "The configuration is documented in the manual.")
    index.add("c_filler", "An unrelated paragraph about cooking recipes.")

    hits = index.search("XQ7Z9", top_k=5)
    assert hits, "the rare-token query must retrieve at least one chunk"
    assert hits[0][0] == "c_rare"
    # Chunks without the token are not returned at all (zero overlap).
    assert {cid for cid, _ in hits} == {"c_rare"}


def test_idf_demotes_common_terms() -> None:
    """A term in every document carries near-zero idf vs. a rare discriminating term."""
    index = BM25Index()
    for i in range(5):
        index.add(f"c{i}", "common common term")
    index.add("c_special", "common rareword")

    # 'common' appears in every doc -> low idf; 'rareword' only in c_special.
    hits = dict(index.search("rareword", top_k=10))
    assert "c_special" in hits
    common_hits = index.search("common", top_k=10)
    # The discriminating query ranks its single chunk strongly; the common query
    # spreads weak scores over many chunks.
    assert hits["c_special"] > max(s for _, s in common_hits)


def test_length_normalization_rewards_focused_match() -> None:
    """For the same term frequency, the shorter (more focused) document scores higher."""
    index = BM25Index()
    index.add("short", "alpha beta")
    index.add("long", "alpha " + "padding " * 50 + "beta")
    hits = dict(index.search("alpha", top_k=5))
    assert hits["short"] > hits["long"]


def test_remove_prunes_index_in_lockstep() -> None:
    """Removing a chunk drops it from results and updates corpus statistics."""
    index = BM25Index()
    index.add("c1", "kubernetes deployment manifest")
    index.add("c2", "docker compose configuration")
    assert index.num_documents == 2
    assert any(cid == "c1" for cid, _ in index.search("kubernetes", top_k=5))

    index.remove("c1")
    assert index.num_documents == 1
    assert index.search("kubernetes", top_k=5) == []
    # The surviving chunk is still retrievable.
    assert any(cid == "c2" for cid, _ in index.search("docker", top_k=5))


def test_readd_replaces_not_doubles() -> None:
    """Re-adding the same chunk_id replaces it rather than double-counting tf/df."""
    index = BM25Index()
    index.add("c1", "alpha alpha alpha")
    first = dict(index.search("alpha", top_k=5))["c1"]
    index.add("c1", "alpha")  # shorter content for the same id
    second = dict(index.search("alpha", top_k=5))["c1"]
    assert index.num_documents == 1
    # Score changes because the content changed (not because it was double-indexed).
    assert first != second


def test_idf_recomputed_lazily_after_mutation() -> None:
    """The IDF cache invalidates on mutation so stats stay correct across edits."""
    index = BM25Index()
    index.add("c1", "term")
    _ = index.search("term", top_k=1)  # warms the IDF cache
    index.add("c2", "term")  # mutation must invalidate the cache
    index.add("c3", "other")
    hits = index.search("term", top_k=5)
    # Both term-bearing chunks come back, proving stats were recomputed.
    assert {cid for cid, _ in hits} == {"c1", "c2"}


def test_tokenizer_shared_with_dense_embedder() -> None:
    """The lexical tokenizer is exactly the offline dense embedder's tokenizer."""
    assert tokenize("Hello, WORLD-123!") == ["hello", "world", "123"]


def test_empty_query_and_empty_index() -> None:
    """An empty query or empty index returns nothing, never an error."""
    index = BM25Index()
    assert index.search("anything", top_k=5) == []
    index.add("c1", "content")
    assert index.search("", top_k=5) == []
    assert index.search("content", top_k=0) == []
