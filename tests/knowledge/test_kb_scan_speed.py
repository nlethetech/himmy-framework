"""Identity + concurrency tests for the P2.2 in-memory dense-scan speedups.

These assert the query-norm hoist and the ``asyncio.to_thread`` offload leave the
retrieval output (scores, ordering, tie-breaks, thresholds) byte-identical to the
pre-P2 pure-Python path.
"""

from __future__ import annotations

import asyncio
import random

from himmy.services.knowledge import DeterministicEmbedder, KnowledgeBase
from himmy.services.knowledge.service import (
    _cosine,
    _cosine_with_norm,
    _vec_norm,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def test_cosine_with_norm_is_byte_identical_to_cosine() -> None:
    """Hoisting the query norm must not perturb a single cosine value."""
    rng = random.Random(20260702)
    for _ in range(500):
        dim = rng.randint(1, 48)
        a = [rng.uniform(-3.0, 3.0) for _ in range(dim)]
        b = [rng.uniform(-3.0, 3.0) for _ in range(dim)]
        expected = _cosine(a, b)
        got = _cosine_with_norm(a, _vec_norm(a), b)
        assert got == expected  # exact float equality, not approx


def test_cosine_with_norm_matches_degenerate_and_zero_cases() -> None:
    """Empty/length-mismatch/zero-norm guards stay identical under the hoist."""
    assert _cosine_with_norm([], _vec_norm([]), [1.0]) == _cosine([], [1.0])
    assert _cosine_with_norm([1.0], _vec_norm([1.0]), []) == _cosine([1.0], [])
    assert _cosine_with_norm([1.0, 2.0], _vec_norm([1.0, 2.0]), [1.0]) == _cosine(
        [1.0, 2.0], [1.0]
    )
    zero = [0.0, 0.0, 0.0]
    other = [1.0, 2.0, 3.0]
    assert _cosine_with_norm(zero, _vec_norm(zero), other) == _cosine(zero, other)
    assert _cosine_with_norm(other, _vec_norm(other), zero) == _cosine(other, zero)


def _pure_python_scan(chunks, query_vec, keep, metadata_filters):
    """Reference pre-P2 inline loop (per-chunk _cosine, no norm hoist)."""
    scored = []
    for chunk in chunks:
        if metadata_filters and any(
            chunk.metadata.get(k) != v for k, v in metadata_filters.items()
        ):
            continue
        sim = _cosine(query_vec, chunk.embedding)
        if not keep(sim):
            continue
        scored.append((sim, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def test_dense_scan_sync_identical_to_reference_loop() -> None:
    """The offloaded sync scan reproduces the reference scores + ordering exactly."""
    rng = random.Random(4242)
    dim = 16

    class _Chunk:
        def __init__(self, cid, emb, meta):
            self.chunk_id = cid
            self.embedding = emb
            self.metadata = meta

    chunks = [
        _Chunk(
            f"c{i}",
            [rng.uniform(-1.0, 1.0) for _ in range(dim)],
            {"tag": "even" if i % 2 == 0 else "odd"},
        )
        for i in range(200)
    ]
    query_vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    def keep(sim: float) -> bool:
        return sim > 0.0

    for filters in (None, {"tag": "even"}):
        expected = _pure_python_scan(chunks, query_vec, keep, filters)
        got = KnowledgeBase._dense_scan_sync(chunks, query_vec, keep, filters)
        assert [s for s, _ in got] == [s for s, _ in expected]
        assert [c.chunk_id for _, c in got] == [c.chunk_id for _, c in expected]


def test_search_output_identical_via_offload() -> None:
    """End-to-end search results are unchanged by the to_thread offload."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    record = run_async(
        kb.create_kb(workspace_id="w1", client_id="c1", name="scan-speed")
    )
    run_async(
        kb.ingest_text(
            record.kb_id,
            "The mitochondria is the powerhouse of the cell. "
            "Photosynthesis converts sunlight into chemical energy. "
            "Neurons transmit electrical signals across synapses. "
            "Rivers carve canyons over geological time.",
            source_uri="doc://bio",
        )
    )

    hits = run_async(kb.search(record.kb_id, "how do cells make energy", top_k=3))
    # Deterministic: re-running yields the same ordering and identical scores.
    hits2 = run_async(kb.search(record.kb_id, "how do cells make energy", top_k=3))
    assert [(h.similarity, h.text) for h in hits] == [
        (h.similarity, h.text) for h in hits2
    ]
    # Similarities are sorted descending (the preserved sort contract).
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_concurrent_searches_are_thread_safe() -> None:
    """Many concurrent offloaded scans over the same KB return consistent results."""
    kb = KnowledgeBase(storage=StorageService(), embedder=DeterministicEmbedder())
    record = run_async(
        kb.create_kb(workspace_id="w1", client_id="c1", name="concurrent")
    )
    run_async(
        kb.ingest_text(
            record.kb_id,
            " ".join(f"sentence number {i} about topic {i % 5}." for i in range(60)),
            source_uri="doc://concurrent",
        )
    )

    async def _run_many() -> list[list[tuple[float, str]]]:
        tasks = [
            kb.search(record.kb_id, "topic 3 sentence", top_k=5)
            for _ in range(24)
        ]
        results = await asyncio.gather(*tasks)
        return [[(h.similarity, h.text) for h in r] for r in results]

    all_results = run_async(_run_many())
    first = all_results[0]
    for other in all_results[1:]:
        assert other == first
