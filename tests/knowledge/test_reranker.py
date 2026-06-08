"""Behavioral tests for the reranker seam (stub + extra-gated real path)."""

from __future__ import annotations

import pytest

from himmy.core.errors import HimmyError
from himmy.services.knowledge.retrieval.reranker import (
    FastEmbedReranker,
    RerankerProtocol,
    build_reranker,
    fastembed_rerank_available,
)


class _KeywordReranker:
    """A deterministic offline reranker: score = count of 'urgent' in the text.

    Stands in for a real cross-encoder so the rerank stage can be exercised with no
    model download — exactly the kind of stub the design calls for.
    """

    async def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        scored = [
            (cid, float(text.lower().count("urgent"))) for cid, text in candidates
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


def test_stub_reranker_satisfies_protocol() -> None:
    """A structurally-matching reranker passes the runtime_checkable Protocol."""
    assert isinstance(_KeywordReranker(), RerankerProtocol)


def test_stub_reranker_reorders_candidates() -> None:
    """The reranker re-scores candidates and returns them in its own order."""
    import asyncio

    reranker = _KeywordReranker()
    candidates = [
        ("c1", "a calm ordinary note"),
        ("c2", "urgent urgent action needed"),
        ("c3", "one urgent matter"),
    ]
    ranked = asyncio.run(reranker.rerank("anything", candidates))
    order = [cid for cid, _ in ranked]
    assert order == ["c2", "c3", "c1"]
    assert dict(ranked)["c2"] == 2.0


def test_build_reranker_unknown_name() -> None:
    """An unknown reranker name is a clear error."""
    with pytest.raises(HimmyError, match="unknown reranker"):
        build_reranker("does-not-exist")


def test_fastembed_reranker_is_import_safe() -> None:
    """Constructing FastEmbedReranker never imports fastembed (lazy)."""
    reranker = build_reranker("fastembed")
    assert isinstance(reranker, FastEmbedReranker)
    assert isinstance(reranker, RerankerProtocol)


@pytest.mark.skipif(
    fastembed_rerank_available(),
    reason="fastembed installed — the missing-extra error path cannot be exercised",
)
def test_fastembed_reranker_raises_clear_error_without_extra() -> None:
    """Without the [embeddings] extra, USING the reranker raises a clear HimmyError."""
    import asyncio

    reranker = FastEmbedReranker()
    with pytest.raises(HimmyError, match=r"\[embeddings\] extra"):
        asyncio.run(reranker.rerank("q", [("c1", "some text")]))


@pytest.mark.skipif(
    not fastembed_rerank_available(),
    reason="fastembed not installed — real ONNX rerank path unavailable",
)
def test_fastembed_reranker_real_path() -> None:  # pragma: no cover - extra-gated
    """With fastembed present, the real cross-encoder ranks a relevant doc first."""
    import asyncio

    reranker = FastEmbedReranker()
    candidates = [
        ("c_off", "Bananas are a yellow fruit grown in the tropics."),
        ("c_on", "Python is a high-level programming language used for scripting."),
    ]
    ranked = asyncio.run(
        reranker.rerank("What programming language is good for scripting?", candidates)
    )
    assert ranked[0][0] == "c_on"
