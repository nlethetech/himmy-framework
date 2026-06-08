"""Validation tests for RetrievalConfig (turned-on stages need collaborators)."""

from __future__ import annotations

import pytest

from himmy.services.knowledge.retrieval.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    RetrievalConfig,
)


def test_default_is_pure_dense() -> None:
    """The default config is exactly the dense path with no extra stages."""
    cfg = RetrievalConfig()
    assert cfg.mode == "dense"
    assert cfg.rerank is False and cfg.query_rewrite is False
    assert cfg.is_default_dense is True
    assert DEFAULT_RETRIEVAL_CONFIG.is_default_dense is True


def test_hybrid_without_extras_is_not_default_dense() -> None:
    """Switching to hybrid leaves the dense fast-path."""
    assert RetrievalConfig(mode="hybrid").is_default_dense is False


def test_rerank_requires_reranker() -> None:
    """rerank=True with no reranker is a construction error (no silent no-op)."""
    with pytest.raises(ValueError, match="reranker"):
        RetrievalConfig(mode="hybrid", rerank=True)


def test_query_rewrite_requires_rewriter() -> None:
    """query_rewrite=True with no rewriter is a construction error."""
    with pytest.raises(ValueError, match="rewriter"):
        RetrievalConfig(mode="hybrid", query_rewrite=True)


def test_numeric_knobs_validated() -> None:
    """rrf_k and candidate_pool must be positive."""
    with pytest.raises(ValueError, match="rrf_k"):
        RetrievalConfig(rrf_k=0)
    with pytest.raises(ValueError, match="candidate_pool"):
        RetrievalConfig(candidate_pool=0)


def test_invalid_mode_rejected() -> None:
    """An unknown mode is rejected at construction."""
    with pytest.raises(ValueError, match="mode"):
        RetrievalConfig(mode="sparse")  # type: ignore[arg-type]
