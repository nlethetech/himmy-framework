"""Tests for the real local embedders + the embedder factory (offline)."""

from __future__ import annotations

import pytest

from himmy.core import HimmyError
from himmy.services.knowledge.embedder import DeterministicEmbedder
from himmy.services.knowledge.local_embedders import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    build_embedder,
    default_dim_for,
)
from tests.conftest import run_async


def _fake_transport(vec: list[float]):
    def transport(path: str, payload: dict) -> dict:
        assert path == "/api/embeddings"
        assert "prompt" in payload
        return {"embedding": vec}

    return transport


def test_ollama_embedder_query_and_documents() -> None:
    """OllamaEmbedder posts to /api/embeddings via the injected transport."""
    emb = OllamaEmbedder(dim=3, transport=_fake_transport([0.1, 0.2, 0.3]))
    assert run_async(emb.embed_query("hi")) == [0.1, 0.2, 0.3]
    docs = run_async(emb.embed_documents(["a", "b"]))
    assert docs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_ollama_embedder_rejects_empty() -> None:
    """An empty embedding from the server is a clear error."""
    emb = OllamaEmbedder(transport=_fake_transport([]))
    with pytest.raises(HimmyError):
        run_async(emb.embed_query("x"))


def test_factory_selects_backends() -> None:
    """build_embedder maps names to embedder types with the right dim."""
    assert isinstance(build_embedder("deterministic", dim=16), DeterministicEmbedder)
    assert build_embedder("deterministic", dim=16).dim == 16
    assert isinstance(build_embedder("ollama", dim=768), OllamaEmbedder)
    assert isinstance(build_embedder("fastembed", dim=384), FastEmbedEmbedder)


def test_factory_unknown_raises() -> None:
    with pytest.raises(HimmyError):
        build_embedder("nope")


def test_default_dims() -> None:
    assert default_dim_for("ollama") == 768
    assert default_dim_for("fastembed") == 384
    assert default_dim_for("deterministic") == 64


def test_fastembed_lazy_import_or_works() -> None:
    """fastembed embeds when installed; otherwise raises a clear extra error."""
    emb = FastEmbedEmbedder(dim=384)
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(HimmyError):
            run_async(emb.embed_query("hello"))
    else:  # pragma: no cover - only when the extra is installed
        vec = run_async(emb.embed_query("hello"))
        assert len(vec) == 384


def test_config_build_embedder_and_dim() -> None:
    """ToolkitConfig builds the configured embedder + dim."""
    from himmy.toolkit.config import ToolkitConfig

    emb, dim = ToolkitConfig(embedder="ollama").build_embedder_and_dim()
    assert isinstance(emb, OllamaEmbedder)
    assert dim == 768
    emb2, dim2 = ToolkitConfig(
        embedder="deterministic", embedder_dim=32
    ).build_embedder_and_dim()
    assert dim2 == 32
