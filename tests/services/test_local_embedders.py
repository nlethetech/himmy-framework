"""Tests for the real local embedders + the embedder factory (offline)."""

from __future__ import annotations

import pytest

from himmy.core import HimmyError
from himmy.services.knowledge import local_embedders
from himmy.services.knowledge.embedder import DeterministicEmbedder
from himmy.services.knowledge.local_embedders import (
    FastEmbedEmbedder,
    OllamaEmbedder,
    build_embedder,
    default_dim_for,
    resolve_auto_backend,
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


# ---- "auto": prefer a real local embedder, fall back to deterministic ----


def test_auto_prefers_fastembed_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When fastembed is importable, "auto" selects the fastembed backend (no network)."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)
    # The Ollama probe must never run when fastembed already won the selection.
    monkeypatch.setattr(
        local_embedders,
        "ollama_reachable",
        lambda *a, **k: pytest.fail("ollama probe should not run when fastembed wins"),
    )
    assert resolve_auto_backend() == "fastembed"
    emb = build_embedder("auto")
    assert isinstance(emb, FastEmbedEmbedder)
    assert emb.dim == 384  # fastembed's native dim, not the deterministic default


def test_auto_prefers_ollama_when_reachable_and_no_fastembed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fastembed but a reachable local Ollama -> "auto" picks the Ollama backend."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    assert resolve_auto_backend() == "ollama"
    emb = build_embedder("auto")
    assert isinstance(emb, OllamaEmbedder)
    assert emb.dim == 768


def test_auto_falls_back_to_deterministic_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fastembed and no reachable Ollama -> "auto" degrades to deterministic."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    assert resolve_auto_backend() == "deterministic"
    emb = build_embedder("auto")
    assert isinstance(emb, DeterministicEmbedder)
    # An offline embedder still embeds with no network.
    assert len(run_async(emb.embed_query("hello world"))) == 64


def test_auto_default_dim_tracks_resolved_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_dim_for("auto") reports the dim of the backend it actually resolves to."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)
    assert default_dim_for("auto") == 384
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    assert default_dim_for("auto") == 64


def test_config_auto_resolves_embedder_and_matching_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ToolkitConfig(embedder="auto") builds a real embedder + a matching dim."""
    from himmy.toolkit.config import ToolkitConfig

    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    emb, dim = ToolkitConfig(embedder="auto").build_embedder_and_dim()
    assert isinstance(emb, OllamaEmbedder)
    assert dim == 768  # the dim matches the resolved backend, not the "auto" alias

    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    emb2, dim2 = ToolkitConfig(embedder="auto").build_embedder_and_dim()
    assert isinstance(emb2, DeterministicEmbedder)
    assert dim2 == 64


def test_ollama_reachable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama probe returns False on any transport error (never raises)."""
    import httpx

    def _boom(*_a: object, **_k: object) -> object:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    # An unreachable server must fail closed, not propagate the connection error.
    assert local_embedders.ollama_reachable("http://localhost:11434") is False
