"""The toolkit embedder defaults to the offline-safe ``"auto"`` cascade.

``"auto"`` resolves fastembed → deterministic, **network-free**: it is genuinely
semantic when the local ONNX ``[embeddings]`` extra is installed, yet stays keyless,
network-free, and dependency-free otherwise. Server-backed embedders (Ollama, OpenAI)
need a running service + a pulled/authorised model, so they are *never* auto-selected
(that would 404 / violate offline determinism) and remain explicit opt-ins.
"""

from __future__ import annotations

import pytest

from himmy.services.knowledge import local_embedders
from himmy.toolkit.config import ToolkitConfig


def test_field_default_is_auto() -> None:
    """A bare ``ToolkitConfig()`` uses the ``"auto"`` cascade by default."""
    assert ToolkitConfig().embedder == "auto"


def test_from_env_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``HIMMY_EMBEDDER`` unset, ``from_env`` still selects ``"auto"``."""
    monkeypatch.delenv("HIMMY_EMBEDDER", raising=False)
    assert ToolkitConfig.from_env().embedder == "auto"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``HIMMY_EMBEDDER`` wins over the ``"auto"`` default."""
    monkeypatch.setenv("HIMMY_EMBEDDER", "deterministic")
    assert ToolkitConfig.from_env().embedder == "deterministic"


def test_resolve_auto_prefers_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When fastembed is importable, the cascade resolves to it first."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: True)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    assert local_embedders.resolve_auto_backend() == "fastembed"


def test_resolve_auto_falls_back_to_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no fastembed and no reachable Ollama, the cascade is deterministic."""
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    assert local_embedders.resolve_auto_backend() == "deterministic"


def test_resolve_auto_uses_ollama_only_when_embed_model_pulled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto picks Ollama when its embed model is pulled, else degrades to deterministic.

    The Ollama leg is gated on the embed model being *present* (not merely the server
    being up): a reachable Ollama with no embed model would 404 at embed time, so it
    falls through to the offline deterministic backend instead.
    """
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: True)
    # Embed model pulled → auto selects ollama.
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: True
    )
    assert local_embedders.resolve_auto_backend() == "ollama"
    # Embed model absent → auto degrades to deterministic (no 404).
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: False
    )
    assert local_embedders.resolve_auto_backend() == "deterministic"


def test_auto_builds_a_working_embedder_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embedder="auto"`` builds and sizes the deterministic fallback offline.

    No fastembed, no Ollama → ``build_embedder_and_dim`` must still produce a
    usable embedder and the deterministic dim (64), never crashing on the
    vector-dim difference vs a semantic backend.
    """
    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(local_embedders, "ollama_reachable", lambda *a, **k: False)
    embedder, dim = ToolkitConfig(embedder="auto").build_embedder_and_dim()
    assert dim == 64
    assert hasattr(embedder, "embed_query")
