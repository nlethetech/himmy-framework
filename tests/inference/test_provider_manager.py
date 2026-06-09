"""Tests for CLI provider construction of the direct Anthropic/OpenAI managers, plus
the pydantic-ai OpenAIModel→OpenAIChatModel deprecation fix.

All offline: managers are constructed (the SDK is only imported lazily inside
``generate``/``_build_client``), the missing-extra path is simulated by forcing the
lazy import to fail, and the pydantic-ai assembly is checked for no DeprecationWarning.
"""

from __future__ import annotations

import builtins
import warnings
from typing import Any

import pytest

from himmy.cli.provider import PROVIDERS, ProviderError, build_manager_for
from himmy.services.inference.anthropic_manager import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicClientManager,
)
from himmy.services.inference.openai_manager import (
    DEFAULT_OPENAI_MODEL,
    OpenAIClientManager,
)


# -------------------------------------------------------- build_manager_for happy
def test_providers_tuple_includes_direct_providers() -> None:
    """The CLI advertises the new direct providers."""
    assert "anthropic" in PROVIDERS
    assert "openai" in PROVIDERS


def test_build_manager_for_anthropic_default_model() -> None:
    """anthropic without a model uses the sane Claude default."""
    mgr = build_manager_for("anthropic")
    assert isinstance(mgr, AnthropicClientManager)
    assert mgr.resolve("default") == DEFAULT_ANTHROPIC_MODEL
    assert mgr.provider_name == "anthropic"


def test_build_manager_for_anthropic_explicit_model() -> None:
    """anthropic honors an explicit model."""
    mgr = build_manager_for("anthropic", "claude-3-7-sonnet-latest")
    assert isinstance(mgr, AnthropicClientManager)
    assert mgr.resolve("default") == "claude-3-7-sonnet-latest"


def test_build_manager_for_openai_default_model() -> None:
    """openai without a model uses the sane GPT default."""
    mgr = build_manager_for("openai")
    assert isinstance(mgr, OpenAIClientManager)
    assert mgr.resolve("default") == DEFAULT_OPENAI_MODEL
    assert mgr.provider_name == "openai"


def test_build_manager_for_openai_explicit_model() -> None:
    """openai honors an explicit model."""
    mgr = build_manager_for("openai", "gpt-4o")
    assert isinstance(mgr, OpenAIClientManager)
    assert mgr.resolve("default") == "gpt-4o"


# -------------------------------------------------------- missing-extra path
def test_anthropic_missing_extra_raises_clear_error_at_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the anthropic SDK installed, building a client raises a clear ImportError.

    The manager imports the SDK lazily, so construction succeeds; the actionable error
    surfaces when a real client must be built. ``generate`` then normalizes it (never
    raises), so the CLI path stays robust.
    """
    from himmy.services.inference.models import (
        InferenceMessage,
        InferenceRequest,
        InferenceStatus,
    )
    from tests.conftest import run_async

    real_import = builtins.__import__

    def _no_anthropic(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_anthropic)

    mgr = build_manager_for("anthropic")  # construction still works (lazy import)
    # _build_client raises a clear, actionable error...
    with pytest.raises(ImportError, match="himmy\\[anthropic\\]"):
        mgr._build_client()
    # ...and generate normalizes that into a FAILED response (never raises).
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
    resp = run_async(mgr.generate(req))
    assert resp.status == InferenceStatus.FAILED


def test_openai_missing_extra_raises_clear_error_at_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the openai SDK installed, building a client raises a clear ImportError."""
    real_import = builtins.__import__

    def _no_openai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_openai)

    mgr = build_manager_for("openai")
    with pytest.raises(ImportError, match="himmy\\[openai\\]"):
        mgr._build_client()


def test_unknown_provider_raises_provider_error() -> None:
    """An unknown provider name still raises a clear ProviderError listing the options."""
    with pytest.raises(ProviderError):
        build_manager_for("nope")


# ----------------------------------------- pydantic-ai deprecation fix
def test_pydantic_ai_openai_model_emits_no_deprecation_warning() -> None:
    """The base_url path prefers OpenAIChatModel → no OpenAIModel DeprecationWarning."""
    pytest.importorskip("pydantic_ai")

    from himmy.services.inference.pydantic_ai_manager import PydanticAIClientManager

    manager = PydanticAIClientManager(
        {"default": "openai:gpt-4o-mini"},
        default_model="openai:gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-unit",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = manager._infer_model("openai:gpt-4o-mini")

    deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "OpenAIModel" in str(w.message)
    ]
    assert deprecations == []
    # the model still carries the configured provider + name
    assert model.model_name == "gpt-4o-mini"


def test_openai_chat_model_helper_prefers_chat_model() -> None:
    """The helper builds OpenAIChatModel when available (the un-deprecated class)."""
    pytest.importorskip("pydantic_ai")

    from pydantic_ai.providers.openai import OpenAIProvider

    from himmy.services.inference.pydantic_ai_manager import _openai_chat_model

    provider = OpenAIProvider(base_url="https://x/v1", api_key="k")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _openai_chat_model("gpt-4o-mini", provider)

    assert type(model).__name__ == "OpenAIChatModel"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
