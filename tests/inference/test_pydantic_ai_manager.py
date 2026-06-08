"""Unit tests for PydanticAIClientManager's OpenAI-compatible (base_url+api_key) path.

These exercise the manager's model assembly only — no network. They are gated on the
``[providers]`` extra (``pydantic_ai``) since ``_infer_model`` imports it at call time.
The OpenRouter provider is built on exactly this path (base_url + explicit api_key).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from himmy.services.inference.pydantic_ai_manager import (  # noqa: E402
    PydanticAIClientManager,
)


def test_base_url_and_api_key_reach_openai_provider() -> None:
    """base_url + api_key are plumbed into the OpenAIProvider behind the model."""
    manager = PydanticAIClientManager(
        {"default": "openai:some-model"},
        default_model="openai:some-model",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-unit",
        provider_name="openrouter",
    )
    model = manager._infer_model("openai:mistralai/mistral-small-3.2-24b-instruct")

    # The returned OpenAIModel carries an OpenAIProvider configured with our values.
    provider = model._provider
    # base_url is surfaced directly on the provider (OpenAI normalizes a trailing /).
    assert str(provider.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    # The api_key reaches the underlying client (not read from the env).
    assert provider.client.api_key == "sk-or-unit"
    # The '/'-bearing slug survives the 'openai:' split as the model name.
    assert model.model_name == "mistralai/mistral-small-3.2-24b-instruct"


def test_api_key_forwarded_via_monkeypatched_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the exact kwargs handed to OpenAIProvider (base_url + api_key)."""
    captured: dict[str, object] = {}

    import pydantic_ai.providers.openai as openai_provider_mod

    real_provider = openai_provider_mod.OpenAIProvider

    def _spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_provider(*args, **kwargs)

    monkeypatch.setattr(openai_provider_mod, "OpenAIProvider", _spy)

    manager = PydanticAIClientManager(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-spy",
    )
    manager._infer_model("openai:foo")

    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "sk-or-spy"


def test_api_key_defaults_none_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compat: no api_key arg → api_key=None passed (provider reads env)."""
    captured: dict[str, object] = {}

    import pydantic_ai.providers.openai as openai_provider_mod

    real_provider = openai_provider_mod.OpenAIProvider

    def _spy(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_provider(*args, **kwargs)

    monkeypatch.setattr(openai_provider_mod, "OpenAIProvider", _spy)

    manager = PydanticAIClientManager(base_url="https://gateway.example/v1")
    assert manager._api_key is None
    manager._infer_model("openai:foo")
    assert captured["api_key"] is None
