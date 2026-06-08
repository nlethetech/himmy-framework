"""Provider selection for the CLI: turn a ``--provider``/``--model`` pair into an
:class:`~himmy.services.inference.service.InferenceService`.

The default (``provider is None``) delegates to the package's
:func:`~himmy.runtime.builder.build_inference`, preserving its auto behavior: a real
pydantic-ai manager when a provider key + the ``providers`` extra + a model are present,
otherwise the offline deterministic stub. Explicit providers let a user opt into the
local ``claude`` CLI (Claude Max), a local Ollama server, or force the stub. Heavier
managers are imported lazily so the common paths stay light and offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.core import HimmyError
from himmy.services.inference.service import InferenceService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.client_manager import ClientManager

PROVIDERS = ("stub", "claude-cli", "ollama", "pydantic-ai", "openrouter")

#: OpenRouter is an OpenAI-compatible aggregator; route through the pydantic-ai path.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"


class ProviderError(HimmyError):
    """Raised when a requested provider cannot be constructed (e.g. missing extra)."""


def build_manager_for(
    provider: str | None = None, model: str | None = None
) -> ClientManager:
    """Build the raw :class:`ClientManager` for a provider (used by the multiplexer).

    ``provider=None`` returns the framework's auto-selected manager (pydantic-ai when a
    key + the ``providers`` extra + a model are present, otherwise the offline stub).
    """
    if provider is None:
        from himmy.runtime.builder import build_inference

        return build_inference()._client_manager

    if provider == "stub":
        from himmy.services.inference.client_manager import StubClientManager

        return StubClientManager()

    if provider == "claude-cli":
        from himmy.services.inference.local import ClaudeCliClientManager

        return ClaudeCliClientManager(model=model or "haiku")

    if provider == "ollama":
        from himmy.services.inference.local import OllamaClientManager

        return OllamaClientManager(model=model) if model else OllamaClientManager()

    if provider == "pydantic-ai":
        try:
            from himmy.services.inference.pydantic_ai_manager import (
                PydanticAIClientManager,
            )
        except Exception as exc:  # pragma: no cover - optional extra missing
            raise ProviderError(
                "provider 'pydantic-ai' needs the 'providers' extra: "
                "pip install 'himmy[providers]'"
            ) from exc
        if model:
            return PydanticAIClientManager({"default": model}, default_model=model)
        return PydanticAIClientManager()

    if provider == "openrouter":
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError(
                "provider 'openrouter' needs OPENROUTER_API_KEY in the environment "
                "(get a key at https://openrouter.ai/keys)."
            )
        try:
            from himmy.services.inference.pydantic_ai_manager import (
                PydanticAIClientManager,
            )
        except Exception as exc:  # pragma: no cover - optional extra missing
            raise ProviderError(
                "provider 'openrouter' needs the 'providers' extra: "
                "pip install 'himmy[providers]'"
            ) from exc
        chosen = model if model and model != "default" else OPENROUTER_DEFAULT_MODEL
        model_string = f"openai:{chosen}"
        return PydanticAIClientManager(
            {"default": model_string},
            default_model=model_string,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            provider_name="openrouter",
        )

    raise ProviderError(
        f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
    )


def build_inference_for(
    provider: str | None = None, model: str | None = None
) -> InferenceService:
    """Build an :class:`InferenceService` for the requested provider.

    ``provider=None`` keeps the framework default (pydantic-ai→stub auto-select).
    """
    return InferenceService(build_manager_for(provider, model))


def is_stub_manager(manager: ClientManager) -> bool:
    """True when ``manager`` is the offline deterministic stub (no real model)."""
    from himmy.services.inference.client_manager import StubClientManager

    return isinstance(manager, StubClientManager)


def resolves_to_stub(provider: str | None = None, model: str | None = None) -> bool:
    """True when this provider/model pair will run on the offline stub.

    Used by the CLI to surface a one-line "you're offline, here's how to use a real
    model" hint. Returns ``False`` if the manager can't be constructed (let the real
    run raise the actionable error instead of guessing here).
    """
    try:
        return is_stub_manager(build_manager_for(provider, model))
    except Exception:  # noqa: BLE001 - never let a hint probe break the command
        return False
