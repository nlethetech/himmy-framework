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

PROVIDERS = ("stub", "claude-cli", "ollama", "pydantic-ai")


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
