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

from himmy.core import HimmyError
from himmy.services.inference.service import InferenceService

PROVIDERS = ("stub", "claude-cli", "ollama", "pydantic-ai")


class ProviderError(HimmyError):
    """Raised when a requested provider cannot be constructed (e.g. missing extra)."""


def build_inference_for(
    provider: str | None = None, model: str | None = None
) -> InferenceService:
    """Build an :class:`InferenceService` for the requested provider.

    ``provider=None`` keeps the framework default (pydantic-ai→stub auto-select).
    """
    if provider is None:
        from himmy.runtime.builder import build_inference

        return build_inference()

    if provider == "stub":
        from himmy.services.inference.client_manager import StubClientManager

        return InferenceService(StubClientManager())

    if provider == "claude-cli":
        from himmy.services.inference.local import ClaudeCliClientManager

        return InferenceService(ClaudeCliClientManager(model=model or "haiku"))

    if provider == "ollama":
        from himmy.services.inference.local import OllamaClientManager

        manager = OllamaClientManager(model=model) if model else OllamaClientManager()
        return InferenceService(manager)

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
            pyd_manager = PydanticAIClientManager(
                {"default": model}, default_model=model
            )
        else:
            pyd_manager = PydanticAIClientManager()
        return InferenceService(pyd_manager)

    raise ProviderError(
        f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
    )
