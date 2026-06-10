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

PROVIDERS = (
    "stub",
    "claude-cli",
    "ollama",
    "pydantic-ai",
    "openrouter",
    "anthropic",
    "openai",
)

#: OpenRouter is an OpenAI-compatible aggregator; route through the pydantic-ai path.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"


def _default_ollama_model() -> str:
    """Pick a sensible default Ollama model: an INSTALLED chat model, not a tag
    the user may never have pulled.

    ``HIMMY_OLLAMA_MODEL`` overrides; otherwise probe the local server's tags
    (sub-second timeout) and take the first non-embedding model. Falls back to
    ``llama3.2`` when the server is down — the historical default, so offline
    behavior is unchanged.
    """
    import os

    override = os.environ.get("HIMMY_OLLAMA_MODEL", "").strip()
    if override:
        return override
    try:
        import httpx

        base = os.environ.get("HIMMY_OLLAMA_URL", "http://localhost:11434")
        resp = httpx.get(f"{base.rstrip('/')}/api/tags", timeout=0.8)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        for m in models:
            name = str(m.get("name", ""))
            if name and "embed" not in name.lower():
                return name
    except Exception:  # noqa: BLE001 - probing is best-effort; offline keeps the old default
        pass
    return "llama3.2"


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

        return OllamaClientManager(model=model or _default_ollama_model())

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

    if provider == "anthropic":
        try:
            from himmy.services.inference.anthropic_manager import (
                DEFAULT_ANTHROPIC_MODEL,
                AnthropicClientManager,
            )
        except Exception as exc:  # pragma: no cover - optional extra missing
            raise ProviderError(
                "provider 'anthropic' needs the 'anthropic' extra: "
                "pip install 'himmy[anthropic]'"
            ) from exc
        return AnthropicClientManager(model=model or DEFAULT_ANTHROPIC_MODEL)

    if provider == "openai":
        try:
            from himmy.services.inference.openai_manager import (
                DEFAULT_OPENAI_MODEL,
                OpenAIClientManager,
            )
        except Exception as exc:  # pragma: no cover - optional extra missing
            raise ProviderError(
                "provider 'openai' needs the 'openai' extra: "
                "pip install 'himmy[openai]'"
            ) from exc
        return OpenAIClientManager(model=model or DEFAULT_OPENAI_MODEL)

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
