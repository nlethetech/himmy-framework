"""Provider selection for the CLI: turn a ``--provider``/``--model`` pair into an
:class:`~himmy.services.inference.service.InferenceService`.

The default (``provider is None``) delegates to the package's
:func:`~himmy.runtime.builder.build_inference`, preserving its auto behavior: a real
pydantic-ai manager when a provider key + the ``providers`` extra + a model are present,
otherwise the offline deterministic stub. Explicit providers let a user opt into the
local ``claude`` CLI (Claude Max), a local Ollama server, the direct Anthropic/OpenAI
SDKs, or any OpenAI-compatible endpoint — ``openrouter`` and the generic
``openai-compatible`` (configurable ``base_url`` + secrets-layer key) cover OpenRouter,
Sarvam, Groq, Together, and friends through one code path. Heavier managers are imported
lazily so the common paths stay light and offline.
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
    "himalayagpt",
    "pydantic-ai",
    "openrouter",
    "openai-compatible",
    "anthropic",
    "openai",
)

#: OpenRouter is an OpenAI-compatible aggregator; talk to it through the direct
#: OpenAI-compatible manager (one code path serves OpenRouter / Sarvam / Groq / Together).
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


#: Default model id for the in-house HimalayaGPT provider (HF repo id + GGUF basename).
HIMALAYAGPT_DEFAULT_MODEL = "himalaya-ai/himalayagpt-0.5b-it"


def _build_himalayagpt_manager(model: str | None) -> ClientManager:
    """Build the HimalayaGPT-0.5b manager backed by a local ``llama-server`` bridge.

    The bridge attaches to a server at ``HIMMY_HIMALAYA_BASE_URL`` (e.g.
    ``http://127.0.0.1:8081``) when set, otherwise spawns and owns its own from the
    nanochat llama.cpp fork (``HIMMY_HIMALAYA_SERVER_BIN`` + ``HIMMY_HIMALAYA_MODEL_PATH``,
    Metal-safe ``-ngl 11``). Tunables: ``HIMMY_HIMALAYA_MAX_TOKENS`` (reply cap).

    A missing binary/GGUF or an unreachable server is surfaced as an actionable
    :class:`ProviderError` (the bridge's ``start`` raises before the first turn) so the
    operator sees the runbook hint instead of a per-turn apology.
    """
    import os

    from himmy.services.inference.himalaya_bridge import HimalayaGptFastBridge
    from himmy.services.inference.local import HimalayaGptClientManager

    base_url = (os.environ.get("HIMMY_HIMALAYA_BASE_URL") or "").strip() or None
    kwargs: dict[str, object] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if server_bin := (os.environ.get("HIMMY_HIMALAYA_SERVER_BIN") or "").strip():
        kwargs["server_bin"] = server_bin
    if model_path := (os.environ.get("HIMMY_HIMALAYA_MODEL_PATH") or "").strip():
        kwargs["model_path"] = model_path
    bridge = HimalayaGptFastBridge(**kwargs)  # type: ignore[arg-type]
    try:
        bridge.start()
    except Exception as exc:  # noqa: BLE001 - surface as an actionable provider error
        raise ProviderError(
            "provider 'himalayagpt' could not reach a llama-server: "
            f"{exc}. Either start one (see experiments/himalayagpt/"
            "FAST_RUNTIME_RUNBOOK.md) and set HIMMY_HIMALAYA_BASE_URL, or build the "
            "GGUF + nanochat llama.cpp fork at ~/himalaya-runtime/ so the bridge can "
            "spawn its own."
        ) from exc
    try:
        max_tokens = int(os.environ.get("HIMMY_HIMALAYA_MAX_TOKENS") or "256")
    except ValueError:
        max_tokens = 256
    return HimalayaGptClientManager(
        model_name=model or HIMALAYAGPT_DEFAULT_MODEL,
        generate_fn=bridge.generate_fn(max_new_tokens=max_tokens, temperature=0.0),
    )


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

    if provider == "himalayagpt":
        return _build_himalayagpt_manager(model)

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
        chosen = model if model and model != "default" else OPENROUTER_DEFAULT_MODEL
        return _build_openai_compatible(
            provider_name="openrouter",
            model=chosen,
            base_url=OPENROUTER_BASE_URL,
            key_names=("OPENROUTER_API_KEY",),
            default_headers=_openrouter_headers(),
            missing_key_hint=(
                "provider 'openrouter' needs OPENROUTER_API_KEY "
                "(get a key at https://openrouter.ai/keys)."
            ),
        )

    if provider == "openai-compatible":
        import os

        base_url = (
            os.environ.get("HIMMY_OPENAI_COMPAT_BASE_URL")
            or os.environ.get("OPENAI_COMPAT_BASE_URL")
            or ""
        ).strip()
        if not base_url:
            raise ProviderError(
                "provider 'openai-compatible' needs a base_url: set "
                "HIMMY_OPENAI_COMPAT_BASE_URL to your endpoint "
                "(e.g. https://api.groq.com/openai/v1, https://api.sarvam.ai/v1, "
                "https://api.together.xyz/v1)."
            )
        if not model or model == "default":
            raise ProviderError(
                "provider 'openai-compatible' needs an explicit --model "
                "(the upstream model id, passed through verbatim)."
            )
        return _build_openai_compatible(
            provider_name="openai-compatible",
            model=model,
            base_url=base_url,
            key_names=(
                "HIMMY_OPENAI_COMPAT_API_KEY",
                "OPENAI_COMPAT_API_KEY",
                "OPENAI_API_KEY",
            ),
            default_headers=None,
            missing_key_hint=(
                "provider 'openai-compatible' needs an API key: set "
                "HIMMY_OPENAI_COMPAT_API_KEY (or OPENAI_API_KEY) to the endpoint's key."
            ),
        )

    raise ProviderError(
        f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
    )


def _openrouter_headers() -> dict[str, str]:
    """OpenRouter's OPTIONAL attribution headers, read from env (never required).

    OpenRouter uses ``HTTP-Referer`` / ``X-Title`` to attribute traffic on its
    dashboard. We forward them only when the operator sets them — an unset env var
    sends no header (blank values are dropped by the manager).
    """
    import os

    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_X_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _build_openai_compatible(
    *,
    provider_name: str,
    model: str,
    base_url: str,
    key_names: tuple[str, ...],
    default_headers: dict[str, str] | None,
    missing_key_hint: str,
) -> ClientManager:
    """Build a direct :class:`OpenAIClientManager` against an OpenAI-compatible endpoint.

    The API key is resolved through the SECRETS layer (env/file/vault/cloud), trying
    ``key_names`` in order — never read straight from ``os.environ`` and never logged.
    ``model`` passes through verbatim (e.g. ``anthropic/claude-3.5-sonnet`` or a Sarvam
    model id). Raises a clear, key-redacting :class:`ProviderError` when no key is set.
    """
    from himmy.config.secrets import get_secret

    api_key: str | None = None
    for name in key_names:
        api_key = get_secret(name)
        if api_key:
            break
    if not api_key:
        raise ProviderError(missing_key_hint)

    try:
        from himmy.services.inference.openai_manager import OpenAIClientManager
    except Exception as exc:  # pragma: no cover - optional extra missing
        raise ProviderError(
            f"provider {provider_name!r} needs the 'openai' extra: "
            "pip install 'himmy[openai]'"
        ) from exc

    return OpenAIClientManager(
        model=model,
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
        provider_name=provider_name,
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
