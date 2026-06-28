"""Runtime kernel: a one-call builder for the full offline-first Himmy stack.

Wiring a usable runtime means assembling inference, storage, an entity registry,
context, prompts, and tools by hand — ~10 constructors a new user must discover.
:func:`build_runtime` does it in one call with sensible offline-first defaults, so
the common case is::

    from himmy import build_runtime

    runtime, inference, tools = build_runtime()
    thread = await runtime.run_task(persona, task)

The default inference path is the deterministic, offline
:class:`~himmy.services.inference.client_manager.StubClientManager` (no network,
no keys). A real pydantic-ai-backed manager is selected ONLY when ``pydantic_ai``
is importable AND a provider key is present AND ``HIMMY_EXAMPLES_MODEL`` names a
model. Every collaborator can be overridden via keyword for tests/advanced wiring.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import (
    ClientManager,
    StubClientManager,
)
from himmy.services.inference.service import InferenceService
from himmy.services.observability import configure_observability
from himmy.services.prompts.manager import PromptManager
from himmy.services.prompts.mapper import ContextPromptMapper
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol


def _pydantic_ai_available() -> bool:
    """True only when the optional ``pydantic_ai`` extra is importable."""
    try:
        import pydantic_ai  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - offline default path
        return False
    return True


def _provider_key_present() -> bool:
    """True when a provider API key is configured for the default model."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY")
    )


def _sdk_available(module: str) -> bool:
    """True only when the optional direct-SDK extra (``anthropic``/``openai``) imports."""
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except Exception:  # pragma: no cover - defensive
        return False


def _direct_manager_for(model: str) -> ClientManager | None:
    """Pick a direct-SDK :class:`ClientManager` for an explicitly named model.

    Preferred over :class:`PydanticAIClientManager` when the matching API key is set
    and the SDK extra is installed: a ``claude*``/``anthropic:`` model routes to
    :class:`AnthropicClientManager`, a ``gpt*``/``openai:`` model to
    :class:`OpenAIClientManager`. Returns ``None`` when no direct manager applies, so
    the caller falls back to pydantic-ai or the offline stub. Honours offline-first:
    nothing here imports an SDK or hits the network until the manager is actually used.
    """
    name = model.split(":", 1)[-1] if ":" in model else model
    is_anthropic = model.startswith("anthropic:") or name.startswith("claude")
    is_openai = model.startswith("openai:") or name.startswith(("gpt", "o1", "o3"))

    if (
        is_anthropic
        and os.environ.get("ANTHROPIC_API_KEY")
        and _sdk_available("anthropic")
    ):
        from himmy.services.inference.anthropic_manager import AnthropicClientManager

        return AnthropicClientManager(model=name)
    if is_openai and os.environ.get("OPENAI_API_KEY") and _sdk_available("openai"):
        from himmy.services.inference.openai_manager import OpenAIClientManager

        return OpenAIClientManager(model=name)
    return None


def build_inference() -> InferenceService:
    """Build the default :class:`InferenceService`.

    Resolution order (first match wins), gated on ``HIMMY_EXAMPLES_MODEL`` naming a
    model so the zero-config default stays the offline :class:`StubClientManager`:

    1. A direct provider SDK — :class:`AnthropicClientManager` / :class:`OpenAIClientManager`
       — when the model routes to it, the matching API key is set, and the SDK extra
       (``himmy[anthropic]`` / ``himmy[openai]``) is installed.
    2. :class:`PydanticAIClientManager` when ``pydantic_ai`` is installed and a key is present.
    3. The offline :class:`StubClientManager` (no network, no keys).
    """
    configure_observability()
    manager: ClientManager
    model = os.environ.get("HIMMY_EXAMPLES_MODEL")
    direct = _direct_manager_for(model) if model else None
    if direct is not None:
        manager = direct
    elif model and _pydantic_ai_available() and _provider_key_present():
        from himmy.services.inference.pydantic_ai_manager import (
            PydanticAIClientManager,
        )

        manager = PydanticAIClientManager({"default": model}, default_model=model)
    else:
        manager = StubClientManager()
    return InferenceService(manager)


def build_storage() -> StorageService:
    """Build the one-shot CLI store: a fresh in-memory :class:`StorageService`.

    Delegates to :meth:`StoreFactory.for_cli`, which is always the in-memory backend
    (EventSink + ThreadEventStore) — a ``himmy run`` / ``himmy chat`` stays zero-setup
    and never touches a durable file or DSN. The durable, server-side default lives in
    :class:`~himmy.services.storage.factory.StoreFactory` (``for_server``).
    """
    from himmy.services.storage.factory import StoreFactory

    return StoreFactory.for_cli()


def build_runtime(
    **overrides: Any,
) -> tuple[SingleAgentRuntime, InferenceService, ToolService]:
    """Wire the full offline-first stack in one call.

    Returns ``(runtime, inference, tools)``. Pass ``overrides`` to substitute any
    of ``inference``, ``storage``, ``registry``, ``context_service``,
    ``prompt_manager``, ``context_prompt_mapper``, ``tool_registry``, or
    ``tool_service`` (handy for tests/apps that need a shared instance).

    ``tool_authorizer`` (P0 confused-deputy fix) is the run principal's tool-capability
    gate. When provided it is threaded into the built :class:`ToolService` so the
    deny-by-default capability check bites before tool dispatch — used by the
    team/workflow orchestration path so member tools are gated by the launching
    principal's grants, exactly as the single-agent path is. ``None`` (the default, and
    every offline / zero-config caller) leaves tool dispatch byte-unchanged.
    """
    configure_observability()

    inference: InferenceService = overrides.get("inference") or build_inference()
    storage: StorageService = overrides.get("storage") or build_storage()
    registry: EntityRegistryProtocol = overrides.get("registry") or EntityRegistry()

    context_service: ContextService = overrides.get(
        "context_service"
    ) or ContextService(storage_service=storage, entity_registry=registry)
    prompt_manager: PromptManager = overrides.get("prompt_manager") or PromptManager()
    context_prompt_mapper: ContextPromptMapper = (
        overrides.get("context_prompt_mapper") or ContextPromptMapper()
    )

    tool_registry: ToolRegistry = overrides.get("tool_registry") or ToolRegistry(
        entity_registry=registry
    )
    tool_service: ToolService = overrides.get("tool_service") or ToolService(
        tool_registry,
        event_sink=storage,
        tool_authorizer=overrides.get("tool_authorizer"),
    )

    runtime = SingleAgentRuntime(
        inference_service=inference,
        memory_store=storage,
        tool_service=tool_service,
        context_service=context_service,
        prompt_manager=prompt_manager,
        context_prompt_mapper=context_prompt_mapper,
        entity_registry=registry,
        input_guardrail=overrides.get("input_guardrail"),
        output_guardrail=overrides.get("output_guardrail"),
        on_event=overrides.get("on_event"),
        capture_io=overrides.get("capture_io"),
        checkpoint_store=overrides.get("checkpoint_store"),
    )
    return runtime, inference, tool_service


__all__ = ["build_runtime", "build_storage", "build_inference"]
