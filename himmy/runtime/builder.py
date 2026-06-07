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
from typing import Any

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


def build_inference() -> InferenceService:
    """Build the default :class:`InferenceService`.

    Uses :class:`PydanticAIClientManager` only when ``pydantic_ai`` is installed
    AND a provider key is present AND ``HIMMY_EXAMPLES_MODEL`` names a model;
    otherwise defaults to the offline :class:`StubClientManager`.
    """
    configure_observability()
    manager: ClientManager
    model = os.environ.get("HIMMY_EXAMPLES_MODEL")
    if model and _pydantic_ai_available() and _provider_key_present():
        from himmy.services.inference.pydantic_ai_manager import (
            PydanticAIClientManager,
        )

        manager = PydanticAIClientManager({"default": model}, default_model=model)
    else:
        manager = StubClientManager()
    return InferenceService(manager)


def build_storage() -> StorageService:
    """Build a fresh in-memory :class:`StorageService` (EventSink + ThreadEventStore)."""
    return StorageService()


def build_runtime(
    **overrides: Any,
) -> tuple[SingleAgentRuntime, InferenceService, ToolService]:
    """Wire the full offline-first stack in one call.

    Returns ``(runtime, inference, tools)``. Pass ``overrides`` to substitute any
    of ``inference``, ``storage``, ``registry``, ``context_service``,
    ``prompt_manager``, ``context_prompt_mapper``, ``tool_registry``, or
    ``tool_service`` (handy for tests/apps that need a shared instance).
    """
    configure_observability()

    inference: InferenceService = overrides.get("inference") or build_inference()
    storage: StorageService = overrides.get("storage") or build_storage()
    registry: EntityRegistry = overrides.get("registry") or EntityRegistry()

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
        tool_registry, event_sink=storage
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
