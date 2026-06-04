"""Examples kernel: shared offline-first runtime builder for every example script.

Provides :func:`build_runtime` (plus :func:`build_storage` and
:func:`build_inference` helpers) that wire the full in-memory OpenSims stack —
storage, entity registry, context service, prompt manager/mapper, tool service —
around an :class:`~opensims.services.inference.service.InferenceService`.

The default inference path is the deterministic, offline
:class:`~opensims.services.inference.client_manager.StubClientManager` (no
network, no keys). A real pydantic-ai-backed manager is selected ONLY when
``pydantic_ai`` is importable AND a provider key is present via
``OPENSIMS_EXAMPLES_MODEL`` (or ``OPENROUTER_API_KEY``). Observability is
configured (no-op unless ``OPENSIMS_LOGFIRE_ENABLED`` is set).
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Ensure the repo root (parent of ``examples/``) is importable so the examples
# run with a plain ``python examples/0N_*.py`` even without ``pip install -e .``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opensims.entities.registry import EntityRegistry
from opensims.runtime.single_agent import SingleAgentRuntime
from opensims.services.context.service import ContextService
from opensims.services.inference.client_manager import (
    ClientManager,
    StubClientManager,
)
from opensims.services.inference.service import InferenceService
from opensims.services.observability import configure_observability
from opensims.services.prompts.manager import PromptManager
from opensims.services.prompts.mapper import ContextPromptMapper
from opensims.services.storage.service import StorageService
from opensims.services.tools.registry import ToolRegistry
from opensims.services.tools.service import ToolService


def _pydantic_ai_available() -> bool:
    """True only when the optional ``pydantic_ai`` extra is importable."""
    try:
        import pydantic_ai  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - offline default path
        return False
    return True


def _provider_key_present() -> bool:
    """True when a provider API key is configured for the examples model."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY")
    )


def build_inference() -> InferenceService:
    """Build the example :class:`InferenceService`.

    Uses :class:`PydanticAIClientManager` only when ``pydantic_ai`` is installed
    AND a provider key is present AND ``OPENSIMS_EXAMPLES_MODEL`` names a model;
    otherwise defaults to the offline :class:`StubClientManager`.
    """
    configure_observability()
    manager: ClientManager
    model = os.environ.get("OPENSIMS_EXAMPLES_MODEL")
    if model and _pydantic_ai_available() and _provider_key_present():
        from opensims.services.inference.pydantic_ai_manager import (
            PydanticAIClientManager,
        )

        manager = PydanticAIClientManager({"default": model}, default_model=model)
    else:
        manager = StubClientManager()
    return InferenceService(manager)


def build_storage() -> StorageService:
    """Build a fresh in-memory :class:`StorageService` (EventSink + MemoryStore)."""
    return StorageService()


def build_runtime(
    **overrides: Any,
) -> tuple[SingleAgentRuntime, InferenceService, ToolService]:
    """Wire the full offline-first example stack.

    Returns ``(runtime, inference, tools)``. Pass ``overrides`` to substitute any
    of ``inference``, ``storage``, ``registry``, ``context_service``,
    ``prompt_manager``, ``context_prompt_mapper``, ``tool_registry``, or
    ``tool_service`` (handy for tests/examples that need a shared instance).
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
    )
    return runtime, inference, tool_service


__all__ = ["build_runtime", "build_storage", "build_inference"]
