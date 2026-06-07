"""Shared pytest fixtures + helpers for the Himmy test suite.

Tests are plain ``def test_*`` functions that drive async code via the local
:func:`run_async` helper (``asyncio.run``); ``pytest-asyncio`` is not required and
is not assumed installed. Fixtures assemble the offline-first stack: in-memory
storage, an entity registry, stub inference, the runtime, and the application
services — everything the kernels need to be exercised without a network.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

# Ensure the repo root is importable when pytest is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Run an awaitable to completion on a fresh event loop (asyncio.run shim)."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def executor_from(
    handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]],
) -> Callable[[str, dict[str, Any]], Awaitable[Any]]:
    """Build a ``ToolExecutor`` from a ``{tool_name: async (args) -> ToolReturnRecord}`` map.

    Mirrors :meth:`ToolService.tool_executor` for tests that build ``BoundTool``s by
    hand and want them to actually execute through the inference path (execution is
    no longer carried on ``BoundTool`` itself).
    """

    async def _exec(name: str, args: dict[str, Any]) -> Any:
        return await handlers[name](args)

    return _exec


# ----------------------------------------------------------------- core builders
@pytest.fixture()
def storage() -> Any:
    """A fresh in-memory :class:`StorageService`."""
    from himmy.services.storage.service import StorageService

    return StorageService()


@pytest.fixture()
def registry() -> Any:
    """A fresh in-memory :class:`EntityRegistry`."""
    from himmy.entities.registry import EntityRegistry

    return EntityRegistry()


@pytest.fixture()
def inference_service(storage: Any) -> Any:
    """An :class:`InferenceService` backed by the offline stub, sinking to storage."""
    from himmy.services.inference.client_manager import StubClientManager
    from himmy.services.inference.service import InferenceService

    return InferenceService(StubClientManager(), event_sink=storage)


@pytest.fixture()
def context_service(storage: Any, registry: Any) -> Any:
    """A :class:`ContextService` over storage + registry (no adapters)."""
    from himmy.services.context.service import ContextService

    return ContextService(storage_service=storage, entity_registry=registry)


@pytest.fixture()
def tool_service() -> Any:
    """A :class:`ToolService` over a fresh registry with no tools registered yet."""
    from himmy.services.tools.registry import ToolRegistry
    from himmy.services.tools.service import ToolService

    return ToolService(ToolRegistry())


@pytest.fixture()
def runtime(
    inference_service: Any,
    storage: Any,
    context_service: Any,
    registry: Any,
) -> Any:
    """A fully-wired :class:`SingleAgentRuntime` (offline, persistent, evidenced)."""
    from himmy.runtime.single_agent import SingleAgentRuntime

    return SingleAgentRuntime(
        inference_service=inference_service,
        memory_store=storage,
        context_service=context_service,
        entity_registry=registry,
    )


@pytest.fixture()
def persona() -> Any:
    """A simple analyst :class:`Persona`."""
    from himmy.agents.personas.persona import Persona

    return Persona(
        name="Analyst",
        description="A careful financial analyst.",
        instructions=["Be precise.", "Cite evidence."],
        metadata={"role": "analyst", "skills": ["valuation"]},
    )


@pytest.fixture()
def task() -> Any:
    """A simple :class:`Task` asking for a brief."""
    from himmy.agents.base_agent.task import Task

    return Task(title="Brief", prompt="Summarize the outlook for ACME.")
