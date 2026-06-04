"""Shared pytest fixtures + helpers for the OpenSims test suite.

Tests are plain ``def test_*`` functions that drive async code via the local
:func:`run_async` helper (``asyncio.run``); ``pytest-asyncio`` is not required and
is not assumed installed. Fixtures assemble the offline-first stack: in-memory
storage, an entity registry, stub inference, the runtime, and the application
services — everything the kernels need to be exercised without a network.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
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


# ----------------------------------------------------------------- core builders
@pytest.fixture()
def storage() -> Any:
    """A fresh in-memory :class:`StorageService`."""
    from opensims.services.storage.service import StorageService

    return StorageService()


@pytest.fixture()
def registry() -> Any:
    """A fresh in-memory :class:`EntityRegistry`."""
    from opensims.entities.registry import EntityRegistry

    return EntityRegistry()


@pytest.fixture()
def inference_service(storage: Any) -> Any:
    """An :class:`InferenceService` backed by the offline stub, sinking to storage."""
    from opensims.services.inference.client_manager import StubClientManager
    from opensims.services.inference.service import InferenceService

    return InferenceService(StubClientManager(), event_sink=storage)


@pytest.fixture()
def context_service(storage: Any, registry: Any) -> Any:
    """A :class:`ContextService` over storage + registry (no adapters)."""
    from opensims.services.context.service import ContextService

    return ContextService(storage_service=storage, entity_registry=registry)


@pytest.fixture()
def tool_service() -> Any:
    """A :class:`ToolService` over a fresh registry with no tools registered yet."""
    from opensims.services.tools.registry import ToolRegistry
    from opensims.services.tools.service import ToolService

    return ToolService(ToolRegistry())


@pytest.fixture()
def runtime(
    inference_service: Any,
    storage: Any,
    context_service: Any,
    registry: Any,
) -> Any:
    """A fully-wired :class:`SingleAgentRuntime` (offline, persistent, evidenced)."""
    from opensims.runtime.single_agent import SingleAgentRuntime

    return SingleAgentRuntime(
        inference_service=inference_service,
        memory_store=storage,
        context_service=context_service,
        entity_registry=registry,
    )


@pytest.fixture()
def persona() -> Any:
    """A simple analyst :class:`Persona`."""
    from opensims.agents.personas.persona import Persona

    return Persona(
        name="Analyst",
        description="A careful financial analyst.",
        instructions=["Be precise.", "Cite evidence."],
        metadata={"role": "analyst", "skills": ["valuation"]},
    )


@pytest.fixture()
def task() -> Any:
    """A simple :class:`Task` asking for a brief."""
    from opensims.agents.base_agent.task import Task

    return Task(title="Brief", prompt="Summarize the outlook for ACME.")
