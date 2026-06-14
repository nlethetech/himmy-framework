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


@pytest.fixture(autouse=True)
def _isolated_spine(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the canonical durable spine at a per-test temp file (T2.1 isolation).

    ``SpineFactory`` resolves ONE durable ``.himmy/spine.db`` so the CLI, Studio, and /v1
    share a spine across processes — the right production behaviour, but it means two
    tests in the same working directory would otherwise accumulate each other's
    security/lineage rows in one on-disk file (the spine used to be a fresh in-memory
    ``EntityRegistry()`` per ``build_default``). Setting ``HIMMY_SPINE_PATH`` to a unique
    temp file per test restores per-test isolation WITHOUT weakening the durable default:
    the production resolution is exercised end-to-end (a real on-disk SQLite spine), just
    scoped to a throwaway path. Tests that want a specific path still override the env
    after this fixture (their ``setenv`` wins). Skipped for ``integration`` tests, which
    own their environment.
    """
    if request.node.get_closest_marker("integration"):
        return
    spine_db = tmp_path_factory.mktemp("spine") / "spine.db"
    monkeypatch.setenv("HIMMY_SPINE_PATH", str(spine_db))


@pytest.fixture(autouse=True)
def _isolated_keyvault(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the durable SubjectKeyVault at a per-test temp file (S2 isolation).

    The vault used to be an in-RAM ``dict`` rebuilt per ``build_default``, so a governed
    container test that crypto-shred a subject (``ledger.withdraw`` -> ``erase_subject``)
    left no trace for the next test. Now the vault persists each subject's KEK to the
    canonical ``.himmy/keyvault.db`` AND a shred is a DURABLE tombstone (the whole point of
    S2) — so without isolation a test that shreds ``subject_X`` would make a LATER test
    re-using that subject id raise on key access. Setting ``HIMMY_KEYVAULT_PATH`` to a unique
    temp file per test restores isolation while still exercising the real durable path (a
    throwaway on-disk SQLite vault). Tests that want a specific path override the env after
    this fixture (their ``setenv`` wins). Skipped for ``integration`` tests.
    """
    if request.node.get_closest_marker("integration"):
        return
    keyvault_db = tmp_path_factory.mktemp("keyvault") / "keyvault.db"
    monkeypatch.setenv("HIMMY_KEYVAULT_PATH", str(keyvault_db))


@pytest.fixture(autouse=True)
def _hermetic_embedder_cascade(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the unit suite offline + deterministic regardless of ambient services.

    The ``"auto"`` embedder cascade prefers fastembed, then a local Ollama that has the
    embed model pulled, then deterministic. On a developer box with Ollama running and an
    embed model pulled, ``"auto"`` would otherwise route the entire knowledge/memory suite
    through a live model — slow and environment-dependent. Force the cascade to the offline
    deterministic backend for every test EXCEPT those marked ``integration`` (which exercise
    live Ollama on purpose). Tests that specifically assert cascade behaviour re-patch these
    probes themselves and win, because their ``setattr`` runs after this fixture's.
    """
    if request.node.get_closest_marker("integration"):
        return
    from himmy.services.knowledge import local_embedders

    monkeypatch.setattr(local_embedders, "fastembed_available", lambda: False)
    monkeypatch.setattr(
        local_embedders, "ollama_embed_model_available", lambda *a, **k: False
    )


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
