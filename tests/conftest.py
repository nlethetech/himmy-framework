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
def _reset_auto_backend_cache() -> Any:
    """Clear the per-process auto-backend decision memo around EVERY test.

    :func:`~himmy.services.knowledge.local_embedders.resolve_auto_backend` memoises its
    ``fastembed``/``ollama``/``deterministic`` decision per Ollama base_url for the process
    lifetime (a latency win: ``build_runtime_for_spec`` calls it several times per turn but
    the blocking HTTP probes then fire at most once). In a shared test process that memo is
    otherwise sticky — a test that monkeypatches the probes to force ``ollama`` would pin
    that decision for every later ``"auto"`` build. Resetting in setup AND teardown keeps
    each test's backend resolution hermetic and order-independent. (Production is unaffected:
    one process, one real environment, one stable decision.)
    """
    from himmy.services.knowledge.local_embedders import reset_auto_backend_cache

    reset_auto_backend_cache()
    yield
    reset_auto_backend_cache()


@pytest.fixture(autouse=True)
def _reset_ambient_auth_flag() -> Any:
    """Reset the process-global tool-gate fail-closed flag around EVERY test.

    ``himmy.services.tools.ambient._auth_configured`` is a deliberate PROCESS-level posture
    (set once at ``create_app`` via :func:`mark_auth_configured` when an authenticator is
    wired): when ``True``, a tool reaching the chokepoint with no authorizer in scope FAILS
    CLOSED. That is exactly right for a single live server, but it makes the flag sticky in a
    shared test process — once ANY API-builder test constructs an app WITH an authenticator,
    the flag stays ``True`` and every SUBSEQUENT offline tool call (runtime/tools suites that
    build a ``ToolService`` with no authorizer) would deny-by-default instead of the
    byte-unchanged pure-pass, turning a monolithic ``pytest tests/`` run order-dependent.

    Resetting to ``False`` in setup AND teardown keeps each test's posture hermetic: the
    offline default for tests that never configure auth, and a clean slate for the next test
    after one that did. The dedicated ambient suite already resets it; this generalises that
    so the whole suite is order-independent. (Production is unaffected — one process, one
    ``create_app``, one fixed posture.)
    """
    from himmy.services.tools.ambient import mark_auth_configured

    mark_auth_configured(False)
    yield
    mark_auth_configured(False)


# Guard-focused suites that deliberately drive the cross-site / CSRF middleware with
# bespoke (or absent) Origin/Referer/Host headers — they must see the RAW guard
# behaviour, so the same-origin default below is NOT injected for them.
_GUARD_TEST_FILES = (
    "test_csrf_origin_guard.py",
    "test_studio_guard.py",
    "test_v1_guard.py",
)


@pytest.fixture(autouse=True)
def _testclient_same_origin(request: pytest.FixtureRequest) -> Any:
    """Make Starlette ``TestClient`` requests look like a real same-origin browser call.

    The Studio/``/v1`` cross-site guard (``_install_studio_guard``) fails CLOSED on a
    guarded *mutating* request (POST/PUT/PATCH/DELETE) that carries no ``Origin``/
    ``Referer`` — correct CSRF hardening, because a same-origin browser fetch ALWAYS
    sends ``Origin``. ``TestClient`` (a server-to-server httpx client) omits it, so
    without this fixture every API test that POSTs to a guarded route would 403.

    A real same-origin request from the Studio SPA sends ``Origin: <app origin>``; the
    TestClient's base origin is ``http://testserver`` (Host ``testserver`` is already in
    the guard's allow-list). We inject that as a DEFAULT header only when the caller did
    not set one of its own — so any test that explicitly passes an ``origin``/``referer``
    (incl. the cross-origin / opaque-Origin negative cases) keeps full control and the
    guard's real decision is exercised unchanged. The guard-focused suites above opt out
    entirely so their no-header / bespoke-header cases stay raw.
    """
    fspath = str(request.node.fspath)
    if any(fspath.endswith(name) for name in _GUARD_TEST_FILES):
        yield
        return

    # Starlette's TestClient subclasses a (possibly vendored, e.g. ``httpx2``) httpx
    # ``Client``; patch the single ``build_request`` choke point it inherits so the
    # default Origin is injected for EVERY call path (``.request`` AND ``.stream``,
    # which the SSE run/team/research tests use) regardless of construction site.
    from starlette.testclient import TestClient as _TestClient

    client_cls = _TestClient.__mro__[1]  # the underlying httpx Client base
    _orig_build_request = client_cls.build_request

    def _build_request_with_origin(
        self: Any, method: str, url: Any, **kwargs: Any
    ) -> Any:
        request = _orig_build_request(self, method, url, **kwargs)
        # Only patch the in-process ASGI transport used by TestClient (base_url
        # http://testserver); never touch a real network client. A same-origin browser
        # fetch always sends Origin, so default it when the caller set neither header.
        base = str(getattr(self, "base_url", "") or "")
        if base.startswith("http://testserver"):
            if "origin" not in request.headers and "referer" not in request.headers:
                request.headers["origin"] = "http://testserver"
        return request

    client_cls.build_request = _build_request_with_origin  # type: ignore[method-assign]
    try:
        yield
    finally:
        client_cls.build_request = _orig_build_request  # type: ignore[method-assign]


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
