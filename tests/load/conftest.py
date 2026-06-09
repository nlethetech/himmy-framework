"""Function-scoped fixtures for the load harness (isolated, xdist-safe).

Every fixture here is **function-scoped** so each test gets a fresh app, container,
and synthetic-data factory — no shared mutable state leaks between tests, which keeps
the suite safe under ``pytest-xdist`` (``-n auto``) and free of ordering effects.

The async HTTP surface is driven in-process via httpx ``AsyncClient`` over an
``ASGITransport`` (no socket, no live server). Tests bodies are plain ``def`` that
call :func:`tests.conftest.run_async` on an async coroutine using the
:func:`client_factory` context manager — mirroring the suite-wide ``asyncio.run``
convention rather than depending on a particular pytest-asyncio mode.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from tests.load.fixtures import DEFAULT_SEED, SyntheticDataFactory


@pytest.fixture()
def api_container() -> Any:
    """A fresh offline-first :class:`ApiContainer` (in-memory + stub inference)."""
    from himmy.api import ApiContainer

    return ApiContainer.build_default()


@pytest.fixture()
def app(api_container: Any) -> Any:
    """A fresh in-process Himmy FastAPI app over the offline container.

    No auth header is required: ``HIMMY_INTERNAL_API_KEY`` is unset on the default
    path, so requests are anonymous and authorised across all workspaces.
    """
    from himmy.api import create_app

    return create_app(api_container)


@pytest.fixture()
def client_factory(
    app: Any,
) -> Callable[..., Any]:
    """Return an async context manager yielding an httpx client bound to ``app``.

    Usage inside an async test body::

        async with client_factory() as client:
            resp = await client.get("/health")

    The client speaks to the app in-process via ``ASGITransport`` — no network, no
    running server, no lifespan side effects — so it is deterministic and offline.
    """

    @asynccontextmanager
    async def _make(base_url: str = "http://loadtest") -> AsyncIterator[AsyncClient]:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=base_url) as client:
            yield client

    return _make


@pytest.fixture()
def factory() -> SyntheticDataFactory:
    """A fresh seeded :class:`SyntheticDataFactory` (deterministic graphs)."""
    return SyntheticDataFactory(seed=DEFAULT_SEED)
