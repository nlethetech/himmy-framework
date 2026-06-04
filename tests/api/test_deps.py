"""Tests for the API dependency container (AAEO-2 env-driven persistence)."""

from __future__ import annotations

from opensims.api.deps import ApiContainer
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


def test_build_default_is_in_memory_and_offline() -> None:
    """build_default always uses the in-memory store (offline-green)."""
    container = ApiContainer.build_default()
    assert isinstance(container.storage, StorageService)
    # The evaluation service is wired so the eval router/dashboard work.
    assert container.evaluation is not None


def test_build_default_async_falls_back_to_memory_without_dsn(monkeypatch) -> None:
    """Without OPENSIMS_DATABASE_URL, the async builder uses the in-memory store."""
    monkeypatch.delenv("OPENSIMS_DATABASE_URL", raising=False)
    container = run_async(ApiContainer.build_default_async())
    assert isinstance(container.storage, StorageService)


def test_build_storage_selects_postgres_when_dsn_set(monkeypatch) -> None:
    """A set DSN routes _build_storage to PostgresStorageService (gated on the extra).

    Without asyncpg/a DB the connect() raises OpenSimsError — proving the Postgres
    path is selected by env, not silently ignored.
    """
    import pytest

    from opensims.core.errors import OpenSimsError

    monkeypatch.setenv("OPENSIMS_DATABASE_URL", "postgresql://localhost/none")
    try:
        import asyncpg  # type: ignore  # noqa: F401

        has_asyncpg = True
    except ImportError:
        has_asyncpg = False

    if has_asyncpg:
        pytest.skip("asyncpg installed; cannot assert the missing-extra path offline")

    with pytest.raises(OpenSimsError):
        run_async(ApiContainer._build_storage())


def test_container_aclose_is_safe() -> None:
    """aclose tolerates an in-memory store with no close()."""
    container = ApiContainer.build_default()
    run_async(container.aclose())  # no exception
