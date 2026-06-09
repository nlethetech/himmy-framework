"""Tests for the API dependency container (AAEO-2 env-driven persistence)."""

from __future__ import annotations

from himmy.api.deps import ApiContainer
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def test_build_default_is_in_memory_and_offline() -> None:
    """build_default always uses the in-memory store (offline-green)."""
    container = ApiContainer.build_default()
    assert isinstance(container.storage, StorageService)
    # The evaluation service is wired so the eval router/dashboard work.
    assert container.evaluation is not None


def test_offline_privacy_audit_wiring_is_transparent() -> None:
    """WS4.7 B4: the always-wired privacy auditor leaves the offline path untouched.

    The zero-config container keeps a *bare* ``StorageService`` (not the governed
    ``ConsentGatedStorage`` wrapper), the runtime carries no ``consent_decider``, the
    consent governance singletons are ``None``, and the wired ``privacy_audit`` is inert:
    its provider gate is off (the stub manager), so no LLM metric ever lights up.
    """
    container = ApiContainer.build_default()
    # Bare store + runtime, exactly as before WS4.6/WS4.7.
    assert type(container.storage) is StorageService
    assert container.runtime._consent_decider is None  # noqa: SLF001 - invariant proof
    assert container.consent_ledger is None
    assert container.consent_policy is None
    # The privacy auditor is wired but the provider gate is off (stub ⇒ no LLM metric).
    assert container.privacy_audit is not None
    assert container.privacy_audit._provider_is_real() is False  # noqa: SLF001


def test_build_default_async_is_durable_sqlite_without_dsn(
    monkeypatch, tmp_path
) -> None:
    """Without HIMMY_DATABASE_URL, the async (server) builder defaults to durable SQLite.

    The durable-default policy (item #3): server/multi-worker entrypoints persist by
    default — SQLite at ``HIMMY_STORE_PATH`` when no Postgres DSN is set — so background
    runs and idempotency survive restarts. The sync ``build_default`` still wires the
    in-memory store for zero-config offline use (covered above). ``HIMMY_STORE_PATH`` is
    steered into ``tmp_path`` so the test never writes a stray ``.himmy/storage.db``.
    """
    from himmy.services.storage.sqlite import SqliteStorageService

    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))
    container = run_async(ApiContainer.build_default_async())
    assert isinstance(container.storage, SqliteStorageService)


def test_build_storage_selects_postgres_when_dsn_set(monkeypatch) -> None:
    """A set DSN routes _build_storage to PostgresStorageService (gated on the extra).

    Without asyncpg/a DB the connect() raises HimmyError — proving the Postgres
    path is selected by env, not silently ignored.
    """
    import pytest

    from himmy.core.errors import HimmyError

    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgresql://localhost/none")
    try:
        import asyncpg  # type: ignore  # noqa: F401

        has_asyncpg = True
    except ImportError:
        has_asyncpg = False

    if has_asyncpg:
        pytest.skip("asyncpg installed; cannot assert the missing-extra path offline")

    with pytest.raises(HimmyError):
        run_async(ApiContainer._build_storage())


def test_container_aclose_is_safe() -> None:
    """aclose tolerates an in-memory store with no close()."""
    container = ApiContainer.build_default()
    run_async(container.aclose())  # no exception
