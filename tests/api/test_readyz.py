"""G3 + G4: durability truth on app.state + GET /readyz fail-loud readiness.

/health stays a 200 liveness probe; /readyz reports whether the pod can actually serve
durable traffic (durable backend wired, Postgres reachable, migrations applied) and
returns 503 otherwise. The durability truth is stamped on app.state both in the
create_app body (so a bare TestClient with no lifespan does not AttributeError) and in
every lifespan branch.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import _readiness, create_app
from himmy.api.route_introspection import collect_route_paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for var in (
        "HIMMY_MULTI_TENANT",
        "HIMMY_AUTH_MODE",
        "HIMMY_INTERNAL_API_KEY",
        "HIMMY_API_KEYS_FILE",
        "HIMMY_DATABASE_URL",
        "HIMMY_DURABLE_STORAGE",
        "HIMMY_BIND_HOST",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


# ----------------------------------------------------------- G3: durability truth


def test_create_app_body_stamps_durability_truth() -> None:
    # Even without entering the lifespan, app.state carries the readiness fields.
    app = create_app()
    assert app.state.durable_requested is False
    assert app.state.built_durable is False
    assert app.state.active_backend == "memory"


def test_durable_requested_recorded_in_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    # Requested but the lifespan has not run → uninitialized.
    assert app.state.durable_requested is True
    assert app.state.built_durable is False


# ------------------------------------------------------------------- G4: /readyz


def test_health_and_readyz_are_distinct_routes() -> None:
    app = create_app()
    paths = collect_route_paths(app)
    assert "/health" in paths
    assert "/readyz" in paths


def test_default_readyz_is_ready_no_db_call() -> None:
    app = create_app()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "backend": "memory"}


def test_durable_requested_but_uninitialized_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # G3 must_fix: a bare TestClient (NO `with` block → no lifespan) over a
    # durable-requested build reports 503, not a 200 nor an AttributeError.
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    resp = TestClient(app).get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "not initialized" in body["reason"]


def test_readyz_is_unguarded(monkeypatch: pytest.MonkeyPatch) -> None:
    # /readyz must answer for a K8s probe regardless of the studio host/origin guard.
    app = create_app()
    client = TestClient(app)
    # A non-loopback Host that the guard would reject on /v1 must still pass for /readyz.
    resp = client.get("/readyz", headers={"Host": "evil.example.com"})
    assert resp.status_code in (200, 503)
    assert resp.json()["status"] in ("ready", "not_ready")


def test_health_stays_200_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    client = TestClient(app)
    assert client.get("/health").status_code == 200  # liveness unaffected
    assert client.get("/readyz").status_code == 503  # readiness fails


# ------------------------------- G4: Postgres readiness branch (offline, fake state)


def _pg_storage(
    monkeypatch: pytest.MonkeyPatch, *, reachable: bool, applied: list[int]
) -> Any:
    """A real PostgresStorageService (pool=None) with its PUBLIC accessors stubbed.

    Using the real class exercises the ``isinstance`` branch in ``_readiness`` and the
    actual ``code_migration_version`` while keeping the test fully offline (no DB).
    """
    from himmy.services.storage.postgres import PostgresStorageService

    storage = PostgresStorageService(pool=None, cipher=None)

    async def _ping(*, timeout: float = 2.0) -> bool:
        return reachable

    async def _applied(*, timeout: float = 2.0) -> list[int]:
        return list(applied)

    monkeypatch.setattr(storage, "ping", _ping)
    monkeypatch.setattr(storage, "applied_migration_versions", _applied)
    return storage


def _wire_pg(app: Any, storage: Any) -> None:
    app.state.durable_requested = True
    app.state.built_durable = True
    app.state.active_backend = "postgres"
    app.state.container = _FakeContainer(storage)


@pytest.mark.asyncio
async def test_readiness_postgres_unreachable_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    _wire_pg(app, _pg_storage(monkeypatch, reachable=False, applied=[]))
    ready, payload = await _readiness(app)
    assert ready is False
    assert payload["reason"] == "postgres unreachable"


@pytest.mark.asyncio
async def test_readiness_postgres_migrations_behind_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    _wire_pg(app, _pg_storage(monkeypatch, reachable=True, applied=[1, 2]))
    ready, payload = await _readiness(app)
    assert ready is False
    assert payload["reason"] == "schema migrations behind code"
    assert payload["applied_version"] == 2
    assert payload["code_version"] > 2


@pytest.mark.asyncio
async def test_readiness_postgres_healthy_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from himmy.services.storage.postgres import PostgresStorageService

    code_max = PostgresStorageService.code_migration_version()
    app = create_app()
    _wire_pg(
        app, _pg_storage(monkeypatch, reachable=True, applied=list(range(1, code_max + 1)))
    )
    ready, payload = await _readiness(app)
    assert ready is True
    assert payload["status"] == "ready"
    assert payload["applied_version"] == code_max


class _FakeContainer:
    def __init__(self, storage: Any) -> None:
        self.storage = storage
