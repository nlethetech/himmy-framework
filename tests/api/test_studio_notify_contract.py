"""Smoke tests for the cross-lane notification contract.

Other Studio lanes import :func:`record_notification` from day one; these
tests pin the parts they rely on (signature defaults, item shape, monotonic
ids, the 500 bound, never-raise) plus the ``GET /api/studio/notify`` surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.routers import studio_notify


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep the durable mirror out of the repo's real ``.himmy/notify.db``.

    The sink now persists every record (notifications survive restarts); these
    contract tests fire hundreds of throwaway items, so they run against a
    temp DB and leave the module memory-seeded afterwards.
    """
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(tmp_path / "notify.db"))
    studio_notify.reset_notify_state()
    yield
    studio_notify.reset_notify_state()
    studio_notify._HYDRATED = True  # later tests must not re-seed from disk


def _drain() -> None:
    with studio_notify._LOCK:
        studio_notify._NOTIFICATIONS.clear()


def test_record_then_get_roundtrip() -> None:
    _drain()
    studio_notify.record_notification(
        "mission", "Mission finished", body="all steps green", link="/missions"
    )
    studio_notify.record_notification("routine", "Routine ran")
    client = TestClient(create_app())
    resp = client.get("/api/studio/notify")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    newest, oldest = items  # newest first
    assert newest["kind"] == "routine"
    assert oldest == {
        "id": oldest["id"],
        "kind": "mission",
        "title": "Mission finished",
        "body": "all steps green",
        "link": "/missions",
        "created_at": oldest["created_at"],
        "read": False,
    }
    assert newest["id"] > oldest["id"]  # monotonic ids
    assert newest["body"] == "" and newest["link"] == ""  # defaults
    _drain()


def test_bounded_at_500_and_never_raises() -> None:
    _drain()
    for i in range(520):
        studio_notify.record_notification("test", f"n{i}")
    with studio_notify._LOCK:
        assert len(studio_notify._NOTIFICATIONS) == 500
    # Hostile input must be swallowed, not raised (contract: never raises).
    studio_notify.record_notification(object(), None)  # type: ignore[arg-type]
    _drain()
