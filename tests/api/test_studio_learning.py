"""Studio ``GET /api/studio/learning`` — the read-only "what Himmy learned" panel.

PARITY: the rows mirror what the CLI ``himmy learned`` prints (same
:func:`himmy.services.learning.report.build_learning_report`).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.config.project import HIMMY_SPINE_PATH_ENV
from himmy.core.events import EventType, RunEvent
from tests.conftest import run_async


@pytest.fixture()
def isolated_spine(tmp_path, monkeypatch):
    """Point the override store's durable spine at a per-test temp ``spine.db``.

    The POST routes write through ``OverrideStore.for_context`` → the canonical
    ``.himmy/spine.db``. Isolating the path keeps each test's overrides from leaking into
    the project's real spine (or into the next test).
    """
    monkeypatch.setenv(HIMMY_SPINE_PATH_ENV, str(tmp_path / "spine.db"))
    yield


def _seed(store: object, tool: str, ok: bool, n: int = 1) -> None:
    for _ in range(n):
        run_async(
            store.append_event(  # type: ignore[attr-defined]
                RunEvent(
                    event_type=EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                    payload={"tool_name": tool},
                )
            )
        )


def test_empty_learning_panel_is_clean() -> None:
    container = ApiContainer.build_default()
    body = TestClient(create_app(container)).get("/api/studio/learning").json()
    assert body["has_data"] is False
    assert body["tools"] == []
    assert body["flaky_count"] == 0


def test_learning_panel_surfaces_flaky_tool() -> None:
    container = ApiContainer.build_default()
    store = container.storage
    _seed(store, "flaky_api", False, n=9)
    _seed(store, "flaky_api", True, n=1)  # 10% -> flaky
    _seed(store, "web_search", True, n=8)  # 100% -> reliable
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.LEARNING_APPLIED,
                payload={"tools_reordered": 1, "deprioritised_tools": ["flaky_api"]},
            )
        )
    )

    body = TestClient(create_app(container)).get("/api/studio/learning").json()
    assert body["has_data"] is True
    assert body["flaky_count"] == 1
    assert body["floor"] == 0.2
    # Worst-first: the flaky tool leads.
    assert body["tools"][0]["tool_name"] == "flaky_api"
    assert body["tools"][0]["flaky"] is True
    assert body["recent_reorders"][0]["deprioritised_tools"] == ["flaky_api"]


def test_outcome_weight_query_is_bounded() -> None:
    container = ApiContainer.build_default()
    c = TestClient(create_app(container))
    assert c.get("/api/studio/learning", params={"outcome_weight": -0.1}).status_code == 422
    assert c.get("/api/studio/learning", params={"outcome_weight": 1.5}).status_code == 422


def test_get_carries_override_and_trust_defaults(isolated_spine) -> None:
    """With nothing set the GET still carries the new fields at their defaults."""
    container = ApiContainer.build_default()
    _seed(container.storage, "web_search", True, n=4)
    body = TestClient(create_app(container)).get("/api/studio/learning").json()
    assert body["trust_feedback"] is False
    assert all(t["override"] is None for t in body["tools"])


def test_post_override_pin_marks_tool_and_drops_flaky(isolated_spine) -> None:
    """Pinning a flaky tool: GET shows override==pinned and the flaky bucket drops."""
    container = ApiContainer.build_default()
    store = container.storage
    _seed(store, "flaky_api", False, n=9)
    _seed(store, "flaky_api", True, n=1)  # 10% -> flaky
    client = TestClient(create_app(container))

    before = client.get("/api/studio/learning").json()
    assert before["flaky_count"] == 1
    assert before["tools"][0]["tool_name"] == "flaky_api"
    assert before["tools"][0]["flaky"] is True
    assert before["tools"][0]["override"] is None

    resp = client.post(
        "/api/studio/learning/overrides",
        json={"tool_name": "flaky_api", "action": "pin"},
    )
    assert resp.status_code == 200
    after = resp.json()
    # The refreshed payload reflects the pin: never flaky, labelled pinned, score 1.0.
    assert after["flaky_count"] == 0
    pinned = next(t for t in after["tools"] if t["tool_name"] == "flaky_api")
    assert pinned["override"] == "pinned"
    assert pinned["flaky"] is False
    assert pinned["score"] == 1.0

    # And a fresh GET sees the same persisted state (shared spine, not just the POST echo).
    again = client.get("/api/studio/learning").json()
    assert again["flaky_count"] == 0
    assert next(
        t for t in again["tools"] if t["tool_name"] == "flaky_api"
    )["override"] == "pinned"


def test_post_override_mute_then_clear(isolated_spine) -> None:
    """Mute labels a tool ignored; clear returns it to pure auto-learning."""
    container = ApiContainer.build_default()
    store = container.storage
    _seed(store, "flaky_api", False, n=9)
    _seed(store, "flaky_api", True, n=1)
    client = TestClient(create_app(container))

    muted = client.post(
        "/api/studio/learning/overrides",
        json={"tool_name": "flaky_api", "action": "mute"},
    ).json()
    assert next(
        t for t in muted["tools"] if t["tool_name"] == "flaky_api"
    )["override"] == "ignored"
    assert muted["flaky_count"] == 0

    cleared = client.post(
        "/api/studio/learning/overrides",
        json={"tool_name": "flaky_api", "action": "clear"},
    ).json()
    flaky_row = next(t for t in cleared["tools"] if t["tool_name"] == "flaky_api")
    assert flaky_row["override"] is None
    # Back to pure auto-learning -> the genuinely-flaky tool is flaky again.
    assert flaky_row["flaky"] is True
    assert cleared["flaky_count"] == 1


def test_post_override_rejects_unknown_action(isolated_spine) -> None:
    container = ApiContainer.build_default()
    client = TestClient(create_app(container))
    resp = client.post(
        "/api/studio/learning/overrides",
        json={"tool_name": "web_search", "action": "boost"},
    )
    assert resp.status_code == 422


def test_post_trust_feedback_toggles(isolated_spine) -> None:
    """Turning trust-feedback on persists and surfaces in the GET payload."""
    container = ApiContainer.build_default()
    client = TestClient(create_app(container))

    on = client.post(
        "/api/studio/learning/trust-feedback", json={"enabled": True}
    )
    assert on.status_code == 200
    assert on.json()["trust_feedback"] is True
    assert client.get("/api/studio/learning").json()["trust_feedback"] is True

    off = client.post(
        "/api/studio/learning/trust-feedback", json={"enabled": False}
    ).json()
    assert off["trust_feedback"] is False
    assert client.get("/api/studio/learning").json()["trust_feedback"] is False


def test_post_override_404_without_storage(isolated_spine) -> None:
    """Unwired storage still 404s on the write path (no event log to refresh from)."""
    container = ApiContainer.build_default()
    container.storage = None  # type: ignore[assignment]
    client = TestClient(create_app(container))
    resp = client.post(
        "/api/studio/learning/overrides",
        json={"tool_name": "web_search", "action": "pin"},
    )
    assert resp.status_code == 404
