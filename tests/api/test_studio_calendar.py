"""Tests for the Studio calendar event store + routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_calendar as cal
from himmy.api.app import create_app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cal.reset_calendar_store()
    yield
    cal.reset_calendar_store()


def test_add_list_by_month_delete(store: None) -> None:
    s = cal.get_calendar_store()
    a = s.add(cal.CalendarEvent(date="2026-06-10", title="Vet", time="09:00"))
    s.add(cal.CalendarEvent(date="2026-06-15", title="Harvest"))
    s.add(cal.CalendarEvent(date="2026-07-01", title="July thing"))
    june = s.list(month="2026-06")
    assert [e.title for e in june] == ["Vet", "Harvest"]  # timed first, ordered
    assert len(s.list()) == 3
    assert s.delete(a.id) is True
    assert len(s.list(month="2026-06")) == 1
    assert s.delete("nope") is False


def test_calendar_routes(store: None) -> None:
    c = TestClient(create_app())
    r = c.post("/api/studio/calendar", json={"date": "2026-06-10", "title": "Vet"})
    assert r.status_code == 200
    eid = r.json()["id"]
    assert c.get("/api/studio/calendar?month=2026-06").json()[0]["title"] == "Vet"
    assert c.delete(f"/api/studio/calendar/{eid}").json()["ok"]


def test_calendar_input_caps_422(store: None) -> None:
    c = TestClient(create_app())
    # empty title rejected
    assert (
        c.post(
            "/api/studio/calendar", json={"date": "2026-06-10", "title": ""}
        ).status_code
        == 422
    )
