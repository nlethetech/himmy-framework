"""Tests for Studio Chats: durable, resumable chat sessions."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_chats as sc
from himmy.api.app import create_app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_CHATS_PATH", str(tmp_path / "chats.db"))
    sc.reset_chats_store()
    yield
    sc.reset_chats_store()


def test_save_get_list_delete(store: None) -> None:
    s = sc.get_chats_store()
    saved = s.save(
        session_id=None,
        title=None,
        agent_path="agent.yaml",
        provider="stub",
        messages=[
            sc.ChatMessage(role="user", text="Plan my week"),
            sc.ChatMessage(role="agent", text="Sure, here's a plan…"),
        ],
    )
    assert saved.title == "Plan my week"  # derived from first user message
    assert saved.message_count == 2

    detail = s.get(saved.id)
    assert detail is not None
    assert [m.role for m in detail.messages] == ["user", "agent"]

    # Re-saving the same id replaces messages and keeps created_at.
    again = s.save(
        session_id=saved.id,
        title=saved.title,
        agent_path="agent.yaml",
        provider="stub",
        messages=[
            sc.ChatMessage(role="user", text="Plan my week"),
            sc.ChatMessage(role="agent", text="Updated plan"),
            sc.ChatMessage(role="user", text="thanks"),
        ],
    )
    assert again.created_at == saved.created_at
    assert again.message_count == 3
    assert len(s.list()) == 1

    assert s.rename(saved.id, "Weekly plan") is True
    assert s.get(saved.id).title == "Weekly plan"  # type: ignore[union-attr]
    assert s.delete(saved.id) is True
    assert s.list() == []


def test_chats_routes(store: None) -> None:
    c = TestClient(create_app())
    saved = c.post(
        "/api/studio/chats",
        json={
            "agent_path": "a.yaml",
            "provider": "stub",
            "messages": [{"role": "user", "text": "Hi there"}],
        },
    ).json()
    assert saved["title"] == "Hi there"
    assert c.get("/api/studio/chats").json()[0]["id"] == saved["id"]
    got = c.get(f"/api/studio/chats/{saved['id']}").json()
    assert got["messages"][0]["text"] == "Hi there"
    assert c.patch(
        f"/api/studio/chats/{saved['id']}", json={"title": "Renamed"}
    ).json()["ok"]
    assert c.delete(f"/api/studio/chats/{saved['id']}").json()["ok"]
    assert c.get("/api/studio/chats/nope").status_code == 404


def test_save_rejects_bad_role(store: None) -> None:
    c = TestClient(create_app())
    r = c.post(
        "/api/studio/chats",
        json={"messages": [{"role": "system", "text": "x"}]},
    )
    assert r.status_code == 422
