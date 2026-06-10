"""Tests for the Studio Memory browser (durable store wrap + HTTP surface)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_memory as sm
from himmy.api.app import create_app
from tests.conftest import run_async


@pytest.fixture
def mem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    sm.reset_memory_service()
    yield
    sm.reset_memory_service()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_MEMORY_PATH", str(tmp_path / "memory.db"))
    monkeypatch.chdir(tmp_path)
    sm.reset_memory_service()
    yield TestClient(create_app())
    sm.reset_memory_service()


def test_add_list_forget(mem: None) -> None:
    a = sm.add_memory("We keep ducks", subject_id="farm")
    sm.add_memory("Pond has carp", subject_id="farm")
    items = sm.list_memories("farm")
    assert {i.text for i in items} == {"We keep ducks", "Pond has carp"}
    assert "farm" in sm.list_subjects()
    assert sm.forget(a.memory_id) is True
    assert len(sm.list_memories("farm")) == 1


def test_recall_ranks_by_similarity(mem: None) -> None:
    sm.add_memory("We keep Khaki Campbell ducks", subject_id="farm")
    sm.add_memory("The tractor needs an oil change", subject_id="farm")
    hits = run_async(sm.recall("what ducks do we have", subject_id="farm", top_k=2))
    assert hits
    assert "duck" in hits[0].text.lower()


# ---- HTTP surface (in-process app) ----------------------------------------


def _add(client: TestClient, text: str, subject: str = "default") -> dict:
    r = client.post("/api/studio/memory", json={"text": text, "subject_id": subject})
    assert r.status_code == 200
    return r.json()


def test_patch_edits_text_preserving_identity(client: TestClient) -> None:
    added = _add(client, "Pond has carp", subject="farm")
    r = client.patch(
        f"/api/studio/memory/{added['memory_id']}",
        json={"text": "Pond has carp and tilapia"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["memory_id"] == added["memory_id"]
    assert body["subject_id"] == "farm"
    assert body["kind"] == added["kind"]
    assert body["created_at"] == added["created_at"]
    assert body["text"] == "Pond has carp and tilapia"
    items = client.get("/api/studio/memory?subject=farm").json()
    assert [i["text"] for i in items] == ["Pond has carp and tilapia"]


def test_patch_unknown_memory_404(client: TestClient) -> None:
    r = client.patch("/api/studio/memory/nope", json={"text": "x"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_patch_input_bounds_422(client: TestClient) -> None:
    added = _add(client, "bounded")
    mid = added["memory_id"]
    assert (
        client.patch(f"/api/studio/memory/{mid}", json={"text": ""}).status_code == 422
    )
    assert (
        client.patch(f"/api/studio/memory/{mid}", json={"text": "   "}).status_code
        == 422
    )
    assert (
        client.patch(
            f"/api/studio/memory/{mid}", json={"text": "x" * 20_001}
        ).status_code
        == 422
    )
    # over-long path ids are rejected, not passed to the store
    assert (
        client.patch("/api/studio/memory/" + "a" * 201, json={"text": "x"}).status_code
        == 422
    )
    # record untouched after the rejected edits
    items = client.get("/api/studio/memory?subject=default").json()
    assert [i["text"] for i in items] == ["bounded"]


def test_patch_refreshes_recall_embedding(client: TestClient) -> None:
    added = _add(client, "The tractor needs an oil change", subject="farm")
    # Populate the in-process embedding cache for the old wording.
    warm = client.post(
        "/api/studio/memory/recall", json={"query": "tractor", "subject_id": "farm"}
    )
    assert warm.status_code == 200
    r = client.patch(
        f"/api/studio/memory/{added['memory_id']}",
        json={"text": "We keep Khaki Campbell ducks"},
    )
    assert r.status_code == 200
    hits = client.post(
        "/api/studio/memory/recall",
        json={"query": "what ducks do we have", "subject_id": "farm"},
    ).json()
    assert hits
    assert hits[0]["memory_id"] == added["memory_id"]
    assert "duck" in hits[0]["text"].lower()


def test_subject_filter_subjects_and_delete(client: TestClient) -> None:
    a = _add(client, "We keep ducks", subject="farm")
    _add(client, "Standup at nine", subject="work")
    assert {"farm", "work"} <= set(client.get("/api/studio/memory/subjects").json())
    farm = client.get("/api/studio/memory?subject=farm").json()
    assert [i["subject_id"] for i in farm] == ["farm"]
    assert client.delete(f"/api/studio/memory/{a['memory_id']}").json()["ok"] is True
    assert client.get("/api/studio/memory?subject=farm").json() == []
    # deleting an unknown id reports ok: false, not an error
    assert client.delete("/api/studio/memory/ghost").json()["ok"] is False
