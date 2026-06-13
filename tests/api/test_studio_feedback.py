"""Tests for P0-A: explicit human feedback on Studio runs (thumbs up/down).

Covers the HTTP surface, the durable entity-spine audit trail, the run-row
current-state copy, analytics tallies, and the invariant that re-saving a run
never clobbers feedback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.studio_entities import get_entity_registry, reset_entity_registry
from himmy.api.studio_feedback import feedback_history
from himmy.api.studio_runs import StudioRun, StudioRunStore, reset_run_store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    reset_run_store()  # both stores are cwd-keyed; start fresh per test
    reset_entity_registry()
    return TestClient(create_app())


def _run(client: TestClient, prompt: str = "hello") -> str:
    frames: list[dict] = []
    with client.stream(
        "POST",
        "/api/studio/run",
        json={"agent_path": "agent.yaml", "prompt": prompt, "provider": "stub"},
    ) as r:
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    return frames[-1]["run_id"]


def test_feedback_up_records_and_reads_back(client: TestClient) -> None:
    run_id = _run(client)

    r = client.post(f"/api/studio/runs/{run_id}/feedback", json={"verdict": "up"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "up"
    assert body["run_id"] == run_id
    assert body["actor"] == "human"
    assert body["version"] == 1

    got = client.get(f"/api/studio/runs/{run_id}/feedback").json()
    assert got["verdict"] == "up"


def test_feedback_with_note_surfaces_on_run_detail(client: TestClient) -> None:
    run_id = _run(client)
    client.post(
        f"/api/studio/runs/{run_id}/feedback",
        json={"verdict": "down", "note": "  hallucinated the price  "},
    )
    det = client.get(f"/api/studio/runs/{run_id}").json()
    assert det["feedback_verdict"] == "down"
    assert det["feedback_note"] == "hallucinated the price"  # trimmed
    assert det["feedback_at"]

    # and it shows up in the list summary too
    item = client.get("/api/studio/runs").json()["items"][0]
    assert item["feedback_verdict"] == "down"


def test_rerating_appends_a_new_spine_version(client: TestClient) -> None:
    run_id = _run(client)
    v1 = client.post(
        f"/api/studio/runs/{run_id}/feedback", json={"verdict": "up"}
    ).json()
    v2 = client.post(
        f"/api/studio/runs/{run_id}/feedback", json={"verdict": "down", "note": "no"}
    ).json()
    assert v1["version"] == 1
    assert v2["version"] == 2

    # current state reflects the latest verdict ...
    assert client.get(f"/api/studio/runs/{run_id}/feedback").json()["verdict"] == "down"
    # ... while the immutable spine retains the full ordered audit trail.
    hist = feedback_history(run_id)
    assert [f.verdict for f in hist] == ["up", "down"]
    assert [f.version for f in hist] == [1, 2]


def test_feedback_counts_in_analytics(client: TestClient) -> None:
    up1 = _run(client, "a")
    up2 = _run(client, "b")
    down1 = _run(client, "c")
    client.post(f"/api/studio/runs/{up1}/feedback", json={"verdict": "up"})
    client.post(f"/api/studio/runs/{up2}/feedback", json={"verdict": "up"})
    client.post(f"/api/studio/runs/{down1}/feedback", json={"verdict": "down"})

    stats = client.get("/api/studio/runs/analytics").json()
    assert stats["feedback_up"] == 2
    assert stats["feedback_down"] == 1


def test_feedback_on_unknown_run_is_404(client: TestClient) -> None:
    r = client.post("/api/studio/runs/nope/feedback", json={"verdict": "up"})
    assert r.status_code == 404
    assert client.get("/api/studio/runs/nope/feedback").status_code == 404


def test_invalid_verdict_is_422(client: TestClient) -> None:
    run_id = _run(client)
    r = client.post(f"/api/studio/runs/{run_id}/feedback", json={"verdict": "meh"})
    assert r.status_code == 422


def test_overlong_note_is_422(client: TestClient) -> None:
    run_id = _run(client)
    r = client.post(
        f"/api/studio/runs/{run_id}/feedback",
        json={"verdict": "up", "note": "x" * 4001},
    )
    assert r.status_code == 422


def test_spine_record_carries_subject_for_erasure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIMMY_MEMORY_SUBJECT", "alice")
    run_id = _run(client)
    client.post(f"/api/studio/runs/{run_id}/feedback", json={"verdict": "up"})

    records = get_entity_registry().list_by_kind("run_feedback")
    assert len(records) == 1
    assert records[0].metadata["subject_id"] == "alice"
    assert records[0].metadata["run_id"] == run_id


def test_save_does_not_clobber_feedback() -> None:
    """An HITL resume re-saves a run under the same id; feedback must survive."""
    store = StudioRunStore(":memory:")
    run = StudioRun(id="r1", created_at="2026-01-01T00:00:00Z", output="first")
    store.save(run)
    assert store.set_feedback("r1", verdict="up", note="good", at="2026-01-01T00:01Z")

    # re-save the run (a continuation) — content updates, feedback preserved
    run2 = StudioRun(id="r1", created_at="2026-01-01T00:00:00Z", output="continued")
    store.save(run2)

    got = store.get("r1")
    assert got is not None
    assert got.output == "continued"
    assert got.feedback_verdict == "up"
    assert got.feedback_note == "good"
    store.close()


def test_set_feedback_unknown_run_returns_false() -> None:
    store = StudioRunStore(":memory:")
    assert store.set_feedback("ghost", verdict="up", note=None, at="t") is False
    store.close()


def test_set_feedback_rejects_bad_verdict() -> None:
    store = StudioRunStore(":memory:")
    store.save(StudioRun(id="r1", created_at="t"))
    with pytest.raises(ValueError):
        store.set_feedback("r1", verdict="sideways", note=None, at="t")
    store.close()
