"""Tests for the hardened Studio notifications lane.

Covers the four backend deliverables: the bounded ring + SQLite persistence
behind the unchanged ``record_notification`` contract, the poll/read API
surface, the approval-creation hook (in-process, via the Studio event
translator), and best-effort Telegram forwarding with an injected fake sender.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.routers import studio_notify as sn


@pytest.fixture
def notify_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the durable store at a temp DB and reset all module state."""
    db = tmp_path / "notify.db"
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(db))
    sn.reset_notify_state()
    yield db
    sn.reset_notify_state()
    # Later suites run with the repo-root DB path again; mark the ring as
    # already seeded so a stale on-disk store can't repopulate their deque.
    sn._HYDRATED = True


def _client() -> TestClient:
    return TestClient(create_app())


# ---- ring bounds + persistence -------------------------------------------------


def test_ring_bounded_in_memory_and_on_disk(notify_env: Path) -> None:
    for i in range(520):
        sn.record_notification("test", f"n{i}")
    client = _client()
    items = client.get("/api/studio/notify").json()["items"]
    assert len(items) == sn.RING_SIZE
    assert items[0]["title"] == "n519"  # newest first
    assert items[-1]["title"] == "n20"  # oldest 20 rolled off
    # The durable mirror is trimmed to the same bound.
    import sqlite3

    with sqlite3.connect(notify_env) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()
    assert count == sn.RING_SIZE


def test_survives_restart_with_read_state(notify_env: Path) -> None:
    sn.record_notification("mission", "Mission finished", link="/missions")
    sn.record_notification("routine", "Routine ran", body="3 steps")
    client = _client()
    first_id = client.get("/api/studio/notify").json()["items"][-1]["id"]
    assert client.post(f"/api/studio/notify/{first_id}/read").status_code == 200

    # Simulate a process restart: all in-memory state dropped, then rehydrated.
    sn.reset_notify_state()
    data = _client().get("/api/studio/notify").json()
    titles = [i["title"] for i in data["items"]]
    assert titles == ["Routine ran", "Mission finished"]
    assert data["items"][1]["read"] is True  # read-state survived
    assert data["items"][0]["read"] is False
    assert data["unread"] == 1
    # Ids keep climbing after the restart (monotonic across hydration).
    sn.record_notification("test", "post-restart")
    newest = _client().get("/api/studio/notify").json()["items"][0]
    assert newest["id"] > data["items"][0]["id"]


def test_degrades_to_memory_when_db_unopenable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A directory at the DB path makes sqlite unable to open it.
    bad = tmp_path / "notify.db"
    bad.mkdir()
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(bad))
    sn.reset_notify_state()
    try:
        sn.record_notification("test", "still recorded")  # must not raise
        items = _client().get("/api/studio/notify").json()["items"]
        assert [i["title"] for i in items] == ["still recorded"]
    finally:
        sn.reset_notify_state()
        sn._HYDRATED = True


# ---- read flow + incremental poll ----------------------------------------------


def test_read_flow_and_unread_count(notify_env: Path) -> None:
    for i in range(3):
        sn.record_notification("test", f"n{i}")
    client = _client()
    data = client.get("/api/studio/notify").json()
    assert data["unread"] == 3
    target = data["items"][0]["id"]

    assert client.post(f"/api/studio/notify/{target}/read").status_code == 200
    data = client.get("/api/studio/notify").json()
    assert data["unread"] == 2
    assert [i["read"] for i in data["items"]] == [True, False, False]

    resp = client.post("/api/studio/notify/read-all")
    assert resp.status_code == 200 and resp.json()["marked"] == 2
    data = client.get("/api/studio/notify").json()
    assert data["unread"] == 0 and all(i["read"] for i in data["items"])


def test_mark_read_unknown_id_404(notify_env: Path) -> None:
    assert _client().post("/api/studio/notify/424242/read").status_code == 404


def test_after_param_polls_incrementally(notify_env: Path) -> None:
    sn.record_notification("test", "old")
    client = _client()
    latest = client.get("/api/studio/notify").json()["latest_id"]
    sn.record_notification("test", "new-1")
    sn.record_notification("test", "new-2")
    data = client.get(f"/api/studio/notify?after={latest}").json()
    assert [i["title"] for i in data["items"]] == ["new-2", "new-1"]
    assert data["unread"] == 3  # unread always counts the whole ring
    assert client.get("/api/studio/notify?after=-1").status_code == 422


# ---- the approval hook -----------------------------------------------------------


def test_approval_required_event_rings_the_bell(notify_env: Path) -> None:
    """The Studio event translator records an 'approval' notification."""
    from himmy.api import studio_service as ss
    from himmy.core.events import EventType, RunEvent

    cog = ss._Cognition({}, "mailer")
    event = RunEvent(
        event_type=EventType.APPROVAL_REQUIRED,
        payload={"checkpoint_id": "cp-1", "tools": ["send_email"]},
    )
    frames = cog.frames(event)
    assert [f["type"] for f in frames] == ["approval_required"]

    data = _client().get("/api/studio/notify").json()
    assert len(data["items"]) == 1
    note = data["items"][0]
    assert note["kind"] == "approval"
    assert "send_email" in note["title"]
    assert note["link"] == "/approvals"
    assert "mailer" in note["body"]


# ---- telegram forwarding ----------------------------------------------------------


def test_forwarding_called_with_fake_sender(
    notify_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(sn, "_telegram_target", lambda: ("tok", "chat-9"))
    monkeypatch.setattr(
        sn, "_dispatch_telegram", lambda t, c, x: sent.append((t, c, x))
    )
    client = _client()

    # Forwarding is off by default — nothing goes out.
    sn.record_notification("mission", "quiet one")
    assert sent == []

    resp = client.post("/api/studio/notify/settings", json={"forward_telegram": True})
    assert resp.status_code == 200
    sn.record_notification("mission", "Mission finished", body="x" * 1000)
    assert len(sent) == 1
    token, chat_id, text = sent[0]
    assert (token, chat_id) == ("tok", "chat-9")
    assert text.startswith("[himmy] Mission finished")
    assert len(text) <= len("[himmy] ") + 400  # truncated

    # The setting is reflected by GET and turns off cleanly.
    assert client.get("/api/studio/notify").json()["settings"] == {
        "forward_telegram": True
    }
    client.post("/api/studio/notify/settings", json={"forward_telegram": False})
    sn.record_notification("mission", "quiet again")
    assert len(sent) == 1


def test_forwarding_failure_never_breaks_recording(
    notify_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> tuple[str, str]:
        raise RuntimeError("telegram is down")

    monkeypatch.setattr(sn, "_telegram_target", _boom)
    client = _client()
    client.post("/api/studio/notify/settings", json={"forward_telegram": True})
    sn.record_notification("mission", "still lands")  # must not raise
    items = client.get("/api/studio/notify").json()["items"]
    assert "still lands" in [i["title"] for i in items]


def test_forward_setting_survives_restart(notify_env: Path) -> None:
    client = _client()
    client.post("/api/studio/notify/settings", json={"forward_telegram": True})
    sn.reset_notify_state()  # simulated restart
    data = _client().get("/api/studio/notify").json()
    assert data["settings"] == {"forward_telegram": True}


# ---- optional live test (local Ollama) ---------------------------------------


def _ollama_up() -> bool:
    try:
        import httpx

        return (
            httpx.get("http://localhost:11434/api/tags", timeout=2.0).status_code == 200
        )
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
def test_live_approval_pause_rings_the_bell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MECHANICS, not prose: a real run that pauses on an approval-gated tool
    streams ``approval_required``, persists the checkpoint in the approvals
    inbox, and records an 'approval' notification linking to /approvals."""
    import json

    from himmy.api import studio_approvals as sa
    from himmy.api.studio_runs import reset_run_store

    (tmp_path / "tg.agent.yaml").write_text(
        "name: tg-notifier\n"
        "description: Sends Telegram messages on request.\n"
        "tool_packs: [telegram]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(tmp_path / "notify.db"))
    # Keep the approval gate ON regardless of the ambient environment.
    monkeypatch.delenv("HIMMY_COMMS_ALLOW_SEND", raising=False)
    sn.reset_notify_state()
    sa.reset_checkpoint_store()
    reset_run_store()
    try:
        client = _client()
        frames: list[dict] = []
        with client.stream(
            "POST",
            "/api/studio/run",
            json={
                "agent_path": "tg.agent.yaml",
                "prompt": (
                    "Use the send_telegram tool to send a message that says "
                    "exactly: hello from the test."
                ),
                "provider": "ollama",
                "model": "qwen2.5:7b-instruct",
            },
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line and line.startswith("data: "):
                    frames.append(json.loads(line[6:]))
        paused = [f for f in frames if f["type"] == "approval_required"]
        if not paused:  # a small model may answer without touching the tool
            pytest.skip(
                f"model never invoked the gated tool: {[f['type'] for f in frames]}"
            )
        # The bell rang, with the tool named and the inbox linked.
        data = client.get("/api/studio/notify").json()
        notes = [n for n in data["items"] if n["kind"] == "approval"]
        assert notes, data
        assert notes[0]["link"] == "/approvals"
        assert "send_telegram" in notes[0]["title"]
        # And the checkpoint truly persisted in the approvals inbox.
        pending = client.get("/api/studio/approvals").json()
        ids = {p["checkpoint_id"] for p in pending}
        assert paused[0]["checkpoint_id"] in ids
    finally:
        sn.reset_notify_state()
        sn._HYDRATED = True
        sa.reset_checkpoint_store()
        reset_run_store()


# ---- rbac-harden(mopup-r1): notify forwarding is a deployment-global, operator-only toggle


def _tenant_admin_client() -> TestClient:
    """A TENANT-BOUND admin principal (all_tenants=False, holds studio.notify:write)."""
    from himmy.api import ApiContainer
    from himmy.api import create_app as _create
    from himmy.api.auth.apikey import ApiKeyAuthenticator
    from himmy.api.auth.principal import Principal

    app = _create(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=["admin"], auth_method="apikey"
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


def test_tenant_bound_principal_cannot_flip_global_forwarding(
    notify_env: Path,
) -> None:
    """A tenant-bound principal must NOT toggle the deployment-global forward setting.

    Even holding studio.notify:write (admin), the forward-Telegram flag is a process-global
    one-row setting shared across co-tenants, so the write is restricted to an operator /
    all_tenants principal. A tenant-bound caller gets 403 instead of silently flipping
    forwarding for every other tenant.
    """
    client = _tenant_admin_client()
    resp = client.post(
        "/api/studio/notify/settings", json={"forward_telegram": False}
    )
    assert resp.status_code == 403, resp.text
    # The global flag was NOT mutated by the rejected request.
    assert sn._FORWARD_TELEGRAM is False  # default, untouched


def test_offline_principal_can_set_forwarding(notify_env: Path) -> None:
    """The offline / all_tenants default (no authenticator) still sets the toggle."""
    client = _client()  # ANONYMOUS / all_tenants -> operator posture
    resp = client.post(
        "/api/studio/notify/settings", json={"forward_telegram": True}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["settings"] == {"forward_telegram": True}
