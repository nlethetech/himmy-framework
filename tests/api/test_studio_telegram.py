"""T3g — Studio Telegram inbound-listener manager.

Covers the manager + router contract entirely offline (a fake Telegram client, the stub
inference provider, an in-memory ConversationStore): start/stop is idempotent and drains
on shutdown; an empty allowlist or a missing token is refused; the per-token flock blocks a
second poller; an inbound (mocked) message runs through the CANONICAL run path — it records
a Studio run, applies guardrails, and lands in the durable conversation store; and the bot
token never appears in a response body.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_telegram as svc
from himmy.api.app import create_app
from himmy.api.studio_runs import reset_run_store
from himmy.services.storage.conversations import ConversationStore
from tests.conftest import run_async

_TOKEN = "123456:ABCDEF-test-token"


# ---------------------------------------------------------------- fake transport


class _FakeClient:
    """Records sent messages and serves scripted update batches (offline)."""

    def __init__(self, batches: list[list[dict[str, Any]]] | None = None) -> None:
        self.sent: list[tuple[Any, str]] = []
        self._batches = list(batches or [])

    async def send_message(self, chat_id: Any, text: str) -> dict[str, Any]:
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    async def get_updates(
        self, *, offset: int | None = None, timeout: int = 25
    ) -> list[dict[str, Any]]:
        if self._batches:
            return self._batches.pop(0)
        # No more scripted updates: yield control so the loop can observe a stop flag
        # without hot-spinning the test event loop.
        await asyncio.sleep(0.01)
        return []


def _msg(update_id: int, chat_id: int, text: str) -> dict[str, Any]:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


# ----------------------------------------------------------------------- fixtures


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A project root with an agent spec + isolated cwd-keyed stores."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_CONVERSATIONS_PATH", str(tmp_path / "conversations.db"))
    monkeypatch.delenv("HIMMY_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("HIMMY_TELEGRAM_ALLOWED_CHATS", raising=False)
    svc.reset_listener_manager()
    reset_run_store()
    yield tmp_path
    # Always drain any running listener so a leaked flock never wedges the next test.
    run_async(svc.get_listener_manager().stop())
    svc.reset_listener_manager()
    reset_run_store()


def _configured(monkeypatch: pytest.MonkeyPatch, *, allowed: str | None = "5") -> None:
    monkeypatch.setenv("HIMMY_TELEGRAM_BOT_TOKEN", _TOKEN)
    if allowed is not None:
        monkeypatch.setenv("HIMMY_TELEGRAM_ALLOWED_CHATS", allowed)
    else:
        monkeypatch.delenv("HIMMY_TELEGRAM_ALLOWED_CHATS", raising=False)


# --------------------------------------------------------- manager: config gating


def test_start_refuses_without_token(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No bot token configured → refuse to start (no listener, no flock)."""
    manager = svc.TelegramListenerManager(client_factory=lambda t, c: _FakeClient())
    with pytest.raises(svc.TelegramListenerError):
        manager.start(agent_path="agent.yaml")
    assert not manager.active


def test_start_refuses_empty_allowlist(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SECURITY: an empty chat allowlist is rejected unless allow_everyone is explicit."""
    _configured(monkeypatch, allowed=None)
    manager = svc.TelegramListenerManager(client_factory=lambda t, c: _FakeClient())
    with pytest.raises(svc.TelegramListenerError) as exc:
        manager.start(agent_path="agent.yaml")
    assert "allowlist" in str(exc.value)
    assert not manager.active


def test_allow_everyone_override_starts_open(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit allow_everyone override lets an open (allowlist-free) bot start."""
    _configured(monkeypatch, allowed=None)
    manager = svc.TelegramListenerManager(client_factory=lambda t, c: _FakeClient())

    async def _go() -> dict[str, Any]:
        status = manager.start(agent_path="agent.yaml", allow_everyone=True)
        await manager.stop()
        return status

    status = run_async(_go())
    assert status["active"] is True
    assert status["allow_everyone"] is True
    assert status["allowed_chat_ids"] == []


# --------------------------------------------------------- manager: idempotency


def test_start_is_idempotent_second_start_is_busy(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second start while a listener is already active is refused (busy)."""
    _configured(monkeypatch)
    manager = svc.TelegramListenerManager(client_factory=lambda t, c: _FakeClient())

    async def _go() -> None:
        manager.start(agent_path="agent.yaml")
        with pytest.raises(svc.TelegramListenerBusy):
            manager.start(agent_path="agent.yaml")
        assert manager.active
        await manager.stop()

    run_async(_go())


def test_stop_is_idempotent_and_releases_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping is idempotent; after stop a fresh start succeeds (flock was released)."""
    _configured(monkeypatch)
    manager = svc.TelegramListenerManager(client_factory=lambda t, c: _FakeClient())

    async def _go() -> None:
        manager.start(agent_path="agent.yaml")
        await manager.stop()
        await manager.stop()  # second stop is a no-op
        assert not manager.active
        # The token flock was released, so the listener can be started again.
        manager.start(agent_path="agent.yaml")
        assert manager.active
        await manager.stop()

    run_async(_go())


# --------------------------------------------------------- flock: cross-process


def test_flock_blocks_a_second_poller_on_the_same_token(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-token flock means a second poller (CLI or another instance) is refused.

    Stand in for ``himmy telegram`` already polling the token by holding its flock; the
    Studio listener must then refuse to start with a busy error rather than spin up a
    competing ``getUpdates`` loop on the same token.
    """
    from himmy.core.process_lock import acquire_process_lock

    _configured(monkeypatch)
    held = acquire_process_lock(svc.telegram_lock_name(_TOKEN))
    try:
        manager = svc.TelegramListenerManager(
            client_factory=lambda t, c: _FakeClient()
        )
        with pytest.raises(svc.TelegramListenerBusy):
            manager.start(agent_path="agent.yaml")
        assert not manager.active
    finally:
        held.release()


def test_lock_name_does_not_leak_the_token(project: Path) -> None:
    """The flock file name is a digest, never the raw token."""
    name = svc.telegram_lock_name(_TOKEN)
    assert _TOKEN not in name
    assert name.startswith("telegram-")


def _cli_args(**kw: Any) -> Any:
    """A minimal namespace `cmd_telegram` reads (real CLI entrypoint, not a stand-in)."""
    import argparse

    base: dict[str, Any] = {
        "file": None,
        "name": None,
        "instruction": None,
        "provider": None,
        "model": None,
        "token": _TOKEN,
        "allow_chat": ["5"],
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_telegram_refuses_when_studio_listener_holds_the_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI `himmy telegram` participates in the shared flock (NOT a stand-in).

    Hold the per-token flock the way the Studio listener does, then drive the ACTUAL CLI
    entrypoint ``cmd_telegram``: it must refuse (exit 2) before ever building/polling the
    bot, so the two interfaces can't both long-poll one token.
    """
    from himmy.cli.commands import cmd_telegram
    from himmy.core.process_lock import acquire_process_lock

    _configured(monkeypatch)
    held = acquire_process_lock(svc.telegram_lock_name(_TOKEN))
    try:
        rc = cmd_telegram(_cli_args())
        assert rc == 2
    finally:
        held.release()


def test_start_endpoint_409_when_cli_telegram_holds_the_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse direction: with the CLI holding the token's flock, /start returns 409.

    Run the REAL ``cmd_telegram`` on a worker thread but replace only its serve loop with a
    barrier (so it acquires + holds the same flock without long-polling the network); while
    it holds the lock, POST /start on the same token must surface a distinct 409.
    """
    import threading

    from himmy.cli import commands as cli_commands
    from himmy.cli.commands import cmd_telegram

    _configured(monkeypatch)
    holding = threading.Event()
    release = threading.Event()

    def _fake_exec(
        factory: Any, registry: Any, mcp_servers: Any, **_kwargs: Any
    ) -> Any:
        # The CLI now holds the per-token flock (acquired before this call); signal the
        # test and block here (standing in for the long-poll) until told to release.
        holding.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(cli_commands, "_exec_with_mcp", _fake_exec)

    rc_box: dict[str, int] = {}

    def _run_cli() -> None:
        rc_box["rc"] = cmd_telegram(_cli_args())

    worker = threading.Thread(target=_run_cli, daemon=True)
    worker.start()
    try:
        assert holding.wait(timeout=5.0), "CLI never acquired the flock"
        with TestClient(create_app()) as http:
            svc.get_listener_manager()._client_factory = (  # type: ignore[return-value]
                lambda t, c: _FakeClient()
            )
            res = http.post(
                "/api/studio/telegram/start", json={"agent_path": "agent.yaml"}
            )
        assert res.status_code == 409, res.text
        assert "already being polled" in res.json()["detail"]
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert rc_box.get("rc") == 0


# -------------------------------------------------- canonical per-message path


def test_inbound_message_runs_through_canonical_path(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked inbound update runs the canonical pipeline: records a run + replies.

    Drive the manager with a fake client scripted with one allow-listed message, run the
    listener until it has answered, then assert (1) the chat got a reply, (2) a Studio run
    row was recorded (the canonical run path — same store /v1 + the CLI read), and (3) the
    turn was persisted to the durable conversation store (T2.3).
    """
    _configured(monkeypatch, allowed="5")
    store = ConversationStore(":memory:")
    client = _FakeClient(batches=[[_msg(1, 5, "hello bot")]])

    manager = svc.TelegramListenerManager(
        client_factory=lambda t, c: client, store=store
    )

    async def _go() -> None:
        manager.start(agent_path="agent.yaml", provider="stub")
        for _ in range(200):
            if client.sent:
                break
            await asyncio.sleep(0.01)
        await manager.stop()

    run_async(_go())

    # (1) the allow-listed chat got a reply
    assert client.sent, "expected the bot to reply to the inbound message"
    assert client.sent[0][0] == 5

    # (2) the run landed in the canonical Studio run store
    from himmy.api.studio_runs import get_run_store

    runs = get_run_store().list()
    assert any(r.agent_name == "helper" for r in runs), "inbound turn recorded no run"

    # (3) the turn is durable + browsable in the shared conversation store
    thread = store.load_thread("telegram:5")
    assert thread is not None
    assert any("hello bot" in m.content for m in thread.messages)


def test_stranger_chat_is_dropped_by_allowlist(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-allow-listed sender never reaches the agent and gets no reply."""
    _configured(monkeypatch, allowed="5")
    store = ConversationStore(":memory:")
    # chat 7 is a stranger; chat 5 is allowed.
    client = _FakeClient(batches=[[_msg(1, 7, "intruder"), _msg(2, 5, "trusted")]])

    manager = svc.TelegramListenerManager(
        client_factory=lambda t, c: client, store=store
    )

    async def _go() -> None:
        manager.start(agent_path="agent.yaml", provider="stub")
        for _ in range(200):
            if client.sent:
                break
            await asyncio.sleep(0.01)
        await manager.stop()

    run_async(_go())

    # Only the allow-listed chat reached the handler.
    assert [c for c, _ in client.sent] == [5]
    assert store.load_thread("telegram:7") is None


# ------------------------------------------------------------------ BFF router


@pytest.fixture
def http(project: Path) -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        yield c


def test_status_endpoint_is_inactive_by_default(http: TestClient) -> None:
    res = http.get("/api/studio/telegram/status")
    assert res.status_code == 200
    body = res.json()
    assert body["active"] is False
    assert body["agent_path"] is None


def test_start_endpoint_rejects_empty_allowlist(
    http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /start with a token but no allowlist → 400 (not a silent open bot)."""
    _configured(monkeypatch, allowed=None)
    res = http.post("/api/studio/telegram/start", json={"agent_path": "agent.yaml"})
    assert res.status_code == 400, res.text
    assert "allowlist" in res.json()["detail"]


def test_start_endpoint_404_for_missing_agent(
    http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configured(monkeypatch)
    res = http.post("/api/studio/telegram/start", json={"agent_path": "nope.yaml"})
    assert res.status_code == 404, res.text


def test_start_stop_endpoints_and_token_never_leaks(
    http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /start launches; the token never appears in any body; /stop halts it."""
    _configured(monkeypatch)
    # Inject the fake client so the endpoint never reaches the network.
    svc.get_listener_manager()._client_factory = lambda t, c: _FakeClient()  # type: ignore[return-value]

    started = http.post(
        "/api/studio/telegram/start", json={"agent_path": "agent.yaml"}
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["active"] is True
    assert body["allowed_chat_ids"] == ["5"]
    # The bot token must never be serialized anywhere in the response.
    assert _TOKEN not in started.text
    assert "bot_token" not in body and "token" not in body

    status = http.get("/api/studio/telegram/status")
    assert status.json()["active"] is True
    assert _TOKEN not in status.text

    stopped = http.post("/api/studio/telegram/stop")
    assert stopped.status_code == 200
    assert stopped.json()["active"] is False


def test_start_endpoint_409_when_token_already_polled(
    http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token already being polled elsewhere surfaces a distinct 409 (no poll storm)."""
    from himmy.core.process_lock import acquire_process_lock

    _configured(monkeypatch)
    held = acquire_process_lock(svc.telegram_lock_name(_TOKEN))
    try:
        res = http.post(
            "/api/studio/telegram/start", json={"agent_path": "agent.yaml"}
        )
        assert res.status_code == 409, res.text
        assert "already being polled" in res.json()["detail"]
    finally:
        held.release()


def test_shutdown_drains_the_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaving the TestClient context (app shutdown) drains a running listener."""
    # Build the app inside an isolated project so the shutdown hook is exercised.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
        monkeypatch.chdir(root)
        monkeypatch.setenv("HIMMY_CONVERSATIONS_PATH", str(root / "conversations.db"))
        _configured(monkeypatch)
        svc.reset_listener_manager()
        reset_run_store()
        manager = svc.get_listener_manager()
        manager._client_factory = lambda t, c: _FakeClient()  # type: ignore[return-value]
        try:
            with TestClient(create_app()) as c:
                assert (
                    c.post(
                        "/api/studio/telegram/start",
                        json={"agent_path": "agent.yaml"},
                    ).status_code
                    == 200
                )
                assert manager.active
            # Context exit ran the app shutdown event → the listener drained.
            assert not manager.active
        finally:
            run_async(manager.stop())
            svc.reset_listener_manager()
