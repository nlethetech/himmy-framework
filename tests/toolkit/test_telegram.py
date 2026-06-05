"""Tests for the telegram pack (send_telegram) and the TelegramBot loop, offline."""

from __future__ import annotations

from typing import Any

from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.telegram import (
    TelegramBot,
    TelegramClient,
    register_telegram_pack,
)
from tests.conftest import run_async


class _FakeClient:
    """Records sent messages and serves scripted update batches."""

    def __init__(self, batches: list[list[dict[str, Any]]] | None = None) -> None:
        self.sent: list[tuple[Any, str]] = []
        self._batches = list(batches or [])

    async def send_message(self, chat_id: Any, text: str) -> dict[str, Any]:
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    async def get_updates(
        self, *, offset: int | None = None, timeout: int = 25
    ) -> list[dict[str, Any]]:
        return self._batches.pop(0) if self._batches else []


# ----------------------------------------------------------------- send_telegram


def _service(client: _FakeClient, **cfg_over: Any) -> ToolService:
    config = ToolkitConfig(
        telegram_bot_token="tok",
        telegram_default_chat_id="999",
        comms_allow_send=True,  # ungate for the test
        **cfg_over,
    )
    registry = ToolRegistry()
    register_telegram_pack(registry, config, client_factory=lambda _c: client)
    return ToolService(registry)


def test_send_telegram_uses_default_chat() -> None:
    client = _FakeClient()
    res = run_async(
        _service(client).execute(
            ToolInvocation(tool_name="send_telegram", args={"text": "hi there"})
        )
    )
    assert res.outcome == "success"
    assert res.result["sent"] is True
    assert client.sent == [("999", "hi there")]


def test_send_telegram_explicit_chat_overrides_default() -> None:
    client = _FakeClient()
    run_async(
        _service(client).execute(
            ToolInvocation(
                tool_name="send_telegram", args={"text": "yo", "chat_id": "42"}
            )
        )
    )
    assert client.sent == [("42", "yo")]


def test_send_telegram_without_token_fails_cleanly() -> None:
    client = _FakeClient()
    config = ToolkitConfig(comms_allow_send=True)  # no token
    registry = ToolRegistry()
    register_telegram_pack(registry, config, client_factory=lambda _c: client)
    res = run_async(
        ToolService(registry).execute(
            ToolInvocation(tool_name="send_telegram", args={"text": "x"})
        )
    )
    assert res.outcome != "success"
    assert client.sent == []


def test_send_telegram_is_approval_gated_by_default() -> None:
    config = ToolkitConfig(telegram_bot_token="tok")  # comms_allow_send defaults False
    registry = ToolRegistry()
    register_telegram_pack(registry, config)
    assert registry.get("send_telegram").requires_approval is True


# --------------------------------------------------------------------- the bot


def test_bot_replies_to_text_messages_and_advances_offset() -> None:
    client = _FakeClient(
        batches=[
            [
                {
                    "update_id": 10,
                    "message": {"chat": {"id": 5}, "text": "hello"},
                },
                {
                    "update_id": 11,
                    "message": {"chat": {"id": 7}, "text": "world"},
                },
            ]
        ]
    )

    async def handler(chat_id: str, text: str) -> str:
        return f"echo:{text}"

    bot = TelegramBot(client, handler)  # type: ignore[arg-type]
    next_offset = run_async(bot.poll_once(None))
    assert next_offset == 12  # last update_id (11) + 1
    assert client.sent == [(5, "echo:hello"), (7, "echo:world")]


def test_bot_skips_non_text_updates() -> None:
    client = _FakeClient(
        batches=[[{"update_id": 1, "message": {"chat": {"id": 5}}}]]  # no text
    )

    async def handler(chat_id: str, text: str) -> str:
        return "should not be called"

    bot = TelegramBot(client, handler)  # type: ignore[arg-type]
    next_offset = run_async(bot.poll_once(None))
    assert next_offset == 2
    assert client.sent == []


def test_bot_run_stops_on_predicate() -> None:
    client = _FakeClient(batches=[[]])
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # allow two poll passes then stop

    async def handler(chat_id: str, text: str) -> str:
        return ""

    bot = TelegramBot(client, handler)  # type: ignore[arg-type]
    run_async(bot.run(stop=stop))
    assert calls["n"] >= 2


def test_client_builds_expected_url() -> None:
    client = TelegramClient("abc", base_url="https://example.test")
    assert client._base == "https://example.test/botabc"
