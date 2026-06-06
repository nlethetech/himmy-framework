"""Telegram pack + bot runner: let himmy agents live in Telegram.

Two surfaces over the Telegram Bot API (HTTP, ``httpx`` only — no SDK):

* ``send_telegram`` — an outbound tool so an agent can proactively message a chat
  (approval-gated like the rest of the ``comms`` "act" tools unless
  ``comms_allow_send`` is set). The bot token and a default chat id come from the
  toolkit config/env (``HIMMY_TELEGRAM_BOT_TOKEN`` / ``HIMMY_TELEGRAM_CHAT_ID``),
  never from the model.
* :class:`TelegramBot` — a long-poll loop that turns an ``agent.yaml`` into a live
  Telegram bot: each incoming message runs the agent and the reply is sent back. The
  loop is injectable (a fake client + a stop predicate) so it is testable offline; the
  CLI's ``himmy telegram`` wires the real client.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from himmy.core.errors import HimmyError
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig

#: Default Telegram Bot API root.
TELEGRAM_API_ROOT = "https://api.telegram.org"


class TelegramError(HimmyError):
    """A Telegram Bot API call returned ``ok: false`` (carries the description)."""


class TelegramClient:
    """A tiny async Telegram Bot API client (``sendMessage`` + ``getUpdates``)."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = TELEGRAM_API_ROOT,
        timeout: float = 30.0,
        transport: Any = None,
    ) -> None:
        """Wrap a bot ``token``; ``transport`` lets tests inject an httpx mock."""
        self._token = token
        self._timeout = timeout
        self._base = f"{base_url}/bot{token}"
        self._transport = transport

    async def _post(self, method: str, payload: dict[str, Any]) -> Any:
        import httpx

        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            resp = await client.post(f"{self._base}/{method}", json=payload)
            data = resp.json()
        if not data.get("ok"):
            raise TelegramError(
                f"telegram {method} failed: {data.get('description', 'unknown error')}"
            )
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        """Return the bot's own profile (``getMe``) — a cheap token-validity check."""
        return await self._post("getMe", {})

    async def send_message(self, chat_id: Any, text: str) -> dict[str, Any]:
        """Send ``text`` to ``chat_id`` (returns the sent Message)."""
        return await self._post("sendMessage", {"chat_id": chat_id, "text": text})

    async def get_updates(
        self, *, offset: int | None = None, timeout: int = 25
    ) -> list[dict[str, Any]]:
        """Long-poll for new updates since ``offset`` (returns the update list)."""
        payload: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        return await self._post("getUpdates", payload) or []


_SEND_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The message text to send."},
        "chat_id": {
            "type": "string",
            "description": "Target chat id (defaults to HIMMY_TELEGRAM_CHAT_ID).",
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}

#: Builds the client used by ``send_telegram`` (the injectable seam for tests).
ClientFactory = Callable[[ToolkitConfig], TelegramClient]


def register_telegram_pack(
    registry: ToolRegistry,
    config: ToolkitConfig,
    *,
    client_factory: ClientFactory | None = None,
) -> None:
    """Register ``send_telegram`` (approval-gated unless ``comms_allow_send``)."""
    gated = not config.comms_allow_send
    make_client = client_factory or (
        lambda cfg: TelegramClient(
            cfg.telegram_bot_token or "", timeout=cfg.http_timeout
        )
    )

    async def send_telegram(args: dict[str, Any]) -> dict[str, Any]:
        if not config.telegram_bot_token:
            raise ToolSecurityError(
                "send_telegram needs HIMMY_TELEGRAM_BOT_TOKEN configured"
            )
        chat_id = args.get("chat_id") or config.telegram_default_chat_id
        if not chat_id:
            raise ToolSecurityError(
                "send_telegram needs a chat_id (or HIMMY_TELEGRAM_CHAT_ID)"
            )
        message = await make_client(config).send_message(chat_id, str(args["text"]))
        return {
            "sent": True,
            "chat_id": str(chat_id),
            "message_id": message.get("message_id"),
        }

    register_local_tool(
        registry,
        name="send_telegram",
        read_only=False,
        handler=send_telegram,
        description="Send a message to a Telegram chat via the configured bot.",
        args_json_schema=_SEND_SCHEMA,
        requires_approval=gated,
        metadata={"pack": "telegram"},
    )


TELEGRAM_TOOL_NAMES = ("send_telegram",)

#: ``handler(chat_id, text) -> reply`` — the agent turn the bot runs per message.
MessageHandler = Callable[[str, str], Awaitable[str]]


class TelegramBot:
    """Long-poll loop that runs ``handler`` for each incoming Telegram message."""

    def __init__(
        self,
        client: TelegramClient,
        handler: MessageHandler,
        *,
        poll_timeout: int = 25,
    ) -> None:
        """Drive ``client`` with ``handler``; ``poll_timeout`` is the long-poll wait."""
        self._client = client
        self._handler = handler
        self._poll_timeout = poll_timeout

    async def poll_once(self, offset: int | None) -> int | None:
        """Fetch one batch of updates, reply to each message, return the next offset."""
        updates = await self._client.get_updates(
            offset=offset, timeout=self._poll_timeout
        )
        for update in updates:
            offset = int(update["update_id"]) + 1
            message = update.get("message") or update.get("edited_message") or {}
            text = message.get("text")
            chat_id = (message.get("chat") or {}).get("id")
            if not text or chat_id is None:
                continue  # non-text update (sticker, join, …) — nothing to answer
            reply = await self._handler(str(chat_id), str(text))
            if reply:
                await self._client.send_message(chat_id, reply)
        return offset

    async def run(self, *, stop: Callable[[], bool] | None = None) -> None:
        """Poll until ``stop()`` returns True (runs forever when ``stop`` is None)."""
        offset: int | None = None
        while not (stop and stop()):
            offset = await self.poll_once(offset)


__all__ = [
    "TelegramClient",
    "TelegramError",
    "TelegramBot",
    "register_telegram_pack",
    "TELEGRAM_TOOL_NAMES",
    "TELEGRAM_API_ROOT",
]
