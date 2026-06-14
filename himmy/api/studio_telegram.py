"""Studio Telegram inbound-listener manager (T3g).

Studio already has an OUTBOUND Telegram connection (an agent can ``send_telegram``);
what it lacked is a way to start/stop the INBOUND long-poll listener the CLI's
``himmy telegram`` runs — the loop that turns each message a user sends the bot into
an agent reply. This module is that listener, modelled on
:class:`himmy.api.routines.RoutineScheduler`:

* **one per process** as a module-wide singleton (``_MANAGER`` /
  :func:`get_listener_manager`, mirroring :data:`~himmy.api.routines._SCHEDULER` /
  :func:`~himmy.api.routines.get_scheduler`), with idempotent
  :meth:`TelegramListenerManager.start` / :meth:`~TelegramListenerManager.stop` and a
  shutdown drain wired into the app lifespan by the router;
* **no parallel run path** — every inbound turn is executed through
  :func:`himmy.api.studio_service.stream_agent_run`, the SAME canonical run pipeline the
  Studio Run/Chat surfaces use, so each message records a run row, applies guardrails,
  and PAUSES (never auto-approves) at an approval-gated tool;
* **config from the existing Telegram Connection** — the bot token and the chat
  allowlist are read from the toolkit config / secrets layer (never stored anew here);
* **allowlist HARD-enforced** — the manager REFUSES to start with an empty allowlist (the
  listener runs the full gated-tool path on attacker-influenced input, so an open bot is
  not a safe default); an explicit ``allow_everyone`` override is required to run open;
* **cross-process single-flight** — a per-bot-token ``flock`` (held for the listener's
  lifetime) means ``himmy telegram`` (CLI) and this Studio listener can never both
  long-poll one token; a second poller is refused with a 409 rather than spinning up a
  competing ``getUpdates`` storm.

Conversation durability (T2.3) rides the shared
:class:`~himmy.services.storage.conversations.ConversationStore`: each chat keeps a
durable per-chat thread keyed ``telegram:<chat_id>`` so context survives a restart and
the conversation is browsable in Studio Chats alongside CLI/Studio threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from himmy.api.studio_canonical import resolve_canonical_storage
from himmy.core.process_lock import (
    ProcessLockBusy,
    ProcessLockHandle,
    acquire_process_lock,
)
from himmy.services.storage.conversations import (
    ConversationStore,
    get_conversation_store,
)
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.telegram import TelegramBot, TelegramClient

logger = logging.getLogger("himmy.api.studio_telegram")

#: ``origin`` stamped on conversations the inbound bot creates (provenance/filtering).
ORIGIN_TELEGRAM = "telegram"

#: Conversation id prefix so a chat's durable thread is unambiguous in the shared store.
_CONV_PREFIX = "telegram:"

#: Reply length cap — a Telegram message is bounded; never blast a huge dump at the chat.
_REPLY_MAX_CHARS = 3500


def telegram_lock_name(token: str) -> str:
    """The cross-process flock name guarding a single bot token's long-poll.

    Keyed by a short, NON-reversible digest of the token (never the token itself, so a
    lock file name can't leak the secret) so the CLI ``himmy telegram`` and the Studio
    listener agree on one lock per token.
    """
    import hashlib

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"telegram-{digest}"


class TelegramListenerError(RuntimeError):
    """A start request that cannot be honored (bad/missing config), surfaced as a 400."""


class TelegramListenerBusy(RuntimeError):
    """The token is already being polled (here or by ``himmy telegram``), surfaced as 409."""


class TelegramListenerManager:
    """The inbound long-poll listener. One per process; idempotent start/stop + drain.

    ``client_factory`` and ``runner`` are injectable so the loop is testable offline (a
    fake Telegram client + a per-message stub) without ever reaching the network or a
    real model. The production wiring uses the real :class:`TelegramClient` and the
    canonical :func:`stream_agent_run` pipeline.
    """

    def __init__(
        self,
        *,
        client_factory: Any | None = None,
        runner: Any | None = None,
        store: ConversationStore | None = None,
        poll_timeout: int = 25,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory
        self._runner = runner or self._run_turn_canonical
        self._store_override = store
        self._poll_timeout = poll_timeout

        self._task: asyncio.Task[None] | None = None
        self._lock: ProcessLockHandle | None = None
        self._stop = False
        # Started-config snapshot (for `status`); the token itself is NEVER stored here so
        # it cannot leak into a response body.
        self._agent_path: str | None = None
        self._allowed_chat_ids: list[str] = []
        self._allow_everyone = False
        self._provider: str | None = None
        self._model: str | None = None
        self._error: str | None = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True while the long-poll loop task is running."""
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        """A token-free snapshot for ``GET /status`` (never includes the bot token)."""
        return {
            "active": self.active,
            "agent_path": self._agent_path,
            "allowed_chat_ids": list(self._allowed_chat_ids),
            "allow_everyone": self._allow_everyone,
            "provider": self._provider,
            "model": self._model,
            "error": self._error,
        }

    def start(
        self,
        *,
        agent_path: str,
        provider: str | None = None,
        model: str | None = None,
        allow_everyone: bool = False,
        config: ToolkitConfig | None = None,
    ) -> dict[str, Any]:
        """Start the listener (idempotent: a second start while active is refused as busy).

        Resolves the bot token + chat allowlist from the Telegram Connection (toolkit
        config / secrets), HARD-rejects an empty allowlist unless ``allow_everyone`` is
        explicitly set, acquires the per-token flock (so the CLI and Studio can't both
        poll), and launches the long-poll loop on the running event loop. Raises
        :class:`TelegramListenerError` for bad config (400) and
        :class:`TelegramListenerBusy` for a token already being polled (409).
        """
        if self.active:
            raise TelegramListenerBusy("a Telegram listener is already running")

        # The bot token + allowlist are read from the Telegram Connection via the
        # secrets/env layer (``from_env`` resolves ``HIMMY_TELEGRAM_BOT_TOKEN`` through the
        # secrets backend + ``HIMMY_TELEGRAM_ALLOWED_CHATS``); they are NEVER stored anew
        # here. ``from_env`` (not ``from_sources``) is deliberate: the telegram secret is
        # not a himmy.toml field, and ``from_sources`` only overlays the toml-backed keys.
        cfg = config or ToolkitConfig.from_env()
        token = cfg.telegram_bot_token
        if not token:
            raise TelegramListenerError(
                "no Telegram bot token configured — set it under Connections → Telegram"
            )
        allowed = [str(c) for c in cfg.telegram_allowed_chat_ids if str(c).strip()]
        if not allowed and not allow_everyone:
            # SECURITY (reviewer must_fix): the listener runs the full gated-tool/approvals
            # path on whatever a sender types. An empty allowlist would mean ANY stranger
            # who finds the bot can drive it, so refuse to start open by default.
            raise TelegramListenerError(
                "refusing to start with an empty chat allowlist — set allowed chat ids "
                "under Connections → Telegram (or pass allow_everyone to run an open bot)"
            )

        # Cross-process single-flight: hold the per-token flock for the listener's whole
        # lifetime so `himmy telegram` and this listener can never both poll one token.
        try:
            self._lock = acquire_process_lock(telegram_lock_name(token))
        except ProcessLockBusy as exc:
            raise TelegramListenerBusy(
                "this bot token is already being polled by another listener "
                "(the `himmy telegram` CLI or another Studio instance)"
            ) from exc

        self._stop = False
        self._error = None
        self._agent_path = agent_path
        self._allowed_chat_ids = allowed
        self._allow_everyone = allow_everyone
        self._provider = provider
        self._model = model

        client = self._client_factory(token, cfg)
        bot = TelegramBot(
            client,
            self._make_handler(agent_path, provider, model),
            poll_timeout=self._poll_timeout,
            # When allow_everyone is set the allowlist is deliberately empty (open bot);
            # otherwise the non-empty allowlist is enforced inside the bot loop too.
            allowed_chat_ids=None if allow_everyone else allowed,
        )
        self._task = asyncio.create_task(
            self._serve(bot), name="himmy-telegram-listener"
        )
        return self.status()

    async def stop(self) -> dict[str, Any]:
        """Stop the loop, await its drain, and release the per-token flock (idempotent)."""
        self._stop = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        self._release_lock()
        return self.status()

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    async def _serve(self, bot: TelegramBot) -> None:
        """Drive the bot until stopped; a poll error backs off rather than hot-looping.

        The offset (the ``getUpdates`` cursor) lives here rather than in
        :meth:`TelegramBot.run` so the loop can interleave the stop flag and a back-off on
        error without losing its place in the update stream.
        """
        offset: int | None = None
        try:
            while not self._stop:
                try:
                    offset = await bot.poll_once(offset)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - one bad poll never kills the listener
                    logger.exception("telegram listener poll failed")
                    # Back off so a persistent error (network/token) is not a hot loop.
                    await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass
        finally:
            # The loop owns lock release on a self-terminating exit; stop() also releases
            # (idempotent) so an externally-cancelled task still frees the token.
            self._release_lock()

    # -- per-message execution (canonical path) -------------------------------

    def _make_handler(
        self, agent_path: str, provider: str | None, model: str | None
    ) -> Any:
        async def _handle(chat_id: str, text: str) -> str:
            return await self._runner(
                agent_path=agent_path,
                provider=provider,
                model=model,
                chat_id=chat_id,
                text=text,
            )

        return _handle

    def _store(self) -> Any:
        # The shared conversation store: a SQLite ``ConversationStore`` OR, under a Postgres
        # DSN (K4), the ``PostgresConversationStore`` mirror — both expose the same
        # save_thread/load_thread/list_summaries surface this agent uses (duck-typed, no
        # nominal base), so the return is typed ``Any``.
        return self._store_override or get_conversation_store()

    async def _run_turn_canonical(
        self,
        *,
        agent_path: str,
        provider: str | None,
        model: str | None,
        chat_id: str,
        text: str,
    ) -> str:
        """Run one inbound message through the canonical Studio pipeline + record it.

        Loads the per-chat durable thread (T2.3) as prior history, runs the agent via
        :func:`stream_agent_run` (records a run row + applies guardrails + pauses on an
        approval-gated tool), then appends the new user/assistant turn to the durable
        thread so the next message keeps context and the conversation is browsable in
        Studio Chats. A run that paused for approval (or errored) returns a short, honest
        note to the chat rather than a blank reply.
        """
        from himmy.agents.base_agent.thread import Message, MessageRole
        from himmy.api import studio_service

        conv_id = f"{_CONV_PREFIX}{chat_id}"
        store = self._store()
        thread = store.load_thread(conv_id)
        history = _thread_to_history(thread)

        try:
            spec = studio_service.load_studio_spec(
                agent_path, provider=provider, model=model
            )
        except (FileNotFoundError, ValueError) as exc:
            return f"(agent unavailable: {exc})"

        reply = ""
        paused = False
        errored: str | None = None
        async for event in studio_service.stream_agent_run(
            spec,
            text,
            history=history,
            provider=provider,
            model=model,
            agent_path=agent_path,
            canonical_storage=resolve_canonical_storage(),
        ):
            kind = event.get("type")
            if kind == "message":
                reply = str(event.get("text") or reply)
            elif kind == "done":
                reply = str(event.get("output_text") or reply)
            elif kind == "paused":
                paused = True
            elif kind == "error":
                errored = str(event.get("message") or "the run failed")

        if errored is not None:
            return f"(sorry — {errored})"
        if paused:
            return (
                "I started on that but a step needs your approval — open Studio → "
                "Approvals to allow or deny it."
            )

        # Persist the turn so context survives a restart and the chat is browsable.
        if thread is None:
            from himmy.agents.base_agent.thread import ChatThread

            thread = ChatThread(agent_id=spec.to_persona().agent_id)
        thread.append_message(Message(role=MessageRole.USER, content=text))
        if reply:
            thread.append_message(Message(role=MessageRole.ASSISTANT, content=reply))
        with contextlib.suppress(Exception):
            store.save_thread(
                conv_id,
                thread,
                origin=ORIGIN_TELEGRAM,
                agent_path=agent_path,
                provider=provider,
            )
        return _cap(reply, _REPLY_MAX_CHARS)


def _thread_to_history(thread: Any) -> list[dict[str, Any]]:
    """Flatten a durable ChatThread into the user/assistant history stream_agent_run wants."""
    if thread is None:
        return []
    from himmy.agents.base_agent.thread import MessageRole

    history: list[dict[str, Any]] = []
    for msg in thread.messages:
        role = getattr(msg, "role", None)
        if role == MessageRole.USER:
            history.append({"role": "user", "content": msg.content})
        elif role == MessageRole.ASSISTANT:
            history.append({"role": "assistant", "content": msg.content})
    return history


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _default_client_factory(token: str, cfg: ToolkitConfig) -> TelegramClient:
    """The production client: long-poll a bit longer than the toolkit HTTP timeout."""
    return TelegramClient(token, timeout=cfg.http_timeout + 30)


_MANAGER: TelegramListenerManager | None = None


def get_listener_manager() -> TelegramListenerManager:
    """The process-wide singleton (created on first use)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TelegramListenerManager()
    return _MANAGER


def reset_listener_manager() -> None:
    """Drop the singleton (tests). The caller must have stopped it first."""
    global _MANAGER
    _MANAGER = None


__all__ = [
    "ORIGIN_TELEGRAM",
    "TelegramListenerBusy",
    "TelegramListenerError",
    "TelegramListenerManager",
    "get_listener_manager",
    "reset_listener_manager",
    "telegram_lock_name",
]
