"""API kernel: Studio Telegram inbound-listener router (T3g).

Mounted under ``/api/studio/telegram`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Behavior lives in
:mod:`himmy.api.studio_telegram` (the :class:`TelegramListenerManager` singleton); this
file is the transport:

* ``POST /start`` — launch the inbound long-poll listener for a selected agent;
* ``POST /stop`` — halt it and release the per-token flock;
* ``GET /status`` — a token-free snapshot (active flag + the started config);
* a shutdown event drains the listener so it dies with the server process (FastAPI's
  ``include_router`` merges this into the app lifespan).

Security mirrors the manager: an empty chat allowlist is REJECTED at ``/start`` (a 400)
unless an explicit ``allow_everyone`` override is set, a token already being polled (by
the CLI or another instance) is refused with a 409, and the bot token NEVER appears in
any response body.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from himmy.api import studio_telegram as svc
from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("telegram", tag="studio-telegram")


# ---- request/response models -------------------------------------------------


class TelegramStart(BaseModel):
    """POST body: which agent to run the inbound bot as, with optional model overrides."""

    agent_path: str = Field(..., min_length=1, max_length=500)
    provider: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=200)
    #: Explicit opt-in to run an OPEN bot (no chat allowlist). Defaults False so an empty
    #: allowlist refuses to start — the listener runs the full gated-tool path on input
    #: from whoever messages the bot, so an open default is unsafe.
    allow_everyone: bool = False


class TelegramStatus(BaseModel):
    """A token-free view of the listener (the bot token is never serialized)."""

    active: bool
    agent_path: str | None
    allowed_chat_ids: list[str]
    allow_everyone: bool
    provider: str | None
    model: str | None
    error: str | None


def _view(status: dict[str, Any]) -> TelegramStatus:
    return TelegramStatus(
        active=bool(status.get("active")),
        agent_path=status.get("agent_path"),
        allowed_chat_ids=list(status.get("allowed_chat_ids") or []),
        allow_everyone=bool(status.get("allow_everyone")),
        provider=status.get("provider"),
        model=status.get("model"),
        error=status.get("error"),
    )


def _validate_agent_path(rel_path: str) -> None:
    """Reject a listener pointing at a missing/escaping agent spec up front."""
    from himmy.api.studio_service import resolve_spec_path

    try:
        resolve_spec_path(rel_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---- endpoints ----------------------------------------------------------------


@router.get("/status", response_model=TelegramStatus)
async def get_status() -> TelegramStatus:
    """The current listener state (never includes the bot token)."""
    return _view(svc.get_listener_manager().status())


@router.post("/start", response_model=TelegramStatus)
async def start(body: TelegramStart) -> TelegramStatus:
    """Start the inbound listener for ``agent_path``.

    400 when the token is missing or the allowlist is empty (without ``allow_everyone``);
    404 when the agent spec doesn't resolve; 409 when this token is already being polled.
    """
    _validate_agent_path(body.agent_path)
    manager = svc.get_listener_manager()
    try:
        status = manager.start(
            agent_path=body.agent_path,
            provider=body.provider,
            model=body.model,
            allow_everyone=body.allow_everyone,
        )
    except svc.TelegramListenerBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except svc.TelegramListenerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _view(status)


@router.post("/stop", response_model=TelegramStatus)
async def stop() -> TelegramStatus:
    """Stop the listener (idempotent — stopping a stopped listener is fine)."""
    status = await svc.get_listener_manager().stop()
    return _view(status)


# ---- lifecycle (merged into the app lifespan by include_router) ----------------


async def _drain_listener() -> None:
    """Stop the listener on shutdown so it dies with the server process."""
    await svc.get_listener_manager().stop()


router.add_event_handler("shutdown", _drain_listener)


__all__ = ["router"]
