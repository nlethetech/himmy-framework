"""Google pack: ``gmail_inbox`` + ``gmail_send`` + ``gcal_events`` + ``gcal_create``.

Lets an agent use the *same* connected Google account the Studio GUI uses (tokens in
the keychain via :mod:`himmy.api.studio_google`). Sending email is approval-gated by
default — the human approves before an agent actually sends. Reads (inbox, events) are
read-only. If no account is connected the tools return a friendly hint, never crash.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async Google call to completion from a sync tool handler.

    Tool handlers are synchronous; the Google client is async. If we're already
    inside a running event loop, run the coroutine on a private loop in a worker
    thread so we never re-enter the live loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _worker() -> None:
        result["value"] = asyncio.run(coro)

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    return result["value"]


_SEND_SCHEMA = {
    "type": "object",
    "properties": {
        "to": {"type": "string", "description": "Recipient email address."},
        "subject": {"type": "string", "description": "Subject line."},
        "body": {"type": "string", "description": "Plain-text body."},
    },
    "required": ["to", "body"],
    "additionalProperties": False,
}

_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Event title."},
        "start": {
            "type": "string",
            "description": "Start, RFC3339 (2026-06-09T14:00:00Z) or date if all_day.",
        },
        "end": {"type": "string", "description": "End, same format as start."},
        "all_day": {"type": "boolean", "description": "True for an all-day event."},
    },
    "required": ["summary", "start", "end"],
    "additionalProperties": False,
}

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def register_google_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register the Gmail + Calendar tools backed by the connected Google account."""
    from himmy.api import studio_google as g

    # Mutating Google tools (send mail, create events) are human-in-the-loop by
    # default — mirroring ``comms_allow_send`` in the comms pack — so a prompt
    # injection in an incoming email can't silently send mail or schedule events
    # from the real account. Set ``google_allow_send`` to opt out of the gate.
    gated = not config.google_allow_send

    def gmail_inbox(args: dict[str, Any]) -> dict[str, Any]:
        if not g.status().connected:
            return {"connected": False, "messages": []}
        msgs = _run(g.gmail_list(15))
        return {
            "connected": True,
            "messages": [
                {"from": m.sender, "subject": m.subject, "snippet": m.snippet}
                for m in msgs
            ],
        }

    def gmail_send(args: dict[str, Any]) -> dict[str, Any]:
        if not g.status().connected:
            return {"ok": False, "detail": "no Google account connected"}
        r = _run(
            g.gmail_send(
                str(args["to"]), str(args.get("subject", "")), str(args.get("body", ""))
            )
        )
        return {"ok": r.ok, "detail": r.detail}

    def gcal_events(args: dict[str, Any]) -> dict[str, Any]:
        if not g.status().connected:
            return {"connected": False, "events": []}
        events = _run(g.calendar_list(15))
        return {
            "connected": True,
            "events": [
                {"summary": e.summary, "start": e.start, "end": e.end} for e in events
            ],
        }

    def gcal_create(args: dict[str, Any]) -> dict[str, Any]:
        if not g.status().connected:
            return {"ok": False, "detail": "no Google account connected"}
        e = _run(
            g.calendar_create(
                str(args["summary"]),
                str(args["start"]),
                str(args["end"]),
                all_day=bool(args.get("all_day", False)),
            )
        )
        return {"ok": True, "id": e.id, "summary": e.summary}

    register_local_tool(
        registry,
        name="gmail_inbox",
        read_only=True,
        handler=gmail_inbox,
        description="Read the most recent messages from the connected Gmail inbox.",
        args_json_schema=_EMPTY_SCHEMA,
        metadata={"pack": "google"},
    )
    register_local_tool(
        registry,
        name="gmail_send",
        read_only=False,
        handler=gmail_send,
        description="Send a plain-text email from the connected Google account.",
        args_json_schema=_SEND_SCHEMA,
        requires_approval=gated,
        metadata={"pack": "google"},
    )
    register_local_tool(
        registry,
        name="gcal_events",
        read_only=True,
        handler=gcal_events,
        description="List upcoming events from the connected Google Calendar.",
        args_json_schema=_EMPTY_SCHEMA,
        metadata={"pack": "google"},
    )
    register_local_tool(
        registry,
        name="gcal_create",
        read_only=False,
        handler=gcal_create,
        description="Create an event on the connected Google Calendar.",
        args_json_schema=_CREATE_SCHEMA,
        requires_approval=gated,
        metadata={"pack": "google"},
    )


GOOGLE_TOOL_NAMES = ("gmail_inbox", "gmail_send", "gcal_events", "gcal_create")

__all__ = ["register_google_pack", "GOOGLE_TOOL_NAMES"]
