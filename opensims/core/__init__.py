"""Core kernel: shared ids, errors, run events, and the event sink protocol."""

from __future__ import annotations

from opensims.core.errors import OpenSimsError
from opensims.core.events import EventSink, EventType, RunEvent
from opensims.core.ids import new_uuid, utc_now_iso

__all__ = [
    "OpenSimsError",
    "EventSink",
    "EventType",
    "RunEvent",
    "new_uuid",
    "utc_now_iso",
]
