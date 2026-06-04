"""Core kernel: shared ids, errors, run events, and the event sink protocol."""

from __future__ import annotations

from himmy.core.errors import HimmyError
from himmy.core.events import EventSink, EventType, RunEvent
from himmy.core.ids import new_uuid, utc_now_iso

__all__ = [
    "HimmyError",
    "EventSink",
    "EventType",
    "RunEvent",
    "new_uuid",
    "utc_now_iso",
]
