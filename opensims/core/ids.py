"""Core kernel: id and timestamp helpers used across every kernel."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def new_uuid() -> str:
    """Return a fresh random UUID4 as a string."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
