"""Core kernel: id and timestamp helpers used across every kernel."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta


def new_uuid() -> str:
    """Return a fresh random UUID4 as a string."""
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def iso_plus_seconds(base_iso: str, seconds: float) -> str:
    """Return ``base_iso`` (an ISO-8601 instant) advanced by ``seconds`` as an ISO string.

    The shared lease-deadline arithmetic for the leased run work-queue: a claimed run's
    ``lease_expires_at`` is ``now + lease_seconds``. Keeping it in one helper means the
    SQLite, Postgres, and in-memory backends compute the same boundary. A malformed/blank
    base falls back to the current UTC instant so a missing timestamp never raises mid-claim.
    """
    try:
        base = datetime.fromisoformat(base_iso)
    except (ValueError, TypeError):
        base = datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return (base + timedelta(seconds=seconds)).isoformat()
