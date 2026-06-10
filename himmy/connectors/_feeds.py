"""Connector-internal feed parsing: bounded, bozo-aware ``feedparser`` invocation.

``feedparser`` happily chews on anything — an HTML error page, a WAF block screen,
a multi-gigabyte body — and reports trouble only via its ``bozo`` flag, which the
connectors previously ignored (structure drift became a silent empty result).
:func:`parse_feed` is the one guarded entry point: it bounds the input size, raises
a clear :class:`~himmy.core.errors.HimmyError` when a broken parse produced no
entries, and caps how many entries any caller may consume.
"""

from __future__ import annotations

from typing import Any

from himmy.core.errors import HimmyError

#: Refuse to parse feed bodies larger than this (real feeds are a few hundred KB).
MAX_FEED_BYTES = 5 * 1024 * 1024
#: Hard cap on entries consumed from one feed, regardless of the caller's limit.
MAX_FEED_ENTRIES = 200


def entry_cap(limit: int) -> int:
    """Clamp a caller-supplied entry ``limit`` into ``0..MAX_FEED_ENTRIES``."""
    return max(0, min(limit, MAX_FEED_ENTRIES))


def parse_feed(raw: bytes, *, source: str) -> Any:
    """Parse RSS/Atom ``raw`` bytes defensively; fail loudly instead of silently.

    Raises :class:`HimmyError` when the body exceeds :data:`MAX_FEED_BYTES`, when
    ``feedparser`` is not installed (the ``[connectors]`` extra), or when the parse
    was *bozo* and yielded no entries (an error page / drifted format — previously
    a silent empty list). A bozo feed that still yielded entries is tolerated:
    real-world feeds are frequently slightly malformed but usable.
    """
    if len(raw) > MAX_FEED_BYTES:
        raise HimmyError(
            f"feed from {source!r} is {len(raw)} bytes, over the "
            f"{MAX_FEED_BYTES}-byte safety bound; refusing to parse"
        )
    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise HimmyError(
            "feed parsing requires the [connectors] extra "
            "(pip install 'himmy[connectors]')."
        ) from exc
    feed = feedparser.parse(raw)
    if getattr(feed, "bozo", False) and not feed.entries:
        reason = getattr(feed, "bozo_exception", None)
        raise HimmyError(
            f"feed from {source!r} is not parseable RSS/Atom "
            f"(bozo, zero entries): {reason!r}"
        )
    return feed


__all__ = ["MAX_FEED_BYTES", "MAX_FEED_ENTRIES", "entry_cap", "parse_feed"]
