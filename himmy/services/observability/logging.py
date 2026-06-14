"""Optional structured JSON logging, toggled by ``HIMMY_LOG_FORMAT=json``.

Default behavior is unchanged: unless ``HIMMY_LOG_FORMAT`` is set to ``json``
this module installs nothing and existing users keep the standard human-readable
log format. When opted in, the root logger's handlers are switched to a JSON
formatter that emits one object per line — friendly to log shippers
(Loki/CloudWatch/ELK) without adding a dependency.

Each JSON line carries at least: ``timestamp`` (ISO-8601, UTC), ``level``,
``logger``, and ``message``. When a request/trace id is present in context (set
by the API's request-context middleware via :func:`bind_request_id`) it is
included as ``request_id`` so a log line can be correlated to the
``X-Request-ID`` already echoed to clients — reusing that existing mechanism
rather than inventing a new one.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import os
from typing import Any

# Holds the current request/trace id for the active task, so log records emitted
# anywhere during request handling can be stamped with it without threading the
# id through every call site.
_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "himmy_request_id", default=None
)

# Standard LogRecord attributes we never want to duplicate into the JSON "extra"
# fields (they are either rendered explicitly or are noise).
_RESERVED_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)

_TRUTHY = {"1", "true", "yes", "on"}


def bind_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    """Bind ``request_id`` to the current context; returns a reset token.

    The token can be passed to :func:`reset_request_id` to restore the previous
    value (use this in a middleware ``finally`` so ids do not leak between
    requests served on the same worker task).
    """
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Restore the request-id context to the value captured by ``token``."""
    _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    """Return the request/trace id bound to the current context, if any."""
    return _REQUEST_ID.get()


class JsonLogFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` to a compact JSON line."""
        payload: dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or current_request_id()
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Surface any user-supplied ``extra=`` fields without clobbering the core
        # keys or leaking internal LogRecord machinery.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in payload or key == "request_id":
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value

        return json.dumps(payload, default=str)


def json_logging_enabled() -> bool:
    """Whether ``HIMMY_LOG_FORMAT`` requests JSON output."""
    return os.environ.get("HIMMY_LOG_FORMAT", "").strip().lower() == "json"


def configure_logging() -> None:
    """Install the JSON formatter on the root logger when opted in; else no-op.

    Idempotent and safe to call from any entry point. When
    ``HIMMY_LOG_FORMAT=json`` it ensures the root logger has a stream handler and
    swaps every handler's formatter to :class:`JsonLogFormatter`. With the env var
    unset (the default) it does nothing at all, so the existing human-readable
    output is preserved byte-for-byte for current users.
    """
    if not json_logging_enabled():
        return
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    formatter = JsonLogFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)


__all__ = [
    "JsonLogFormatter",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "json_logging_enabled",
    "reset_request_id",
]
