"""HIMMY_LOG_FORMAT=json: structured JSON log lines.

When opted in, each emitted log line must parse as JSON and carry the core keys
(timestamp/level/logger/message) plus the request_id when one is bound in
context. With the env var unset, the formatter is not installed (default
human-readable output is preserved) — asserted via json_logging_enabled().
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from himmy.services.observability.logging import (
    JsonLogFormatter,
    bind_request_id,
    configure_logging,
    json_logging_enabled,
    reset_request_id,
)


def _emit(record_factory) -> str:
    """Run one record through the JSON formatter and return the line."""
    formatter = JsonLogFormatter()
    return formatter.format(record_factory())


def _make_record(msg: str = "hello world") -> logging.LogRecord:
    return logging.LogRecord(
        name="himmy.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_json_formatter_emits_valid_json_with_core_keys() -> None:
    line = _emit(_make_record)
    parsed = json.loads(line)  # must be valid JSON

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "himmy.api"
    assert parsed["message"] == "hello world"
    # timestamp is ISO-8601 (parses without error).
    import datetime as dt

    dt.datetime.fromisoformat(parsed["timestamp"])
    assert {"timestamp", "level", "logger", "message"} <= set(parsed)


def test_json_line_includes_bound_request_id() -> None:
    token = bind_request_id("req-abc123")
    try:
        parsed = json.loads(_emit(_make_record))
    finally:
        reset_request_id(token)
    assert parsed["request_id"] == "req-abc123"


def test_configure_logging_installs_json_only_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default (unset): no JSON, formatter not installed.
    monkeypatch.delenv("HIMMY_LOG_FORMAT", raising=False)
    assert json_logging_enabled() is False

    # Opt in: configure_logging swaps root handler formatters to JSON and a real
    # emitted record parses as JSON with the expected keys.
    monkeypatch.setenv("HIMMY_LOG_FORMAT", "json")
    assert json_logging_enabled() is True

    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root.addHandler(handler)
    try:
        configure_logging()
        test_logger = logging.getLogger("himmy.test")
        test_logger.setLevel(logging.INFO)
        test_logger.info("structured line")
        handler.flush()
        out = stream.getvalue().strip().splitlines()[-1]
        parsed = json.loads(out)
        assert parsed["message"] == "structured line"
        assert parsed["logger"] == "himmy.test"
        assert {"timestamp", "level", "logger", "message"} <= set(parsed)
    finally:
        root.removeHandler(handler)
        root.setLevel(prior_level)
