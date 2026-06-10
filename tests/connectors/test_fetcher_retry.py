"""Tests for HttpxFetcher's retry/backoff (mock transport, never the network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from himmy.connectors.fetcher import HttpxFetcher, _parse_retry_after


class _ScriptedTransport(httpx.BaseTransport):
    """Plays back a script of responses/exceptions, one per request."""

    def __init__(self, script: list[httpx.Response | Exception]) -> None:
        self.script = list(script)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _fetcher(transport: httpx.BaseTransport, *, retries: int = 2) -> HttpxFetcher:
    # Zero backoff so tests never actually sleep.
    return HttpxFetcher(
        retries=retries, backoff_base=0.0, backoff_max=0.0, transport=transport
    )


def test_429_then_200_is_retried_to_success() -> None:
    """A rate-limited first attempt is retried and the batch survives."""
    transport = _ScriptedTransport(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, text="ok"),
        ]
    )
    assert _fetcher(transport).get_text("https://example.org/x") == "ok"
    assert transport.calls == 2


def test_503_then_200_is_retried_for_bytes() -> None:
    """Transient 5xx responses are retried on the bytes path too."""
    transport = _ScriptedTransport(
        [httpx.Response(503), httpx.Response(200, content=b"payload")]
    )
    assert _fetcher(transport).get_bytes("https://example.org/x") == b"payload"
    assert transport.calls == 2


def test_transport_error_then_200_is_retried() -> None:
    """A connection reset / DNS blip is retried, not fatal."""
    transport = _ScriptedTransport(
        [httpx.ConnectError("boom"), httpx.Response(200, text="ok")]
    )
    assert _fetcher(transport).get_text("https://example.org/x") == "ok"
    assert transport.calls == 2


def test_retries_are_bounded_then_raise() -> None:
    """Persistent 503s exhaust the budget (retries+1 attempts) and raise."""
    transport = _ScriptedTransport([httpx.Response(503)] * 5)
    with pytest.raises(httpx.HTTPStatusError):
        _fetcher(transport, retries=2).get_text("https://example.org/x")
    assert transport.calls == 3


def test_retries_zero_restores_single_attempt() -> None:
    """retries=0 opts out: exactly one request, immediate failure."""
    transport = _ScriptedTransport([httpx.Response(429)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        _fetcher(transport, retries=0).get_text("https://example.org/x")
    assert transport.calls == 1


def test_non_retryable_status_fails_fast() -> None:
    """A 404 is not transient — no retry is attempted."""
    transport = _ScriptedTransport([httpx.Response(404)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        _fetcher(transport).get_text("https://example.org/x")
    assert transport.calls == 1


def test_persistent_transport_error_raises_after_budget() -> None:
    """Transport errors are also bounded by the retry budget."""
    transport = _ScriptedTransport([httpx.ConnectError("boom")] * 5)
    with pytest.raises(httpx.ConnectError):
        _fetcher(transport, retries=1).get_text("https://example.org/x")
    assert transport.calls == 2


def test_retry_after_seconds_is_honored_and_capped() -> None:
    """A numeric Retry-After drives the delay, capped at backoff_max."""
    fetcher = HttpxFetcher(backoff_base=0.5, backoff_max=30.0)
    assert fetcher._backoff_delay(0, "7") == 7.0
    assert fetcher._backoff_delay(0, "9999") == 30.0  # capped


def test_retry_after_http_date_is_honored() -> None:
    """The HTTP-date form of Retry-After parses into a sane positive delay."""
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=10))
    delay = _parse_retry_after(when)
    assert delay is not None
    assert 5.0 < delay <= 10.0


def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    """Garbage Retry-After falls back to exponential backoff (with jitter)."""
    fetcher = HttpxFetcher(backoff_base=1.0, backoff_max=100.0)
    assert 1.0 <= fetcher._backoff_delay(0, "garbage") <= 1.25
    assert 4.0 <= fetcher._backoff_delay(2, None) <= 5.0  # 1.0 * 2**2 + jitter


def test_per_call_timeout_override() -> None:
    """get_text/get_bytes accept a per-call timeout overriding the default."""
    seen: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, text="ok")

    fetcher = HttpxFetcher(timeout=30.0, transport=httpx.MockTransport(handler))
    fetcher.get_text("https://example.org/x")
    fetcher.get_text("https://example.org/x", timeout=5.0)
    fetcher.get_bytes("https://example.org/x", timeout=2.5)
    assert seen == [30.0, 5.0, 2.5]
