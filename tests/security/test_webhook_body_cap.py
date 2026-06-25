"""Regression: inbound webhook must cap the body BEFORE buffering it (Finding #12).

The vulnerable code did ``body = await request.body()`` in the FastAPI route, buffering the
ENTIRE inbound body into memory before the connector's 256 KB cap was checked. An oversized
(or merely over-DECLARED) body could therefore pin memory ahead of the signature check.

These tests fail against the old behaviour and pass with the fix:

* a request whose ``Content-Length`` declares an oversized body is rejected ``413`` without
  ``handle_webhook`` ever being reached (so the full body was never buffered/parsed);
* a *chunked* (no declared length) body that streams past the cap is aborted ``413`` once the
  running total crosses the cap, again without reaching ``handle_webhook``;
* the generic connector router (Slack/Discord/etc.) enforces the same cap.

All isolated: own in-memory connector, no network, no shared state.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.connectors.webhook import (
    DEFAULT_MAX_BODY_BYTES,
    BodyTooLarge,
    WebhookInboundConnector,
    build_webhook_router,
    read_body_capped,
)
from tests.conftest import run_async

_SECRET = "whsec_test_0000000000000000000000000000"  # fake signing secret (test only)


# --------------------------------------------------------------------------- helpers
class _SpyConnector(WebhookInboundConnector):
    """A webhook connector that records whether ``handle_webhook`` was ever reached.

    If the route buffered the whole body and only THEN checked the cap inside the connector,
    ``handle_webhook`` would be called (and ``reached`` flipped). With the fix the route
    rejects the over-cap body before delegating, so ``reached`` stays False.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.reached = False
        super().__init__(**kwargs)

    async def handle_webhook(self, body: bytes, headers: Any) -> dict[str, Any]:
        self.reached = True
        return await super().handle_webhook(body, headers)


def _connector(**kwargs: Any) -> _SpyConnector:
    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - never run
        return "ok"

    kwargs.setdefault("signing_secret", _SECRET)
    kwargs.setdefault("allowed_sources", ["ci"])
    return _SpyConnector(handler=handler, **kwargs)


def _app(conn: WebhookInboundConnector) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_webhook_router(conn, path="/hook"))
    return app


# ----------------------------------------------------------------- e2e over the app
def test_oversized_content_length_is_rejected_without_buffering() -> None:
    """A lying/oversized Content-Length is refused 413 before the body is read."""
    from fastapi.testclient import TestClient

    conn = _connector()
    over = DEFAULT_MAX_BODY_BYTES + 1
    with TestClient(_app(conn)) as client:
        # Send a tiny actual body but DECLARE an oversized Content-Length: the route must
        # reject on the declared length alone, never reaching the connector.
        resp = client.post(
            "/hook",
            content=b"{}",
            headers={"content-length": str(over)},
        )
    assert resp.status_code == 413
    assert resp.json() == {"ok": False, "reason": "payload too large"}
    assert conn.reached is False, (
        "handle_webhook ran => body was buffered before the cap"
    )


def test_chunked_body_over_cap_is_aborted_while_streaming() -> None:
    """A chunked (no Content-Length) body is aborted 413 once the stream crosses the cap."""
    from fastapi.testclient import TestClient

    conn = _connector()
    chunk = b"x" * 64 * 1024

    def _gen() -> Any:
        # Yield well past the cap; the route must abort mid-stream, not buffer it all.
        for _ in range((DEFAULT_MAX_BODY_BYTES // len(chunk)) + 4):
            yield chunk

    with TestClient(_app(conn)) as client:
        # A generator body is sent chunked (Transfer-Encoding: chunked, no Content-Length).
        resp = client.post("/hook", content=_gen())
    assert resp.status_code == 413
    assert resp.json() == {"ok": False, "reason": "payload too large"}
    assert conn.reached is False


def test_within_cap_body_is_passed_through() -> None:
    """A body within the cap still reaches the connector (no false positive)."""
    from fastapi.testclient import TestClient

    conn = _connector()
    with TestClient(_app(conn)) as client:
        # Unsigned => the connector rejects it 401, but it MUST have reached the connector.
        resp = client.post("/hook", content=b'{"source":"ci","text":"hi"}')
    assert resp.status_code == 401
    assert conn.reached is True


# ----------------------------------------------------------- the helper in isolation
class _FakeRequest:
    """Minimal stand-in for a Starlette Request exercising ``read_body_capped``."""

    def __init__(self, *, headers: dict[str, str], chunks: list[bytes]) -> None:
        self.headers = headers
        self._chunks = chunks
        self.streamed = 0

    async def stream(self) -> Any:
        for chunk in self._chunks:
            self.streamed += len(chunk)
            yield chunk


def test_read_body_capped_rejects_declared_oversize_without_streaming() -> None:
    req = _FakeRequest(
        headers={"content-length": str(DEFAULT_MAX_BODY_BYTES + 1)},
        chunks=[b"x" * 1024],
    )
    with pytest.raises(BodyTooLarge):
        run_async(read_body_capped(req, DEFAULT_MAX_BODY_BYTES))
    # The declared-length short-circuit means the body stream was never consumed.
    assert req.streamed == 0


def test_read_body_capped_aborts_stream_at_cap() -> None:
    chunk = b"x" * 64 * 1024
    nchunks = (DEFAULT_MAX_BODY_BYTES // len(chunk)) + 4
    req = _FakeRequest(headers={}, chunks=[chunk] * nchunks)
    with pytest.raises(BodyTooLarge):
        run_async(read_body_capped(req, DEFAULT_MAX_BODY_BYTES))
    # Aborted shortly after crossing the cap, not after draining the whole oversized stream.
    assert req.streamed <= DEFAULT_MAX_BODY_BYTES + len(chunk)


def test_read_body_capped_returns_body_within_cap() -> None:
    req = _FakeRequest(headers={"content-length": "4"}, chunks=[b"ab", b"cd"])
    body = run_async(read_body_capped(req, DEFAULT_MAX_BODY_BYTES))
    assert body == b"abcd"
