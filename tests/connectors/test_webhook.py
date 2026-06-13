"""Tests for the generic inbound webhook connector — fully OFFLINE (in-process e2e).

Unlike Slack/Discord/GitHub, this connector talks to NO external service on the inbound
path: a sender POSTs a signed JSON payload to an HTTP endpoint we mount, so the accept /
reject behaviour is end-to-end testable on this machine through FastAPI's ``TestClient``.
These tests therefore:

* mount the connector on a real (standalone) FastAPI app and drive it over HTTP — a
  correctly-signed payload is accepted AND triggers the agent run (the handler observes the
  text and returns a reply that comes back in the 200 body); a TAMPERED body and an UNSIGNED
  request are both rejected with 401 and never reach the agent;
* assert the source allowlist is default-deny (an empty allowlist answers no one; an
  unlisted source is 403);
* assert the replay guard (stale signed timestamp) and delivery de-duplication (a repeated
  ``id`` is not re-run), the body-size cap, and the malformed-body handling;
* contract-test the OPTIONAL async result-callback over a mocked httpx transport: the result
  is POSTed to the operator-fixed URL, signed with the shared secret, with redirects off and
  SSRF-guarded (a private/loopback callback URL is refused). The real network leg of the
  callback is the only part that needs a live receiver to verify; it is exercised here
  against a mock transport and marked ``live-pending`` for the wire.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.sdk import AuditSink, ConnectorContext, ConnectorError
from himmy.connectors.webhook import (
    DEFAULT_SIGNATURE_HEADER,
    DEFAULT_TIMESTAMP_HEADER,
    WEBHOOK_SIGNING_SECRET,
    GuardedCallbackPoster,
    WebhookInboundConnector,
    build_webhook_router,
    make_webhook_connector,
    sign_webhook_body,
)
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async

_SECRET = "whsec_4d2f8a1b3c9e7f60aa11bb22cc33dd44"  # a fake signing secret (test only)


class _MemSecrets:
    """In-memory secret provider for tests (installed via configure_secrets)."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _RecordingAudit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _reset_secrets() -> Any:
    """Keep each test's secret provider isolated."""
    yield
    configure_secrets(None)


# --------------------------------------------------------------------------- helpers
def _sign(secret: str, body: bytes, *, timestamp: str | None = None) -> str:
    """Independent re-derivation of the wire signature (not via the module under test)."""
    material = body if timestamp is None else (f"{timestamp}.".encode() + body)
    digest = hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _payload(**fields: Any) -> bytes:
    base = {"source": "ci", "text": "build 1234 passed", "id": "evt-1"}
    base.update(fields)
    return json.dumps(base).encode("utf-8")


def _connector(**kwargs: Any) -> tuple[WebhookInboundConnector, list[str]]:
    """A connector whose handler records the prompts it saw and echoes a reply."""
    seen: list[str] = []

    async def handler(sender_id: str, text: str) -> str:
        seen.append(f"{sender_id}:{text}")
        return f"ran agent for: {text}"

    kwargs.setdefault("signing_secret", _SECRET)
    kwargs.setdefault("allowed_sources", ["ci"])
    conn = WebhookInboundConnector(handler, **kwargs)
    return conn, seen


# ----------------------------------------------------------------- e2e over the app
def _app(conn: WebhookInboundConnector) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_webhook_router(conn, path="/hook"))
    return app


def test_signed_payload_is_accepted_and_triggers_the_agent() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _connector()
    body = _payload()
    sig = _sign(_SECRET, body)
    with TestClient(_app(conn)) as client:
        resp = client.post(
            "/hook", content=body, headers={DEFAULT_SIGNATURE_HEADER: sig}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["handled"] is True
    assert data["reply"] == "ran agent for: build 1234 passed"
    # The agent actually ran, with the source as sender id and the payload text.
    assert seen == ["ci:build 1234 passed"]


def test_tampered_body_is_rejected_and_agent_never_runs() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _connector()
    body = _payload()
    sig = _sign(_SECRET, body)  # signature over the ORIGINAL body
    tampered = _payload(text="rm -rf / now please")  # different bytes, same signature
    with TestClient(_app(conn)) as client:
        resp = client.post(
            "/hook", content=tampered, headers={DEFAULT_SIGNATURE_HEADER: sig}
        )
    assert resp.status_code == 401
    assert resp.json()["ok"] is False
    assert seen == []  # the forged payload never reached the agent


def test_unsigned_request_is_rejected() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _connector()
    with TestClient(_app(conn)) as client:
        resp = client.post("/hook", content=_payload())  # no signature header at all
    assert resp.status_code == 401
    assert resp.json()["reason"] == "invalid signature"
    assert seen == []


def test_wrong_secret_signature_is_rejected() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _connector()
    body = _payload()
    sig = _sign("not-the-real-secret", body)
    with TestClient(_app(conn)) as client:
        resp = client.post(
            "/hook", content=body, headers={DEFAULT_SIGNATURE_HEADER: sig}
        )
    assert resp.status_code == 401
    assert seen == []


def test_unlisted_source_is_forbidden_even_when_signed() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _connector(allowed_sources=["ci"])
    body = _payload(source="attacker")
    sig = _sign(_SECRET, body)  # correctly signed, but the source is not allow-listed
    with TestClient(_app(conn)) as client:
        resp = client.post(
            "/hook", content=body, headers={DEFAULT_SIGNATURE_HEADER: sig}
        )
    assert resp.status_code == 403
    assert resp.json()["reason"] == "source not allowed"
    assert seen == []


# ----------------------------------------------------------------- unit / direct gate
def test_empty_allowlist_denies_everyone_by_default() -> None:
    conn, seen = _connector(allowed_sources=[])
    body = _payload()
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result == {"ok": False, "reason": "source not allowed"}
    assert seen == []


def test_signature_header_is_case_insensitive() -> None:
    conn, seen = _connector()
    body = _payload()
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {"x-himmy-signature": sig}))
    assert result["ok"] is True
    assert seen == ["ci:build 1234 passed"]


def test_malformed_json_is_acknowledged_but_not_run() -> None:
    conn, seen = _connector()
    body = b"{not json"
    sig = _sign(_SECRET, body)  # validly signed garbage
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result == {"ok": True, "handled": False}
    assert seen == []


def test_missing_text_is_acknowledged_but_not_run() -> None:
    conn, seen = _connector()
    body = json.dumps({"source": "ci", "id": "x"}).encode("utf-8")
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result == {"ok": True, "handled": False}
    assert seen == []


def test_oversize_body_is_rejected_before_signature_check() -> None:
    conn, seen = _connector(max_body_bytes=64)
    body = _payload(text="x" * 200)
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result == {"ok": False, "reason": "payload too large"}
    assert seen == []


# ------------------------------------------------------------------ replay + timestamp
def test_timestamp_binding_accepts_a_fresh_signed_request() -> None:
    fixed = 1_000_000.0
    conn, seen = _connector(clock=lambda: fixed)
    body = _payload()
    ts = str(int(fixed))
    sig = _sign(_SECRET, body, timestamp=ts)
    result = run_async(
        conn.handle_webhook(
            body, {DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: ts}
        )
    )
    assert result["ok"] is True
    assert seen == ["ci:build 1234 passed"]


def test_stale_timestamp_is_rejected_as_replay() -> None:
    now = 1_000_000.0
    conn, seen = _connector(clock=lambda: now, max_timestamp_skew=300)
    body = _payload()
    old_ts = str(int(now - 4000))  # well outside the 300s window
    sig = _sign(_SECRET, body, timestamp=old_ts)
    result = run_async(
        conn.handle_webhook(
            body, {DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: old_ts}
        )
    )
    assert result == {"ok": False, "reason": "invalid signature"}
    assert seen == []


def test_timestamp_cannot_be_swapped_without_breaking_the_signature() -> None:
    now = 1_000_000.0
    conn, seen = _connector(clock=lambda: now)
    body = _payload()
    ts = str(int(now))
    sig = _sign(_SECRET, body, timestamp=ts)
    # Attacker forwards the same body+sig but a different (fresh) timestamp header.
    result = run_async(
        conn.handle_webhook(
            body,
            {DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: str(int(now) + 1)},
        )
    )
    assert result["ok"] is False
    assert seen == []


# ------------------------------------------------------------------ de-duplication
def test_duplicate_delivery_id_is_not_run_twice() -> None:
    conn, seen = _connector()
    body = _payload(id="evt-dup")
    sig = _sign(_SECRET, body)
    first = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    second = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert first["handled"] is True
    assert second == {"ok": True, "handled": False, "deduplicated": True}
    assert seen == ["ci:build 1234 passed"]  # ran exactly once


# ------------------------------------------------------------------ capability + secrets
def test_capability_is_unavailable_without_a_secret() -> None:
    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - not run
        return ""

    configure_secrets(_MemSecrets({}))  # no webhook secret present
    conn = WebhookInboundConnector(handler, allowed_sources=["ci"])
    cap = conn.capability()
    assert cap.ok is False
    assert WEBHOOK_SIGNING_SECRET in cap.reason
    assert conn.available() is False


def test_secret_is_sourced_from_the_secrets_layer() -> None:
    seen: list[str] = []

    async def handler(sender_id: str, text: str) -> str:
        seen.append(text)
        return "ok"

    configure_secrets(_MemSecrets({WEBHOOK_SIGNING_SECRET: _SECRET}))
    conn = WebhookInboundConnector(handler, allowed_sources=["ci"])
    assert conn.available() is True
    body = _payload()
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result["ok"] is True
    assert seen == ["build 1234 passed"]


def test_make_webhook_connector_reads_the_secret_and_wires_a_guarded_poster() -> None:
    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover
        return ""

    configure_secrets(_MemSecrets({WEBHOOK_SIGNING_SECRET: _SECRET}))
    conn = make_webhook_connector(
        handler,
        allowed_sources=["ci"],
        callback_url="https://hooks.example.com/result",
        egress_allow_hosts=["hooks.example.com"],
    )
    assert conn.available() is True
    assert conn._callback_url == "https://hooks.example.com/result"  # noqa: SLF001
    assert isinstance(conn._result_callback, GuardedCallbackPoster)  # noqa: SLF001


def test_callback_and_url_must_be_set_together() -> None:
    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover
        return ""

    with pytest.raises(ConnectorError):  # misconfiguration: url without a callback
        WebhookInboundConnector(
            handler,
            signing_secret=_SECRET,
            allowed_sources=["ci"],
            callback_url="https://x.example.com/cb",  # callback_url without result_callback
        )


# ------------------------------------------------------------------ audit trail
def test_denied_signature_is_audited() -> None:
    audit = _RecordingAudit()
    conn, _seen = _connector(context=ConnectorContext(audit=AuditSink(audit.record)))
    body = _payload()
    run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: "sha256=bad"}))
    outcomes = [(e.action, e.outcome) for e in audit.events]
    assert ("webhook", "deny") in outcomes


def test_allowed_run_is_audited() -> None:
    audit = _RecordingAudit()
    conn, _seen = _connector(context=ConnectorContext(audit=AuditSink(audit.record)))
    body = _payload()
    sig = _sign(_SECRET, body)
    run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    outcomes = [(e.action, e.outcome) for e in audit.events]
    assert ("inbound_message", "allow") in outcomes


# ----------------------------------------------- async ack + result callback (mocked)
class _MockTransport:
    """Captures the single outbound callback POST without opening a socket."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def handle_request(self, request: Any) -> Any:
        import httpx

        self.requests.append(request)
        return httpx.Response(200, json={"ok": True})


def test_result_callback_posts_a_signed_result() -> None:
    seen: list[str] = []

    async def handler(sender_id: str, text: str) -> str:
        seen.append(text)
        return "the agent's answer"

    import httpx

    transport = _MockTransport()
    poster = GuardedCallbackPoster(
        secret=_SECRET,
        egress_allow_hosts=["hooks.example.com"],
        transport=httpx.MockTransport(transport.handle_request),
    )
    conn = WebhookInboundConnector(
        handler,
        signing_secret=_SECRET,
        allowed_sources=["ci"],
        result_callback=poster,
        callback_url="https://hooks.example.com/result",
        ack_only=True,
        clock=lambda: 1_000_000.0,
    )
    body = _payload()
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))

    # Async-ack mode: the response is a bare ack, the reply went to the callback.
    assert result == {"ok": True, "handled": True, "accepted": True}
    assert seen == ["build 1234 passed"]

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert str(req.url) == "https://hooks.example.com/result"
    posted = json.loads(req.content)
    assert posted["reply"] == "the agent's answer"
    assert posted["source"] == "ci"
    # The callback body is signed with the same shared secret so the receiver can verify us.
    expected = sign_webhook_body(
        secret=_SECRET, body=req.content, timestamp="1000000"
    )
    assert req.headers[DEFAULT_SIGNATURE_HEADER] == expected


def test_async_ack_router_returns_202() -> None:
    from fastapi.testclient import TestClient

    async def handler(sender_id: str, text: str) -> str:
        return "done"

    import httpx

    transport = _MockTransport()
    poster = GuardedCallbackPoster(
        secret=_SECRET,
        egress_allow_hosts=["hooks.example.com"],
        transport=httpx.MockTransport(transport.handle_request),
    )
    conn = WebhookInboundConnector(
        handler,
        signing_secret=_SECRET,
        allowed_sources=["ci"],
        result_callback=poster,
        callback_url="https://hooks.example.com/result",
        ack_only=True,
    )
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_webhook_router(conn, path="/hook"))
    body = _payload()
    sig = _sign(_SECRET, body)
    with TestClient(app) as client:
        resp = client.post(
            "/hook", content=body, headers={DEFAULT_SIGNATURE_HEADER: sig}
        )
    assert resp.status_code == 202
    assert resp.json()["accepted"] is True
    assert len(transport.requests) == 1


def test_callback_to_a_private_url_is_refused_by_the_ssrf_guard() -> None:
    poster = GuardedCallbackPoster(secret=_SECRET)
    with pytest.raises(ToolSecurityError):
        run_async(
            poster("http://169.254.169.254/latest/meta-data/", {"reply": "x"}, "sha256=z")
        )


def test_callback_failure_does_not_crash_intake() -> None:
    """A flaky receiver is audited (secret-safe) but never raised to the sender."""

    async def handler(sender_id: str, text: str) -> str:
        return "answer"

    async def failing_callback(url: str, payload: Any, sig: str) -> None:
        raise RuntimeError(f"boom with secret {_SECRET}")  # would leak if surfaced

    audit = _RecordingAudit()
    conn = WebhookInboundConnector(
        handler,
        signing_secret=_SECRET,
        allowed_sources=["ci"],
        result_callback=failing_callback,
        callback_url="https://hooks.example.com/result",
        context=ConnectorContext(audit=AuditSink(audit.record)),
    )
    body = _payload()
    sig = _sign(_SECRET, body)
    result = run_async(conn.handle_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}))
    assert result["ok"] is True  # intake survived
    # The callback failure was audited and the secret was scrubbed from the detail.
    callback_denies = [
        e for e in audit.events if e.action == "webhook_callback" and e.outcome == "deny"
    ]
    assert callback_denies
    assert all(_SECRET not in (e.detail or "") for e in callback_denies)


def test_sign_webhook_body_matches_the_verifier() -> None:
    conn, _seen = _connector()
    body = _payload()
    sig = sign_webhook_body(secret=_SECRET, body=body)
    assert conn.verify_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}) is True
    # With a timestamp the helper binds it; the verifier must agree.
    ts = str(int(time.time()))
    sig_ts = sign_webhook_body(secret=_SECRET, body=body, timestamp=ts)
    assert (
        conn.verify_webhook(
            body, {DEFAULT_SIGNATURE_HEADER: sig_ts, DEFAULT_TIMESTAMP_HEADER: ts}
        )
        is True
    )
