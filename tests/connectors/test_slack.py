"""Contract tests for the Slack connector — fully offline (mocked transport).

Slack is an EXTERNAL service: end-to-end verification against the real API needs a live
Slack app + bot/signing/app-level tokens and a workspace, and cannot run on this machine.
These tests therefore CONTRACT-test the connector against a mocked ``httpx`` transport and
a synthetic signing secret: they assert the exact API calls, payloads, and auth header it
would make, that the auth header is present-on-the-wire but redacted-in-logs/errors, that
the v0 request signature is verified (valid accepted, tampered/stale/missing rejected),
that the team/channel/user allowlist drops unauthorized senders (default-deny), that an
event round-trips through the signed webhook and is answered IN THREAD, and that the bot
never answers its own message (loop guard). What still needs a real Slack app + workspace
to verify is marked ``live-pending`` in the module docstrings.
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
from himmy.connectors.slack import (
    SLACK_BOT_TOKEN_SECRET,
    SLACK_SIGNING_SECRET,
    SlackClient,
    SlackEventsConnector,
    SlackOutboundConnector,
    SlackSocketModeListener,
    verify_slack_signature,
)
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async


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


# ------------------------------------------------------------- v0 signing helpers

_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"  # a fake signing secret (test only)


def _sign(secret: str, timestamp: str, body: bytes) -> str:
    """Compute the X-Slack-Signature value Slack would send for this request."""
    basestring = f"v0:{timestamp}:".encode() + body
    digest = hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def test_verify_slack_signature_accepts_correct_and_denies_tampered_or_missing() -> None:
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts = str(int(time.time()))
    good = _sign(_SECRET, ts, body)
    assert verify_slack_signature(
        signing_secret=_SECRET, timestamp=ts, body=body, signature=good
    )
    # Tampered body, wrong sig, missing sig, bad/empty timestamp, wrong secret → all denied.
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp=ts, body=b"tampered", signature=good
    )
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp=ts, body=body, signature="v0=" + "00" * 32
    )
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp=ts, body=body, signature=None
    )
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp="", body=body, signature=good
    )
    assert not verify_slack_signature(
        signing_secret="wrong-secret", timestamp=ts, body=body, signature=good
    )


def test_verify_slack_signature_rejects_stale_timestamp_replay() -> None:
    """A signature over a timestamp older than the skew window is denied (replay guard)."""
    body = b'{"x":1}'
    old_ts = str(int(time.time()) - 60 * 60)  # an hour old
    good_over_old = _sign(_SECRET, old_ts, body)
    # Even though the HMAC over this (old) basestring is correct, the stale timestamp denies.
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp=old_ts, body=body, signature=good_over_old
    )
    # A non-integer timestamp is a clean deny, not a crash.
    assert not verify_slack_signature(
        signing_secret=_SECRET, timestamp="not-an-int", body=body, signature=good_over_old
    )


# --------------------------------------------------- inbound: signed events api


def _event_body(
    *,
    team: str = "T1",
    channel: str = "C1",
    user: str = "U1",
    text: str = "hello himmy",
    etype: str = "app_mention",
    ts: str = "1700000000.000100",
    thread_ts: str | None = None,
    bot_id: str | None = None,
    subtype: str | None = None,
) -> bytes:
    """A realistic event_callback envelope body."""
    event: dict[str, Any] = {
        "type": etype,
        "channel": channel,
        "user": user,
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if bot_id is not None:
        event["bot_id"] = bot_id
    if subtype is not None:
        event["subtype"] = subtype
    return json.dumps(
        {"type": "event_callback", "team_id": team, "event": event}
    ).encode()


def _fixed_clock() -> float:
    """A clock pinned to the event timestamps used in these tests (avoids skew flakiness)."""
    return 1700000000.0


def _events_connector(
    handler: Any,
    audit: _RecordingAudit | None = None,
    poster: Any = None,
    **kw: Any,
) -> SlackEventsConnector:
    ctx = ConnectorContext(audit=AuditSink(audit.record)) if audit else None
    return SlackEventsConnector(
        handler,
        signing_secret=_SECRET,
        poster=poster,
        context=ctx,
        clock=_fixed_clock,
        **kw,
    )


def _signed_headers(body: bytes, ts: str = "1700000000") -> dict[str, str]:
    return {
        "X-Slack-Signature": _sign(_SECRET, ts, body),
        "X-Slack-Request-Timestamp": ts,
    }


def test_events_url_verification_echoes_challenge_without_agent() -> None:
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "should not run"

    conn = _events_connector(handler)
    body = json.dumps({"type": "url_verification", "challenge": "tok-123"}).encode()
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is True
    assert res["challenge"] == "tok-123"
    assert ran == []  # the handshake never reaches the agent


def test_events_bad_signature_denied_before_parse_or_agent() -> None:
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    audit = _RecordingAudit()
    conn = _events_connector(handler, audit=audit, allowed_team_ids=["T1"])
    body = _event_body()
    res = run_async(
        conn.handle_webhook(
            body,
            {
                "X-Slack-Signature": "v0=" + "ff" * 32,
                "X-Slack-Request-Timestamp": "1700000000",
            },
        )
    )
    assert res["ok"] is False
    assert ran == []  # agent never ran on an unverified body
    assert audit.events[-1].outcome == "deny"


def test_events_stale_timestamp_denied() -> None:
    """A correctly-signed but old request is denied (replay guard), agent never runs."""
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    conn = _events_connector(handler, allowed_team_ids=["T1"])
    body = _event_body()
    old_ts = "1699990000"  # ~2.7h before the fixed clock
    res = run_async(conn.handle_webhook(body, _signed_headers(body, ts=old_ts)))
    assert res["ok"] is False
    assert ran == []


def test_events_allowlist_admits_listed_team_and_runs_agent_and_replies_in_thread() -> None:
    seen: list[tuple[str, str]] = []
    posted: list[tuple[str, str, str | None]] = []

    async def handler(sender: str, text: str) -> str:
        seen.append((sender, text))
        return "answer: 42"

    def poster(channel: str, text: str, thread_ts: str | None) -> Any:
        posted.append((channel, text, thread_ts))
        return {"ok": True, "ts": "1700000000.000200"}

    conn = _events_connector(handler, poster=poster, allowed_team_ids=["T1"])
    body = _event_body(text="what is the meaning of life")
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is True
    assert res["handled"] is True
    # The agent saw the event; sender carries team/channel/user.
    assert seen == [("team:T1:channel:C1:user:U1", "what is the meaning of life")]
    # The reply was posted back to the source channel, threaded under the message ts.
    assert posted == [("C1", "answer: 42", "1700000000.000100")]


def test_events_reply_uses_existing_thread_ts_when_present() -> None:
    async def handler(sender: str, text: str) -> str:
        return "ok"

    posted: list[tuple[str, str, str | None]] = []

    def poster(channel: str, text: str, thread_ts: str | None) -> Any:
        posted.append((channel, text, thread_ts))
        return {"ok": True}

    conn = _events_connector(handler, poster=poster, allowed_channel_ids=["C1"])
    # An event already inside a thread → reply uses that thread_ts, not the message ts.
    body = _event_body(thread_ts="1699999999.000001", ts="1700000000.000100")
    run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert posted == [("C1", "ok", "1699999999.000001")]


def test_events_allowlist_admits_by_channel_or_user_too() -> None:
    async def handler(sender: str, text: str) -> str:
        return "ok"

    # Channel-only allowlist admits that channel's event even with an unlisted team/user.
    conn = _events_connector(handler, allowed_channel_ids=["C1"])
    body = _event_body(team="Tx", channel="C1", user="Ux")
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is True and res["handled"] is True

    # And a user-only allowlist admits that user from any team/channel.
    conn2 = _events_connector(handler, allowed_user_ids=["U9"])
    body2 = _event_body(team="Tz", channel="Cz", user="U9")
    res2 = run_async(conn2.handle_webhook(body2, _signed_headers(body2)))
    assert res2["ok"] is True and res2["handled"] is True


def test_events_default_deny_drops_unlisted_sender() -> None:
    """An empty allowlist denies everyone even with a valid signature (default-deny)."""
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "leak"

    audit = _RecordingAudit()
    conn = _events_connector(handler, audit=audit)  # no allowlist
    body = _event_body()
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is False
    assert ran == []  # signature valid, but the sender is not allow-listed
    assert audit.events[-1].outcome == "deny"


def test_events_unlisted_sender_dropped_even_if_signed() -> None:
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    conn = _events_connector(handler, allowed_team_ids=["GOOD"])
    body = _event_body(team="EVIL")
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is False
    assert ran == []


def test_events_ignores_bot_messages_loop_guard() -> None:
    """A message authored by a bot (incl. ourselves) is dropped — the bot won't self-loop."""
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "echo"

    conn = _events_connector(handler, allowed_channel_ids=["C1"])
    body = _event_body(etype="message", bot_id="B999")
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is True and res["handled"] is False
    assert ran == []  # never act on a bot's own post


def test_events_ignores_message_subtypes() -> None:
    """A message_changed/deleted subtype carries no real user turn and is ignored."""
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    conn = _events_connector(handler, allowed_channel_ids=["C1"])
    body = _event_body(etype="message", subtype="message_changed")
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    assert res["ok"] is True and res["handled"] is False
    assert ran == []


def test_events_capability_requires_signing_secret() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    # No signing secret configured anywhere → unavailable, skipped cleanly.
    configure_secrets(_MemSecrets({}))
    try:
        conn = SlackEventsConnector(handler)  # reads secret (absent)
        cap = conn.capability()
        assert cap.ok is False
        assert SLACK_SIGNING_SECRET in cap.reason
    finally:
        configure_secrets(None)


def test_events_signing_secret_read_from_secrets_layer() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    configure_secrets(_MemSecrets({SLACK_SIGNING_SECRET: _SECRET}))
    try:
        conn = SlackEventsConnector(handler, allowed_team_ids=["T1"])
        assert conn.capability().ok is True
    finally:
        configure_secrets(None)


def test_events_is_webhook_only_not_polling() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    conn = _events_connector(handler)
    with pytest.raises(ConnectorError, match="webhook-only"):
        run_async(conn.fetch_updates(None))


def test_events_reply_failure_does_not_crash_intake() -> None:
    """A failing poster is audited but never propagates out of the webhook handler."""

    async def handler(sender: str, text: str) -> str:
        return "answer"

    def poster(channel: str, text: str, thread_ts: str | None) -> Any:
        raise RuntimeError("slack down")

    audit = _RecordingAudit()
    conn = _events_connector(
        handler, audit=audit, poster=poster, allowed_channel_ids=["C1"]
    )
    body = _event_body()
    res = run_async(conn.handle_webhook(body, _signed_headers(body)))
    # The agent ran + was authorized; only the reply send failed (audited deny).
    assert res["ok"] is True
    assert any(
        e.outcome == "deny" and e.action == "inbound_reply" for e in audit.events
    )


# ------------------------------------------------------ outbound: post-message


def _mock_transport(app: Any) -> Any:
    import httpx

    return httpx.MockTransport(app)


def test_post_message_sends_bearer_auth_and_body_via_mock_transport() -> None:
    """Contract: assert the exact request slack_post_message puts on the wire."""
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True, "channel": "C1", "ts": "1.2"})

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb-secret-token"}))
    try:
        # Inject a SlackClient bound to the mock transport (no real socket / DNS).
        def factory(token: str) -> SlackClient:
            return SlackClient(token, transport=_mock_transport(app))

        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=factory,
        )
        registry = ToolRegistry()
        names = conn.register_tools(registry)
        assert names == ["slack_post_message"]
        out = run_async(
            registry.handler_for("slack_post_message")(
                {"text": "hi from himmy", "channel": "C1"}
            )
        )
        assert out["sent"] is True
        assert out["ts"] == "1.2"
        assert out["channel"] == "C1"
        assert seen["method"] == "POST"
        assert seen["url"] == "https://slack.com/api/chat.postMessage"
        # Slack Web API auth uses the standard Bearer scheme, carrying the secret token.
        assert seen["auth"] == "Bearer xoxb-secret-token"
        assert seen["body"]["channel"] == "C1"
        assert seen["body"]["text"] == "hi from himmy"
    finally:
        configure_secrets(None)


def test_post_message_threads_when_thread_ts_supplied() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True, "ts": "9.9"})

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        run_async(
            registry.handler_for("slack_post_message")(
                {"text": "reply", "channel": "C1", "thread_ts": "1700.0001"}
            )
        )
        assert seen["body"]["thread_ts"] == "1700.0001"
    finally:
        configure_secrets(None)


def test_post_message_denies_unlisted_channel_default_deny() -> None:
    """A channel NOT on the allowlist is refused before any HTTP call (default-deny)."""
    calls = {"n": 0}

    def factory(token: str) -> SlackClient:
        class _NoNet(SlackClient):
            def post_message(self_inner, *a: Any, **k: Any) -> dict[str, Any]:
                calls["n"] += 1
                return {"ok": True}

        return _NoNet(token)

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        audit = _RecordingAudit()
        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=factory,
            context=ConnectorContext(audit=AuditSink(audit.record)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        with pytest.raises(ToolSecurityError, match="not on the Slack post allowlist"):
            run_async(
                registry.handler_for("slack_post_message")(
                    {"text": "x", "channel": "C-EVIL"}
                )
            )
        assert calls["n"] == 0  # never hit the network for a denied channel
        assert audit.events[-1].outcome == "deny"
    finally:
        configure_secrets(None)


def test_post_message_uses_default_channel_when_omitted() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True, "ts": "1"})

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(
            default_channel_id="C-DEFAULT",
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        out = run_async(registry.handler_for("slack_post_message")({"text": "hi"}))
        assert out["channel"] == "C-DEFAULT"
        assert seen["body"]["channel"] == "C-DEFAULT"
    finally:
        configure_secrets(None)


def test_post_message_idempotent_dedupes_repeat() -> None:
    """The same (channel, thread, text) posts exactly once across a repeated tool call."""
    import httpx

    hits = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json={"ok": True, "ts": str(hits["n"])})

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        args = {"text": "same message", "channel": "C1"}
        first = run_async(registry.handler_for("slack_post_message")(dict(args)))
        second = run_async(registry.handler_for("slack_post_message")(dict(args)))
        assert first == second
        assert hits["n"] == 1  # the effect fired exactly once
    finally:
        configure_secrets(None)


def test_post_message_capability_gated_on_token() -> None:
    """Without the bot token the connector is unavailable and registers no tools."""
    configure_secrets(_MemSecrets({}))
    try:
        conn = SlackOutboundConnector(allowed_channel_ids=["C1"])
        assert conn.available() is False
        registry = ToolRegistry()
        assert conn.register_tools(registry) == []  # skipped cleanly, no crash
    finally:
        configure_secrets(None)


def test_post_message_token_read_from_secrets_layer_not_env(monkeypatch: Any) -> None:
    """The token comes from the secrets layer, never straight from os.environ."""
    import httpx

    monkeypatch.setenv(SLACK_BOT_TOKEN_SECRET, "from-env-MUST-NOT-USE")
    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True, "ts": "1"})

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "from-secrets-layer"}))
    try:
        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        run_async(
            registry.handler_for("slack_post_message")({"text": "x", "channel": "C1"})
        )
        assert seen["auth"] == "Bearer from-secrets-layer"
    finally:
        configure_secrets(None)


def test_post_message_error_never_leaks_token_in_message() -> None:
    """A Slack API failure surfaces its error code, not the body — and never the token."""
    import httpx

    token = "xoxb-SUPER-SECRET"

    def app(request: httpx.Request) -> httpx.Response:
        # Slack returns 200 with ok=false; an error body that (maliciously) echoes the
        # token must never ride out in the surfaced error.
        return httpx.Response(
            200, json={"ok": False, "error": f"not_in_channel auth Bearer {token}"}
        )

    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: token}))
    try:
        conn = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        with pytest.raises(ConnectorError) as exc:
            run_async(
                registry.handler_for("slack_post_message")(
                    {"text": "x", "channel": "C1"}
                )
            )
        assert token not in str(exc.value)  # the token never rides out in an error
    finally:
        configure_secrets(None)


def test_post_message_is_approval_gated_by_default() -> None:
    """The side-effecting post tool defaults to human-in-the-loop approval."""
    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(allowed_channel_ids=["C1"])  # default approval
        registry = ToolRegistry()
        conn.register_tools(registry)
        assert registry.get("slack_post_message").requires_approval is True
    finally:
        configure_secrets(None)


def test_outbound_default_fetcher_carries_ssrf_guard() -> None:
    """The connector's default fetcher refuses an SSRF target (the guard is wired)."""
    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(allowed_channel_ids=["C1"])
        with pytest.raises(ToolSecurityError):
            conn.guard("http://169.254.169.254/latest/meta-data/")
        # And the egress allow-list pins it to the Slack host.
        with pytest.raises(ToolSecurityError, match="egress allow-list"):
            conn.guard("https://evil.example.test/webhook")
    finally:
        configure_secrets(None)


def test_outbound_post_helper_reapplies_channel_allowlist() -> None:
    """The shared post() helper (used for inbound replies) also denies an unlisted channel."""
    configure_secrets(_MemSecrets({SLACK_BOT_TOKEN_SECRET: "xoxb"}))
    try:
        conn = SlackOutboundConnector(allowed_channel_ids=["C1"])
        with pytest.raises(ConnectorError, match="not on the Slack post allowlist"):
            conn.post("C-EVIL", "hi", thread_ts=None)
    finally:
        configure_secrets(None)


# --------------------------------------------- inbound + outbound wired together


def test_events_reply_goes_through_outbound_allowlist() -> None:
    """End-to-end: a signed event's reply funnels through the outbound post() allowlist.

    The reply is posted to the SAME (allow-listed) channel and threaded; a reply can never
    escape to a non-allow-listed channel because the outbound poster re-applies the gate.
    """
    import httpx

    sent: list[dict[str, Any]] = []

    def app(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read().decode()))
        return httpx.Response(200, json={"ok": True, "channel": "C1", "ts": "1.3"})

    configure_secrets(
        _MemSecrets(
            {SLACK_BOT_TOKEN_SECRET: "xoxb", SLACK_SIGNING_SECRET: _SECRET}
        )
    )
    try:
        outbound = SlackOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: SlackClient(t, transport=_mock_transport(app)),
        )

        async def handler(sender: str, text: str) -> str:
            return f"you said: {text}"

        conn = SlackEventsConnector(
            handler,
            signing_secret=_SECRET,
            poster=outbound.post,
            allowed_channel_ids=["C1"],
            clock=_fixed_clock,
        )
        body = _event_body(text="ping")
        res = run_async(conn.handle_webhook(body, _signed_headers(body)))
        assert res["ok"] is True and res["handled"] is True
        assert len(sent) == 1
        assert sent[0]["channel"] == "C1"
        assert sent[0]["text"] == "you said: ping"
        assert sent[0]["thread_ts"] == "1700000000.000100"
    finally:
        configure_secrets(None)


# ---------------------------------------------------- optional socket mode listener


def test_socket_mode_listener_default_deny_allowlist() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    # No allowlist at all → deny everyone (a bot can see every message it has access to).
    listener = SlackSocketModeListener(handler)
    assert listener.is_allowed(team_id="T", channel_id="C", user_id="U") is False

    listener2 = SlackSocketModeListener(handler, allowed_channel_ids=["C1"])
    assert listener2.is_allowed(team_id="Tx", channel_id="C1", user_id="Ux") is True
    assert listener2.is_allowed(team_id="Tx", channel_id="Cx", user_id="Ux") is False


def test_socket_mode_listener_capability_gated_on_lib_and_tokens() -> None:
    """The socket path is unavailable without BOTH the `slack_sdk` extra and the tokens.

    It is skipped cleanly (capability not OK) whenever any is missing — the offline test
    asserts that invariant rather than which one happens to be absent here. The live socket
    run itself is live-pending (needs a real Slack app + app-level token).
    """
    configure_secrets(_MemSecrets({}))  # no tokens
    try:
        cap = SlackSocketModeListener.capability()
        assert cap.ok is False  # tokens absent ⇒ unavailable regardless of the lib
    finally:
        configure_secrets(None)
