"""Contract tests for the Discord connector — fully offline (mocked transport).

Discord is an EXTERNAL service: end-to-end verification against the real API needs a
live bot token + a registered application/guild and cannot run on this machine. These
tests therefore CONTRACT-test the connector against a mocked ``httpx`` transport and a
synthetic Ed25519 keypair: they assert the exact API calls, payloads, and auth header it
would make, that the auth header is present-on-the-wire but redacted-in-logs/errors, that
the guild/channel/user allowlist drops unauthorized senders (default-deny), and that an
interaction round-trips through the signed webhook. What still needs a real bot token +
server to verify is marked ``live-pending`` in the module docstrings.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.discord import (
    DISCORD_BOT_TOKEN_SECRET,
    DISCORD_PUBLIC_KEY_SECRET,
    DiscordClient,
    DiscordGatewayListener,
    DiscordInteractionsConnector,
    DiscordOutboundConnector,
    verify_ed25519_signature,
)
from himmy.connectors.sdk import AuditSink, ConnectorContext, ConnectorError
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


# ------------------------------------------------------------- Ed25519 signing


def _keypair() -> tuple[Any, str]:
    """A synthetic Ed25519 keypair; returns (private_key, public_key_hex)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_hex = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    return priv, pub_hex


def _sign(priv: Any, timestamp: str, body: bytes) -> str:
    return priv.sign(timestamp.encode("utf-8") + body).hex()


def test_verify_ed25519_accepts_correct_and_denies_tampered_or_missing() -> None:
    priv, pub_hex = _keypair()
    ts, body = "1700000000", b'{"type":1}'
    good = _sign(priv, ts, body)
    assert verify_ed25519_signature(
        public_key_hex=pub_hex, timestamp=ts, body=body, signature_hex=good
    )
    # Tampered body, wrong sig, missing sig, bad timestamp → all denied, never crash.
    assert not verify_ed25519_signature(
        public_key_hex=pub_hex, timestamp=ts, body=b"tampered", signature_hex=good
    )
    assert not verify_ed25519_signature(
        public_key_hex=pub_hex, timestamp=ts, body=body, signature_hex="00" * 64
    )
    assert not verify_ed25519_signature(
        public_key_hex=pub_hex, timestamp=ts, body=body, signature_hex=None
    )
    assert not verify_ed25519_signature(
        public_key_hex=pub_hex, timestamp="", body=body, signature_hex=good
    )
    # A malformed (non-hex) public key is a clean deny, not a crash.
    assert not verify_ed25519_signature(
        public_key_hex="not-hex", timestamp=ts, body=body, signature_hex=good
    )


# --------------------------------------------------- inbound: signed interactions


def _command_body(
    *,
    guild: str = "G1",
    channel: str = "C1",
    user: str = "U1",
    name: str = "ask",
    question: str = "hello",
) -> bytes:
    """A realistic slash-command interaction (type=2) body."""
    return json.dumps(
        {
            "type": 2,
            "guild_id": guild,
            "channel_id": channel,
            "member": {"user": {"id": user, "username": "alice"}},
            "data": {
                "name": name,
                "options": [{"name": "question", "value": question}],
            },
        }
    ).encode()


def _interactions_connector(
    handler: Any,
    priv: Any,
    pub_hex: str,
    audit: _RecordingAudit | None = None,
    **kw: Any,
) -> DiscordInteractionsConnector:
    ctx = ConnectorContext(audit=AuditSink(audit.record)) if audit else None
    return DiscordInteractionsConnector(handler, public_key=pub_hex, context=ctx, **kw)


def _signed_headers(priv: Any, body: bytes, ts: str = "1700000000") -> dict[str, str]:
    return {
        "X-Signature-Ed25519": _sign(priv, ts, body),
        "X-Signature-Timestamp": ts,
    }


def test_interactions_ping_is_answered_with_pong_without_agent() -> None:
    priv, pub_hex = _keypair()
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "should not run"

    conn = _interactions_connector(handler, priv, pub_hex)
    body = b'{"type":1}'
    res = run_async(conn.handle_webhook(body, _signed_headers(priv, body)))
    assert res["ok"] is True
    assert res["response"] == {"type": 1}  # PONG
    assert ran == []  # a PING never reaches the agent


def test_interactions_bad_signature_denied_before_parse_or_agent() -> None:
    priv, pub_hex = _keypair()
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    audit = _RecordingAudit()
    conn = _interactions_connector(
        handler, priv, pub_hex, audit=audit, allowed_guild_ids=["G1"]
    )
    body = _command_body()
    res = run_async(
        conn.handle_webhook(
            body,
            {"X-Signature-Ed25519": "ff" * 64, "X-Signature-Timestamp": "1700000000"},
        )
    )
    assert res["ok"] is False
    assert ran == []  # agent never ran on an unverified body
    assert audit.events[-1].outcome == "deny"


def test_interactions_allowlist_admits_listed_guild_and_runs_agent() -> None:
    priv, pub_hex = _keypair()
    seen: list[tuple[str, str]] = []

    async def handler(sender: str, text: str) -> str:
        seen.append((sender, text))
        return "answer: 42"

    conn = _interactions_connector(handler, priv, pub_hex, allowed_guild_ids=["G1"])
    body = _command_body(question="meaning of life")
    res = run_async(conn.handle_webhook(body, _signed_headers(priv, body)))
    assert res["ok"] is True
    # Inline CHANNEL_MESSAGE_WITH_SOURCE reply (type=4).
    assert res["response"]["type"] == 4
    assert res["response"]["data"]["content"] == "answer: 42"
    # Command + options flattened for the agent; sender carries guild/channel/user.
    assert seen == [("guild:G1:channel:C1:user:U1", "ask question=meaning of life")]


def test_interactions_allowlist_admits_by_channel_or_user_too() -> None:
    priv, pub_hex = _keypair()

    async def handler(sender: str, text: str) -> str:
        return "ok"

    # A connector that only allows a specific channel admits that channel's interaction
    # even though its guild/user are not listed.
    conn = _interactions_connector(handler, priv, pub_hex, allowed_channel_ids=["C1"])
    body = _command_body(guild="Gx", channel="C1", user="Ux")
    res = run_async(conn.handle_webhook(body, _signed_headers(priv, body)))
    assert res["ok"] is True and res["response"]["type"] == 4

    # And a user-only allowlist admits that user from any guild/channel.
    conn2 = _interactions_connector(handler, priv, pub_hex, allowed_user_ids=["U9"])
    body2 = _command_body(guild="Gz", channel="Cz", user="U9")
    res2 = run_async(conn2.handle_webhook(body2, _signed_headers(priv, body2)))
    assert res2["ok"] is True and res2["response"]["type"] == 4


def test_interactions_default_deny_drops_unlisted_sender() -> None:
    """An empty allowlist denies everyone even with a valid signature (default-deny)."""
    priv, pub_hex = _keypair()
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "leak"

    audit = _RecordingAudit()
    conn = _interactions_connector(handler, priv, pub_hex, audit=audit)  # no allowlist
    body = _command_body()
    res = run_async(conn.handle_webhook(body, _signed_headers(priv, body)))
    assert res["ok"] is False
    assert ran == []  # signature was valid, but the sender is not allow-listed
    assert audit.events[-1].outcome == "deny"


def test_interactions_unlisted_sender_dropped_even_if_signed() -> None:
    priv, pub_hex = _keypair()
    ran: list[str] = []

    async def handler(sender: str, text: str) -> str:
        ran.append(text)
        return "x"

    conn = _interactions_connector(handler, priv, pub_hex, allowed_guild_ids=["GOOD"])
    body = _command_body(guild="EVIL")
    res = run_async(conn.handle_webhook(body, _signed_headers(priv, body)))
    assert res["ok"] is False
    assert ran == []


def test_interactions_capability_requires_public_key() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    # No public key configured anywhere → unavailable, skipped cleanly.
    configure_secrets(_MemSecrets({}))
    try:
        conn = DiscordInteractionsConnector(handler)  # reads secret (absent)
        cap = conn.capability()
        assert cap.ok is False
        assert DISCORD_PUBLIC_KEY_SECRET in cap.reason
    finally:
        configure_secrets(None)


def test_interactions_public_key_read_from_secrets_layer() -> None:
    _, pub_hex = _keypair()

    async def handler(sender: str, text: str) -> str:
        return ""

    configure_secrets(_MemSecrets({DISCORD_PUBLIC_KEY_SECRET: pub_hex}))
    try:
        conn = DiscordInteractionsConnector(handler, allowed_guild_ids=["G1"])
        assert conn.capability().ok is True
    finally:
        configure_secrets(None)


def test_interactions_is_webhook_only_not_polling() -> None:
    priv, pub_hex = _keypair()

    async def handler(sender: str, text: str) -> str:
        return ""

    conn = _interactions_connector(handler, priv, pub_hex)
    with pytest.raises(ConnectorError, match="webhook-only"):
        run_async(conn.fetch_updates(None))


# ------------------------------------------------------ outbound: post-message


def _mock_transport(app: Any) -> Any:
    import httpx

    return httpx.MockTransport(app)


def test_post_message_sends_bot_auth_and_body_via_mock_transport() -> None:
    """Contract: assert the exact request discord_post_message puts on the wire."""
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"id": "987654321", "channel_id": "C1"})

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "bot.secret.token"}))
    try:
        # Inject a DiscordClient bound to the mock transport (no real socket / DNS).
        def factory(token: str) -> DiscordClient:
            return DiscordClient(token, transport=_mock_transport(app))

        conn = DiscordOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=factory,
        )
        registry = ToolRegistry()
        names = conn.register_tools(registry)
        assert names == ["discord_post_message"]
        out = run_async(
            registry.handler_for("discord_post_message")(
                {"content": "hi from himmy", "channel_id": "C1"}
            )
        )
        assert out["sent"] is True
        assert out["message_id"] == "987654321"
        assert seen["method"] == "POST"
        assert seen["url"] == "https://discord.com/api/v10/channels/C1/messages"
        # Discord bot auth uses the "Bot " scheme, carrying the secret-layer token.
        assert seen["auth"] == "Bot bot.secret.token"
        assert seen["body"]["content"] == "hi from himmy"
        assert "nonce" in seen["body"]  # idempotency token rides to Discord too
    finally:
        configure_secrets(None)


def test_post_message_denies_unlisted_channel_default_deny() -> None:
    """A channel NOT on the allowlist is refused before any HTTP call (default-deny)."""
    calls = {"n": 0}

    def factory(token: str) -> DiscordClient:
        class _NoNet(DiscordClient):
            def post_message(self_inner, *a: Any, **k: Any) -> dict[str, Any]:
                calls["n"] += 1
                return {"id": "x"}

        return _NoNet(token)

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        audit = _RecordingAudit()
        conn = DiscordOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=factory,
            context=ConnectorContext(audit=AuditSink(audit.record)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        with pytest.raises(
            ToolSecurityError, match="not on the Discord post allowlist"
        ):
            run_async(
                registry.handler_for("discord_post_message")(
                    {"content": "x", "channel_id": "C-EVIL"}
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
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "1"})

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        conn = DiscordOutboundConnector(
            default_channel_id="C-DEFAULT",
            requires_approval=False,
            client_factory=lambda t: DiscordClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        out = run_async(registry.handler_for("discord_post_message")({"content": "hi"}))
        assert out["channel_id"] == "C-DEFAULT"
        assert seen["url"].endswith("/channels/C-DEFAULT/messages")
    finally:
        configure_secrets(None)


def test_post_message_idempotent_dedupes_repeat() -> None:
    """The same (channel, content) posts exactly once across a repeated tool call."""
    import httpx

    hits = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json={"id": str(hits["n"])})

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        conn = DiscordOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: DiscordClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        args = {"content": "same message", "channel_id": "C1"}
        first = run_async(registry.handler_for("discord_post_message")(dict(args)))
        second = run_async(registry.handler_for("discord_post_message")(dict(args)))
        assert first == second
        assert hits["n"] == 1  # the effect fired exactly once
    finally:
        configure_secrets(None)


def test_post_message_capability_gated_on_token() -> None:
    """Without the bot token the connector is unavailable and registers no tools."""
    configure_secrets(_MemSecrets({}))
    try:
        conn = DiscordOutboundConnector(allowed_channel_ids=["C1"])
        assert conn.available() is False
        registry = ToolRegistry()
        assert conn.register_tools(registry) == []  # skipped cleanly, no crash
    finally:
        configure_secrets(None)


def test_post_message_token_read_from_secrets_layer_not_env(monkeypatch: Any) -> None:
    """The token comes from the secrets layer, never straight from os.environ."""
    import httpx

    monkeypatch.setenv(DISCORD_BOT_TOKEN_SECRET, "from-env-MUST-NOT-USE")
    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": "1"})

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "from-secrets-layer"}))
    try:
        conn = DiscordOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: DiscordClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        run_async(
            registry.handler_for("discord_post_message")(
                {"content": "x", "channel_id": "C1"}
            )
        )
        assert seen["auth"] == "Bot from-secrets-layer"
    finally:
        configure_secrets(None)


def test_post_message_error_never_leaks_token_in_message() -> None:
    """A Discord API failure surfaces a status, not the body — and never the token."""
    import httpx

    token = "bot.SUPER.SECRET"

    def app(request: httpx.Request) -> httpx.Response:
        # An error body that (maliciously) echoes the auth header back.
        return httpx.Response(
            403, json={"message": f"forbidden auth Bot {token}", "code": 50001}
        )

    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: token}))
    try:
        conn = DiscordOutboundConnector(
            allowed_channel_ids=["C1"],
            requires_approval=False,
            client_factory=lambda t: DiscordClient(t, transport=_mock_transport(app)),
        )
        registry = ToolRegistry()
        conn.register_tools(registry)
        with pytest.raises(ConnectorError) as exc:
            run_async(
                registry.handler_for("discord_post_message")(
                    {"content": "x", "channel_id": "C1"}
                )
            )
        assert token not in str(exc.value)  # the token never rides out in an error
        assert "403" in str(exc.value)  # but the operator still learns it was a 403
    finally:
        configure_secrets(None)


def test_post_message_is_approval_gated_by_default() -> None:
    """The side-effecting post tool defaults to human-in-the-loop approval."""
    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        conn = DiscordOutboundConnector(allowed_channel_ids=["C1"])  # default approval
        registry = ToolRegistry()
        conn.register_tools(registry)
        assert registry.get("discord_post_message").requires_approval is True
    finally:
        configure_secrets(None)


def test_outbound_default_fetcher_carries_ssrf_guard() -> None:
    """The connector's default fetcher refuses an SSRF target (the guard is wired)."""
    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        conn = DiscordOutboundConnector(allowed_channel_ids=["C1"])
        with pytest.raises(ToolSecurityError):
            conn.guard("http://169.254.169.254/latest/meta-data/")
        # And the egress allow-list pins it to the Discord host.
        with pytest.raises(ToolSecurityError, match="egress allow-list"):
            conn.guard("https://evil.example.test/webhook")
    finally:
        configure_secrets(None)


# ---------------------------------------------------- optional gateway listener


def test_gateway_listener_default_deny_allowlist() -> None:
    async def handler(sender: str, text: str) -> str:
        return ""

    # No allowlist at all → deny everyone (a bot can see every message it has access to).
    listener = DiscordGatewayListener(handler)
    assert listener.is_allowed(guild_id="G", channel_id="C", user_id="U") is False

    listener2 = DiscordGatewayListener(handler, allowed_channel_ids=["C1"])
    assert listener2.is_allowed(guild_id="Gx", channel_id="C1", user_id="Ux") is True
    assert listener2.is_allowed(guild_id="Gx", channel_id="Cx", user_id="Ux") is False


def test_gateway_listener_capability_gated_on_lib_and_token() -> None:
    """The gateway path is unavailable without BOTH the `discord` extra and a token.

    It is skipped cleanly (capability not OK) whenever either is missing — the offline
    test asserts that invariant rather than which one happens to be absent here. The
    live gateway run itself is live-pending (needs a real bot + Message Content intent).
    """
    configure_secrets(_MemSecrets({}))  # no bot token
    try:
        cap = DiscordGatewayListener.capability()
        assert cap.ok is False  # token absent ⇒ unavailable regardless of the lib
        assert "discord" in cap.reason or DISCORD_BOT_TOKEN_SECRET in cap.reason
    finally:
        configure_secrets(None)
    # With both present it would be available; we can only force one side offline (the
    # token), since whether the `discord` lib is installed varies by environment.
    _, _pub = _keypair()
    configure_secrets(_MemSecrets({DISCORD_BOT_TOKEN_SECRET: "tok"}))
    try:
        import importlib.util

        cap2 = DiscordGatewayListener.capability()
        if importlib.util.find_spec("discord") is not None:
            assert cap2.ok is True  # lib + token both present ⇒ available
        else:
            assert cap2.ok is False and "discord" in cap2.reason
    finally:
        configure_secrets(None)
