"""Tests for the connector SDK: registry, allowlist default-deny, secret redaction,
egress guard enforcement, idempotency, signatures, rate limiting, audit events.

All offline — no network, no real credentials. External-service connectors (the ones
that need a live Slack/GitHub token) are contract-tested against mocked transports
elsewhere; this suite proves the security spine the whole SDK rests on.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.sdk import (
    AuditSink,
    Capability,
    Connector,
    ConnectorContext,
    ConnectorError,
    ConnectorKind,
    ConnectorRegistry,
    IdempotencyStore,
    InboundChannelConnector,
    InboundMessage,
    OutboundToolConnector,
    RateLimiter,
    RetryPolicy,
    capability,
    redact_args,
    safe_error,
    verify_hmac_signature,
)
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async


class _MemSecrets:
    """An in-memory secret provider for tests (installed via configure_secrets)."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _RecordingAudit:
    """Captures recorded SecurityEvents."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


# --------------------------------------------------------------------- registry


class _StubInbound(InboundChannelConnector):
    name = "stub-inbound"

    async def fetch_updates(self, offset: int | None) -> list[Any]:
        return []

    def parse_update(self, update: Any) -> InboundMessage | None:
        return None

    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        return None


class _StubOutbound(OutboundToolConnector):
    name = "stub-outbound"
    required_secrets = ("STUB_TOKEN",)

    def register(self, registry: ToolRegistry) -> list[str]:
        return []


def _make_inbound(**kw: Any) -> _StubInbound:
    async def handler(sender: str, text: str) -> str:
        return f"echo:{text}"

    return _StubInbound(handler, **kw)


def test_registry_register_and_discover_by_kind() -> None:
    reg = ConnectorRegistry()
    reg.register("stub-inbound", _make_inbound)
    reg.register("stub-outbound", _StubOutbound)
    assert reg.names() == ["stub-inbound", "stub-outbound"]
    inbound = reg.discover(kind=ConnectorKind.INBOUND, available_only=False)
    assert [c.name for c in inbound] == ["stub-inbound"]


def test_registry_unknown_connector_raises() -> None:
    reg = ConnectorRegistry()
    with pytest.raises(ConnectorError, match="no connector registered"):
        reg.create("nope")
    assert reg.get("nope") is None


def test_registry_discover_skips_unavailable_connectors() -> None:
    """A connector whose required secret is absent is skipped cleanly by discovery."""
    configure_secrets(_MemSecrets({}))  # no STUB_TOKEN
    try:
        reg = ConnectorRegistry()
        reg.register("stub-outbound", _StubOutbound)
        assert reg.discover(available_only=True) == []  # skipped, not crashed
        assert [c.name for c in reg.discover(available_only=False)] == ["stub-outbound"]
    finally:
        configure_secrets(None)


# ------------------------------------------------------------ allowlist (deny)


def test_inbound_allowlist_default_denies_with_empty_list() -> None:
    """An empty allowlist denies everyone by default (the webhook-safe posture)."""
    conn = _make_inbound(allowed_senders=[])
    assert conn.is_allowed("anyone") is False


def test_inbound_allowlist_admits_only_listed_senders() -> None:
    conn = _make_inbound(allowed_senders=["5", "7"])
    assert conn.is_allowed("5") is True
    assert conn.is_allowed(5) is True  # int coerced to string
    assert conn.is_allowed("9") is False


def test_inbound_empty_allowlist_opt_in_answers_everyone() -> None:
    """A private poll bot may opt into answering everyone via allow_empty_allowlist."""
    conn = _make_inbound(allowed_senders=[], allow_empty_allowlist=True)
    assert conn.is_allowed("anyone") is True


def test_inbound_dispatch_drops_unlisted_sender_and_audits_deny() -> None:
    audit = _RecordingAudit()
    replies: list[str] = []

    async def handler(sender: str, text: str) -> str:
        replies.append(text)
        return "hi"

    conn = _StubInbound(
        handler,
        allowed_senders=["good"],
        context=ConnectorContext(audit=AuditSink(audit.record)),
    )
    out = run_async(
        conn._dispatch(InboundMessage(sender_id="evil", text="attack"))
    )
    assert out is None
    assert replies == []  # handler never ran for the unlisted sender
    assert audit.events and audit.events[-1].outcome == "deny"


# -------------------------------------------------------- HMAC signature verify


def test_verify_hmac_signature_accepts_correct_and_rejects_wrong() -> None:
    secret = "shhh"
    body = b'{"event":"ping"}'
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_hmac_signature(secret=secret, body=body, provided=good) is True
    assert verify_hmac_signature(secret=secret, body=body, provided="deadbeef") is False
    # Missing signature is denied, never crashes.
    assert verify_hmac_signature(secret=secret, body=body, provided=None) is False


def test_verify_hmac_signature_strips_scheme_prefix() -> None:
    secret = "k"
    body = b"payload"
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_hmac_signature(
        secret=secret, body=body, provided=f"sha256={digest}", prefix="sha256="
    )


def test_webhook_requires_valid_signature_before_handler_runs() -> None:
    """A bad signature is denied on the raw body before the handler/parse ever runs."""
    ran: list[str] = []

    class _Webhookable(InboundChannelConnector):
        name = "wh"

        async def fetch_updates(self, offset: int | None) -> list[Any]:
            return []

        def parse_update(self, update: Any) -> InboundMessage | None:
            return None

        def parse_webhook(self, body: bytes, headers: Any) -> InboundMessage | None:
            ran.append("parsed")
            return InboundMessage(sender_id="ok", text="hi")

        async def send_reply(self, msg: InboundMessage, reply: str) -> None:
            return None

    async def handler(sender: str, text: str) -> str:
        ran.append("handled")
        return "reply"

    audit = _RecordingAudit()
    conn = _Webhookable(
        handler,
        allowed_senders=["ok"],
        signing_secret="topsecret",
        signature_header="X-Signature",
        context=ConnectorContext(audit=AuditSink(audit.record)),
    )
    body = b'{"x":1}'
    # Wrong signature: denied, parser+handler never touched.
    res = run_async(conn.handle_webhook(body, {"X-Signature": "wrong"}))
    assert res["ok"] is False
    assert ran == []
    assert audit.events[-1].outcome == "deny"
    # Correct signature: parsed, allow-listed, handled.
    good = hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    res = run_async(conn.handle_webhook(body, {"X-Signature": good}))
    assert res["ok"] is True and res["reply"] == "reply"
    assert ran == ["parsed", "handled"]


# ------------------------------------------------------------- secret handling


def test_outbound_secret_read_from_secrets_layer_not_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("STUB_TOKEN", "from-env")  # must NOT be read directly
    configure_secrets(_MemSecrets({"STUB_TOKEN": "from-secrets-layer"}))
    try:
        conn = _StubOutbound()
        assert conn.secret("STUB_TOKEN") == "from-secrets-layer"
    finally:
        configure_secrets(None)


def test_outbound_missing_required_secret_names_var_only() -> None:
    configure_secrets(_MemSecrets({}))
    try:
        conn = _StubOutbound()
        with pytest.raises(ConnectorError, match="STUB_TOKEN"):
            conn.secret("STUB_TOKEN")
    finally:
        configure_secrets(None)


def test_safe_error_never_leaks_a_secret_value() -> None:
    """An arbitrary exception is reduced to its type; literal secrets are scrubbed."""
    leaky = RuntimeError("auth failed for token=SUPERSECRET123")
    msg = safe_error(leaky, secrets=["SUPERSECRET123"])
    assert "SUPERSECRET123" not in msg
    assert msg == "RuntimeError"  # message body dropped for an unknown exception type
    # A ConnectorError message is authored safe, so it is preserved — but a stray
    # secret still gets scrubbed as a belt-and-braces guard.
    safe = ConnectorError("needs token=SUPERSECRET123")
    out = safe_error(safe, secrets=["SUPERSECRET123"])
    assert "SUPERSECRET123" not in out
    assert "ConnectorError" in out


def test_redact_args_masks_secret_looking_keys() -> None:
    """Arg dicts headed for an audit detail have secret-looking values redacted."""
    redacted = redact_args(
        {"city": "kathmandu", "api_key": "sk-live-123", "token": "abc"}
    )
    assert redacted["city"] == "kathmandu"
    assert redacted["api_key"] != "sk-live-123"
    assert redacted["token"] != "abc"


# ------------------------------------------------------- egress guard (SSRF)


def test_outbound_guard_rejects_private_and_loopback_hosts() -> None:
    conn = _StubOutbound()
    for bad in ("http://127.0.0.1/x", "http://169.254.169.254/latest", "http://10.0.0.1/"):
        with pytest.raises(ToolSecurityError):
            conn.guard(bad)


def test_outbound_guard_enforces_egress_allowlist() -> None:
    """With an egress allow-list set, a public host NOT on the list is denied."""
    conn = _StubOutbound(egress_allow_hosts=["api.allowed.test"])
    with pytest.raises(ToolSecurityError, match="egress allow-list"):
        conn.guard("https://api.other.test/v1")


def test_outbound_default_fetcher_carries_a_guard() -> None:
    """The default outbound fetcher refuses an SSRF target (proves the guard is wired)."""
    conn = _StubOutbound()
    with pytest.raises(ToolSecurityError):
        conn.fetcher.get_text("http://127.0.0.1:9/secret")


# --------------------------------------------------------------- idempotency


def test_idempotency_store_runs_once_and_caches() -> None:
    store = IdempotencyStore()
    calls = {"n": 0}

    def effect() -> str:
        calls["n"] += 1
        return f"result-{calls['n']}"

    first = store.run_once("k1", effect)
    second = store.run_once("k1", effect)  # same key → cached, effect not re-fired
    assert first == "result-1"
    assert second == "result-1"
    assert calls["n"] == 1
    # A different key fires the effect again.
    assert store.run_once("k2", effect) == "result-2"
    assert calls["n"] == 2


def test_call_idempotent_dedupes_a_retry() -> None:
    """A side-effecting call wrapped with the same key fires its effect exactly once."""
    conn = _StubOutbound()
    fired = {"n": 0}

    def post() -> dict[str, Any]:
        fired["n"] += 1
        return {"id": fired["n"]}

    a = conn.call_idempotent("order:42", post)
    b = conn.call_idempotent("order:42", post)
    assert a == b == {"id": 1}
    assert fired["n"] == 1


# --------------------------------------------------------------- retry policy


def test_retry_policy_does_not_retry_non_idempotent_calls() -> None:
    """A non-idempotent call is attempted exactly once even if it raises (no double-send)."""
    policy = RetryPolicy(attempts=3, retry_on=(RuntimeError,))
    calls = {"n": 0}

    def call() -> None:
        calls["n"] += 1
        raise RuntimeError("timeout")

    with pytest.raises(RuntimeError):
        policy.run(call, idempotent=False, sleep=lambda _s: None)
    assert calls["n"] == 1  # NOT retried — the effect might have landed


def test_retry_policy_retries_idempotent_calls_up_to_attempts() -> None:
    policy = RetryPolicy(attempts=3, retry_on=(RuntimeError,))
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert policy.run(call, idempotent=True, sleep=lambda _s: None) == "ok"
    assert calls["n"] == 3


def test_retry_policy_only_retries_declared_exceptions() -> None:
    policy = RetryPolicy(attempts=3, retry_on=(RuntimeError,))

    def call() -> None:
        raise ValueError("not retryable")

    calls = {"n": 0}

    def counting() -> None:
        calls["n"] += 1
        call()

    with pytest.raises(ValueError):
        policy.run(counting, idempotent=True, sleep=lambda _s: None)
    assert calls["n"] == 1


# --------------------------------------------------------------- rate limiting


def test_rate_limiter_blocks_when_bucket_empty() -> None:
    clock = {"t": 0.0}
    limiter = RateLimiter(rate=2, capacity=2, window=1.0, clock=lambda: clock["t"])
    limiter.check("c")  # token 1
    limiter.check("c")  # token 2
    with pytest.raises(ConnectorError, match="rate limit exceeded"):
        limiter.check("c")  # empty


def test_rate_limiter_refills_over_time() -> None:
    clock = {"t": 0.0}
    limiter = RateLimiter(rate=1, capacity=1, window=1.0, clock=lambda: clock["t"])
    limiter.check("c")
    with pytest.raises(ConnectorError):
        limiter.check("c")
    clock["t"] = 1.0  # one window later → one token back
    limiter.check("c")


def test_rate_limiter_is_per_key() -> None:
    clock = {"t": 0.0}
    limiter = RateLimiter(rate=1, capacity=1, window=1.0, clock=lambda: clock["t"])
    limiter.check("a")
    limiter.check("b")  # separate bucket, not throttled by 'a'
    with pytest.raises(ConnectorError):
        limiter.check("a")


# --------------------------------------------------------------- capability


def test_capability_reports_missing_module() -> None:
    cap = capability(requires_modules=["a_module_that_does_not_exist_xyz"])
    assert cap.ok is False
    assert "missing dependency" in cap.reason


def test_capability_reports_missing_secret_by_name_only() -> None:
    configure_secrets(_MemSecrets({}))
    try:
        cap = capability(requires_secrets=["MY_API_KEY"])
        assert cap.ok is False
        assert "MY_API_KEY" in cap.reason
    finally:
        configure_secrets(None)


def test_capability_require_raises_when_unavailable() -> None:
    with pytest.raises(ConnectorError):
        Capability(ok=False, reason="nope").require()


def test_connector_requires_a_name() -> None:
    class _Nameless(Connector):
        kind = ConnectorKind.OUTBOUND

    with pytest.raises(ConnectorError, match="non-empty `name`"):
        _Nameless()


# --------------------------------------------- audit spine (real EntityRegistry)


def test_connector_action_lands_on_the_real_audit_spine() -> None:
    """A connector's allow/deny is recorded as a tamper-evident security_event record."""
    from himmy.entities.registry import EntityRegistry
    from himmy.services.audit.log import SECURITY_EVENT_KIND, SecurityAuditLog

    registry = EntityRegistry()
    audit = AuditSink.from_registry(registry)

    async def handler(sender: str, text: str) -> str:
        return "ok"

    conn = _StubInbound(
        handler,
        allowed_senders=["good"],
        context=ConnectorContext(audit=audit),
    )
    # One allowed + one denied dispatch.
    run_async(conn._dispatch(InboundMessage(sender_id="good", text="hi")))
    run_async(conn._dispatch(InboundMessage(sender_id="evil", text="hi")))

    events = SecurityAuditLog(registry).recent()
    outcomes = {e.outcome for e in events}
    assert outcomes == {"allow", "deny"}
    assert all(e.event_type == "connector_action" for e in events)
    assert all(e.resource == "stub-inbound" for e in events)
    # Records are on the audit spine as the dedicated tamper-evident kind.
    assert registry.list_by_kind(SECURITY_EVENT_KIND)
