"""Generic inbound webhook connector on the SDK: a signed HTTP trigger for an agent.

This is the channel-agnostic counterpart to the Slack/Discord inbound connectors. Where
those speak a specific provider's envelope, :class:`WebhookInboundConnector` accepts a
*plain JSON* payload over a single authenticated HTTP endpoint and runs the agent with the
text it carries. It is the connector you wire when something that is NOT Slack/Discord/etc.
(a CI job, a form, a monitoring alert, your own service) needs to trigger an agent turn.

The security posture is the SDK's, inherited whole — default-deny everywhere:

1. **HMAC signature over the raw body** (a shared secret from the secrets layer,
   :data:`WEBHOOK_SIGNING_SECRET`) is verified FIRST, on the bytes, before any parsing —
   so a forged or unsigned request never reaches the JSON decoder, let alone the agent. A
   missing/empty/wrong signature is denied. A timestamp header, when sent, is folded into
   the signed material and a stale one is rejected (replay guard).
2. **Source allowlist (default-deny)**: the payload's ``source`` field (configurable) must
   be on the allowlist. An empty allowlist answers no one on this publicly-reachable path.
3. **Delivery de-duplication**: a repeated ``id`` (a webhook re-delivery) is acknowledged
   without re-running the agent, so an at-least-once sender can't double-trigger.
4. Only then does the agent run. Delivery is **synchronous by default** (the reply is
   returned in the HTTP response). An operator may instead opt into an **async ack +
   result callback**: the endpoint returns ``202 Accepted`` immediately and the agent's
   result is POSTed back to a pre-registered, SSRF-guarded callback URL (itself signed with
   the same secret so the receiver can verify us).

:func:`build_webhook_router` mounts the connector as a FastAPI router — on the BFF
(``app.include_router(build_webhook_router(...))``) or on a standalone app — reading the
raw request body so the signature is checked over exactly the bytes that were sent.

Nothing here imports a network SDK at module load; the optional async callback lazily
imports ``httpx`` only when a callback URL is configured.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

from himmy.connectors.sdk import (
    Capability,
    ConnectorContext,
    ConnectorError,
    IdempotencyStore,
    InboundChannelConnector,
    InboundMessage,
    MessageHandler,
    safe_error,
    verify_hmac_signature,
)
from himmy.toolkit._net import build_pinned_transport, guard_url

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps fastapi/httpx lazy
    from fastapi import APIRouter

# --------------------------------------------------------------------------- secrets
#: Secret name for the shared HMAC secret that signs an inbound webhook (and, when a
#: result callback is configured, the outbound callback too). Read through the secrets
#: layer — NEVER from ``os.environ`` directly, NEVER logged. (A NAME, not a value — the
#: S105 "hardcoded password" check is a false positive here.)
WEBHOOK_SIGNING_SECRET = "HIMMY_WEBHOOK_SIGNING_SECRET"  # noqa: S105

# --------------------------------------------------------------------------- headers
#: Header carrying the hex HMAC of the raw body. Matches the GitHub-style ``sha256=<hex>``
#: convention so off-the-shelf senders interoperate; the ``sha256=`` prefix is stripped
#: before comparison.
DEFAULT_SIGNATURE_HEADER = "X-Himmy-Signature"
#: Prefix on the signature value (scheme marker), stripped before the constant-time check.
DEFAULT_SIGNATURE_PREFIX = "sha256="
#: Optional header carrying the unix timestamp the request was signed at. When present it
#: is bound into the signed material (``{timestamp}.{body}``) and a stale one is rejected.
DEFAULT_TIMESTAMP_HEADER = "X-Himmy-Timestamp"

#: Reject a request whose timestamp is older than this many seconds (replay guard). Only
#: enforced when the sender includes the timestamp header.
DEFAULT_MAX_TIMESTAMP_SKEW = 60 * 5

#: Hard ceiling on an accepted webhook body. A multi-MB JSON blob is almost certainly an
#: abuse / misconfiguration, never a legitimate trigger; cap it so a flood can't exhaust
#: memory before the signature is even checked.
DEFAULT_MAX_BODY_BYTES = 256 * 1024

#: Payload fields the connector reads (all configurable on the connector). ``text`` is the
#: prompt handed to the agent; ``source`` is gated by the allowlist; ``id`` de-dupes a
#: re-delivery.
DEFAULT_TEXT_FIELD = "text"
DEFAULT_SOURCE_FIELD = "source"
DEFAULT_ID_FIELD = "id"


# --------------------------------------------------------------------- signing helper
def sign_webhook_body(
    *, secret: str, body: bytes, timestamp: str | None = None
) -> str:
    """Compute the ``sha256=<hex>`` signature a sender (or our callback) puts on the wire.

    Mirrors :meth:`WebhookInboundConnector.verify_webhook`: when ``timestamp`` is given the
    signed material is ``{timestamp}.{body}`` (binding the timestamp so it can't be swapped
    to defeat the replay guard); otherwise it is the raw ``body``. Exposed so a sender — and
    the connector's own result callback — produce a signature the verifier accepts, and so
    tests don't re-derive the scheme by hand.
    """
    import hashlib
    import hmac

    material = body if timestamp is None else (f"{timestamp}.".encode() + body)
    digest = hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()
    return f"{DEFAULT_SIGNATURE_PREFIX}{digest}"


def _header_lookup(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
    if not name:
        return None
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


# ------------------------------------------------------------------- result callback
#: ``callback(url, payload, signature_header_value)`` — POSTs the agent result back. The
#: default is a guarded httpx POST; injectable as a test seam.
ResultCallback = Callable[[str, Mapping[str, Any], str], Awaitable[None]]


class GuardedCallbackPoster:
    """POSTs the agent's result back to a pre-registered URL over the SSRF-guarded transport.

    Used only when the connector is configured with an async result callback. The URL is
    fixed by the operator (NOT taken from the inbound payload — a payload-supplied URL would
    be an SSRF hole), validated with :func:`guard_url` against the egress policy, and the
    request rides the DNS-rebinding-safe **pinned transport** with redirects OFF, so a 3xx
    can't bounce the signed result to an internal host. The body is signed with the shared
    secret so the receiver can authenticate us in turn.
    """

    def __init__(
        self,
        *,
        secret: str,
        timeout: float = 15.0,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = None,
        transport: Any = None,
    ) -> None:
        self._secret = secret
        self._timeout = timeout
        self._allow_private = allow_private_hosts
        self._allow_hosts = set(egress_allow_hosts) if egress_allow_hosts else None
        self._transport = transport

    async def __call__(
        self, url: str, payload: Mapping[str, Any], signature: str
    ) -> None:
        import asyncio

        await asyncio.to_thread(self._post, url, payload, signature)

    def _post(self, url: str, payload: Mapping[str, Any], signature: str) -> None:
        import httpx

        # SSRF gate, in two layers (mirrors the OutboundToolConnector seam):
        #  - here, the cheap OFFLINE checks (http/https, no embedded creds, host present,
        #    egress allow-list) — ``resolve=False`` so a name we won't actually dial (a
        #    test mock, or a deployment that resolves at connect time) isn't rejected by a
        #    premature DNS lookup;
        #  - the public-IP rule is enforced at CONNECT time by the pinned transport, which
        #    resolves once and dials the vetted literal IP (DNS-rebinding-safe). With a
        #    private/loopback LITERAL in the URL this offline+connect pair still refuses it.
        guarded = guard_url(
            url,
            allow_private=self._allow_private,
            allow_hosts=self._allow_hosts,
            resolve=False,
        )
        transport = self._transport
        if transport is None:
            transport = build_pinned_transport(
                allow_private=self._allow_private, allow_hosts=self._allow_hosts
            )
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            DEFAULT_SIGNATURE_HEADER: signature,
        }
        with httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,  # never bounce the signed result to another host
            transport=transport,
        ) as client:
            resp = client.post(guarded, content=body, headers=headers)
            if resp.status_code >= 400:
                # Surface only the status, never the response body (it may echo our payload).
                raise ConnectorError(
                    f"webhook result callback failed: HTTP {resp.status_code}"
                )


# ----------------------------------------------------------------- inbound connector
class WebhookInboundConnector(InboundChannelConnector):
    """Generic signed-webhook trigger: verify → allowlist → de-dupe → run the agent.

    Construct it with the agent ``handler`` (``async (sender_id, text) -> reply``) and the
    HMAC ``signing_secret`` (defaults to the secrets-layer :data:`WEBHOOK_SIGNING_SECRET`).
    Gate senders with ``allowed_sources``: the payload's ``source`` field must match
    (default-deny — an empty allowlist answers no one, the only safe posture for a public
    URL). The payload shape is minimal and configurable::

        {"source": "ci", "text": "deploy finished for build 1234", "id": "evt-1"}

    Delivery modes:

    * **Synchronous** (default) — :meth:`handle_webhook` runs the agent inline and returns
      ``{"ok": True, "handled": ..., "reply": ...}``; the router serializes that as the HTTP
      response body.
    * **Async ack + result callback** — pass a ``result_callback`` (and optionally
      ``ack_status=202``): :meth:`handle_webhook` still runs the agent (so back-pressure /
      rate-limit / errors are observed) but POSTs the result to the operator-fixed callback
      URL, signed with the same secret, instead of (or in addition to) returning it inline.

    Replay/duplicate safety: a repeated ``id`` is acknowledged without re-running the agent
    (an :class:`IdempotencyStore`), and a stale signed timestamp is rejected.
    """

    name = "webhook"

    def __init__(
        self,
        handler: MessageHandler,
        *,
        signing_secret: str | None = None,
        allowed_sources: Collection[str] | None = None,
        signature_header: str = DEFAULT_SIGNATURE_HEADER,
        signature_prefix: str = DEFAULT_SIGNATURE_PREFIX,
        timestamp_header: str = DEFAULT_TIMESTAMP_HEADER,
        max_timestamp_skew: int = DEFAULT_MAX_TIMESTAMP_SKEW,
        text_field: str = DEFAULT_TEXT_FIELD,
        source_field: str = DEFAULT_SOURCE_FIELD,
        id_field: str = DEFAULT_ID_FIELD,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        result_callback: ResultCallback | None = None,
        callback_url: str | None = None,
        ack_only: bool = False,
        idempotency: IdempotencyStore | None = None,
        clock: Callable[[], float] | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        """Wire the agent ``handler``, the signing secret, the source allowlist + options.

        ``signing_secret`` defaults to the secrets-layer value so production reads it from
        the configured backend; tests pass it explicitly. ``allowed_sources`` are compared
        as strings against the payload's ``source`` field — empty ⇒ default-deny.

        ``result_callback`` + ``callback_url`` enable the async-callback mode: the agent's
        result is POSTed to ``callback_url`` (an operator-fixed URL, never payload-supplied)
        rather than returned inline. ``ack_only`` additionally suppresses the inline reply so
        the HTTP response is a bare ack (use with the router's ``202`` status). Passing a
        callback without a URL, or vice versa, is a configuration error.
        """
        from himmy.config.secrets import get_secret

        secret = (
            signing_secret
            if signing_secret is not None
            else (get_secret(WEBHOOK_SIGNING_SECRET) or "")
        )
        super().__init__(
            handler,
            allowed_senders={str(s) for s in (allowed_sources or [])},
            signing_secret=secret or None,
            signature_header=signature_header,
            signature_prefix=signature_prefix,
            signature_algorithm="sha256",
            # default-deny: an empty allowlist denies everyone (webhook-safe posture)
            allow_empty_allowlist=False,
            context=context,
        )
        self._secret = secret
        self._timestamp_header = timestamp_header
        self._max_skew = max_timestamp_skew
        self._text_field = text_field
        self._source_field = source_field
        self._id_field = id_field
        self._max_body_bytes = max_body_bytes
        self._clock = clock or time.time
        self._dedupe = idempotency or IdempotencyStore()

        if (result_callback is None) != (callback_url is None):
            raise ConnectorError(
                "webhook: result_callback and callback_url must be set together"
            )
        self._result_callback = result_callback
        self._callback_url = callback_url
        self._ack_only = ack_only

    # -- capability ---------------------------------------------------------
    def capability(self) -> Capability:
        """Available only when a signing secret is configured (a public URL MUST be signed).

        Refusing to come up without a secret is deliberate: an unsigned generic webhook is a
        trivially-forgeable agent trigger. The reason names only the secret variable.
        """
        if not self._secret:
            return Capability(
                ok=False, reason=f"missing credential {WEBHOOK_SIGNING_SECRET!r}"
            )
        return Capability(ok=True)

    # -- security gate: HMAC over the raw body (+ optional timestamp binding) ----
    def verify_webhook(self, body: bytes, headers: Mapping[str, str]) -> bool:
        """Verify the body's HMAC signature; reject a stale timestamp (default-deny).

        With no configured secret this connector is unavailable (see :meth:`capability`), so
        verification here always has a secret to check against. When the sender includes the
        timestamp header it is bound into the signed material (``{ts}.{body}``) and a
        timestamp older than ``max_timestamp_skew`` is rejected before the HMAC is even
        computed — closing a capture-replay window. A missing/forged signature returns False.
        """
        if not self._secret:
            return False
        provided = _header_lookup(headers, self._signature_header)
        if not provided:
            return False
        ts = _header_lookup(headers, self._timestamp_header)
        material = body
        if ts is not None:
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                return False
            if abs(self._clock() - ts_int) > self._max_skew:
                return False  # stale capture — replay guard
            material = f"{ts}.".encode() + body
        return verify_hmac_signature(
            secret=self._secret,
            body=material,
            provided=provided,
            algorithm="sha256",
            prefix=self._signature_prefix,
        )

    # -- the source allowlist sits on the SDK's sender allowlist ----------------
    def is_allowed(self, sender_id: str) -> bool:
        """Whether the payload ``source`` is allow-listed (default-deny on empty)."""
        if not self._allowed:
            return self._allow_empty_allowlist
        return str(sender_id) in self._allowed

    # -- webhook delivery ---------------------------------------------------
    async def handle_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """Verify → parse → allowlist → de-dupe → run the agent (sync reply or async callback).

        The order is load-bearing and matches the rest of the connector layer: the HMAC is
        checked on the RAW bytes before the JSON is parsed, so a forged payload never reaches
        the decoder. An over-large body, a bad signature, an unlisted source, or a malformed
        payload is audited as a deny and returns ``ok=False`` WITHOUT running the agent. A
        re-delivered ``id`` is acknowledged (``deduplicated=True``) without re-running.

        De-duplication is mark-AFTER-success with an in-flight lease (Q4): the delivery id is
        recorded as consumed only once the agent turn COMPLETES, not before it starts. This
        closes two holes the prior seen()/run_once() split had — (1) a TOCTOU between the
        pre-check and the run, and (2) with a DURABLE store, a crash AFTER a mark-before-run
        would have permanently dropped the redelivery. Now a crash mid-run leaves only an
        expiring in-flight lease, so the redelivery cleanly re-runs; a genuine duplicate (a
        completed delivery within its TTL) is acknowledged without re-running; a CONCURRENT
        duplicate (a live in-flight lease) is acknowledged without firing a second agent turn.
        """
        if len(body) > self._max_body_bytes:
            self._audit("webhook", "deny", detail="body exceeds size cap")
            return {"ok": False, "reason": "payload too large"}
        if not self.verify_webhook(body, headers):
            self._audit("webhook", "deny", detail="signature verification failed")
            return {"ok": False, "reason": "invalid signature"}

        msg = self.parse_webhook(body, headers)
        if msg is None or not msg.text:
            return {"ok": True, "handled": False}
        if not self._authorize(msg):  # source allowlist gate (audits the deny)
            return {"ok": False, "reason": "source not allowed"}

        delivery_id = str(msg.raw.get(self._id_field) or "")
        # No delivery id ⇒ no dedup key; run unconditionally (the prior behaviour).
        if not delivery_id:
            self._throttle(msg.sender_id)
            reply = await self._handler(msg.sender_id, msg.text)
            self._audit(
                "inbound_message",
                "allow",
                actor={"subject": msg.sender_id},
                detail="handled",
            )
            return await self._deliver(msg, reply)

        # Dedup AROUND the agent turn: ``run_once_async`` runs the handler at most once per
        # id and records the consumed mark only AFTER it succeeds. A duplicate (completed) or
        # concurrent duplicate (in flight) is recognised by the store and never re-fires.
        # The closure flips ``ran`` only when the handler ACTUALLY executes, so a cached
        # replay (same id, already completed) is distinguishable from a fresh run.
        ran = False

        async def _run() -> str:
            nonlocal ran
            ran = True
            self._throttle(msg.sender_id)
            return await self._handler(msg.sender_id, msg.text)

        try:
            reply = await self._dedupe.run_once_async(delivery_id, _run)
        except ConnectorError:
            # A concurrent duplicate (another in-flight delivery of the same id): treat as a
            # de-duplicated ack rather than firing a second agent turn.
            ran = False
            reply = ""

        if not ran:
            self._audit(
                "webhook",
                "allow",
                actor={"subject": msg.sender_id},
                detail="duplicate delivery ignored",
            )
            return {"ok": True, "handled": False, "deduplicated": True}

        self._audit(
            "inbound_message",
            "allow",
            actor={"subject": msg.sender_id},
            detail="handled",
        )
        return await self._deliver(msg, reply)

    async def _deliver(self, msg: InboundMessage, reply: str) -> dict[str, Any]:
        """Return the reply inline, or POST it to the result callback (async-ack mode)."""
        if self._result_callback is None or self._callback_url is None:
            return {"ok": True, "handled": bool(reply), "reply": reply}
        await self._post_result(msg, reply)
        if self._ack_only:
            return {"ok": True, "handled": bool(reply), "accepted": True}
        return {"ok": True, "handled": bool(reply), "reply": reply, "accepted": True}

    async def _post_result(self, msg: InboundMessage, reply: str) -> None:
        """POST the agent result to the configured callback URL, signed with the secret.

        Best-effort: a callback failure is audited (secret-safe) but never raised, so a
        flaky receiver can't crash intake or make the sender retry the whole webhook.
        """
        if self._result_callback is None or self._callback_url is None:
            return
        delivery_id = str(msg.raw.get(self._id_field) or "")
        payload = {
            "source": msg.sender_id,
            "id": delivery_id,
            "reply": reply,
        }
        ts = str(int(self._clock()))
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = sign_webhook_body(secret=self._secret, body=body, timestamp=ts)
        try:
            await self._result_callback(self._callback_url, payload, signature)
        except Exception as exc:  # noqa: BLE001 - a callback failure must not crash intake
            self._audit(
                "webhook_callback",
                "deny",
                actor={"subject": msg.sender_id},
                detail=safe_error(exc, secrets=(self._secret,)),
            )

    def parse_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> InboundMessage | None:
        """Parse a signed JSON body into an :class:`InboundMessage` (None on malformed).

        Runs only AFTER :meth:`verify_webhook` has accepted the bytes. Reads the configured
        ``source`` / ``text`` / ``id`` fields; a non-object body, a missing/empty source, or
        a non-string text yields None (acknowledged, agent not run). The whole decoded object
        is kept on ``raw`` so the id (and any extra fields) are available downstream.
        """
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        source = payload.get(self._source_field)
        text = payload.get(self._text_field)
        if not isinstance(source, str) or not source:
            return None
        if not isinstance(text, str) or not text:
            return None
        return InboundMessage(sender_id=source, text=text, raw=dict(payload))

    # The webhook connector is push-only: there is no poll loop to implement.
    async def fetch_updates(self, offset: int | None) -> list[Any]:
        raise ConnectorError(
            "webhook is push-only; use handle_webhook, not polling"
        )

    def parse_update(self, update: Any) -> InboundMessage | None:  # pragma: no cover
        return None

    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        """No-op: a synchronous webhook returns its reply in the HTTP response.

        The reply is delivered either inline (the router serializes :meth:`handle_webhook`'s
        return) or via the result callback (:meth:`_post_result`); there is no separate
        per-message channel to push to, unlike Slack/Telegram.
        """
        return None


# ------------------------------------------------------------------- router factory
def build_webhook_router(
    connector: WebhookInboundConnector,
    *,
    path: str = "/v1/connectors/webhook",
    tags: Collection[str] = ("connectors", "webhook"),
) -> APIRouter:
    """Mount ``connector`` as a FastAPI router — on the BFF or a standalone app.

    The route reads the RAW request body (so the HMAC is checked over exactly the bytes the
    sender signed, not a re-serialized copy) and delegates to
    :meth:`WebhookInboundConnector.handle_webhook`. A rejected request (bad signature,
    unlisted source, oversize/malformed body) returns ``401``/``403``/``400`` with a
    generic, secret-free reason — never echoing the body or any internal detail. An accepted
    request returns ``200`` with the agent's reply, or ``202`` (ack) when the connector is in
    async-callback mode.

    Mount on the BFF::

        from himmy.connectors.webhook import WebhookInboundConnector, build_webhook_router
        app.include_router(build_webhook_router(WebhookInboundConnector(handler, ...)))

    The route inherits the app's global dependencies (auth/rate-limit) when mounted on the
    BFF; the connector's OWN HMAC + allowlist gate is independent and always applies, so it
    is equally safe mounted on a bare standalone app.
    """
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=list(tags))
    # 202 only makes sense when the reply is delivered out-of-band (the callback).
    ack_status = 202 if connector._result_callback is not None else 200  # noqa: SLF001

    async def receive_webhook(request: Request) -> JSONResponse:
        """Receive a signed webhook, run the agent (gated), return the reply or an ack."""
        body = await request.body()
        # Map the SDK's structured outcome onto HTTP status codes. The outcome strings are
        # the connector's own (secret-free) reasons; we never surface the request body.
        result = await connector.handle_webhook(body, dict(request.headers))
        if result.get("ok"):
            status = ack_status if result.get("accepted") else 200
            return JSONResponse(status_code=status, content=result)
        reason = str(result.get("reason") or "rejected")
        status_code = {
            "invalid signature": 401,
            "source not allowed": 403,
            "payload too large": 413,
        }.get(reason, 400)
        return JSONResponse(
            status_code=status_code, content={"ok": False, "reason": reason}
        )

    # ``from __future__ import annotations`` makes every annotation a STRING, so FastAPI
    # would evaluate ``"Request"`` against this module's globals — where fastapi is not
    # imported (it stays a lazy dependency) — and mistake the parameter for a query field
    # (a 422). Bind the already-resolved class objects onto the endpoint's annotations so
    # FastAPI sees the real ``Request`` type without any module-level fastapi import.
    receive_webhook.__annotations__["request"] = Request
    receive_webhook.__annotations__["return"] = JSONResponse
    router.add_api_route(path, receive_webhook, methods=["POST"], include_in_schema=True)

    return router


# ----------------------------------------------------- convenience: secret-sourced build
def make_webhook_connector(
    handler: MessageHandler,
    *,
    allowed_sources: Collection[str] | None = None,
    callback_url: str | None = None,
    egress_allow_hosts: Collection[str] | None = None,
    allow_private_hosts: bool = False,
    ack_only: bool = False,
    context: ConnectorContext | None = None,
) -> WebhookInboundConnector:
    """Build a :class:`WebhookInboundConnector` from the secrets/allowlist environment.

    Reads the signing secret from the secrets layer (:data:`WEBHOOK_SIGNING_SECRET`) and, if
    a ``callback_url`` is given, wires the SSRF-guarded :class:`GuardedCallbackPoster` (so the
    result POST is validated against the same egress policy and signed with the same secret).
    A deployment that sets only the secret gets a connector that is capability-OK but refuses
    every delivery (default-deny) until a source is allow-listed.
    """
    from himmy.config.secrets import get_secret

    secret = get_secret(WEBHOOK_SIGNING_SECRET) or ""
    poster: ResultCallback | None = None
    if callback_url is not None:
        poster = GuardedCallbackPoster(
            secret=secret,
            allow_private_hosts=allow_private_hosts,
            egress_allow_hosts=egress_allow_hosts,
        )
    return WebhookInboundConnector(
        handler,
        signing_secret=secret or None,
        allowed_sources=allowed_sources,
        result_callback=poster,
        callback_url=callback_url,
        ack_only=ack_only,
        context=context,
    )


__all__ = [
    "WEBHOOK_SIGNING_SECRET",
    "DEFAULT_SIGNATURE_HEADER",
    "DEFAULT_SIGNATURE_PREFIX",
    "DEFAULT_TIMESTAMP_HEADER",
    "DEFAULT_MAX_TIMESTAMP_SKEW",
    "DEFAULT_MAX_BODY_BYTES",
    "WebhookInboundConnector",
    "GuardedCallbackPoster",
    "ResultCallback",
    "build_webhook_router",
    "make_webhook_connector",
    "sign_webhook_body",
]
