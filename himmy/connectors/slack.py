"""Slack connector on the SDK: an Events-API bot + a post-message tool.

Two surfaces over Slack's public API, both inheriting the connector SDK's security
posture (default-deny allowlist, secrets-layer credentials, SSRF-guarded egress,
idempotent side effects, secret-safe errors, structured audit events):

* :class:`SlackEventsConnector` — the **inbound** side. Slack delivers events (a message
  in a channel, an ``app_mention``) to a single HTTP **Events API** request URL. Every
  delivery is signed: Slack computes ``HMAC-SHA256`` over the basestring
  ``v0:{timestamp}:{raw_body}`` keyed by the app's **signing secret** and sends it as
  ``X-Slack-Signature: v0=<hex>`` alongside ``X-Slack-Request-Timestamp``. We verify that
  signature on the RAW body BEFORE parsing (overriding the SDK's plain-body HMAC default,
  because Slack hashes a *versioned, timestamped* basestring), and we reject a stale
  timestamp to blunt replay. Only then is the sender gated by a
  **team/channel/user allowlist (default-deny)** before the agent ever runs. Slack's
  ``url_verification`` handshake is answered with its ``challenge`` without touching the
  allowlist or the agent. A real event runs the agent and the reply is posted back IN
  THREAD via the outbound client (Slack events do not carry an inline reply channel).

  Socket Mode (a WebSocket alternative to a public request URL) is the optional,
  capability-gated path for a deployment that cannot expose an inbound URL:
  :class:`SlackSocketModeListener` drives the ``slack_sdk`` library (an OPTIONAL extra +
  an app-level token). It shares the exact same signing-is-moot-but-allowlist-still-gates
  spine; it is **live-pending** (needs a live Slack app socket) so it carries no offline
  contract test beyond capability detection.

* :class:`SlackOutboundConnector` — the **outbound** side: a ``slack_post_message`` tool
  so an agent can post to a channel (optionally threaded). The bot token comes from the
  **secrets layer** (:data:`SLACK_BOT_TOKEN_SECRET`), never the model; the call goes
  through the connector's **SSRF-guarded pinned transport** pinned to the Slack API host;
  the target channel must be on the **allowlist**; and the post is **idempotent** (deduped
  on a per-message key so a retry cannot double-post).

Nothing here imports ``httpx`` or ``slack_sdk`` at module load — capability detection
probes lazily — so importing this module is free in an air-gapped install.

LIVE-PENDING: the request shapes, auth header, and v0 signing scheme are contract-tested
against a mocked transport (see ``tests/connectors/test_slack.py``); verifying against the
REAL Slack API needs a live Slack app + bot/signing/app-level tokens and a workspace, and
cannot run on this machine.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

from himmy.connectors.sdk import (
    Capability,
    ConnectorContext,
    ConnectorError,
    InboundChannelConnector,
    InboundMessage,
    MessageHandler,
    OutboundToolConnector,
    RateLimiter,
    capability,
    redact_args,
    safe_error,
)
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit._net import build_pinned_transport

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx stays a lazy import)
    import httpx

#: Slack Web API root. The connector pins its egress to this host.
SLACK_API_ROOT = "https://slack.com/api"
#: The single API host the connector is allowed to reach (egress allow-list).
SLACK_API_HOST = "slack.com"

#: Secret name for the bot (xoxb-) token, resolved via the secrets layer, NEVER the model.
#: This is the NAME of the env/secret var, not a credential value (S105 false positive).
SLACK_BOT_TOKEN_SECRET = "HIMMY_SLACK_BOT_TOKEN"  # noqa: S105
#: Secret name for the app **signing secret**, used to verify that an inbound Events API
#: request genuinely came from Slack. From the app's Basic Information page. (A NAME, not
#: a secret value — S105 false positive.)
SLACK_SIGNING_SECRET = "HIMMY_SLACK_SIGNING_SECRET"  # noqa: S105
#: Secret name for the app-level (xapp-) token used ONLY by the optional Socket Mode
#: listener. (A NAME, not a secret value — S105 false positive.)
SLACK_APP_TOKEN_SECRET = "HIMMY_SLACK_APP_TOKEN"  # noqa: S105

#: Slack's signature version prefix and the headers it signs each request with.
_SIG_VERSION = "v0"
_SIG_HEADER = "X-Slack-Signature"
_TS_HEADER = "X-Slack-Request-Timestamp"

#: Reject a request whose timestamp is older than this (seconds) to blunt replay — the
#: window Slack itself recommends.
_MAX_TIMESTAMP_SKEW = 60 * 5

#: Slack accepts very large messages, but a single ``chat.postMessage`` text over ~40k
#: chars is rejected; we clip defensively so an over-long agent reply degrades to a
#: truncated message instead of an API error.
_MAX_MESSAGE_LEN = 40000


def _truncate(text: str) -> str:
    """Clip ``text`` to Slack's practical per-message limit (keeps a trailing ellipsis)."""
    if len(text) <= _MAX_MESSAGE_LEN:
        return text
    return text[: _MAX_MESSAGE_LEN - 1] + "…"


def verify_slack_signature(
    *,
    signing_secret: str,
    timestamp: str,
    body: bytes,
    signature: str | None,
    now: float | None = None,
    max_skew: int = _MAX_TIMESTAMP_SKEW,
) -> bool:
    """Constant-time check of a Slack Events API request signature (default-deny).

    Slack signs the basestring ``v0:{timestamp}:{raw_body}`` with HMAC-SHA256 keyed by the
    app's signing secret and sends ``X-Slack-Signature: v0=<hex>`` plus
    ``X-Slack-Request-Timestamp``. This recomputes that and compares with
    :func:`hmac.compare_digest` (timing-safe). It returns False — never raises — for a
    missing/empty/malformed signature, a missing secret, a non-integer timestamp, or a
    timestamp older than ``max_skew`` (a replayed capture), so a caller treats any failure
    exactly like a forgery and denies it.
    """
    if not signature or not signing_secret or not timestamp:
        return False
    # Reject a stale timestamp (replay protection) before doing the HMAC work.
    try:
        ts_int = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - ts_int) > max_skew:
        return False
    basestring = f"{_SIG_VERSION}:{timestamp}:".encode() + body
    expected = (
        _SIG_VERSION
        + "="
        + hmac.new(
            signing_secret.encode("utf-8"), basestring, hashlib.sha256
        ).hexdigest()
    )
    return hmac.compare_digest(expected, signature.strip())


class SlackClient:
    """A tiny Web API client for the one outbound call: post a message to a channel.

    All HTTP rides on the SSRF-guarded **pinned transport** (the DNS-rebinding-safe one
    the rest of the connector layer uses) with redirects OFF, so a 3xx can never bounce an
    authenticated request to an internal host. The bot token is passed by the caller
    (resolved from the secrets layer) and sent as ``Authorization: Bearer <token>``; it is
    never logged. ``transport`` is a test seam for injecting an ``httpx`` mock.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = SLACK_API_ROOT,
        timeout: float = 20.0,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = (SLACK_API_HOST,),
        transport: Any = None,
    ) -> None:
        self._token = token
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._allow_private = allow_private_hosts
        self._allow_hosts = set(egress_allow_hosts) if egress_allow_hosts else None
        self._transport = transport

    def _auth_headers(self) -> dict[str, str]:
        # Slack Web API uses a standard Bearer token.
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _client(self) -> httpx.Client:
        import httpx

        transport = self._transport
        if transport is None:
            transport = build_pinned_transport(
                allow_private=self._allow_private, allow_hosts=self._allow_hosts
            )
        return httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,  # never auto-follow a 3xx with the bot token attached
            transport=transport,
        )

    def post_message(
        self,
        channel: str,
        text: str,
        *,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """POST ``text`` to ``channel`` (``chat.postMessage``), optionally in a thread.

        ``thread_ts`` replies in an existing thread (so an inbound mention is answered IN
        THREAD rather than as a new top-level message). Returns the API response object.

        Slack's Web API returns HTTP 200 even for a logical failure, signalling it with
        ``{"ok": false, "error": "<code>"}`` in the body; we surface ONLY that token-free
        error code (and the HTTP status for a transport-level failure), never the response
        body verbatim, so the token can never ride out in an error string.
        """
        url = f"{self._base}/chat.postMessage"
        payload: dict[str, Any] = {"channel": channel, "text": _truncate(text)}
        if thread_ts:
            payload["thread_ts"] = thread_ts
        with self._client() as client:
            resp = client.post(url, json=payload, headers=self._auth_headers())
            if resp.status_code >= 400:
                raise ConnectorError(
                    f"slack chat.postMessage failed: HTTP {resp.status_code}"
                )
            try:
                data = dict(resp.json())
            except Exception:  # noqa: BLE001 - non-JSON body
                raise ConnectorError(
                    "slack chat.postMessage returned a non-JSON body"
                ) from None
            if not data.get("ok", False):
                # Surface only Slack's own (token-free) error code, never the raw body.
                code = str(data.get("error", "")) or "unknown"
                raise ConnectorError(f"slack chat.postMessage failed: {code}")
            return data


# --------------------------------------------------------------- outbound tool
_POST_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The message text to post (clipped to Slack's limit).",
        },
        "channel": {
            "type": "string",
            "description": (
                "Target Slack channel id (e.g. C0123ABCD). Must be on the connector's "
                "channel allowlist; defaults to the configured default channel."
            ),
        },
        "thread_ts": {
            "type": "string",
            "description": (
                "Optional parent message ts to reply in-thread instead of posting a new "
                "top-level message."
            ),
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}


class SlackOutboundConnector(OutboundToolConnector):
    """Exposes ``slack_post_message`` — post to an allow-listed channel, idempotently.

    Security contract (all inherited from :class:`OutboundToolConnector` + enforced here):

    * the bot token is read from the **secrets layer** (:data:`SLACK_BOT_TOKEN_SECRET`) at
      call time, never from a constructor arg the model could set, never logged;
    * the connector is **unavailable** (skipped cleanly by discovery / ``register_tools``)
      when the token is absent — capability-gated;
    * every request goes through the SSRF-guarded **pinned transport**, pinned to the Slack
      API host (egress allow-list);
    * the destination channel must be on the **channel allowlist** (default-deny: an empty
      allowlist with no configured default rejects every post);
    * the post is **idempotent** — deduped per ``(channel, thread, text)`` key so a retry
      or a re-issued tool call posts exactly once.
    """

    name = "slack"
    required_secrets = (SLACK_BOT_TOKEN_SECRET,)

    def __init__(
        self,
        *,
        allowed_channel_ids: Collection[str] | None = None,
        default_channel_id: str | None = None,
        requires_approval: bool = True,
        client_factory: Callable[[str | None], SlackClient] | None = None,
        context: ConnectorContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wire the channel allowlist + default channel.

        ``allowed_channel_ids`` is the allowlist (compared as strings). With it empty and
        no ``default_channel_id``, the tool denies every post — the default-deny posture.
        ``requires_approval`` gates the tool behind human-in-the-loop approval by default
        (a side-effecting "act" tool). ``client_factory`` is the injectable seam tests use
        to supply a :class:`SlackClient` over a mock transport.
        """
        # Default the egress allow-list to the Slack API host for defence in depth.
        kwargs.setdefault("egress_allow_hosts", (SLACK_API_HOST,))
        # A sane per-connector self-throttle so a runaway agent can't flood the API
        # (Slack's chat.postMessage tier is ~1 msg/sec/channel).
        if context is None:
            context = ConnectorContext(rate_limiter=RateLimiter(rate=1, window=1.0))
        super().__init__(context=context, **kwargs)
        self._allowed_channels = {str(c) for c in (allowed_channel_ids or [])}
        self._default_channel = str(default_channel_id) if default_channel_id else None
        self._requires_approval = requires_approval
        self._client_factory = client_factory

    def capability(self) -> Capability:
        """Available only when the bot token resolves through the secrets layer."""
        return capability(requires_secrets=self.required_secrets)

    def _channel_allowed(self, channel_id: str) -> bool:
        """Whether ``channel_id`` may be posted to (default-deny when unconfigured)."""
        if not self._allowed_channels:
            # No explicit allowlist: only the single configured default channel is
            # permitted (and only if one was configured). Never "any channel".
            return (
                self._default_channel is not None
                and channel_id == self._default_channel
            )
        return channel_id in self._allowed_channels

    def _build_client(self) -> SlackClient:
        token = self.secret(SLACK_BOT_TOKEN_SECRET)  # raises (named) if absent
        if self._client_factory is not None:
            return self._client_factory(token)
        return SlackClient(
            token or "",
            allow_private_hosts=self._allow_private,
            egress_allow_hosts=self._allow_hosts or (SLACK_API_HOST,),
        )

    def post(
        self, channel: str, text: str, thread_ts: str | None = None
    ) -> dict[str, Any]:
        """Post ``text`` to an allow-listed ``channel`` once (idempotent, audited).

        ``thread_ts`` is positional-or-keyword so this method satisfies the inbound
        connector's ``poster(channel, text, thread_ts)`` contract directly (it is wired as
        the reply poster).

        Shared by the tool handler AND the inbound connector's in-thread reply path: both
        funnel through the SAME channel-allowlist gate, throttle, idempotency key, and
        secret-safe error handling, so a reply can never escape to a non-allow-listed
        channel any more than a tool call can. Raises :class:`ConnectorError` on failure
        (the message is secret-safe).
        """
        if not self._channel_allowed(channel):
            self._audit(
                "slack_post_message",
                "deny",
                actor={"subject": channel},
                detail="channel not on allowlist",
            )
            raise ConnectorError(
                f"channel {channel!r} is not on the Slack post allowlist"
            )
        client = self._build_client()
        # Idempotency over (channel, thread, text). A SHA-256 digest is the local store
        # key: it is collision-resistant (unlike a PYTHONHASHSEED-salted, 32-bit hash()
        # that could silently drop a distinct message as a duplicate).
        digest = hashlib.sha256(
            f"{channel}:{thread_ts or ''}:{text}".encode()
        ).hexdigest()

        def _send() -> dict[str, Any]:
            # Throttle INSIDE the idempotent body so a deduplicated repeat (which fires no
            # network call) does not consume a rate-limit token — only an actual send does.
            self._throttle()
            return client.post_message(channel, text, thread_ts=thread_ts)

        try:
            return dict(self.call_idempotent(f"post:{digest}", _send))
        except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
            token = self.secret(SLACK_BOT_TOKEN_SECRET, required=False) or ""
            self._audit(
                "slack_post_message",
                "deny",
                actor={"subject": channel},
                detail=safe_error(exc, secrets=[token]),
            )
            raise ConnectorError(
                f"slack_post_message failed: {safe_error(exc, secrets=[token])}"
            ) from None

    def register(self, registry: ToolRegistry) -> list[str]:
        async def slack_post_message(args: dict[str, Any]) -> dict[str, Any]:
            import asyncio

            text = str(args.get("text") or "")
            if not text:
                raise ToolSecurityError("slack_post_message requires `text`")
            channel = str(args.get("channel") or self._default_channel or "")
            if not channel:
                raise ToolSecurityError(
                    "slack_post_message needs a channel (or a default channel)"
                )
            thread_ts = args.get("thread_ts")
            thread = str(thread_ts) if thread_ts else None
            # Channel-allowlist denial is raised as a ToolSecurityError (pre-flight, no
            # network) so it is distinguishable from a downstream API failure.
            if not self._channel_allowed(channel):
                self._audit(
                    "slack_post_message",
                    "deny",
                    actor={"subject": channel},
                    detail="channel not on allowlist",
                )
                raise ToolSecurityError(
                    f"channel {channel!r} is not on the Slack post allowlist"
                )
            try:
                message = await asyncio.to_thread(
                    self.post, channel, text, thread_ts=thread
                )
            except ConnectorError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
                token = self.secret(SLACK_BOT_TOKEN_SECRET, required=False) or ""
                raise ConnectorError(
                    f"slack_post_message failed: {safe_error(exc, secrets=[token])}"
                ) from None
            self._audit(
                "slack_post_message",
                "allow",
                actor={"subject": channel},
                detail=f"posted {dict(redact_args(args))}",
            )
            return {
                "sent": True,
                "channel": str(message.get("channel", channel)),
                "ts": str(message.get("ts", "")),
            }

        register_local_tool(
            registry,
            name="slack_post_message",
            read_only=False,
            handler=slack_post_message,
            description="Post a message to an allow-listed Slack channel via the bot.",
            args_json_schema=_POST_SCHEMA,
            requires_approval=self._requires_approval,
            sensitive_arg_names=[],
            metadata={"connector": "slack"},
        )
        return ["slack_post_message"]


# ------------------------------------------------------------ inbound events api
class SlackEventsConnector(InboundChannelConnector):
    """Slack bot over the signed **Events API** request URL (message / app_mention).

    Delivery is a webhook, so :meth:`handle_webhook` is the entry point. The security order
    is load-bearing:

    1. **Slack v0 signature** over the RAW ``v0:{timestamp}:{body}`` basestring is verified
       FIRST (we override the SDK's plain-body HMAC default with :meth:`verify_webhook`),
       and a stale timestamp is rejected (replay guard). A missing/forged signature is
       denied before the body is even parsed.
    2. The ``url_verification`` handshake (Slack confirming the request URL) is answered
       with the echoed ``challenge`` — it carries no sender and never reaches the allowlist
       or the agent.
    3. A real event (``message`` / ``app_mention``) is gated by the
       **team/channel/user allowlist (default-deny)**: the ``sender_id`` is the invoking
       ``team:channel:user`` triple, and the allowlist may name a whole team (workspace), a
       channel, or a specific user (any match admits). An empty allowlist denies everyone —
       a public request URL must name its senders. A message FROM a bot (including
       ourselves) is dropped first, so the bot can never answer its own post (loop guard).
    4. Only then does the agent run, and its reply is **posted back IN THREAD** via the
       supplied outbound poster (Slack events do not carry an inline reply channel).

    The signing secret (for step 1) comes from the **secrets layer**
    (:data:`SLACK_SIGNING_SECRET`); the connector is unavailable when it is absent.
    """

    name = "slack-events"

    def __init__(
        self,
        handler: MessageHandler,
        *,
        signing_secret: str | None = None,
        poster: Callable[[str, str, str | None], Any] | None = None,
        allowed_team_ids: Collection[str] | None = None,
        allowed_channel_ids: Collection[str] | None = None,
        allowed_user_ids: Collection[str] | None = None,
        max_timestamp_skew: int = _MAX_TIMESTAMP_SKEW,
        clock: Callable[[], float] | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        """Wire the agent ``handler`` plus the signing secret + team/channel/user allowlist.

        ``signing_secret`` defaults to the secrets-layer value (:data:`SLACK_SIGNING_SECRET`)
        so production reads it from the configured backend; tests pass it explicitly. The
        three allowlists are unioned into the SDK's sender allowlist as ``team:<id>`` /
        ``channel:<id>`` / ``user:<id>`` tokens, and a parsed event is admitted when ANY of
        its team/channel/user appears — so an operator can allow a whole workspace, a single
        channel, or a named user. All empty ⇒ default-deny (answer no one).

        ``poster(channel, text, thread_ts)`` sends the reply (normally
        :meth:`SlackOutboundConnector.post`, which re-applies the channel allowlist). A
        connector built without one verifies + gates + runs the agent but does not send the
        reply (useful for tests / a deployment that posts replies elsewhere).
        """
        from himmy.config.secrets import get_secret

        self._signing = (
            signing_secret
            if signing_secret is not None
            else (get_secret(SLACK_SIGNING_SECRET) or "")
        )
        self._poster = poster
        self._max_skew = max_timestamp_skew
        self._clock = clock
        allowed: set[str] = set()
        allowed.update(f"team:{t}" for t in (allowed_team_ids or []))
        allowed.update(f"channel:{c}" for c in (allowed_channel_ids or []))
        allowed.update(f"user:{u}" for u in (allowed_user_ids or []))
        super().__init__(
            handler,
            allowed_senders=allowed,
            signing_secret=self._signing,
            # default-deny: an empty allowlist denies everyone (webhook-safe posture)
            allow_empty_allowlist=False,
            context=context,
        )

    def capability(self) -> Capability:
        """Available only when the app signing secret is configured."""
        if not self._signing:
            return Capability(
                ok=False, reason=f"missing credential {SLACK_SIGNING_SECRET!r}"
            )
        return Capability(ok=True)

    # -- security gate: Slack's v0 signature, not the SDK's plain-body HMAC ----
    def verify_webhook(self, body: bytes, headers: Mapping[str, str]) -> bool:
        """Verify the request's Slack v0 signature over ``v0:{timestamp}:{body}``."""
        return verify_slack_signature(
            signing_secret=self._signing,
            timestamp=_header(headers, _TS_HEADER) or "",
            body=body,
            signature=_header(headers, _SIG_HEADER),
            now=self._clock() if self._clock else None,
            max_skew=self._max_skew,
        )

    def is_allowed(self, sender_id: str) -> bool:
        """Admit when the event's team OR channel OR user is allow-listed.

        ``sender_id`` is the ``team:<t>:channel:<c>:user:<u>`` triple parsed from the event;
        we split it back out and admit if any component token is on the allowlist. With an
        empty allowlist this is default-deny (False).
        """
        if not self._allowed:
            return self._allow_empty_allowlist
        for token in _sender_tokens(sender_id):
            if token in self._allowed:
                return True
        return False

    # -- webhook delivery (overrides the SDK to handle url_verification + thread) --
    async def handle_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """Verify, answer url_verification, allowlist-gate an event, run the agent, reply.

        Returns Slack's expected response: ``{"challenge": ...}`` for the handshake, and an
        internal ``ok``/``handled`` flag for the caller otherwise (Slack only needs a 200
        for an event). A bad signature, an event from a bot, or an unlisted sender returns
        ``ok=False``/``handled=False`` and is audited as a deny without running the agent.
        """
        if not self.verify_webhook(body, headers):
            self._audit("webhook", "deny", detail="invalid Slack signature")
            return {"ok": False, "reason": "invalid signature"}
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "reason": "malformed body"}
        if not isinstance(envelope, dict):
            return {"ok": False, "reason": "malformed body"}

        etype = envelope.get("type")
        if etype == "url_verification":
            # Slack confirming the request URL — echo the challenge, no agent, no gate.
            self._audit("webhook", "allow", detail="url_verification")
            return {
                "ok": True,
                "handled": False,
                "challenge": str(envelope.get("challenge", "")),
            }
        if etype != "event_callback":
            # We only act on event callbacks; anything else is acknowledged but ignored.
            return {"ok": True, "handled": False}

        msg = self._parse_event(envelope)
        if msg is None or not msg.text:
            return {"ok": True, "handled": False}
        if not self._authorize(msg):  # allowlist gate (audits the deny)
            return {"ok": False, "reason": "sender not allowed"}
        self._throttle(msg.sender_id)
        reply = await self._handler(msg.sender_id, msg.text)
        self._audit(
            "inbound_message",
            "allow",
            actor={"subject": msg.sender_id},
            detail="handled",
        )
        if reply and self._poster is not None:
            await self.send_reply(msg, reply)
        return {"ok": True, "handled": bool(reply), "reply": reply}

    def _parse_event(self, envelope: Mapping[str, Any]) -> InboundMessage | None:
        """Normalize a ``message`` / ``app_mention`` event into an :class:`InboundMessage`.

        Returns None for a non-message event, a message with no text, OR a message authored
        by a bot (a ``bot_id`` present, or our own retry/edit subtypes) — the loop guard
        that stops the bot answering its own posts. The ``sender_id`` encodes the
        team/channel/user so the allowlist can gate on any of them, and ``raw`` carries the
        event (so the caller can read ``thread_ts`` for the reply).
        """
        event = envelope.get("event")
        if not isinstance(event, dict):
            return None
        if event.get("type") not in ("message", "app_mention"):
            return None
        # Loop guard: never act on a bot's message (including our own). Slack flags a bot
        # message with `bot_id`; a `subtype` like message_changed/deleted is also ignored.
        if event.get("bot_id"):
            return None
        if event.get("subtype"):
            return None
        text = str(event.get("text") or "")
        if not text:
            return None
        team_id = str(envelope.get("team_id") or event.get("team") or "")
        channel_id = str(event.get("channel") or "")
        user_id = str(event.get("user") or "")
        if not user_id:
            return None
        sender_id = f"team:{team_id}:channel:{channel_id}:user:{user_id}"
        return InboundMessage(sender_id=sender_id, text=text, raw=dict(event))

    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        """Post ``reply`` back to the event's channel, threaded under the source message.

        Replies IN THREAD: Slack uses the source message's ``thread_ts`` when present (a
        reply to an existing thread) else its ``ts`` (start a thread under the message), so
        the bot's answer stays attached to the question. The reply goes through the supplied
        ``poster`` (normally the outbound connector's :meth:`SlackOutboundConnector.post`,
        which re-applies the channel allowlist), run in a worker thread so the blocking HTTP
        call does not stall the event loop.
        """
        if self._poster is None:
            return
        import asyncio

        raw = msg.raw or {}
        channel = str(raw.get("channel") or "")
        if not channel:
            return
        thread_ts = str(raw.get("thread_ts") or raw.get("ts") or "") or None

        def _post() -> Any:
            return self._poster(channel, reply, thread_ts)  # type: ignore[misc]

        try:
            await asyncio.to_thread(_post)
        except Exception as exc:  # noqa: BLE001 - a reply failure must not crash intake
            self._audit(
                "inbound_reply",
                "deny",
                actor={"subject": msg.sender_id},
                detail=safe_error(exc),
            )

    # The Events API is push-only — there is no poll loop to implement.
    async def fetch_updates(self, offset: int | None) -> list[Any]:
        raise ConnectorError(
            "slack-events is webhook-only; use handle_webhook, not polling"
        )

    def parse_update(self, update: Any) -> InboundMessage | None:  # pragma: no cover
        return None

    def parse_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> InboundMessage | None:
        """Parse a raw Events API body into an :class:`InboundMessage` (envelope-aware)."""
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(envelope, dict):
            return None
        return self._parse_event(envelope)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _sender_tokens(sender_id: str) -> list[str]:
    """Split a ``team:<t>:channel:<c>:user:<u>`` sender id into its allowlist tokens."""
    out: list[str] = []
    for field in ("team", "channel", "user"):
        marker = f"{field}:"
        start = sender_id.find(marker)
        if start == -1:
            continue
        rest = sender_id[start + len(marker) :]
        # value runs until the next "<field>:" marker
        end = len(rest)
        for nxt in ("team:", "channel:", "user:"):
            pos = rest.find(nxt)
            if pos != -1:
                end = min(end, pos)
        value = rest[:end].strip().rstrip(":")
        if value:
            out.append(f"{field}:{value}")
    return out


# ----------------------------------------------------- optional socket mode listener
class SlackSocketModeListener:
    """OPTIONAL path for receiving events over a WebSocket (Socket Mode).

    A deployment that cannot expose a public Events API request URL can instead connect
    OUTBOUND to Slack over Socket Mode, which needs an app-level (xapp-) token and the
    ``slack_sdk`` library — therefore an OPTIONAL dependency: :meth:`available` reports
    whether it can run, and the connector layer skips it cleanly when the extra (or the
    tokens) is absent.

    The same security spine applies: the tokens come from the secrets layer, and the
    ``allowed_*`` allowlists (default-deny) gate every received event before the agent runs;
    Socket Mode events are delivered over an authenticated socket (no per-message signature
    to verify). This surface is **live-pending** — it cannot be exercised offline (it needs
    a live socket), so it carries no offline contract test beyond capability detection; the
    fully-tested inbound surface is :class:`SlackEventsConnector`.
    """

    def __init__(
        self,
        handler: MessageHandler,
        *,
        poster: Callable[[str, str, str | None], Any] | None = None,
        allowed_team_ids: Collection[str] | None = None,
        allowed_channel_ids: Collection[str] | None = None,
        allowed_user_ids: Collection[str] | None = None,
    ) -> None:
        self._handler = handler
        self._poster = poster
        self._allowed_teams = {str(t) for t in (allowed_team_ids or [])}
        self._allowed_channels = {str(c) for c in (allowed_channel_ids or [])}
        self._allowed_users = {str(u) for u in (allowed_user_ids or [])}

    @staticmethod
    def capability() -> Capability:
        """Available only with the ``slack_sdk`` library AND both Slack tokens present."""
        return capability(
            requires_modules=("slack_sdk",),
            requires_secrets=(SLACK_APP_TOKEN_SECRET, SLACK_BOT_TOKEN_SECRET),
        )

    @staticmethod
    def available() -> bool:
        """Convenience boolean over :meth:`capability`."""
        return SlackSocketModeListener.capability().ok

    def is_allowed(self, *, team_id: str, channel_id: str, user_id: str) -> bool:
        """Default-deny gate over the event's team/channel/user.

        With all three allowlists empty this denies (the safe default for a bot that can
        see every message in every channel it is in). A non-empty allowlist admits when the
        team OR the channel OR the user matches.
        """
        if not (self._allowed_teams or self._allowed_channels or self._allowed_users):
            return False
        return (
            str(team_id) in self._allowed_teams
            or str(channel_id) in self._allowed_channels
            or str(user_id) in self._allowed_users
        )

    def run(self) -> None:  # pragma: no cover - needs a live socket (live-pending)
        """Connect over Socket Mode and run the agent for each allowed message.

        Requires the ``slack_sdk`` extra and a live app-level + bot token; cannot run
        offline, so it is excluded from the offline test suite.
        """
        cap = self.capability()
        cap.require()
        import asyncio
        from typing import cast

        import slack_sdk.socket_mode  # type: ignore[import-not-found]
        import slack_sdk.socket_mode.response  # type: ignore[import-not-found]
        import slack_sdk.web  # type: ignore[import-not-found]

        from himmy.config.secrets import get_secret

        socket_mode_client = slack_sdk.socket_mode.SocketModeClient
        socket_mode_response = slack_sdk.socket_mode.response.SocketModeResponse
        web_client = slack_sdk.web.WebClient

        app_token = get_secret(SLACK_APP_TOKEN_SECRET) or ""
        bot_token = get_secret(SLACK_BOT_TOKEN_SECRET) or ""
        client = socket_mode_client(
            app_token=app_token, web_client=web_client(token=bot_token)
        )
        listener = self

        def _on_request(socket_client: Any, req: Any) -> None:
            # Always ack first so Slack does not redeliver.
            socket_client.send_socket_mode_response(
                socket_mode_response(envelope_id=req.envelope_id)
            )
            if req.type != "events_api":
                return
            event = (req.payload or {}).get("event") or {}
            if event.get("type") not in ("message", "app_mention"):
                return
            if event.get("bot_id") or event.get("subtype"):
                return  # loop guard / non-user message
            team_id = str((req.payload or {}).get("team_id") or "")
            channel_id = str(event.get("channel") or "")
            user_id = str(event.get("user") or "")
            text = str(event.get("text") or "")
            if not (user_id and text):
                return
            if not listener.is_allowed(
                team_id=team_id, channel_id=channel_id, user_id=user_id
            ):
                return
            reply: str = asyncio.run(
                cast(
                    "Any",
                    listener._handler(
                        f"team:{team_id}:channel:{channel_id}:user:{user_id}", text
                    ),
                )
            )
            if reply and listener._poster is not None:
                thread_ts = str(event.get("thread_ts") or event.get("ts") or "") or None
                listener._poster(channel_id, reply, thread_ts)

        client.socket_mode_request_listeners.append(_on_request)  # type: ignore[attr-defined]
        client.connect()
        from threading import Event

        Event().wait()


# ----------------------------------------------------- registry self-registration
def _make_outbound() -> SlackOutboundConnector:
    """Build the default outbound connector from the secrets/allowlist env.

    Reads the channel allowlist + default channel from the environment so a deployment that
    only sets ``HIMMY_SLACK_BOT_TOKEN`` gets a connector that is capability-OK but refuses
    every post (default-deny) until a channel is allow-listed.
    """
    import os

    allow = [
        c.strip()
        for c in (os.environ.get("HIMMY_SLACK_ALLOWED_CHANNELS") or "").split(",")
        if c.strip()
    ]
    return SlackOutboundConnector(
        allowed_channel_ids=allow or None,
        default_channel_id=os.environ.get("HIMMY_SLACK_DEFAULT_CHANNEL"),
    )


def register_slack_connectors() -> None:
    """Register the Slack outbound connector on the process-wide registry.

    The inbound events connector is NOT auto-registered: it needs an agent ``handler`` (the
    registry stores zero-arg factories), so the BFF/CLI wires it explicitly with the agent
    turn. Safe to call more than once.
    """
    from himmy.connectors.sdk import register_connector

    register_connector("slack", _make_outbound)


__all__ = [
    "SLACK_API_ROOT",
    "SLACK_API_HOST",
    "SLACK_BOT_TOKEN_SECRET",
    "SLACK_SIGNING_SECRET",
    "SLACK_APP_TOKEN_SECRET",
    "SlackClient",
    "SlackOutboundConnector",
    "SlackEventsConnector",
    "SlackSocketModeListener",
    "verify_slack_signature",
    "register_slack_connectors",
]
