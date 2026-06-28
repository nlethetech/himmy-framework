"""Discord connector on the SDK: a slash-command bot + a post-message tool.

Two surfaces over Discord's public API, both inheriting the connector SDK's security
posture (default-deny allowlist, secrets-layer credentials, SSRF-guarded egress,
idempotent side effects, secret-safe errors, structured audit events):

* :class:`DiscordInteractionsConnector` — the **inbound** side. We use Discord's HTTP
  **Interactions** endpoint (a signed webhook) rather than a persistent gateway
  WebSocket: it is stateless, needs no always-on socket, and Discord cryptographically
  signs every delivery, so it is the cleaner fit for a command bot AND for the SDK's
  webhook gate. Each interaction is signed with **Ed25519** over
  ``timestamp + raw_body`` (headers ``X-Signature-Ed25519`` / ``X-Signature-Timestamp``);
  we verify it on the RAW body before parsing (overriding the SDK's HMAC default), then
  apply a **guild/channel/user allowlist (default-deny)** before the agent ever runs.
  A ``PING`` (Discord's endpoint health check) is answered with a ``PONG`` without
  touching the allowlist or the agent; a slash command runs the agent and replies inline
  with a ``CHANNEL_MESSAGE_WITH_SOURCE`` response.

  Receiving *arbitrary* (non-command) channel messages is a different capability: Discord
  does not webhook those, so it requires a gateway WebSocket connection with the
  privileged *Message Content* intent. :class:`DiscordGatewayListener` is the optional,
  capability-gated path for that (needs the ``discord`` extra + a live bot); the
  fully-offline-tested surface is the interactions webhook above.

* :class:`DiscordOutboundConnector` — the **outbound** side: a ``discord_post_message``
  tool so an agent can post to a channel. The bot token comes from the **secrets layer**
  (:data:`DISCORD_BOT_TOKEN_SECRET`), never the model; the call goes through the
  connector's **SSRF-guarded pinned transport**; the target channel must be on the
  **allowlist**; and the post is **idempotent** (deduped on a per-message key so a retry
  cannot double-post).

Nothing here imports ``cryptography`` or ``httpx`` at module load — capability detection
probes lazily — so importing this module is free in an air-gapped install.

LIVE-PENDING: the request shapes, auth header, and Ed25519 scheme are contract-tested
against a mocked transport (see ``tests/connectors/test_discord.py``); verifying against
the REAL Discord API needs a live bot token + a registered application/guild and cannot
run on this machine.
"""

from __future__ import annotations

import hashlib
import json
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

#: Discord REST API root (v10). The connector pins its egress to this host.
DISCORD_API_ROOT = "https://discord.com/api/v10"
#: The single API host the connector is allowed to reach (egress allow-list).
DISCORD_API_HOST = "discord.com"

#: Secret name for the bot token (resolved via the secrets layer, NEVER the model).
#: This is the NAME of the env/secret var, not a credential value (S105 false positive).
DISCORD_BOT_TOKEN_SECRET = "HIMMY_DISCORD_BOT_TOKEN"  # noqa: S105
#: Secret name for the application's Ed25519 public key (hex), used to verify that an
#: inbound interaction genuinely came from Discord. From the Developer Portal. (A NAME,
#: not a secret value — S105 false positive.)
DISCORD_PUBLIC_KEY_SECRET = "HIMMY_DISCORD_PUBLIC_KEY"  # noqa: S105

#: Discord interaction `type` values we handle.
_INTERACTION_PING = 1
_INTERACTION_APPLICATION_COMMAND = 2  # a slash command
#: Discord interaction `response` types.
_RESPONSE_PONG = 1
_RESPONSE_CHANNEL_MESSAGE = 4  # reply with a visible message

#: Discord caps a message at 2000 characters; we truncate defensively so an
#: over-long agent reply degrades to a clipped message instead of an API rejection.
_MAX_MESSAGE_LEN = 2000

#: Headers Discord signs each interaction with.
_SIG_HEADER = "X-Signature-Ed25519"
_TS_HEADER = "X-Signature-Timestamp"


def _truncate(text: str) -> str:
    """Clip ``text`` to Discord's per-message limit (keeps a trailing ellipsis)."""
    if len(text) <= _MAX_MESSAGE_LEN:
        return text
    return text[: _MAX_MESSAGE_LEN - 1] + "…"


def verify_ed25519_signature(
    *, public_key_hex: str, timestamp: str, body: bytes, signature_hex: str | None
) -> bool:
    """Constant-time Ed25519 check for a Discord interaction (default-deny).

    Discord signs ``timestamp || body`` with the application's Ed25519 key and sends the
    signature (hex) in ``X-Signature-Ed25519`` and the timestamp in
    ``X-Signature-Timestamp``. This recomputes the verification with the public key from
    the secrets layer and returns False — never raises — for a missing, malformed, or
    wrong signature (or an unparsable key), so the caller treats any failure exactly like
    a forgery and denies it. ``cryptography`` is imported lazily so this module loads in
    an air-gapped install without the dependency.
    """
    if not signature_hex or not public_key_hex or not timestamp:
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError:  # pragma: no cover - only without the `auth`/`encryption` extra
        # No crypto backend: we cannot verify, so we MUST deny (fail closed) rather than
        # wave the interaction through unverified.
        return False
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        signature = bytes.fromhex(signature_hex.strip())
    except (ValueError, TypeError):
        return False
    message = timestamp.encode("utf-8") + body
    try:
        key.verify(signature, message)
    except InvalidSignature:
        return False
    except Exception:  # noqa: BLE001 - any verification fault is a deny, never a crash
        return False
    return True


class DiscordClient:
    """A tiny REST client for the one outbound call: post a message to a channel.

    All HTTP rides on the SSRF-guarded **pinned transport** (the DNS-rebinding-safe one
    the rest of the connector layer uses) with redirects OFF, so a 3xx can never bounce
    an authenticated request to an internal host. The bot token is passed by the caller
    (resolved from the secrets layer) and sent as ``Authorization: Bot <token>``; it is
    never logged. ``transport`` is a test seam for injecting an ``httpx`` mock.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DISCORD_API_ROOT,
        timeout: float = 20.0,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = (DISCORD_API_HOST,),
        transport: Any = None,
    ) -> None:
        self._token = token
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._allow_private = allow_private_hosts
        self._allow_hosts = set(egress_allow_hosts) if egress_allow_hosts else None
        self._transport = transport

    def _auth_headers(self) -> dict[str, str]:
        # Discord bot auth uses the literal "Bot " scheme (not "Bearer").
        return {
            "Authorization": f"Bot {self._token}",
            "Content-Type": "application/json",
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
        self, channel_id: str, content: str, *, nonce: str | None = None
    ) -> dict[str, Any]:
        """POST ``content`` to ``channel_id`` (``POST /channels/{id}/messages``).

        ``nonce`` is Discord's own de-duplication token: passing a stable value AND
        ``enforce_nonce: true`` makes a repeated send idempotent on Discord's side too
        (belt-and-braces with the connector's idempotency store) — Discord only checks the
        nonce for uniqueness when ``enforce_nonce`` is set, returning the existing message
        on a duplicate. The nonce must be <=25 chars (Discord rejects a longer one with a
        50035 Invalid Form Body), so the caller passes a clipped value. Returns the created
        Message object. Raises a secret-safe :class:`ConnectorError` on a non-2xx so the
        token can never ride out in an error string.
        """
        url = f"{self._base}/channels/{channel_id}/messages"
        payload: dict[str, Any] = {"content": _truncate(content)}
        if nonce:
            # Discord caps the nonce at 25 chars; clip defensively so an over-long value
            # degrades instead of 400-ing. ``enforce_nonce`` makes Discord honour it for
            # de-dupe (without it the nonce is ignored for uniqueness).
            payload["nonce"] = nonce[:25]
            payload["enforce_nonce"] = True
        with self._client() as client:
            resp = client.post(url, json=payload, headers=self._auth_headers())
            if resp.status_code >= 400:
                # Surface ONLY the status + Discord's own (token-free) error code, never
                # the response body verbatim (which could echo a request header).
                code = ""
                try:
                    code = str((resp.json() or {}).get("code", ""))
                except Exception:  # noqa: BLE001 - non-JSON error body
                    code = ""
                raise ConnectorError(
                    f"discord post_message failed: HTTP {resp.status_code}"
                    + (f" (code {code})" if code else "")
                )
            return dict(resp.json())


# --------------------------------------------------------------- outbound tool
_POST_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "The message text to post (clipped to 2000 chars).",
        },
        "channel_id": {
            "type": "string",
            "description": (
                "Target Discord channel id. Must be on the connector's channel "
                "allowlist; defaults to the configured default channel."
            ),
        },
    },
    "required": ["content"],
    "additionalProperties": False,
}


class DiscordOutboundConnector(OutboundToolConnector):
    """Exposes ``discord_post_message`` — post to an allow-listed channel, idempotently.

    Security contract (all inherited from :class:`OutboundToolConnector` + enforced here):

    * the bot token is read from the **secrets layer** (:data:`DISCORD_BOT_TOKEN_SECRET`)
      at call time, never from a constructor arg the model could set, never logged;
    * the connector is **unavailable** (skipped cleanly by discovery / ``register_tools``)
      when the token is absent — capability-gated;
    * every request goes through the SSRF-guarded **pinned transport**, pinned to the
      Discord API host (egress allow-list);
    * the destination channel must be on the **channel allowlist** (default-deny: an empty
      allowlist with no configured default rejects every post);
    * the post is **idempotent** — deduped per ``(channel, content)`` key so a retry or a
      re-issued tool call posts exactly once.
    """

    name = "discord"
    required_secrets = (DISCORD_BOT_TOKEN_SECRET,)

    def __init__(
        self,
        *,
        allowed_channel_ids: Collection[str] | None = None,
        default_channel_id: str | None = None,
        requires_approval: bool = True,
        client_factory: Callable[[str | None], DiscordClient] | None = None,
        context: ConnectorContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wire the channel allowlist + default channel.

        ``allowed_channel_ids`` is the allowlist (compared as strings). With it empty and
        no ``default_channel_id``, the tool denies every post — the default-deny posture.
        ``requires_approval`` gates the tool behind human-in-the-loop approval by default
        (a side-effecting "act" tool). ``client_factory`` is the injectable seam tests use
        to supply a :class:`DiscordClient` over a mock transport.
        """
        # Default the egress allow-list to the Discord API host for defence in depth.
        kwargs.setdefault("egress_allow_hosts", (DISCORD_API_HOST,))
        # A sane per-connector self-throttle so a runaway agent can't flood the API.
        if context is None:
            context = ConnectorContext(rate_limiter=RateLimiter(rate=5, window=1.0))
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

    def _build_client(self) -> DiscordClient:
        token = self.secret(DISCORD_BOT_TOKEN_SECRET)  # raises (named) if absent
        if self._client_factory is not None:
            return self._client_factory(token)
        return DiscordClient(
            token or "",
            allow_private_hosts=self._allow_private,
            egress_allow_hosts=self._allow_hosts or (DISCORD_API_HOST,),
        )

    def register(self, registry: ToolRegistry) -> list[str]:
        async def discord_post_message(args: dict[str, Any]) -> dict[str, Any]:
            import asyncio

            content = str(args.get("content") or "")
            if not content:
                raise ToolSecurityError("discord_post_message requires `content`")
            channel_id = str(args.get("channel_id") or self._default_channel or "")
            if not channel_id:
                raise ToolSecurityError(
                    "discord_post_message needs a channel_id (or a default channel)"
                )
            if not self._channel_allowed(channel_id):
                self._audit(
                    "discord_post_message",
                    "deny",
                    actor={"subject": channel_id},
                    detail="channel not on allowlist",
                )
                raise ToolSecurityError(
                    f"channel {channel_id!r} is not on the Discord post allowlist"
                )
            self._throttle()
            client = self._build_client()
            # Idempotency over the (channel, content). A SHA-256 digest is the local store
            # key: it is collision-resistant (unlike PYTHONHASHSEED-salted, 32-bit-truncated
            # hash(), which could silently drop a distinct message as a duplicate). Discord's
            # `nonce` is capped at 25 chars, so the client sends a clipped slice of this same
            # digest for upstream de-dupe; the FULL digest is what we dedupe on locally.
            digest = hashlib.sha256(f"{channel_id}:{content}".encode()).hexdigest()

            def _send() -> dict[str, Any]:
                return client.post_message(channel_id, content, nonce=digest)

            try:
                message = await asyncio.to_thread(
                    self.call_idempotent, f"post:{digest}", _send
                )
            except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
                token = self.secret(DISCORD_BOT_TOKEN_SECRET, required=False) or ""
                self._audit(
                    "discord_post_message",
                    "deny",
                    actor={"subject": channel_id},
                    detail=safe_error(exc, secrets=[token]),
                )
                raise ConnectorError(
                    f"discord_post_message failed: {safe_error(exc, secrets=[token])}"
                ) from None
            self._audit(
                "discord_post_message",
                "allow",
                actor={"subject": channel_id},
                detail=f"posted {dict(redact_args(args))}",
            )
            return {
                "sent": True,
                "channel_id": channel_id,
                "message_id": str(message.get("id", "")),
            }

        register_local_tool(
            registry,
            name="discord_post_message",
            read_only=False,
            handler=discord_post_message,
            description="Post a message to an allow-listed Discord channel via the bot.",
            args_json_schema=_POST_SCHEMA,
            requires_approval=self._requires_approval,
            sensitive_arg_names=[],
            metadata={"connector": "discord"},
        )
        return ["discord_post_message"]


# --------------------------------------------------------- inbound interactions
class DiscordInteractionsConnector(InboundChannelConnector):
    """Discord slash-command bot over the signed HTTP **Interactions** webhook.

    Delivery is a webhook, so :meth:`handle_webhook` is the entry point. The security
    order is load-bearing:

    1. **Ed25519 signature** over the RAW ``timestamp + body`` is verified FIRST (we
       override the SDK's HMAC default with :meth:`verify_webhook`). A missing/forged
       signature is denied before the body is even parsed.
    2. A ``PING`` (Discord's periodic endpoint check) is answered with a ``PONG`` — it
       carries no sender and never reaches the allowlist or the agent.
    3. A slash command is gated by the **guild/channel/user allowlist (default-deny)**:
       the ``sender_id`` is the invoking ``guild:channel:user`` triple, and the allowlist
       may name a whole guild, a channel, or a specific user (any match admits). An empty
       allowlist denies everyone — a public interactions URL must name its senders.
    4. Only then does the agent run, and its reply is returned as a
       ``CHANNEL_MESSAGE_WITH_SOURCE`` interaction response (the inline reply Discord
       shows under the command).

    The public key (for step 1) comes from the **secrets layer**
    (:data:`DISCORD_PUBLIC_KEY_SECRET`); the connector is unavailable when it is absent.
    """

    name = "discord-interactions"

    def __init__(
        self,
        handler: MessageHandler,
        *,
        public_key: str | None = None,
        allowed_guild_ids: Collection[str] | None = None,
        allowed_channel_ids: Collection[str] | None = None,
        allowed_user_ids: Collection[str] | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        """Wire the agent ``handler`` plus the public key + guild/channel/user allowlist.

        ``public_key`` defaults to the secrets-layer value (:data:`DISCORD_PUBLIC_KEY_SECRET`)
        so production reads it from the configured backend; tests pass it explicitly. The
        three allowlists are unioned into the SDK's sender allowlist as ``guild:<id>`` /
        ``channel:<id>`` / ``user:<id>`` tokens, and a parsed interaction is admitted when
        ANY of its guild/channel/user appears — so an operator can allow a whole guild, a
        single channel, or a named user. All empty ⇒ default-deny (answer no one).
        """
        from himmy.config.secrets import get_secret

        self._public_key = (
            public_key
            if public_key is not None
            else (get_secret(DISCORD_PUBLIC_KEY_SECRET) or "")
        )
        allowed: set[str] = set()
        allowed.update(f"guild:{g}" for g in (allowed_guild_ids or []))
        allowed.update(f"channel:{c}" for c in (allowed_channel_ids or []))
        allowed.update(f"user:{u}" for u in (allowed_user_ids or []))
        super().__init__(
            handler,
            allowed_senders=allowed,
            # default-deny: an empty allowlist denies everyone (webhook-safe posture)
            allow_empty_allowlist=False,
            context=context,
        )

    def capability(self) -> Capability:
        """Available only when the application public key is configured."""
        if not self._public_key:
            return Capability(
                ok=False, reason=f"missing credential {DISCORD_PUBLIC_KEY_SECRET!r}"
            )
        return Capability(ok=True)

    # -- security gate: Discord's Ed25519, not the SDK's HMAC default --------
    def verify_webhook(self, body: bytes, headers: Mapping[str, str]) -> bool:
        """Verify the interaction's Ed25519 signature over ``timestamp + body``."""
        return verify_ed25519_signature(
            public_key_hex=self._public_key,
            timestamp=_header(headers, _TS_HEADER) or "",
            body=body,
            signature_hex=_header(headers, _SIG_HEADER),
        )

    def is_allowed(self, sender_id: str) -> bool:
        """Admit when the interaction's guild OR channel OR user is allow-listed.

        ``sender_id`` is the ``guild:<g>:channel:<c>:user:<u>`` triple parsed from the
        interaction; we split it back out and admit if any component token is on the
        allowlist. With an empty allowlist this is default-deny (False).
        """
        if not self._allowed:
            return self._allow_empty_allowlist
        for token in _sender_tokens(sender_id):
            if token in self._allowed:
                return True
        return False

    # -- webhook delivery (overrides the SDK to handle PING + inline reply) --
    async def handle_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """Verify, answer a PING, allowlist-gate a command, run the agent, reply inline.

        Returns the JSON interaction response Discord expects (``{"type": 1}`` for a
        PONG, ``{"type": 4, "data": {...}}`` for a command reply) plus an internal
        ``ok``/``handled`` flag for the caller. A bad signature or unlisted sender returns
        ``ok=False`` and is audited as a deny without running the agent.
        """
        if not self.verify_webhook(body, headers):
            self._audit("webhook", "deny", detail="invalid interaction signature")
            return {"ok": False, "reason": "invalid signature"}
        try:
            interaction = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "reason": "malformed body"}
        if not isinstance(interaction, dict):
            return {"ok": False, "reason": "malformed body"}

        itype = interaction.get("type")
        if itype == _INTERACTION_PING:
            # Discord's endpoint health check — answer immediately, no agent, no gate.
            self._audit("webhook", "allow", detail="ping")
            return {"ok": True, "handled": False, "response": {"type": _RESPONSE_PONG}}
        if itype != _INTERACTION_APPLICATION_COMMAND:
            # We only act on slash commands; anything else is acknowledged but ignored.
            return {"ok": True, "handled": False}

        msg = self.parse_webhook(body, headers)
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
        response = {
            "type": _RESPONSE_CHANNEL_MESSAGE,
            "data": {"content": _truncate(reply or "")},
        }
        return {"ok": True, "handled": bool(reply), "response": response}

    def parse_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> InboundMessage | None:
        """Normalize a slash-command interaction into an :class:`InboundMessage`.

        The text is the command name plus its option values (``/ask question:hello`` →
        ``"ask question=hello"``); the ``sender_id`` encodes the guild/channel/user so the
        allowlist can gate on any of them.
        """
        try:
            interaction = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(interaction, dict):
            return None
        data = interaction.get("data") or {}
        name = str(data.get("name") or "")
        if not name:
            return None
        # Flatten the command options into "key=value" tokens for the agent.
        parts = [name]
        for opt in data.get("options") or []:
            if isinstance(opt, dict) and "value" in opt:
                parts.append(f"{opt.get('name')}={opt.get('value')}")
        text = " ".join(parts).strip()

        guild_id = str(interaction.get("guild_id") or "")
        channel_id = str(interaction.get("channel_id") or "")
        member = interaction.get("member") or {}
        user = (member.get("user") if isinstance(member, dict) else None) or (
            interaction.get("user") or {}
        )
        user_id = str(user.get("id") or "") if isinstance(user, dict) else ""
        sender_id = f"guild:{guild_id}:channel:{channel_id}:user:{user_id}"
        return InboundMessage(sender_id=sender_id, text=text, raw=interaction)

    # The interactions webhook is push-only — there is no poll loop to implement.
    async def fetch_updates(self, offset: int | None) -> list[Any]:
        raise ConnectorError(
            "discord-interactions is webhook-only; use handle_webhook, not polling"
        )

    def parse_update(self, update: Any) -> InboundMessage | None:  # pragma: no cover
        return None

    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        # The reply rides back inline in the interaction RESPONSE (handled in
        # handle_webhook), so there is no separate outbound send here.
        return None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _sender_tokens(sender_id: str) -> list[str]:
    """Split a ``guild:<g>:channel:<c>:user:<u>`` sender id into its allowlist tokens."""
    out: list[str] = []
    for field in ("guild", "channel", "user"):
        marker = f"{field}:"
        start = sender_id.find(marker)
        if start == -1:
            continue
        rest = sender_id[start + len(marker) :]
        # value runs until the next "<field>:" marker
        end = len(rest)
        for nxt in ("guild:", "channel:", "user:"):
            pos = rest.find(nxt)
            if pos != -1:
                end = min(end, pos)
        value = rest[:end].strip().rstrip(":")
        if value:
            out.append(f"{field}:{value}")
    return out


# ----------------------------------------------------- optional gateway listener
class DiscordGatewayListener:
    """OPTIONAL path for receiving arbitrary channel MESSAGES (needs the gateway).

    Discord does not deliver non-command channel messages via webhook, so reading them
    requires a persistent gateway WebSocket connection with the privileged *Message
    Content* intent. This thin wrapper drives the ``discord`` (discord.py) library, which
    is therefore an OPTIONAL dependency: :meth:`available` reports whether it can run, and
    the connector layer skips it cleanly when the extra (or the bot token) is absent.

    The same security spine applies: the bot token comes from the secrets layer, and the
    ``allowed_*`` allowlists (default-deny) gate every received message before the agent
    runs. This surface is **live-pending** — it cannot be exercised offline (it needs a
    live gateway connection), so it carries no offline contract test beyond capability
    detection; the fully-tested inbound surface is :class:`DiscordInteractionsConnector`.
    """

    def __init__(
        self,
        handler: MessageHandler,
        *,
        allowed_guild_ids: Collection[str] | None = None,
        allowed_channel_ids: Collection[str] | None = None,
        allowed_user_ids: Collection[str] | None = None,
    ) -> None:
        self._handler = handler
        self._allowed_guilds = {str(g) for g in (allowed_guild_ids or [])}
        self._allowed_channels = {str(c) for c in (allowed_channel_ids or [])}
        self._allowed_users = {str(u) for u in (allowed_user_ids or [])}

    @staticmethod
    def capability() -> Capability:
        """Available only with the ``discord`` library AND the bot token present."""
        return capability(
            requires_modules=("discord",),
            requires_secrets=(DISCORD_BOT_TOKEN_SECRET,),
        )

    @staticmethod
    def available() -> bool:
        """Convenience boolean over :meth:`capability`."""
        return DiscordGatewayListener.capability().ok

    def is_allowed(self, *, guild_id: str, channel_id: str, user_id: str) -> bool:
        """Default-deny gate over the message's guild/channel/user.

        With all three allowlists empty this denies (the safe default for a bot that can
        see every message in every server it is in). A non-empty allowlist admits when the
        guild OR the channel OR the user matches.
        """
        if not (self._allowed_guilds or self._allowed_channels or self._allowed_users):
            return False
        return (
            str(guild_id) in self._allowed_guilds
            or str(channel_id) in self._allowed_channels
            or str(user_id) in self._allowed_users
        )

    def run(self) -> None:  # pragma: no cover - needs a live gateway (live-pending)
        """Connect to the gateway and run the agent for each allowed message.

        Requires the ``discord`` extra and a live bot token + Message Content intent;
        cannot run offline, so it is excluded from the offline test suite.
        """
        cap = self.capability()
        cap.require()
        import discord  # type: ignore[import-not-found]

        from himmy.config.secrets import get_secret

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        listener = self

        @client.event  # type: ignore[untyped-decorator]
        async def on_message(message: Any) -> None:
            if message.author.bot:
                return  # never act on another bot (incl. ourselves) — loop guard
            guild_id = str(getattr(message.guild, "id", "") or "")
            channel_id = str(getattr(message.channel, "id", "") or "")
            user_id = str(getattr(message.author, "id", "") or "")
            if not listener.is_allowed(
                guild_id=guild_id, channel_id=channel_id, user_id=user_id
            ):
                return
            reply = await listener._handler(
                f"guild:{guild_id}:channel:{channel_id}:user:{user_id}",
                str(message.content),
            )
            if reply:
                await message.channel.send(_truncate(reply))

        token = get_secret(DISCORD_BOT_TOKEN_SECRET) or ""
        client.run(token)


# ----------------------------------------------------- registry self-registration
def _make_outbound() -> DiscordOutboundConnector:
    """Build the default outbound connector from the secrets/allowlist env.

    Reads the channel allowlist + default channel from the environment so a deployment
    that only sets ``HIMMY_DISCORD_BOT_TOKEN`` gets a connector that is capability-OK but
    refuses every post (default-deny) until a channel is allow-listed.
    """
    import os

    allow = [
        c.strip()
        for c in (os.environ.get("HIMMY_DISCORD_ALLOWED_CHANNELS") or "").split(",")
        if c.strip()
    ]
    return DiscordOutboundConnector(
        allowed_channel_ids=allow or None,
        default_channel_id=os.environ.get("HIMMY_DISCORD_DEFAULT_CHANNEL"),
    )


def register_discord_connectors() -> None:
    """Register the Discord outbound connector on the process-wide registry.

    The inbound interactions connector is NOT auto-registered: it needs an agent
    ``handler`` (the registry stores zero-arg factories), so the BFF/CLI wires it
    explicitly with the agent turn. Safe to call more than once.
    """
    from himmy.connectors.sdk import register_connector

    register_connector("discord", _make_outbound)


__all__ = [
    "DISCORD_API_ROOT",
    "DISCORD_API_HOST",
    "DISCORD_BOT_TOKEN_SECRET",
    "DISCORD_PUBLIC_KEY_SECRET",
    "DiscordClient",
    "DiscordOutboundConnector",
    "DiscordInteractionsConnector",
    "DiscordGatewayListener",
    "verify_ed25519_signature",
    "register_discord_connectors",
]
