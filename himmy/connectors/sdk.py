"""Connector SDK: the one secure pattern every Himmy connector follows.

A *connector* bridges a Himmy agent to an external system. There are two kinds and
this module gives each a base class so the long tail of community connectors inherits
the security posture for free instead of re-deriving it (and re-introducing the same
SSRF / allowlist / secret-leak holes) each time:

* :class:`InboundChannelConnector` — receives a message FROM an external channel,
  runs the agent, and replies. It must authenticate the sender against an **allowlist
  (default-deny)** and, for webhook delivery, verify an **HMAC signature** before the
  message ever reaches the agent. Polling reuses the Telegram poll/offset shape; the
  same connector serves webhook delivery through :meth:`handle_webhook`.

* :class:`OutboundToolConnector` — exposes typed tools to the agent. Credentials come
  from the **secrets layer** (:func:`himmy.config.secrets.get_secret`), every outbound
  HTTP call goes through the **SSRF-guarded** fetcher / :func:`guard_url`, and
  side-effecting calls are **idempotent** (an idempotency key dedupes a blind retry).

Cross-cutting machinery both kinds share:

* a :class:`ConnectorRegistry` for discovery (register + look up by name/kind);
* :func:`capability` / ``available()`` so a connector whose dependency or credential is
  absent is *skipped cleanly* rather than exploding at import or call time;
* a per-connector token-bucket :class:`RateLimiter` and bounded :class:`RetryPolicy`;
* uniform error handling that **never leaks a secret** (:func:`safe_error`) and a
  structured :class:`SecurityEvent` per connector action, recorded on the existing
  entity-backed audit spine.

Nothing here imports a network SDK at module load — capability detection probes lazily
— so importing :mod:`himmy.connectors.sdk` is free even in an air-gapped install.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from himmy.config.secrets import get_secret
from himmy.connectors.fetcher import Fetcher, HttpxFetcher
from himmy.core.errors import HimmyError
from himmy.services.audit.models import SecurityEvent
from himmy.services.tools.security import ToolSecurityError, redact_mapping
from himmy.toolkit._net import build_pinned_transport, guard_url

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.tools.registry import ToolRegistry


class ConnectorError(HimmyError):
    """A connector failed in a way the caller can surface (message is safe to show).

    Raised by the SDK for connector-level failures (unknown connector, missing
    credential, denied sender, idempotency conflict). The message is constructed to
    never carry a secret value — only NAMES of missing/required things — so it is safe
    to log and return to a model.
    """


class ConnectorKind(str, Enum):
    """The two connector shapes the SDK supports."""

    INBOUND = "inbound"  # receive from a channel -> run agent -> reply
    OUTBOUND = "outbound"  # expose typed tools the agent can call


# --------------------------------------------------------------------------- audit
#: EntityRecord kind for a connector action on the audit spine. Distinct from
#: ``security_event`` (auth/authz) but recorded through the same tamper-evident log.
CONNECTOR_EVENT_TYPE = "connector_action"


def _record_event(
    audit: AuditSink | None,
    *,
    connector: str,
    kind: ConnectorKind,
    action: str,
    outcome: str,
    actor: Mapping[str, Any] | None = None,
    detail: str = "",
) -> None:
    """Record one structured connector action on the audit spine (no-op if unwired).

    ``outcome`` is ``allow`` (the action ran) or ``deny`` (it was refused — a denied
    sender, a bad signature, a blocked URL). ``detail`` MUST already be secret-safe:
    callers pass :func:`safe_error` output, never a raw exception that might echo a
    credential. Recording is best-effort: an audit-store hiccup never fails the action
    that is being audited.
    """
    if audit is None:
        return
    event = SecurityEvent(
        event_type=CONNECTOR_EVENT_TYPE,
        outcome=outcome,
        actor=dict(actor or {"subject": connector}),
        resource=connector,
        action=action,
        method=kind.value,
        detail=detail[:500],
    )
    try:
        audit.record(event)
    except Exception:  # noqa: BLE001 - auditing must never break the audited path
        pass


class AuditSink:
    """Adapts :class:`SecurityAuditLog` (or any ``.record(SecurityEvent)``) for the SDK.

    The SDK only needs ``record``; this thin Protocol-shaped wrapper lets a connector
    accept the existing :class:`~himmy.services.audit.log.SecurityAuditLog`, a raw
    :class:`EntityRegistry` (wrapped via :meth:`from_registry`), or a test double.
    """

    def __init__(self, record: Callable[[SecurityEvent], Any]) -> None:
        self._record = record

    def record(self, event: SecurityEvent) -> None:
        self._record(event)

    @classmethod
    def from_registry(cls, registry: EntityRegistryProtocol) -> AuditSink:
        """Wrap an entity registry in a :class:`SecurityAuditLog` sink."""
        from himmy.services.audit.log import SecurityAuditLog

        return cls(SecurityAuditLog(registry).record)


def redact_args(
    args: Mapping[str, Any], *, extra_keys: Collection[str] = ()
) -> dict[str, Any]:
    """Redact secret-looking values in an arg/header mapping for safe auditing.

    A thin wrapper over the kernel's :func:`redact_mapping` so a connector can put a
    summary of a tool's arguments into an audit ``detail`` without risking that an
    arg named ``token`` / ``api_key`` / ``authorization`` (or an explicitly-named
    ``extra_keys`` field) leaks its value onto the audit trail.
    """
    return redact_mapping(dict(args), extra_keys=tuple(extra_keys))


def safe_error(exc: BaseException, *, secrets: Collection[str] = ()) -> str:
    """A loggable, model-safe string for ``exc`` that cannot leak a secret value.

    Two-layer defense. First, only the exception TYPE and (for the SDK's own
    :class:`ConnectorError` / :class:`ToolSecurityError`, which are authored to be
    safe) its message are surfaced; an arbitrary third-party exception is reduced to
    its class name so a credential a library echoed into ``str(exc)`` never escapes.
    Second, any literal secret value passed in ``secrets`` is scrubbed from whatever
    text remains, as a belt-and-braces guard for the safe-by-type messages too.
    """
    if isinstance(exc, (ConnectorError, ToolSecurityError)):
        text = f"{type(exc).__name__}: {exc}"
    else:
        # An unknown exception's message may embed a header/token the library logged;
        # surface only the type so nothing sensitive rides along.
        text = type(exc).__name__
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***REDACTED***")
    return text


# ----------------------------------------------------------------- rate limiting
class RateLimiter:
    """A thread-safe token-bucket limiter, one bucket per (connector) key.

    Mirrors :class:`himmy.api.ratelimit.TokenBucketRateLimiter` but as a plain
    library object (no FastAPI dependency): refill ``rate`` tokens per ``window``
    seconds up to ``capacity`` (the burst). :meth:`check` raises
    :class:`ConnectorError` when the bucket is empty — the connector's own
    self-throttle so a misbehaving agent (or a webhook flood) can't hammer an
    upstream and trip *its* rate limit / ban the credential.
    """

    def __init__(
        self,
        *,
        rate: float,
        capacity: float | None = None,
        window: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if rate <= 0 or window <= 0:
            raise ValueError("rate and window must be positive")
        self._refill_per_sec = rate / window
        self._capacity = capacity if capacity is not None else rate
        if self._capacity <= 0:
            raise ValueError("capacity must be positive")
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str = "default") -> None:
        """Consume one token for ``key``; raise :class:`ConnectorError` when empty."""
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(
                self._capacity, tokens + (now - last) * self._refill_per_sec
            )
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                retry = max(1, int((1.0 - tokens) / self._refill_per_sec + 0.999))
                raise ConnectorError(
                    f"rate limit exceeded for {key!r}; retry in ~{retry}s"
                )
            self._buckets[key] = (tokens - 1.0, now)


# -------------------------------------------------------------------- retries
@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy for a connector's outbound calls.

    Defaults to NO retries on a side-effecting call — a blind retry of a POST that
    timed out can double-send. A call is retried only when it (a) declares itself
    ``idempotent`` (safe to repeat) or (b) raises an exception class the policy
    marks retryable. Reuses the tool-retry safety rule: never retry a non-idempotent
    mutation just because it timed out.
    """

    attempts: int = 1  # total attempts incl. the first; 1 = no retry
    backoff_base: float = 0.2
    backoff_max: float = 5.0
    retry_on: tuple[type[BaseException], ...] = ()

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")

    def run(
        self,
        call: Callable[[], Any],
        *,
        idempotent: bool,
        sleep: Callable[[float], None] | None = None,
    ) -> Any:
        """Invoke ``call``, retrying ONLY when ``idempotent`` and the error is retryable.

        A non-idempotent call is attempted exactly once: if it raises, the caller can't
        know whether the side effect landed, so a retry could duplicate it. An
        idempotent call retries up to ``attempts`` times on a ``retry_on`` exception
        with exponential backoff.
        """
        nap = sleep or time.sleep
        max_attempts = self.attempts if idempotent else 1
        last: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                return call()
            except self.retry_on as exc:  # type: ignore[misc]
                last = exc
                if attempt + 1 >= max_attempts:
                    break
                nap(min(self.backoff_base * (2.0**attempt), self.backoff_max))
        assert last is not None  # the loop only breaks after catching
        raise last


# ----------------------------------------------------------- idempotency tracking
class IdempotencyStore:
    """Remembers idempotency keys so a side-effecting call runs at most once.

    A connector wraps a mutating call with :meth:`run_once`: the first time a key is
    seen the call runs and its result is cached; a repeat of the SAME key returns the
    cached result WITHOUT re-invoking the call. This makes a retry (or a duplicate
    webhook delivery) safe — the outbound effect happens once. In-process by default
    (fits a single connector instance); inject a shared store for multi-replica.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done: dict[str, Any] = {}
        self._inflight: set[str] = set()

    def run_once(self, key: str, call: Callable[[], Any]) -> Any:
        """Run ``call`` once for ``key``; return the cached result on a repeat.

        Raises :class:`ConnectorError` if ``key`` is already in flight on another
        thread (a concurrent duplicate) so the two callers can't both fire the effect.
        """
        with self._lock:
            if key in self._done:
                return self._done[key]
            if key in self._inflight:
                raise ConnectorError(
                    f"idempotency key {key!r} is already in flight (duplicate request)"
                )
            self._inflight.add(key)
        try:
            result = call()
        finally:
            with self._lock:
                self._inflight.discard(key)
        with self._lock:
            self._done[key] = result
        return result

    async def run_once_async(
        self, key: str, call: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Async ``run_once``: run the async ``call`` once per ``key``, mark-after-success.

        The async analog of :meth:`run_once` used by the (async) webhook handler so the
        dedup MARK lands only AFTER the agent turn succeeds — never the mark-before-run
        order that, made durable, would turn a crash into a permanently-dropped delivery. A
        concurrent duplicate (key already in flight) raises :class:`ConnectorError`; a key
        that already completed returns the cached result without re-running; a failure
        releases the in-flight reservation so a redelivery re-runs. The durable
        :class:`~himmy.services.storage.trigger_dedup.DurableIdempotencyStore` overrides this
        to persist the same protocol across restarts.
        """
        with self._lock:
            if key in self._done:
                return self._done[key]
            if key in self._inflight:
                raise ConnectorError(
                    f"idempotency key {key!r} is already in flight (duplicate request)"
                )
            self._inflight.add(key)
        try:
            result = await call()
        except BaseException:
            with self._lock:
                self._inflight.discard(key)
            raise
        with self._lock:
            self._inflight.discard(key)
            self._done[key] = result
        return result

    def seen(self, key: str) -> bool:
        """Whether ``key`` has already completed (used by webhook de-duplication)."""
        with self._lock:
            return key in self._done

    async def seen_async(self, key: str) -> bool:
        """Async ``seen`` (the durable store overrides to hit storage)."""
        return self.seen(key)


# ------------------------------------------------------------- capability detection
@dataclass(frozen=True)
class Capability:
    """The result of probing whether a connector can actually run.

    ``ok`` is True only when every required dependency imports AND every required
    secret resolves. ``reason`` explains a miss in plain words (never echoing a secret
    value — only its NAME), so discovery can *skip a connector cleanly* and tell the
    operator why instead of crashing.
    """

    ok: bool
    reason: str = ""

    def require(self) -> None:
        """Raise :class:`ConnectorError` when this capability is unavailable."""
        if not self.ok:
            raise ConnectorError(self.reason or "connector is unavailable")


def capability(
    *, requires_modules: Collection[str] = (), requires_secrets: Collection[str] = ()
) -> Capability:
    """Probe importability + secret presence; return a :class:`Capability`.

    Each module in ``requires_modules`` must import (a missing optional extra makes
    the connector unavailable rather than crashing the registry); each name in
    ``requires_secrets`` must resolve through :func:`get_secret` to a non-empty value.
    Only the NAME of a missing secret appears in ``reason`` — never its value.
    """
    import importlib.util

    for module in requires_modules:
        if importlib.util.find_spec(module) is None:
            return Capability(
                ok=False, reason=f"missing dependency {module!r} (install the extra)"
            )
    for name in requires_secrets:
        if not get_secret(name):
            return Capability(ok=False, reason=f"missing credential {name!r}")
    return Capability(ok=True)


# --------------------------------------------------------------------- signatures
def verify_hmac_signature(
    *,
    secret: str,
    body: bytes,
    provided: str | None,
    algorithm: str = "sha256",
    prefix: str = "",
) -> bool:
    """Constant-time HMAC check for an inbound webhook body (default-deny).

    Computes ``HMAC(secret, body)`` with ``algorithm`` and compares it to the
    ``provided`` header value using :func:`hmac.compare_digest` (timing-safe).
    ``prefix`` strips a scheme marker such as ``"sha256="`` (GitHub) before comparing.
    Returns False — never raises — for a missing/empty/malformed signature so a
    caller can treat the absence of a valid signature exactly like a wrong one (deny).
    """
    if not provided or not secret:
        return False
    candidate = provided.strip()
    if prefix and candidate.startswith(prefix):
        candidate = candidate[len(prefix) :]
    try:
        expected = hmac.new(
            secret.encode("utf-8"), body, getattr(hashlib, algorithm)
        ).hexdigest()
    except (AttributeError, TypeError):
        return False
    return hmac.compare_digest(expected, candidate)


# ------------------------------------------------------------- base connector
@dataclass
class ConnectorContext:
    """Shared services every connector is constructed with.

    Bundles the audit sink, the rate limiter, and the retry policy so a subclass's
    ``__init__`` stays small and every connector audits / throttles consistently. All
    fields are optional: a connector built with the bare default audits nothing,
    throttles at a sane default rate, and does not retry — the safe baseline.
    """

    audit: AuditSink | None = None
    rate_limiter: RateLimiter | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)


class Connector(ABC):
    """Base for every connector: identity, capability, audit + rate-limit plumbing."""

    #: Stable, unique connector name (the registry key). Subclasses MUST set this.
    name: str = ""
    #: Which shape this connector is — drives discovery filtering.
    kind: ConnectorKind

    def __init__(self, *, context: ConnectorContext | None = None) -> None:
        if not self.name:
            raise ConnectorError(
                f"{type(self).__name__} must set a non-empty `name` class attribute"
            )
        self._ctx = context or ConnectorContext()

    # -- capability ---------------------------------------------------------
    def capability(self) -> Capability:
        """Whether this connector can run now (deps + credentials present).

        The default reports available; override to declare required modules/secrets via
        :func:`capability`. Discovery calls this to *skip cleanly* when a dependency or
        credential is absent.
        """
        return Capability(ok=True)

    def available(self) -> bool:
        """Convenience boolean over :meth:`capability`."""
        return self.capability().ok

    # -- shared plumbing ----------------------------------------------------
    def _throttle(self, key: str | None = None) -> None:
        """Consume a rate-limit token (no-op when no limiter is configured)."""
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.check(key or self.name)

    def _audit(
        self,
        action: str,
        outcome: str,
        *,
        actor: Mapping[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        """Record a structured connector action on the audit spine."""
        _record_event(
            self._ctx.audit,
            connector=self.name,
            kind=self.kind,
            action=action,
            outcome=outcome,
            actor=actor,
            detail=detail,
        )


# --------------------------------------------------------- inbound channel base
#: ``handler(sender_id, text) -> reply`` — the agent turn an inbound connector runs.
MessageHandler = Callable[[str, str], Awaitable[str]]


@dataclass(frozen=True)
class InboundMessage:
    """One normalized inbound message a channel hands the agent."""

    sender_id: str  # the channel-native id used for the allowlist check
    text: str
    raw: Mapping[str, Any] = field(default_factory=dict)


class InboundChannelConnector(Connector):
    """Base for "receive a message -> run the agent -> reply" channel connectors.

    Two delivery modes share ONE security gate (:meth:`_authorize`):

    * **Polling** — :meth:`poll_once` pulls a batch via :meth:`fetch_updates`, parses
      each through :meth:`parse_update`, and runs the handler for every ALLOWED sender,
      advancing an offset (the Telegram poll/offset shape).
    * **Webhook** — :meth:`handle_webhook` verifies the HMAC signature over the raw
      body FIRST, then applies the same sender allowlist, then runs the handler.

    Default-deny is the spine: an empty allowlist means *answer no one* on the webhook
    path (a public URL must name its senders) — the opposite of a private poll bot,
    which may set ``allow_empty_allowlist`` to answer everyone it is privately given.
    """

    kind = ConnectorKind.INBOUND

    def __init__(
        self,
        handler: MessageHandler,
        *,
        allowed_senders: Collection[str] | None = None,
        signing_secret: str | None = None,
        signature_header: str = "",
        signature_prefix: str = "",
        signature_algorithm: str = "sha256",
        allow_empty_allowlist: bool = False,
        context: ConnectorContext | None = None,
    ) -> None:
        """Wire the agent ``handler`` plus the sender allowlist + signature policy.

        ``allowed_senders`` is the allowlist (compared as strings). With
        ``allow_empty_allowlist=False`` (the default, and the only safe setting for a
        publicly reachable webhook), an empty allowlist denies EVERY sender. A private
        poll bot may pass ``allow_empty_allowlist=True`` to answer everyone.
        ``signing_secret`` enables HMAC verification on :meth:`handle_webhook`.
        """
        super().__init__(context=context)
        self._handler = handler
        self._allowed = {str(s) for s in (allowed_senders or [])}
        self._signing_secret = signing_secret
        self._signature_header = signature_header
        self._signature_prefix = signature_prefix
        self._signature_algorithm = signature_algorithm
        self._allow_empty_allowlist = allow_empty_allowlist

    # -- the security gate --------------------------------------------------
    def is_allowed(self, sender_id: str) -> bool:
        """Whether ``sender_id`` is on the allowlist (default-deny semantics).

        With a non-empty allowlist, only listed senders pass. With an EMPTY allowlist
        the answer is ``allow_empty_allowlist`` — False by default, so an unconfigured
        connector denies everyone rather than silently answering the world.
        """
        if not self._allowed:
            return self._allow_empty_allowlist
        return str(sender_id) in self._allowed

    def _authorize(self, msg: InboundMessage) -> bool:
        """Allowlist-gate one parsed message, auditing the allow/deny decision."""
        if self.is_allowed(msg.sender_id):
            return True
        self._audit(
            "inbound_message",
            "deny",
            actor={"subject": msg.sender_id},
            detail="sender not on allowlist",
        )
        return False

    def verify_webhook(self, body: bytes, headers: Mapping[str, str]) -> bool:
        """Verify the inbound webhook's HMAC signature (default-deny when configured).

        When a ``signing_secret`` was provided, the signature header MUST be present
        and valid — a missing or wrong signature returns False. When NO secret was
        configured the method returns True (signature verification not in use), so an
        operator must consciously opt out of it; the allowlist still gates the sender.
        """
        if not self._signing_secret:
            return True
        provided = _header_lookup(headers, self._signature_header)
        return verify_hmac_signature(
            secret=self._signing_secret,
            body=body,
            provided=provided,
            algorithm=self._signature_algorithm,
            prefix=self._signature_prefix,
        )

    # -- polling delivery ---------------------------------------------------
    async def poll_once(self, offset: int | None) -> int | None:
        """Fetch one batch, run the handler for each ALLOWED message, return the offset.

        Mirrors the Telegram loop: every update advances the offset (so a denied sender
        is consumed, not retried forever), but only an allow-listed sender reaches the
        agent handler and gets a reply.
        """
        updates = await self.fetch_updates(offset)
        for update in updates:
            offset = self.next_offset(update, offset)
            msg = self.parse_update(update)
            if msg is None or not msg.text:
                continue
            await self._dispatch(msg)
        return offset

    async def run(self, *, stop: Callable[[], bool] | None = None) -> None:
        """Poll until ``stop()`` is True (forever when ``stop`` is None)."""
        offset: int | None = None
        while not (stop and stop()):
            offset = await self.poll_once(offset)

    # -- webhook delivery ---------------------------------------------------
    async def handle_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """Verify signature, then allowlist, then run the agent for a webhook delivery.

        Order is load-bearing: the cryptographic signature check happens on the RAW
        body before parsing, so an attacker can't get the parser to run on a forged
        payload. A failed signature audits a deny and returns ``{"ok": False}`` without
        touching the handler. A valid-but-unlisted sender is likewise denied.
        """
        if not self.verify_webhook(body, headers):
            self._audit("webhook", "deny", detail="signature verification failed")
            return {"ok": False, "reason": "invalid signature"}
        msg = self.parse_webhook(body, headers)
        if msg is None or not msg.text:
            return {"ok": True, "handled": False}
        reply = await self._dispatch(msg)
        return {"ok": True, "handled": reply is not None, "reply": reply}

    async def _dispatch(self, msg: InboundMessage) -> str | None:
        """Run the gate + handler for one message; return the reply (or None)."""
        if not self._authorize(msg):
            return None
        self._throttle(msg.sender_id)
        reply = await self._handler(msg.sender_id, msg.text)
        self._audit(
            "inbound_message",
            "allow",
            actor={"subject": msg.sender_id},
            detail="handled",
        )
        if reply:
            await self.send_reply(msg, reply)
        return reply

    # -- channel-specific seams (subclasses implement) ---------------------
    @abstractmethod
    async def fetch_updates(self, offset: int | None) -> list[Any]:
        """Pull one batch of raw updates from the channel (polling delivery)."""

    @abstractmethod
    def parse_update(self, update: Any) -> InboundMessage | None:
        """Normalize one raw poll update into an :class:`InboundMessage` (None = skip)."""

    def parse_webhook(
        self, body: bytes, headers: Mapping[str, str]
    ) -> InboundMessage | None:
        """Normalize a webhook body into an :class:`InboundMessage`.

        Defaults to "this channel does not support webhook delivery"; override for a
        channel that does (after the SDK has already verified the signature).
        """
        raise ConnectorError(f"{self.name} does not support webhook delivery")

    def next_offset(self, update: Any, current: int | None) -> int | None:
        """Compute the poll offset after ``update`` (default: leave it unchanged)."""
        return current

    @abstractmethod
    async def send_reply(self, msg: InboundMessage, reply: str) -> None:
        """Send ``reply`` back to the sender of ``msg`` over the channel."""


def _header_lookup(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (HTTP header names are case-insensitive)."""
    if not name:
        return None
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


# --------------------------------------------------------- outbound tool base
class OutboundToolConnector(Connector):
    """Base for connectors that expose typed tools the agent can call.

    The contract every subclass inherits:

    * **Credentials from the secrets layer.** :meth:`secret` reads through
      :func:`get_secret` — never ``os.environ`` directly, never a constructor arg the
      model could populate. A missing required secret is a clean
      :class:`ConnectorError` (naming the var), never a leak of its value.
    * **No SSRF hole.** :meth:`fetcher` returns a :class:`Fetcher` whose transport pins
      the DNS-vetted IP and re-guards every redirect hop; :meth:`guard` validates a
      model-supplied URL before any request. Subclasses route ALL HTTP through these.
    * **Idempotent side effects.** :meth:`call_idempotent` wraps a mutating call with an
      idempotency key so a retry (or a re-issued tool call) fires the effect once.
    * **Typed tools.** :meth:`register` adds the connector's tools to a
      :class:`ToolRegistry`, so each one flows through the normal tool pipeline
      (validation / approval / events / lineage).
    """

    kind = ConnectorKind.OUTBOUND

    #: Secret names this connector needs; surfaced by :meth:`capability`.
    required_secrets: tuple[str, ...] = ()
    #: Importable modules this connector needs; surfaced by :meth:`capability`.
    required_modules: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        fetcher: Fetcher | None = None,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = None,
        idempotency: IdempotencyStore | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        """Wire the guarded HTTP seam, egress policy, and idempotency store.

        ``fetcher`` defaults to an :class:`HttpxFetcher` carrying :func:`guard_url` and
        a pinned transport — the SSRF-safe default. ``allow_private_hosts`` /
        ``egress_allow_hosts`` narrow what :meth:`guard` permits (an air-gapped or
        allow-listed deployment). ``idempotency`` defaults to an in-process store.
        """
        super().__init__(context=context)
        self._allow_private = allow_private_hosts
        self._allow_hosts = set(egress_allow_hosts) if egress_allow_hosts else None
        self._idempotency = idempotency or IdempotencyStore()
        self._fetcher = fetcher or HttpxFetcher(
            guard=self._guard_for_fetcher(),
            transport=build_pinned_transport(
                allow_private=allow_private_hosts, allow_hosts=self._allow_hosts
            ),
        )

    def capability(self) -> Capability:
        """Available only when every required module imports + secret resolves."""
        return capability(
            requires_modules=self.required_modules,
            requires_secrets=self.required_secrets,
        )

    # -- credentials --------------------------------------------------------
    def secret(self, name: str, *, required: bool = True) -> str | None:
        """Resolve secret ``name`` via the secrets layer; raise if required + absent.

        NEVER reads ``os.environ`` directly and NEVER echoes the value — a missing
        required secret raises a :class:`ConnectorError` naming only the variable.
        """
        value = get_secret(name)
        if not value and required:
            raise ConnectorError(
                f"{self.name} requires the secret {name!r} (not configured)"
            )
        return value

    # -- guarded HTTP -------------------------------------------------------
    def _guard_for_fetcher(self) -> Callable[[str], str] | None:
        """The URL guard the default fetcher applies to its URL + every redirect hop."""
        if self._allow_private and not self._allow_hosts:
            return None  # private explicitly allowed and no allow-list → no guard

        def _guard(url: str) -> str:
            return guard_url(
                url,
                allow_private=self._allow_private,
                allow_hosts=self._allow_hosts,
            )

        return _guard

    def guard(self, url: str) -> str:
        """SSRF-validate a (possibly model-supplied) URL; raise to refuse it.

        Call this on any URL whose host the model or upstream could influence BEFORE
        fetching. Enforces the connector's egress policy (public-only, plus an optional
        allow-list). Returns the URL unchanged when it passes.
        """
        return guard_url(
            url, allow_private=self._allow_private, allow_hosts=self._allow_hosts
        )

    @property
    def fetcher(self) -> Fetcher:
        """The SSRF-guarded fetcher all outbound HTTP must go through."""
        return self._fetcher

    # -- idempotent side effects -------------------------------------------
    def call_idempotent(self, key: str, call: Callable[[], Any]) -> Any:
        """Run a side-effecting ``call`` at most once for ``key`` (dedupe retries).

        Combines the :class:`IdempotencyStore` (run-once per key) with the connector's
        :class:`RetryPolicy`: the call is treated as idempotent because the store
        guarantees the effect fires once, so a bounded retry of a transient transport
        error is safe. A repeat of the same key returns the first result without
        re-firing the effect.
        """
        retry = self._ctx.retry

        def _guarded() -> Any:
            return retry.run(call, idempotent=True)

        return self._idempotency.run_once(key, _guarded)

    # -- tool registration --------------------------------------------------
    @abstractmethod
    def register(self, registry: ToolRegistry) -> list[str]:
        """Register this connector's typed tools; return the registered tool names.

        Capability is checked first so a connector missing a dependency/credential is
        skipped cleanly (no tools registered, no crash). Subclasses call
        :meth:`register_tools` after their own capability gate.
        """

    def register_tools(self, registry: ToolRegistry) -> list[str]:
        """Register the connector's tools, skipping cleanly if it is unavailable.

        Wraps :meth:`register` with the capability gate: if a dependency or credential
        is missing this returns ``[]`` (and audits the skip) instead of raising, so a
        toolkit assembling many connectors degrades gracefully.
        """
        cap = self.capability()
        if not cap.ok:
            self._audit("register", "deny", detail=cap.reason)
            return []
        names = self.register(registry)
        self._audit("register", "allow", detail=f"registered {len(names)} tool(s)")
        return names


# --------------------------------------------------------------- registry
class ConnectorRegistry:
    """Discovery surface: register connector factories, look them up by name/kind.

    A *factory* (zero-arg callable returning a :class:`Connector`) is registered, not a
    live instance, so capability is evaluated lazily — listing available connectors
    can't be derailed by one whose construction needs a missing credential. Discovery
    methods filter to the *available* ones so the caller wires only what can actually
    run in this environment.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Connector]] = {}

    def register(self, name: str, factory: Callable[[], Connector]) -> None:
        """Register ``factory`` under ``name`` (replacing any prior registration)."""
        if not name:
            raise ConnectorError("a connector must be registered under a non-empty name")
        self._factories[name] = factory

    def names(self) -> list[str]:
        """Every registered connector name (regardless of availability)."""
        return sorted(self._factories)

    def create(self, name: str) -> Connector:
        """Instantiate the connector registered under ``name``.

        Raises :class:`ConnectorError` for an unknown name — discovery never silently
        returns a wrong/None connector.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise ConnectorError(f"no connector registered under {name!r}")
        return factory()

    def get(self, name: str) -> Connector | None:
        """Instantiate ``name`` or return None when it is not registered."""
        factory = self._factories.get(name)
        return factory() if factory is not None else None

    def discover(
        self, *, kind: ConnectorKind | None = None, available_only: bool = True
    ) -> list[Connector]:
        """Instantiate the registered connectors, filtered by kind/availability.

        With ``available_only`` (the default), a connector whose capability probe fails
        (missing dependency or credential) is *skipped cleanly* — it never appears in
        the discovered list, so callers wire only connectors that can run here.
        """
        out: list[Connector] = []
        for name in self.names():
            connector = self.create(name)
            if kind is not None and connector.kind != kind:
                continue
            if available_only and not connector.available():
                continue
            out.append(connector)
        return out


#: The process-wide default registry. Connectors self-register here at import so
#: discovery finds them; tests build their own :class:`ConnectorRegistry` for isolation.
_DEFAULT_REGISTRY = ConnectorRegistry()


def default_registry() -> ConnectorRegistry:
    """Return the process-wide connector registry."""
    return _DEFAULT_REGISTRY


def register_connector(name: str, factory: Callable[[], Connector]) -> None:
    """Register a connector factory on the process-wide :func:`default_registry`."""
    _DEFAULT_REGISTRY.register(name, factory)


__all__ = [
    "ConnectorError",
    "ConnectorKind",
    "Connector",
    "ConnectorContext",
    "InboundChannelConnector",
    "OutboundToolConnector",
    "InboundMessage",
    "MessageHandler",
    "ConnectorRegistry",
    "default_registry",
    "register_connector",
    "AuditSink",
    "RateLimiter",
    "RetryPolicy",
    "IdempotencyStore",
    "Capability",
    "capability",
    "verify_hmac_signature",
    "safe_error",
    "redact_args",
    "CONNECTOR_EVENT_TYPE",
]
