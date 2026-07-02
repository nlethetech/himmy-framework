"""Connector management: ONE service over the secrets layer + the connector registry.

The connector classes (Slack/Discord/GitHub/web-search outbound, the generic HMAC
inbound webhook) are fully built and hardened in :mod:`himmy.connectors`, but until now
they were reachable from no interface — there was no place that *registered* the
built-ins, no way to write their credentials, no enable/disable switch, and no read-only
catalog. This module is that single unifying surface; the CLI ``himmy connectors``
subcommand and the read-only ``GET /v1/connectors`` endpoint are thin adapters over it.

What it gives every interface, once:

* **catalog** — a :class:`ConnectorDescriptor` per built-in connector: its kind, the
  secret names it needs (required + optional), its allow-list field, and a live
  capability probe (deps + credentials present).
* **configure** — write a connector's secrets through the *writable* secrets backend
  (:func:`himmy.config.secrets.get_writable_provider`), the SAME path Studio Connections
  uses. On the bare env-only backend this FAILS LOUDLY (:class:`ReadOnlyBackendError`),
  never a silent no-op.
* **enable / disable** — a *non-secret* ``HIMMY_CONNECTOR_<NAME>_<SURFACE>_ENABLED`` flag
  (default absent ⇒ OFF), written through the same backend; a connector is wired into a
  runtime only when its flag is on AND it is configured.
* **status / test** — reuse :meth:`Connector.capability` for the structural readiness
  check, plus an optional live read-only ping (declared per descriptor) that exercises
  the real provider so a green check means the credential actually works.
* **build_outbound / build_inbound** — construct the live connector from the registry
  (outbound) or with an agent ``handler`` (inbound), refusing cleanly when it is not
  configured.

:func:`ensure_builtin_connectors_registered` populates the process-wide
:func:`~himmy.connectors.sdk.default_registry` with the built-in outbound connectors,
idempotently. The whole surface is secret-safe: a status/catalog view NEVER echoes a
secret value — only whether each named secret is present.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from himmy.config.secrets import get_secret, get_writable_provider
from himmy.connectors.discord import (
    DISCORD_BOT_TOKEN_SECRET,
    DISCORD_PUBLIC_KEY_SECRET,
    DiscordInteractionsConnector,
    DiscordOutboundConnector,
    register_discord_connectors,
)
from himmy.connectors.github import (
    GITHUB_TOKEN_SECRET,
    GitHubOutboundConnector,
    register_github_connectors,
)
from himmy.connectors.sdk import (
    Capability,
    Connector,
    ConnectorContext,
    ConnectorError,
    ConnectorKind,
    ConnectorRegistry,
    InboundChannelConnector,
    MessageHandler,
    OutboundToolConnector,
    default_registry,
)
from himmy.connectors.slack import (
    SLACK_APP_TOKEN_SECRET,
    SLACK_BOT_TOKEN_SECRET,
    SLACK_SIGNING_SECRET,
    SlackEventsConnector,
    SlackOutboundConnector,
    register_slack_connectors,
)
from himmy.connectors.webhook import (
    WEBHOOK_SIGNING_SECRET,
    make_webhook_connector,
)
from himmy.connectors.websearch import (
    BRAVE_API_KEY_SECRET,
    TAVILY_API_KEY_SECRET,
    WebSearchConnector,
    register_websearch_connectors,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable


# Non-secret per-connector "default target" store keys (the optional channel/repo a post
# goes to when the caller names none). They are NOT credentials but ARE configurable, so
# they live in the secrets backend alongside the token and are written via configure();
# _direct_outbound reads them through get_secret so they compose with the allow-list.
SLACK_DEFAULT_CHANNEL_SECRET = "HIMMY_SLACK_DEFAULT_CHANNEL"  # noqa: S105 - a NAME, not a value
DISCORD_DEFAULT_CHANNEL_SECRET = "HIMMY_DISCORD_DEFAULT_CHANNEL"  # noqa: S105 - a NAME
GITHUB_DEFAULT_REPO_SECRET = "HIMMY_GITHUB_DEFAULT_REPO"  # noqa: S105 - a NAME, not a value


class ReadOnlyBackendError(RuntimeError):
    """Raised when configure/enable/disable is attempted on a read-only secrets backend.

    The bare env-only backend (the offline default) cannot persist a secret or a flag,
    so a write through it would be a silent no-op. We refuse loudly instead — the
    operator must select a writable backend (``HIMMY_SECRETS=keychain`` or ``file``)
    before configuring a connector.
    """


@dataclass(frozen=True)
class SecretSpec:
    """One secret a connector reads, plus whether it is required for the connector to run."""

    name: str  # underlying HIMMY_* store key (never a literal value)
    label: str
    required: bool = True
    placeholder: str = ""


@dataclass(frozen=True)
class ConnectorDescriptor:
    """Declarative description of one built-in connector (no secret values).

    The catalog the CLI and ``/v1/connectors`` render is a list of these. ``probe`` is an
    optional live read-only ping (an async callable returning a short detail string) used
    by :meth:`ConnectorService.test`; it MUST be a read-only call (a search, an
    authenticated-user GET) so a "test" never has a side effect.
    """

    name: str  # the registry/identity key (e.g. "slack")
    title: str
    description: str
    kind: ConnectorKind  # outbound (tools) | inbound (channel)
    secrets: tuple[SecretSpec, ...]
    #: The env name that holds this connector's allow-list (channels/repos/senders), or
    #: None when the connector has none. Non-secret, default-deny when unset.
    allowlist_env: str | None = None
    #: Optional live read-only ping (None ⇒ structural capability check only).
    probe: Callable[[], Awaitable[str]] | None = field(default=None, compare=False)

    def required_secret_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.secrets if s.required)


# --------------------------------------------------------------- live read-only probes


async def _probe_websearch() -> str:
    """A real (keyless or keyed) one-result search — read-only, no side effect."""
    import asyncio

    from himmy.services.tools.registry import ToolRegistry

    connector = WebSearchConnector()
    registry = ToolRegistry()
    connector.register(registry)
    handler = registry.handler_for("web_search")
    if handler is None:  # pragma: no cover - defensive
        raise ConnectorError("web_search tool is not available")
    result = await asyncio.wait_for(
        handler({"query": "himmy connectivity test", "count": "1"}), timeout=20.0
    )
    count = (result or {}).get("count")
    return f"{(result or {}).get('backend', 'search')}: {count} result(s)"


async def _probe_github() -> str:
    """GET the authenticated user — a read-only call that proves the token works."""
    import asyncio

    from himmy.connectors.github import GitHubClient

    token = get_secret(GITHUB_TOKEN_SECRET)
    if not token:
        raise ConnectorError(f"missing credential {GITHUB_TOKEN_SECRET!r}")

    def _call() -> str:
        user = GitHubClient(token).get("/user")
        login = (user or {}).get("login") if isinstance(user, dict) else None
        return f"@{login}" if login else "authenticated"

    return await asyncio.wait_for(asyncio.to_thread(_call), timeout=20.0)


# ----------------------------------------------------------------- the catalog


def _build_catalog() -> dict[str, ConnectorDescriptor]:
    """The built-in connector descriptors (declared once, secret-safe)."""
    descriptors = (
        ConnectorDescriptor(
            name="slack",
            title="Slack",
            description="Post messages to Slack channels (and receive events).",
            kind=ConnectorKind.OUTBOUND,
            secrets=(
                SecretSpec(SLACK_BOT_TOKEN_SECRET, "Bot token", placeholder="xoxb-..."),
                SecretSpec(
                    SLACK_SIGNING_SECRET,
                    "Signing secret (inbound)",
                    required=False,
                ),
                SecretSpec(
                    SLACK_APP_TOKEN_SECRET,
                    "App token (socket mode)",
                    required=False,
                    placeholder="xapp-...",
                ),
                SecretSpec(
                    SLACK_DEFAULT_CHANNEL_SECRET,
                    "Default channel",
                    required=False,
                    placeholder="C0123ABCDEF",
                ),
            ),
            allowlist_env="HIMMY_SLACK_ALLOWED_CHANNELS",
        ),
        ConnectorDescriptor(
            name="discord",
            title="Discord",
            description="Post messages to Discord channels (and receive interactions).",
            kind=ConnectorKind.OUTBOUND,
            secrets=(
                SecretSpec(DISCORD_BOT_TOKEN_SECRET, "Bot token"),
                SecretSpec(
                    DISCORD_PUBLIC_KEY_SECRET,
                    "Public key (inbound)",
                    required=False,
                ),
                SecretSpec(
                    DISCORD_DEFAULT_CHANNEL_SECRET,
                    "Default channel",
                    required=False,
                ),
            ),
            allowlist_env="HIMMY_DISCORD_ALLOWED_CHANNELS",
        ),
        ConnectorDescriptor(
            name="github",
            title="GitHub",
            description="Read/act on GitHub issues and pull requests.",
            kind=ConnectorKind.OUTBOUND,
            secrets=(
                SecretSpec(GITHUB_TOKEN_SECRET, "API token", placeholder="ghp_..."),
                SecretSpec(
                    GITHUB_DEFAULT_REPO_SECRET,
                    "Default repository",
                    required=False,
                    placeholder="owner/repo",
                ),
            ),
            allowlist_env="HIMMY_GITHUB_ALLOWED_REPOS",
            probe=_probe_github,
        ),
        ConnectorDescriptor(
            name="websearch",
            title="Web search",
            description="Search the web (Tavily/Brave need a key; DuckDuckGo is keyless).",
            kind=ConnectorKind.OUTBOUND,
            secrets=(
                SecretSpec(
                    TAVILY_API_KEY_SECRET, "Tavily API key", required=False,
                ),
                SecretSpec(
                    BRAVE_API_KEY_SECRET, "Brave API key", required=False,
                ),
            ),
            probe=_probe_websearch,
        ),
        ConnectorDescriptor(
            name="webhook",
            title="Inbound webhook",
            description="Receive HMAC-signed webhook deliveries and run the agent.",
            kind=ConnectorKind.INBOUND,
            secrets=(
                SecretSpec(
                    WEBHOOK_SIGNING_SECRET,
                    "Signing secret",
                    placeholder="a long random string",
                ),
            ),
            allowlist_env="HIMMY_WEBHOOK_ALLOWED_SOURCES",
        ),
    )
    return {d.name: d for d in descriptors}


#: The built-in connector catalog (name → descriptor).
CONNECTOR_CATALOG: dict[str, ConnectorDescriptor] = _build_catalog()


# ----------------------------------------------------- enable/disable flag naming

#: Surfaces a connector can be enabled on. Outbound → wired as agent tools; inbound →
#: wired as a channel listener (webhook BFF / Slack events / Discord interactions).
_VALID_SURFACES = ("outbound", "inbound")


def _enabled_flag_name(connector: str, surface: str) -> str:
    """The non-secret env flag for ``connector`` on ``surface`` (default absent ⇒ OFF)."""
    return f"HIMMY_CONNECTOR_{connector.upper()}_{surface.upper()}_ENABLED"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------- registration (idempotent)

_REGISTER_LOCK = threading.Lock()
_REGISTERED = False


def ensure_builtin_connectors_registered() -> None:
    """Register the built-in outbound connectors on the process-wide registry, once.

    The connector modules ship ``register_*_connectors`` self-registration helpers but
    nothing called them; this is the single choke point that wires them so discovery
    (and the catalog's registry lookups) find them. Idempotent and thread-safe — safe to
    call at CLI start, at app startup, and from every catalog read.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        register_slack_connectors()
        register_discord_connectors()
        register_github_connectors()
        register_websearch_connectors()
        _REGISTERED = True


# ------------------------------------------------------------------- status models


@dataclass(frozen=True)
class SecretStatus:
    """Per-secret presence (never the value)."""

    name: str
    label: str
    required: bool
    present: bool
    placeholder: str = ""


@dataclass(frozen=True)
class ConnectorStatus:
    """The catalog row for one connector: descriptor + configured/enabled state.

    Contains NO secret values — only each secret's NAME and whether it is present.
    """

    name: str
    title: str
    description: str
    kind: str
    allowlist_env: str | None
    allowlist_set: bool
    configured: bool  # every REQUIRED secret resolves
    available: bool  # capability probe ok (deps + credentials)
    capability_reason: str
    writable: bool  # a writable secrets backend is active
    enabled_outbound: bool
    enabled_inbound: bool
    secrets: tuple[SecretStatus, ...]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict (used by the /v1 read-only catalog response)."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "kind": self.kind,
            "allowlist_env": self.allowlist_env,
            "allowlist_set": self.allowlist_set,
            "configured": self.configured,
            "available": self.available,
            "capability_reason": self.capability_reason,
            "writable": self.writable,
            "enabled": {
                "outbound": self.enabled_outbound,
                "inbound": self.enabled_inbound,
            },
            "secrets": [
                {
                    "name": s.name,
                    "label": s.label,
                    "required": s.required,
                    "present": s.present,
                    "placeholder": s.placeholder,
                }
                for s in self.secrets
            ],
        }


@dataclass(frozen=True)
class ConnectorTestResult:
    """The outcome of a connector test (capability + optional live ping)."""

    name: str
    ok: bool
    detail: str
    checked: str  # what was actually exercised ("capability" | "live ping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "checked": self.checked,
        }


# --------------------------------------------------------------------- the service


class ConnectorService:
    """ONE service over the secrets layer + the connector registry + the catalog.

    Construct once (cheap); call from the CLI, the BFF, and the runtime adapter. The
    registry defaults to the process-wide :func:`default_registry` (with the built-ins
    ensured-registered); a test passes its own registry/catalog for isolation.
    """

    def __init__(
        self,
        *,
        registry: ConnectorRegistry | None = None,
        catalog: dict[str, ConnectorDescriptor] | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        ensure_builtin_connectors_registered()
        self._registry = registry or default_registry()
        self._catalog = catalog if catalog is not None else CONNECTOR_CATALOG
        self._context = context

    # -- catalog / status ---------------------------------------------------
    def names(self) -> list[str]:
        """Every connector name in the catalog (sorted)."""
        return sorted(self._catalog)

    def descriptor(self, name: str) -> ConnectorDescriptor:
        """The descriptor for ``name``; raise :class:`KeyError` for an unknown connector."""
        descriptor = self._catalog.get(name)
        if descriptor is None:
            raise KeyError(name)
        return descriptor

    def catalog(self) -> list[ConnectorStatus]:
        """The full catalog with per-connector configured/enabled status (no secrets)."""
        return [self.status(name) for name in self.names()]

    def status(self, name: str) -> ConnectorStatus:
        """Configured/enabled/available status for one connector (no secret values)."""
        descriptor = self.descriptor(name)
        writable = get_writable_provider() is not None
        secrets: list[SecretStatus] = []
        configured = True
        for spec in descriptor.secrets:
            present = bool(get_secret(spec.name))
            if spec.required and not present:
                configured = False
            secrets.append(
                SecretStatus(
                    name=spec.name,
                    label=spec.label,
                    required=spec.required,
                    present=present,
                    placeholder=spec.placeholder,
                )
            )
        allowlist_set = bool(
            descriptor.allowlist_env and get_secret(descriptor.allowlist_env)
        )
        cap = self._capability(descriptor)
        return ConnectorStatus(
            name=descriptor.name,
            title=descriptor.title,
            description=descriptor.description,
            kind=descriptor.kind.value,
            allowlist_env=descriptor.allowlist_env,
            allowlist_set=allowlist_set,
            configured=configured,
            available=cap.ok,
            capability_reason=cap.reason,
            writable=writable,
            enabled_outbound=self.is_enabled(name, "outbound"),
            enabled_inbound=self.is_enabled(name, "inbound"),
            secrets=tuple(secrets),
        )

    def _capability(self, descriptor: ConnectorDescriptor) -> Capability:
        """Probe the live connector's capability without firing any request.

        Outbound connectors expose ``capability()`` from a freshly-built instance;
        inbound connectors are credential-gated on their signing secret.
        """
        if descriptor.kind == ConnectorKind.OUTBOUND:
            try:
                connector = self._build_outbound_instance(descriptor.name)
            except ConnectorError as exc:
                return Capability(ok=False, reason=str(exc))
            return connector.capability()
        # Inbound: available once every required secret is present.
        missing = [
            s.name for s in descriptor.secrets if s.required and not get_secret(s.name)
        ]
        if missing:
            return Capability(ok=False, reason=f"missing credential {missing[0]!r}")
        return Capability(ok=True)

    # -- configure ----------------------------------------------------------
    def configure(self, name: str, values: dict[str, Any]) -> ConnectorStatus:
        """Write the connector's secrets/allow-list through the writable backend.

        ``values`` is keyed by secret NAME (and may include the descriptor's
        ``allowlist_env``); an empty value DELETEs the key. Unknown keys are ignored
        (never write an arbitrary store key). FAILS LOUDLY on a read-only backend.
        """
        descriptor = self.descriptor(name)
        writable = self._require_writable()
        allowed = {s.name for s in descriptor.secrets}
        if descriptor.allowlist_env:
            allowed.add(descriptor.allowlist_env)
        for key, raw in values.items():
            if key not in allowed:
                continue
            value = "" if raw is None else str(raw)
            if value == "":
                writable.delete(key)
            else:
                writable.set(key, value)
        return self.status(name)

    def unconfigure(self, name: str) -> ConnectorStatus:
        """Delete every secret + allow-list this connector owns (fails loud if read-only)."""
        descriptor = self.descriptor(name)
        writable = self._require_writable()
        for spec in descriptor.secrets:
            writable.delete(spec.name)
        if descriptor.allowlist_env:
            writable.delete(descriptor.allowlist_env)
        return self.status(name)

    # -- enable / disable ---------------------------------------------------
    def is_enabled(self, name: str, surface: str) -> bool:
        """Whether ``name`` is enabled on ``surface`` (default absent ⇒ OFF)."""
        self.descriptor(name)  # validate the name
        self._validate_surface(surface)
        return _truthy(get_secret(_enabled_flag_name(name, surface)))

    def enable(self, name: str, surface: str) -> ConnectorStatus:
        """Turn the connector ON for ``surface`` (writes the non-secret flag).

        FAILS LOUDLY on a read-only backend — an enable that could not persist would be a
        silent no-op the next process never sees.
        """
        self.descriptor(name)
        self._validate_surface(surface)
        writable = self._require_writable()
        writable.set(_enabled_flag_name(name, surface), "1")
        return self.status(name)

    def disable(self, name: str, surface: str) -> ConnectorStatus:
        """Turn the connector OFF for ``surface`` (deletes the flag; fails loud if read-only)."""
        self.descriptor(name)
        self._validate_surface(surface)
        writable = self._require_writable()
        writable.delete(_enabled_flag_name(name, surface))
        return self.status(name)

    # -- test ---------------------------------------------------------------
    async def test(self, name: str) -> ConnectorTestResult:
        """Validate a connector: structural capability, then an optional live read ping.

        The capability check (deps + credentials present) runs first and short-circuits a
        live ping when it fails — so a "test" of an unconfigured connector reports the
        missing credential by NAME rather than throwing a network error. When the
        descriptor declares a read-only ``probe`` and capability passed, the probe runs
        and a green result means the credential actually works.
        """
        descriptor = self.descriptor(name)
        cap = self._capability(descriptor)
        if not cap.ok:
            return ConnectorTestResult(
                name=name, ok=False, detail=cap.reason, checked="capability"
            )
        if descriptor.probe is None:
            return ConnectorTestResult(
                name=name,
                ok=True,
                detail="configured (no live ping for this connector)",
                checked="capability",
            )
        try:
            detail = await descriptor.probe()
        except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe report
            from himmy.connectors.sdk import safe_error

            return ConnectorTestResult(
                name=name,
                ok=False,
                detail=safe_error(exc),
                checked="live ping",
            )
        return ConnectorTestResult(
            name=name, ok=True, detail=detail, checked="live ping"
        )

    # -- build ---------------------------------------------------------------
    def build_outbound(self, name: str) -> OutboundToolConnector:
        """Construct the live outbound connector, refusing cleanly if not configured.

        Raises :class:`ConnectorError` when the connector is the wrong kind, unknown, or
        not configured (its required credential is absent) — never returns a half-wired
        tool that would explode at call time.
        """
        descriptor = self.descriptor(name)
        if descriptor.kind != ConnectorKind.OUTBOUND:
            raise ConnectorError(f"{name!r} is not an outbound connector")
        connector = self._build_outbound_instance(name)
        cap = connector.capability()
        if not cap.ok:
            raise ConnectorError(f"{name!r} is not configured: {cap.reason}")
        return connector

    def build_inbound(
        self,
        name: str,
        handler: MessageHandler,
        **kwargs: Any,
    ) -> InboundChannelConnector:
        """Construct the live inbound connector wired to an agent ``handler``.

        Inbound connectors carry the agent turn, so (unlike outbound) the registry's
        zero-arg factory can't build them — this passes the ``handler`` (and any extra
        kwargs the channel accepts). Refuses cleanly when the required credential is
        absent.
        """
        descriptor = self.descriptor(name)
        if descriptor.kind != ConnectorKind.INBOUND:
            raise ConnectorError(f"{name!r} is not an inbound connector")
        missing = [
            s.name for s in descriptor.secrets if s.required and not get_secret(s.name)
        ]
        if missing:
            raise ConnectorError(
                f"{name!r} is not configured: missing credential {missing[0]!r}"
            )
        connector = self._build_inbound_instance(name, handler, **kwargs)
        return connector

    # -- internal builders --------------------------------------------------
    def _build_outbound_instance(self, name: str) -> OutboundToolConnector:
        """Instantiate the outbound connector with SECRETS-AWARE allow-lists/defaults.

        For a built-in connector we construct it through :func:`_direct_outbound`, which
        reads the allow-list/default via :func:`_split_env`/:func:`get_secret` — so an
        allow-list written by :meth:`configure` (into the *writable secrets backend*, not
        ``os.environ``) is actually honored. The registry factory is intentionally NOT
        used for built-ins: ``_make_outbound`` reads ONLY ``os.environ`` (github.py /
        slack.py / discord.py), so routing through it would silently ignore the
        just-configured allow-list and default-deny every call while :meth:`status`
        (which reads the secrets backend) reported the connector configured+available.

        The registry remains the path for any NON-built-in connector a caller registered
        directly (a third-party/custom connector the catalog has no descriptor for).
        """
        if name in _DIRECT_OUTBOUND_NAMES:
            connector: Connector = _direct_outbound(name, context=self._context)
        else:
            registered = self._registry.get(name)
            connector = registered if registered is not None else _direct_outbound(name)
        if not isinstance(connector, OutboundToolConnector):
            raise ConnectorError(f"{name!r} is not an outbound connector")
        return connector

    def _build_inbound_instance(
        self, name: str, handler: MessageHandler, **kwargs: Any
    ) -> InboundChannelConnector:
        """Instantiate the inbound connector for ``name`` wired to ``handler``."""
        if name == "webhook":
            allow = _split_env("HIMMY_WEBHOOK_ALLOWED_SOURCES")
            require_ts = _secret_truthy("HIMMY_WEBHOOK_REQUIRE_TIMESTAMP")
            return make_webhook_connector(
                handler,
                allowed_sources=kwargs.pop("allowed_sources", None) or allow or None,
                require_timestamp=kwargs.pop("require_timestamp", None) or require_ts,
                context=self._context,
                **kwargs,
            )
        if name == "slack":
            return SlackEventsConnector(
                handler,
                allowed_channel_ids=kwargs.pop("allowed_channel_ids", None)
                or _split_env("HIMMY_SLACK_ALLOWED_CHANNELS")
                or None,
                context=self._context,
                **kwargs,
            )
        if name == "discord":
            return DiscordInteractionsConnector(
                handler,
                allowed_channel_ids=kwargs.pop("allowed_channel_ids", None)
                or _split_env("HIMMY_DISCORD_ALLOWED_CHANNELS")
                or None,
                context=self._context,
                **kwargs,
            )
        raise ConnectorError(f"{name!r} has no inbound channel")

    # -- helpers ------------------------------------------------------------
    def _require_writable(self) -> Any:
        writable = get_writable_provider()
        if writable is None:
            raise ReadOnlyBackendError(
                "the active secrets backend is read-only; set HIMMY_SECRETS=keychain "
                "or file to configure connectors"
            )
        return writable

    @staticmethod
    def _validate_surface(surface: str) -> None:
        if surface not in _VALID_SURFACES:
            raise ValueError(
                f"unknown surface {surface!r} (use {' | '.join(_VALID_SURFACES)})"
            )


def _split_env(name: str) -> list[str]:
    """Split a comma-separated allow-list env into a clean list (resolved via secrets)."""
    raw = get_secret(name) or os.environ.get(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _secret_truthy(name: str) -> bool:
    """Whether a boolean config flag (resolved via the secrets layer) is on."""
    from himmy.config.flags import truthy

    raw = get_secret(name) or os.environ.get(name) or ""
    return truthy(raw)


#: Built-in outbound connectors that :func:`_direct_outbound` can construct directly,
#: reading their allow-list/default through the secrets layer (not just ``os.environ``).
#: These are built secrets-aware in :meth:`ConnectorService._build_outbound_instance`
#: rather than via the env-only registry factory, so :meth:`ConnectorService.configure`
#: composes with :meth:`ConnectorService.build_outbound`.
_DIRECT_OUTBOUND_NAMES = frozenset({"slack", "discord", "github", "websearch"})

#: Each direct-built outbound connector's per-connector self-throttle (tokens/sec). These
#: MIRROR the defaults each connector applies when it is built with no context — so when
#: the runtime supplies a context (to wire the run's audit spine), the rate limiter is
#: preserved, not dropped. Keep in sync with the connectors' ``__init__`` defaults.
_OUTBOUND_RATE: dict[str, float] = {
    "slack": 1.0,
    "discord": 5.0,
    "github": 10.0,
    "websearch": 5.0,
}


def _outbound_context(
    name: str, context: ConnectorContext | None
) -> ConnectorContext | None:
    """Merge the run's audit/retry context with this connector's own rate-limit default.

    A connector applies its standard :class:`RateLimiter` only when built with NO context;
    passing one (to carry the run's audit sink) would otherwise silence the self-throttle.
    This returns a context that keeps the caller's ``audit``/``retry`` AND restores the
    connector's default rate limiter when the caller did not set one.
    """
    if context is None:
        return None
    if context.rate_limiter is not None:
        return context
    from himmy.connectors.sdk import RateLimiter

    rate = _OUTBOUND_RATE.get(name)
    limiter = RateLimiter(rate=rate, window=1.0) if rate else None
    return ConnectorContext(
        audit=context.audit, rate_limiter=limiter, retry=context.retry
    )


def _direct_outbound(name: str, *, context: ConnectorContext | None = None) -> Connector:
    """Build an outbound connector directly (secrets-aware, registry-independent).

    ``context`` (when given) carries the run's audit sink so a tool call the connector
    makes lands on the run's spine; the connector's standard rate-limit default is
    preserved via :func:`_outbound_context`.
    """
    ctx = _outbound_context(name, context)
    if name == "slack":
        return SlackOutboundConnector(
            allowed_channel_ids=_split_env("HIMMY_SLACK_ALLOWED_CHANNELS") or None,
            default_channel_id=get_secret(SLACK_DEFAULT_CHANNEL_SECRET),
            context=ctx,
        )
    if name == "discord":
        return DiscordOutboundConnector(
            allowed_channel_ids=_split_env("HIMMY_DISCORD_ALLOWED_CHANNELS") or None,
            default_channel_id=get_secret(DISCORD_DEFAULT_CHANNEL_SECRET),
            context=ctx,
        )
    if name == "github":
        return GitHubOutboundConnector(
            allowed_repos=_split_env("HIMMY_GITHUB_ALLOWED_REPOS") or None,
            default_repo=get_secret(GITHUB_DEFAULT_REPO_SECRET),
            context=ctx,
        )
    if name == "websearch":
        return WebSearchConnector(context=ctx)
    raise ConnectorError(f"no outbound connector registered under {name!r}")


__all__ = [
    "ReadOnlyBackendError",
    "SecretSpec",
    "ConnectorDescriptor",
    "ConnectorStatus",
    "SecretStatus",
    "ConnectorTestResult",
    "ConnectorService",
    "CONNECTOR_CATALOG",
    "ensure_builtin_connectors_registered",
]
