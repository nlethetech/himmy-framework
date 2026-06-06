"""Studio Connections: connect Email/Telegram/Web so agents can act for you.

A connection is a friendly grouping of the ``HIMMY_*`` env/secret names a tool pack
already reads. The Studio UI sets the fields here; values are written to the active
*writable* secrets backend (macOS keychain or a local file store — see
:func:`himmy.config.secrets.get_writable_provider`) and never returned to the GUI.

Non-secret fields are also pushed into ``os.environ`` (:func:`apply_connections_to_env`)
so the env-reading :class:`~himmy.toolkit.config.ToolkitConfig` picks them up without a
restart. "Test" actions reuse the real tool plumbing (SMTP login, Telegram ``getMe``,
a search ping) so a green check means the credential actually works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from himmy.config.secrets import (
    get_secret,
    get_writable_provider,
)


@dataclass(frozen=True)
class ConnectionField:
    """One field of a connection → the underlying secret/env name it maps to."""

    name: str  # GUI field id
    secret_name: str  # underlying HIMMY_* store key
    label: str
    secret: bool = False  # True → value never returned; stored as a secret
    required: bool = True
    placeholder: str = ""
    kind: str = "text"  # text | password | number | bool | enum
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionType:
    """A connectable account type: its fields + an optional live test."""

    type: str
    title: str
    description: str
    hue: str  # CSS hue token name (email/telegram/web/…)
    fields: tuple[ConnectionField, ...]


# ---- the registry -------------------------------------------------------

CONNECTION_TYPES: dict[str, ConnectionType] = {
    "email": ConnectionType(
        type="email",
        title="Email",
        description="Send email via your SMTP server.",
        hue="email",
        fields=(
            ConnectionField(
                "smtp_host",
                "HIMMY_SMTP_HOST",
                "SMTP host",
                placeholder="smtp.gmail.com",
            ),
            ConnectionField(
                "smtp_port",
                "HIMMY_SMTP_PORT",
                "Port",
                required=False,
                placeholder="587",
                kind="number",
            ),
            ConnectionField(
                "smtp_user", "HIMMY_SMTP_USER", "Username", placeholder="you@gmail.com"
            ),
            ConnectionField(
                "smtp_password",
                "HIMMY_SMTP_PASSWORD",
                "Password",
                secret=True,
                kind="password",
            ),
            ConnectionField(
                "smtp_from",
                "HIMMY_SMTP_FROM",
                "From address",
                required=False,
                placeholder="you@gmail.com",
            ),
            ConnectionField(
                "smtp_use_tls",
                "HIMMY_SMTP_USE_TLS",
                "Use TLS",
                required=False,
                kind="bool",
            ),
        ),
    ),
    "telegram": ConnectionType(
        type="telegram",
        title="Telegram",
        description="Message a Telegram chat through your bot.",
        hue="telegram",
        fields=(
            ConnectionField(
                "bot_token",
                "HIMMY_TELEGRAM_BOT_TOKEN",
                "Bot token",
                secret=True,
                kind="password",
                placeholder="123456:ABC-...",
            ),
            ConnectionField(
                "chat_id",
                "HIMMY_TELEGRAM_CHAT_ID",
                "Default chat id",
                required=False,
                placeholder="987654321",
            ),
        ),
    ),
    "web_search": ConnectionType(
        type="web_search",
        title="Web search",
        description="Search the web (DuckDuckGo is keyless; Tavily/Brave need a key).",
        hue="web",
        fields=(
            ConnectionField(
                "backend",
                "HIMMY_SEARCH_BACKEND",
                "Backend",
                required=False,
                kind="enum",
                options=("duckduckgo", "tavily", "brave"),
            ),
            ConnectionField(
                "api_key",
                "HIMMY_SEARCH_API_KEY",
                "API key",
                secret=True,
                required=False,
                kind="password",
            ),
        ),
    ),
}


# ---- response models ----------------------------------------------------


class ConnectionFieldStatus(BaseModel):
    name: str
    label: str
    present: bool
    secret: bool
    required: bool
    kind: str
    placeholder: str = ""
    options: list[str] = []
    masked_hint: str | None = None  # never the secret itself


class ConnectionStatus(BaseModel):
    type: str
    title: str
    description: str
    hue: str
    configured: bool
    writable: bool
    fields: list[ConnectionFieldStatus]


class ConnectionTestResult(BaseModel):
    ok: bool
    detail: str
    checked: str


class ReadOnlyBackendError(RuntimeError):
    """Raised when set/delete is attempted but no writable secrets backend is active."""


# ---- helpers ------------------------------------------------------------


def _present(name: str) -> str | None:
    """The current value of a field's secret/env (secret-aware), or None."""
    return get_secret(name)


def _masked(field_def: ConnectionField, value: str | None) -> str | None:
    if not value:
        return None
    if field_def.secret or field_def.kind == "password":
        return "••••"
    if "@" in value:  # email-ish: show first char + domain
        local, _, domain = value.partition("@")
        return f"{local[:1]}•••@{domain}"
    return value if len(value) <= 24 else value[:22] + "…"


def _status(ct: ConnectionType) -> ConnectionStatus:
    writable = get_writable_provider() is not None
    fields: list[ConnectionFieldStatus] = []
    configured = True
    for f in ct.fields:
        val = _present(f.secret_name)
        present = val is not None and val != ""
        if f.required and not present:
            configured = False
        fields.append(
            ConnectionFieldStatus(
                name=f.name,
                label=f.label,
                present=present,
                secret=f.secret,
                required=f.required,
                kind=f.kind,
                placeholder=f.placeholder,
                options=list(f.options),
                masked_hint=_masked(f, val),
            )
        )
    return ConnectionStatus(
        type=ct.type,
        title=ct.title,
        description=ct.description,
        hue=ct.hue,
        configured=configured,
        writable=writable,
        fields=fields,
    )


# ---- service ------------------------------------------------------------


def list_connections() -> list[ConnectionStatus]:
    return [_status(ct) for ct in CONNECTION_TYPES.values()]


def get_connection(ctype: str) -> ConnectionStatus | None:
    ct = CONNECTION_TYPES.get(ctype)
    return _status(ct) if ct else None


def set_connection(ctype: str, fields: dict[str, Any]) -> ConnectionStatus:
    """Persist the given fields to the writable backend; refresh the process env."""
    ct = CONNECTION_TYPES.get(ctype)
    if ct is None:
        raise KeyError(ctype)
    writable = get_writable_provider()
    if writable is None:
        raise ReadOnlyBackendError(
            "the active secrets backend is read-only; set HIMMY_SECRETS=keychain or file"
        )
    by_name = {f.name: f for f in ct.fields}
    for name, raw in fields.items():
        f = by_name.get(name)
        if f is None:
            continue  # ignore unknown fields rather than write arbitrary keys
        value = "" if raw is None else str(raw)
        if value == "":
            writable.delete(f.secret_name)
        else:
            writable.set(f.secret_name, value)
    apply_connections_to_env()
    return _status(ct)


def delete_connection(ctype: str) -> ConnectionStatus:
    ct = CONNECTION_TYPES.get(ctype)
    if ct is None:
        raise KeyError(ctype)
    writable = get_writable_provider()
    if writable is None:
        raise ReadOnlyBackendError("the active secrets backend is read-only")
    for f in ct.fields:
        writable.delete(f.secret_name)
        os.environ.pop(f.secret_name, None)
    return _status(ct)


async def test_connection(ctype: str) -> ConnectionTestResult:
    """Live-validate a connection by exercising the real tool plumbing."""
    import asyncio

    from himmy.toolkit.config import ToolkitConfig

    apply_connections_to_env()
    cfg = ToolkitConfig.from_env()

    if ctype == "email":
        from himmy.toolkit.comms import smtp_login_check

        try:
            await asyncio.to_thread(smtp_login_check, cfg)
            return ConnectionTestResult(
                ok=True, detail="SMTP login ok", checked="smtp login"
            )
        except Exception as exc:  # noqa: BLE001 - surface a sanitized message
            return ConnectionTestResult(
                ok=False, detail=_sanitize(str(exc), cfg), checked="smtp login"
            )

    if ctype == "telegram":
        from himmy.toolkit.telegram import TelegramClient

        if not cfg.telegram_bot_token:
            return ConnectionTestResult(
                ok=False, detail="no bot token", checked="getMe"
            )
        try:
            me = await TelegramClient(cfg.telegram_bot_token).get_me()
            who = me.get("username") or me.get("first_name") or "bot"
            return ConnectionTestResult(ok=True, detail=f"@{who}", checked="getMe")
        except Exception as exc:  # noqa: BLE001
            return ConnectionTestResult(
                ok=False, detail=_sanitize(str(exc), cfg), checked="getMe"
            )

    if ctype == "web_search":
        if cfg.search_backend == "duckduckgo":
            return ConnectionTestResult(
                ok=True, detail="DuckDuckGo (keyless)", checked="search"
            )
        from himmy.toolkit.web import (
            _httpx_caller,
            build_search_backend,
            ddg_post_html,
        )

        def _ping() -> int:
            backend = build_search_backend(
                cfg, _httpx_caller, lambda q: ddg_post_html(q, cfg.http_timeout)
            )
            return len(backend.search("himmy connectivity test", 1))

        try:
            n = await asyncio.to_thread(_ping)
            return ConnectionTestResult(
                ok=True, detail=f"{cfg.search_backend}: {n} result(s)", checked="search"
            )
        except Exception as exc:  # noqa: BLE001
            return ConnectionTestResult(
                ok=False, detail=_sanitize(str(exc), cfg), checked="search"
            )

    raise KeyError(ctype)


def _sanitize(message: str, cfg: Any) -> str:
    """Strip any secret value that an error string might have echoed back."""
    out = message
    for secret in (cfg.smtp_password, cfg.telegram_bot_token, cfg.search_api_key):
        if secret:
            out = out.replace(secret, "••••")
    return out[:200]


_ENV_APPLIED = False


def apply_connections_to_env() -> None:
    """Load non-secret connection fields from the writable backend into ``os.environ``.

    Secret fields resolve through ``get_secret`` already; non-secret fields (host,
    port, backend, …) are read from the env by the tool config, so they must be
    materialized here. Idempotent; safe to call at startup and after each ``set``.
    """
    global _ENV_APPLIED
    writable = get_writable_provider()
    if writable is None:
        return
    for ct in CONNECTION_TYPES.values():
        for f in ct.fields:
            if f.secret:
                continue
            value = writable.get(f.secret_name)
            if value is not None and value != "":
                # The stored connection is authoritative once a user sets it.
                os.environ[f.secret_name] = value
    _ENV_APPLIED = True


__all__ = [
    "ConnectionType",
    "ConnectionField",
    "ConnectionStatus",
    "ConnectionFieldStatus",
    "ConnectionTestResult",
    "ReadOnlyBackendError",
    "CONNECTION_TYPES",
    "list_connections",
    "get_connection",
    "set_connection",
    "delete_connection",
    "test_connection",
    "apply_connections_to_env",
]
