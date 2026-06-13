"""Toolkit networking guard: SSRF protection for tools that fetch arbitrary URLs.

The kernel's :mod:`himmy.services.tools.security` pins HTTP tools to a *configured*
base host — but ``web_fetch``/``http_request`` accept a URL the model itself supplies,
so they need a different check: the URL must be http/https, carry no embedded
credentials, and resolve to a public address (never loopback, private, link-local, or
otherwise reserved ranges). :func:`guard_url` enforces that before any request is made.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from himmy.services.tools.security import ToolSecurityError

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx/httpcore stay lazy)
    import httpx


def _host_allowed(host: str, allow_hosts: Collection[str]) -> bool:
    """Whether ``host`` matches the allow-list (exact or a subdomain of an entry)."""
    host = host.lower().rstrip(".")
    for entry in allow_hosts:
        allowed = entry.lower().strip().lstrip("*.").rstrip(".")
        if allowed and (host == allowed or host.endswith("." + allowed)):
            return True
    return False


def _is_blocked_ip(ip: str) -> bool:
    """True when ``ip`` is loopback/private/link-local/reserved/multicast/unspecified."""
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def guard_url(
    url: str,
    *,
    allow_private: bool = False,
    allow_hosts: Collection[str] | None = None,
    resolve: bool = True,
) -> str:
    """Validate ``url`` for outbound fetching; return it unchanged or raise.

    Raises :class:`ToolSecurityError` for a non-http(s) scheme, embedded userinfo
    credentials, a missing host, or (unless ``allow_private``) a host that resolves
    to a private/loopback/reserved address. DNS resolution is attempted so a public
    name that points at an internal IP is still rejected.

    When ``allow_hosts`` is provided (egress allow-list, WS3.3), the host must match
    an entry (exact or a subdomain) — everything else is denied, for restricted /
    air-gapped deployments. The allow-list narrows on top of the public-IP check.

    ``resolve=False`` runs only the cheap, offline checks (scheme, no embedded
    credentials, host present, allow-list) and skips the DNS/IP step — for callers that
    enforce the public-IP rule at connect time through a pinned transport (so the same
    resolution isn't done twice, and a DNS hiccup doesn't mask the real connect).
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolSecurityError(
            f"unsupported URL scheme {parts.scheme!r} (expected http/https)"
        )
    if parts.username or parts.password:
        raise ToolSecurityError("URL must not contain embedded credentials")
    host = parts.hostname
    if not host:
        raise ToolSecurityError("URL has no host")
    if allow_hosts is not None and not _host_allowed(host, allow_hosts):
        raise ToolSecurityError(f"host {host!r} is not in the egress allow-list")
    if allow_private or not resolve:
        return url

    # A literal IP host is checked directly; a name is resolved and every
    # returned address must be public.
    _validate_resolved_ips(host, parts.port)
    return url


def _validate_resolved_ips(host: str, port: int | None) -> list[str]:
    """Resolve ``host`` and require every address be public; return the addresses.

    A literal-IP host is checked directly (no DNS); a name is resolved via
    ``getaddrinfo`` and EVERY returned address must pass :func:`_is_blocked_ip`.
    Raises :class:`ToolSecurityError` on an unresolvable host or a non-public hit.
    """
    try:
        ipaddress.ip_address(host)
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port or None)
        except OSError as exc:
            raise ToolSecurityError(f"could not resolve host {host!r}") from exc
        candidates = [str(info[4][0]) for info in infos]

    for ip in candidates:
        if _is_blocked_ip(ip):
            raise ToolSecurityError(
                f"host {host!r} resolves to a non-public address ({ip})"
            )
    return candidates


def _pin_connect_target(
    host: str, port: int | None, allow_hosts: Collection[str] | None
) -> str:
    """Resolve + vet a connect target at connect time; return the literal IP to dial.

    Shared by the sync and async pinning backends (installed only when ``allow_private``
    is False). A literal-IP host is still run through :func:`_is_blocked_ip` — so a
    request whose URL was rewritten to (or supplied as) ``127.0.0.1`` / ``169.254.169.254``
    / an RFC1918 literal is refused, not trusted. A name is allow-list-checked then
    resolved, and the FIRST vetted address is the one dialed (check==connect, no
    rebinding window). Raises :class:`ToolSecurityError` on any non-public hit.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if allow_hosts is not None and not _host_allowed(host, allow_hosts):
            raise ToolSecurityError(
                f"host {host!r} is not in the egress allow-list"
            ) from None
        return _validate_resolved_ips(host, port)[0]
    # A literal IP: still enforce the public-only rule (the backend is only installed
    # when allow_private is False).
    if _is_blocked_ip(host):
        raise ToolSecurityError(
            f"host {host!r} resolves to a non-public address ({host})"
        )
    return host


def build_pinned_transport(
    *,
    allow_private: bool = False,
    allow_hosts: Collection[str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> httpx.BaseTransport:
    """An httpx transport that re-validates + PINS the resolved IP at connect time.

    :func:`guard_url` resolves DNS and vets the IPs, but a brand-new ``httpx.Client``
    re-resolves at connect time — a DNS-rebinding (check-to-use / TOCTOU) window where
    an attacker who controls a low-TTL name answers the guard lookup with a public IP
    and the connect lookup with ``169.254.169.254`` / loopback / RFC1918. This wraps
    the connection layer so the host is resolved ONCE, every candidate address is
    re-checked with the same :func:`_is_blocked_ip` rule, and the socket is dialed at
    the validated literal IP (the original Host header + TLS SNI are preserved by
    httpcore, which derives them from the request, not the connect target). Check and
    connect therefore share one resolution: the IP that is vetted is the IP that is
    dialed. With ``allow_private`` the wrapper is a passthrough (no pinning needed).

    ``transport`` is a test seam: when given, it is returned as-is so suites can inject
    a mock transport instead of opening real sockets.
    """
    import httpx

    if transport is not None:
        return transport
    http_transport = httpx.HTTPTransport()
    if allow_private:
        return http_transport

    from httpcore._backends.sync import SyncBackend

    base_backend = SyncBackend()

    class _PinningBackend(SyncBackend):  # type: ignore[misc]
        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[Any] | None = None,
        ) -> Any:
            # Resolve + vet here, at connect time, and dial the literal IP we just
            # validated — closing the gap where httpx would re-resolve independently.
            pinned = _pin_connect_target(host, port, allow_hosts)
            return base_backend.connect_tcp(
                pinned,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

    # Swap the pool's network backend (httpcore's documented custom-backend hook)
    # so the actual TCP connect dials the IP we vetted, leaving httpx's TLS/proxy
    # config — and the Host header + SNI derived from the request — intact.
    http_transport._pool._network_backend = _PinningBackend()
    return http_transport


def build_async_pinned_transport(
    *,
    allow_private: bool = False,
    allow_hosts: Collection[str] | None = None,
    transport: Any = None,
) -> Any:
    """The async twin of :func:`build_pinned_transport` for ``httpx.AsyncClient``.

    The generic OpenAPI/REST connector dispatches on the event loop through a shared
    ``httpx.AsyncClient``, so it needs the SAME DNS-rebinding (TOCTOU) defense the sync
    connectors get: resolve the host ONCE at connect time, re-check every candidate IP,
    dial the validated literal IP, and preserve the Host header + TLS SNI (derived by
    httpcore from the request, not the connect target). With ``allow_private`` the
    wrapper is a passthrough. ``transport`` is a test seam: a supplied transport (e.g. a
    ``MockTransport``) is returned as-is so suites never open real sockets.
    """
    import httpx

    if transport is not None:
        return transport
    http_transport = httpx.AsyncHTTPTransport()
    if allow_private:
        return http_transport

    from httpcore._backends.anyio import AnyIOBackend

    base_backend = AnyIOBackend()

    class _AsyncPinningBackend(AnyIOBackend):  # type: ignore[misc]
        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[Any] | None = None,
        ) -> Any:
            pinned = _pin_connect_target(host, port, allow_hosts)
            return await base_backend.connect_tcp(
                pinned,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

    http_transport._pool._network_backend = _AsyncPinningBackend()
    return http_transport


__all__ = ["guard_url", "build_pinned_transport", "build_async_pinned_transport"]
