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
from urllib.parse import urlsplit

from himmy.services.tools.security import ToolSecurityError


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


def guard_url(url: str, *, allow_private: bool = False) -> str:
    """Validate ``url`` for outbound fetching; return it unchanged or raise.

    Raises :class:`ToolSecurityError` for a non-http(s) scheme, embedded userinfo
    credentials, a missing host, or (unless ``allow_private``) a host that resolves
    to a private/loopback/reserved address. DNS resolution is attempted so a public
    name that points at an internal IP is still rejected.
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
    if allow_private:
        return url

    # A literal IP host is checked directly; a name is resolved and every
    # returned address must be public.
    try:
        ipaddress.ip_address(host)
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parts.port or None)
        except OSError as exc:
            raise ToolSecurityError(f"could not resolve host {host!r}") from exc
        candidates = [str(info[4][0]) for info in infos]

    for ip in candidates:
        if _is_blocked_ip(ip):
            raise ToolSecurityError(
                f"host {host!r} resolves to a non-public address ({ip})"
            )
    return url


__all__ = ["guard_url"]
