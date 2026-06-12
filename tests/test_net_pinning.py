"""DNS-rebinding (TOCTOU) defense: build_pinned_transport re-validates AND pins the
resolved IP at connect time, so a low-TTL name that passes guard_url cannot rebind to
an internal address before httpx connects. Offline (resolution is monkeypatched).
"""

from __future__ import annotations

import socket

import pytest

from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit import _net
from himmy.toolkit._net import build_pinned_transport, guard_url


def _backend(transport):
    """The pinning network backend swapped into the transport's connection pool."""
    return transport._pool._network_backend


def test_passthrough_when_a_transport_is_injected() -> None:
    """An injected (mock) transport is returned as-is — the test seam is preserved."""
    sentinel = object()
    assert build_pinned_transport(transport=sentinel) is sentinel  # type: ignore[arg-type]


def test_allow_private_does_not_pin() -> None:
    """With allow_private the wrapper is a plain transport (no connect-time guard)."""
    transport = build_pinned_transport(allow_private=True)
    # No pinning backend is installed: the default backend stays in place.
    assert _backend(transport).__class__.__name__ != "_PinningBackend"


def _patch_real_connect(monkeypatch, dialed: list[tuple[str, int]]) -> None:
    """Replace the real TCP connect (the base SyncBackend) with a recorder."""
    from httpcore._backends.sync import SyncBackend

    monkeypatch.setattr(
        SyncBackend,
        "connect_tcp",
        lambda self, host, port, **kw: dialed.append((host, port)) or object(),
    )


def test_connect_dials_the_validated_literal_ip(monkeypatch) -> None:
    """A name resolving to a public IP is dialed at THAT literal IP (resolve-and-pin)."""
    dialed: list[tuple[str, int]] = []

    # The name resolves to a single public address.
    monkeypatch.setattr(
        _net.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))
        ],
    )

    transport = build_pinned_transport()
    backend = _backend(transport)
    _patch_real_connect(monkeypatch, dialed)

    backend.connect_tcp("example.com", 80)
    # The socket targets the literal vetted IP, not the hostname (which httpx would
    # otherwise re-resolve independently).
    assert dialed == [("93.184.216.34", 80)]


def test_connect_refuses_rebind_to_internal_ip(monkeypatch) -> None:
    """A name that (re)resolves to an internal IP at connect time is refused there."""
    # guard_url at check time saw a public IP...
    monkeypatch.setattr(
        _net.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))
        ],
    )
    assert guard_url("http://rebind.example/") == "http://rebind.example/"

    transport = build_pinned_transport()
    backend = _backend(transport)

    # ...but by connect time the attacker's DNS now answers with the metadata IP.
    monkeypatch.setattr(
        _net.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 80))
        ],
    )
    with pytest.raises(ToolSecurityError):
        backend.connect_tcp("rebind.example", 80)


def test_connect_enforces_egress_allow_list(monkeypatch) -> None:
    """A host outside the egress allow-list is refused at connect time too."""
    monkeypatch.setattr(
        _net.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))
        ],
    )
    transport = build_pinned_transport(allow_hosts={"allowed.example"})
    backend = _backend(transport)
    with pytest.raises(ToolSecurityError):
        backend.connect_tcp("evil.example", 80)


def test_connect_passes_through_literal_ip_host(monkeypatch) -> None:
    """A literal-IP host needs no DNS and is dialed directly (no getaddrinfo call)."""
    dialed: list[tuple[str, int]] = []

    def boom(*a, **k):  # getaddrinfo must NOT be called for a literal IP
        raise AssertionError("literal IP host must not be resolved")

    monkeypatch.setattr(_net.socket, "getaddrinfo", boom)
    transport = build_pinned_transport()
    backend = _backend(transport)
    _patch_real_connect(monkeypatch, dialed)
    backend.connect_tcp("93.184.216.34", 443)
    assert dialed == [("93.184.216.34", 443)]
