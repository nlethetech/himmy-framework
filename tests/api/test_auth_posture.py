"""Fail-closed posture: an unauthenticated build refuses to start off-loopback.

The zero-config default principal is admin/all-tenants — safe only on a loopback bind.
``create_app`` must refuse to expose that unauthenticated admin surface to the network
unless the operator either configures auth or explicitly opts in.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.core.errors import HimmyError


def test_default_loopback_bind_starts(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    # No bind_host arg → loopback default → safe, must start.
    assert TestClient(create_app()).get("/health").status_code == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])  # noqa: S104
def test_non_loopback_bind_without_auth_refuses(host: str, monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("HIMMY_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    monkeypatch.delenv("HIMMY_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(HimmyError) as exc:
        create_app(bind_host=host)
    assert "non-loopback" in str(exc.value)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_without_auth_starts(host: str, monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    app = create_app(bind_host=host)
    assert TestClient(app).get("/health").status_code == 200


def test_non_loopback_bind_with_auth_starts(monkeypatch: Any) -> None:
    # An authenticator removes the exposure; off-loopback is then allowed.
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "secret-key")
    app = create_app(bind_host="0.0.0.0")  # noqa: S104
    assert app.state.authenticator is not None


def test_non_loopback_bind_explicit_opt_in_starts(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    monkeypatch.setenv("HIMMY_ALLOW_UNAUTHENTICATED", "1")
    app = create_app(bind_host="0.0.0.0")  # noqa: S104
    assert app.state.authenticator is None


def test_bind_host_read_from_env(monkeypatch: Any) -> None:
    # When bind_host is omitted, HIMMY_BIND_HOST drives the posture check.
    monkeypatch.delenv("HIMMY_INTERNAL_API_KEY", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    monkeypatch.delenv("HIMMY_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("HIMMY_BIND_HOST", "0.0.0.0")  # noqa: S104
    with pytest.raises(HimmyError):
        create_app()
