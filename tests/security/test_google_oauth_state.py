"""Security regression: the Google OAuth callback must ENFORCE the CSRF ``state``.

Finding #7 (login CSRF / mailbox grafting): ``exchange_code`` previously only
*discarded* a known state and never rejected a missing/unknown/forged one, so an
attacker could complete the flow with their own auth code and connect THEIR
mailbox to the victim Studio. These tests fail against the old behaviour (where
the code was exchanged + a token stored) and pass once an unverifiable state is
rejected before any token exchange.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from himmy.api import studio_google as g
from himmy.config import secrets as secrets_mod


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Writable file secrets + a configured client + a token-minting transport.

    The transport returns a fresh token for ANY exchange, so if the state guard
    is broken the attacker's code WOULD store a refresh token — exactly what we
    assert never happens for an unverifiable state.
    """
    monkeypatch.setenv("HIMMY_SECRETS", "file")
    monkeypatch.setenv("HIMMY_SECRETS_DIR", str(tmp_path / "secrets"))
    secrets_mod.configure_secrets(None)
    g.set_client("cid", "csecret")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(g._TOKEN_URL):
            return httpx.Response(
                200,
                json={
                    "access_token": "attacker_atok",
                    "refresh_token": "attacker_rtok",
                    "expires_in": 3600,
                },
            )
        if url.startswith(g._USERINFO_URL):
            return httpx.Response(200, json={"email": "attacker@evil.example"})
        return httpx.Response(404, json={"error": {"message": "unexpected"}})

    g._TRANSPORT = httpx.MockTransport(handler)
    yield
    g._TRANSPORT = None
    g._PENDING_STATES.clear()
    secrets_mod.configure_secrets(None)


def _no_token_stored() -> None:
    assert secrets_mod.get_secret(g.REFRESH_TOKEN) is None
    assert secrets_mod.get_secret(g.ACCESS_TOKEN) is None
    assert secrets_mod.get_secret(g.ACCOUNT_EMAIL) is None


def test_missing_state_is_rejected(configured: None) -> None:
    with pytest.raises(g.GoogleError):
        asyncio.run(g.exchange_code("attacker_code", "http://localhost/cb", None))
    _no_token_stored()


def test_forged_state_is_rejected(configured: None) -> None:
    # A plausible-looking but never-issued / wrongly-signed state.
    with pytest.raises(g.GoogleError):
        asyncio.run(
            g.exchange_code("attacker_code", "http://localhost/cb", "nonce.0.deadbeef")
        )
    _no_token_stored()


def test_unknown_opaque_state_is_rejected(configured: None) -> None:
    # Old behaviour silently accepted this and stored the attacker's token.
    with pytest.raises(g.GoogleError):
        asyncio.run(
            g.exchange_code(
                "attacker_code", "http://localhost/cb", "totally-unknown-state"
            )
        )
    _no_token_stored()


def test_pending_state_proceeds(configured: None) -> None:
    # A state issued by auth_url is pending in-memory and must succeed.
    url = g.auth_url("http://localhost/cb")
    state = parse_qs(urlparse(url).query)["state"][0]
    st = asyncio.run(g.exchange_code("good_code", "http://localhost/cb", state))
    assert st.connected
    assert secrets_mod.get_secret(g.REFRESH_TOKEN) == "attacker_rtok"
    # The pending state is single-use (consumed on success).
    assert state not in g._PENDING_STATES


def test_signed_state_survives_restart(configured: None) -> None:
    # Restart-resilience: a validly-signed, unexpired state works even if it is
    # NOT in the in-memory pending set (simulating a process restart).
    state = g._sign_state("noncexyz", int(time.time()))
    assert state not in g._PENDING_STATES
    st = asyncio.run(g.exchange_code("good_code", "http://localhost/cb", state))
    assert st.connected


def test_expired_signed_state_is_rejected(configured: None) -> None:
    old = int(time.time()) - (g._STATE_TTL_SECONDS + 60)
    state = g._sign_state("noncexyz", old)
    with pytest.raises(g.GoogleError):
        asyncio.run(g.exchange_code("good_code", "http://localhost/cb", state))
    _no_token_stored()
