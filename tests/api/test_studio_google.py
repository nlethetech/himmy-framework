"""Tests for Studio Google (OAuth + Gmail + Calendar) with a mocked transport."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_google as g
from himmy.api.app import create_app
from himmy.config import secrets as secrets_mod


@pytest.fixture
def connected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A writable file-secrets backend + a mock Google transport."""
    monkeypatch.setenv("HIMMY_SECRETS", "file")
    monkeypatch.setenv("HIMMY_SECRETS_DIR", str(tmp_path / "secrets"))
    secrets_mod.configure_secrets(None)  # force rebuild from env

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(g._TOKEN_URL):
            return httpx.Response(
                200,
                json={
                    "access_token": "atok",
                    "refresh_token": "rtok",
                    "expires_in": 3600,
                },
            )
        if url.startswith(g._USERINFO_URL):
            return httpx.Response(200, json={"email": "me@example.com"})
        if url.startswith(f"{g._GMAIL_BASE}/messages/send"):
            return httpx.Response(200, json={"id": "sent1"})
        if "/messages/" in url:
            return httpx.Response(
                200,
                json={
                    "id": "m1",
                    "threadId": "t1",
                    "snippet": "hi there",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "a@b.com"},
                            {"name": "Subject", "value": "Hello"},
                            {"name": "Date", "value": "Mon, 1 Jun 2026"},
                        ]
                    },
                },
            )
        if url.startswith(f"{g._GMAIL_BASE}/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        if request.method == "POST" and url.startswith(g._CAL_BASE):
            return httpx.Response(
                200,
                json={
                    "id": "e1",
                    "summary": "Standup",
                    "start": {"dateTime": "2026-06-09T09:00:00Z"},
                    "end": {"dateTime": "2026-06-09T09:30:00Z"},
                    "htmlLink": "https://cal/e1",
                },
            )
        if url.startswith(g._CAL_BASE):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "e1",
                            "summary": "Standup",
                            "start": {"dateTime": "2026-06-09T09:00:00Z"},
                            "end": {"dateTime": "2026-06-09T09:30:00Z"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": {"message": "unexpected"}})

    g._TRANSPORT = httpx.MockTransport(handler)
    yield
    g._TRANSPORT = None
    secrets_mod.configure_secrets(None)


def test_oauth_and_apis(connected: None) -> None:
    import asyncio

    g.set_client("cid", "csecret")
    st = g.status()
    assert st.configured is True and st.connected is False

    url = g.auth_url("http://127.0.0.1:8800/api/studio/google/callback")
    assert "accounts.google.com" in url and "gmail.send" in url

    st = asyncio.run(
        g.exchange_code("authcode", "http://127.0.0.1:8800/cb", state=None)
    )
    assert st.connected is True
    assert st.email == "me@example.com"

    msgs = asyncio.run(g.gmail_list(10))
    assert msgs[0].subject == "Hello" and msgs[0].sender == "a@b.com"

    sent = asyncio.run(g.gmail_send("x@y.com", "Hi", "Body"))
    assert sent.ok is True and sent.id == "sent1"
    # the send payload is a base64url-encoded RFC822 message
    assert base64.urlsafe_b64encode(b"x")

    events = asyncio.run(g.calendar_list())
    assert events[0].summary == "Standup"

    created = asyncio.run(
        g.calendar_create("Standup", "2026-06-09T09:00:00Z", "2026-06-09T09:30:00Z")
    )
    assert created.id == "e1"

    st = g.disconnect()
    assert st.connected is False and st.configured is True


def test_routes_status_and_guard(connected: None) -> None:
    c = TestClient(create_app())
    r = c.get("/api/studio/google")
    assert r.status_code == 200
    assert r.json()["configured"] is False  # nothing set yet
    # auth-url before client creds → 409
    assert c.get("/api/studio/google/auth-url").status_code == 409
    # set client, then auth-url works
    assert (
        c.put(
            "/api/studio/google/client",
            json={"client_id": "cid", "client_secret": "sec"},
        ).status_code
        == 200
    )
    body = c.get("/api/studio/google/auth-url").json()
    assert "accounts.google.com" in body["url"]


def test_gmail_requires_connection(connected: None) -> None:
    c = TestClient(create_app())
    # configured client but no tokens → 409 not connected
    c.put(
        "/api/studio/google/client",
        json={"client_id": "cid", "client_secret": "sec"},
    )
    assert c.get("/api/studio/google/gmail").status_code == 409
