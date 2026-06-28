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


def _valid_state() -> str:
    """Mint a currently-valid, signed OAuth state (the callback now REJECTS a missing
    or forged state as login-CSRF, so tests must present a real one — same token the
    ``auth_url`` consent step issues)."""
    import time

    return g._sign_state(g._secrets.token_urlsafe(8), int(time.time()))


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
    assert "gmail.compose" in url  # drafts.create needs compose, not just send

    st = asyncio.run(
        g.exchange_code("authcode", "http://127.0.0.1:8800/cb", state=_valid_state())
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


def test_gmail_get_reply_and_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gmail_get returns the full body + Message-ID; gmail_send threads a reply / saves a draft."""
    import asyncio
    import json

    monkeypatch.setenv("HIMMY_SECRETS", "file")
    monkeypatch.setenv("HIMMY_SECRETS_DIR", str(tmp_path / "secrets"))
    secrets_mod.configure_secrets(None)

    captured: list[tuple[str, dict]] = []

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
        if request.method == "GET" and "/messages/m1" in url:
            data = (
                base64.urlsafe_b64encode(b"Hello, can you join Tuesday?")
                .decode()
                .rstrip("=")
            )
            return httpx.Response(
                200,
                json={
                    "id": "m1",
                    "threadId": "t1",
                    "snippet": "Hello, can you join",
                    "payload": {
                        "mimeType": "text/plain",
                        "headers": [
                            {"name": "From", "value": "Acme <a@acme.com>"},
                            {"name": "To", "value": "me@example.com"},
                            {"name": "Subject", "value": "Project kickoff"},
                            {"name": "Date", "value": "Mon, 1 Jun 2026"},
                            {"name": "Message-ID", "value": "<orig123@acme.com>"},
                        ],
                        "body": {"data": data},
                    },
                },
            )
        if request.method == "POST" and url.startswith(
            f"{g._GMAIL_BASE}/messages/send"
        ):
            captured.append(("send", json.loads(request.content)))
            return httpx.Response(200, json={"id": "sent1", "threadId": "t1"})
        if request.method == "POST" and url.startswith(f"{g._GMAIL_BASE}/drafts"):
            captured.append(("draft", json.loads(request.content)))
            return httpx.Response(
                200, json={"id": "d1", "message": {"id": "m2", "threadId": "t1"}}
            )
        return httpx.Response(404, json={"error": {"message": f"unexpected {url}"}})

    g._TRANSPORT = httpx.MockTransport(handler)
    try:
        g.set_client("cid", "csecret")
        asyncio.run(g.exchange_code("authcode", "http://cb", state=_valid_state()))

        # full fetch carries the body + the Message-ID header needed to thread
        m = asyncio.run(g.gmail_get("m1"))
        assert m.message_id_header == "<orig123@acme.com>"
        assert "join Tuesday" in m.body
        assert m.sender == "Acme <a@acme.com>"

        # a threaded reply: posts to messages/send with the threadId + In-Reply-To header
        r = asyncio.run(
            g.gmail_send(
                "a@acme.com",
                "Re: Project kickoff",
                "Yes, I'll join.",
                thread_id="t1",
                in_reply_to=m.message_id_header,
            )
        )
        assert r.ok and r.id == "sent1"
        kind, payload = captured[-1]
        assert kind == "send" and payload.get("threadId") == "t1"
        raw = base64.urlsafe_b64decode(
            payload["raw"] + "=" * (-len(payload["raw"]) % 4)
        ).decode()
        assert (
            "In-Reply-To: <orig123@acme.com>" in raw
            and "References: <orig123@acme.com>" in raw
        )

        # a draft goes to the drafts endpoint and never sends
        d = asyncio.run(
            g.gmail_send(
                "a@acme.com",
                "Re: Project kickoff",
                "Draft only.",
                thread_id="t1",
                in_reply_to=m.message_id_header,
                draft=True,
            )
        )
        assert d.ok and d.id == "d1" and "draft" in d.detail
        kind2, payload2 = captured[-1]
        assert kind2 == "draft" and payload2["message"].get("threadId") == "t1"
    finally:
        g._TRANSPORT = None
        secrets_mod.configure_secrets(None)
