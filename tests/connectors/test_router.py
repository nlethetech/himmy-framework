"""T1.4(b) — the generic inbound-connector FastAPI router (build_connector_router).

``build_connector_router`` generalizes ``build_webhook_router`` over any
:class:`~himmy.connectors.sdk.InboundChannelConnector`. These tests mount each inbound
connector on a standalone FastAPI app and drive it over HTTP — entirely offline — to prove
the ONE router preserves each connector's default-deny gate AND returns the provider-shaped
response body:

* the generic webhook: a correctly-signed payload runs the agent and returns the reply;
  a forged/unsigned one is 401 and the agent never runs;
* Slack Events: the signed ``url_verification`` handshake echoes the challenge; a bad
  signature is 401;
* Discord Interactions: a signed PING returns the ``{"type": 1}`` PONG body (the
  provider-response mapper), a bad signature is 401.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from himmy.connectors.router import build_connector_router


def _app(connector: Any, path: str) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_connector_router(connector, path=path))
    return app


# ----------------------------------------------------------------- generic webhook
def _webhook_connector() -> tuple[Any, list[str]]:
    from himmy.connectors.webhook import WebhookInboundConnector

    seen: list[str] = []

    async def handler(sender_id: str, text: str) -> str:
        seen.append(f"{sender_id}:{text}")
        return f"ran: {text}"

    conn = WebhookInboundConnector(
        handler, signing_secret="whsec_test", allowed_sources=["ci"]
    )
    return conn, seen


def test_webhook_signed_runs_agent_via_generic_router() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _webhook_connector()
    body = json.dumps({"source": "ci", "text": "go", "id": "e1"}).encode()
    sig = "sha256=" + hmac.new(b"whsec_test", body, hashlib.sha256).hexdigest()
    with TestClient(_app(conn, "/hook")) as client:
        resp = client.post("/hook", content=body, headers={"X-Himmy-Signature": sig})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "ran: go"
    assert seen == ["ci:go"]


def test_webhook_unsigned_is_401_and_agent_never_runs() -> None:
    from fastapi.testclient import TestClient

    conn, seen = _webhook_connector()
    body = json.dumps({"source": "ci", "text": "go", "id": "e1"}).encode()
    with TestClient(_app(conn, "/hook")) as client:
        resp = client.post("/hook", content=body)
    assert resp.status_code == 401
    assert seen == []


# ----------------------------------------------------------------- Slack events
def test_slack_url_verification_echoes_challenge() -> None:
    from fastapi.testclient import TestClient

    from himmy.connectors.slack import SlackEventsConnector

    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - not run
        return "x"

    secret = "slacksigningsecret"
    conn = SlackEventsConnector(handler, signing_secret=secret)
    ts = str(int(time.time()))
    body = json.dumps(
        {"type": "url_verification", "challenge": "abc123"}
    ).encode()
    basestring = b"v0:" + ts.encode() + b":" + body
    sig = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    with TestClient(_app(conn, "/slack")) as client:
        resp = client.post(
            "/slack",
            content=body,
            headers={
                "X-Slack-Request-Timestamp": ts,
                "X-Slack-Signature": sig,
            },
        )
    assert resp.status_code == 200
    assert resp.json()["challenge"] == "abc123"


def test_slack_bad_signature_is_401() -> None:
    from fastapi.testclient import TestClient

    from himmy.connectors.slack import SlackEventsConnector

    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - not run
        return "x"

    conn = SlackEventsConnector(handler, signing_secret="slacksigningsecret")
    ts = str(int(time.time()))
    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()
    with TestClient(_app(conn, "/slack")) as client:
        resp = client.post(
            "/slack",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": "v0=bad"},
        )
    assert resp.status_code == 401


# ----------------------------------------------------------------- Discord interactions
def _ed25519_keypair() -> tuple[Any, str]:
    """A synthetic Ed25519 keypair (private_key, public_key_hex) via ``cryptography``."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    pub_hex = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    return priv, pub_hex


def test_discord_ping_returns_pong_body() -> None:
    from fastapi.testclient import TestClient

    from himmy.connectors.discord import DiscordInteractionsConnector

    priv, public_key_hex = _ed25519_keypair()

    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - not run
        return "x"

    conn = DiscordInteractionsConnector(handler, public_key=public_key_hex)
    body = json.dumps({"type": 1}).encode()  # PING
    ts = str(int(time.time()))
    signature = priv.sign(ts.encode() + body).hex()
    with TestClient(_app(conn, "/discord")) as client:
        resp = client.post(
            "/discord",
            content=body,
            headers={
                "X-Signature-Ed25519": signature,
                "X-Signature-Timestamp": ts,
            },
        )
    assert resp.status_code == 200
    # the provider-response mapper returns the interaction response object itself
    assert resp.json() == {"type": 1}


def test_discord_bad_signature_is_401() -> None:
    from fastapi.testclient import TestClient

    from himmy.connectors.discord import DiscordInteractionsConnector

    _priv, public_key_hex = _ed25519_keypair()

    async def handler(sender_id: str, text: str) -> str:  # pragma: no cover - not run
        return "x"

    conn = DiscordInteractionsConnector(handler, public_key=public_key_hex)
    body = json.dumps({"type": 1}).encode()
    with TestClient(_app(conn, "/discord")) as client:
        resp = client.post(
            "/discord",
            content=body,
            headers={"X-Signature-Ed25519": "00" * 64, "X-Signature-Timestamp": "1"},
        )
    assert resp.status_code == 401
