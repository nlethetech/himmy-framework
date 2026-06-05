"""Tests for the comms pack: send_email + send_webhook (injected seams, offline)."""

from __future__ import annotations

from email.message import EmailMessage
from typing import Any

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.comms import register_comms_pack
from himmy.toolkit.config import ToolkitConfig

PUBLIC = "http://93.184.216.34/hook"


def test_send_email_builds_and_sends() -> None:
    captured: list[EmailMessage] = []
    registry = ToolRegistry()
    register_comms_pack(
        registry,
        ToolkitConfig(comms_allow_send=True, smtp_from="bot@x.com"),
        email_sender=lambda msg, cfg: captured.append(msg),
    )
    out = registry.handler_for("send_email")(
        {"to": "a@b.com", "subject": "Hi", "body": "Hello there"}
    )
    assert out["sent"] is True
    assert len(captured) == 1
    msg = captured[0]
    assert msg["To"] == "a@b.com"
    assert msg["Subject"] == "Hi"
    assert msg["From"] == "bot@x.com"
    assert "Hello there" in msg.get_content()


def test_send_email_html_alternative() -> None:
    captured: list[EmailMessage] = []
    registry = ToolRegistry()
    register_comms_pack(
        registry,
        ToolkitConfig(comms_allow_send=True),
        email_sender=lambda msg, cfg: captured.append(msg),
    )
    registry.handler_for("send_email")(
        {"to": "a@b.com", "subject": "S", "body": "<b>hi</b>", "html": True}
    )
    assert captured[0].is_multipart()


def test_send_webhook_posts_and_guards() -> None:
    calls: list[tuple[str, Any]] = []

    def caller(url: str, payload: Any, headers, timeout: float) -> int:
        calls.append((url, payload))
        return 204

    registry = ToolRegistry()
    register_comms_pack(
        registry, ToolkitConfig(comms_allow_send=True), webhook_caller=caller
    )
    out = registry.handler_for("send_webhook")(
        {"url": PUBLIC, "payload": {"event": "ping"}}
    )
    assert out["status_code"] == 204
    assert out["ok"] is True
    assert calls == [(PUBLIC, {"event": "ping"})]


def test_send_webhook_rejects_private_host() -> None:
    registry = ToolRegistry()
    register_comms_pack(
        registry,
        ToolkitConfig(comms_allow_send=True),
        webhook_caller=lambda *a: 200,
    )
    with pytest.raises(ToolSecurityError):
        registry.handler_for("send_webhook")(
            {"url": "http://127.0.0.1/x", "payload": {}}
        )


def test_send_tools_are_approval_gated_by_default() -> None:
    registry = ToolRegistry()
    register_comms_pack(registry, ToolkitConfig())  # comms_allow_send=False
    assert registry.get("send_email").requires_approval is True
    assert registry.get("send_webhook").requires_approval is True
