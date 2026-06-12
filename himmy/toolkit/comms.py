"""Comms pack: ``send_email`` + ``send_webhook`` — the agent's outbound "act" tools.

These let an agent reach the outside world. Both are approval-gated by default (sending
is a human-in-the-loop decision) unless ``comms_allow_send`` is set. ``send_email`` uses
stdlib ``smtplib`` with credentials read from the toolkit config/env (never from the
model); ``send_webhook`` POSTs JSON to a URL that must pass the SSRF
:func:`~himmy.toolkit._net.guard_url` check, so an agent cannot be steered into hitting
an internal endpoint.

Injectable seams (``email_sender`` / ``webhook_caller``) keep the suite offline.
"""

from __future__ import annotations

from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit._net import guard_url
from himmy.toolkit.config import ToolkitConfig

#: Sends a prepared message via SMTP (the injectable seam for tests).
EmailSender = Callable[[EmailMessage, ToolkitConfig], None]
#: POSTs ``payload`` to ``url`` and returns the HTTP status code.
WebhookCaller = Callable[[str, Any, "dict[str, str] | None", float], int]


def _smtp_send(message: EmailMessage, config: ToolkitConfig) -> None:
    """Default :data:`EmailSender`: connect to the configured SMTP server and send."""
    import smtplib

    if not config.smtp_host:
        raise ToolSecurityError("send_email needs HIMMY_SMTP_HOST configured")
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as smtp:
        if config.smtp_use_tls:
            smtp.starttls()
        if config.smtp_user and config.smtp_password:
            smtp.login(config.smtp_user, config.smtp_password)
        smtp.send_message(message)


def smtp_login_check(config: ToolkitConfig) -> None:
    """Connect + (TLS) + login to the SMTP server without sending — the "Test" path.

    Validates host/port/TLS and, when present, the user/password, raising on
    failure. Used by the Studio Connections page; no message is sent.
    """
    import smtplib

    if not config.smtp_host:
        raise ToolSecurityError("email needs HIMMY_SMTP_HOST configured")
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15) as smtp:
        if config.smtp_use_tls:
            smtp.starttls()
        if config.smtp_user and config.smtp_password:
            smtp.login(config.smtp_user, config.smtp_password)
        smtp.noop()


def _httpx_webhook(
    url: str, payload: Any, headers: dict[str, str] | None, timeout: float
) -> int:
    """Default :data:`WebhookCaller`: POST ``payload`` as JSON, return the status.

    The transport pins the DNS-vetted IP at connect time so the name ``send_webhook``
    already passed through :func:`guard_url` can't rebind to an internal address
    between that check and httpx's own resolution (the rebinding TOCTOU gap).
    """
    import httpx

    from himmy.toolkit._net import build_pinned_transport

    with httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        transport=build_pinned_transport(),
    ) as client:
        resp = client.post(url, json=payload, headers=headers)
        return resp.status_code


_EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "to": {"type": "string", "description": "Recipient email address."},
        "subject": {"type": "string"},
        "body": {"type": "string"},
        "html": {"type": "boolean", "default": False},
    },
    "required": ["to", "subject", "body"],
    "additionalProperties": False,
}

_WEBHOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "An http(s) webhook URL."},
        "payload": {"type": "object"},
        "headers": {"type": "object"},
    },
    "required": ["url", "payload"],
    "additionalProperties": False,
}


def register_comms_pack(
    registry: ToolRegistry,
    config: ToolkitConfig,
    *,
    email_sender: EmailSender | None = None,
    webhook_caller: WebhookCaller | None = None,
) -> None:
    """Register ``send_email`` and ``send_webhook`` (approval-gated by default)."""
    send_mail = email_sender or _smtp_send
    call_webhook = webhook_caller or _httpx_webhook
    gated = not config.comms_allow_send

    def send_email(args: dict[str, Any]) -> dict[str, Any]:
        message = EmailMessage()
        message["From"] = config.smtp_from or config.smtp_user or "agent@localhost"
        message["To"] = str(args["to"])
        message["Subject"] = str(args["subject"])
        body = str(args["body"])
        if args.get("html"):
            message.set_content("This message requires an HTML viewer.")
            message.add_alternative(body, subtype="html")
        else:
            message.set_content(body)
        send_mail(message, config)
        return {"sent": True, "to": args["to"]}

    def send_webhook(args: dict[str, Any]) -> dict[str, Any]:
        url = str(args["url"])
        guard_url(
            url,
            allow_private=config.allow_private_hosts,
            allow_hosts=set(config.egress_allow_hosts) or None,
        )
        status = call_webhook(
            url, args.get("payload"), args.get("headers"), config.http_timeout
        )
        return {"url": url, "status_code": status, "ok": 200 <= status < 300}

    register_local_tool(
        registry,
        name="send_email",
        read_only=False,
        handler=send_email,
        description="Send an email via the configured SMTP server.",
        args_json_schema=_EMAIL_SCHEMA,
        requires_approval=gated,
        metadata={"pack": "comms"},
    )
    register_local_tool(
        registry,
        name="send_webhook",
        read_only=False,
        handler=send_webhook,
        description="POST a JSON payload to an http(s) webhook URL.",
        args_json_schema=_WEBHOOK_SCHEMA,
        requires_approval=gated,
        sensitive_arg_names=["headers"],
        metadata={"pack": "comms"},
    )


__all__ = ["register_comms_pack", "EmailSender", "WebhookCaller"]
