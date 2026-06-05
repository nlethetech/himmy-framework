"""The security audit event: a who/what/when record of a security-relevant action."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso


class SecurityEvent(BaseModel):
    """One security-relevant action: an auth/authz decision or a data access.

    Recorded as a tamper-evident ``EntityRecord`` (kind ``security_event``) so the
    audit trail inherits immutability, lineage, and the signed-bundle integrity check.
    ``actor`` is the compact principal descriptor (subject/auth_method/roles/ip);
    ``outcome`` is ``allow`` or ``deny``.
    """

    event_id: str = Field(default_factory=new_uuid)
    event_type: str  # auth_failure | authz_denied | access | admin
    outcome: str = "deny"  # allow | deny
    created_at: str = Field(default_factory=utc_now_iso)
    actor: dict[str, Any] = Field(default_factory=dict)
    resource: str | None = None
    action: str | None = None
    workspace_id: str | None = None
    method: str | None = None
    path: str | None = None
    detail: str = ""


__all__ = ["SecurityEvent"]
