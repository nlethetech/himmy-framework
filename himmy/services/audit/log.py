"""The security audit log: record security events as tamper-evident entities.

Each :class:`~himmy.services.audit.models.SecurityEvent` is registered as an
``EntityRecord`` (kind ``security_event``) in the :class:`EntityRegistry`, so the
audit trail is immutable, content-addressed, and covered by the signed audit bundle
(`himmy/entities/integrity.py`) — you can later prove the log was not altered. The
durability of the trail follows the registry backend (in-memory by default, Postgres
when a `PostgresEntityRegistry` is wired).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.entities.records import EntityRecord
from himmy.services.audit.models import SecurityEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol

#: The EntityRecord kind used for audit events.
SECURITY_EVENT_KIND = "security_event"


class SecurityAuditLog:
    """Append-only security audit trail backed by the entity registry."""

    def __init__(self, entity_registry: EntityRegistryProtocol) -> None:
        """Record events into ``entity_registry`` as ``security_event`` records."""
        self._registry = entity_registry

    def record(self, event: SecurityEvent) -> SecurityEvent:
        """Persist ``event`` as a tamper-evident entity; returns the event."""
        self._registry.register(
            EntityRecord.create(
                stable_id=event.event_id,
                version=1,
                kind=SECURITY_EVENT_KIND,
                payload=event.model_dump(),
                metadata={
                    "workspace_id": event.workspace_id,
                    "actor": event.actor.get("subject"),
                    "outcome": event.outcome,
                },
            )
        )
        return event

    def recent(
        self,
        *,
        limit: int = 100,
        workspace_id: str | None = None,
        workspace_ids: frozenset[str] | set[str] | None = None,
        event_type: str | None = None,
    ) -> list[SecurityEvent]:
        """Return recent events (newest first), optionally filtered.

        ``workspace_id`` pins to a single tenant; ``workspace_ids`` (red-team scope-r5)
        pins to an ALLOW-LIST of tenants — the shape a tenant-bound principal's
        :func:`~himmy.api.auth.context.studio_tenant_filter` returns — so the seclog
        viewer surfaces only events stamped to a workspace the caller is entitled to.
        Both default to ``None`` (no tenant filter — the single-box / unrestricted path),
        and both are applied BEFORE the ``limit`` slice so scoping never silently drops
        in-window rows. Passing both intersects them.
        """
        events = [
            SecurityEvent.model_validate(r.payload)
            for r in self._registry.list_by_kind(SECURITY_EVENT_KIND)
        ]
        if workspace_id is not None:
            events = [e for e in events if e.workspace_id == workspace_id]
        if workspace_ids is not None:
            events = [e for e in events if e.workspace_id in workspace_ids]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        events.sort(key=lambda e: e.created_at, reverse=True)
        return events[:limit]


__all__ = ["SecurityAuditLog", "SECURITY_EVENT_KIND"]
