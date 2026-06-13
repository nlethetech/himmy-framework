"""WS4.6 — the consent ledger: record/read consent as immutable EntityRecords.

Each ``(subject_id, purpose)`` owns ONE append-only version chain (grant=v1, withdraw=v2,
re-grant=v3, ...) keyed by :func:`consent_stable_id` and evolved via
:meth:`EntityRegistry.new_version`, so the consent history is content-addressed,
tamper-evident, and covered by the signed audit bundle for free. ``withdraw`` additionally
routes through the already-shipped :class:`~himmy.services.governance.retention.RetentionService`
to crypto-shred the subject's key and write a signed erasure tombstone — so a withdrawal both
flips the latest state to ``WITHDRAWN`` *and* renders any data persisted under the subject's
key unrecoverable.

The ledger is the runtime ``consent_decider``: :meth:`decision` resolves the latest state and
applies the pure :class:`~himmy.services.governance.consent.ConsentPolicy`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.core.errors import HimmyError
from himmy.core.ids import utc_now_iso
from himmy.entities.records import EntityQuery
from himmy.services.governance.consent import (
    CONSENT_KIND,
    ConsentPolicy,
    ConsentRecord,
    ConsentState,
    Decision,
    Purpose,
    consent_stable_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.entities.records import EntityRecord
    from himmy.services.governance.retention import RetentionService

#: How many times to retry a version write when another writer raced us.
_MAX_RETRY = 8


class ConsentLedger:
    """Append-only consent records over an :class:`EntityRegistry` + the PDP."""

    def __init__(
        self,
        entity_registry: EntityRegistryProtocol,
        *,
        policy: ConsentPolicy | None = None,
        retention_service: RetentionService | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """Wire the registry, decision policy, optional erasure service, and clock."""
        self._registry = entity_registry
        self._policy = policy or ConsentPolicy()
        self._retention = retention_service
        self._clock = clock or utc_now_iso

    # ------------------------------------------------------------------ writes
    def set(
        self,
        subject_id: str,
        purpose: Purpose,
        state: ConsentState,
        *,
        workspace_id: str | None = None,
        actor: str = "",
        source: str = "",
        basis: str | None = None,
        expires_at: str | None = None,
    ) -> EntityRecord:
        """Append a new version recording ``state`` for ``(subject_id, purpose)``."""
        record = ConsentRecord(
            subject_id=subject_id,
            purpose=purpose,
            state=state,
            workspace_id=workspace_id,
            actor=actor,
            source=source,
            basis=basis,
            expires_at=expires_at,
            recorded_at=self._clock(),
        )
        payload = record.model_dump(mode="json")
        metadata = {
            "subject_id": subject_id,
            "purpose": purpose.value,
            "state": state.value,
            "workspace_id": workspace_id,
        }
        stable_id = consent_stable_id(subject_id, purpose)
        last_error: HimmyError | None = None
        for _ in range(_MAX_RETRY):
            latest = self._registry.get_latest(stable_id)
            expected = latest.version if latest is not None else 0
            try:
                return self._registry.new_version(
                    stable_id=stable_id,
                    kind=CONSENT_KIND,
                    payload=payload,
                    metadata=metadata,
                    expected_version=expected,
                )
            except (
                HimmyError
            ) as exc:  # optimistic-concurrency conflict — re-read + retry
                last_error = exc
        raise HimmyError(
            f"consent write for ({subject_id!r}, {purpose.value}) lost too many races"
        ) from last_error

    def grant(self, subject_id: str, purpose: Purpose, **kw: object) -> EntityRecord:
        """Record a GRANTED consent."""
        return self.set(subject_id, purpose, ConsentState.GRANTED, **kw)  # type: ignore[arg-type]

    def deny(self, subject_id: str, purpose: Purpose, **kw: object) -> EntityRecord:
        """Record a DENIED consent."""
        return self.set(subject_id, purpose, ConsentState.DENIED, **kw)  # type: ignore[arg-type]

    def withdraw(
        self,
        subject_id: str,
        purpose: Purpose | None = None,
        *,
        reason: str = "",
        actor: str = "",
        source: str = "",
    ) -> list[EntityRecord]:
        """Withdraw consent and crypto-shred the subject.

        Appends a WITHDRAWN version for ``purpose`` (or every purpose the subject has a
        record for when ``purpose`` is ``None``), then — if a
        :class:`RetentionService` is wired — calls ``erase_subject`` to destroy the
        subject's key and write a signed erasure tombstone.
        """
        purposes = [purpose] if purpose is not None else self._purposes_for(subject_id)
        records = [
            self.set(
                subject_id,
                p,
                ConsentState.WITHDRAWN,
                actor=actor,
                source=source,
                basis=reason or None,
            )
            for p in purposes
        ]
        if self._retention is not None:
            self._retention.erase_subject(
                subject_id, reason=reason or "consent_withdrawn"
            )
        return records

    # ------------------------------------------------------------------- reads
    def state(self, subject_id: str, purpose: Purpose) -> ConsentState:
        """The latest recorded state, or UNKNOWN when there is no record."""
        latest = self._registry.get_latest(consent_stable_id(subject_id, purpose))
        if latest is None:
            return ConsentState.UNKNOWN
        return ConsentState(latest.payload["state"])

    def latest(self, subject_id: str, purpose: Purpose) -> ConsentRecord | None:
        """The latest :class:`ConsentRecord`, or ``None``."""
        latest = self._registry.get_latest(consent_stable_id(subject_id, purpose))
        return ConsentRecord.model_validate(latest.payload) if latest else None

    def history(self, subject_id: str, purpose: Purpose) -> list[ConsentRecord]:
        """The full version chain (ascending) for ``(subject_id, purpose)``."""
        return [
            ConsentRecord.model_validate(r.payload)
            for r in self._registry.get_history(consent_stable_id(subject_id, purpose))
        ]

    def decision(
        self, subject_id: str, purpose: Purpose, *, surface: str | None = None
    ) -> Decision:
        """Resolve the latest state and apply the policy (the runtime decider)."""
        return self._policy.decide(
            subject_id, purpose, self.state(subject_id, purpose), surface=surface
        )

    def _purposes_for(self, subject_id: str) -> list[Purpose]:
        """Every purpose this subject has at least one consent record for."""
        records = self._registry.query(
            EntityQuery(kind=CONSENT_KIND, metadata_filters={"subject_id": subject_id})
        )
        seen: dict[str, Purpose] = {}
        for record in records:
            value = str(record.metadata.get("purpose") or record.payload.get("purpose"))
            if value and value not in seen:
                seen[value] = Purpose(value)
        return list(seen.values())


__all__ = ["ConsentLedger"]
