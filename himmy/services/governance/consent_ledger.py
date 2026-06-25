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
        stable_id = consent_stable_id(subject_id, purpose, workspace_id=workspace_id)
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
        workspace_id: str | None = None,
        reason: str = "",
        actor: str = "",
        source: str = "",
    ) -> list[EntityRecord]:
        """Withdraw consent and crypto-shred the subject **within one tenant**.

        Appends a WITHDRAWN version for ``purpose`` (or every purpose the subject has a
        record for *in ``workspace_id``* when ``purpose`` is ``None``), then — only when the
        caller provably OWNS the subject's vault key — calls ``erase_subject`` to destroy the
        key and write a signed erasure tombstone.

        ``workspace_id`` scopes the operation to the caller's tenant: a withdrawal in tenant A
        never touches tenant B's records. The crypto-shred is FAIL SAFE: it fires only for a
        single-tenant/offline caller (``workspace_id is None``) or when the subject's key is
        provably bound to the caller's own workspace. A multi-tenant caller acting on a
        GLOBAL/unbound key (the production default, since keys are minted without a workspace)
        does NOT shred — a per-tenant erasure of a shared key would destroy other tenants' data.
        """
        if purpose is not None:
            existing = self.latest(subject_id, purpose, workspace_id=workspace_id)
            purposes = [purpose] if existing is not None else []
        else:
            purposes = self._purposes_for(subject_id, workspace_id=workspace_id)
        records = [
            self.set(
                subject_id,
                p,
                ConsentState.WITHDRAWN,
                workspace_id=workspace_id,
                actor=actor,
                source=source,
                basis=reason or None,
            )
            for p in purposes
        ]
        # Only crypto-shred when the caller's tenant actually OWNS the subject's vault key.
        # A "records non-empty" check alone is forgeable: a tenant-A operator can manufacture
        # a grant for any subject_id in workspace A, making ``records`` non-empty, then trigger
        # an erase that destroys a key minted under tenant B.
        #
        # FAIL SAFE on the subject's BOUND workspace (the vault's authoritative tenant linkage).
        # In production subject keys are minted via ``encryptor_for(subject)`` with NO
        # workspace_id, so ``workspace_of`` is almost always None (a GLOBAL/unbound key). A
        # global key cannot be safely shredded on behalf of a single tenant — doing so destroys
        # data shared across every tenant. So the rule is:
        #
        #   * caller is single-tenant/offline (``workspace_id is None`` — the legacy CLI path):
        #     erase proceeds (there is no other tenant to harm; legacy erasure is unchanged);
        #   * caller is multi-tenant (``workspace_id`` set) AND the key is provably bound to the
        #     caller's OWN workspace (``bound_ws == workspace_id``): erase (rightful tenant);
        #   * caller is multi-tenant on an UNBOUND/global key (``bound_ws is None``) OR a key
        #     bound to a DIFFERENT tenant: SKIP the erase entirely (still append the caller's own
        #     WITHDRAWN versions, but NEVER crypto-shred a key the caller does not provably own).
        #
        # NOTE: this intentionally DISABLES crypto-shred for multi-tenant callers on global keys.
        # Per-tenant shredding of a global key is unsound; the proper fix is per-tenant keys
        # (thread workspace_id into every encryptor_for/key_for mint), an owner design decision.
        if self._retention is not None and records:
            bound_ws = self._retention.workspace_of(subject_id)
            owns_subject = workspace_id is None or bound_ws == workspace_id
            if owns_subject:
                # erase_subject also re-checks tenant ownership at the vault (defense-in-depth).
                self._retention.erase_subject(
                    subject_id,
                    reason=reason or "consent_withdrawn",
                    workspace_id=workspace_id,
                )
        return records

    # ------------------------------------------------------------------- reads
    def state(
        self, subject_id: str, purpose: Purpose, *, workspace_id: str | None = None
    ) -> ConsentState:
        """The latest recorded state, or UNKNOWN when there is no record."""
        latest = self._registry.get_latest(
            consent_stable_id(subject_id, purpose, workspace_id=workspace_id)
        )
        if latest is None:
            return ConsentState.UNKNOWN
        return ConsentState(latest.payload["state"])

    def latest(
        self, subject_id: str, purpose: Purpose, *, workspace_id: str | None = None
    ) -> ConsentRecord | None:
        """The latest :class:`ConsentRecord`, or ``None``."""
        latest = self._registry.get_latest(
            consent_stable_id(subject_id, purpose, workspace_id=workspace_id)
        )
        return ConsentRecord.model_validate(latest.payload) if latest else None

    def history(
        self, subject_id: str, purpose: Purpose, *, workspace_id: str | None = None
    ) -> list[ConsentRecord]:
        """The full version chain (ascending) for ``(subject_id, purpose)``."""
        return [
            ConsentRecord.model_validate(r.payload)
            for r in self._registry.get_history(
                consent_stable_id(subject_id, purpose, workspace_id=workspace_id)
            )
        ]

    def decision(
        self,
        subject_id: str,
        purpose: Purpose,
        *,
        workspace_id: str | None = None,
        surface: str | None = None,
    ) -> Decision:
        """Resolve the latest state and apply the policy (the runtime decider)."""
        return self._policy.decide(
            subject_id,
            purpose,
            self.state(subject_id, purpose, workspace_id=workspace_id),
            surface=surface,
        )

    def _purposes_for(
        self, subject_id: str, *, workspace_id: str | None = None
    ) -> list[Purpose]:
        """Every purpose this subject has at least one consent record for.

        Scoped to ``workspace_id`` when given so a withdrawal never enumerates (and then
        withdraws) purposes recorded under a different tenant.
        """
        filters: dict[str, object] = {"subject_id": subject_id}
        if workspace_id is not None:
            filters["workspace_id"] = workspace_id
        records = self._registry.query(
            EntityQuery(kind=CONSENT_KIND, metadata_filters=filters)
        )
        seen: dict[str, Purpose] = {}
        for record in records:
            value = str(record.metadata.get("purpose") or record.payload.get("purpose"))
            if value and value not in seen:
                seen[value] = Purpose(value)
        return list(seen.values())


__all__ = ["ConsentLedger"]
