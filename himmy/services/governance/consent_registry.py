"""WS4.6 — ``ConsentAwareRegistry``: the linchpin gate on the immutable audit spine.

The single-agent runtime writes subject transcripts to the ``EntityRegistry`` *in parallel*
with the StorageService (``_emit`` registers every run event; ``_register_message`` writes
messages), so gating storage alone would leave a full cleartext copy on the content-addressed
spine — the ephemeral guarantee would be a lie. This wrapper closes that hole: for a
configured set of subject-bearing ``kind``s it consults the consent decider at
purpose=RETAIN and, on :attr:`Effect.EPHEMERAL`/:attr:`Effect.DENY`, **never lets the record
reach the spine**. On :attr:`Effect.ALLOW` it registers the record with its subject-bearing
payload fields **encrypted under the subject's shreddable key**, so a later
``RetentionService.erase_subject`` (which cannot delete an append-only record) renders the
ciphertext permanently unrecoverable.

Infrastructure records (personas, prompts, agent/env state) and the audit kinds (``consent``,
``security_event``, ``erasure_tombstone``) are **not** in the gated set and pass through
untouched, so the default audit trail that examples/tests rely on is never starved. The set of
gated kinds and the subject/field extractors are **injected** by the wiring layer, which knows
the runtime's real record kinds — keeping this mechanism decoupled and unit-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from himmy.entities.records import EntityRecord
from himmy.services.governance.consent import Effect, Purpose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Sequence

    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.governance.consent import Decision
    from himmy.services.governance.retention import SubjectKeyVault

    ConsentDecider = Callable[[str, Purpose], Decision]
    SubjectExtractor = Callable[[EntityRecord], str | None]
    FieldsForKind = Callable[[str], Sequence[str]]


class ConsentAwareRegistry:
    """Gate subject-bearing ``register()`` writes to the spine; delegate the rest."""

    def __init__(
        self,
        inner: Any,
        *,
        decider: ConsentDecider,
        gated_kinds: Iterable[str],
        subject_extractor: SubjectExtractor,
        key_vault: SubjectKeyVault | None = None,
        encrypted_fields_for: FieldsForKind | None = None,
        audit: SecurityAuditLog | None = None,
    ) -> None:
        """Wrap ``inner`` registry; gate ``gated_kinds`` at purpose=RETAIN."""
        self._inner = inner
        self._decider = decider
        self._gated = frozenset(gated_kinds)
        self._subject_of = subject_extractor
        self._vault = key_vault
        self._fields_for: FieldsForKind = encrypted_fields_for or (lambda _kind: ())
        self._audit = audit

    def register(self, record: EntityRecord) -> EntityRecord:
        """Register ``record`` unless it is a non-consented subject-bearing record."""
        if record.kind not in self._gated:
            return cast(EntityRecord, self._inner.register(record))
        subject = self._subject_of(record)
        if not subject:  # subject-bearing kind but unresolved → fail closed
            self._audit_deny("", record.kind, "unresolved subject")
            return record
        decision = self._decider(subject, Purpose.RETAIN)
        if decision.effect is not Effect.ALLOW:
            self._audit_deny(subject, record.kind, decision.reason)
            return record  # ephemeral/deny: the spine never sees this subject's data
        return cast(EntityRecord, self._inner.register(self._encrypt(record, subject)))

    def _encrypt(self, record: EntityRecord, subject: str) -> EntityRecord:
        """Encrypt the kind's subject-bearing fields under the subject's key (if wired).

        ``record_id`` is derived from ``(kind, stable_id, version)`` only, so encrypting
        the payload leaves the content-addressed id (and thus lineage) unchanged while the
        signed-bundle hash now covers ciphertext.
        """
        fields = tuple(self._fields_for(record.kind))
        if not fields or self._vault is None:
            return record
        from himmy.services.storage.encryption import RecordCipher

        cipher = RecordCipher(self._vault.encryptor_for(subject))
        payload = cipher.encrypt_fields(record.payload, fields, aad=subject.encode())
        return EntityRecord.create(
            stable_id=record.stable_id,
            version=record.version,
            kind=record.kind,
            payload=payload,
            metadata=record.metadata,
        )

    def _audit_deny(self, subject: str, kind: str, reason: str) -> None:
        """Record a ``consent_denied_persist`` security event (no-op without a log)."""
        if self._audit is None:
            return
        from himmy.services.audit.models import SecurityEvent

        self._audit.record(
            SecurityEvent(
                event_type="consent_denied_persist",
                outcome="deny",
                actor={"subject": subject} if subject else {},
                resource=kind,
                action="register",
                detail=reason,
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate every non-gated registry method to the wrapped registry."""
        return getattr(self._inner, name)


__all__ = ["ConsentAwareRegistry"]
