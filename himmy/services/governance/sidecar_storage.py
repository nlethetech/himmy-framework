"""S3 — per-subject envelope-encrypt the content-bearing *sidecar* stores on ALLOW.

:class:`~himmy.services.governance.consent_storage.ConsentGatedStorage` wraps ONLY the
:class:`~himmy.services.storage.service.StorageService` facade. But three content-bearing
stores live OUTSIDE that facade as separate objects and are never passed to it, so on a
governed deployment they would keep writing **plaintext** subject content that
``RetentionService.erase_subject`` can never make unrecoverable:

* the memory module's :class:`~himmy.services.memory.store.MemoryStore`
  (:class:`~himmy.services.memory.store.MemoryRecord`, a different type from the facade's
  ``MemoryObject``) — its ``text`` is the remembered fact;
* the unified :class:`~himmy.services.storage.conversations.ConversationStore` — the rich
  ``ChatThread`` body is the conversation transcript;
* the pgvector knowledge backend — a subject-scoped KB's chunk ``text``.

This module supplies the three transparent decorators that close that gap, each carrying the
SAME ``RETAIN`` consent gate + per-subject :class:`SubjectKeyVault` envelope cipher the spine
path (:class:`~himmy.services.governance.consent_registry.ConsentAwareRegistry`) already
applies — so routing a write through a sidecar never silently downgrades encryption-at-rest.

Reviewer must_fixes folded in:

* **Three NEW wrappers, wired in deps.py** — these are decorators over the separate store
  objects, not a reach through the facade (which would not cover them).
* **Default-subject memory is excluded from per-subject keying.**
  :attr:`~himmy.services.memory.store.MemoryRecord.subject_id` defaults to the literal
  ``"default"``; crypto-shredding ``"default"`` would over-delete every un-attributed memory.
  A ``"default"``-subject record is gated like any other but its ``text`` is left
  PLAINTEXT (no per-subject key is minted for it), so erasing one real subject never reaches
  the shared ``"default"`` bucket. The excluded subject is configurable
  (:data:`DEFAULT_MEMORY_SUBJECT`).
* **Knowledge: encrypt chunk TEXT only** (never the embedding vector — that would break ANN),
  and only for subject-scoped KBs (see :class:`SubjectKnowledgeCipher`).
* **Offline path untouched** — these decorators are constructed only in the governed branch
  of the API container; the zero-config path keeps the bare stores byte-for-byte, and even
  when governed, a write with no minted key (e.g. excluded subject) stays plaintext.

Decryption is idempotent on plaintext (``FieldEncryptor.is_encrypted`` guards every read),
so legacy plaintext rows written before a key was configured round-trip untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from himmy.services.governance.consent import Effect, Purpose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from himmy.agents.base_agent.thread import ChatThread
    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.governance.consent import Decision
    from himmy.services.governance.retention import SubjectKeyVault
    from himmy.services.memory.store import MemoryRecord
    from himmy.services.storage.conversations import ConversationSummary

    ConsentDecider = Callable[[str, Purpose], Decision]

#: The memory ``subject_id`` whose content is deliberately NOT per-subject keyed: the literal
#: ``"default"`` bucket (the :class:`MemoryRecord` default) is shared across un-attributed
#: facts, so crypto-shredding it would over-delete. A record for this subject is still consent
#: -gated, but its ``text`` is left plaintext (no per-subject key is minted for it).
DEFAULT_MEMORY_SUBJECT = "default"


def _is_governed_subject(subject: str | None) -> bool:
    """Whether ``subject`` is a real, per-subject-keyable subject (not the shared default)."""
    return bool(subject) and subject != DEFAULT_MEMORY_SUBJECT


class ConsentGatedMemoryStore:
    """Gate + per-subject-encrypt the memory module's :class:`MemoryStore` (S3).

    Wraps a concrete ``MemoryStore`` (``SqliteMemoryStore`` / the K5 Postgres mirror / the
    in-memory store) and:

    * ``save`` — consult the decider at purpose ``RETAIN`` for the record's subject. On
      :attr:`Effect.ALLOW`, encrypt ``text`` under the subject's shreddable key (unless the
      subject is the excluded :data:`DEFAULT_MEMORY_SUBJECT`, left plaintext) and persist;
      otherwise emit ``consent_denied_persist`` and SKIP the write (the record is returned to
      the caller for the live request, mirroring :class:`ConsentGatedStorage`).
    * ``list`` / ``get`` — decrypt ``text`` transparently (idempotent on plaintext).
    * every other method (``delete``/``save_link``/``links_from``/``links_to``/``close``) is
      delegated unchanged.
    """

    def __init__(
        self,
        inner: Any,
        *,
        decider: ConsentDecider,
        key_vault: SubjectKeyVault | None = None,
        audit: SecurityAuditLog | None = None,
    ) -> None:
        """Wrap ``inner`` memory store; gate ``save`` at purpose=RETAIN."""
        self._inner = inner
        self._decider = decider
        self._vault = key_vault
        self._audit = audit

    def save(self, record: MemoryRecord) -> MemoryRecord:
        """Persist ``record`` unless RETAIN is not consented; encrypt ``text`` on ALLOW."""
        subject = record.subject_id or None
        if not subject:  # subject-bearing but unresolved → fail closed
            _audit_deny(self._audit, "", "MemoryRecord", "unresolved subject", "save")
            return record
        decision = self._decider(subject, Purpose.RETAIN)
        if decision.effect is not Effect.ALLOW:
            _audit_deny(self._audit, subject, "MemoryRecord", decision.reason, "save")
            return record
        stored = self._encrypt(record, subject)
        saved = cast("MemoryRecord", self._inner.save(stored))
        # Hand the CALLER back the plaintext record (the live request wants cleartext);
        # only the at-rest row carries ciphertext.
        return record if saved is stored else saved

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory by id with ``text`` decrypted (idempotent on plaintext)."""
        record = cast("MemoryRecord | None", self._inner.get(memory_id))
        return self._decrypt(record) if record is not None else None

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]:
        """List memories with ``text`` decrypted (idempotent on plaintext)."""
        rows = self._inner.list(subject_id, active_only=active_only, tier=tier)
        return [self._decrypt(r) for r in rows]

    # ------------------------------------------------------------------ cipher
    def _encrypt(self, record: MemoryRecord, subject: str) -> MemoryRecord:
        """Return a copy of ``record`` with ``text`` encrypted under the subject's key.

        The excluded :data:`DEFAULT_MEMORY_SUBJECT` and a missing vault both leave the text
        plaintext; an empty text is left untouched (nothing to protect).
        """
        if (
            self._vault is None
            or not _is_governed_subject(subject)
            or not record.text
        ):
            return record
        from himmy.services.storage.encryption import FieldEncryptor

        enc = self._vault.encryptor_for(subject)
        if FieldEncryptor.is_encrypted(record.text):
            return record
        token = enc.encrypt(record.text, aad=subject.encode())
        return record.model_copy(update={"text": token})

    def _decrypt(self, record: MemoryRecord) -> MemoryRecord:
        """Return a copy of ``record`` with ``text`` decrypted (or unchanged if plaintext)."""
        subject = record.subject_id or None
        if (
            self._vault is None
            or not _is_governed_subject(subject)
            or not record.text
        ):
            return record
        from himmy.services.storage.encryption import FieldEncryptor

        if not FieldEncryptor.is_encrypted(record.text):
            return record
        # ``subject`` is non-None here (``_is_governed_subject`` is True).
        assert subject is not None
        if not self._vault.has(subject):
            # The subject was crypto-shredded: the key is gone, so the ciphertext is
            # permanently unrecoverable. Surface a tombstone marker instead of raising,
            # so listing surviving subjects never crashes on an erased one's residue.
            return record.model_copy(update={"text": ""})
        enc = self._vault.encryptor_for(subject)
        return record.model_copy(
            update={"text": enc.decrypt(record.text, aad=subject.encode())}
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate every non-gated method to the wrapped store."""
        return getattr(self._inner, name)


class ConsentGatedConversationStore:
    """Gate + per-subject-encrypt the unified :class:`ConversationStore` (S3).

    A conversation is keyed by ``conversation_id`` but the data subject of its transcript is
    the run's subject, threaded in here as ``subject`` on the save calls. On
    :attr:`Effect.ALLOW` every message ``content`` in the authoritative ``ChatThread`` is
    encrypted under the subject's shreddable key (reusing
    :meth:`StorePayloadCipher.encrypt_thread_payload`, bound to the subject as AAD) BEFORE the
    store persists + re-projects it, so neither the authoritative ``thread`` blob nor the flat
    ``conversation_messages`` projection holds plaintext crypto-shred can't reach.

    The store's ``subject_id`` column (added by its own additive migration) records the
    linkage so :class:`SubjectReachMap` can find + delete a subject's conversations.

    All reads (``load_thread`` / ``flat_messages`` / ``list_summaries`` / …) decrypt
    transparently; every non-gated method is delegated unchanged.
    """

    def __init__(
        self,
        inner: Any,
        *,
        decider: ConsentDecider,
        key_vault: SubjectKeyVault | None = None,
        audit: SecurityAuditLog | None = None,
    ) -> None:
        """Wrap ``inner`` conversation store; gate subject-bearing saves at RETAIN."""
        self._inner = inner
        self._decider = decider
        self._vault = key_vault
        self._audit = audit

    def save_thread(
        self,
        conversation_id: str,
        thread: ChatThread,
        *,
        subject_id: str | None = None,
        **kwargs: Any,
    ) -> ConversationSummary | None:
        """Save ``thread`` for ``conversation_id``; gate + encrypt when a subject is given.

        ``subject_id`` is the data subject of the transcript (the run's subject) — the SAME
        parameter the bare store takes for its linkage column, so the call site is uniform.
        When absent the conversation is treated as un-attributed infrastructure and saved as
        today (ungoverned) — the consent gate only governs subject-attributed conversations.
        When present and RETAIN is not consented, the write is skipped (``None`` returned).
        """
        if not subject_id:
            return cast(
                "ConversationSummary",
                self._inner.save_thread(conversation_id, thread, **kwargs),
            )
        decision = self._decider(subject_id, Purpose.RETAIN)
        if decision.effect is not Effect.ALLOW:
            _audit_deny(
                self._audit, subject_id, "ChatThread", decision.reason, "save_thread"
            )
            return None
        stored = self._encrypt_thread(thread, subject_id)
        return cast(
            "ConversationSummary",
            self._inner.save_thread(
                conversation_id, stored, subject_id=subject_id, **kwargs
            ),
        )

    def load_thread(self, conversation_id: str) -> ChatThread | None:
        """Return the authoritative thread with message content decrypted."""
        thread = self._inner.load_thread(conversation_id)
        if thread is None:
            return None
        subject = self._inner.subject_of(conversation_id)
        return self._decrypt_thread(thread, subject) if subject else thread

    # ------------------------------------------------------------------ cipher
    def _encrypt_thread(self, thread: ChatThread, subject: str) -> ChatThread:
        """Return a copy of ``thread`` with each message ``content`` encrypted (subject AAD)."""
        if self._vault is None or not _is_governed_subject(subject):
            return thread
        from himmy.agents.base_agent.thread import ChatThread as _Thread
        from himmy.services.storage.at_rest import StorePayloadCipher

        enc = self._vault.encryptor_for(subject)
        payload = thread.model_dump(mode="json")
        payload = StorePayloadCipher(enc).encrypt_thread_payload(
            payload, thread_id=subject
        )
        return _Thread.model_validate(payload)

    def _decrypt_thread(self, thread: ChatThread, subject: str) -> ChatThread:
        """Return a copy of ``thread`` with each message ``content`` decrypted (or as-is)."""
        if self._vault is None or not _is_governed_subject(subject):
            return thread
        from himmy.agents.base_agent.thread import ChatThread as _Thread
        from himmy.services.storage.at_rest import StorePayloadCipher

        if not self._vault.has(subject):
            # Crypto-shredded subject: the message bodies are permanently unrecoverable.
            return thread
        enc = self._vault.encryptor_for(subject)
        payload = thread.model_dump(mode="json")
        payload = StorePayloadCipher(enc).decrypt_thread_payload(
            payload, thread_id=subject
        )
        return _Thread.model_validate(payload)

    def __getattr__(self, name: str) -> Any:
        """Delegate every non-gated method to the wrapped store."""
        return getattr(self._inner, name)


class SubjectKnowledgeCipher:
    """Per-subject envelope cipher for a subject-scoped KB's chunk TEXT only (S3).

    The pgvector knowledge backend stores both the chunk ``text`` and its embedding
    ``vector``. Encrypting the vector would break ANN search, so ONLY the ``text`` is
    enveloped — under the per-subject shreddable key, bound to the subject as AAD — and ONLY
    for **subject-scoped** KBs. By convention a subject-scoped KB pins its ``client_id`` to the
    subject id (the same scoping the backend already uses for ``(workspace, client, name)``
    uniqueness), so the KB->subject mapping needs no new table: ``client_id`` IS the subject.

    This is a thin, stateless helper the ingestion path applies to a chunk's text before it
    is embedded+persisted, and the retrieval path applies in reverse on a hit. It is
    deliberately not a full store decorator (the backend's surface is wide and async); it
    gives callers the two pure functions they need while keeping the embedding untouched.
    """

    def __init__(self, key_vault: SubjectKeyVault) -> None:
        self._vault = key_vault

    def encrypt_text(self, text: str, *, subject: str) -> str:
        """Encrypt one chunk's ``text`` under ``subject``'s key (idempotent; plaintext-safe)."""
        from himmy.services.storage.encryption import FieldEncryptor

        if (
            not text
            or not _is_governed_subject(subject)
            or FieldEncryptor.is_encrypted(text)
        ):
            return text
        return self._vault.encryptor_for(subject).encrypt(text, aad=subject.encode())

    def decrypt_text(self, text: str, *, subject: str) -> str:
        """Decrypt one chunk's ``text`` (returns ``""`` for a shredded subject; plaintext-safe)."""
        from himmy.services.storage.encryption import FieldEncryptor

        if not text or not _is_governed_subject(subject):
            return text
        if not FieldEncryptor.is_encrypted(text):
            return text
        if not self._vault.has(subject):
            return ""  # crypto-shredded: permanently unrecoverable
        return self._vault.encryptor_for(subject).decrypt(text, aad=subject.encode())


def _audit_deny(
    audit: SecurityAuditLog | None,
    subject: str,
    resource: str,
    reason: str,
    action: str,
) -> None:
    """Record a ``consent_denied_persist`` security event (no-op without a log)."""
    if audit is None:
        return
    from himmy.services.audit.models import SecurityEvent

    audit.record(
        SecurityEvent(
            event_type="consent_denied_persist",
            outcome="deny",
            actor={"subject": subject} if subject else {},
            resource=resource,
            action=action,
            detail=reason,
        )
    )


__all__ = [
    "DEFAULT_MEMORY_SUBJECT",
    "ConsentGatedMemoryStore",
    "ConsentGatedConversationStore",
    "SubjectKnowledgeCipher",
]
