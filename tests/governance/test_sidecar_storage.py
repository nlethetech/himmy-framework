"""S3 — per-subject envelope-encrypt the content-bearing sidecar stores on ALLOW.

Covers the three NEW gating/cipher wrappers (memory module store, unified conversation
store, subject-scoped knowledge chunk cipher), the default-subject exclusion, and the
offline (ungoverned) byte-identity.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from himmy.agents.base_agent.thread import (  # noqa: E402
    ChatThread,
    Message,
    MessageRole,
)
from himmy.entities.registry import EntityRegistry  # noqa: E402
from himmy.services.audit.log import SecurityAuditLog  # noqa: E402
from himmy.services.governance.consent import (  # noqa: E402
    ConsentPolicy,
    Purpose,
)
from himmy.services.governance.consent_ledger import ConsentLedger  # noqa: E402
from himmy.services.governance.retention import SubjectKeyVault  # noqa: E402
from himmy.services.governance.sidecar_storage import (  # noqa: E402
    DEFAULT_MEMORY_SUBJECT,
    ConsentGatedConversationStore,
    ConsentGatedMemoryStore,
    SubjectKnowledgeCipher,
)
from himmy.services.memory.store import InMemoryMemoryStore, MemoryRecord  # noqa: E402
from himmy.services.storage.conversations import ConversationStore  # noqa: E402
from himmy.services.storage.encryption import FieldEncryptor  # noqa: E402


def _governed() -> tuple[ConsentLedger, SubjectKeyVault, SecurityAuditLog]:
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    vault = SubjectKeyVault()
    audit = SecurityAuditLog(registry)
    return ledger, vault, audit


# --------------------------------------------------------------------- memory


def test_memory_allow_encrypts_text_at_rest_and_decrypts_on_read() -> None:
    ledger, vault, audit = _governed()
    ledger.grant("alice", Purpose.RETAIN)
    inner = InMemoryMemoryStore()
    gated = ConsentGatedMemoryStore(
        inner, decider=ledger.decision, key_vault=vault, audit=audit
    )

    rec = MemoryRecord(subject_id="alice", text="alice loves mangoes")
    gated.save(rec)

    # At rest (the inner store) the text is ciphertext.
    raw = inner.get(rec.memory_id)
    assert raw is not None
    assert FieldEncryptor.is_encrypted(raw.text)
    assert "mangoes" not in raw.text

    # The gated read decrypts symmetrically.
    got = gated.get(rec.memory_id)
    assert got is not None and got.text == "alice loves mangoes"
    listed = gated.list("alice")
    assert listed and listed[0].text == "alice loves mangoes"


def test_memory_denied_subject_is_skipped() -> None:
    ledger, vault, audit = _governed()
    inner = InMemoryMemoryStore()
    gated = ConsentGatedMemoryStore(
        inner, decider=ledger.decision, key_vault=vault, audit=audit
    )
    rec = MemoryRecord(subject_id="bob", text="secret")  # no consent → DENY
    returned = gated.save(rec)
    assert returned is rec  # caller keeps it for the live request
    assert inner.get(rec.memory_id) is None  # nothing persisted
    assert audit.recent(event_type="consent_denied_persist")


def test_memory_default_subject_is_excluded_from_per_subject_keying() -> None:
    ledger, vault, audit = _governed()
    ledger.grant(DEFAULT_MEMORY_SUBJECT, Purpose.RETAIN)
    inner = InMemoryMemoryStore()
    gated = ConsentGatedMemoryStore(
        inner, decider=ledger.decision, key_vault=vault, audit=audit
    )
    rec = MemoryRecord(text="an un-attributed fact")  # subject_id defaults to "default"
    assert rec.subject_id == DEFAULT_MEMORY_SUBJECT
    gated.save(rec)
    # The shared "default" bucket is gated but NOT per-subject-keyed (over-delete guard).
    raw = inner.get(rec.memory_id)
    assert raw is not None and raw.text == "an un-attributed fact"
    assert not FieldEncryptor.is_encrypted(raw.text)
    # No per-subject key was minted for "default".
    assert vault.has(DEFAULT_MEMORY_SUBJECT) is False


def test_memory_shredded_subject_text_is_unrecoverable_on_read() -> None:
    ledger, vault, _ = _governed()
    ledger.grant("carol", Purpose.RETAIN)
    inner = InMemoryMemoryStore()
    gated = ConsentGatedMemoryStore(inner, decider=ledger.decision, key_vault=vault)
    gated.save(MemoryRecord(subject_id="carol", text="phone number"))

    # Crypto-shred the subject's key, then a read can no longer recover the text.
    assert vault.destroy("carol") is True
    [hit] = gated.list("carol")
    assert hit.text == ""  # tombstone marker, not a crash


def test_memory_ungoverned_path_is_byte_identical() -> None:
    # No vault, ALLOW-everything policy → the wrapper writes plaintext exactly as the bare
    # store would (the offline byte-identity invariant for a non-keyed deployment).
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=False))
    inner = InMemoryMemoryStore()
    gated = ConsentGatedMemoryStore(inner, decider=ledger.decision, key_vault=None)
    gated.save(MemoryRecord(subject_id="alice", text="plain"))
    raw = inner.get(inner.list("alice")[0].memory_id)
    assert raw is not None and raw.text == "plain"
    assert not FieldEncryptor.is_encrypted(raw.text)


# ---------------------------------------------------------------- conversations


def _thread(*turns: tuple[MessageRole, str]) -> ChatThread:
    t = ChatThread(thread_id="c1")
    for role, text in turns:
        t.append_message(Message(role=role, content=text))
    return t


def test_conversation_allow_encrypts_messages_and_links_subject() -> None:
    ledger, vault, audit = _governed()
    ledger.grant("dana", Purpose.RETAIN)
    inner = ConversationStore(":memory:")
    gated = ConsentGatedConversationStore(
        inner, decider=ledger.decision, key_vault=vault, audit=audit
    )

    thread = _thread(
        (MessageRole.USER, "my SSN is 123"),
        (MessageRole.ASSISTANT, "noted"),
    )
    summary = gated.save_thread("c1", thread, subject_id="dana")
    assert summary is not None

    # At rest, the authoritative thread + flat projection carry no plaintext content.
    stored = inner.load_thread("c1")
    assert stored is not None
    assert all(FieldEncryptor.is_encrypted(m.content) for m in stored.messages)
    flat_text = " ".join(m.text for m in inner.flat_messages("c1"))
    assert "123" not in flat_text
    # The subject linkage is recorded for the reach map.
    assert inner.subject_of("c1") == "dana"
    assert inner.conversation_ids_for_subject("dana") == ["c1"]

    # The gated read decrypts symmetrically.
    back = gated.load_thread("c1")
    assert back is not None
    assert [m.content for m in back.messages] == ["my SSN is 123", "noted"]


def test_conversation_denied_subject_is_skipped() -> None:
    ledger, vault, audit = _governed()
    inner = ConversationStore(":memory:")
    gated = ConsentGatedConversationStore(
        inner, decider=ledger.decision, key_vault=vault, audit=audit
    )
    out = gated.save_thread("c2", _thread((MessageRole.USER, "hi")), subject_id="eve")
    assert out is None  # no consent → skipped
    assert inner.load_thread("c2") is None
    assert audit.recent(event_type="consent_denied_persist")


def test_conversation_unattributed_save_is_ungoverned() -> None:
    ledger, vault, _ = _governed()
    inner = ConversationStore(":memory:")
    gated = ConsentGatedConversationStore(
        inner, decider=ledger.decision, key_vault=vault
    )
    # No subject → un-attributed infra conversation, saved plaintext as today.
    gated.save_thread("c3", _thread((MessageRole.USER, "hello")))
    stored = inner.load_thread("c3")
    assert stored is not None
    assert stored.messages[0].content == "hello"
    assert inner.subject_of("c3") is None


def test_conversation_legacy_db_gains_subject_id_column() -> None:
    import sqlite3
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "legacy.db")
        # Simulate a pre-S3 conversations.db WITHOUT the subject_id column.
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, "
            "thread TEXT NOT NULL, origin TEXT NOT NULL DEFAULT 'cli', title TEXT, "
            "agent_path TEXT, provider TEXT, project_id TEXT, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        )
        conn.commit()
        conn.close()
        # Opening through the store migrates it in place (additive ADD COLUMN).
        store = ConversationStore(path)
        cols = {
            r["name"]
            for r in store._conn.execute("PRAGMA table_info(conversations)").fetchall()
        }
        assert "subject_id" in cols
        store.close()


# ------------------------------------------------------------------- knowledge


def test_knowledge_cipher_encrypts_text_only_for_governed_subject() -> None:
    vault = SubjectKeyVault()
    cipher = SubjectKnowledgeCipher(vault)

    token = cipher.encrypt_text("clinical note", subject="frank")
    assert FieldEncryptor.is_encrypted(token)
    assert "clinical" not in token
    assert cipher.decrypt_text(token, subject="frank") == "clinical note"

    # Default subject + empty text are left plaintext (no per-subject key minted).
    assert cipher.encrypt_text("x", subject=DEFAULT_MEMORY_SUBJECT) == "x"
    assert cipher.encrypt_text("", subject="frank") == ""


def test_knowledge_cipher_shredded_subject_is_unrecoverable() -> None:
    vault = SubjectKeyVault()
    cipher = SubjectKnowledgeCipher(vault)
    token = cipher.encrypt_text("secret note", subject="gwen")
    assert vault.destroy("gwen") is True
    assert cipher.decrypt_text(token, subject="gwen") == ""  # unrecoverable


# ----------------------------------------------------- durable cross-restart


def test_durable_memory_survives_restart_then_erase_leaves_no_plaintext(
    tmp_path: object,
) -> None:
    """Write subject memory, restart the vault+store, erase: no plaintext remains; a
    control subject still decrypts (the S3+S4 cross-restart acceptance)."""
    import sqlite3
    from pathlib import Path

    from himmy.services.governance.retention import (
        RetentionService,
        SubjectReachMap,
    )
    from himmy.services.memory.store import SqliteMemoryStore

    base = Path(str(tmp_path))
    kv_path = str(base / "keyvault.db")
    mem_path = str(base / "memory.db")
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    ledger.grant("alice", Purpose.RETAIN)
    ledger.grant("bob", Purpose.RETAIN)

    # --- process 1: write ---
    vault1 = SubjectKeyVault(kv_path)
    store1 = SqliteMemoryStore(mem_path)
    gated1 = ConsentGatedMemoryStore(store1, decider=ledger.decision, key_vault=vault1)
    gated1.save(MemoryRecord(subject_id="alice", text="alice-medical-secret"))
    gated1.save(MemoryRecord(subject_id="bob", text="bob-fact"))
    vault1.close()
    store1.close()

    # --- process 2: restart, then erase alice ---
    vault2 = SubjectKeyVault(kv_path)
    store2 = SqliteMemoryStore(mem_path)
    reach = SubjectReachMap(memory_store=store2)
    service = RetentionService(registry, key_vault=vault2, reach_map=reach)
    cert = service.erase_subject("alice")
    assert cert.crypto_shredded is True
    assert {r.store: r.deleted for r in cert.per_store}["memory"] == 1
    store2.close()
    vault2.close()

    # Scan the raw memory.db file: no plaintext of alice's secret remains anywhere.
    raw_bytes = Path(mem_path).read_bytes()
    assert b"alice-medical-secret" not in raw_bytes

    # A control subject (bob) still decrypts after the restart + alice's erase.
    vault3 = SubjectKeyVault(kv_path)
    store3 = SqliteMemoryStore(mem_path)
    gated3 = ConsentGatedMemoryStore(store3, decider=ledger.decision, key_vault=vault3)
    assert gated3.list("alice") == []  # alice's rows are gone
    [bob] = gated3.list("bob")
    assert bob.text == "bob-fact"
    store3.close()
    vault3.close()
    _ = sqlite3  # noqa: F841 - imported for explicitness of the raw scan above
