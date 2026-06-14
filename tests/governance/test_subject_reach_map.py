"""S4 — subject reach map + signed DeletionCertificate (provable whole-system erasure)."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from himmy.agents.base_agent.thread import (  # noqa: E402
    ChatThread,
    Message,
    MessageRole,
)
from himmy.entities.registry import EntityRegistry  # noqa: E402
from himmy.services.governance.consent import ConsentPolicy, Purpose  # noqa: E402
from himmy.services.governance.consent_ledger import ConsentLedger  # noqa: E402
from himmy.services.governance.retention import (  # noqa: E402
    DELETION_CERTIFICATE_KIND,
    ERASURE_KIND,
    RetentionService,
    StoreErasureResult,
    SubjectKeyVault,
    SubjectReachMap,
)
from himmy.services.governance.sidecar_storage import (  # noqa: E402
    ConsentGatedConversationStore,
    ConsentGatedMemoryStore,
)
from himmy.services.memory.store import InMemoryMemoryStore, MemoryRecord  # noqa: E402
from himmy.services.storage.conversations import ConversationStore  # noqa: E402


def _wire() -> tuple[
    RetentionService,
    SubjectKeyVault,
    EntityRegistry,
    InMemoryMemoryStore,
    ConversationStore,
    ConsentGatedMemoryStore,
    ConsentGatedConversationStore,
]:
    registry = EntityRegistry()
    ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
    vault = SubjectKeyVault()
    mem_inner = InMemoryMemoryStore()
    conv_inner = ConversationStore(":memory:")
    gated_mem = ConsentGatedMemoryStore(
        mem_inner, decider=ledger.decision, key_vault=vault
    )
    gated_conv = ConsentGatedConversationStore(
        conv_inner, decider=ledger.decision, key_vault=vault
    )
    reach = SubjectReachMap(memory_store=mem_inner, conversation_store=conv_inner)
    service = RetentionService(registry, key_vault=vault, reach_map=reach)
    ledger.grant("subj", Purpose.RETAIN)
    return service, vault, registry, mem_inner, conv_inner, gated_mem, gated_conv


def _thread() -> ChatThread:
    t = ChatThread(thread_id="x")
    t.append_message(Message(role=MessageRole.USER, content="my data"))
    return t


def test_erase_reaches_memory_and_conversations_and_signs_certificate() -> None:
    service, vault, registry, mem_inner, conv_inner, gated_mem, gated_conv = _wire()

    gated_mem.save(MemoryRecord(subject_id="subj", text="a fact"))
    gated_conv.save_thread("conv1", _thread(), subject_id="subj")
    assert mem_inner.list("subj")
    assert conv_inner.conversation_ids_for_subject("subj") == ["conv1"]

    key_version = vault.key_version_of("subj")
    cert = service.erase_subject("subj", reason="GDPR #7")

    # The mutable sidecars are hard-deleted (the spine stays as undecryptable ciphertext).
    assert mem_inner.list("subj") == []
    assert conv_inner.conversation_ids_for_subject("subj") == []
    assert vault.has("subj") is False

    # The certificate lists per-store action+counts, the destroyed key_version, erased_at.
    assert cert.subject_id == "subj"
    assert cert.crypto_shredded is True
    assert cert.key_version == key_version
    assert cert.erased_at
    assert cert.complete is True
    stores = {r.store: r for r in cert.per_store}
    assert stores["memory"].deleted == 1
    assert stores["conversations"].deleted == 1

    # The certificate is persisted on the spine (signed by the audit bundle).
    certs = registry.list_by_kind(DELETION_CERTIFICATE_KIND)
    assert certs and certs[0].payload["subject_id"] == "subj"
    assert certs[0].payload["per_store"]
    # The historical tombstone is still registered too.
    assert registry.list_by_kind(ERASURE_KIND)


def test_partial_failure_is_recorded_and_rerun_idempotent() -> None:
    registry = EntityRegistry()
    vault = SubjectKeyVault()

    class _Boom:
        def delete_by_subject(self, subject_id: str) -> int:
            raise RuntimeError("store offline")

    healthy = InMemoryMemoryStore()
    healthy.save(MemoryRecord(subject_id="z", text="x"))
    reach = SubjectReachMap(memory_store=healthy, conversation_store=_Boom())
    service = RetentionService(registry, key_vault=vault, reach_map=reach)

    cert = service.erase_subject("z")
    by_store = {r.store: r for r in cert.per_store}
    assert by_store["memory"].ok is True and by_store["memory"].deleted == 1
    assert by_store["conversations"].ok is False
    assert "offline" in by_store["conversations"].error
    assert cert.complete is False  # a store failed → not complete

    # Re-running is idempotent: the healthy store already has nothing to delete.
    cert2 = service.erase_subject("z")
    assert {r.store: r.deleted for r in cert2.per_store}["memory"] == 0


def test_reach_map_skips_unwired_stores() -> None:
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    reach = SubjectReachMap()  # nothing wired (offline shape)
    service = RetentionService(registry, key_vault=vault, reach_map=reach)
    cert = service.erase_subject("nobody")
    assert cert.per_store == ()
    assert cert.complete is True


def test_spine_eraser_is_walked_and_recorded_in_certificate() -> None:
    """The S3 must_fix #1: storage.db chat_threads/run_events are reached by erasure.

    The runtime persists a governed run's transcript + events under the store-WIDE KEK
    (not per-subject), so a crypto-shred alone leaves them recoverable. The reach map's
    ``spine_eraser`` hard-deletes them and the certificate records the per-store count.
    """
    import asyncio

    from himmy.services.storage.models import RunRecord, RunStatus
    from himmy.services.storage.service import StorageService

    storage = StorageService()

    async def _seed() -> None:
        await storage.save_run(
            RunRecord(
                workspace_id="w",
                subject_id="subj",
                thread_id="t1",
                trace_id="tr1",
                status=RunStatus.SUCCEEDED,
            )
        )
        t = ChatThread(thread_id="t1")
        t.append_message(Message(role=MessageRole.USER, content="transcript secret"))
        await storage.save_thread(t)

    asyncio.run(_seed())

    registry = EntityRegistry()
    vault = SubjectKeyVault()
    reach = SubjectReachMap(spine_eraser=storage.delete_by_subject)
    service = RetentionService(registry, key_vault=vault, reach_map=reach)

    cert = service.erase_subject("subj")
    by_store = {r.store: r for r in cert.per_store}
    assert by_store["spine_storage"].deleted == 1  # the one thread row
    assert by_store["spine_storage"].ok is True
    assert cert.complete is True
    # The transcript is physically gone from the spine storage.
    assert asyncio.run(storage.load_thread("t1")) is None


def test_knowledge_eraser_callable_is_walked() -> None:
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    erased: list[str] = []

    def _erase_kb(subject_id: str) -> int:
        erased.append(subject_id)
        return 3

    reach = SubjectReachMap(knowledge_eraser=_erase_kb)
    service = RetentionService(registry, key_vault=vault, reach_map=reach)
    cert = service.erase_subject("kbsubj")
    assert erased == ["kbsubj"]
    assert {r.store: r.deleted for r in cert.per_store}["knowledge"] == 3


def test_store_erasure_result_defaults() -> None:
    r = StoreErasureResult(store="memory", action="delete", deleted=2)
    assert r.ok is True and r.error == ""
