"""Conformance tests for the hand-mirrored auxiliary conversation store.

The auxiliary conversation store is mirrored across two backends —
:class:`~himmy.services.storage.conversations.ConversationStore` (durable SQLite) and
:class:`~himmy.services.storage.postgres_aux.PostgresConversationStore` (the K4 Postgres
mirror) — that :func:`~himmy.services.storage.conversations.get_conversation_store` swaps
behind one handle by ``HIMMY_DATABASE_URL``. Callers hold that handle polymorphically, so
the two twins MUST expose the same method surface; a drift means a Postgres deployment
silently loses a method the SQLite path has.

These tests are the structural safety net for that parity — entirely offline (no Postgres,
no asyncpg, no live pool): they assert every backend implementation satisfies
:class:`~himmy.services.storage.protocols.ConversationStoreProtocol`, mirroring
``tests/storage/test_store_protocols.py`` for the focused core stores. They FAIL LOUDLY on
drift so a method added to one twin without its sibling is caught at test time rather than
at runtime under Postgres.

``ConversationStoreProtocol`` is a method-only ``runtime_checkable`` protocol, so
``issubclass`` verifies the surface directly on the classes — no instance construction, so
the shared aux loop is never spun up here.
"""

from __future__ import annotations

from himmy.services.storage.conversations import ConversationStore
from himmy.services.storage.postgres_aux import PostgresConversationStore
from himmy.services.storage.protocols import ConversationStoreProtocol

# Every concrete backend of the hand-mirrored conversation store.
_CONVERSATION_BACKENDS = (ConversationStore, PostgresConversationStore)


def test_conversation_backends_satisfy_the_protocol() -> None:
    """Both conversation-store twins are structural subtypes of the shared protocol."""
    for backend in _CONVERSATION_BACKENDS:
        assert issubclass(backend, ConversationStoreProtocol), (
            f"{backend.__name__} does not satisfy ConversationStoreProtocol"
        )


def test_no_backend_drops_a_sibling_method() -> None:
    """Report the exact missing method(s) if a backend drifts from the protocol surface."""
    surface = [
        name
        for name in dir(ConversationStoreProtocol)
        if not name.startswith("_")
    ]
    for backend in _CONVERSATION_BACKENDS:
        missing = [name for name in surface if not hasattr(backend, name)]
        assert not missing, f"{backend.__name__} missing {missing}"


def test_protocol_covers_prune() -> None:
    """``prune`` is part of the shared surface — the drift this cluster fixed on Postgres."""
    assert "prune" in dir(ConversationStoreProtocol)
    assert hasattr(ConversationStore, "prune")
    assert hasattr(PostgresConversationStore, "prune")
