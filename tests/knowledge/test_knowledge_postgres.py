"""Docker-gated integration tests for the pgvector knowledge backend (CK-1, CK-2).

These run only when ``HIMMY_TEST_POSTGRES_DSN`` points at a live
``pgvector/pgvector`` database (see ``docker/docker-compose.yml``) and ``asyncpg`` +
``pgvector`` are installed; otherwise every test SKIPs so the offline-first suite
stays green with only pydantic/pyyaml/httpx/fastapi present.

Run locally with:

    docker compose -f docker/docker-compose.yml up -d
    HIMMY_TEST_POSTGRES_DSN=postgresql://himmy:himmy@localhost:5433/himmy \\
        python3 -m pytest -q tests/knowledge/test_knowledge_postgres.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.conftest import run_async

_DSN = os.environ.get("HIMMY_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not _DSN, reason="HIMMY_TEST_POSTGRES_DSN not set (docker-gated)"
)


def _require_backend_deps() -> None:
    """Skip the test when asyncpg / pgvector are not importable."""
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        pytest.skip("asyncpg not installed")


def _new_storage():
    """Connect a PostgresStorageService + create both schemas (storage + knowledge)."""
    from himmy.services.storage.postgres import PostgresStorageService

    storage = run_async(PostgresStorageService.connect(_DSN))
    run_async(storage.create_schema())
    run_async(storage.create_knowledge_schema(vector_dim=64))
    return storage


def test_pgvector_roundtrip_ingest_and_search() -> None:
    """Create KB -> ingest -> cosine search end-to-end against a real database."""
    _require_backend_deps()
    from himmy.services.knowledge import DeterministicEmbedder, KnowledgeBase

    storage = _new_storage()
    try:
        kb_service = KnowledgeBase(
            storage=storage,
            embedder=DeterministicEmbedder(dim=64),
            backend=storage.knowledge_backend(),
        )
        name = f"kb-{uuid.uuid4().hex[:8]}"
        rec = run_async(
            kb_service.create_kb(
                workspace_id="w1", client_id="c1", name=name, vector_dim=64
            )
        )
        run_async(
            kb_service.ingest_text(
                rec.kb_id, "ACME quarterly revenue rose on strong widget demand."
            )
        )
        run_async(kb_service.ingest_text(rec.kb_id, "Unrelated note about tomatoes."))
        hits = run_async(kb_service.search(rec.kb_id, "ACME revenue widgets", top_k=1))
        assert hits and "ACME" in (hits[0].text or "")
        assert hits[0].similarity > 0.0
        # Tenancy guard: wrong workspace cannot reach the KB.
        from himmy.core.errors import HimmyError

        with pytest.raises(HimmyError):
            run_async(
                kb_service.search(
                    rec.kb_id, "ACME", workspace_id="other", client_id="c1"
                )
            )
        # Cascade delete removes the KB.
        assert run_async(kb_service.delete_kb(rec.kb_id)) is True
    finally:
        run_async(storage.close())


def test_pgvector_duplicate_kb_name_raises() -> None:
    """The (workspace, client, name) unique index is enforced at the DB."""
    _require_backend_deps()
    from himmy.core.errors import HimmyError
    from himmy.services.knowledge import DeterministicEmbedder, KnowledgeBase

    storage = _new_storage()
    try:
        kb_service = KnowledgeBase(
            storage=storage,
            embedder=DeterministicEmbedder(dim=64),
            backend=storage.knowledge_backend(),
        )
        name = f"dup-{uuid.uuid4().hex[:8]}"
        run_async(
            kb_service.create_kb(
                workspace_id="w1", client_id="c1", name=name, vector_dim=64
            )
        )
        with pytest.raises(HimmyError):
            run_async(
                kb_service.create_kb(
                    workspace_id="w1", client_id="c1", name=name, vector_dim=64
                )
            )
    finally:
        run_async(storage.close())


def test_openai_compatible_embedder_real_call() -> None:
    """Real OpenAI-compatible embedding call (skips without configured creds)."""
    if not os.environ.get("OPENAI_COMPATIBLE_API_KEY") and not os.environ.get(
        "OPENAI_API_KEY"
    ):
        pytest.skip("no OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY configured")
    if not os.environ.get("OPENAI_COMPATIBLE_EMBED_MODEL"):
        pytest.skip("OPENAI_COMPATIBLE_EMBED_MODEL not set")
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed")

    from himmy.services.knowledge import build_openai_compatible_embedder

    emb = build_openai_compatible_embedder()
    vecs = run_async(emb.embed_documents(["hello", "world"]))
    assert len(vecs) == 2 and all(len(v) > 0 for v in vecs)
    q = run_async(emb.embed_query("hello"))
    assert len(q) == len(vecs[0])
