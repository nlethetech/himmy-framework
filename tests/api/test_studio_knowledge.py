"""Tests for the Studio Knowledge browser (KB wrap)."""

from __future__ import annotations

import pytest

from himmy.api import studio_knowledge as k
from tests.conftest import run_async


@pytest.fixture(autouse=True)
def _fresh():
    k.reset_kb_service()
    yield
    k.reset_kb_service()


def test_create_ingest_search() -> None:
    async def _go():
        kb = await k.create_kb("docs")
        await k.ingest_text(
            kb.kb_id, "Ducks lay around 300 eggs a year.", title="ducks"
        )
        kbs = k.list_kbs()
        assert kbs[0].name == "docs" and kbs[0].documents == 1
        # Retrieval runs and returns a list (semantic quality depends on the
        # embedder; the bundled deterministic one is not meaning-aware).
        hits = await k.search(kb.kb_id, "ducks eggs", top_k=3)
        assert isinstance(hits, list)
        assert await k.delete_kb(kb.kb_id) is True
        assert k.list_kbs() == []

    run_async(_go())
