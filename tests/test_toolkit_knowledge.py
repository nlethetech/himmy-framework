"""Tests for the knowledge pack: kb_ingest + kb_search (offline, deterministic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.knowledge import register_knowledge_pack
from tests.conftest import run_async


def _registry(root: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    register_knowledge_pack(registry, ToolkitConfig(fs_root=root or Path.cwd()))
    return registry


def test_ingest_then_search_finds_text() -> None:
    reg = _registry()
    ingest = reg.handler_for("kb_ingest")
    search = reg.handler_for("kb_search")
    out = run_async(
        ingest(
            {
                "text": "Permaculture food forests layer canopy shrubs and roots",
                "title": "pc",
            }
        )
    )
    assert out["ingested"] == 1
    assert out["document_ids"]
    # The offline DeterministicEmbedder matches on exact token overlap, so the query
    # shares words with the ingested text.
    found = run_async(search({"query": "food forests layer canopy", "top_k": 3}))
    assert found["query"] == "food forests layer canopy"
    assert len(found["results"]) >= 1
    assert any("food forests" in (r["text"] or "").lower() for r in found["results"])


def test_ingest_requires_text_or_path() -> None:
    with pytest.raises(ValueError):
        run_async(_registry().handler_for("kb_ingest")({}))


def test_ingest_from_file(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("bees pollinate the orchard each spring")
    reg = _registry(tmp_path)
    out = run_async(reg.handler_for("kb_ingest")({"path": "doc.txt"}))
    assert out["ingested"] == 1
    found = run_async(
        reg.handler_for("kb_search")({"query": "bees pollinate the orchard"})
    )
    assert len(found["results"]) >= 1


def test_search_empty_kb_returns_no_results() -> None:
    found = run_async(_registry().handler_for("kb_search")({"query": "anything"}))
    assert found["results"] == []
