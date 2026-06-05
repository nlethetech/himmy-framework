"""Tests for the memory toolkit pack: remember + recall tools."""

from __future__ import annotations

from pathlib import Path

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.memory import register_memory_pack
from tests.conftest import run_async


def _registry(config: ToolkitConfig) -> ToolRegistry:
    registry = ToolRegistry()
    register_memory_pack(registry, config)
    return registry


def test_remember_and_recall_tools() -> None:
    reg = _registry(ToolkitConfig())
    out = reg.handler_for("remember")({"text": "the duck pond feeds the orchard"})
    assert out["memory_id"]
    found = run_async(
        reg.handler_for("recall")({"query": "duck pond orchard", "top_k": 3})
    )
    assert found["results"]
    assert "duck pond" in found["results"][0]["text"]


def test_memory_pack_durable_via_config(tmp_path: Path) -> None:
    """With memory_path set, a remembered fact is recallable from a fresh pack."""
    cfg = ToolkitConfig(memory_path=str(tmp_path / "m.db"), memory_subject="s")
    _registry(cfg).handler_for("remember")({"text": "harvest is in autumn season"})

    reg2 = _registry(cfg)  # fresh pack, same sqlite file
    found = run_async(reg2.handler_for("recall")({"query": "harvest autumn season"}))
    assert found["results"]
    assert "harvest" in found["results"][0]["text"]
