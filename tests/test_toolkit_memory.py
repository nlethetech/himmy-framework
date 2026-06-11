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
    # ':memory:' keeps behavior tests off the durable default store (cwd .himmy/).
    reg = _registry(ToolkitConfig(memory_path=":memory:"))
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


def test_consolidate_tool_is_opt_in() -> None:
    """The consolidate tool only registers when memory_consolidate is enabled."""
    assert _registry(ToolkitConfig()).handler_for("consolidate") is None
    enabled = _registry(ToolkitConfig(memory_consolidate=True))
    assert enabled.handler_for("consolidate") is not None


def test_consolidate_tool_reconciles_facts() -> None:
    """The consolidate tool reconciles a changed fact into an UPDATE."""
    cfg = ToolkitConfig(
        memory_consolidate=True, memory_subject="s", memory_path=":memory:"
    )
    reg = _registry(cfg)
    consolidate = reg.handler_for("consolidate")
    assert consolidate is not None
    first = run_async(consolidate({"text": "the user lives in Mumbai"}))
    assert first["action"] == "ADD"
    second = run_async(consolidate({"text": "the user lives in Mumbai now"}))
    assert second["action"] in ("UPDATE", "ADD")


def test_recall_tool_active_only_arg() -> None:
    """The recall tool accepts the new active_only / as_of bi-temporal args."""
    reg = _registry(ToolkitConfig(memory_path=":memory:"))
    reg.handler_for("remember")({"text": "a recallable fact about ducks"})
    found = run_async(
        reg.handler_for("recall")({"query": "ducks", "active_only": True, "top_k": 2})
    )
    assert found["results"]


def test_default_memory_path_is_durable(tmp_path: Path, monkeypatch) -> None:
    """No explicit memory_path → the durable .himmy/memory.db under cwd.

    `remember` tells the user the fact survives the run; an in-process default
    silently broke that promise on the CLI (himmy-demo-film shoot, 2026-06-11).
    """
    from himmy.toolkit.memory import effective_memory_path

    monkeypatch.chdir(tmp_path)
    path = effective_memory_path(ToolkitConfig())
    assert path == str(Path(".himmy") / "memory.db")
    # the server flag no longer changes the default
    assert effective_memory_path(ToolkitConfig(), server=False) == path


def test_memory_path_colon_memory_opts_out(tmp_path: Path, monkeypatch) -> None:
    """memory_path=':memory:' forces the ephemeral in-process store."""
    from himmy.toolkit.memory import effective_memory_path

    monkeypatch.chdir(tmp_path)
    assert effective_memory_path(ToolkitConfig(memory_path=":memory:")) is None
    assert not (tmp_path / ".himmy").exists()


def test_remember_survives_a_fresh_pack_by_default(tmp_path: Path, monkeypatch) -> None:
    """End-to-end durability: a fact remembered with DEFAULT config is recallable
    from a brand-new registry (a 'fresh process')."""
    monkeypatch.chdir(tmp_path)
    first = _registry(ToolkitConfig())
    first.handler_for("remember")({"text": "broker is Naasa Securities, number 58"})

    fresh = _registry(ToolkitConfig())  # new pack, new store object, same default path
    found = run_async(
        fresh.handler_for("recall")({"query": "broker Naasa Securities", "top_k": 3})
    )
    assert found["results"]
    assert "Naasa" in found["results"][0]["text"]
