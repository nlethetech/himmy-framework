"""P1 self-learning: the spec opt-in flag + from_spec wiring (default OFF).

``spec.self_learning`` (default False) is the single switch. Off → no ``learned_hints``
context key and no reputation provider (zero behaviour change). On → the key is present,
COMPOSES with memory, and the runtime wires the learned-hints adapter + injects the
reputation provider into the ToolService so ``bound_tools`` reordering is active.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec
from himmy.services.tools.registry import register_local_tool

_TOOLS_MODULE = "tests.application._per_run_tools"


@pytest.fixture(autouse=True)
def _store_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep any durable store under tmp_path and clear any exported DSN."""
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))


# ------------------------------------------------------------------------- spec defaults
def test_self_learning_defaults_false_no_context_key() -> None:
    """The flag defaults off and make_task injects no learned_hints key."""
    spec = AgentSpec(name="t")
    assert spec.self_learning is False
    task = spec.make_task("hello")
    assert "context_build_spec" not in task.context


def test_self_learning_on_adds_learned_hints_key() -> None:
    """With the flag on, make_task wires the learned_hints context key + system key."""
    spec = AgentSpec(name="t", self_learning=True)
    task = spec.make_task("hello")
    keys = task.context["context_build_spec"]["keys"]
    assert [k["key"] for k in keys] == ["learned_hints"]
    assert keys[0]["adapter_name"] == "learned_hints"
    assert keys[0]["source_preference"] == "tool_only"
    assert keys[0]["metadata"]["query"] == "hello"
    assert task.context["context_prompt_map_spec"]["system_keys"] == ["learned_hints"]


def test_memory_and_self_learning_compose_both_keys() -> None:
    """Both flags on → both context keys present (neither clobbers the other)."""
    spec = AgentSpec(name="t", memory=True, self_learning=True)
    task = spec.make_task("hello")
    keys = {k["key"] for k in task.context["context_build_spec"]["keys"]}
    assert keys == {"agent_memory", "learned_hints"}
    system_keys = set(task.context["context_prompt_map_spec"]["system_keys"])
    assert system_keys == {"agent_memory", "learned_hints"}


# ----------------------------------------------------------------------- from_spec wiring
def _adapter_names(runtime: object) -> set[str]:
    ctx = runtime.context_service  # type: ignore[attr-defined]
    return {getattr(a, "name", "") for a in getattr(ctx, "_adapters", {}).values()}


def _learned_hints_adapter(runtime: object) -> object:
    ctx = runtime.context_service  # type: ignore[attr-defined]
    return getattr(ctx, "_adapters", {})["learned_hints"]


def test_self_learning_off_no_reputation_provider() -> None:
    """Off → the ToolService has no reputation provider (bound_tools unchanged)."""
    spec = AgentSpec(name="t", tools_module=_TOOLS_MODULE)
    runtime, _registry = build_runtime_for_spec(spec)
    assert runtime.tool_service._reputation_provider is None  # type: ignore[union-attr]


def test_self_learning_on_wires_adapter_and_provider() -> None:
    """On → learned_hints adapter is registered AND the reputation provider is injected."""
    spec = AgentSpec(
        name="t",
        self_learning=True,
        tools_module=_TOOLS_MODULE,
    )
    runtime, _registry = build_runtime_for_spec(spec)
    assert "learned_hints" in _adapter_names(runtime)
    assert runtime.tool_service._reputation_provider is not None  # type: ignore[union-attr]
    # The adapter must be PINNED with the registry's bound tools — the snapshot scope never
    # carries the run's tool list, so without this the hint half of the loop is a no-op.
    adapter = _learned_hints_adapter(runtime)
    assert set(adapter._tool_names or []) == {"ping", "wire_money"}  # type: ignore[attr-defined]


def test_self_learning_on_no_tools_still_wires_adapter() -> None:
    """On with no tool registry → the adapter is still wired (hints from scope tools)."""
    spec = AgentSpec(name="t", self_learning=True)
    runtime, _registry = build_runtime_for_spec(spec)
    assert "learned_hints" in _adapter_names(runtime)


def test_reprime_picks_up_late_registered_mcp_tools() -> None:
    """The re-prime hook re-reads the registry so MCP tools (registered AFTER build) are
    covered by BOTH halves of the loop — the reputation snapshot and the hint candidates.
    """
    spec = AgentSpec(name="t", self_learning=True, tools_module=_TOOLS_MODULE)
    runtime, registry = build_runtime_for_spec(spec)

    # Build-time priming only saw the tools_module tools, never the (async, late) MCP ones.
    provider = runtime.tool_service._reputation_provider  # type: ignore[union-attr]
    adapter = _learned_hints_adapter(runtime)
    assert "mcp_remote" not in provider._snapshot  # type: ignore[attr-defined]
    assert "mcp_remote" not in (adapter._tool_names or [])  # type: ignore[attr-defined]

    # Simulate attach_mcp_servers registering a tool, then driving the re-prime hook.
    async def _mcp_remote(args: dict[str, object]) -> dict[str, object]:
        return {"ok": True}

    register_local_tool(
        registry,
        name="mcp_remote",
        handler=_mcp_remote,
        description="A late-registered (MCP-like) tool.",
        args_json_schema={"type": "object", "properties": {}},
    )
    assert runtime.reprime_self_learning is not None  # type: ignore[attr-defined]
    asyncio.run(runtime.reprime_self_learning())  # type: ignore[attr-defined]

    # After the re-prime both halves now include the MCP tool (snapshot + hint candidates).
    assert "mcp_remote" in provider._snapshot  # type: ignore[attr-defined]
    assert "mcp_remote" in (adapter._tool_names or [])  # type: ignore[attr-defined]
