"""P1 self-learning: the spec opt-in flag + from_spec wiring (default OFF).

``spec.self_learning`` (default False) is the single switch. Off → no ``learned_hints``
context key and no reputation provider (zero behaviour change). On → the key is present,
COMPOSES with memory, and the runtime wires the learned-hints adapter + injects the
reputation provider into the ToolService so ``bound_tools`` reordering is active.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec

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


def test_self_learning_on_no_tools_still_wires_adapter() -> None:
    """On with no tool registry → the adapter is still wired (hints from scope tools)."""
    spec = AgentSpec(name="t", self_learning=True)
    runtime, _registry = build_runtime_for_spec(spec)
    assert "learned_hints" in _adapter_names(runtime)
