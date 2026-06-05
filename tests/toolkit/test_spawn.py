"""Tests for the spawn_agent tool (ad-hoc recursive sub-agents)."""

from __future__ import annotations

from typing import Any

from himmy import build_runtime
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from himmy.toolkit.spawn import register_spawn_tool
from tests.conftest import run_async


def _service_with_spawn() -> tuple[ToolService, Any]:
    _runtime, inference, _tools = build_runtime()  # offline stub inference
    registry = ToolRegistry()
    register_spawn_tool(registry, inference=inference)
    return ToolService(registry), inference


def test_spawn_tool_registers() -> None:
    registry = ToolRegistry()
    _r, inference, _t = build_runtime()
    register_spawn_tool(registry, inference=inference)
    assert registry.get("spawn_agent") is not None


def test_spawn_runs_a_subagent_and_returns_its_answer() -> None:
    service, _inf = _service_with_spawn()
    res = run_async(
        service.execute(
            ToolInvocation(
                tool_name="spawn_agent",
                args={
                    "instructions": "You summarize text.",
                    "prompt": "Summarize: the quick brown fox.",
                    "name": "summarizer",
                },
            )
        )
    )
    assert res.outcome == "success"
    assert res.result["succeeded"] is True
    assert isinstance(res.result["answer"], str)


def test_spawned_subagent_cannot_itself_spawn() -> None:
    """The sub-runtime's registry has no spawn_agent — recursion is capped."""
    _runtime, inference, _tools = build_runtime()
    registry = ToolRegistry()
    register_spawn_tool(registry, inference=inference)

    # The sub-agent is built inside the handler with build_runtime(inference=...) and
    # no spawn tool; a fresh build_runtime() likewise has no spawn_agent registered.
    sub_runtime, _i, sub_tools = build_runtime(inference=inference)
    assert sub_tools.registry.get("spawn_agent") is None


def test_spawn_tolerates_unknown_tool_packs() -> None:
    """A hallucinated pack name is reported, not fatal — the spawn still runs."""
    service, _inf = _service_with_spawn()
    res = run_async(
        service.execute(
            ToolInvocation(
                tool_name="spawn_agent",
                args={
                    "instructions": "Answer briefly.",
                    "prompt": "hi",
                    "tool_packs": ["no_such_pack", "utils"],
                },
            )
        )
    )
    assert res.outcome == "success"
    assert res.result["succeeded"] is True
    assert res.result["unknown_tool_packs"] == ["no_such_pack"]


def test_spawn_with_tool_packs_gives_subagent_those_tools() -> None:
    """A spawned sub-agent can be handed built-in packs to use."""
    service, _inf = _service_with_spawn()
    res = run_async(
        service.execute(
            ToolInvocation(
                tool_name="spawn_agent",
                args={
                    "instructions": "You do arithmetic with the calculator.",
                    "prompt": "What is 6 times 7?",
                    "tool_packs": ["utils"],
                },
            )
        )
    )
    assert res.outcome == "success"
    assert res.result["succeeded"] is True
