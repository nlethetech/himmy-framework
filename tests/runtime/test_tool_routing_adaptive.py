"""Adaptive tool routing: the tri-state ``route_tools`` default.

Explicit True/False always win. None (the spec default) routes automatically
only when the bound toolset is LARGE (> AUTO_ROUTE_OVER_TOOLS) — where the
routing call pays for itself and small models need the help — and never
routes small toolsets or tool-less runtimes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService


class _StubManager:
    provider_name = "stub"

    def resolve(self, model_key: str) -> str:
        return "stub:model"

    async def generate(self, request: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("not called")


def _runtime(tool_count: int | None) -> SingleAgentRuntime:
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_StubManager()),
        memory_store=StorageService(),
    )
    if tool_count is not None:
        defs = [
            SimpleNamespace(name=f"t{i}", description="d") for i in range(tool_count)
        ]
        rt.tool_service = SimpleNamespace(registry=SimpleNamespace(list=lambda: defs))
    return rt


def test_explicit_flag_always_wins() -> None:
    big = _runtime(tool_count=20)
    assert big._should_route(False) is False
    small = _runtime(tool_count=2)
    assert small._should_route(True) is True


def test_none_routes_only_large_toolsets() -> None:
    assert _runtime(tool_count=20)._should_route(None) is True
    assert _runtime(tool_count=9)._should_route(None) is True  # > 8
    assert _runtime(tool_count=8)._should_route(None) is False
    assert _runtime(tool_count=3)._should_route(None) is False


def test_none_without_tools_never_routes() -> None:
    assert _runtime(tool_count=None)._should_route(None) is False


def test_spec_default_is_adaptive() -> None:
    from himmy.config.agent_spec import AgentSpec

    assert AgentSpec(name="x").tool_router is None
