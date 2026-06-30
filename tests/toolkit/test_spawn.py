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


# ---- rbac-harden(mopup-r1): spawned sub-agent inherits the parent's tenant/subject scope


def test_spawn_threads_tenant_subject_scope_into_subagent_packs(
    monkeypatch: Any,
) -> None:
    """The sub-agent's memory/KB packs must key off the PARENT run's tenancy axes.

    Regression for the confused-deputy leak: ``spawn_agent`` rebuilt the sub-agent's packs
    with a bare ``ToolkitConfig.from_env()`` that carried NO tenant/subject scope, so the
    sub-agent reverted to the shared static ``default`` subject / ``(local, local)`` KB —
    cross-tenant on a durable shared store. Assert the threaded scope reaches the
    ``register_packs`` config.
    """
    captured: dict[str, Any] = {}

    import himmy.toolkit as toolkit

    real_register_packs = toolkit.register_packs

    def _spy(registry: Any, packs: Any, config: Any) -> Any:
        captured["tenant_scope"] = config.tenant_scope
        captured["subject_scope"] = config.subject_scope
        return real_register_packs(registry, packs, config)

    monkeypatch.setattr(toolkit, "register_packs", _spy)

    _runtime, inference, _tools = build_runtime()
    registry = ToolRegistry()
    register_spawn_tool(
        registry, inference=inference, tenant_scope="t1", subject_scope="userA"
    )
    service = ToolService(registry)
    run_async(
        service.execute(
            ToolInvocation(
                tool_name="spawn_agent",
                args={
                    "instructions": "Use memory.",
                    "prompt": "remember nothing",
                    "tool_packs": ["memory"],
                },
            )
        )
    )
    assert captured == {"tenant_scope": "t1", "subject_scope": "userA"}, (
        "spawned sub-agent's packs did not inherit the parent's tenant/subject scope"
    )


def test_spawn_offline_scope_none_is_unchanged(monkeypatch: Any) -> None:
    """Offline / non-subject-scoped (None/None) leaves the sub-packs on the static scope."""
    captured: dict[str, Any] = {}

    import himmy.toolkit as toolkit

    real_register_packs = toolkit.register_packs

    def _spy(registry: Any, packs: Any, config: Any) -> Any:
        captured["tenant_scope"] = config.tenant_scope
        captured["subject_scope"] = config.subject_scope
        return real_register_packs(registry, packs, config)

    monkeypatch.setattr(toolkit, "register_packs", _spy)

    _runtime, inference, _tools = build_runtime()
    registry = ToolRegistry()
    register_spawn_tool(registry, inference=inference)  # no scope
    service = ToolService(registry)
    run_async(
        service.execute(
            ToolInvocation(
                tool_name="spawn_agent",
                args={
                    "instructions": "Use memory.",
                    "prompt": "x",
                    "tool_packs": ["memory"],
                },
            )
        )
    )
    assert captured == {"tenant_scope": None, "subject_scope": None}
