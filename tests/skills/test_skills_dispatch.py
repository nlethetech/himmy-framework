"""Tier 4: skill-as-subagent dispatch + skill-aware benchmark wiring."""

from __future__ import annotations

from pathlib import Path

from himmy.benchmark.models import BenchmarkSuite
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.tools.registry import ToolRegistry
from himmy.skills import (
    SkillDispatcher,
    register_skill_dispatch_tool,
)
from tests.conftest import run_async


def _inference() -> InferenceService:
    return InferenceService(client_manager=StubClientManager())


def test_dispatcher_runs_a_skill_in_isolation() -> None:
    # data_analysis binds the data pack; the stub drives one tool turn then answers.
    result = run_async(
        SkillDispatcher(inference=_inference()).run("data_analysis", "list the tables")
    )
    assert result["skill"] == "data_analysis"
    assert result["tool_packs"] == ["data"]
    assert "answer" in result


def test_dispatcher_toolless_skill() -> None:
    result = run_async(
        SkillDispatcher(inference=_inference()).run("summarize", "shorten this text")
    )
    assert result["skill"] == "summarize"
    assert result["tool_packs"] == []


def test_dispatch_tool_registers_and_lists_skills() -> None:
    from himmy.skills import SkillRegistry

    registry = ToolRegistry()
    register_skill_dispatch_tool(
        registry,
        inference=_inference(),
        skill_registry=SkillRegistry.with_builtins(),
    )
    tool = registry.get("dispatch_skill")
    assert tool is not None
    assert "data_analysis" in tool.description  # advertises the catalog


def test_dispatch_tool_handles_unknown_skill() -> None:
    registry = ToolRegistry()
    register_skill_dispatch_tool(registry, inference=_inference())
    handler = registry.handler_for("dispatch_skill")
    out = run_async(handler({"skill": "nope_not_real", "prompt": "x"}))
    assert out["answer"] == ""
    assert "error" in out


def test_benchmark_task_accepts_skills() -> None:
    suite = BenchmarkSuite.from_dict(
        {
            "name": "t",
            "tasks": [
                {
                    "id": "s1",
                    "prompt": "q",
                    "skills": ["data_analysis"],
                    "grade": {"type": "numeric", "value": 1},
                }
            ],
        }
    )
    assert suite.tasks[0].skills == ["data_analysis"]
    # The suite's pack union picks up the pack the skill implies.
    assert "data" in suite.packs


def test_skills_suite_file_loads() -> None:
    path = Path("himmy/benchmark/suites/skills.yaml")
    suite = BenchmarkSuite.from_yaml(path)
    assert suite.name == "skills"
    ids = {t.id for t in suite.tasks}
    assert {"skill_sql", "skill_file", "skill_summarize"} <= ids


# ---- rbac-harden(mopup-r1): dispatched skill sub-agent inherits parent tenant/subject scope


def test_dispatch_threads_tenant_subject_scope_into_skill_packs(monkeypatch) -> None:
    """A dispatched skill's tool packs must key off the parent run's tenancy axes.

    Regression for the same confused-deputy leak as spawn: ``dispatch_skill`` rebuilt the
    skill's packs with a bare ``ToolkitConfig.from_env()`` carrying no tenant/subject scope,
    so memory/KB reverted to the shared static store. Assert the threaded scope reaches the
    ``register_packs`` config.
    """
    import himmy.toolkit as toolkit

    captured: dict[str, object] = {}
    real = toolkit.register_packs

    def _spy(registry, packs, config):
        captured["tenant_scope"] = config.tenant_scope
        captured["subject_scope"] = config.subject_scope
        return real(registry, packs, config)

    monkeypatch.setattr(toolkit, "register_packs", _spy)

    run_async(
        SkillDispatcher(
            inference=_inference(), tenant_scope="t1", subject_scope="userA"
        ).run("data_analysis", "list the tables")
    )
    assert captured == {"tenant_scope": "t1", "subject_scope": "userA"}, (
        "dispatched skill's packs did not inherit the parent's tenant/subject scope"
    )


def test_dispatch_offline_scope_none_is_unchanged(monkeypatch) -> None:
    """Offline / non-subject-scoped (None/None) leaves the skill packs on the static scope."""
    import himmy.toolkit as toolkit

    captured: dict[str, object] = {}
    real = toolkit.register_packs

    def _spy(registry, packs, config):
        captured["tenant_scope"] = config.tenant_scope
        captured["subject_scope"] = config.subject_scope
        return real(registry, packs, config)

    monkeypatch.setattr(toolkit, "register_packs", _spy)

    run_async(SkillDispatcher(inference=_inference()).run("data_analysis", "list tables"))
    assert captured == {"tenant_scope": None, "subject_scope": None}
