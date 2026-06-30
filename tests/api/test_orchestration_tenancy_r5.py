"""The /v1 orchestration path threads the run's tenant+subject into member tool packs (rbac r5).

The single-agent and Studio team paths scope every member's memory/KB packs to the launching
run's tenant (+ within-tenant subject). The ``/v1`` team / group-chat / graph / workflow path
(``himmy.application.orchestration_runner._build_team_runtime``) historically called
``build_team`` WITHOUT a ``toolkit_config``, so the members fell back to
``ToolkitConfig.from_env()`` — the static shared ``default`` memory subject / ``("local","local")``
KB scope. Two tenants' (or two users' of one tenant's) orchestration runs then pooled onto ONE
durable memory/KB namespace: a cross-tenant confused-deputy read/write.

These tests capture the ``toolkit_config`` ``build_team`` receives from ``_build_team_runtime`` and
assert it is namespaced to the owner — and that the offline / unscoped path passes no config
(byte-unchanged).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import himmy.config.team_spec as team_spec_mod
from himmy.application.orchestration_runner import _build_team_runtime


class _Member:
    def __init__(self) -> None:
        self.provider = None


class _TeamSpec:
    members = [_Member()]


def _capture_build_team(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    real = team_spec_mod.build_team

    def _spy(spec, *, toolkit_config=None, resolve_tools_module=None):  # type: ignore[no-untyped-def]
        captured["config"] = toolkit_config
        # Return a trivial (team, registry) without touching real packs.
        return object(), None

    monkeypatch.setattr(team_spec_mod, "build_team", _spy)
    # Keep a handle so the test module name resolves even if real is unused.
    captured["_real"] = real
    return captured


def _run_build(
    *, owner_workspace_id: str | None, owner_subject_scope: str | None
) -> None:
    # _build_team_runtime also builds a runtime; stub build_runtime so we exercise only the
    # toolkit_config threading into build_team (the chokepoint under test).
    import himmy.runtime.builder as builder_mod

    def _stub_build_runtime(**kwargs):  # type: ignore[no-untyped-def]
        return object(), None, None

    orig = builder_mod.build_runtime
    builder_mod.build_runtime = _stub_build_runtime  # type: ignore[assignment]
    try:
        _build_team_runtime(
            _TeamSpec(),
            storage=None,
            shared_inference=object(),
            owner_workspace_id=owner_workspace_id,
            owner_subject_scope=owner_subject_scope,
        )
    finally:
        builder_mod.build_runtime = orig  # type: ignore[assignment]


def test_orchestration_threads_tenant_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_build_team(monkeypatch)
    _run_build(owner_workspace_id="A", owner_subject_scope=None)
    cfg = captured["config"]
    assert cfg is not None, "orchestration did not pass a scoped toolkit_config to build_team"
    assert cfg.tenant_scope == "A"
    assert cfg.scoped_memory_subject() == "t:A:default"
    assert cfg.scoped_kb_keys() == ("t:A", "t:A")


def test_orchestration_threads_subject_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_build_team(monkeypatch)
    _run_build(owner_workspace_id="acme", owner_subject_scope="alice")
    cfg = captured["config"]
    assert cfg is not None
    assert cfg.tenant_scope == "acme"
    assert cfg.subject_scope == "alice"
    # Two users of ONE tenant get distinct memory/KB namespaces (cross-user isolation).
    assert cfg.scoped_memory_subject() == "t:acme:s:alice:default"
    assert cfg.scoped_kb_keys() == ("t:acme:s:alice", "t:acme:s:alice")


def test_orchestration_offline_passes_no_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tenant/subject binding → build_team gets no scoped config (byte-unchanged)."""
    captured = _capture_build_team(monkeypatch)
    _run_build(owner_workspace_id=None, owner_subject_scope=None)
    assert captured["config"] is None


def test_two_tenants_never_share_memory_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: tenant A's and tenant B's member memory subjects are disjoint."""
    seen: list[str] = []
    real = team_spec_mod.build_team

    def _spy(spec, *, toolkit_config=None, resolve_tools_module=None):  # type: ignore[no-untyped-def]
        seen.append(toolkit_config.scoped_memory_subject())
        return object(), None

    monkeypatch.setattr(team_spec_mod, "build_team", _spy)
    assert real is not None
    _run_build(owner_workspace_id="A", owner_subject_scope=None)
    _run_build(owner_workspace_id="B", owner_subject_scope=None)
    assert seen == ["t:A:default", "t:B:default"]
    assert len(set(seen)) == 2


def test_run_orchestration_forwards_owner_axes() -> None:
    """run_orchestration accepts + forwards owner_workspace_id/owner_subject_scope to the runner."""
    import himmy.application.orchestration_runner as runner

    captured: dict[str, Any] = {}

    async def _fake_multi(named, prompt, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        from himmy.application.orchestration_runner import OrchestrationOutcome

        return OrchestrationOutcome()

    orig = runner._run_multi_agent
    runner._run_multi_agent = _fake_multi  # type: ignore[assignment]
    try:
        asyncio.run(
            runner.run_orchestration(
                kind="multi_agent",
                members=[],
                prompt="hi",
                resource_kind="team",
                storage=None,
                shared_inference=None,
                operator_provisioned=False,
                owner_workspace_id="acme",
                owner_subject_scope="alice",
            )
        )
    finally:
        runner._run_multi_agent = orig  # type: ignore[assignment]
    assert captured.get("owner_workspace_id") == "acme"
    assert captured.get("owner_subject_scope") == "alice"
