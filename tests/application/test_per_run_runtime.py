"""UNIT 5 — secure per-run /v1 runtime: T0.2 + T0.3 + T0.4.

* T0.2: a run resolved from an ``AgentSpec`` with a tool ACTUALLY calls it (a per-run
  tool-bearing runtime built via ``from_spec.build_runtime_for_spec``) — impossible
  before this unit because the shared runtime carries no tool service. The
  inline-persona path stays on the shared tool-less runtime, unchanged.
* T0.3: a tenant-submitted spec carrying ``tools_module``/``http_tools``/
  ``mcp_servers`` (RCE/SSRF/subprocess surface) is rejected; an operator-provisioned
  spec with the operator opt-in keeps them.
* T0.4: a workspace's outstanding-run cap is enforced and concurrency is bounded
  per-workspace without affecting another workspace.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application.services import (
    RunAppService,
    WorkspaceRunQuotaExceeded,
)
from himmy.config.agent_spec import AgentSpec
from himmy.config.http_tool_spec import HttpToolSpec
from himmy.config.mcp_spec import MCPServerConfig
from himmy.config.spec_sanitizer import (
    SpecSanitizationError,
    flagged_fields,
    sanitize_tenant_spec,
)
from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.application import _per_run_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._per_run_tools"


def _run_app(
    *,
    workspace_concurrency: int = 8,
    workspace_max_outstanding: int = 64,
) -> tuple[StorageService, RunAppService, SingleAgentRuntime]:
    """Build a RunAppService over a stub-backed shared runtime + in-memory store."""
    storage = StorageService()
    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        workspace_concurrency=workspace_concurrency,
        workspace_max_outstanding=workspace_max_outstanding,
    )
    return storage, app, runtime


# --------------------------------------------------------------------- T0.2


def test_v1_run_from_spec_calls_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run resolved from a tool-bearing AgentSpec ACTUALLY calls the tool (T0.2).

    The spec names ``tools_module`` (operator field) so the only way the tool can
    fire is the per-run runtime built from ``build_runtime_for_spec`` — the shared
    runtime has no tool service. ``CALLS`` recording + TOOL_CALLED events on the
    shared store prove both the tool execution and the storage threading.
    """
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    _per_run_tools.CALLS.clear()
    _storage, app, _shared = _run_app()
    spec = AgentSpec(name="probe", tools_module=_TOOLS_MODULE, tools=["ping"])

    async def _scenario() -> None:
        run = await app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=spec.to_persona(),
            task=spec.make_task("please ping"),
            agent_spec=spec,
            operator_provisioned=True,
        )
        terminal = await app.await_run(run.run_id, timeout=10.0)
        assert terminal is not None and terminal.status.value == "SUCCEEDED"
        events = await app.get_run_events(run.run_id)
        kinds = {getattr(e, "event_type", None) for e in events}
        assert EventType.TOOL_CALLED in kinds
        assert terminal.metadata.get("agent_name") == "probe"

    run_async(_scenario())
    assert _per_run_tools.CALLS, "the agent's tool never fired on the per-run runtime"


def test_v1_run_from_spec_tool_pack_fires_for_tenant() -> None:
    """A TENANT spec (no privileged fields) with a tool_pack still calls a tool (T0.2).

    ``tool_packs`` is not part of the RCE/SSRF surface, so a plain tenant spec is
    allowed and the per-run runtime binds + the stub auto-calls one of the pack's
    tools (``calculator``/``current_time``).
    """
    _storage, app, _shared = _run_app()
    spec = AgentSpec(name="calc", tool_packs=["utils"])

    async def _scenario() -> None:
        run = await app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=spec.to_persona(),
            task=spec.make_task("what is 2+2"),
            agent_spec=spec,  # no operator_provisioned needed: no privileged fields
        )
        terminal = await app.await_run(run.run_id, timeout=10.0)
        assert terminal is not None and terminal.status.value == "SUCCEEDED"
        events = await app.get_run_events(run.run_id)
        kinds = {getattr(e, "event_type", None) for e in events}
        assert EventType.TOOL_CALLED in kinds

    run_async(_scenario())


def test_inline_persona_path_stays_tool_less_and_unchanged() -> None:
    """With NO agent_spec the run uses the shared tool-less runtime (back-compat, T0.2).

    The shared runtime carries no tool service, so even though the persona's metadata
    might hint at tools, no tool fires — proving the inline path is byte-unchanged.
    """
    _per_run_tools.CALLS.clear()
    _storage, app, shared = _run_app()
    persona = Persona(name="plain")
    task = Task(title="t", prompt="hello")

    captured: dict[str, object] = {}

    async def _scenario() -> None:
        # _resolve_runtime(None) must return the SHARED runtime object identity.
        resolved = await app._resolve_runtime(None)
        captured["is_shared"] = resolved is shared
        run = await app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        terminal = await app.await_run(run.run_id, timeout=10.0)
        assert terminal is not None and terminal.status.value == "SUCCEEDED"

    run_async(_scenario())
    assert captured["is_shared"] is True
    assert not _per_run_tools.CALLS


# --------------------------------------------------------------------- T0.3


@pytest.mark.parametrize(
    ("field", "spec_kwargs"),
    [
        ("tools_module", {"tools_module": _TOOLS_MODULE}),
        (
            "http_tools",
            {"http_tools": [HttpToolSpec(name="x", base_url="http://10.0.0.1", path="/")]},
        ),
        ("mcp_servers", {"mcp_servers": [MCPServerConfig(command="echo")]}),
        # red-team r2: sub-agent orchestration amplifiers a tenant must not self-provision.
        ("allow_spawn", {"allow_spawn": True}),
        ("allow_skill_dispatch", {"allow_skill_dispatch": True}),
    ],
)
def test_tenant_spec_with_privileged_field_is_rejected(
    field: str, spec_kwargs: dict[str, object]
) -> None:
    """Each operator-only surface field is rejected for a tenant spec (T0.3 + r2)."""
    spec = AgentSpec(name="evil", **spec_kwargs)  # type: ignore[arg-type]
    assert field in flagged_fields(spec)
    with pytest.raises(SpecSanitizationError) as exc:
        sanitize_tenant_spec(spec)  # tenant: not operator_provisioned
    assert field in exc.value.fields


def test_create_run_rejects_tenant_privileged_spec() -> None:
    """create_run itself rejects a tenant spec with tools_module (choke point, T0.3)."""
    _storage, app, _shared = _run_app()
    spec = AgentSpec(name="evil", tools_module=_TOOLS_MODULE)

    async def _scenario() -> None:
        with pytest.raises(SpecSanitizationError):
            await app.create_run(
                workspace_id="w1",
                subject_id="s1",
                persona=spec.to_persona(),
                task=spec.make_task("go"),
                agent_spec=spec,  # tenant (operator_provisioned defaults False)
            )

    run_async(_scenario())
    # Nothing was persisted for the rejected write.
    runs = run_async(_storage.list_runs(workspace_id="w1"))
    assert runs == []


def test_operator_provisioned_spec_keeps_privileged_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator-provisioned spec keeps tools_module when the operator opted in (T0.3)."""
    monkeypatch.setenv("HIMMY_ALLOW_OPERATOR_SPEC_TOOLS", "1")
    spec = AgentSpec(name="trusted", tools_module=_TOOLS_MODULE)
    result = sanitize_tenant_spec(spec, operator_provisioned=True)
    assert result.spec.tools_module == _TOOLS_MODULE
    assert result.operator_provisioned is True
    assert not result.stripped


def test_operator_provisioned_keeps_amplifiers_without_optin() -> None:
    """INVARIANT (r2): an operator-provisioned spec KEEPS allow_spawn/allow_skill_dispatch
    WITHOUT the RCE opt-in flag — so the offline / single-box path (all_tenants ⇒
    operator-provisioned) is byte-unchanged. The amplifiers are gate-protected; only a
    real tenant is blocked from self-provisioning them.
    """
    spec = AgentSpec(name="trusted", allow_spawn=True, allow_skill_dispatch=True)
    result = sanitize_tenant_spec(spec, operator_provisioned=True)
    assert result.spec is spec  # byte-identical, nothing stripped
    assert result.stripped is False
    assert result.spec.allow_spawn is True
    assert result.spec.allow_skill_dispatch is True


def test_operator_provisioned_rce_still_rejected_even_with_amplifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator spec carrying an RCE field (no opt-in) is still rejected, amplifier or not."""
    spec = AgentSpec(
        name="trusted", tools_module=_TOOLS_MODULE, allow_spawn=True
    )
    with pytest.raises(SpecSanitizationError) as exc:
        sanitize_tenant_spec(spec, operator_provisioned=True)
    # The reject names the RCE field (the high-blast-radius one needing the opt-in).
    assert "tools_module" in exc.value.fields


def test_operator_spec_still_sanitized_without_optin() -> None:
    """Even an operator-provisioned spec is rejected when the opt-in flag is OFF (T0.3).

    The privileged fields cannot be reached by configuration accident — the operator
    must explicitly set ``HIMMY_ALLOW_OPERATOR_SPEC_TOOLS``.
    """
    spec = AgentSpec(name="trusted", tools_module=_TOOLS_MODULE)
    with pytest.raises(SpecSanitizationError):
        sanitize_tenant_spec(spec, operator_provisioned=True)


def test_strip_mode_clears_fields_instead_of_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip mode clears the privileged fields and returns a sanitized copy (T0.3)."""
    monkeypatch.setenv("HIMMY_SANITIZE_SPEC_STRIP", "1")
    spec = AgentSpec(
        name="evil",
        tools_module=_TOOLS_MODULE,
        http_tools=[HttpToolSpec(name="x", base_url="http://10.0.0.1", path="/")],
        mcp_servers=[MCPServerConfig(command="echo")],
    )
    result = sanitize_tenant_spec(spec)
    assert result.stripped is True
    assert set(result.flagged) == {"tools_module", "http_tools", "mcp_servers"}
    assert result.spec.tools_module is None
    assert result.spec.http_tools == []
    assert result.spec.mcp_servers == []
    # The caller's input object was not mutated.
    assert spec.tools_module == _TOOLS_MODULE


def test_clean_spec_is_passthrough() -> None:
    """A spec with no privileged fields is returned unchanged (no-op fast path, T0.3)."""
    spec = AgentSpec(name="ok", tool_packs=["utils"])
    result = sanitize_tenant_spec(spec)
    assert result.spec is spec
    assert result.flagged == []
    assert result.stripped is False


# --------------------------------------------------------------------- T0.4


def test_workspace_outstanding_cap_rejects_burst() -> None:
    """A burst beyond the per-workspace outstanding cap is rejected (T0.4).

    The runtime is made to block so runs stay in-flight; the (cap+1)-th create_run
    for the same workspace raises WorkspaceRunQuotaExceeded, while a DIFFERENT
    workspace is unaffected.
    """
    _storage, app, _shared = _run_app(workspace_max_outstanding=2)
    gate = asyncio.Event()

    async def _blocking_run_task_detailed(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        await gate.wait()
        raise AssertionError("unreachable in this test")

    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario() -> None:
        # Block the shared runtime so every accepted run stays outstanding.
        app._runtime.run_task_detailed = _blocking_run_task_detailed  # type: ignore[assignment,method-assign]
        await app.create_run(
            workspace_id="wA", subject_id="s", persona=persona, task=task
        )
        await app.create_run(
            workspace_id="wA", subject_id="s", persona=persona, task=task
        )
        # Let the background tasks reach the blocking point.
        await asyncio.sleep(0.05)
        assert app.workspace_outstanding("wA") == 2
        with pytest.raises(WorkspaceRunQuotaExceeded) as exc:
            await app.create_run(
                workspace_id="wA", subject_id="s", persona=persona, task=task
            )
        assert exc.value.cap == 2
        # A different workspace is unaffected by wA's cap.
        await app.create_run(
            workspace_id="wB", subject_id="s", persona=persona, task=task
        )
        assert app.workspace_outstanding("wB") == 1
        # Release everything and drain.
        gate.set()
        await app.drain(timeout=5.0)

    run_async(_scenario())


def test_outstanding_slot_released_on_completion() -> None:
    """A completed run frees its outstanding slot so the cap is not permanently spent."""
    _storage, app, _shared = _run_app(workspace_max_outstanding=1)
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario() -> None:
        run1 = await app.create_run(
            workspace_id="wA", subject_id="s", persona=persona, task=task
        )
        t1 = await app.await_run(run1.run_id, timeout=10.0)
        assert t1 is not None and t1.status.value == "SUCCEEDED"
        assert app.workspace_outstanding("wA") == 0
        # The freed slot lets a second run through (cap is 1).
        run2 = await app.create_run(
            workspace_id="wA", subject_id="s", persona=persona, task=task
        )
        t2 = await app.await_run(run2.run_id, timeout=10.0)
        assert t2 is not None and t2.status.value == "SUCCEEDED"

    run_async(_scenario())


def test_idempotent_resubmit_does_not_consume_a_slot() -> None:
    """An idempotent re-submit returns the prior run without spending a new slot (T0.4)."""
    _storage, app, _shared = _run_app(workspace_max_outstanding=1)
    gate = asyncio.Event()

    async def _blocking(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        await gate.wait()
        raise AssertionError("unreachable")

    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario() -> None:
        app._runtime.run_task_detailed = _blocking  # type: ignore[assignment,method-assign]
        first = await app.create_run(
            workspace_id="wA",
            subject_id="s",
            persona=persona,
            task=task,
            idempotency_key="k",
        )
        await asyncio.sleep(0.02)
        assert app.workspace_outstanding("wA") == 1
        # Same key: returns the prior run, no new slot, no quota error even at cap=1.
        second = await app.create_run(
            workspace_id="wA",
            subject_id="s",
            persona=persona,
            task=task,
            idempotency_key="k",
        )
        assert second.run_id == first.run_id
        assert app.workspace_outstanding("wA") == 1
        gate.set()
        await app.drain(timeout=5.0)

    run_async(_scenario())


def test_per_workspace_concurrency_is_bounded() -> None:
    """At most ``workspace_concurrency`` of a workspace's runs execute at once (T0.4)."""
    _storage, app, _shared = _run_app(
        workspace_concurrency=2, workspace_max_outstanding=0
    )
    active = 0
    peak = 0
    release = asyncio.Event()

    async def _counting(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await release.wait()
        finally:
            active -= 1
        raise AssertionError("unreachable")

    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario() -> None:
        app._runtime.run_task_detailed = _counting  # type: ignore[assignment,method-assign]
        for _ in range(5):
            await app.create_run(
                workspace_id="wA", subject_id="s", persona=persona, task=task
            )
        await asyncio.sleep(0.05)
        # The semaphore caps concurrent execution at 2 even with 5 runs queued.
        assert peak == 2
        release.set()
        await app.drain(timeout=5.0)

    run_async(_scenario())


def test_run_quota_breach_surfaces_as_http_429() -> None:
    """A workspace over its outstanding-run cap returns a clean 429 over HTTP (T0.4).

    The quota error must not fall through to the generic 500 handler — it is
    backpressure, so the BFF maps :class:`WorkspaceRunQuotaExceeded` to a structured
    ``429`` (the exception's documented contract). We block the runtime so the first
    run stays in-flight, set the cap to 1, and assert the second create is a 429 with
    the ``run_quota_exceeded`` code while a different workspace still gets a 200.
    """
    from fastapi.testclient import TestClient

    from himmy.api import ApiContainer, create_app

    container = ApiContainer.build_default()
    container.run_app._workspace_max_outstanding = 1  # type: ignore[attr-defined]
    gate = asyncio.Event()

    async def _blocking(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
        await gate.wait()
        raise AssertionError("unreachable")

    container.run_app._runtime.run_task_detailed = _blocking  # type: ignore[attr-defined]

    body = {
        "workspace_id": "wA",
        "subject_id": "s1",
        "persona": {"name": "A", "description": ""},
        "task": {"title": "t", "prompt": "hi"},
    }
    try:
        with TestClient(create_app(container)) as client:
            first = client.post("/v1/runs", json=body)
            assert first.status_code == 200

            # wA is now at cap (1 in-flight, blocked) — the next create is rejected.
            second = client.post("/v1/runs", json=body)
            assert second.status_code == 429
            payload = second.json()
            assert payload["code"] == "run_quota_exceeded"
            assert "cap" in payload["detail"].lower()

            # A DIFFERENT workspace is unaffected by wA's cap.
            other = client.post("/v1/runs", json={**body, "workspace_id": "wB"})
            assert other.status_code == 200
    finally:
        gate.set()
        run_async(container.run_app.drain(timeout=5.0))
