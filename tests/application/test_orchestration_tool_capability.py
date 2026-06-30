"""Red-team r3: the P0 tool-capability gate is threaded into the team/workflow path.

Single-agent runs already enforce the tool-capability gate (a run started by a tenant
whose role grants only narrow tool capabilities cannot invoke a write tool). The same
caller wrapping that agent in a 1-member team / workflow MUST NOT bypass the gate — that
is a confused-deputy privilege escalation reaching side-effecting member tools.

These tests build the orchestration member runtime via ``_build_team_runtime`` with an
ENFORCING :class:`ToolCapabilityAuthorizer` (the launcher's gate, as the run service now
threads it) and dispatch member tools through that runtime's tool service to prove:

* a tool the launcher's role was NOT granted (``write_action``, a side-effecting tool) is
  DENIED on the member runtime — the confused-deputy hole is closed; and
* a tool the launcher WAS granted (``read_lookup``, ``tool:read_lookup:invoke``) still
  runs;
* the OFFLINE invariant: with NO authorizer (the zero-config default) every member tool
  dispatches unconditionally — byte-unchanged.

``write_action`` is deliberately NOT approval-gated, so a denial is unambiguously the
CAPABILITY gate (never the HITL gate).
"""

from __future__ import annotations

from himmy.api.auth.rbac import AccessPolicy
from himmy.application.orchestration_runner import (
    _build_team_runtime,
    _build_team_spec,
)
from himmy.config.agent_spec import AgentSpec
from himmy.services.storage.service import StorageService
from himmy.services.tools import ToolInvocation
from himmy.services.tools.capability import ToolCapabilityAuthorizer
from tests.application import _capability_tools
from tests.conftest import run_async

_TOOLS_MODULE = "tests.application._capability_tools"


def _member_runtime(authorizer: ToolCapabilityAuthorizer | None) -> object:
    """Build a 1-member team runtime carrying the recording tools + the given gate."""
    spec = AgentSpec(
        name="worker",
        description="a member that carries a read + a write tool",
        tools=["read_lookup", "write_action"],
        tools_module=_TOOLS_MODULE,
    )
    team_spec = _build_team_spec([("worker", spec)], kind="multi_agent")
    _team, _registry, runtime = _build_team_runtime(
        team_spec,
        storage=StorageService(),
        shared_inference=None,
        tool_authorizer=authorizer,
    )
    return runtime


def _dispatch(runtime: object, tool_name: str) -> str:
    """Dispatch ``tool_name`` on the member runtime's tool service; return the outcome."""
    tool_service = runtime.tool_service  # type: ignore[attr-defined]
    result = run_async(
        tool_service.execute(
            ToolInvocation(tool_call_id="t1", tool_name=tool_name, args={})
        )
    )
    return result.outcome


def _narrow_authorizer() -> ToolCapabilityAuthorizer:
    """An enforcing gate granting ONLY tool:read_lookup:invoke (least-privilege launcher)."""
    policy = AccessPolicy.from_mapping(
        {"tenant": ["run:write", "tool:read_lookup:invoke"]}
    )
    return ToolCapabilityAuthorizer(
        enforce=True, roles=frozenset({"tenant"}), policy=policy
    )


def setup_function() -> None:
    _capability_tools.READ_CALLS.clear()
    _capability_tools.WRITE_CALLS.clear()


def test_member_write_tool_denied_when_launcher_lacks_capability() -> None:
    """A member's write tool is DENIED when the launcher's gate did not grant it."""
    runtime = _member_runtime(_narrow_authorizer())
    # ``write_action`` is side-effecting and the gate grants no tool:write_action:* — the
    # confused-deputy escalation is refused on the member runtime (capability gate, not HITL).
    assert _dispatch(runtime, "write_action") == "denied"
    assert _capability_tools.WRITE_CALLS == []  # the write tool never ran


def test_member_granted_tool_still_runs() -> None:
    """A member tool the launcher's gate DID grant (tool:read_lookup:invoke) still runs."""
    runtime = _member_runtime(_narrow_authorizer())
    assert _dispatch(runtime, "read_lookup") == "success"
    assert len(_capability_tools.READ_CALLS) == 1


def test_member_tools_unrestricted_without_authorizer() -> None:
    """INVARIANT: with NO authorizer (offline default) every member tool dispatches."""
    runtime = _member_runtime(None)
    assert _dispatch(runtime, "write_action") == "success"
    assert _dispatch(runtime, "read_lookup") == "success"
    assert len(_capability_tools.WRITE_CALLS) == 1
    assert len(_capability_tools.READ_CALLS) == 1
