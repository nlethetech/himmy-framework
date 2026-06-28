"""Red-team r6 (vuln 3): a CAPABILITY denial is NOT a resumable approval checkpoint.

The approval gate (``requires_approval``) and the capability gate (missing tool RBAC
grant) used to ``_fail`` with the IDENTICAL ``(outcome="denied", POLICY_BLOCKED)`` tuple.
The runtime's pending-approval selector keys on exactly that tuple, so a capability denial
was mis-presented as a pending HUMAN-APPROVAL checkpoint — approving it re-ran the tool,
which the capability gate re-denied (the persisted actor's roles are unchanged), wedging
the run until ``max_turns`` and misleading the operator.

Root-cause fix: the capability gate emits the DISTINCT ``ToolErrorCode.CAPABILITY_DENIED``,
and ``_pending_approvals`` keys ONLY on the approval gate's ``POLICY_BLOCKED`` — so a
capability denial is a hard failure the model sees, never a resumable checkpoint.
"""

from __future__ import annotations

from himmy.runtime.single_agent import RunResult, SingleAgentRuntime
from himmy.services.inference.models import ToolCallRecord, ToolReturnRecord
from himmy.services.tools.models import ToolErrorCode


def _result(error_code: str) -> RunResult:
    """A one-call RunResult whose single tool return is ``denied`` with ``error_code``."""
    call = ToolCallRecord(tool_call_id="c1", tool_name="send_email", args={})
    ret = ToolReturnRecord(
        tool_call_id="c1",
        tool_name="send_email",
        outcome="denied",
        metadata={"error_code": error_code},
    )
    return RunResult(
        thread=None,  # type: ignore[arg-type]
        status="paused",
        tool_calls=[call],
        tool_returns=[ret],
    )


def test_approval_denial_is_a_pending_checkpoint() -> None:
    """A ``requires_approval`` denial (POLICY_BLOCKED) IS selected as pending approval."""
    pending = SingleAgentRuntime._pending_approvals(
        _result(ToolErrorCode.POLICY_BLOCKED.value)
    )
    assert [p.tool_name for p in pending] == ["send_email"]


def test_capability_denial_is_not_a_pending_checkpoint() -> None:
    """A CAPABILITY denial is NOT mistaken for a pending human-approval checkpoint."""
    pending = SingleAgentRuntime._pending_approvals(
        _result(ToolErrorCode.CAPABILITY_DENIED.value)
    )
    assert pending == [], "capability denial must not resume as an approval gate"


def test_capability_and_approval_codes_are_distinct() -> None:
    """The two deny reasons no longer collapse to the same error code."""
    assert ToolErrorCode.CAPABILITY_DENIED != ToolErrorCode.POLICY_BLOCKED


def test_multi_agent_gated_selector_excludes_capability_denial() -> None:
    """The orchestrator's gated-tool selector mirrors the single-agent fix.

    A ``requires_approval`` denial fails the multi-agent run closed (HITL unsupported),
    but a CAPABILITY denial must NOT — it should surface as an ordinary tool failure.
    """
    from himmy.orchestrators.multi_agent import _gated_tool_denied

    assert (
        _gated_tool_denied(_result(ToolErrorCode.POLICY_BLOCKED.value)) == "send_email"
    )
    assert _gated_tool_denied(_result(ToolErrorCode.CAPABILITY_DENIED.value)) is None
