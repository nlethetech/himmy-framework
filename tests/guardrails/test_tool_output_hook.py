"""WS5 — guard tool RETURN content (not just args) before it reaches the thread.

A tool that hands back unredacted PII/secrets used to land on the thread, the
next-turn context, and the final answer unfiltered. ``build_guardrail_post_hook``
closes that leak at the :class:`~himmy.services.tools.service.ToolService` post-hook
seam: PII is redacted/tokenized per the DLP policy, a BLOCK-class match withholds the
whole result behind a placeholder, detections are counted to the audit sink (never the
values), and a no-guardrail (zero-config) path passes output through unchanged.
"""

from __future__ import annotations

from himmy.services.guardrails import (
    BLOCKED_OUTPUT_PLACEHOLDER,
    GuardrailPipeline,
    build_guardrail_pipeline,
    build_guardrail_post_hook,
)
from himmy.services.guardrails.dlp import DlpAction, DlpGuardrail, DlpPolicy
from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
)
from himmy.services.tools.models import (
    ToolBackendKind,
    ToolDefinition,
    ToolExecutionResult,
)
from tests.conftest import run_async


def _result(value: object) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="c1", tool_name="t", outcome="success", result=value
    )


def _defn() -> ToolDefinition:
    return ToolDefinition(name="t", kind=ToolBackendKind.LOCAL)


# ------------------------------------------------------------- hook unit tests
def test_post_hook_redacts_pii_in_string_output() -> None:
    """A tool returning an email -> output is redacted before it leaves the hook."""
    hook = build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))
    out = run_async(hook(_result("reach me at a@b.com"), _defn()))
    assert "a@b.com" not in out
    assert "[REDACTED-EMAIL]" in out


def test_post_hook_redacts_pii_nested_in_structured_output() -> None:
    """PII nested in a dict/list result is redacted; structure is preserved."""
    hook = build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))
    payload = {"records": [{"email": "a@b.com", "id": 7}], "ok": True}
    out = run_async(hook(_result(payload), _defn()))
    assert out["ok"] is True
    assert out["records"][0]["id"] == 7
    assert "a@b.com" not in out["records"][0]["email"]


def test_post_hook_blocks_card_with_placeholder_not_leaked() -> None:
    """A BLOCK-class (card) -> whole result replaced with the placeholder, not leaked."""
    pipeline = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    hook = build_guardrail_post_hook(pipeline)
    out = run_async(hook(_result("the card is 4111 1111 1111 1111"), _defn()))
    assert out == BLOCKED_OUTPUT_PLACEHOLDER
    assert "4111" not in out


def test_post_hook_blocks_card_nested_in_structure() -> None:
    """A BLOCK match anywhere in a structure withholds the ENTIRE result."""
    pipeline = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    hook = build_guardrail_post_hook(pipeline)
    payload = {"note": "ok", "secret": "card 4111 1111 1111 1111"}
    out = run_async(hook(_result(payload), _defn()))
    assert out == BLOCKED_OUTPUT_PLACEHOLDER  # not a dict still carrying the card


def test_post_hook_audit_sink_counts_never_values() -> None:
    """Flagged classes are counted to the audit sink; the raw values never appear.

    The count is the number of result leaves that carried each guardrail flag (here
    two separate string leaves each containing an email), so a security log can record
    *that* and *how often* PII was found in tool output without ever seeing it.
    """
    seen: list[dict] = []
    hook = build_guardrail_post_hook(
        build_guardrail_pipeline(["pii"]), audit_sink=seen.append
    )
    payload = {"a": "ping a@b.com", "b": "ping c@d.com"}
    run_async(hook(_result(payload), _defn()))
    assert seen, "audit sink should have been called"
    flat = " ".join(f"{k}:{v}" for k, v in seen[0].items())
    assert "a@b.com" not in flat and "c@d.com" not in flat
    assert seen[0].get("pii:email") == 2  # two leaves flagged, counted (not values)


def test_post_hook_passes_clean_output_through() -> None:
    """A clean, structured result is returned unchanged (and stays structured)."""
    hook = build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))
    payload = {"answer": "all clear", "n": 3}
    out = run_async(hook(_result(payload), _defn()))
    assert out == payload


def test_post_hook_non_string_leaf_untouched() -> None:
    """Numeric/None leaves are left alone (nothing to inspect)."""
    hook = build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))
    assert run_async(hook(_result(42), _defn())) == 42


# --------------------------------------------------- end-to-end via ToolService
def _registry_returning(value: object) -> ToolRegistry:
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="leaky",
        handler=lambda args: value,
        description="returns a fixed value (possibly with PII)",
    )
    return registry


def test_service_post_hook_redacts_return_content() -> None:
    """End-to-end: the tool's return content is redacted before the service yields it."""
    hook = build_guardrail_post_hook(build_guardrail_pipeline(["pii"]))
    svc = ToolService(_registry_returning("contact a@b.com"), post_execution_hook=hook)
    res = run_async(svc.execute(ToolInvocation(tool_name="leaky")))
    assert res.outcome == "success"
    assert "a@b.com" not in res.result


def test_service_post_hook_blocks_card_return_content() -> None:
    """End-to-end: a BLOCK-class return is replaced with the placeholder, not leaked."""
    pipeline = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    svc = ToolService(
        _registry_returning("card 4111 1111 1111 1111"),
        post_execution_hook=build_guardrail_post_hook(pipeline),
    )
    res = run_async(svc.execute(ToolInvocation(tool_name="leaky")))
    assert res.outcome == "success"
    assert res.result == BLOCKED_OUTPUT_PLACEHOLDER
    assert "4111" not in res.result


def test_service_without_post_hook_passes_pii_through() -> None:
    """Offline regression: zero-config (no post-hook) -> output passes through verbatim."""
    svc = ToolService(_registry_returning("contact a@b.com"))  # no hook wired
    res = run_async(svc.execute(ToolInvocation(tool_name="leaky")))
    assert res.outcome == "success"
    assert res.result == "contact a@b.com"  # unredacted — guard is opt-in


def test_service_post_hook_blocked_placeholder_fails_object_schema() -> None:
    """A BLOCK on a tool with an object output schema fails closed (no leak)."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="carded",
        handler=lambda args: {"card": "4111 1111 1111 1111"},
        description="returns a card in a structured result",
        output_json_schema={
            "type": "object",
            "properties": {"card": {"type": "string"}},
        },
    )
    pipeline = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    svc = ToolService(registry, post_execution_hook=build_guardrail_post_hook(pipeline))
    res = run_async(svc.execute(ToolInvocation(tool_name="carded")))
    # Placeholder (a string) violates the object schema -> the call is denied rather
    # than leaking the card; output validation runs AFTER the post-hook by design.
    assert res.outcome == "failed"
    assert res.error_code == ToolErrorCode.OUTPUT_VALIDATION
    assert res.result is None or "4111" not in str(res.result)


# --------------------------------------------------------- from_spec wiring
def test_from_spec_wires_output_post_hook_when_guardrails_declared() -> None:
    """A spec with guardrails + tools wires the post-hook (output guard on returns)."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.from_spec import build_runtime_for_spec

    spec = AgentSpec(
        name="Guarded", provider="stub", guardrails=["pii"], tool_packs=["utils"]
    )
    runtime, _registry = build_runtime_for_spec(spec)

    # The post-hook is wired exactly where the output guardrail is configured.
    assert getattr(runtime.tool_service, "_post_hook", None) is not None
    assert getattr(runtime.tool_service, "_pre_hook", None) is not None


def test_from_spec_secrets_default_wires_hooks_zero_config() -> None:
    """Secure default: tools + NO explicit guardrails still wires the secret-redaction
    pre/post hooks (``redact_secrets`` defaults True), so a credential can never leak
    through a tool arg/return even on the zero-config path."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.from_spec import build_runtime_for_spec

    spec = AgentSpec(name="Bare", provider="stub", tool_packs=["utils"])
    runtime, _registry = build_runtime_for_spec(spec)

    assert getattr(runtime.tool_service, "_post_hook", None) is not None
    assert getattr(runtime.tool_service, "_pre_hook", None) is not None


def test_from_spec_no_hooks_when_secrets_default_opted_out() -> None:
    """Opt-out regression: tools, no guardrails, and ``redact_secrets=False`` -> the
    tool path is true passthrough (no pre/post hook)."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.from_spec import build_runtime_for_spec

    spec = AgentSpec(
        name="Bare", provider="stub", tool_packs=["utils"], redact_secrets=False
    )
    runtime, _registry = build_runtime_for_spec(spec)

    assert getattr(runtime.tool_service, "_post_hook", None) is None
    assert getattr(runtime.tool_service, "_pre_hook", None) is None
