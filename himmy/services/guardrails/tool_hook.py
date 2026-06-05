"""Wire a guardrail pipeline onto the tool surface (the highest-risk 'act' path).

``build_guardrail_pre_hook`` adapts a :class:`GuardrailPipeline` into a
``PreExecutionHook`` for the :class:`~himmy.services.tools.service.ToolService`: every
string tool argument is inspected before the tool runs — blocked args deny the call,
redacted args are passed through transformed. This means an agent cannot exfiltrate PII
through ``send_email``/``http_request`` or be steered into a blocked action via tool args.
"""

from __future__ import annotations

from typing import Any

from himmy.services.guardrails.base import GuardrailPipeline
from himmy.services.tools.models import (
    ToolDefinition,
    ToolInvocation,
    ToolPolicyDecision,
)


def build_guardrail_pre_hook(pipeline: GuardrailPipeline) -> Any:
    """Return a ``PreExecutionHook`` that guards a tool invocation's string args."""

    async def _hook(
        invocation: ToolInvocation, definition: ToolDefinition
    ) -> ToolPolicyDecision:
        transformed: dict[str, Any] = {}
        changed = False
        for key, value in invocation.args.items():
            if not isinstance(value, str):
                transformed[key] = value
                continue
            verdict = pipeline.inspect(
                value,
                context={"stage": "tool_arg", "tool": definition.name, "arg": key},
            )
            if not verdict.allowed:
                return ToolPolicyDecision(
                    allow=False,
                    reason=f"guardrail blocked arg {key!r}: "
                    + "; ".join(verdict.reasons),
                )
            transformed[key] = verdict.text
            if verdict.text != value:
                changed = True
        return ToolPolicyDecision(
            allow=True, transformed_args=transformed if changed else None
        )

    return _hook


__all__ = ["build_guardrail_pre_hook"]
