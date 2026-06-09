"""Wire a guardrail pipeline onto the tool surface (the highest-risk 'act' path).

Two seams onto the :class:`~himmy.services.tools.service.ToolService` hooks:

* ``build_guardrail_pre_hook`` adapts a :class:`GuardrailPipeline` into a
  ``PreExecutionHook``: every string tool *argument* is inspected before the tool
  runs — blocked args deny the call, redacted args are passed through transformed.
  This means an agent cannot exfiltrate PII through ``send_email``/``http_request``
  or be steered into a blocked action via tool args.
* ``build_guardrail_post_hook`` adapts the same pipeline into a ``PostExecutionHook``
  that guards a tool's *return* content. Without it, a tool could hand back
  unredacted PII/secrets that the runtime would append to the thread, the next-turn
  context, and the final answer unfiltered — the args-only guard does not cover the
  return path. The post-hook redacts/tokenizes PII per the (DLP) policy in place, and
  on a BLOCK-class match (e.g. a credit-card number) replaces the *entire* return
  content with a safe placeholder so the secret never reaches the model or the thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from himmy.services.guardrails.base import GuardrailPipeline, GuardrailVerdict
from himmy.services.tools.models import (
    ToolDefinition,
    ToolExecutionResult,
    ToolInvocation,
    ToolPolicyDecision,
)

#: Placeholder substituted for tool return content when the guardrail pipeline
#: returns a BLOCK-class verdict (e.g. a credit-card number). The secret is never
#: surfaced; only a redaction marker reaches the model/thread.
BLOCKED_OUTPUT_PLACEHOLDER = "[BLOCKED: tool output withheld by data-loss policy]"


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


def _guard_value(
    pipeline: GuardrailPipeline,
    value: Any,
    *,
    context: dict[str, Any],
    flags: list[str],
    blocked: list[str],
) -> Any:
    """Guard one value, recursing into containers; collect flags + block reasons.

    String leaves are inspected by ``pipeline`` and replaced with the (possibly
    redacted/tokenized) verdict text. Dicts/lists/tuples are walked so PII nested in
    a structured tool result is guarded too. A BLOCK verdict anywhere records its
    reasons in ``blocked`` (the caller then withholds the *whole* result); non-block
    redactions are applied in place so partial structures stay usable.
    """
    if isinstance(value, str):
        verdict: GuardrailVerdict = pipeline.inspect(value, context=context)
        flags.extend(verdict.flags)
        if not verdict.allowed:
            blocked.extend(verdict.reasons or [f"blocked content in {context['arg']}"])
        return verdict.text
    if isinstance(value, dict):
        return {
            key: _guard_value(
                pipeline, item, context=context, flags=flags, blocked=blocked
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        guarded = [
            _guard_value(pipeline, item, context=context, flags=flags, blocked=blocked)
            for item in value
        ]
        return type(value)(guarded) if isinstance(value, tuple) else guarded
    # Non-textual leaf (int/float/bool/None): nothing to inspect.
    return value


def build_guardrail_post_hook(
    pipeline: GuardrailPipeline,
    *,
    audit_sink: Callable[[dict[str, int]], None] | None = None,
) -> Any:
    """Return a ``PostExecutionHook`` that guards a tool's return content.

    Mirrors :func:`build_guardrail_pre_hook` but on the output path: the tool result's
    ``result`` is walked (strings, and strings nested in dicts/lists) and run through
    ``pipeline``. PII is redacted/tokenized per the policy *in place* so the structure
    stays usable; a BLOCK-class match (e.g. ``card``) withholds the *entire* result and
    substitutes :data:`BLOCKED_OUTPUT_PLACEHOLDER`, so the secret never reaches the
    model, the thread, or the next-turn context. Per-class detection *counts* — the
    number of result leaves that raised each guardrail flag, derived from the flags
    and never the values — are reported to ``audit_sink`` when supplied (the DLP
    guardrail additionally reports its own per-match counts via its own sink).
    """

    async def _hook(result: ToolExecutionResult, definition: ToolDefinition) -> Any:
        context = {"stage": "tool_output", "tool": definition.name, "arg": "result"}
        flags: list[str] = []
        blocked: list[str] = []
        guarded = _guard_value(
            pipeline,
            result.result,
            context=context,
            flags=flags,
            blocked=blocked,
        )
        if audit_sink is not None and flags:
            counts: dict[str, int] = {}
            for flag in flags:
                counts[flag] = counts.get(flag, 0) + 1
            audit_sink(counts)
        if blocked:
            # A BLOCK-class match: withhold the entire result so the secret never
            # reaches the model/thread, rather than silently dropping it.
            return BLOCKED_OUTPUT_PLACEHOLDER
        return guarded

    return _hook


__all__ = [
    "build_guardrail_pre_hook",
    "build_guardrail_post_hook",
    "BLOCKED_OUTPUT_PLACEHOLDER",
]
