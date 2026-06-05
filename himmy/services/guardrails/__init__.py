"""Composable guardrails: PII redaction, prompt-injection + blocklist, over three
surfaces — tool arguments, the input prompt, and the model's output.

    from himmy.services.guardrails import build_guardrail_pipeline

    pipeline = build_guardrail_pipeline(["pii", "injection"])
    verdict = pipeline.inspect("email me at a@b.com")  # → redacted text + flags
"""

from __future__ import annotations

from himmy.services.guardrails.base import (
    Guardrail,
    GuardrailPipeline,
    GuardrailVerdict,
)
from himmy.services.guardrails.builtins import (
    BUILTIN_GUARDRAILS,
    BlocklistGuardrail,
    InjectionGuardrail,
    PIIGuardrail,
    build_guardrail_pipeline,
)
from himmy.services.guardrails.tool_hook import build_guardrail_pre_hook

__all__ = [
    "Guardrail",
    "GuardrailVerdict",
    "GuardrailPipeline",
    "PIIGuardrail",
    "InjectionGuardrail",
    "BlocklistGuardrail",
    "BUILTIN_GUARDRAILS",
    "build_guardrail_pipeline",
    "build_guardrail_pre_hook",
]
