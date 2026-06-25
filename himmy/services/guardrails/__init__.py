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
    GuardrailTrigger,
    GuardrailVerdict,
)
from himmy.services.guardrails.builtins import (
    BUILTIN_GUARDRAILS,
    BlocklistGuardrail,
    GroundingGuardrail,
    InjectionGuardrail,
    NepalPIIGuardrail,
    PIIGuardrail,
    PIIRule,
    SecretsGuardrail,
    build_guardrail_pipeline,
)
from himmy.services.guardrails.dlp import (
    DEFAULT_VAULT_CAPACITY,
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    InMemoryTokenVaultBackend,
    PresidioAnalyzerAdapter,
    SqliteTokenVaultBackend,
    TokenVault,
    TokenVaultBackend,
    build_dlp_guardrail,
)
from himmy.services.guardrails.tool_hook import (
    BLOCKED_OUTPUT_PLACEHOLDER,
    build_guardrail_post_hook,
    build_guardrail_pre_hook,
)

__all__ = [
    "Guardrail",
    "GuardrailTrigger",
    "GuardrailVerdict",
    "GuardrailPipeline",
    "PIIRule",
    "PIIGuardrail",
    "SecretsGuardrail",
    "InjectionGuardrail",
    "BlocklistGuardrail",
    "NepalPIIGuardrail",
    "GroundingGuardrail",
    "BUILTIN_GUARDRAILS",
    "build_guardrail_pipeline",
    "build_guardrail_pre_hook",
    "build_guardrail_post_hook",
    "BLOCKED_OUTPUT_PLACEHOLDER",
    "DEFAULT_VAULT_CAPACITY",
    "DlpAction",
    "DlpPolicy",
    "DlpGuardrail",
    "TokenVault",
    "TokenVaultBackend",
    "InMemoryTokenVaultBackend",
    "SqliteTokenVaultBackend",
    "PresidioAnalyzerAdapter",
    "build_dlp_guardrail",
]
