"""Tools kernel: one registry, two execution backends, policy hooks, and events."""

from __future__ import annotations

from opensims.services.tools.models import (
    HttpAuthConfig,
    HttpAuthMode,
    HttpToolConfig,
    ToolBackendKind,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
    ToolPolicyDecision,
)
from opensims.services.tools.registry import (
    ToolRegistry,
    register_http_tool,
    register_local_tool,
)
from opensims.services.tools.runtime_adapter import ToolServiceToolset, build_arg_model
from opensims.services.tools.service import ToolService
from opensims.services.tools.validation import validate_against_schema

__all__ = [
    "ToolRegistry",
    "ToolService",
    "ToolServiceToolset",
    "ToolDefinition",
    "ToolBackendKind",
    "ToolInvocation",
    "ToolPolicyDecision",
    "ToolErrorCode",
    "ToolExecutionResult",
    "HttpToolConfig",
    "HttpAuthConfig",
    "HttpAuthMode",
    "register_local_tool",
    "register_http_tool",
    "validate_against_schema",
    "build_arg_model",
]
