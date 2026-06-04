"""Prompts kernel: YAML-templated prompts and the context→prompt mapper."""

from __future__ import annotations

from himmy.services.prompts.manager import (
    PromptManager,
    PromptTemplate,
    SystemPromptVariables,
    TaskPromptVariables,
)
from himmy.services.prompts.mapper import (
    ContextPromptKey,
    ContextPromptMapper,
    ContextPromptMapSpec,
)

__all__ = [
    "PromptManager",
    "PromptTemplate",
    "SystemPromptVariables",
    "TaskPromptVariables",
    "ContextPromptKey",
    "ContextPromptMapSpec",
    "ContextPromptMapper",
]
