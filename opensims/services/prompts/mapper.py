"""Prompts kernel: project a ContextSnapshot's fields into prompt sections.

Each selected snapshot key is rendered as a clearly delimited block. To keep the
real-inference path token-budget-safe and avoid leaking secrets into the model
context, each key supports an optional ``max_chars`` truncation cap and a
``redact`` flag. The block delimiter is a fenced, attributed wrapper rather than
a raw markdown ``###`` heading so arbitrary key names cannot be misread by the
model as instruction structure.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a context import cycle
    from opensims.services.context.models import ContextSnapshot

#: Placeholder substituted for a redacted field value.
_REDACTED = "[REDACTED]"

#: Default truncation cap applied when a key sets ``truncate=True`` without a
#: per-key ``max_chars``. ``None`` (the default) means no truncation.
_DEFAULT_MAX_CHARS = 2000


class ContextPromptKey(BaseModel):
    """A single snapshot key selected for projection into a prompt block.

    ``max_chars`` caps the rendered value length (with a truncation marker);
    ``redact`` replaces the value entirely (for sensitive fields that should be
    referenced but never shown to the model).
    """

    key: str
    required: bool = False
    max_chars: int | None = None
    redact: bool = False

    @field_validator("key", mode="before")
    @classmethod
    def _coerce_key(cls, value: Any) -> Any:
        """Allow a bare string key to coerce into ``{"key": value}``."""
        return value


def _coerce_keys(value: Any) -> Any:
    """Coerce a list of raw strings/dicts into ContextPromptKey-shaped dicts."""
    if not isinstance(value, list):
        return value
    coerced: list[Any] = []
    for item in value:
        if isinstance(item, str):
            coerced.append({"key": item})
        else:
            coerced.append(item)
    return coerced


class ContextPromptMapSpec(BaseModel):
    """Declares which snapshot keys flow into the system vs. task prompt blocks.

    ``default_max_chars`` applies a truncation cap to every key that does not set
    its own ``max_chars`` (None = unlimited, the back-compatible default).
    """

    system_keys: list[ContextPromptKey] = []
    task_keys: list[ContextPromptKey] = []
    default_max_chars: int | None = None

    @field_validator("system_keys", "task_keys", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _coerce_keys(value)


def _render_value(value: Any) -> str:
    """Render a field value as text (JSON for non-string structures)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


def _truncate(text: str, max_chars: int | None) -> str:
    """Truncate ``text`` to ``max_chars`` with a clear marker (no-op if None)."""
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"... [truncated {len(text) - max_chars} chars]"
    head = max(0, max_chars - len(marker))
    return text[:head] + marker


class ContextPromptMapper:
    """The bridge between a ``ContextSnapshot`` and a rendered prompt."""

    def project(
        self,
        snapshot: ContextSnapshot | None,
        spec: ContextPromptMapSpec | dict[str, Any] | None,
    ) -> tuple[str, str, list[str]]:
        """Project selected snapshot keys into (system_block, task_block, missing).

        Each selected key renders as a delimited, attributed block. Required keys
        absent from the snapshot are returned in the missing list. Per-key
        ``max_chars``/``redact`` (and the spec's ``default_max_chars``) control
        size and sensitivity.
        """
        if spec is None:
            return "", "", []
        if not isinstance(spec, ContextPromptMapSpec):
            spec = ContextPromptMapSpec.model_validate(spec)

        fields = self._snapshot_fields(snapshot)

        system_block, system_missing = self._render_keys(
            spec.system_keys, fields, spec.default_max_chars
        )
        task_block, task_missing = self._render_keys(
            spec.task_keys, fields, spec.default_max_chars
        )
        return system_block, task_block, system_missing + task_missing

    @staticmethod
    def _snapshot_fields(snapshot: Any) -> dict[str, Any]:
        """Extract a ``{key: value}`` mapping from a snapshot (or empty)."""
        if snapshot is None:
            return {}
        raw_fields = getattr(snapshot, "fields", None) or {}
        resolved: dict[str, Any] = {}
        for key, field in raw_fields.items():
            # ContextField exposes ``.value``; tolerate plain values too.
            resolved[key] = getattr(field, "value", field)
        return resolved

    @staticmethod
    def _format_block(key: str, rendered: str) -> str:
        """Wrap a key/value in a delimited, attributed block.

        Uses an explicit ``<context key="...">`` fence rather than a raw ``###``
        heading so an arbitrary key name cannot collide with prompt structure or
        be misread by the model as an instruction header.
        """
        return f'<context key="{key}">\n{rendered}\n</context>'

    def _render_keys(
        self,
        keys: list[ContextPromptKey],
        fields: dict[str, Any],
        default_max_chars: int | None,
    ) -> tuple[str, list[str]]:
        blocks: list[str] = []
        missing: list[str] = []
        for prompt_key in keys:
            if prompt_key.key not in fields or fields[prompt_key.key] is None:
                if prompt_key.required:
                    missing.append(prompt_key.key)
                continue
            if prompt_key.redact:
                blocks.append(self._format_block(prompt_key.key, _REDACTED))
                continue
            rendered = _render_value(fields[prompt_key.key])
            if rendered == "" and prompt_key.required:
                missing.append(prompt_key.key)
                continue
            cap = (
                prompt_key.max_chars
                if prompt_key.max_chars is not None
                else default_max_chars
            )
            rendered = _truncate(rendered, cap)
            blocks.append(self._format_block(prompt_key.key, rendered))
        return "\n\n".join(blocks), missing
