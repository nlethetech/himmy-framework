"""Prompts kernel: YAML-templated system/task prompt rendering.

Prompts are sectioned YAML templates with structured variables — never strings
hardcoded in Python. The business team owns wording; editing a template never
requires a code deploy. The renderer substitutes only ``{name}`` placeholders
that map to a supplied value, leaving every other brace and ``$`` literal intact,
so a template like ``Use set {a, b}`` or ``Budget is $total`` survives verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from opensims.core.errors import OpenSimsError

#: Bundled default template, resolved relative to this file so cwd is irrelevant.
_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).parent / "configs" / "prompts" / "default_prompt.yaml"
)

#: Matches a ``{identifier}`` placeholder (valid Python identifier inside braces).
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Module-level parsed-template cache keyed by (resolved_path, mtime_ns).
_TEMPLATE_CACHE: dict[tuple[str, int], dict[str, dict[str, str]]] = {}


class SystemPromptVariables(BaseModel):
    """Typed inputs for the persona-side (system) template."""

    role: str = ""
    persona: str = ""
    objectives: list[str] = []
    skills: list[str] = []
    datetime: str = ""


class TaskPromptVariables(BaseModel):
    """Typed inputs for the task-side (user) template."""

    task: str = ""
    output_format: str = ""
    output_schema: dict[str, Any] | None = None


def _render(template_text: str, values: dict[str, str]) -> str:
    """Substitute only ``{name}`` placeholders present in ``values``.

    Placeholders whose key is absent from ``values`` are left as-is (``{name}``),
    and all other braces / ``$`` characters are preserved verbatim. This is a
    real ``{name}``-only formatter, not a global brace rewrite, so literal braces
    and dollar signs in templates or values can never be corrupted.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        return match.group(0)  # leave unknown placeholders untouched

    return _PLACEHOLDER_RE.sub(_replace, template_text)


def _join_list(items: list[str]) -> str:
    """Render a list of strings as a newline-bulleted block."""
    return "\n".join(f"- {item}" for item in items)


def _load_validated_yaml(path: Path) -> dict[str, dict[str, str]]:
    """Load one template file, validating it is a dict-of-dict-of-str.

    Caches the parsed result keyed by (resolved path, mtime) so repeated
    ``PromptManager()`` construction does not re-read or re-parse the file.
    Raises :class:`OpenSimsError` with an actionable message on malformed input.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise OpenSimsError(f"prompt template not found: {resolved}")
    mtime = resolved.stat().st_mtime_ns
    cache_key = (str(resolved), mtime)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    text = resolved.read_text(encoding="utf-8")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OpenSimsError(
            f"prompt template {resolved} is not valid YAML: {exc}"
        ) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise OpenSimsError(
            f"prompt template {resolved} must be a mapping of sections, "
            f"got {type(loaded).__name__}"
        )

    validated: dict[str, dict[str, str]] = {}
    for section, entries in loaded.items():
        if not isinstance(entries, dict):
            raise OpenSimsError(
                f"prompt template {resolved}: section {section!r} must be a "
                f"mapping of key -> template string, got {type(entries).__name__}"
            )
        section_map: dict[str, str] = {}
        for key, value in entries.items():
            if not isinstance(value, str):
                raise OpenSimsError(
                    f"prompt template {resolved}: section {section!r} key "
                    f"{key!r} must be a string, got {type(value).__name__}"
                )
            section_map[str(key)] = value
        validated[str(section)] = section_map

    _TEMPLATE_CACHE[cache_key] = validated
    return validated


class PromptTemplate:
    """Loads one or more sectioned YAML files and merges them deterministically.

    Sections map to dicts of keyed sub-templates. When multiple files are loaded,
    later files win on a per-(section, key) basis. Parsed files are cached by
    (path, mtime); malformed structure raises a clear :class:`OpenSimsError`.
    """

    def __init__(self, sections: dict[str, dict[str, str]] | None = None) -> None:
        self.sections: dict[str, dict[str, str]] = sections or {}

    @classmethod
    def from_paths(cls, paths: list[Path]) -> PromptTemplate:
        """Load and merge templates from the given YAML file paths (later wins)."""
        merged: dict[str, dict[str, str]] = {}
        for path in paths:
            data = _load_validated_yaml(Path(path))
            for section, entries in data.items():
                merged.setdefault(section, {}).update(entries)
        return cls(sections=merged)

    def section(self, name: str) -> dict[str, str]:
        """Return the sub-templates for a named section (empty dict if absent)."""
        return self.sections.get(name, {})


class PromptManager:
    """Renders system + task prompts from typed variables and a YAML template."""

    def __init__(self, *, template_paths: list[Path] | None = None) -> None:
        paths = template_paths or [_DEFAULT_TEMPLATE_PATH]
        self.template = PromptTemplate.from_paths(paths)

    def get_system_prompt(self, vars: SystemPromptVariables) -> str:
        """Render the system prompt, including only sections with non-empty values."""
        section = self.template.section("persona")
        blocks: list[str] = []

        if vars.role:
            blocks.append(
                _render(
                    section.get("role", ""),
                    {"role": vars.role, "datetime": vars.datetime},
                )
            )
        if vars.persona:
            blocks.append(
                _render(section.get("background", ""), {"persona": vars.persona})
            )
        if vars.objectives:
            blocks.append(
                _render(
                    section.get("objectives", ""),
                    {"objectives": _join_list(vars.objectives)},
                )
            )
        if vars.skills:
            blocks.append(
                _render(section.get("skills", ""), {"skills": _join_list(vars.skills)})
            )
        return "\n\n".join(b for b in blocks if b).strip()

    def get_task_prompt(self, vars: TaskPromptVariables) -> str:
        """Render the task prompt, including only sections with non-empty values."""
        section = self.template.section("task")
        blocks: list[str] = []

        if vars.task:
            blocks.append(_render(section.get("task", ""), {"task": vars.task}))
        if vars.output_format:
            blocks.append(
                _render(
                    section.get("output", ""),
                    {"output_format": vars.output_format},
                )
            )
        if vars.output_schema:
            blocks.append(
                _render(
                    section.get("schema", ""),
                    {"output_schema": json.dumps(vars.output_schema, indent=2)},
                )
            )
        return "\n\n".join(b for b in blocks if b).strip()
