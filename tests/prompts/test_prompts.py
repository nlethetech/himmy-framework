"""Tests for the prompts kernel: YAML rendering + context→prompt projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.core.errors import HimmyError
from himmy.services.context.models import ContextField, ContextSnapshot
from himmy.services.prompts import (
    ContextPromptKey,
    ContextPromptMapper,
    ContextPromptMapSpec,
    PromptManager,
    SystemPromptVariables,
    TaskPromptVariables,
)
from himmy.services.prompts.manager import (
    PromptTemplate,
    _load_validated_yaml,
    _render,
)


def test_system_prompt_includes_only_nonempty_sections() -> None:
    """Role/objectives/skills only render when their variables are present."""
    mgr = PromptManager()
    full = mgr.get_system_prompt(
        SystemPromptVariables(
            role="analyst",
            persona="careful",
            objectives=["assess risk"],
            skills=["valuation"],
        )
    )
    assert "analyst" in full
    assert "assess risk" in full
    assert "valuation" in full

    minimal = mgr.get_system_prompt(SystemPromptVariables(role="analyst"))
    assert "analyst" in minimal
    assert "assess risk" not in minimal


def test_task_prompt_renders_task_and_schema() -> None:
    """The task prompt includes the task text and JSON schema block when supplied."""
    mgr = PromptManager()
    rendered = mgr.get_task_prompt(
        TaskPromptVariables(
            task="Summarize ACME",
            output_format="bullet list",
            output_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
    )
    assert "Summarize ACME" in rendered
    assert "bullet list" in rendered
    assert '"type": "object"' in rendered


def test_empty_task_prompt_is_blank() -> None:
    """An empty TaskPromptVariables renders to an empty string."""
    assert PromptManager().get_task_prompt(TaskPromptVariables()) == ""


def _snapshot() -> ContextSnapshot:
    return ContextSnapshot(
        subject_id="s1",
        fields={
            "company": ContextField(key="company", value="ACME Corp"),
            "metrics": ContextField(key="metrics", value={"pe": 12}),
        },
    )


def test_mapper_projects_keys_into_labelled_blocks() -> None:
    """Selected snapshot keys render as delimited, attributed blocks."""
    mapper = ContextPromptMapper()
    spec = ContextPromptMapSpec(system_keys=["company"], task_keys=["metrics"])
    sys_block, task_block, missing = mapper.project(_snapshot(), spec)
    assert '<context key="company">' in sys_block
    assert "ACME Corp" in sys_block
    assert '<context key="metrics">' in task_block
    assert '"pe": 12' in task_block
    assert missing == []


def test_mapper_reports_required_missing_keys() -> None:
    """A required key absent from the snapshot surfaces in the missing list."""
    mapper = ContextPromptMapper()
    spec = ContextPromptMapSpec(system_keys=[{"key": "absent", "required": True}])
    sys_block, _task_block, missing = mapper.project(_snapshot(), spec)
    assert "absent" in missing
    assert sys_block == ""


def test_mapper_handles_none_spec_and_snapshot() -> None:
    """A None spec/snapshot degrades to empty output (no crash)."""
    mapper = ContextPromptMapper()
    assert mapper.project(None, None) == ("", "", [])
    assert mapper.project(_snapshot(), None) == ("", "", [])


# ----------------------------------------------------------------- renderer (TP)
def test_render_leaves_literal_braces_intact() -> None:
    """A template with non-placeholder braces is preserved verbatim."""
    assert _render("Use set {a, b, c}", {"x": "1"}) == "Use set {a, b, c}"


def test_render_leaves_dollar_signs_intact() -> None:
    """A literal '$word' is not treated as a placeholder and survives."""
    assert _render("Budget is $total", {"total": "ignored"}) == "Budget is $total"


def test_render_substitutes_known_placeholders_only() -> None:
    """Only {name} placeholders present in values are substituted."""
    out = _render("Hi {name}, your id is {id}", {"name": "Bob"})
    assert out == "Hi Bob, your id is {id}"


def test_render_value_with_braces_is_safe() -> None:
    """A value that itself contains braces is inserted literally, not re-parsed."""
    out = _render("Result: {payload}", {"payload": "{nested: 1}"})
    assert out == "Result: {nested: 1}"


# ----------------------------------------------------------------- YAML validation
def test_template_validation_rejects_non_dict_section(tmp_path: Path) -> None:
    """A section authored as a list raises a clear HimmyError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("persona:\n  - just\n  - a\n  - list\n", encoding="utf-8")
    with pytest.raises(HimmyError):
        PromptTemplate.from_paths([bad])


def test_template_validation_rejects_non_string_value(tmp_path: Path) -> None:
    """A non-string template value raises a clear HimmyError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("persona:\n  role: 123\n", encoding="utf-8")
    with pytest.raises(HimmyError):
        PromptTemplate.from_paths([bad])


def test_template_caching_by_path_mtime(tmp_path: Path) -> None:
    """Parsed templates are cached, then reparsed once the file mtime changes."""
    p = tmp_path / "t.yaml"
    p.write_text('persona:\n  role: "v1 {role}"\n', encoding="utf-8")
    first = _load_validated_yaml(p)
    second = _load_validated_yaml(p)
    assert first is second  # cache hit returns the same object
    # Rewrite with a newer mtime; the new content must be picked up.
    import os

    new_mtime = p.stat().st_mtime + 10
    p.write_text('persona:\n  role: "v2 {role}"\n', encoding="utf-8")
    os.utime(p, (new_mtime, new_mtime))
    third = _load_validated_yaml(p)
    assert third["persona"]["role"] == "v2 {role}"


# ----------------------------------------------------------------- mapper guards
def test_mapper_truncates_large_values() -> None:
    """A per-key max_chars truncates the rendered value with a marker."""
    snap = ContextSnapshot(
        subject_id="s1",
        fields={"big": ContextField(key="big", value="x" * 5000)},
    )
    mapper = ContextPromptMapper()
    spec = ContextPromptMapSpec(
        system_keys=[ContextPromptKey(key="big", max_chars=100)]
    )
    sys_block, _task, _missing = mapper.project(snap, spec)
    assert "truncated" in sys_block
    # The rendered value portion stays within the cap (block fence adds a bit).
    assert len(sys_block) < 300


def test_mapper_redacts_sensitive_keys() -> None:
    """A redact=True key shows a placeholder, never the raw value."""
    snap = ContextSnapshot(
        subject_id="s1",
        fields={"api_key": ContextField(key="api_key", value="super-secret")},
    )
    mapper = ContextPromptMapper()
    spec = ContextPromptMapSpec(
        system_keys=[ContextPromptKey(key="api_key", redact=True)]
    )
    sys_block, _task, _missing = mapper.project(snap, spec)
    assert "super-secret" not in sys_block
    assert "[REDACTED]" in sys_block


def test_mapper_default_max_chars_applies() -> None:
    """A spec-level default_max_chars applies to keys without their own cap."""
    snap = ContextSnapshot(
        subject_id="s1",
        fields={"big": ContextField(key="big", value="y" * 5000)},
    )
    mapper = ContextPromptMapper()
    spec = ContextPromptMapSpec(system_keys=["big"], default_max_chars=50)
    sys_block, _task, _missing = mapper.project(snap, spec)
    assert "truncated" in sys_block
