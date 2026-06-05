"""Tests for the declarative agent spec (himmy.config.agent_spec)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from himmy.config import AgentSpec, load_agent_spec
from himmy.services.inference.models import ResponseFormat


def test_to_persona_folds_role_into_metadata() -> None:
    """``role`` surfaces via Persona.metadata['role'] (and the role property)."""
    spec = AgentSpec(name="analyst", role="Research Analyst", instructions=["be terse"])
    persona = spec.to_persona()
    assert persona.name == "analyst"
    assert persona.role == "Research Analyst"
    assert persona.instructions == ["be terse"]


def test_make_task_wires_tools_and_schema_context() -> None:
    """make_task threads tool names + output schema into Task.context."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    spec = AgentSpec(name="a", tools=["word_count"], output_schema=schema)
    task = spec.make_task("hi", title="t")
    assert task.title == "t"
    assert task.prompt == "hi"
    assert task.context["tool_names"] == ["word_count"]
    assert task.context["output_schema"] == schema


def test_to_llm_config_derives_structured_output() -> None:
    """An output_schema makes the LLMConfig auto-derive STRUCTURED_OUTPUT."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    spec = AgentSpec(name="a", model="haiku", output_schema=schema, temperature=0.2)
    cfg = spec.to_llm_config()
    assert cfg.model_key == "haiku"
    assert cfg.temperature == 0.2
    assert cfg.response_format == ResponseFormat.STRUCTURED_OUTPUT


def test_to_llm_config_text_default_without_schema() -> None:
    """No schema leaves response_format unset (plain-text default path)."""
    cfg = AgentSpec(name="a").to_llm_config()
    assert cfg.model_key == "default"
    assert cfg.response_format is None


def test_load_agent_spec_yaml_roundtrip(tmp_path: Path) -> None:
    """A YAML file loads into a fully-populated AgentSpec."""
    (tmp_path / "agent.yaml").write_text(
        "name: market-analyst\n"
        "description: tech analyst\n"
        "role: Research Analyst\n"
        "instructions:\n"
        "  - Provide actionable insights.\n"
        "model: haiku\n"
        "provider: stub\n"
        "tools: [word_count]\n"
    )
    spec = load_agent_spec(tmp_path / "agent.yaml")
    assert spec.name == "market-analyst"
    assert spec.role == "Research Analyst"
    assert spec.model == "haiku"
    assert spec.provider == "stub"
    assert spec.tools == ["word_count"]


def test_load_agent_spec_resolves_schema_path(tmp_path: Path) -> None:
    """A string output_schema is read as a JSON file relative to the YAML."""
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "agent.yaml").write_text(
        "name: a\noutput_schema: schema.json\n"
    )
    spec = load_agent_spec(tmp_path / "agent.yaml")
    assert spec.output_schema == schema


def test_load_agent_spec_rejects_non_mapping(tmp_path: Path) -> None:
    """A non-mapping YAML document is a clear error, not a cryptic crash."""
    (tmp_path / "bad.yaml").write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_agent_spec(tmp_path / "bad.yaml")
