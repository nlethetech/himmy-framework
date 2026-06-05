"""Tests for the Nepal pack, Nepal PII guardrail, and Nepali agent mode."""

from __future__ import annotations

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.services.guardrails import build_guardrail_pipeline
from himmy.services.knowledge.local_embedders import build_embedder
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.nepal import register_nepal_pack


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_nepal_pack(registry, ToolkitConfig())
    return registry


def test_nepali_format_grouping_and_devanagari() -> None:
    out = _registry().handler_for("nepali_format")({"amount": 1234567.5})
    assert out["grouped"] == "12,34,567.50"  # Nepali lakh grouping
    assert out["npr"].startswith("रू ")
    assert "१२" in out["devanagari"]  # Devanagari numerals


def test_nepali_transliterate() -> None:
    out = _registry().handler_for("nepali_transliterate")({"text": "नेपाल"})
    assert out["normalized"] == "nepal"
    assert out["roman"]


def test_nepali_date_conversion() -> None:
    """AD→BS conversion (needs the `nepal` extra: nepali-datetime)."""
    pytest.importorskip("nepali_datetime")
    out = _registry().handler_for("nepali_date")({"date": "2025-01-01"})
    assert out["ad"] == "2025-01-01"
    assert "-" in out["bs"]  # a BS date string
    assert "/" in out["fiscal_year"]  # e.g. 2081/82


def test_nepal_pii_redacts() -> None:
    pipeline = build_guardrail_pipeline(["nepal_pii"])
    v = pipeline.inspect("reach me at +977-9812345678", context={})
    assert "9812345678" not in v.text
    assert "[REDACTED-NP-PHONE]" in v.text
    assert "pii:np_phone" in v.flags


def test_nepali_language_mode_adds_instruction() -> None:
    persona = AgentSpec(name="a", language="ne").to_persona()
    assert any("नेपाली" in instr for instr in persona.instructions)
    # default (en) adds nothing
    assert not any(
        "नेपाली" in instr for instr in AgentSpec(name="b").to_persona().instructions
    )


def test_nepali_embedder_selectable() -> None:
    from himmy.nepal.language import NepaliEmbedder

    assert isinstance(build_embedder("nepali"), NepaliEmbedder)
