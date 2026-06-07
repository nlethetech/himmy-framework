"""Tests for the grounding guardrail — the enforceable anti-hallucination backstop.

It blocks an agent OUTPUT that answers a real-world question from stale built-in
knowledge instead of grounding it in a tool, replacing the text with a refusal. It
is a no-op on the input stage and on already-grounded answers.
"""

from __future__ import annotations

from himmy.services.guardrails import (
    GroundingGuardrail,
    build_guardrail_pipeline,
)
from himmy.services.guardrails.builtins import _GROUNDING_REFUSAL


def _out(text: str):
    return GroundingGuardrail().inspect(text, context={"stage": "output"})


def test_blocks_and_replaces_training_data_answer() -> None:
    """The exact observed failure: 'Based on my last available data, ...' is blocked."""
    v = _out(
        "Based on my last available data, the Prime Minister of Nepal was Prachanda."
    )
    assert v.allowed is False
    assert "ungrounded" in v.flags
    assert v.text == _GROUNDING_REFUSAL
    assert "Prachanda" not in v.text  # the stale claim never reaches the user


def test_blocks_various_staleness_tells() -> None:
    """A range of 'answering from memory' disclaimers are caught."""
    for tell in (
        "As of my last knowledge update, the exchange rate was 130.",
        "I don't have access to real-time data, but the PM is X.",
        "My training data suggests the mayor is Y.",
        "I was last trained in 2023, so the latest news may differ: Z happened.",
    ):
        v = _out(tell)
        assert v.allowed is False, tell
        assert v.text == _GROUNDING_REFUSAL


def test_allows_grounded_answer() -> None:
    """A normal, tool-grounded answer passes through unchanged."""
    grounded = "According to the search results, the USD buy rate is 136.63."
    v = _out(grounded)
    assert v.allowed is True
    assert v.text == grounded


def test_noop_on_input_stage() -> None:
    """The guard only acts on output; an input prompt is never blocked/altered."""
    text = "Based on my last available data, what is X?"
    v = GroundingGuardrail().inspect(text, context={"stage": "input"})
    assert v.allowed is True
    assert v.text == text


def test_registered_and_pipeline_blocks() -> None:
    """'grounding' is resolvable by name and blocks via a pipeline (output stage)."""
    pipeline = build_guardrail_pipeline(["grounding"])
    assert "grounding" in pipeline.names
    v = pipeline.inspect(
        "Based on my training data, the president is X.",
        context={"stage": "output"},
    )
    assert v.allowed is False
    assert v.text == _GROUNDING_REFUSAL


def test_from_spec_always_enforces_grounding_on_output() -> None:
    """Every spec-built agent gets grounding on output, even with no guardrails set."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.from_spec import build_runtime_for_spec

    spec = AgentSpec(name="Bare", provider="stub")  # declares NO guardrails
    runtime, _registry = build_runtime_for_spec(spec)

    assert runtime._output_guardrail is not None
    assert "grounding" in runtime._output_guardrail.names
    # No input guardrail is forced when the spec declares none (grounding is output-only).
    assert runtime._input_guardrail is None


def test_from_spec_keeps_declared_guardrails_and_adds_grounding() -> None:
    """A spec's own guardrails are preserved on output, with grounding prepended once."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime.from_spec import build_runtime_for_spec

    spec = AgentSpec(name="Safe", provider="stub", guardrails=["pii", "grounding"])
    runtime, _registry = build_runtime_for_spec(spec)

    names = runtime._output_guardrail.names
    assert names.count("grounding") == 1  # deduped, not doubled
    assert "pii" in names
    # The spec's guardrails still guard the input prompt.
    assert runtime._input_guardrail is not None
    assert "pii" in runtime._input_guardrail.names
