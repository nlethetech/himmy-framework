"""Declarative agent specification: define an agent in YAML, run it without code.

The framework's primitives (:class:`~himmy.agents.personas.persona.Persona`,
:class:`~himmy.agents.base_agent.task.Task`, :class:`LLMConfig`) are normally
hand-wired in Python. :class:`AgentSpec` is a thin, declarative façade over them so a
user can describe an agent in a single ``agent.yaml`` file and drive it from the CLI::

    name: market-analyst
    description: A market research analyst specializing in tech.
    role: Research Analyst
    instructions:
      - Provide actionable insights backed by clear reasoning.
    model: default            # model_key handed to the active provider
    provider: claude-cli      # optional: stub | claude-cli | ollama | pydantic-ai
    tools: []                 # tool names to bind (registered via tools_module)
    tools_module: tools:register   # optional dotted path to register(registry)
    output_schema: null       # inline JSON Schema dict, or a path to a .json file

The loader resolves an ``output_schema`` given as a path relative to the YAML file.
Imports stay light/offline: the heavier ``LLMConfig`` is pulled in lazily inside the
methods that need it, so ``import himmy`` (which re-exports this) stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.models import LLMConfig


class AgentSpec(BaseModel):
    """A declarative description of an agent, loadable from YAML.

    Maps cleanly onto the runtime primitives: :meth:`to_persona` builds the identity,
    :meth:`make_task` builds a unit of work, and :meth:`to_llm_config` packs the model
    knobs. ``provider``/``model`` steer the inference backend the CLI selects.
    """

    name: str
    description: str = ""
    instructions: list[str] = []
    role: str | None = None
    model: str = "default"
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[str] = []
    tool_packs: list[str] = []
    tools_module: str | None = None
    guardrails: list[str] = []
    memory: bool = False  # auto-recall long-term memory into the prompt each run
    memory_top_k: int = 5
    language: str = "en"  # "ne" → instruct the agent to respond in Nepali (Devanagari)
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}

    def to_persona(self) -> Persona:
        """Project the spec into a :class:`Persona` (role folded into metadata)."""
        metadata = dict(self.metadata)
        if self.role:
            metadata.setdefault("role", self.role)
        instructions = list(self.instructions)
        if self.language == "ne":
            instructions.append(
                "कृपया सधैं नेपाली भाषामा (देवनागरी लिपिमा) जवाफ दिनुहोस्। "
                "(Always respond in the Nepali language, in Devanagari script.)"
            )
        return Persona(
            name=self.name,
            description=self.description,
            instructions=instructions,
            metadata=metadata,
        )

    def to_llm_config(self) -> LLMConfig:
        """Build the :class:`LLMConfig` carrying model_key + generation knobs.

        ``response_format`` is left for ``LLMConfig`` to auto-derive: an
        ``output_schema`` yields ``STRUCTURED_OUTPUT``, otherwise plain text.
        """
        from himmy.services.inference.models import LLMConfig

        return LLMConfig(
            model_key=self.model,
            output_json_schema=self.output_schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def make_task(self, prompt: str, *, title: str | None = None) -> Task:
        """Build a :class:`Task` for ``prompt``, wiring tool/schema context keys.

        ``context`` mirrors the LLM config so a run still binds the declared tools and
        structured-output schema even when a caller does not pass ``llm_config``.
        """
        context: dict[str, Any] = {"model_key": self.model}
        if self.tools:
            context["tool_names"] = list(self.tools)
        if self.output_schema is not None:
            context["output_schema"] = self.output_schema
        if self.memory:
            # Recall relevant long-term memory and inject it into the system prompt,
            # with no tool call. Needs a runtime wired with a memory ContextAdapter.
            context["context_build_spec"] = {
                "keys": [
                    {
                        "key": "agent_memory",
                        "adapter_name": "memory",
                        "source_preference": "tool_only",
                        "metadata": {"query": prompt},
                    }
                ]
            }
            context["context_prompt_map_spec"] = {"system_keys": ["agent_memory"]}
        return Task(
            title=title or f"{self.name}-task",
            prompt=prompt,
            context=context,
            metadata=dict(self.metadata),
        )


def load_agent_spec(path: str | Path) -> AgentSpec:
    """Load an :class:`AgentSpec` from a YAML file.

    A string ``output_schema`` is treated as a path to a JSON Schema file, resolved
    relative to the YAML file's directory, and inlined into the returned spec.
    """
    spec_path = Path(path).expanduser()
    raw = yaml.safe_load(spec_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"agent spec {spec_path} must be a YAML mapping")

    schema = raw.get("output_schema")
    if isinstance(schema, str):
        schema_path = (spec_path.parent / schema).expanduser()
        raw["output_schema"] = json.loads(schema_path.read_text())

    return AgentSpec.model_validate(raw)
