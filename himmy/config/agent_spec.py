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
    provider: claude-cli      # optional: stub | claude-cli | ollama | pydantic-ai | openrouter
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
from himmy.config.http_tool_spec import HttpToolSpec
from himmy.config.mcp_spec import MCPServerConfig
from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.models import LLMConfig
    from himmy.skills import SkillRegistry


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
    skills: list[str] = []  # capability bundles → tools + injected know-how
    tools: list[str] = []
    tool_packs: list[str] = []
    tools_module: str | None = None
    http_tools: list[HttpToolSpec] = []  # declarative REST tools (no Python)
    mcp_servers: list[MCPServerConfig] = []  # stdio MCP servers → agent tools
    # Built-in OUTBOUND connectors to bind as agent tools (e.g. ["slack", "github"]).
    # A named connector is wired only when it is ENABLED for outbound AND configured
    # (its capability probe passes) — an unconfigured/disabled name is skipped cleanly,
    # not an error, so a shared spec degrades gracefully across environments. Inbound
    # connectors (webhook/slack/discord intake) are mounted by the BFF, not bound here.
    connectors: list[str] = []
    # Give the agent a spawn_agent tool (one-level recursive sub-agents):
    allow_spawn: bool = False
    # Give the agent a dispatch_skill tool (run a named capability as a sub-agent):
    allow_skill_dispatch: bool = False
    # Route to the few relevant tools per query. Tri-state: True/False are
    # explicit; None (the default) is ADAPTIVE — the runtime routes
    # automatically when the bound toolset is large (>8 tools), where routing
    # pays for itself on every provider (one small routing call instead of
    # resending the full tool-schema block each turn) and rescues small local
    # models from tool-choice overload. Small toolsets skip routing entirely.
    tool_router: bool | None = None
    guardrails: list[str] = []
    memory: bool = False  # auto-recall long-term memory into the prompt each run
    memory_top_k: int = 5
    # Files/dirs to auto-ingest into the agent's knowledge base at startup (no driver
    # code). The agent gains kb_search and can answer grounded in these docs.
    knowledge: list[str] = []
    # Auto-compaction: summarize old turns once the thread crosses the token budget,
    # so long multi-turn runs don't overflow the context window.
    compact_context: bool = False
    compact_after_tokens: int = 3000
    compact_keep_recent: int = 6
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
        if self.metadata.get("skill_routing_hints"):
            context["skill_routing_hints"] = list(self.metadata["skill_routing_hints"])
        if self.compact_context:
            context["compaction_spec"] = {
                "max_tokens": self.compact_after_tokens,
                "keep_recent": self.compact_keep_recent,
            }
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

    def builds_tool_registry(self) -> bool:
        """Whether ``build_runtime_for_spec`` will construct a per-run tool registry.

        Mirrors the exact gate in :func:`himmy.runtime.from_spec.build_runtime_for_spec`
        (which is the single caller that decides this): a spec that declares any
        tool-bearing surface — packs, a tools module, declarative HTTP/MCP tools,
        knowledge auto-ingest, recursive spawn, skill dispatch, or outbound connectors —
        yields a :class:`ToolRegistry`; a bare persona spec (e.g. ``{"name": "x"}``) does
        NOT. Callers that need a registry to exist (HITL/plan mode, which register a gated
        tool onto it) gate on this so a tool-less stored agent is rejected up front rather
        than silently producing a run that can never pause. ``tools`` (bare names) alone
        does not build a registry — those names resolve against tools a ``tools_module``
        registers, so they are not counted here.
        """
        return bool(
            self.tool_packs
            or self.tools_module
            or self.http_tools
            or self.knowledge
            or self.mcp_servers
            or self.allow_spawn
            or self.allow_skill_dispatch
            or self.connectors
        )


def apply_skills(spec: AgentSpec, registry: SkillRegistry) -> AgentSpec:
    """Expand ``spec.skills`` into the spec: tools, packs, guardrails, and know-how.

    Returns a new spec with the resolved skills *consumed*: their ``tool_packs`` /
    ``tools`` / ``guardrails`` unioned in (order-stable), their instruction blocks
    appended to the description (so they reliably render as the agent's background),
    and the contributing skill names recorded in ``metadata['skills']`` — which the
    runtime's prompt renderer already surfaces as the agent's skills list. Explicit
    skill-required tool names are stashed in ``metadata`` for the runtime to validate
    against the live tool registry. A spec with no skills is returned unchanged.
    """
    if not spec.skills:
        return spec
    from himmy.skills import resolve_skills
    from himmy.skills.resolve import format_examples

    bundle = resolve_skills(spec.skills, registry)
    sections: list[str] = []
    if spec.description:
        sections.append(spec.description)
    sections.extend(bundle.instruction_blocks)
    if bundle.examples:
        sections.append(format_examples(bundle.examples))
    description = "\n\n".join(sections).strip()
    metadata = dict(spec.metadata)
    metadata["skills"] = list(bundle.skills)
    if bundle.tools:
        metadata["resolved_skill_tools"] = list(bundle.tools)
    if bundle.when_to_use:
        # Routing hints: surfaced to the tool router so it knows when each skill's
        # tools are relevant (only consulted when tool_router is enabled).
        metadata["skill_routing_hints"] = list(bundle.when_to_use)
    return spec.model_copy(
        update={
            "skills": [],  # consumed
            "description": description,
            "tools": list(dict.fromkeys([*spec.tools, *bundle.tools])),
            "tool_packs": list(dict.fromkeys([*spec.tool_packs, *bundle.tool_packs])),
            "guardrails": list(dict.fromkeys([*spec.guardrails, *bundle.guardrails])),
            "metadata": metadata,
        }
    )


def load_agent_spec(path: str | Path) -> AgentSpec:
    """Load an :class:`AgentSpec` from a YAML file.

    A string ``output_schema`` is treated as a path to a JSON Schema file, resolved
    relative to the YAML file's directory, and inlined into the returned spec.
    """
    spec_path = Path(path).expanduser()
    if not spec_path.is_file():
        raise HimmyError(
            f"agent spec not found: {spec_path} (create one with `himmy init`)"
        )
    raw = yaml.safe_load(spec_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"agent spec {spec_path} must be a YAML mapping")

    schema = raw.get("output_schema")
    if isinstance(schema, str):
        schema_path = (spec_path.parent / schema).expanduser()
        raw["output_schema"] = json.loads(schema_path.read_text())

    return AgentSpec.model_validate(raw)
