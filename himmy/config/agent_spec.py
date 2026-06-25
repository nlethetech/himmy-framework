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
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.config.http_tool_spec import HttpToolSpec
from himmy.config.mcp_spec import MCPServerConfig
from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.models import LLMConfig
    from himmy.skills import SkillRegistry

#: A pydantic model type the shared spec validator returns (preserves the concrete type).
_SpecModelT = TypeVar("_SpecModelT", bound=BaseModel)


class AgentSpec(BaseModel):
    """A declarative description of an agent, loadable from YAML.

    Maps cleanly onto the runtime primitives: :meth:`to_persona` builds the identity,
    :meth:`make_task` builds a unit of work, and :meth:`to_llm_config` packs the model
    knobs. ``provider``/``model`` steer the inference backend the CLI selects.

    ``extra="forbid"`` so a typo'd top-level field (e.g. ``gaurdrails:``) fails loudly
    instead of being silently dropped — leaving the agent running without its config.
    The open ``metadata`` dict stays for forward-compat / free-form keys.
    """

    model_config = ConfigDict(extra="forbid")

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
    # Secure-by-default credential redaction. When True (the default), himmy redacts
    # CREDENTIALS ONLY (API keys, JWTs, URL-embedded credentials) from every outbound
    # vector — tool arguments, tool results, and the final answer — for EVERY spec-built
    # agent (including ones created in Studio). Unlike full ``pii`` (email/phone/...), the
    # secrets subset never appears in legitimate agent output, so this is safe to enforce
    # by default and closes the "a spec that omits guardrails leaks secrets" hole. Set
    # False to opt out entirely; add ``"pii"`` to ``guardrails`` for personal-data redaction.
    redact_secrets: bool = True
    memory: bool = False  # auto-recall long-term memory into the prompt each run
    memory_top_k: int = 5
    # Self-learning (P1, opt-in): mine recorded TOOL_FAILED/TOOL_COMPLETED signals into a
    # per-tool reputation that (a) stable-sorts flaky tools after reliable ones in the
    # bound toolset and (b) injects a short reliability hint into the prompt. Default
    # False → zero behaviour change. Needs a runtime wired with the learning service.
    self_learning: bool = False
    # Self-learning (P2, outcome quality, opt-in): blend an OUTCOME-quality signal (was the
    # ANSWER good, from the LLM judge / a deterministic grader / user feedback) into the
    # P1 operational reputation (did the tool RUN). ``outcome_weight`` in [0, 1] is the
    # convex blend weight: 0.0 (the default) means outcome scores are recorded but NEVER
    # blended, so reputation/hints/reorder stay byte-identical to P1; >0.0 mixes the
    # outcome-weighted success rate into the score. Inert unless ``self_learning`` is also
    # on. ``trajectory_hints`` (default off) adds run-sequence-aware hints ("you appear to
    # be looping / don't call X first") mined from the run's OWN tool-call order.
    outcome_weight: float = 0.0
    trajectory_hints: bool = False
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

    @field_validator("outcome_weight")
    @classmethod
    def _check_outcome_weight(cls, v: float) -> float:
        """``outcome_weight`` is a convex blend weight — it must live in [0, 1]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"outcome_weight must be in [0, 1], got {v}")
        return v

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
        # Memory + self-learning both inject a system-prompt context block, with no tool
        # call, via a context adapter. They COMPOSE: each appends its key rather than
        # replacing the whole spec, so turning on both wires both blocks (a naive
        # re-assignment would silently clobber the other). Needs a runtime wired with the
        # matching ContextAdapter(s).
        build_keys: list[dict[str, Any]] = []
        system_keys: list[str] = []
        if self.memory:
            # Recall relevant long-term memory and inject it into the system prompt.
            build_keys.append(
                {
                    "key": "agent_memory",
                    "adapter_name": "memory",
                    "source_preference": "tool_only",
                    "metadata": {"query": prompt},
                }
            )
            system_keys.append("agent_memory")
        if self.self_learning:
            # Inject a short reliability note about recently-flaky tools.
            build_keys.append(
                {
                    "key": "learned_hints",
                    "adapter_name": "learned_hints",
                    "source_preference": "tool_only",
                    "metadata": {"query": prompt},
                }
            )
            system_keys.append("learned_hints")
        if build_keys:
            context["context_build_spec"] = {"keys": build_keys}
            context["context_prompt_map_spec"] = {"system_keys": system_keys}
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


#: Top-level keys owned by a SIBLING schema that may legitimately share one YAML file
#: with an agent spec. A ``workflow.yaml`` is loaded both as a :class:`AgentSpec` (for
#: name/model/provider/tools the steps run under) and as a workflow ``steps`` list, so
#: ``steps`` is a known cross-schema field — not a typo — and is dropped here before the
#: strict (``extra="forbid"``) validation rather than rejected.
_SIBLING_SCHEMA_KEYS = frozenset({"steps"})


def parse_spec_yaml(path: Path, *, kind: str = "spec") -> dict[str, Any]:
    """Read + YAML-parse ``path`` into a mapping, naming the file on any failure.

    A raw ``yaml.YAMLError`` is a multi-line parser dump with no filename — useless
    in a CLI traceback and an opaque 500 in Studio. We catch it and re-raise a clean
    :class:`HimmyError` that leads with the offending file (and the parser's first
    line), so the user knows *which* file to fix. ``kind`` labels the spec in the
    message (e.g. ``"agent spec"``).
    """
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise HimmyError(f"{kind} {path}: invalid YAML — {detail}") from exc
    if not isinstance(raw, dict):
        raise HimmyError(f"{kind} {path} must be a YAML mapping")
    return raw


def validate_spec(
    model: type[_SpecModelT], raw: dict[str, Any], path: Path, *, kind: str = "spec"
) -> _SpecModelT:
    """Validate ``raw`` against ``model``, naming the file on a schema error.

    A raw pydantic ``ValidationError`` escapes as a multi-line dump with no filename;
    we re-raise a :class:`HimmyError` that leads with the file and keeps the field-level
    error lines (dropping pydantic's "For further information…" footer), mirroring the
    one-liner findings that ``himmy validate`` shows.
    """
    from pydantic import ValidationError

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        lines = [
            line.strip()
            for line in str(exc).splitlines()
            if line.strip() and not line.strip().startswith(("For further", "Traceback"))
        ]
        detail = "; ".join(lines) if lines else str(exc)
        raise HimmyError(f"{kind} {path} is invalid: {detail}") from exc


def load_agent_spec(path: str | Path) -> AgentSpec:
    """Load an :class:`AgentSpec` from a YAML file.

    A string ``output_schema`` is treated as a path to a JSON Schema file, resolved
    relative to the YAML file's directory, and inlined into the returned spec.

    ``AgentSpec`` forbids unknown fields so a typo'd key fails loudly; known
    sibling-schema keys (see ``_SIBLING_SCHEMA_KEYS``) are dropped first so a
    dual-purpose file (e.g. ``workflow.yaml``) still loads. Malformed YAML or a
    schema error is re-raised as a :class:`HimmyError` naming the file, not a raw
    parser/pydantic traceback.
    """
    spec_path = Path(path).expanduser()
    if not spec_path.is_file():
        raise HimmyError(
            f"agent spec not found: {spec_path} (create one with `himmy init`)"
        )
    raw = parse_spec_yaml(spec_path, kind="agent spec")
    raw = {k: v for k, v in raw.items() if k not in _SIBLING_SCHEMA_KEYS}

    schema = raw.get("output_schema")
    if isinstance(schema, str):
        schema_path = (spec_path.parent / schema).expanduser()
        raw["output_schema"] = json.loads(schema_path.read_text())

    return validate_spec(AgentSpec, raw, spec_path, kind="agent spec")
