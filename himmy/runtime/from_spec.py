"""Build a runtime from a declarative :class:`AgentSpec`, independent of the CLI.

This is the single source of truth for turning an ``agent.yaml`` (loaded as an
``AgentSpec``) into a wired :class:`SingleAgentRuntime` plus its tool registry —
honoring provider/model overrides, project defaults, guardrails, memory, tool
packs, knowledge, HTTP tools, custom tool modules, sub-agent spawning, and skill
dispatch. Both the ``himmy`` CLI and Himmy Studio's API call into it, so the two
front ends wire agents identically. The CLI keeps thin wrappers over these for
backwards compatibility.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from himmy.config.agent_spec import AgentSpec, load_agent_spec

# Text file extensions auto-ingested from a `knowledge:` directory.
KNOWLEDGE_EXTS = {".md", ".markdown", ".txt", ".rst", ".csv"}

_PROJECT_CACHE: dict[str, Any] | None = None


def load_project(refresh: bool = False) -> dict[str, Any]:
    """Load (and cache) the ``himmy.toml`` project config for this process."""
    global _PROJECT_CACHE
    if _PROJECT_CACHE is None or refresh:
        from himmy.config.project import load_project_config

        _PROJECT_CACHE = load_project_config()
    return _PROJECT_CACHE


def apply_project_defaults(spec: AgentSpec) -> AgentSpec:
    """Fill unset spec fields from ``himmy.toml`` ``[defaults]`` (spec/flag still win)."""
    defaults = load_project().get("defaults", {})
    if spec.provider is None and defaults.get("provider"):
        spec.provider = defaults["provider"]
    if spec.model == "default" and defaults.get("model"):
        spec.model = defaults["model"]
    if not spec.tool_packs and defaults.get("tool_packs"):
        spec.tool_packs = list(defaults["tool_packs"])
    if not spec.guardrails and defaults.get("guardrails"):
        spec.guardrails = list(defaults["guardrails"])
    return spec


def resolve_tools_module(dotted: str) -> Callable[[Any], Any]:
    """Resolve a ``module:attr`` (or ``module`` → ``register``) tool registrar."""
    module_name, _, attr = dotted.partition(":")
    if not attr:
        attr = "register"
    # Allow running against a tools.py sitting next to the spec / in CWD.
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"tools_module {dotted!r} has no callable {attr!r}")
    return fn


async def ingest_knowledge_sources(registry: Any, sources: list[str]) -> int:
    """Ingest each declared file/dir of text docs into the KB; returns the count."""
    ingest = registry.handler_for("kb_ingest")
    count = 0
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(
                f for f in p.rglob("*") if f.suffix.lower() in KNOWLEDGE_EXTS
            )
        else:
            continue
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            await ingest({"text": text, "title": f.stem, "source_uri": f.name})
            count += 1
    return count


def load_spec_file(
    path: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    expand_skills: bool = True,
) -> AgentSpec:
    """Load an ``agent.yaml`` into an :class:`AgentSpec`, ready to wire.

    Applies project defaults, optional provider/model overrides, and expands any
    declared ``skills`` into tools + injected know-how (so skill-contributed packs
    and guardrails flow through the normal wiring path).
    """
    spec = load_agent_spec(path)
    spec = apply_project_defaults(spec)
    if provider:
        spec.provider = provider
    if model and model != "default":
        spec.model = model
    if spec.skills and expand_skills:
        from himmy.config.agent_spec import apply_skills
        from himmy.skills import build_skill_registry

        spec = apply_skills(spec, build_skill_registry())
    return spec


def build_runtime_for_spec(
    spec: AgentSpec,
    *,
    provider: str | None = None,
    model: str | None = None,
    on_event: Any = None,
    inference: Any = None,
    on_log: Callable[[str], None] | None = None,
) -> Any:
    """Wire a runtime for ``spec`` honoring provider/model overrides + tools.

    Returns ``(runtime, registry)``. The registry is non-``None`` whenever the agent
    has tools/packs/MCP servers; MCP tools are registered into it later, inside the
    event loop, since connecting is async. ``inference`` overrides the provider-derived
    service (used by record/replay). ``on_log`` receives human-readable progress notes
    (e.g. a knowledge-ingest count); ``None`` is silent.

    Knowledge ingestion runs an inner ``asyncio.run``; call this from a worker thread
    (``asyncio.to_thread``) when invoking from inside an event loop.
    """
    from himmy import build_runtime
    from himmy.cli.provider import build_inference_for
    from himmy.services.tools.registry import ToolRegistry

    provider = provider or spec.provider
    model = model or (spec.model if spec.model != "default" else None)
    if inference is None:
        inference = build_inference_for(provider, model)

    overrides: dict[str, Any] = {"inference": inference}
    if on_event is not None:
        overrides["on_event"] = on_event

    pipeline = None
    if spec.guardrails:
        from himmy.services.guardrails import build_guardrail_pipeline

        pipeline = build_guardrail_pipeline(spec.guardrails)
        overrides["input_guardrail"] = pipeline
        overrides["output_guardrail"] = pipeline

    if spec.memory:
        from himmy import build_storage
        from himmy.services.context.service import ContextService
        from himmy.services.memory import (
            InMemoryMemoryStore,
            MemoryContextAdapter,
            MemoryService,
            SqliteMemoryStore,
        )
        from himmy.toolkit import ToolkitConfig

        tk = ToolkitConfig.from_sources(load_project().get("toolkit"))
        store = (
            SqliteMemoryStore(tk.memory_path)
            if tk.memory_path
            else InMemoryMemoryStore()
        )
        memory = MemoryService(store, embedder=tk.build_embedder_and_dim()[0])
        adapter = MemoryContextAdapter(
            memory, top_k=spec.memory_top_k, subject_id=tk.memory_subject
        )
        overrides["context_service"] = ContextService(
            storage_service=build_storage(), adapters=[adapter]
        )

    registry = None
    if (
        spec.tool_packs
        or spec.tools_module
        or spec.http_tools
        or spec.knowledge
        or spec.mcp_servers
        or spec.allow_spawn
        or spec.allow_skill_dispatch
    ):
        registry = ToolRegistry()
        if spec.tool_packs:
            from himmy.toolkit import ToolkitConfig, register_packs

            tk_config = ToolkitConfig.from_sources(load_project().get("toolkit"))
            register_packs(registry, spec.tool_packs, tk_config)
        if spec.knowledge:
            # Auto-ingest the declared docs into a local knowledge base and give the
            # agent kb_search — a grounded doc agent with no driver code.
            from himmy.toolkit import ToolkitConfig, register_packs

            if "knowledge" not in spec.tool_packs:
                register_packs(
                    registry,
                    ["knowledge"],
                    ToolkitConfig.from_sources(load_project().get("toolkit")),
                )
            n = asyncio.run(ingest_knowledge_sources(registry, spec.knowledge))
            if on_log is not None:
                on_log(f"ingested {n} document(s) into the knowledge base")
        if spec.http_tools:
            from himmy.config.http_tool_spec import register_http_tools

            register_http_tools(registry, spec.http_tools)
        if spec.tools_module:
            resolve_tools_module(spec.tools_module)(registry)
        if spec.allow_spawn:
            # Recursive sub-agents share the parent's inference backend; the spawned
            # worker's runtime has no spawn_agent tool, capping recursion at one level.
            from himmy.toolkit.spawn import register_spawn_tool

            register_spawn_tool(registry, inference=inference)
        if spec.allow_skill_dispatch:
            # dispatch_skill runs a named capability as a tool-scoped sub-agent; the
            # sub-runtime lacks the tool, so a dispatched skill can't dispatch again.
            from himmy.skills import build_skill_registry, register_skill_dispatch_tool

            register_skill_dispatch_tool(
                registry, inference=inference, skill_registry=build_skill_registry()
            )
        if pipeline is not None:
            # Guard tool arguments too (the highest-risk "act" surface).
            from himmy.services.guardrails import build_guardrail_pre_hook
            from himmy.services.tools.service import ToolService

            overrides["tool_service"] = ToolService(
                registry, pre_execution_hook=build_guardrail_pre_hook(pipeline)
            )
        else:
            overrides["tool_registry"] = registry

    # A skill that names explicit tools must find them in the wired registry — fail
    # loudly here rather than letting the agent silently lack a capability's tools.
    skill_tools = spec.metadata.get("resolved_skill_tools") or []
    if skill_tools:
        from himmy.skills import SkillToolError

        available = {d.name for d in registry.list()} if registry is not None else set()
        missing = [t for t in skill_tools if t not in available]
        if missing:
            raise SkillToolError(
                f"skill(s) {spec.metadata.get('skills')} require tool(s) not "
                f"available: {', '.join(missing)} — add the providing tool_pack "
                f"or tools_module"
            )

    runtime, _inference, _tools = build_runtime(**overrides)
    return runtime, registry


__all__ = [
    "KNOWLEDGE_EXTS",
    "load_project",
    "apply_project_defaults",
    "resolve_tools_module",
    "ingest_knowledge_sources",
    "load_spec_file",
    "build_runtime_for_spec",
]
