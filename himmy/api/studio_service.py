"""Himmy Studio backend service: agent discovery + running agents for the GUI.

Kept separate from the FastAPI router so the GUI's behavior is unit-testable without
HTTP. Everything wires through :mod:`himmy.runtime.from_spec`, so a Studio run is
configured exactly like a ``himmy run``. Agents are discovered relative to a project
root (the directory ``himmy studio`` was launched in).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from himmy.config.agent_spec import AgentSpec, load_agent_spec
from himmy.runtime import from_spec


def project_root() -> Path:
    """The directory Studio resolves agent files against (the launch CWD)."""
    return Path.cwd()


# Directories never scanned for agent specs.
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".himmy"}


class AgentSummary(BaseModel):
    """A discovered ``agent.yaml`` — enough to list and pick it in the GUI."""

    name: str
    path: str  # relative to the project root (stable id for the GUI)
    description: str = ""
    provider: str | None = None
    model: str = "default"
    skills: list[str] = []
    tool_packs: list[str] = []
    has_tools: bool = False
    error: str | None = None  # set when the file failed to parse


def _candidate_files(root: Path) -> list[Path]:
    """Likely agent-spec files under ``root`` (loose files + agents/ + one level deep)."""
    found: set[Path] = set()
    for pat in ("agent.yaml", "*.agent.yaml", "*/agent.yaml", "agents/*.yaml"):
        found.update(root.glob(pat))
    # Loose top-level *.yaml too (so a hand-named spec is discoverable).
    found.update(root.glob("*.yaml"))
    return sorted(
        f
        for f in found
        if f.is_file() and not any(part in _SKIP_DIRS for part in f.parts)
    )


def _summarize(path: Path, root: Path) -> AgentSummary | None:
    """Load a spec file into an :class:`AgentSummary`; None if it isn't an agent spec."""
    rel = str(path.relative_to(root))
    try:
        spec = load_agent_spec(str(path))
    except Exception as exc:  # noqa: BLE001 - a non-spec YAML is simply skipped
        # A YAML that doesn't look like an agent (no name/description) isn't an error
        # worth surfacing; only flag files that look intended-as-agents but broke.
        if path.name == "agent.yaml" or path.name.endswith(".agent.yaml"):
            return AgentSummary(name=path.stem, path=rel, error=str(exc))
        return None
    return AgentSummary(
        name=spec.name,
        path=rel,
        description=spec.description,
        provider=spec.provider,
        model=spec.model,
        skills=list(spec.skills),
        tool_packs=list(spec.tool_packs),
        has_tools=bool(
            spec.skills
            or spec.tool_packs
            or spec.tools
            or spec.tools_module
            or spec.http_tools
            or spec.mcp_servers
        ),
    )


def list_agents(root: Path | None = None) -> list[AgentSummary]:
    """Discover agent specs under the project root, de-duplicated by name+path."""
    root = root or project_root()
    out: list[AgentSummary] = []
    seen: set[str] = set()
    for f in _candidate_files(root):
        summary = _summarize(f, root)
        if summary is None or summary.path in seen:
            continue
        seen.add(summary.path)
        out.append(summary)
    return out


def resolve_spec_path(rel_path: str, root: Path | None = None) -> Path:
    """Resolve a GUI-supplied relative agent path safely under the project root."""
    root = (root or project_root()).resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path escapes project root: {rel_path!r}")
    if not target.is_file():
        raise FileNotFoundError(f"agent spec not found: {rel_path!r}")
    return target


def load_studio_spec(
    rel_path: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    root: Path | None = None,
) -> AgentSpec:
    """Load + fully prepare an agent spec selected in the GUI (defaults, skills)."""
    path = resolve_spec_path(rel_path, root)
    return from_spec.load_spec_file(str(path), provider=provider, model=model)


# ---- Running an agent (streaming) ---------------------------------------


def _rebuild_thread(spec: AgentSpec, history: list[dict[str, Any]]) -> Any:
    """Reconstruct a ChatThread from prior GUI turns (user/assistant only).

    The runtime injects the system prompt itself on the first turn, so history holds
    only user/assistant messages — replayed verbatim to preserve multi-turn context.
    """
    from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole

    persona = spec.to_persona()
    thread = ChatThread(agent_id=persona.agent_id)
    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
    for turn in history:
        role = role_map.get(str(turn.get("role")))
        content = str(turn.get("content") or "")
        if role is None or not content:
            continue
        thread.append_message(Message(role=role, content=content))
    return thread


async def stream_agent_run(
    spec: AgentSpec,
    prompt: str,
    *,
    history: list[dict[str, Any]] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run an agent for one user turn, yielding GUI events.

    Event shapes (each delivered as one SSE ``data:`` frame):
      * ``{"type": "start", "agent": name, "streaming": bool}``
      * ``{"type": "token", "delta": str}``         — incremental text (no-tool agents)
      * ``{"type": "tool", "name": str}``           — a tool the agent invoked
      * ``{"type": "message", "text": str}``        — full assistant text (tool agents)
      * ``{"type": "done", "output_text", "thread_id", "succeeded"}``
      * ``{"type": "error", "message": str}``

    No-tool agents stream token-by-token; tool-using agents run the bounded
    act→observe loop (which can't stream mid-loop) and return the final answer plus
    the tools they touched.
    """
    history = history or []
    collected_events: list[Any] = []

    async def _on_event(event: Any) -> None:
        collected_events.append(event)

    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    # Build the runtime off the event loop: knowledge ingestion runs its own
    # asyncio.run() internally, which must not happen inside this running loop.
    runtime, registry = await asyncio.to_thread(
        from_spec.build_runtime_for_spec,
        spec,
        provider=provider,
        model=model,
        on_event=_on_event,
    )
    has_tools = registry is not None
    thread = _rebuild_thread(spec, history)

    # Attach any MCP servers for the lifetime of this run.
    mcp_clients: list[Any] = []
    if spec.mcp_servers:
        from himmy.config.mcp_spec import attach_mcp_servers

        mcp_clients = await attach_mcp_servers(registry, list(spec.mcp_servers))

    yield {"type": "start", "agent": spec.name, "streaming": not has_tools}
    try:
        if has_tools:
            task = spec.make_task(prompt)
            loop = await runtime.run_agent_loop(
                persona,
                task,
                thread,
                llm_config=llm_config,
                max_turns=8,
                route_tools=spec.tool_router,
            )
            for name in _tool_names(collected_events):
                yield {"type": "tool", "name": name}
            text = loop.final.output_text or ""
            yield {"type": "message", "text": text}
            yield {
                "type": "done",
                "output_text": text,
                "thread_id": thread.thread_id,
                "succeeded": bool(loop.final.succeeded)
                if hasattr(loop.final, "succeeded")
                else True,
            }
        else:
            task = spec.make_task(prompt)
            final_text = ""
            async for delta in runtime.stream_task(
                persona, task, thread, llm_config=llm_config
            ):
                if delta.delta:
                    final_text += delta.delta
                    yield {"type": "token", "delta": delta.delta}
                if delta.done and delta.response is not None:
                    final_text = delta.response.output_text or final_text
            yield {
                "type": "done",
                "output_text": final_text,
                "thread_id": thread.thread_id,
                "succeeded": True,
            }
    finally:
        if mcp_clients:
            from himmy.config.mcp_spec import close_mcp_clients

            await close_mcp_clients(mcp_clients)


def _tool_names(events: list[Any]) -> list[str]:
    """Extract the ordered, de-duplicated tool names from collected run events."""
    from himmy.core.events import EventType

    names: list[str] = []
    for ev in events:
        if getattr(ev, "event_type", None) != EventType.TOOL_CALLED:
            continue
        payload = getattr(ev, "payload", None) or {}
        name = payload.get("tool") or payload.get("name") or payload.get("tool_name")
        if name and name not in names:
            names.append(name)
    return names


__all__ = [
    "AgentSummary",
    "list_agents",
    "load_studio_spec",
    "resolve_spec_path",
    "stream_agent_run",
    "project_root",
]
