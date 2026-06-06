"""Himmy Studio backend service: agent discovery + running agents for the GUI.

Kept separate from the FastAPI router so the GUI's behavior is unit-testable without
HTTP. Everything wires through :mod:`himmy.runtime.from_spec`, so a Studio run is
configured exactly like a ``himmy run``. Agents are discovered relative to a project
root (the directory ``himmy studio`` was launched in).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from himmy.config.agent_spec import AgentSpec, load_agent_spec
from himmy.core.ids import new_uuid, utc_now_iso
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


def _looks_like_team(path: Path) -> dict | None:
    """Return the parsed mapping if ``path`` is a team spec (entry + members), else None."""
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - non-YAML / unreadable → not a team
        return None
    if isinstance(raw, dict) and raw.get("entry") and isinstance(raw.get("members"), list):
        return raw
    return None


def list_agents(root: Path | None = None) -> list[AgentSummary]:
    """Discover single-agent specs under the project root (excludes team specs)."""
    root = root or project_root()
    out: list[AgentSummary] = []
    seen: set[str] = set()
    for f in _candidate_files(root):
        if _looks_like_team(f) is not None:
            continue  # teams are listed separately
        summary = _summarize(f, root)
        if summary is None or summary.path in seen:
            continue
        seen.add(summary.path)
        out.append(summary)
    return out


class TeamMemberInfo(BaseModel):
    name: str
    role: str | None = None
    provider: str | None = None
    model: str = "default"
    delegates: list[str] = []
    handoffs: list[str] = []


class TeamSummary(BaseModel):
    """A discovered team.yaml — enough to list and pick it in the GUI."""

    name: str
    path: str  # relative to the project root
    entry: str
    members: list[TeamMemberInfo] = []
    is_team: bool = True


def list_teams(root: Path | None = None) -> list[TeamSummary]:
    """Discover multi-agent team specs under the project root."""
    root = root or project_root()
    out: list[TeamSummary] = []
    seen: set[str] = set()
    for f in _candidate_files(root):
        raw = _looks_like_team(f)
        if raw is None:
            continue
        rel = str(f.relative_to(root))
        if rel in seen:
            continue
        seen.add(rel)
        members = [
            TeamMemberInfo(
                name=str(m.get("name", "?")),
                role=m.get("role"),
                provider=m.get("provider"),
                model=str(m.get("model", "default")),
                delegates=list(m.get("delegates", []) or []),
                handoffs=list(m.get("handoffs", []) or []),
            )
            for m in raw.get("members", [])
            if isinstance(m, dict)
        ]
        out.append(
            TeamSummary(
                name=f.stem.replace("-team", "").replace(".team", "") or f.stem,
                path=rel,
                entry=str(raw.get("entry", "")),
                members=members,
            )
        )
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
    agent_path: str | None = None,
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
        lambda: from_spec.build_runtime_for_spec(
            spec,
            provider=provider,
            model=model,
            on_event=_on_event,
            capture_io=True,  # power the trace inspector
        )
    )
    has_tools = registry is not None
    thread = _rebuild_thread(spec, history)

    # Attach any MCP servers for the lifetime of this run.
    mcp_clients: list[Any] = []
    if spec.mcp_servers:
        from himmy.config.mcp_spec import attach_mcp_servers

        mcp_clients = await attach_mcp_servers(registry, list(spec.mcp_servers))

    run_id = new_uuid()
    started = time.monotonic()
    output_text = ""
    tools: list[str] = []
    status = "ok"
    error_msg: str | None = None

    yield {"type": "start", "agent": spec.name, "streaming": not has_tools}
    try:
        task = spec.make_task(prompt)
        if has_tools:
            loop = await runtime.run_agent_loop(
                persona,
                task,
                thread,
                llm_config=llm_config,
                max_turns=8,
                route_tools=spec.tool_router,
            )
            tools = _tool_names(collected_events)
            for name in tools:
                yield {"type": "tool", "name": name}
            output_text = loop.final.output_text or ""
            yield {"type": "message", "text": output_text}
        else:
            async for delta in runtime.stream_task(
                persona, task, thread, llm_config=llm_config
            ):
                if delta.delta:
                    output_text += delta.delta
                    yield {"type": "token", "delta": delta.delta}
                if delta.done and delta.response is not None:
                    output_text = delta.response.output_text or output_text
    except Exception as exc:  # noqa: BLE001 - record the failed run, then surface it
        status = "error"
        error_msg = str(exc)
    finally:
        if mcp_clients:
            from himmy.config.mcp_spec import close_mcp_clients

            await close_mcp_clients(mcp_clients)

    duration_ms = (time.monotonic() - started) * 1000.0
    # Persist the run (best-effort) so it's browsable under Runs.
    try:
        await asyncio.to_thread(
            _record_run,
            run_id=run_id,
            spec=spec,
            agent_path=agent_path,
            provider=provider,
            model=model,
            prompt=prompt,
            history=history,
            output=output_text,
            tools=tools,
            events=collected_events,
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
            thread_id=thread.thread_id,
        )
    except Exception:  # noqa: BLE001 - persistence must never break the stream
        pass

    if status == "error":
        yield {"type": "error", "message": error_msg or "run failed", "run_id": run_id}
        return
    yield {
        "type": "done",
        "output_text": output_text,
        "thread_id": thread.thread_id,
        "run_id": run_id,
        "succeeded": True,
    }


def _record_run(
    *,
    run_id: str,
    spec: AgentSpec,
    agent_path: str | None,
    provider: str | None,
    model: str | None,
    prompt: str,
    history: list[dict[str, Any]],
    output: str,
    tools: list[str],
    events: list[Any],
    status: str,
    error: str | None,
    duration_ms: float,
    thread_id: str,
) -> None:
    """Build a :class:`StudioRun` and persist it (runs in a worker thread)."""
    from himmy.api.studio_runs import (
        StudioRun,
        TranscriptMessage,
        get_run_store,
    )

    messages = [
        TranscriptMessage(role=str(t.get("role")), content=str(t.get("content") or ""))
        for t in history
    ]
    messages.append(TranscriptMessage(role="user", content=prompt))
    messages.append(
        TranscriptMessage(role="assistant", content=output or (error or ""))
    )
    run = StudioRun(
        id=run_id,
        created_at=utc_now_iso(),
        agent_name=spec.name,
        agent_path=agent_path,
        provider=provider or spec.provider,
        model=(model or (spec.model if spec.model != "default" else None)),
        prompt=prompt,
        output=output,
        output_preview=output,
        status=status,
        duration_ms=duration_ms,
        thread_id=thread_id,
        tools=tools,
        messages=messages,
        timeline=_build_timeline(prompt, events, output, status, error),
    )
    get_run_store().save(run)


def _build_timeline(
    prompt: str, events: list[Any], output: str, status: str, error: str | None
) -> list[Any]:
    """Compose a step-by-step timeline from synthetic bookends + runtime events."""
    from himmy.api.studio_runs import TimelineStep

    steps: list[TimelineStep] = []

    def add(
        type_: str,
        label: str,
        detail: str = "",
        ts: str | None = None,
        io: dict | None = None,
    ) -> None:
        steps.append(
            TimelineStep(
                seq=len(steps) + 1, type=type_, label=label, detail=detail, ts=ts, io=io
            )
        )

    add("run_started", "Run started", _trim(prompt, 200))
    for ev in events:
        et = getattr(ev, "event_type", None)
        name = et.value if et is not None else "event"
        payload = getattr(ev, "payload", None) or {}
        ts = getattr(ev, "timestamp", None)
        add(
            name.lower(),
            name.replace("_", " ").title(),
            _summarize_payload(payload),
            ts,
            io=payload.get("io"),
        )
    if status == "ok":
        add("run_completed", "Completed", _trim(output, 200))
    else:
        add("run_failed", "Failed", error or "")
    return steps


def _trim(text: str, n: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _summarize_payload(payload: dict[str, Any]) -> str:
    """A compact one-line summary of a run event's payload."""
    if not payload:
        return ""
    for key in ("tool", "name", "tool_name", "summary", "text", "reason"):
        if payload.get(key):
            return _trim(str(payload[key]), 160)
    import json

    return _trim(json.dumps(payload, default=str), 160)


# ---- Running a TEAM (manager → workers), streamed live ------------------

# Synthetic tool-name prefixes the orchestrator registers (hidden from the trail).
_SYNTHETIC_TOOL_PREFIXES = ("ask_", "transfer_to_")


def load_team(rel_path: str, root: Path | None = None) -> Any:
    """Resolve + load a team spec selected in the GUI."""
    from himmy.config.team_spec import load_team_spec

    path = resolve_spec_path(rel_path, root)
    return load_team_spec(str(path))


def _tool_name_of(event: Any) -> str | None:
    from himmy.core.events import EventType

    if getattr(event, "event_type", None) != EventType.TOOL_CALLED:
        return None
    payload = getattr(event, "payload", None) or {}
    return payload.get("tool") or payload.get("name") or payload.get("tool_name")


async def stream_team_run(
    spec: Any,
    prompt: str,
    *,
    team_name: str = "team",
    team_path: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run a multi-agent team for one request, streaming the live routing trail.

    Members run on their own providers (e.g. a Claude-CLI manager delegating to local
    Ollama workers). Emits ``start`` → ``tool``/``delegate``/``handoff`` frames as they
    happen → ``message`` (the final synthesized answer) → ``done``. The whole run is
    persisted for the Runs browser.
    """
    from himmy import build_runtime
    from himmy.config.team_spec import build_team, build_team_inference
    from himmy.core.events import EventType
    from himmy.orchestrators import MultiAgentOrchestrator

    queue: asyncio.Queue = asyncio.Queue()
    collected: list[Any] = []

    async def _push(event: Any) -> None:
        collected.append(event)
        await queue.put(event)

    # Build off the event loop (tool-module import + pack registration are sync).
    def _build() -> tuple[Any, Any]:
        team, registry = build_team(
            spec, resolve_tools_module=from_spec.resolve_tools_module
        )
        inference = build_team_inference(spec)
        runtime, _i, _t = build_runtime(
            inference=inference,
            tool_registry=registry,
            on_event=_push,
            capture_io=True,  # power the trace inspector
        )
        return runtime, registry, team

    runtime, registry, team = await asyncio.to_thread(_build)
    orch = MultiAgentOrchestrator(runtime, team, registry, on_event=_push)

    mcp_clients: list[Any] = []
    mcp_servers = list(getattr(spec, "mcp_servers", []) or [])
    if mcp_servers:
        from himmy.config.mcp_spec import attach_mcp_servers

        mcp_clients = await attach_mcp_servers(registry, mcp_servers)

    run_id = new_uuid()
    started = time.monotonic()
    output_text = ""
    status = "ok"
    error_msg: str | None = None
    tools_used: list[str] = []
    delegate_answers: list[tuple[str, str]] = []  # (worker, answer) for a fallback

    yield {"type": "start", "agent": team_name, "streaming": False, "team": True}

    def _frame(event: Any) -> dict[str, Any] | None:
        et = getattr(event, "event_type", None)
        payload = getattr(event, "payload", None) or {}
        if et == EventType.AGENT_DELEGATED:
            worker = payload.get("worker", "?")
            answer = str(payload.get("answer", "") or "")
            if answer:
                delegate_answers.append((worker, answer))
            return {
                "type": "delegate",
                "worker": worker,
                "task": _trim(str(payload.get("task", "")), 140),
            }
        if et == EventType.AGENT_HANDOFF:
            return {"type": "handoff", "to": payload.get("to", "?")}
        name = _tool_name_of(event)
        if name and not name.startswith(_SYNTHETIC_TOOL_PREFIXES) and name != "final_answer":
            if name not in tools_used:
                tools_used.append(name)
            return {"type": "tool", "name": name}
        return None

    run_task = asyncio.create_task(orch.run(prompt))
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {getter, run_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                frame = _frame(getter.result())
                if frame is not None:
                    yield frame
            else:
                getter.cancel()
            if run_task.done():
                # Drain whatever events are still queued, then stop.
                while not queue.empty():
                    frame = _frame(queue.get_nowait())
                    if frame is not None:
                        yield frame
                break
        result = await run_task
        output_text = (result.output_text or "").strip()
        if not output_text and delegate_answers:
            # The manager ended without a closing synthesis — fall back to the
            # specialists' findings so the user always sees a real answer.
            output_text = "Here's what the team found:\n\n" + "\n".join(
                f"• **{worker}** — {ans.strip()}" for worker, ans in delegate_answers
            )
        yield {"type": "message", "text": output_text}
    except Exception as exc:  # noqa: BLE001 - record + surface a terminal error
        status = "error"
        error_msg = str(exc)
    finally:
        if mcp_clients:
            from himmy.config.mcp_spec import close_mcp_clients

            await close_mcp_clients(mcp_clients)

    duration_ms = (time.monotonic() - started) * 1000.0
    try:
        await asyncio.to_thread(
            _record_team_run,
            run_id=run_id,
            team_name=team_name,
            team_path=team_path,
            prompt=prompt,
            output=output_text,
            tools=tools_used,
            events=collected,
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 - persistence must never break the stream
        pass

    if status == "error":
        yield {"type": "error", "message": error_msg or "team run failed", "run_id": run_id}
        return
    yield {
        "type": "done",
        "output_text": output_text,
        "run_id": run_id,
        "succeeded": True,
    }


def _record_team_run(
    *,
    run_id: str,
    team_name: str,
    team_path: str | None,
    prompt: str,
    output: str,
    tools: list[str],
    events: list[Any],
    status: str,
    error: str | None,
    duration_ms: float,
) -> None:
    """Persist a team run (transcript = prompt + final answer; timeline = the trail)."""
    from himmy.api.studio_runs import StudioRun, TranscriptMessage, get_run_store

    messages = [
        TranscriptMessage(role="user", content=prompt),
        TranscriptMessage(role="assistant", content=output or (error or "")),
    ]
    run = StudioRun(
        id=run_id,
        created_at=utc_now_iso(),
        agent_name=team_name,
        agent_path=team_path,
        provider="team",
        model="multi-provider",
        prompt=prompt,
        output=output,
        output_preview=output,
        status=status,
        duration_ms=duration_ms,
        thread_id=None,
        tools=tools,
        messages=messages,
        timeline=_build_timeline(prompt, events, output, status, error),
    )
    get_run_store().save(run)


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
    "TeamSummary",
    "TeamMemberInfo",
    "list_agents",
    "list_teams",
    "load_studio_spec",
    "load_team",
    "resolve_spec_path",
    "stream_agent_run",
    "stream_team_run",
    "project_root",
]
