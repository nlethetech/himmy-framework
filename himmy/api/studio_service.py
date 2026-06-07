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
    if (
        isinstance(raw, dict)
        and raw.get("entry")
        and isinstance(raw.get("members"), list)
    ):
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


# ---- Live cognition: map runtime events into rich GUI frames ------------

# Synthetic orchestration tools, never shown as ordinary tool steps (they're the
# delegation/handoff/final-answer machinery, surfaced via their own frames).
_SYNTHETIC_TOOLS = ("ask_", "transfer_to_")


def _cap(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# Tools whose result grounds the answer in retrieved evidence (recall = memory,
# *kb_search* / *knowledge* = RAG). Matched by name; the result is parsed for
# citations so the GUI can show what the agent actually relied on.
def _grounding_from_tool(name: str, result: Any) -> dict[str, Any] | None:
    """Parse a grounding tool's result into ``{source, query, citations}`` or None."""
    low = name.lower()
    is_memory = low == "recall"
    is_kb = ("kb_search" in low) or ("knowledge" in low) or ("search_kb" in low)
    if not (is_memory or is_kb):
        return None
    import json

    if not isinstance(result, str):
        return None
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    citations: list[dict[str, Any]] = []
    # memory recall → {query, results: [{text, similarity}]}
    for r in data.get("results") or []:
        if isinstance(r, dict):
            citations.append(
                {
                    "text": _cap(str(r.get("text", "")), 400),
                    "similarity": r.get("similarity"),
                    "source_uri": None,
                }
            )
    # kb search → {chunks: [{text|context_window, similarity, source_uri}]}
    for c in data.get("chunks") or []:
        if isinstance(c, dict):
            snippet = c.get("text") or c.get("context_window") or ""
            citations.append(
                {
                    "text": _cap(str(snippet), 400),
                    "similarity": c.get("similarity"),
                    "source_uri": c.get("source_uri"),
                }
            )
    if not citations:
        return None
    return {
        "source": "memory" if is_memory else "knowledge",
        "query": data.get("query"),
        "citations": citations,
    }


class _Cognition:
    """Turn the runtime's event stream into rich, live cognition frames.

    Tracks which agent is active, pairs each tool call with its result + intent +
    latency, and records a persistable trace (think → act → observe). Used both for
    the live SSE stream and for the run-detail cognition view.
    """

    def __init__(self, read_only: dict[str, bool | None], initial_agent: str) -> None:
        from himmy.core.events import EventType

        self._E = EventType
        self._read_only = read_only
        self.active = initial_agent
        self.active_model: str | None = None
        self.tools_used: list[str] = []
        self.delegate_answers: list[tuple[str, str]] = []
        self.steps: list[dict[str, Any]] = []
        # Usage rollup (tokens/cost/latency) accumulated across inferences.
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.inferences = 0
        self._latencies: list[float] = []
        self._by_model: dict[str, dict[str, Any]] = {}

    def _record(self, **kw: Any) -> None:
        kw["seq"] = len(self.steps) + 1
        self.steps.append(kw)

    def _account(self, model: str, in_tok: int, out_tok: int, cost: float) -> None:
        """Fold one inference into the run-level + per-model usage rollup."""
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cost += cost
        self.inferences += 1
        m = self._by_model.setdefault(
            model,
            {
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "inferences": 0,
            },
        )
        m["input_tokens"] += in_tok
        m["output_tokens"] += out_tok
        m["cost"] += cost
        m["inferences"] += 1

    @property
    def usage_by_model(self) -> list[dict[str, Any]]:
        return list(self._by_model.values())

    def _intent(self, name: str) -> bool | None:
        ro = self._read_only.get(name)
        if ro is None:
            from himmy.services.tools.access import classify_read_only

            ro = classify_read_only(name)
        return ro

    def frames(self, event: Any) -> list[dict[str, Any]]:
        """Map one runtime event to zero or more GUI frames (and record the step)."""
        E = self._E
        et = getattr(event, "event_type", None)
        payload = getattr(event, "payload", None) or {}
        ts = getattr(event, "timestamp", None)
        out: list[dict[str, Any]] = []

        if et == E.AGENT_RUN_STARTED:
            name = payload.get("persona_name")
            model = payload.get("model_key")
            if model:
                self.active_model = model
            if name and name != self.active:
                self.active = name
                out.append({"type": "agent", "name": name, "model": model})
                self._record(kind="agent", agent=name, model=model, ts=ts)
            return out

        if et in (E.INFERENCE_SUCCEEDED, E.INFERENCE_FAILED):
            io = payload.get("io") or {}
            model = io.get("model") or self.active_model or "?"
            in_tok = int(payload.get("input_tokens") or 0)
            out_tok = int(payload.get("output_tokens") or 0)
            cost = float(getattr(event, "cost", None) or 0.0)
            latency = getattr(event, "latency_ms", None)
            if latency is not None:
                self._latencies.append(float(latency))
            # Account every inference (a failed call can still cost), then emit a
            # live usage frame carrying both the delta and the running totals.
            self._account(model, in_tok, out_tok, cost)
            out.append(
                {
                    "type": "usage",
                    "agent": self.active,
                    "model": model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost": cost,
                    "latency_ms": latency,
                    "total_input_tokens": self.input_tokens,
                    "total_output_tokens": self.output_tokens,
                    "total_cost": self.cost,
                    "inferences": self.inferences,
                }
            )
            # Reasoning that PRECEDES an action (the final answer has no tool calls
            # and is delivered as the message, not as a reasoning step).
            text = (io.get("response_text") or "").strip()
            if text and (io.get("tool_calls") or []):
                out.append({"type": "reason", "agent": self.active, "text": text})
                self._record(kind="reason", agent=self.active, text=text, ts=ts)
            return out

        if et in (E.TOOL_COMPLETED, E.TOOL_FAILED):
            name = payload.get("tool_name") or ""
            if name.startswith(_SYNTHETIC_TOOLS) or name == "final_answer":
                return out
            args = dict(payload.get("tool_args") or {})
            result = payload.get("result")
            result_str = _cap(str(result), 800) if result is not None else ""
            outcome = payload.get("tool_outcome") or (
                "success" if et == E.TOOL_COMPLETED else "failed"
            )
            ro = self._intent(name)
            latency = payload.get("latency_ms")
            if name not in self.tools_used:
                self.tools_used.append(name)
            frame = {
                "type": "tool",
                "agent": self.active,
                "name": name,
                "args": args,
                "result": result_str,
                "outcome": outcome,
                "read_only": ro,
                "latency_ms": latency,
            }
            out.append(frame)
            self._record(
                kind="tool",
                agent=self.active,
                name=name,
                args=args,
                result=result_str,
                outcome=outcome,
                read_only=ro,
                latency_ms=latency,
                ts=ts,
            )
            # Memory recall / in-run KB search also grounds the answer — surface it
            # as a citation alongside the plain tool step ("why it said that").
            if et == E.TOOL_COMPLETED:
                grounding = _grounding_from_tool(name, payload.get("result"))
                if grounding is not None:
                    out.append({"type": "grounding", "agent": self.active, **grounding})
                    self._record(
                        kind="grounding", agent=self.active, ts=ts, **grounding
                    )
            return out

        if et == E.CONTEXT_SNAPSHOT_BUILT:
            for g in payload.get("grounding") or []:
                citations = g.get("citations") or []
                frame = {
                    "type": "grounding",
                    "agent": self.active,
                    "source": "knowledge",
                    "query": g.get("query") or g.get("key"),
                    "citations": citations,
                }
                out.append(frame)
                self._record(
                    kind="grounding",
                    agent=self.active,
                    ts=ts,
                    source="knowledge",
                    query=g.get("query") or g.get("key"),
                    citations=citations,
                )
            return out

        if et == E.APPROVAL_REQUIRED:
            out.append(
                {
                    "type": "approval_required",
                    "agent": self.active,
                    "checkpoint_id": payload.get("checkpoint_id"),
                    "tools": list(payload.get("tools") or []),
                }
            )
            return out

        if et == E.GUARDRAIL_APPLIED:
            safety = {
                "stage": payload.get("stage"),
                "blocked": bool(payload.get("blocked")),
                "redacted": bool(payload.get("redacted")),
                "flags": list(payload.get("flags") or []),
                "reasons": list(payload.get("reasons") or []),
            }
            out.append({"type": "safety", "agent": self.active, **safety})
            self._record(kind="safety", agent=self.active, ts=ts, **safety)
            return out

        if et == E.AGENT_DELEGATED:
            worker = payload.get("worker", "?")
            answer = str(payload.get("answer", "") or "")
            if answer:
                self.delegate_answers.append((worker, answer))
            task = _cap(str(payload.get("task", "")), 200)
            out.append(
                {
                    "type": "delegate",
                    "from": self.active,
                    "worker": worker,
                    "task": task,
                }
            )
            self._record(
                kind="delegate", agent=self.active, worker=worker, task=task, ts=ts
            )
            return out

        if et == E.AGENT_HANDOFF:
            frm = payload.get("from")
            to = payload.get("to", "?")
            out.append({"type": "handoff", "from": frm, "to": to})
            self._record(kind="handoff", agent=frm, to=to, ts=ts)
            return out

        return out


async def _drain_cognition(
    queue: asyncio.Queue, run_task: asyncio.Task, cog: _Cognition
) -> AsyncIterator[dict[str, Any]]:
    """Yield cognition frames live as runtime events arrive, until the run finishes."""
    while True:
        getter = asyncio.create_task(queue.get())
        done, _pending = await asyncio.wait(
            {getter, run_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if getter in done:
            for frame in cog.frames(getter.result()):
                yield frame
        else:
            getter.cancel()
        if run_task.done():
            while not queue.empty():
                for frame in cog.frames(queue.get_nowait()):
                    yield frame
            break


def _usage_of(cog: _Cognition) -> dict[str, Any]:
    """The run-level usage rollup to persist (tokens, cost, per-model)."""
    return {
        "input_tokens": cog.input_tokens,
        "output_tokens": cog.output_tokens,
        "cost": cog.cost,
        "by_model": cog.usage_by_model,
    }


def _read_only_map(registry: Any) -> dict[str, bool | None]:
    """Tool name → read-only intent, from the wired registry (empty if none)."""
    if registry is None:
        return {}
    out: dict[str, bool | None] = {}
    for d in registry.list():
        out[d.name] = getattr(d, "read_only", None)
    return out


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
    queue: asyncio.Queue = asyncio.Queue()

    async def _on_event(event: Any) -> None:
        collected_events.append(event)  # for the raw-I/O inspector timeline
        await queue.put(event)  # for the live cognition stream

    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    # Build the runtime off the event loop: knowledge ingestion runs its own
    # asyncio.run() internally, which must not happen inside this running loop.
    from himmy.api.studio_approvals import get_checkpoint_store

    runtime, registry = await asyncio.to_thread(
        lambda: from_spec.build_runtime_for_spec(
            spec,
            provider=provider,
            model=model,
            on_event=_on_event,
            capture_io=True,  # power the cognition stream + trace inspector
            checkpoint_store=get_checkpoint_store(),  # pause on approval-gated tools
        )
    )
    has_tools = registry is not None
    thread = _rebuild_thread(spec, history)
    cog = _Cognition(_read_only_map(registry), spec.name)

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
    checkpoint_id: str | None = None

    yield {"type": "start", "agent": spec.name, "streaming": not has_tools}
    try:
        task = spec.make_task(prompt)
        if has_tools:
            # Tool-using agents stream their cognition (act → observe) live, and
            # PAUSE (hitl) when a tool needs approval instead of auto-denying it.
            run_task = asyncio.create_task(
                runtime.run_agent_loop(
                    persona,
                    task,
                    thread,
                    llm_config=llm_config,
                    max_turns=8,
                    route_tools=spec.tool_router,
                    hitl=True,
                )
            )
            async for frame in _drain_cognition(queue, run_task, cog):
                yield frame
            loop = await run_task
            tools = cog.tools_used
            if loop.stopped_reason == "awaiting_approval":
                # The run paused at an approval gate. The approval_required frame
                # already streamed (via _Cognition); end here without a final
                # message/done — the user resumes it later from Approvals.
                status = "awaiting_approval"
                checkpoint_id = loop.checkpoint_id
            else:
                output_text = loop.final.output_text or ""
                if not output_text.strip():
                    # A tool-using turn can end without a closing synthesis (the
                    # model emitted tool calls but no final prose, or a tool came
                    # back empty). Never show a blank card — say what happened.
                    used = ", ".join(tools) if tools else ""
                    output_text = (
                        "I ran "
                        + (f"`{used}` " if used else "some tools ")
                        + "but didn't get enough back to answer. This often means a "
                        "web search returned nothing (try rephrasing, or connect a "
                        "search key under Connections → Web search) or a needed "
                        "account isn't connected yet."
                    )
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
            # Streaming can't interleave cognition mid-token, so fold the events it
            # emitted (usage, guardrails, grounding) in afterwards — both to surface
            # them in the GUI and to record cog.steps for persistence. Reasoning is
            # skipped: its text already streamed as the answer.
            while not queue.empty():
                for frame in cog.frames(queue.get_nowait()):
                    if frame["type"] != "reason":
                        yield frame
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
            steps=cog.steps,
            usage=_usage_of(cog),
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
            thread_id=thread.thread_id,
            checkpoint_id=checkpoint_id,
        )
    except Exception:  # noqa: BLE001 - persistence must never break the stream
        pass

    if status == "error":
        yield {"type": "error", "message": error_msg or "run failed", "run_id": run_id}
        return
    if status == "awaiting_approval":
        # Terminal for this stream; the approval_required frame already carried the
        # checkpoint. The user resumes from the Approvals inbox.
        yield {
            "type": "paused",
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
        }
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
    steps: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    status: str,
    error: str | None,
    duration_ms: float,
    thread_id: str | None,
    checkpoint_id: str | None = None,
) -> None:
    """Build a :class:`StudioRun` and persist it (runs in a worker thread)."""
    from himmy.api.studio_runs import (
        ModelUsage,
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
    usage = usage or {}
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
        steps=list(steps or []),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost=float(usage.get("cost", 0.0)),
        usage_by_model=[ModelUsage(**u) for u in usage.get("by_model", [])],
        checkpoint_id=checkpoint_id,
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


# Verbose / redundant payload keys never shown in a timeline detail line (the full
# prompt etc. live in the raw-I/O inspector instead).
_NOISY_PAYLOAD_KEYS = frozenset(
    {
        "rendered_prompt",
        "snapshot_id",
        "route_override",
        "request_id",
        "trace_id",
        "model_key",
        "response_format",
        "io",
    }
)


def _summarize_payload(payload: dict[str, Any]) -> str:
    """A compact, readable one-line summary of a run event's payload.

    Inference events collapse to ``<model>  <in>→<out> tok``; everything else prefers a
    meaningful key (tool/worker/persona/status). The verbose prompt/ids are dropped —
    they're available in the raw-I/O inspector — so the timeline reads cleanly.
    """
    if not payload:
        return ""
    if "input_tokens" in payload or "output_tokens" in payload:
        model = payload.get("model_path") or payload.get("model_key") or ""
        toks = f"{payload.get('input_tokens', 0)}→{payload.get('output_tokens', 0)} tok"
        return f"{model}  {toks}".strip()
    for key in (
        "tool",
        "tool_name",
        "worker",
        "persona_name",
        "to",
        "status",
        "summary",
        "text",
        "reason",
        "name",
    ):
        if payload.get(key):
            return _trim(str(payload[key]), 150)
    import json

    clean = {
        k: v
        for k, v in payload.items()
        if k not in _NOISY_PAYLOAD_KEYS and v not in (None, "", [], {})
    }
    return _trim(json.dumps(clean, default=str), 150) if clean else ""


# ---- Running a TEAM (manager → workers), streamed live ------------------


def load_team(rel_path: str, root: Path | None = None) -> Any:
    """Resolve + load a team spec selected in the GUI."""
    from himmy.config.team_spec import load_team_spec

    path = resolve_spec_path(rel_path, root)
    return load_team_spec(str(path))


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
    from himmy.orchestrators import MultiAgentOrchestrator

    queue: asyncio.Queue = asyncio.Queue()
    collected: list[Any] = []

    async def _push(event: Any) -> None:
        collected.append(event)
        await queue.put(event)

    # Build off the event loop (tool-module import + pack registration are sync).
    def _build() -> tuple[Any, Any, Any]:
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
    entry = getattr(spec, "entry", None) or team_name
    cog = _Cognition(_read_only_map(registry), entry)

    yield {"type": "start", "agent": team_name, "streaming": False, "team": True}

    run_task = asyncio.create_task(orch.run(prompt))
    try:
        async for frame in _drain_cognition(queue, run_task, cog):
            yield frame
        result = await run_task
        output_text = (result.output_text or "").strip()
        if not output_text and cog.delegate_answers:
            # The manager ended without a closing synthesis — fall back to the
            # specialists' findings so the user always sees a real answer (skip any
            # specialist that itself came back empty).
            findings = [
                f"• **{worker}** — {ans.strip()}"
                for worker, ans in cog.delegate_answers
                if ans.strip()
            ]
            if findings:
                output_text = "Here's what the team found:\n\n" + "\n".join(findings)
        if not output_text.strip():
            # Nothing usable came back — never show a blank card.
            output_text = (
                "The team ran but didn't get enough back to answer. This often "
                "means a web search returned nothing (try rephrasing, or connect a "
                "search key under Connections → Web search) or a needed account "
                "isn't connected yet."
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
            tools=cog.tools_used,
            events=collected,
            steps=cog.steps,
            usage=_usage_of(cog),
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
        )
    except Exception:  # noqa: BLE001 - persistence must never break the stream
        pass

    if status == "error":
        yield {
            "type": "error",
            "message": error_msg or "team run failed",
            "run_id": run_id,
        }
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
    steps: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    status: str,
    error: str | None,
    duration_ms: float,
) -> None:
    """Persist a team run (transcript = prompt + final answer; timeline = the trail)."""
    from himmy.api.studio_runs import (
        ModelUsage,
        StudioRun,
        TranscriptMessage,
        get_run_store,
    )

    messages = [
        TranscriptMessage(role="user", content=prompt),
        TranscriptMessage(role="assistant", content=output or (error or "")),
    ]
    usage = usage or {}
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
        steps=list(steps or []),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost=float(usage.get("cost", 0.0)),
        usage_by_model=[ModelUsage(**u) for u in usage.get("by_model", [])],
    )
    get_run_store().save(run)


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
