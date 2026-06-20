"""Command handlers for the ``himmy`` CLI.

Each ``cmd_*`` is a synchronous function returning a process exit code; the ones that
drive the async runtime wrap it in :func:`asyncio.run` internally so the argparse
dispatcher in :mod:`himmy.cli.__main__` stays plain. Everything defaults to the
offline stub, so ``himmy run``/``himmy chat`` work with no keys and no network.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from himmy.cli.provider import build_inference_for
from himmy.config.agent_spec import AgentSpec, load_agent_spec
from himmy.runtime import from_spec


def _eprint(*args: Any) -> None:
    """Print to stderr (diagnostics, never mixed into machine-readable stdout)."""
    print(*args, file=sys.stderr)


def _effective_provider_model(
    spec: AgentSpec, args: argparse.Namespace
) -> tuple[str | None, str | None]:
    """The provider/model a run will actually use: CLI flags override the spec."""
    provider = getattr(args, "provider", None) or spec.provider
    model = getattr(args, "model", None) or (
        spec.model if spec.model != "default" else None
    )
    return provider, model


def _maybe_hint_stub(spec: AgentSpec, args: argparse.Namespace) -> None:
    """Tell a human at the terminal when a run is falling back to the offline stub.

    The stub returns canned, deterministic text — fine for wiring/tests, useless as a
    real answer. Without this, a first-time user sees nonsense and assumes himmy is
    broken. Only fires on an interactive terminal so piped/CI output and the test
    harness stay clean, and never when the user explicitly asked for the stub.
    """
    if not sys.stderr.isatty() or os.environ.get("HIMMY_NO_HINTS"):
        return
    provider, model = _effective_provider_model(spec, args)
    if provider == "stub":  # explicit choice — don't nag
        return
    from himmy.cli.provider import resolves_to_stub

    if not resolves_to_stub(provider, model):
        return
    _eprint(
        "note: running offline on the stub — canned deterministic output, not a real "
        "model.\n"
        "  for real answers, pick a backend:\n"
        "    • local, free:  ollama pull llama3.2   then add  --provider ollama\n"
        "    • Claude Max:   --provider claude-cli\n"
        "    • cloud:        set OPENAI_API_KEY or ANTHROPIC_API_KEY\n"
        "  details: himmy doctor\n"
    )


def _trace_db() -> str:
    """Path to the durable trace event log (``.himmy/trace.db``), dir created."""
    path = Path(".himmy")
    path.mkdir(exist_ok=True)
    return str(path / "trace.db")


class _TraceCollector:
    """Collects run events in-process and persists them to the trace log."""

    def __init__(self) -> None:
        from himmy.services.observability.trace import SqliteEventStore

        self.events: list[Any] = []
        self._store = SqliteEventStore(_trace_db())

    async def handle(self, event: Any) -> None:
        self.events.append(event)
        await self._store.append_event(event)


# The spec→runtime wiring lives in himmy.runtime.from_spec (shared with the Studio
# API). These thin aliases preserve the CLI's internal call sites.
_resolve_register = from_spec.resolve_tools_module
_project = from_spec.load_project
_apply_defaults = from_spec.apply_project_defaults


def _model_label(spec: AgentSpec, args: argparse.Namespace) -> str:
    """``sonnet · claude-cli``-style tag for the live spinner/✓ lines."""
    model = getattr(args, "model", None) or spec.model
    provider = getattr(args, "provider", None) or spec.provider
    parts = [p for p in (None if model == "default" else model, provider) if p]
    return " · ".join(parts)


def _read_piped_stdin() -> str:
    """Piped stdin content, or "" on a TTY / when stdin is unreadable (pytest, cron)."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:  # noqa: BLE001 - no stdin is never an error, just no input
        return ""


def _discover_spec_file() -> Path | None:
    """git-style upward search from cwd for the nearest ``agent.yaml``."""
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "agent.yaml"
        if candidate.is_file():
            return candidate
    return None


#: Cap on the HIMMY.md project-context block folded into agent instructions (~20 KB).
_HIMMY_MD_CAP = 20 * 1024


def _discover_himmy_md() -> Path | None:
    """git-style upward search from cwd for the nearest ``HIMMY.md`` project note."""
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / "HIMMY.md"
        if candidate.is_file():
            return candidate
    return None


def _apply_himmy_md(spec: AgentSpec) -> AgentSpec:
    """Fold a discovered ``HIMMY.md`` into ``spec``'s instructions as project context.

    Walks upward from cwd for ``HIMMY.md`` (same pattern as :func:`_discover_spec_file`);
    when found, appends its contents (capped at :data:`_HIMMY_MD_CAP`) to the agent's
    instructions, framed as project notes — for BOTH discovered-spec and house agents.
    A dim ``using <path>`` line is printed to stderr only on an interactive terminal
    (it's context, not decoration, so it still applies under pipes/CI — just silently).
    Returns ``spec`` unchanged when no ``HIMMY.md`` exists.
    """
    path = _discover_himmy_md()
    if path is None:
        return spec
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return spec
    if len(raw) > _HIMMY_MD_CAP:
        raw = raw[:_HIMMY_MD_CAP] + "\n… (truncated)"
    note = f"Project notes (HIMMY.md):\n{raw.strip()}"
    if sys.stderr.isatty():
        _eprint(f"using {path}")
    return spec.model_copy(update={"instructions": [*spec.instructions, note]})


def _house_spec() -> AgentSpec:
    """The capable default agent for a bare ask (`himmy "what's the weather"`).

    No spec file anywhere, no flags: instead of a toothless echo agent, build
    one on the best backend detected on this machine (same probe as the init
    wizard) with the keyless core packs, so a fresh install can just be asked
    things. With only the stub available it stays minimal — but still answers.
    """
    from himmy.cli.wizard import detect_provider_choices

    choice = detect_provider_choices()[0]
    real = choice.key != "stub"
    return AgentSpec(
        name="himmy",
        description="Your terminal assistant.",
        instructions=[
            "Be concise; answer in a few short lines.",
            "Use your tools rather than guessing.",
        ],
        provider=choice.key if real else None,
        model=choice.model or "default",
        tool_packs=["web", "utils", "data-sources", "news", "memory"]
        if real
        else ["utils"],
        memory=real,
        tool_router=True,
    )


def _spec_from_args(args: argparse.Namespace) -> AgentSpec:
    """Build the AgentSpec from ``-f file``, the nearest ``agent.yaml`` (searched
    upward from cwd, git-style), or ad-hoc ``--name``/``--instruction`` flags.
    With none of those, the capable house agent answers (see :func:`_house_spec`).

    A discovered ``HIMMY.md`` (upward from cwd) is folded into the agent's
    instructions as project context for every build path (see :func:`_apply_himmy_md`),
    and ``--provider``/``--model`` flags are folded INTO the spec so the per-request
    ``model_key`` (from ``spec.to_llm_config()``) matches the manager the flags
    built — otherwise an override sends the spec's old model string to the new
    provider every request (found live: OpenRouter 400 on a leaked ollama tag).
    """
    return _apply_cli_overrides(_apply_himmy_md(_spec_from_args_inner(args)), args)


def _apply_cli_overrides(spec: AgentSpec, args: argparse.Namespace) -> AgentSpec:
    """Fold ``--provider``/``--model`` into the spec itself (see _spec_from_args)."""
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if provider:
        spec.provider = provider
    if model and model != "default":
        spec.model = model
    return spec


def _spec_from_args_inner(args: argparse.Namespace) -> AgentSpec:
    """Build the bare AgentSpec (before HIMMY.md project context is folded in)."""
    if getattr(args, "file", None):
        return from_spec.load_spec_file(args.file)
    if not getattr(args, "name", None) and not getattr(args, "instruction", None):
        discovered = _discover_spec_file()
        if discovered is not None:
            _eprint(f"using {discovered}")
            return from_spec.load_spec_file(str(discovered))
        if not getattr(args, "provider", None):
            spec = _house_spec()
            spec = from_spec.apply_project_defaults(spec)
            return spec
    spec = AgentSpec(
        name=getattr(args, "name", None) or "himmy-agent",
        description="Ad-hoc agent created from CLI flags.",
        instructions=list(getattr(args, "instruction", None) or []),
    )
    spec = from_spec.apply_project_defaults(spec)
    if spec.skills:
        # Expand declared skills into tools + injected know-how before the runtime is
        # wired, so skill-contributed packs/guardrails flow through the normal path.
        from himmy.config.agent_spec import apply_skills
        from himmy.skills import build_skill_registry

        spec = apply_skills(spec, build_skill_registry())
    return spec


def _exec_with_mcp(
    factory: Any, registry: Any, mcp_servers: Any, *, runtime: Any = None
) -> Any:
    """Run ``factory()`` in one event loop, with MCP servers attached for its duration.

    MCP clients bind their reader task to the running loop, so connect + run + close
    must share a single ``asyncio.run``. With no MCP servers this is a plain run.

    ``runtime`` (when self-learning is on) is re-primed after the MCP tools are
    registered, so the reputation snapshot + hint candidates include them — the
    build-time priming ran before those async tools existed.
    """
    if not mcp_servers:
        return asyncio.run(factory())

    from himmy.config.mcp_spec import attach_mcp_servers, close_mcp_clients

    async def _outer() -> Any:
        clients = await attach_mcp_servers(registry, list(mcp_servers), runtime=runtime)
        try:
            return await factory()
        finally:
            await close_mcp_clients(clients)

    return asyncio.run(_outer())


async def _answer(
    runtime: Any,
    persona: Any,
    task: Any,
    *,
    thread: Any = None,
    llm_config: Any = None,
    has_tools: bool = False,
    max_turns: int = 8,
    route_tools: bool | None = None,
    cost_budget: float | None = None,
    on_stop: Any = None,
) -> Any:
    """Produce a final answer, driving the tool loop when the agent has tools.

    ``run_task_detailed`` is a single inference call — correct for a no-tool agent,
    but for a tool-using one the model's first turn is the tool *call*; the answer
    only comes after the runtime feeds the tool result back. So when tools are wired
    we use ``run_agent_loop`` (act → observe → answer) and return its final turn.
    ``cost_budget`` (when set) bounds the loop; ``on_stop`` is called with the loop's
    ``stopped_reason`` so the caller can report a budget-stopped run.
    """
    if has_tools:
        loop = await runtime.run_agent_loop(
            persona,
            task,
            thread,
            llm_config=llm_config,
            max_turns=max_turns,
            route_tools=route_tools,
            cost_budget=cost_budget,
        )
        if on_stop is not None:
            on_stop(loop.stopped_reason)
        return loop.final
    return await runtime.run_task_detailed(persona, task, thread, llm_config=llm_config)


async def _run_with_in_run_approvals(
    runtime: Any,
    checkpoint_store: Any,
    spec: AgentSpec,
    args: argparse.Namespace,
    live: Any,
    cost_budget: float | None = None,
    on_stop: Any = None,
) -> Any:
    """Run one prompt under hitl, servicing approval pauses, and return the RunResult.

    Mirrors the chat REPL's approval loop for one-shot ``himmy run``: run the agent
    loop with ``hitl=True``; while it pauses (``checkpoint_id`` set), stop the live
    spinner, prompt approve? [y/N/always], resume with the decision, and loop until
    the run finishes. Returns ``AgentLoopResult.final`` so the caller's existing
    RunResult-shaped printing/JSON path is unchanged.
    """
    from himmy.cli.repl import prompt_approval

    llm_config = spec.to_llm_config()
    loop_result = await runtime.run_agent_loop(
        spec.to_persona(),
        spec.make_task(args.prompt),
        llm_config=llm_config,
        max_turns=8,
        route_tools=spec.tool_router,
        hitl=True,
        cost_budget=cost_budget,
    )
    while getattr(loop_result, "checkpoint_id", None):
        if live is not None:
            live.finish()
        approved = prompt_approval(
            checkpoint_store,
            loop_result.checkpoint_id,
            role=getattr(args, "role", None),
        )
        loop_result = await runtime.resume_agent_loop(
            loop_result.checkpoint_id, approved=approved, llm_config=llm_config
        )
    if on_stop is not None:
        on_stop(loop_result.stopped_reason)
    return loop_result.final


# Knowledge ingestion lives in himmy.runtime.from_spec (shared with the Studio API).
_ingest_knowledge = from_spec.ingest_knowledge_sources


def _build_runtime_for(
    spec: AgentSpec,
    args: argparse.Namespace,
    *,
    on_event: Any = None,
    inference: Any = None,
    checkpoint_store: Any = None,
) -> Any:
    """Wire a runtime for ``spec`` honoring CLI provider/model overrides + tools.

    Thin CLI adapter over :func:`himmy.runtime.from_spec.build_runtime_for_spec`
    (the shared wiring used by Himmy Studio too). Returns ``(runtime, registry)``.
    ``checkpoint_store`` (optional) wires durable HITL pause/resume — passed when
    a run/chat may pause at an approval-gated tool.
    """
    return from_spec.build_runtime_for_spec(
        spec,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        on_event=on_event,
        inference=inference,
        on_log=_eprint,
        checkpoint_store=checkpoint_store,
    )


def _persist_run_result(
    result: Any,
    spec: AgentSpec,
    args: argparse.Namespace,
    *,
    prompt: str,
) -> str | None:
    """Persist a SYNCHRONOUSLY-completed RunResult to the canonical run store (T2b).

    ``--persist`` is OPT-IN: a plain one-shot ``himmy run``/``himmy chat`` never reaches
    here and stays in-RAM (the zero-config default). When set, we capture the run that
    ALREADY completed at the call site and write it via ``store.save_run`` — deliberately
    NOT through :meth:`RunAppService.create_run` (reviewer must_fix): ``create_run`` is
    fire-and-forget background execution, and a one-shot ``asyncio.run`` tears the event
    loop down the moment this returns, so a background task would be killed mid-flight. The
    run is finished; we only need to RECORD it, in the SAME ``.himmy/storage.db`` the
    server reads, so it lands in ``himmy runs list`` AND ``GET /v1/runs`` AND the Studio
    runs screen (the connectedness payoff). Stamped with the reserved ``__local__``
    workspace/subject so it is browsable locally but excluded from cross-tenant admin lists.

    Best-effort: returns the persisted ``run_id`` on success, or ``None`` (with a stderr
    note) on failure — persistence never changes the run's own exit code.
    """
    from himmy.cli.app_services import build_app_container
    from himmy.services.inference.models import InferenceStatus
    from himmy.services.storage.models import (
        LOCAL_SUBJECT,
        LOCAL_WORKSPACE,
        RunRecord,
        RunStatus,
    )

    succeeded = getattr(result, "status", None) == InferenceStatus.SUCCESS.value
    thread = getattr(result, "thread", None)
    thread_id = getattr(thread, "thread_id", None)
    persona = spec.to_persona()
    record = RunRecord(
        workspace_id=LOCAL_WORKSPACE,
        subject_id=LOCAL_SUBJECT,
        thread_id=thread_id,
        trace_id=getattr(result, "trace_id", None) or thread_id,
        persona_name=getattr(persona, "agent_id", None) or spec.name,
        model_key=getattr(result, "model_path", None) or None,
        status=RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED,
        output_text=getattr(result, "output_text", None) or None,
        output_structured=getattr(result, "output_structured", None),
        error=getattr(result, "error", None),
        metadata={
            "source": "cli",
            "prompt": prompt,
            "provider_name": getattr(result, "provider_name", "") or "",
            "cost": getattr(result, "cost", 0.0) or 0.0,
            "input_tokens": getattr(result, "input_tokens", 0) or 0,
            "output_tokens": getattr(result, "output_tokens", 0) or 0,
            "latency_ms": getattr(result, "latency_ms", 0.0) or 0.0,
        },
    )
    container = build_app_container()
    try:
        saved = asyncio.run(container.storage.save_run(record))
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        _eprint(f"⚠ --persist failed to save run: {exc}")
        return None
    finally:
        container.close()
    _eprint(f"persisted run {saved.run_id} → himmy runs show {saved.run_id}")
    return saved.run_id


# --------------------------------------------------------------------- run/chat


def _record_replay_inference(
    spec: AgentSpec, args: argparse.Namespace
) -> tuple[Any, Any]:
    """Build a record/replay inference service from --record/--replay; else (None, None).

    Returns ``(inference, recorder)``: ``inference`` overrides the provider service;
    ``recorder`` (a RecordingClientManager) is non-None under --record so the caller can
    dump its cassette after the run.
    """
    from himmy.cli.provider import build_manager_for
    from himmy.services.inference.service import InferenceService

    replay = getattr(args, "replay", None)
    record = getattr(args, "record", None)
    if replay:
        from himmy.services.inference.replay import ReplayClientManager

        return InferenceService(ReplayClientManager.from_file(replay)), None
    if record:
        from himmy.services.inference.replay import RecordingClientManager

        provider = getattr(args, "provider", None) or spec.provider
        model = getattr(args, "model", None) or (
            spec.model if spec.model != "default" else None
        )
        recorder = RecordingClientManager(
            build_manager_for(provider, model), label=spec.name
        )
        return InferenceService(recorder), recorder
    return None, None


def cmd_run(args: argparse.Namespace) -> int:
    """One-shot: run a single prompt through the agent and print the answer.

    The prompt can arrive three ways, composable: positional words, ``-p``, and
    piped stdin (appended as input, or used as the whole prompt when alone) —
    so ``git diff | himmy run "review this"`` does what it reads like it does.
    """
    prompt = (args.prompt or " ".join(getattr(args, "words", None) or [])).strip()
    piped = _read_piped_stdin().strip()
    if piped:
        prompt = f"{prompt}\n\n--- piped input ---\n{piped}" if prompt else piped
    if not prompt:
        _eprint("error: give a prompt — positional, -p/--prompt, or piped stdin")
        return 2
    # @file affordance: inline the contents of any @path token that names a real file.
    from himmy.cli.input_affordances import expand_at_files

    prompt = expand_at_files(prompt, on_warn=lambda msg: _eprint(msg))
    args.prompt = prompt
    if getattr(args, "record", None) and getattr(args, "replay", None):
        _eprint("error: --record and --replay are mutually exclusive")
        return 2
    if getattr(args, "yolo", False) and getattr(args, "safe", False):
        _eprint("error: --yolo and --safe are mutually exclusive")
        return 2
    spec = _spec_from_args(args)
    from himmy.cli import permissions as _perms

    flag_budget = getattr(args, "budget", None)
    cost_budget = (
        float(flag_budget) if flag_budget is not None else _perms.load_session_budget()
    )
    tracer = _TraceCollector() if getattr(args, "trace", False) else None
    inference, recorder = _record_replay_inference(spec, args)
    if inference is None:  # record/replay supplies its own (non-stub) manager
        _maybe_hint_stub(spec, args)
    # Live desk-style rendering (spinner + tool lines on stderr, TTY only).
    # Streaming runs own the terminal with their token flow, so no overlay there.
    from himmy.cli.ui import LiveRunUI, compose_event_handlers, styles

    live = (
        LiveRunUI(model_label=_model_label(spec, args))
        if not getattr(args, "stream", False)
        else None
    )
    # In-run approvals only when stdin AND stderr are real TTYs (a human can answer
    # the prompt); scripts/CI stay non-interactive so gated tools fail closed.
    interactive = bool(sys.stdin.isatty() and sys.stderr.isatty())
    checkpoint_store = None
    if interactive:
        from himmy.cli.repl import _cli_checkpoint_db
        from himmy.runtime.checkpoint import SqliteCheckpointStore

        checkpoint_store = SqliteCheckpointStore(_cli_checkpoint_db())
    runtime, registry = _build_runtime_for(
        spec,
        args,
        on_event=compose_event_handlers(
            tracer.handle if tracer else None,
            live.handle if live is not None and live.enabled else None,
        ),
        inference=inference,
        checkpoint_store=checkpoint_store,
    )
    # Permission profile: --yolo grants everything, --safe ignores the allowlist,
    # otherwise the himmy.toml [permissions] auto_approve list ungates its tools.
    from himmy.cli import permissions
    from himmy.cli.rbac_cmd import born_gated_names

    # Snapshot which tools are approval-gated AS BUILT, before the profile ungates
    # any — RBAC enforcement needs this to gate plain tools vs. born-gated ones.
    born_gated = born_gated_names(registry)
    if getattr(args, "yolo", False):
        c = styles(sys.stderr)
        _eprint(f"{c['crimson']}⚠ --yolo: every tool runs without approval{c['reset']}")
        permissions.grant_all_approvals(registry)
    elif getattr(args, "safe", False):
        pass  # ignore the allowlist: everything gated prompts
    else:
        permissions.apply_allowlist(registry, permissions.load_auto_approve())
    # RBAC: hard-deny a restricted role's gated tools (re-gate so they fail closed),
    # so a viewer in CI / non-TTY can't slip a gated tool past the allowlist or --yolo.
    from himmy.cli.rbac_cmd import enforce_role_on_registry

    enforce_role_on_registry(
        registry,
        role=getattr(args, "role", None),
        on_line=_eprint,
        born_gated=born_gated,
    )

    def _print_trace() -> None:
        if tracer is not None:
            from himmy.services.observability.trace import format_timeline

            _eprint("\n--- trace ---")
            _eprint(format_timeline(tracer.events))

    if getattr(args, "stream", False):

        async def _stream() -> None:
            async for delta in runtime.stream_task(
                spec.to_persona(),
                spec.make_task(args.prompt),
                llm_config=spec.to_llm_config(),
            ):
                if delta.delta:
                    sys.stdout.write(delta.delta)
                    sys.stdout.flush()
            sys.stdout.write("\n")

        _exec_with_mcp(_stream, registry, spec.mcp_servers, runtime=runtime)
        _print_trace()
        return 0

    if getattr(args, "plan", False):
        from himmy.orchestrators import PlannerOrchestrator

        async def _plan() -> Any:
            return await PlannerOrchestrator(runtime).run(
                args.prompt, spec.to_persona(), tool_names=spec.tools or None
            )

        result = _exec_with_mcp(_plan, registry, spec.mcp_servers, runtime=runtime)
        _eprint(f"plan: {len(result.plan)} step(s)")
        for i, step in enumerate(result.plan, start=1):
            _eprint(f"  {i}. {step}")
        print(result.output_text or "")
        _print_trace()
        return 0

    # Interactive + tool-using: run under hitl and service approval pauses in-run
    # (render the pending call, prompt approve? [y/N/always], resume, loop).
    stop_reason: list[str] = []
    if interactive and registry is not None and checkpoint_store is not None:

        async def _go() -> Any:
            return await _run_with_in_run_approvals(
                runtime,
                checkpoint_store,
                spec,
                args,
                live,
                cost_budget=cost_budget,
                on_stop=stop_reason.append,
            )
    else:

        async def _go() -> Any:
            return await _answer(
                runtime,
                spec.to_persona(),
                spec.make_task(args.prompt),
                llm_config=spec.to_llm_config(),
                has_tools=registry is not None,
                route_tools=spec.tool_router,
                cost_budget=cost_budget,
                on_stop=stop_reason.append,
            )

    result = _exec_with_mcp(_go, registry, spec.mcp_servers, runtime=runtime)
    if live is not None:
        live.finish()
    if cost_budget is not None and stop_reason and stop_reason[0] == "budget":
        _eprint(
            f"⚠ stopped: cost budget ${cost_budget:.2f} reached "
            f"(spent ${result.cost or 0.0:.4f}) — answer may be incomplete"
        )

    # --persist (opt-in, T2b): record the now-completed run in the canonical store so it
    # appears in `himmy runs` / `GET /v1/runs` / the Studio runs screen. Default-off keeps
    # the one-shot path in-RAM and byte-identical to before.
    if getattr(args, "persist", False):
        _persist_run_result(result, spec, args, prompt=args.prompt)

    if args.json:
        print(
            json.dumps(
                {
                    "status": result.status,
                    "output_text": result.output_text,
                    "output_structured": result.output_structured,
                    "cost": result.cost,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "model_path": result.model_path,
                    "provider_name": result.provider_name,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif spec.output_schema is not None and result.output_structured is not None:
        print(json.dumps(result.output_structured, indent=2, ensure_ascii=False))
    else:
        from himmy.cli.ui import render_markdown_lite

        print(render_markdown_lite(result.output_text or "", stream=sys.stdout))
    if live is not None:
        live.footer()
    _print_trace()
    if recorder is not None:
        path = recorder.dump(args.record)
        _eprint(f"recorded {len(recorder.cassette.entries)} model exchange(s) → {path}")
    elif getattr(args, "replay", None):
        _eprint(f"replayed from {args.replay}")
    return 0 if result.succeeded else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL keeping one thread; `--message` runs a single turn.

    The interactive path lives in :class:`himmy.cli.repl.ChatRepl` (which adds
    in-REPL approvals, interrupt-safe turns, and plan mode); this stays the thin
    entry that handles the non-interactive ``--message`` one-shot and otherwise
    delegates to the REPL.
    """
    spec = _spec_from_args(args)
    _maybe_hint_stub(spec, args)

    if getattr(args, "yolo", False) and getattr(args, "safe", False):
        _eprint("error: --yolo and --safe are mutually exclusive")
        return 2

    if getattr(args, "message", None):
        from himmy.cli.ui import LiveRunUI, render_markdown_lite

        live = LiveRunUI(model_label=_model_label(spec, args))
        runtime, registry = _build_runtime_for(
            spec, args, on_event=live.handle if live.enabled else None
        )
        from himmy.agents.base_agent.thread import ChatThread

        persona = spec.to_persona()
        llm_config = spec.to_llm_config()
        # Session: an explicit --session resumes that id; -c/--continue resumes the
        # implicit "last" session. With neither, this one-shot stays ephemeral.
        explicit = getattr(args, "session", None)
        continue_last = bool(getattr(args, "continue_last", False))
        session_id = str(explicit) if explicit else ("last" if continue_last else None)
        store = None
        if session_id:
            from himmy.config.project import conversations_db_path
            from himmy.runtime.session import SqliteSessionStore

            Path(".himmy").mkdir(exist_ok=True)
            store = SqliteSessionStore(conversations_db_path())
        thread = ChatThread(agent_id=persona.agent_id)
        if store is not None and session_id:
            existing = store.load(session_id)
            if existing is not None:
                thread = existing
        from himmy.cli.input_affordances import expand_at_files

        message = expand_at_files(args.message, on_warn=lambda m: _eprint(m))
        from himmy.cli import permissions as _perms

        _flag_budget = getattr(args, "budget", None)
        _cost_budget = (
            float(_flag_budget)
            if _flag_budget is not None
            else _perms.load_session_budget()
        )
        result = _exec_with_mcp(
            lambda: _answer(
                runtime,
                persona,
                spec.make_task(message),
                thread=thread,
                llm_config=llm_config,
                has_tools=registry is not None,
                route_tools=spec.tool_router,
                cost_budget=_cost_budget,
            ),
            registry,
            spec.mcp_servers,
            runtime=runtime,
        )
        if store is not None and session_id:
            store.save(session_id, result.thread)
        # --persist (opt-in, T2b): record this completed turn in the canonical run store.
        if getattr(args, "persist", False):
            _persist_run_result(result, spec, args, prompt=message)
        live.finish()
        print(render_markdown_lite(result.output_text or "", stream=sys.stdout))
        return 0 if result.succeeded else 1

    from himmy.cli.repl import ChatRepl

    return ChatRepl(spec, args).run()


# -------------------------------------------------------------------- telegram


def cmd_telegram(args: argparse.Namespace) -> int:
    """Run an agent as a live Telegram bot: each message → an agent reply."""
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.toolkit import ToolkitConfig
    from himmy.toolkit.telegram import TelegramBot, TelegramClient

    spec = _spec_from_args(args)
    runtime, registry = _build_runtime_for(spec, args)
    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    cfg = ToolkitConfig.from_sources(_project().get("toolkit"))
    token = getattr(args, "token", None) or cfg.telegram_bot_token
    if not token:
        _eprint("error: set HIMMY_TELEGRAM_BOT_TOKEN (or pass --token) to run the bot.")
        return 2

    # Cross-process single-flight (must share the lock the Studio listener uses): the CLI
    # and the Studio inbound listener must NEVER both long-poll one bot token, or Telegram
    # returns 409 / drops+duplicates updates across the two competing `getUpdates` loops.
    # Acquire the SAME per-token flock `studio_telegram.start()` takes (keyed by a digest of
    # the token, never the token itself) and hold it for the whole serve loop; refuse — do
    # NOT retry into a poll storm — if it is already held here or by the Studio listener.
    from himmy.api.studio_telegram import telegram_lock_name
    from himmy.core.process_lock import ProcessLockBusy, acquire_process_lock

    try:
        token_lock = acquire_process_lock(telegram_lock_name(token))
    except ProcessLockBusy:
        _eprint(
            "error: this bot token is already being polled (the Studio Telegram listener "
            "or another `himmy telegram`). Stop the other poller first."
        )
        return 2

    # One conversation thread per chat, so each user gets continuous context.
    threads: dict[str, Any] = {}

    has_tools = registry is not None

    async def _handle(chat_id: str, text: str) -> str:
        thread = threads.get(chat_id)
        if thread is None:
            thread = ChatThread(agent_id=persona.agent_id)
            threads[chat_id] = thread
        result = await _answer(
            runtime,
            persona,
            spec.make_task(text),
            thread=thread,
            llm_config=llm_config,
            has_tools=has_tools,
            route_tools=spec.tool_router,
        )
        threads[chat_id] = result.thread
        return result.output_text or ""

    # Sender allowlist: --allow-chat flags win over HIMMY_TELEGRAM_ALLOWED_CHATS; an
    # empty allowlist keeps the answer-everyone behaviour for a private single-chat bot.
    allowed_chat_ids = getattr(args, "allow_chat", None) or cfg.telegram_allowed_chat_ids
    bot = TelegramBot(
        TelegramClient(token, timeout=cfg.http_timeout + 30),
        _handle,
        allowed_chat_ids=allowed_chat_ids,
    )
    if allowed_chat_ids:
        _eprint(f"  (restricted to chat ids: {', '.join(allowed_chat_ids)})")
    else:
        _eprint(
            "  (no chat allowlist — set HIMMY_TELEGRAM_ALLOWED_CHATS or --allow-chat "
            "to restrict who can drive the bot)"
        )
    _eprint(f"himmy telegram — {persona.name} is live. Ctrl-C to stop.")

    async def _serve() -> None:
        try:
            await bot.run()
        except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover
            pass

    try:
        # MCP servers (if any) stay connected for the whole session.
        _exec_with_mcp(_serve, registry, spec.mcp_servers, runtime=runtime)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _eprint("\n(stopped)")
    finally:
        # Release the per-token flock so the Studio listener (or a fresh CLI) can poll
        # this token once we stop; the OS also drops it automatically on process exit.
        token_lock.release()
    return 0


# ------------------------------------------------------------------------ team


def cmd_team(args: argparse.Namespace) -> int:
    """Run a multi-agent team from a team.yaml: route a prompt and print the trail."""
    if not args.prompt:
        _eprint("error: --prompt/-p is required for `himmy team`")
        return 2
    from himmy import build_runtime
    from himmy.config.team_spec import build_team, build_team_inference, load_team_spec
    from himmy.orchestrators import MultiAgentOrchestrator

    spec = load_team_spec(args.file)
    team, registry = build_team(spec, resolve_tools_module=_resolve_register)
    # Members may each declare their own provider — a Claude-CLI brain orchestrating
    # local Ollama workers, etc.; otherwise one backend is used for the whole team.
    inference = build_team_inference(
        spec,
        default_provider=getattr(args, "provider", None),
        default_model=getattr(args, "model", None),
    )
    runtime, _inference, _tools = build_runtime(
        inference=inference, tool_registry=registry
    )
    orch = MultiAgentOrchestrator(runtime, team, registry)

    result = _exec_with_mcp(lambda: orch.run(args.prompt), registry, spec.mcp_servers)

    if args.json:
        print(
            json.dumps(
                {
                    "handoff_chain": result.handoff_chain,
                    "final_agent": result.final_agent,
                    "stopped_reason": result.stopped_reason,
                    "turn_count": result.turn_count,
                    "total_cost": result.total_cost,
                    "output_text": result.output_text,
                    "transcript": [
                        {"agent": name, "output": turn.output_text}
                        for name, turn in result.turns
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _eprint(f"route: {' → '.join(result.handoff_chain)}  ({result.stopped_reason})")
        print(result.output_text or "")
    return 0


# ------------------------------------------------------------------------ eval


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate an agent (or team) against a suite.yaml and print the scorecard."""
    from himmy import build_runtime
    from himmy.config.eval_spec import load_eval_suite
    from himmy.services.evaluation.agent_harness import AgentEvalHarness
    from himmy.services.evaluation.service import EvaluationService

    suite = load_eval_suite(args.file)
    inference = build_inference_for(
        getattr(args, "provider", None), getattr(args, "model", None)
    )
    # LLM-judge metrics activate only when a real (non-stub) provider is selected.
    judge = inference if getattr(args, "provider", None) not in (None, "stub") else None
    eval_service = EvaluationService(inference_service=judge)

    if args.team:
        from himmy.config.team_spec import build_team, load_team_spec

        team, registry = build_team(
            load_team_spec(args.team), resolve_tools_module=_resolve_register
        )
        runtime, _i, _t = build_runtime(inference=inference, tool_registry=registry)
        harness = AgentEvalHarness(runtime, eval_service)
        run = asyncio.run(harness.evaluate_team(suite, team, registry))
    else:
        if not args.agent:
            _eprint(
                "error: `himmy eval` needs --agent <agent.yaml> or --team <team.yaml>"
            )
            return 2
        spec = load_agent_spec(args.agent)
        runtime, registry = _build_runtime_for(spec, args)
        harness = AgentEvalHarness(runtime, eval_service)
        run = _exec_with_mcp(
            lambda: harness.evaluate_agent(
                suite, spec.to_persona(), llm_config=spec.to_llm_config()
            ),
            registry,
            spec.mcp_servers,
            runtime=runtime,
        )

    if args.json:
        print(json.dumps(run.model_dump(), indent=2, ensure_ascii=False, default=str))
        return 0
    print(f"suite: {run.suite_name}   aggregate: {run.aggregate_score:.3f}")
    for case in run.case_results:
        mark = "PASS" if case.passed else "FAIL"
        metrics = " ".join(f"{m.metric}={m.score:.2f}" for m in case.metric_scores)
        print(f"  [{mark}] {case.case_id[:8]}  {case.aggregate:.2f}  {metrics}")
    passed = sum(1 for c in run.case_results if c.passed)
    print(f"\n{passed}/{len(run.case_results)} cases passed")
    return 0 if passed == len(run.case_results) else 1


# ------------------------------------------------------------------------ init

_AGENT_YAML = """\
name: my-agent
description: A helpful assistant. Describe its job here.
role: Assistant
instructions:
  - Be concise and accurate.
  - Cite reasoning when it helps the user decide.
model: default
# provider: claude-cli      # stub | claude-cli | ollama | pydantic-ai | openrouter (default: auto)
# skills: [web_research]    # capability bundles: tools + know-how (run `himmy skills`)
# tool_packs: [web, utils]  # built-in tool packs (run `himmy tools` to list them)
tools: []                   # names to bind (from tool_packs and/or tools_module)
# tools_module: tools:register   # uncomment to wire the example tool in tools.py
# output_schema: schema.json     # path to a JSON Schema for structured output
"""

_SKILL_YAML = """\
# A skill bundles know-how with the tools it needs. Reference it from an agent with
# `skills: [my_skill]` (run `himmy skills` to list available skills, built-in + here).
name: my_skill
description: One line on what this capability does.
when_to_use: a short cue for when the agent should lean on this skill.
tool_packs: [utils]          # packs this skill grants (run `himmy tools`)
# tools: []                  # or specific tool names
instructions:
  - Concrete guidance the agent should follow when using this capability.
  - Prefer running a tool over guessing.
"""

_TOOLS_PY = '''\
"""Example tool module for a himmy agent.

Wire it by setting ``tools_module: tools:register`` and listing the tool under
``tools:`` in agent.yaml. ``register(registry)`` is called once at startup.
"""

from __future__ import annotations

from himmy.services.tools.registry import ToolRegistry, register_local_tool


def word_count(text: str) -> dict[str, int]:
    """Return the number of whitespace-separated words in ``text``."""
    return {"words": len(text.split())}


def register(registry: ToolRegistry) -> None:
    """Register this module's tools onto the runtime's tool registry."""
    register_local_tool(
        registry,
        name="word_count",
        handler=word_count,
        description="Count the words in a piece of text.",
        args_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
'''


_TEAM_YAML = """\
# A multi-agent team: a triage agent that hands off to specialists.
entry: triage
members:
  - name: triage
    description: Decide which specialist should handle the request, then hand off.
    handoffs: [researcher, writer]
  - name: researcher
    description: Gather facts from the web.
    tool_packs: [web]
    tools: [web_search, web_fetch]
    handoffs: [writer]            # may hand the findings to the writer
  - name: writer
    description: Write a clear final answer from the conversation so far.
"""


_HIMMY_TOML = """\
# Project defaults for the himmy CLI. Precedence: CLI flag > env > this file > built-in.
[defaults]
# provider = "ollama"          # stub | claude-cli | ollama | pydantic-ai | openrouter
# model = "qwen2.5:3b-instruct"
# tool_packs = ["web", "utils"]
# guardrails = ["pii"]

[toolkit]
# embedder = "ollama"          # deterministic | ollama | fastembed | openai
# memory_path = ".himmy/memory.db"
"""


# --- `himmy init --template <name>` starters -----------------------------------------
# Each is a working specialised agent showcasing one no-code feature.

_TPL_HELPDESK_AGENT = """\
# Answers questions from YOUR documents — grounded, offline, no code.
name: helpdesk
description: Answers questions from the documents in ./docs, grounded and cited.
provider: ollama
model: qwen2.5:3b-instruct
knowledge: [./docs]          # auto-ingested at startup; drop your own .md files in here
tools: [kb_search]
instructions:
  - You have NO knowledge of this company's policies — ONLY the documents do. You must
    call kb_search for EVERY question before answering. Never answer from memory and
    never ask the user for more context.
  - Answer in one or two sentences using the retrieved passage, and name the source.
  - Only if kb_search returns nothing relevant, say "That's not in the handbook."
"""

_TPL_HELPDESK_DOC = """\
# Company Handbook

## Time Off
Full-time employees get **20 days** of paid time off (PTO) per year.
Part-time employees accrue PTO prorated by their hours.

## Security
Report any security issue or phishing email to **security@example.com**.
All laptops must have full-disk encryption enabled.

Replace this file with your own documents — anything in ./docs is ingested.
"""

_TPL_ANALYST_AGENT = """\
# Answers questions by calling a live REST API — a declarative HTTP tool, no code.
name: data-analyst
description: Looks up live foreign-exchange rates via a public API.
provider: ollama
model: qwen2.5:3b-instruct
http_tools:
  - name: exchange_rate
    description: Get the exchange rate from a base currency to others (ISO codes).
    base_url: https://api.frankfurter.dev
    path: /v1/latest
    query: [base, symbols]
instructions:
  - Use exchange_rate (base and symbols are ISO currency codes) and state the rate.
"""

_TPL_RESEARCHER_AGENT = """\
# Researches a question on the web and answers with citations — uses a skill.
name: researcher
description: Searches the web and answers with sources.
provider: ollama
model: qwen2.5:3b-instruct
skills: [web_research]        # bundles the web tools + the know-how to use them
instructions:
  - Search before answering; ground the answer in what you find and cite the URLs.
"""

#: template name -> (files to write, the suggested next command, an offline note).
_TEMPLATES: dict[str, dict[str, Any]] = {
    "helpdesk": {
        "files": {
            "agent.yaml": _TPL_HELPDESK_AGENT,
            "docs/handbook.md": _TPL_HELPDESK_DOC,
        },
        "prompt": "How many PTO days do I get?",
        "note": "offline — needs the [knowledge] extra",
    },
    "analyst": {
        "files": {"agent.yaml": _TPL_ANALYST_AGENT},
        "prompt": "What is the USD to EUR rate? Use base=USD and symbols=EUR.",
        "note": "calls a live public API (needs network)",
    },
    "researcher": {
        "files": {"agent.yaml": _TPL_RESEARCHER_AGENT},
        "prompt": "What is permaculture? Cite a source.",
        "note": "searches the web (needs network + the [web] tools)",
    },
}


def cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark one or more models on a task suite; print a comparative scorecard."""
    from himmy.benchmark import (
        BenchmarkRunner,
        BenchmarkSuite,
        ModelSpec,
        bfcl_suite,
        default_suite,
        irrelevance_suite,
        multiagent_suite,
        nepali_suite,
        render_leaderboard,
        render_markdown,
        to_json,
    )

    # `--suite` accepts a built-in name (resolved offline) or a path to a suite.yaml.
    _BUILTIN_SUITES = {
        "core": default_suite,
        "nepali": nepali_suite,
        "irrelevance": irrelevance_suite,
        "multiagent": multiagent_suite,
        "bfcl": bfcl_suite,
    }
    suite_arg = getattr(args, "suite", None)
    if not suite_arg:
        suite = default_suite()
    elif suite_arg in _BUILTIN_SUITES:
        suite = _BUILTIN_SUITES[suite_arg]()
    else:
        suite = BenchmarkSuite.from_yaml(suite_arg)
    extra = [
        p.strip()
        for p in (getattr(args, "extra_packs", None) or "").split(",")
        if p.strip()
    ]
    specs: list[ModelSpec] = []
    for raw in (args.models or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        provider, _, model = raw.partition(":")  # model may itself contain ':'
        if not model:
            _eprint(f"error: model spec {raw!r} must be provider:model")
            return 2
        specs.append(
            ModelSpec(
                provider=provider,
                model=model,
                tool_router=bool(getattr(args, "router", False)),
                temperature=getattr(args, "temperature", 0.0),
                extra_packs=extra,
                # Judge tier (LLM-graded tasks): unset ⇒ judge-tier trials recorded
                # ungraded; the judge provider defaults to the candidate's own provider.
                judge_provider=getattr(args, "judge_provider", None) or "",
                judge_model=getattr(args, "judge_model", None) or "",
            )
        )
    if not specs:
        _eprint(
            "error: --models is required, e.g. "
            "--models ollama:qwen2.5:3b-instruct,claude-cli:haiku"
        )
        return 2

    def _progress(spec: Any, task: Any, i: int, n: int) -> None:
        _eprint(f"  [{spec.name}] {task.id}  trial {i}/{n}")

    runner = BenchmarkRunner(trials=args.trials, on_progress=_progress)
    _eprint(
        f"benchmarking {len(specs)} model(s) on '{suite.name}' "
        f"({len(suite.tasks)} tasks × {args.trials} trials)…"
    )
    cards = asyncio.run(runner.run(suite, specs))
    if getattr(args, "leaderboard", False):
        print(render_leaderboard(cards, suite_name=suite.name))
    else:
        print(render_markdown(cards, suite_name=suite.name))

    # Cache the scorecards so Studio's Doctor can show per-model reliability.
    from datetime import datetime

    from himmy.benchmark.cache import save_scorecards

    try:
        save_scorecards(
            cards,
            suite_name=suite.name,
            when=datetime.now(UTC).isoformat(),
        )
    except Exception:  # noqa: BLE001 - caching is best-effort, never fail the run
        pass

    if getattr(args, "json", None):
        Path(args.json).write_text(
            json.dumps(to_json(cards, suite_name=suite.name), indent=2),
            encoding="utf-8",
        )
        _eprint(f"\nwrote full results to {args.json}")

    # Regression gate (for CI): fail if any model scores below the floor. This is what
    # makes "did my change make agents worse?" answerable — a real regression trips it.
    floor = getattr(args, "fail_under", None)
    if floor is not None:
        below = [c for c in cards if c.accuracy < floor]
        for c in below:
            _eprint(
                f"FAIL: {c.spec.name} accuracy {c.accuracy:.0%} < floor {floor:.0%}"
            )
        if below:
            return 1
        _eprint(f"OK: all models ≥ {floor:.0%} floor")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold an agent: a default one, a ``--template`` starter, or a ``--team``.

    On a real terminal the default form goes interactive (see
    :mod:`himmy.cli.wizard`); pipes/CI and ``--classic`` keep the example scaffold.
    """
    target = Path(args.directory).expanduser()

    template = getattr(args, "template", None)
    if not template and not args.team and not getattr(args, "classic", False):
        from himmy.cli.wizard import run_wizard, wizard_available

        if wizard_available():
            return run_wizard(target, force=args.force)

    target.mkdir(parents=True, exist_ok=True)
    next_msg = ""
    if template:
        tmpl = _TEMPLATES[template]
        files = dict(tmpl["files"])
        next_msg = (
            f"\nNext ({tmpl['note']}):\n"
            f'  himmy run -f {target / "agent.yaml"} -p "{tmpl["prompt"]}"'
        )
    elif args.team:
        files = {"team.yaml": _TEAM_YAML}
        next_msg = f'\nNext: himmy team -f {target / "team.yaml"} -p "your question"'
    else:
        files = {
            "agent.yaml": _AGENT_YAML,
            "tools.py": _TOOLS_PY,
            "himmy.toml": _HIMMY_TOML,
            "skills/my_skill.yaml": _SKILL_YAML,
        }
        next_msg = f'\nNext: himmy run -f {target / "agent.yaml"} -p "hello"'

    existing = [name for name in files if (target / name).exists()]
    if existing and not args.force:
        _eprint(
            f"error: {', '.join(existing)} already exist in {target} "
            "(use --force to overwrite)"
        )
        return 1

    for name, content in files.items():
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        print(f"wrote {dest}")
    print(next_msg)
    return 0


# ------------------------------------------------------------------ demo-video


def cmd_demo_video(args: argparse.Namespace) -> int:
    """Scaffold a demo-video workspace, or render it to an MP4 with ``--render``."""
    from himmy.demovideo import render, scaffold

    target = Path(args.directory).expanduser()
    if args.render:
        render(target, only=args.only, output_name=args.output)
        return 0
    written = scaffold(target)
    for path in written:
        print(f"wrote {path}")
    if not written:
        print(f"{target} already scaffolded (files left untouched)")
    print(
        "\nNext: capture REAL command output, write it into "
        f"{target / 'script.json'} (see the README's two rules), then:\n"
        f"  himmy demo-video {target} --render"
    )
    return 0


# ----------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    """Boot the FastAPI BFF via uvicorn (requires the ``api`` extra)."""
    try:
        import uvicorn

        from himmy.api.app import create_app
    except Exception as exc:  # pragma: no cover - optional extra missing
        _eprint(
            "error: `himmy serve` needs the 'api' extra: "
            f"pip install 'himmy[api]'  ({exc})"
        )
        return 1

    # Pass the bind host so create_app can fail closed when an unauthenticated
    # build would be exposed off-loopback (see _enforce_auth_posture).
    app = create_app(bind_host=args.host)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------- studio


def cmd_studio(args: argparse.Namespace) -> int:
    """Serve Himmy Studio — the local web GUI — from the built frontend.

    Boots the same FastAPI BFF (so the Studio API + SPA share one origin) bound to
    loopback by default, since Studio can run agents and write files. If the GUI
    hasn't been built yet, prints exactly how to build it (or run the Vite dev
    server) instead of starting an empty shell.
    """
    try:
        import uvicorn

        from himmy.api.app import create_app, studio_is_built
    except Exception as exc:  # pragma: no cover - optional extra missing
        _eprint(
            "error: `himmy studio` needs the 'studio' extra: "
            f"pip install 'himmy[studio]'  ({exc})"
        )
        return 1

    if not studio_is_built():
        from himmy.api.app import STUDIO_STATIC_DIR

        _eprint(
            "Himmy Studio hasn't been built yet.\n"
            "\n"
            "  Build it once (needs Node 18+):\n"
            "    cd studio && npm install && npm run build\n"
            f"    → emits the SPA into {STUDIO_STATIC_DIR}\n"
            "  then re-run:  himmy studio\n"
            "\n"
            "  Or develop with hot-reload (two terminals):\n"
            "    himmy serve            # API on :8000\n"
            "    cd studio && npm run dev   # UI on :5173 (proxies /api → :8000)\n"
        )
        return 1

    url = f"http://{args.host}:{args.port}"
    _eprint(f"Himmy Studio → {url}  (Ctrl-C to stop)")
    if not getattr(args, "no_browser", False):
        # Open the browser shortly after the server comes up, in a daemon thread.
        import threading
        import webbrowser

        def _open() -> None:
            import time

            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception:  # pragma: no cover - headless / no browser
                pass

        threading.Thread(target=_open, daemon=True).start()

    # Pass the bind host so create_app can fail closed when an unauthenticated
    # build would be exposed off-loopback (see _enforce_auth_posture).
    uvicorn.run(create_app(bind_host=args.host), host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------- doctor


def _can_import(module: str) -> bool:
    """True when ``module`` imports cleanly (optional extra is installed)."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report installed optional extras, local providers, and provider keys.

    Renders the shared :func:`himmy.runtime.diagnostics.collect_doctor_report`
    snapshot (the same data Himmy Studio serves as JSON) as a text table.
    """
    from himmy.runtime.diagnostics import collect_doctor_report

    report = collect_doctor_report()
    print(f"himmy doctor — Python {report.python}")

    print("\noptional extras:")
    for extra in report.extras:
        print(f"  [{'ok ' if extra.ok else '-- '}] {extra.label}")

    print("\nlocal providers (PATH):")
    for prov in report.providers:
        print(
            f"  [{'ok ' if prov.ok else '-- '}] {prov.name}"
            + (f" → {prov.path}" if prov.path else "")
        )

    print("\nprovider keys (env):")
    for key in report.keys:
        print(f"  [{'ok ' if key.present else '-- '}] {key.name}")

    print(f"\nembedders (auto cascade, selected: {report.embedder_selected}):")
    for emb in report.embedder_statuses:
        flag = "ok " if emb.available else "-- "
        suffix = " (selected)" if emb.name == report.embedder_selected else ""
        if not suffix and emb.reason:
            suffix = f" ({emb.reason})"
        print(f"  [{flag}] {emb.name}{suffix}")

    print(
        f"\nguardrails (agent.yaml `guardrails: [...]`): {', '.join(report.guardrails)}"
    )
    print(
        f"\nproject config: {report.project_config or '(none — using env + defaults)'}"
    )

    if report.next_step is not None:
        print("\nnext step:")
        print(f"  → {report.next_step.message}")

    if getattr(args, "storage", False):
        _doctor_storage_section()
    return 0


def _redact_dsn(dsn: str) -> str:
    """Return ``dsn`` with any password component replaced by ``***``.

    Parses with :func:`urllib.parse.urlsplit` and rebuilds the netloc so a raw
    password is never echoed to the console. If the DSN cannot be parsed we fall
    back to a fully-masked placeholder rather than risk leaking it.
    """
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(dsn)
        if parts.password is None:
            return dsn
        userinfo = parts.username or ""
        if userinfo:
            userinfo = f"{userinfo}:***"
        host = parts.hostname or ""
        # ``urlsplit`` strips the brackets from an IPv6 host (``[::1]`` → ``::1``);
        # re-bracket it so the rebuilt netloc stays a valid URL (and the trailing
        # ``:port`` isn't ambiguous with the address's own colons).
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        netloc = f"{userinfo}@{host}" if userinfo else host
        return urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )
    except Exception:  # noqa: BLE001 - never risk echoing the raw DSN
        return "<redacted>"


def _doctor_postgres_report(dsn: str) -> None:
    """Print the Postgres backend report; degrade to one warning line on any error.

    Compares the ``schema_migrations`` versions applied in the live database against
    ``max(STORAGE_MIGRATIONS)`` in the shipped code, so an operator can see whether
    pending migrations exist (they are auto-applied at server startup). Any failure
    — asyncpg missing, connection refused, query error — collapses to a single
    ``warning:`` line; this command must never crash.
    """
    import asyncio

    from himmy.services.storage.postgres import STORAGE_MIGRATIONS

    print(f"  dsn: {_redact_dsn(dsn)}")
    code_max = max(version for version, _, _ in STORAGE_MIGRATIONS)

    async def _query() -> list[int]:
        import asyncpg  # noqa: PLC0415 - lazy: optional dependency

        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=5.0)
        try:
            rows = await conn.fetch("SELECT version FROM schema_migrations")
        finally:
            await conn.close()
        return sorted(int(row["version"]) for row in rows)

    try:
        applied = asyncio.run(_query())
    except Exception as exc:  # noqa: BLE001 - offline-first: degrade, never crash
        print(f"  warning: could not query schema_migrations ({exc})")
        return

    print(f"  applied: {applied}")
    print(f"  code max: {code_max}")
    pending = sorted(
        version for version, _, _ in STORAGE_MIGRATIONS if version not in applied
    )
    if pending:
        print(f"  PENDING: {pending} (auto-applied at server startup)")
    else:
        print("  up to date")


def _doctor_sqlite_report(base_dir: Path) -> None:
    """Print one line per ``*.db`` store under ``base_dir`` (size + journal mode)."""
    import sqlite3

    if not base_dir.is_dir():
        print(f"  (no {base_dir}/ stores yet)")
        return
    dbs = sorted(base_dir.glob("*.db"))
    if not dbs:
        print(f"  (no {base_dir}/ stores yet)")
        return
    for db in dbs:
        try:
            size = db.stat().st_size
        except OSError as exc:
            print(f"  {db.name}: (unreadable: {exc})")
            continue
        try:
            conn = sqlite3.connect(db)
            try:
                row = conn.execute("PRAGMA journal_mode").fetchone()
                mode = str(row[0]) if row else "?"
            finally:
                conn.close()
        except sqlite3.Error as exc:
            print(f"  {db.name}: {size} bytes (unreadable: {exc})")
            continue
        print(f"  {db.name}: {size} bytes, journal_mode={mode}")


def _doctor_storage_section() -> None:
    """Report the durable storage backend (Postgres vs SQLite) and its state.

    Mirrors :mod:`himmy.services.storage.factory` backend selection so the report
    matches what a server entrypoint would actually use: a ``postgres://`` DSN in
    ``HIMMY_DATABASE_URL`` selects Postgres (migration report), otherwise the
    file-backed SQLite stores under the configured store path's directory.
    """
    from himmy.config.secrets import get_secret
    from himmy.services.storage.factory import (  # noqa: PLC2701
        HIMMY_STORE_PATH,
        _is_postgres_dsn,
    )

    print("\nstorage:")
    dsn = get_secret("HIMMY_DATABASE_URL")
    if _is_postgres_dsn(dsn):
        print("  backend: postgresql")
        _doctor_postgres_report(dsn or "")
        return
    print("  backend: sqlite")
    store_path = get_secret("HIMMY_STORE_PATH") or HIMMY_STORE_PATH
    _doctor_sqlite_report(Path(store_path).parent)


# ----------------------------------------------------------------------- trace


def cmd_trace(args: argparse.Namespace) -> int:
    """Inspect saved run traces: list recent runs, or show one run's timeline."""
    from himmy.services.observability.trace import SqliteEventStore, format_timeline

    if not Path(_trace_db()).exists():
        _eprint("no traces yet — run with `himmy run --trace ...` first")
        return 1
    store = SqliteEventStore(_trace_db())
    if args.thread:
        print(format_timeline(store.list_events(thread_id=args.thread)))
        return 0
    runs = store.recent_threads(limit=args.limit)
    if not runs:
        _eprint("no traced runs found")
        return 1
    print("recent runs (use `himmy trace <thread_id>` for the timeline):\n")
    for run in runs:
        print(f"  {run['thread_id']}  {run['events']:>3} events  {run['last']}")
    return 0


# ----------------------------------------------------------------------- tools


def cmd_tools(args: argparse.Namespace) -> int:
    """List the built-in tool packs and the tools each one provides."""
    from himmy.toolkit import BUILTIN_PACKS

    print("built-in tool packs (use in agent.yaml: `tool_packs: [...]`):\n")
    for pack in BUILTIN_PACKS.values():
        print(f"  {pack.name:<7} {pack.description}")
        print(f"          tools: {', '.join(pack.tool_names)}")
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    """Manage the model price table: `sync` (refresh), `show <model>`, or list."""
    from himmy.services.inference import pricing

    action = getattr(args, "action", None)

    if action == "sync":
        url = getattr(args, "url", None) or pricing.LITELLM_PRICES_URL
        try:
            n = pricing.sync_prices(url=url)
        except Exception as exc:  # noqa: BLE001 - report network/parse failures cleanly
            _eprint(f"error: price sync failed: {exc}")
            return 1
        print(f"synced {n} model prices → {pricing.SYNCED_PRICES_PATH}")
        return 0

    if action == "show":
        model = getattr(args, "model", None)
        if not model:
            _eprint("error: `himmy prices show` needs a model, e.g. openai:gpt-4o-mini")
            return 2
        p = pricing.price_for(model)
        if not pricing.is_priced(model):
            print(
                f"{model}: unpriced ($0.00) — run `himmy prices sync` for current data"
            )
            return 0
        # Show in the conventional USD-per-1M-tokens form.
        print(f"{model}:")
        print(f"  input:  ${p.input_per_1k * 1000:.2f} / 1M tokens")
        print(f"  output: ${p.output_per_1k * 1000:.2f} / 1M tokens")
        return 0

    # Default: list the table.
    table = pricing.load_price_table()
    synced = pricing.SYNCED_PRICES_PATH.exists()
    print(
        f"{len(table)} model prices loaded "
        f"({'synced + ' if synced else ''}bundled snapshot).\n"
        f"  input/output are USD per 1M tokens.\n"
    )
    for name in sorted(table):
        p = table[name]
        print(
            f"  {name:<24} ${p.input_per_1k * 1000:>7.2f} / ${p.output_per_1k * 1000:.2f}"
        )
    if not synced:
        print("\nRefresh with the latest community prices: himmy prices sync")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    """List available skills (or show one in detail) for ``skills: [...]``."""
    from himmy.skills import BUILTIN_SKILLS, build_skill_registry, discover_skill_dirs

    registry = build_skill_registry()

    name = getattr(args, "name", None)
    if name:
        skill = registry.get(name)
        if skill is None:
            import difflib

            close = difflib.get_close_matches(name, registry.names(), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            _eprint(f"error: unknown skill {name!r}{hint}")
            return 1
        origin = "built-in" if skill.name in BUILTIN_SKILLS else "project"
        print(f"{skill.name}  (v{skill.version}, {origin})")
        print(f"  {skill.description}")
        if skill.when_to_use:
            print(f"\n  use when: {skill.when_to_use}")
        binds = ", ".join((*skill.tool_packs, *skill.tools)) or "(no tools)"
        print(f"  binds: {binds}")
        if skill.requires_skills:
            print(f"  requires: {', '.join(skill.requires_skills)}")
        if skill.guardrails:
            print(f"  guardrails: {', '.join(skill.guardrails)}")
        if skill.instructions:
            print("\n  instructions:")
            for line in skill.instructions:
                print(f"    - {line}")
        if skill.examples:
            print("\n  examples:")
            for ex in skill.examples:
                print(f'    - "{ex.input}" → {ex.action}')
        return 0

    print("available skills (use in agent.yaml: `skills: [...]`):\n")
    for skill in registry.list():
        origin = "" if skill.name in BUILTIN_SKILLS else "  [project]"
        binds = ", ".join((*skill.tool_packs, *skill.tools)) or "(no tools)"
        print(f"  {skill.name:<16} {skill.description}{origin}")
        print(f"          binds: {binds}")
        if skill.requires_skills:
            print(f"          requires: {', '.join(skill.requires_skills)}")
    scanned = [str(d) for d in discover_skill_dirs() if d.is_dir()]
    if scanned:
        print(f"\nproject skill dirs scanned: {', '.join(scanned)}")
    print("\nDetail: himmy skills <name>")
    return 0


# The consent (WS4.6) and privacy-audit (WS4.7 B4) subcommand trees live in their own
# modules (each has a richer parser); re-exported here so call sites that reference CLI
# handlers via ``commands`` keep working.
from himmy.cli.audit import (  # noqa: E402
    add_audit_parser,
    cmd_audit_privacy,
)
from himmy.cli.consent import (  # noqa: E402
    add_consent_parser,
    cmd_consent_deny,
    cmd_consent_grant,
    cmd_consent_history,
    cmd_consent_revoke,
    cmd_consent_status,
)

__all__ = [
    "add_consent_parser",
    "cmd_consent_grant",
    "cmd_consent_deny",
    "cmd_consent_status",
    "cmd_consent_history",
    "cmd_consent_revoke",
    "add_audit_parser",
    "cmd_audit_privacy",
]
