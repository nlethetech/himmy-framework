"""Command handlers for the ``himmy`` CLI.

Each ``cmd_*`` is a synchronous function returning a process exit code; the ones that
drive the async runtime wrap it in :func:`asyncio.run` internally so the argparse
dispatcher in :mod:`himmy.cli.__main__` stays plain. Everything defaults to the
offline stub, so ``himmy run``/``himmy chat`` work with no keys and no network.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from himmy.cli.provider import build_inference_for
from himmy.config.agent_spec import AgentSpec, load_agent_spec
from himmy.core.errors import HimmyError
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

    # Probe the machine the same way the init wizard does. When a REAL backend already
    # exists here, print the single exact flag to append to THIS command — never tell a
    # user to install a model they already have. Only fall back to install lines when
    # nothing real is detected.
    from himmy.cli.wizard import detect_provider_choices

    real = [c for c in detect_provider_choices() if c.key != "stub"]
    if real:
        best = real[0]
        rerun = f"--provider {best.key}"
        if best.model:
            rerun += f" --model {best.model}"
        _eprint(
            "note: running offline on the stub — canned deterministic output, not a real "
            "model.\n"
            f"  {best.label} is available here — re-run with a real backend:\n"
            f"    add:  {rerun}\n"
            "  or pin it in agent.yaml. details: himmy doctor\n"
        )
        return
    _eprint(
        "note: running offline on the stub — canned deterministic output, not a real "
        "model.\n"
        "  no real backend detected here — install one:\n"
        "    • local, free:  ollama pull llama3.2   then add  --provider ollama\n"
        "    • Claude Max:   --provider claude-cli   (needs the claude CLI on PATH)\n"
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

    Finally, a spec that would fall through to the offline stub because it named no
    provider and left ``model: default`` is resolved against the machine
    (:func:`_resolve_default_provider`) so a run picks up the local claude-cli/ollama
    doctor reports as ``[ok]`` — turning the misleading ``[stub:…]`` headline into a
    real answer without touching an explicit backend or the CLI overrides above.
    """
    spec = _apply_cli_overrides(_apply_himmy_md(_spec_from_args_inner(args)), args)
    return _resolve_default_provider(spec, args)


def _apply_cli_overrides(spec: AgentSpec, args: argparse.Namespace) -> AgentSpec:
    """Fold ``--provider``/``--model`` into the spec itself (see _spec_from_args)."""
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if provider:
        spec.provider = provider
    if model and model != "default":
        spec.model = model
    return spec


def _resolve_default_provider(spec: AgentSpec, args: argparse.Namespace) -> AgentSpec:
    """Resolve a provider-less ``model: default`` spec against THIS machine before a run.

    The framework's auto-select (``provider=None``) only reaches for a cloud SDK when a key
    is present — it never notices a local ``claude`` CLI or ``ollama`` server, so a spec that
    named no provider and left ``model: default`` runs on the offline stub even on a box where
    ``himmy doctor`` reports claude-cli/ollama ``[ok]``. That is the misleading ``[stub:…]``
    headline. Here we run the SAME probe the init wizard uses
    (:func:`~himmy.cli.wizard.detect_provider_choices`) and, when a real backend is detected,
    fold it into the spec so the run answers for real.

    Conservative by construction — only fires when the user has expressed no backend at all:
    a ``--provider``/``--model`` flag (already folded in by :func:`_apply_cli_overrides`), a
    ``provider:`` in the YAML, or a pinned ``model:`` all short-circuit it, so explicit config
    (including a deliberate ``provider: stub``) always stands. When nothing real is detected the
    spec is left untouched on the honest offline stub. Never writes to disk (that is the
    scaffold/deploy stamp's job — see :func:`_stamp_spec_provider_in_place`).
    """
    if getattr(args, "provider", None) or getattr(args, "model", None):
        return spec  # explicit CLI override — leave it alone
    if spec.provider or spec.model not in (None, "default"):
        return spec  # explicit spec config — never clobber
    from himmy.cli.wizard import detect_provider_choices

    best = detect_provider_choices()[0]
    if best.key == "stub":
        return spec  # nothing real detected — honest offline stub stands
    return spec.model_copy(
        update={"provider": best.key, "model": best.model or "default"}
    )


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
            f'  himmy run -f {target / "agent.yaml"} -p "{tmpl["prompt"]}"\n'
            + _deploy_next_tail(target / "agent.yaml")
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
        next_msg = spec_next_steps(target / "agent.yaml")

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
    # The classic/non-TTY default scaffold ships `model: default` (provider commented), which
    # resolves to the stub. Stamp the best detected backend into it so `himmy run` answers for
    # real out of the box — same probe the interactive wizard runs, minus the questions.
    if "agent.yaml" in files and not template and not args.team:
        _stamp_scaffold_provider(target / "agent.yaml")
        # Emit a ready-to-build container front door next to the spec: `docker build` from
        # this folder layers the agent onto the published runtime image (no framework
        # checkout). Never clobbers a Dockerfile the user already has.
        _emit_scaffold_dockerfile(target)
    print(next_msg)
    return 0


def _emit_scaffold_dockerfile(target: Path) -> None:
    """Write a 3-line agent ``Dockerfile`` next to a freshly-scaffolded ``agent.yaml``.

    Layers the user's spec onto the published runtime image (see :func:`_agent_dockerfile_text`)
    so ``docker build`` works straight from the scaffolded folder with no framework checkout. A
    pre-existing ``Dockerfile`` is left untouched (idempotent; never clobbers user content) —
    ``himmy init --force`` re-scaffolds the spec but the container recipe is the user's to own.
    """
    dockerfile = target / "Dockerfile"
    if dockerfile.exists():
        return
    try:
        dockerfile.write_text(_agent_dockerfile_text("agent.yaml"), encoding="utf-8")
    except OSError:
        return
    print(f"wrote {dockerfile}")


def _stamp_scaffold_provider(agent_yaml: Path) -> None:
    """Stamp the best detected backend into a freshly-written classic/non-TTY scaffold.

    Runs the SAME machine probe the init wizard uses
    (:func:`~himmy.cli.wizard.detect_provider_choices`) and, when a real backend is detected,
    fills in the scaffold's provider/model so ``himmy run -f agent.yaml`` answers for real out
    of the box. The edit is TEXTUAL (uncomment the ``# provider:`` line, set ``model:``) rather
    than a YAML re-dump, so the scaffold's teaching comments (the ``# skills:``/``# tool_packs:``
    guidance) survive. Only touches the two lines the template ships as unset defaults, so it is
    idempotent and never clobbers anything a user later pins.

    When NOTHING real is detected the file is left exactly as scaffolded (a commented
    ``provider:`` line + ``model: default``) and a note explains it will answer on the offline
    stub until a backend is installed — the honest fallback, never a fake provider.
    """
    from himmy.cli.wizard import detect_provider_choices

    best = detect_provider_choices()[0]
    if best.key == "stub":
        _eprint(
            "note: no real backend detected — the scaffold will answer on the offline stub "
            "(canned output).\n"
            "  install one for real answers:  ollama pull llama3.2   "
            "(or set --provider on run), then it just works."
        )
        return
    try:
        text = agent_yaml.read_text(encoding="utf-8")
    except OSError:
        return
    # Only stamp the scaffold's unset defaults: a commented `# provider:` line and
    # `model: default`. If the file has already been edited to pin either, leave it be.
    if "\nprovider:" in f"\n{text}" or "\nmodel: default" not in f"\n{text}":
        return
    model = best.model or "default"
    stamped = text.replace("model: default", f"model: {model}", 1).replace(
        "# provider: claude-cli", f"provider: {best.key}", 1
    )
    try:
        agent_yaml.write_text(stamped, encoding="utf-8")
    except OSError:
        return
    label = best.key + (f" · {best.model}" if best.model else "")
    _eprint(f"note: wired {label} (detected on this machine) into {agent_yaml}")


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


# --------------------------------------------------------------- agent-as-service

#: The one inbound channel `himmy serve -f` / `himmy worker -f` turn on: the generic
#: signed-webhook trigger. The Slack/Discord connectors need provider-specific secrets,
#: so the one-command front door wires only the channel-agnostic webhook.
_SERVICE_INBOUND_CONNECTOR = "webhook"
#: The sample payload source the summary's ready-to-paste curl uses (and that the
#: front door allow-lists when the operator has set no allowlist of their own).
_SERVICE_SAMPLE_SOURCE = "local"


def _service_agent_path(args: argparse.Namespace) -> str | None:
    """The agent.yaml a service should expose: ``-f``/``--agent``, or the nearest one.

    ``himmy serve -f agent.yaml`` names the file explicitly; ``--agent`` is an alias so the
    flag reads the same as ``himmy eval --agent``. With neither, an ``agent.yaml`` discovered
    upward from cwd (git-style, :func:`_discover_spec_file`) is used — so ``himmy serve`` in a
    scaffolded project just works. Returns ``None`` when nothing is configured (the BFF then
    boots with NO agent endpoint mounted, byte-identical to a bare ``himmy serve``).
    """
    explicit = getattr(args, "file", None) or getattr(args, "agent", None)
    if explicit:
        return str(Path(explicit).expanduser())
    discovered = _discover_spec_file()
    return str(discovered) if discovered is not None else None


def _stamp_inbound_provider(args: argparse.Namespace) -> None:
    """Honour ``--provider`` on a service by stamping the inbound provider override.

    ``himmy serve --provider ollama`` (and the same on ``worker``) makes the SERVED agent
    use that provider, mirroring the CLI run/chat ``--provider`` override. Written to the
    process env (the ``EnvSecrets`` link) so :func:`inbound_provider` — read at spec load in
    :func:`_build_inbound_handler` — applies it. Idempotent and never clobbering: with no
    ``--provider`` the spec's own provider stands (the override is only set when asked for),
    and an operator-set ``HIMMY_INBOUND_PROVIDER`` is left untouched.
    """
    from himmy.api.connector_inbound import INBOUND_PROVIDER_ENV

    provider = getattr(args, "provider", None)
    if provider and INBOUND_PROVIDER_ENV not in os.environ:
        os.environ[INBOUND_PROVIDER_ENV] = provider


def _enable_inbound_webhook(agent_path: str) -> str:
    """Point the inbound webhook at ``agent_path`` and ensure it can mount; return its secret.

    Wires the SAME machinery ``mount_inbound_connectors`` reads (never a parallel runtime):

    * ``HIMMY_INBOUND_AGENT_PATH`` names the agent an inbound delivery runs;
    * the webhook connector is enabled for the ``inbound`` surface;
    * a signing secret is ensured — an operator-configured one is honoured (never
      clobbered), else a fresh random one is generated so the public trigger is signed by
      construction (an unsigned webhook is a forgeable agent trigger, so the connector
      refuses to mount without a secret). A generated secret is PERSISTED through the active
      writable secrets provider (keychain/file) when one exists, so a restart reuses the SAME
      secret and previously-configured signed senders keep working; when the backend is
      read-only (the default ``env`` mode) it falls back to the process env for this run;
    * the anti-replay guard is turned ON for the auto-wired front door
      (``HIMMY_WEBHOOK_REQUIRE_TIMESTAMP``) so an accepted delivery MUST carry a fresh,
      skew-checked timestamp — a captured signed request goes stale instead of replaying
      forever;
    * the sample source is allow-listed ONLY when the operator has set no allowlist (an
      empty allowlist is default-deny — the safe posture — but then the ready-to-paste curl
      would 403; adding one source keeps default-deny for everyone else).

    Idempotent: an already-configured secret/allowlist/agent is left untouched. Returns the
    effective signing secret so the caller can render a VALID sample signature — the raw
    secret itself is NEVER printed or logged.
    """
    import secrets as _secrets

    from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV
    from himmy.config.secrets import get_secret, get_writable_provider
    from himmy.connectors.manage import _enabled_flag_name
    from himmy.connectors.webhook import WEBHOOK_SIGNING_SECRET

    os.environ[INBOUND_AGENT_PATH_ENV] = agent_path
    os.environ.setdefault(
        _enabled_flag_name(_SERVICE_INBOUND_CONNECTOR, "inbound"), "1"
    )
    # The auto-wired public trigger must not be replayable: require a fresh timestamp per
    # delivery. Set (not clobber) so an operator can still opt out deliberately.
    os.environ.setdefault("HIMMY_WEBHOOK_REQUIRE_TIMESTAMP", "1")
    secret = get_secret(WEBHOOK_SIGNING_SECRET)
    if not secret:
        secret = "whsec_" + _secrets.token_hex(24)
        # Persist through the writable provider FIRST so a restart reuses this exact secret
        # (previously-configured signed senders keep verifying). A read-only backend (env
        # mode) has none — fall back to the process env for this run.
        provider = get_writable_provider()
        persisted = False
        if provider is not None:
            try:
                provider.set(WEBHOOK_SIGNING_SECRET, secret)
                persisted = True
            except Exception:  # noqa: BLE001 - persistence is best-effort; env still works
                persisted = False
        if not persisted:
            os.environ[WEBHOOK_SIGNING_SECRET] = secret
    if not get_secret("HIMMY_WEBHOOK_ALLOWED_SOURCES"):
        os.environ["HIMMY_WEBHOOK_ALLOWED_SOURCES"] = _SERVICE_SAMPLE_SOURCE
    return secret


def _durable_store_path() -> str:
    """The durable SQLite run-store path the summary reports (``HIMMY_STORE_PATH``)."""
    from himmy.config.secrets import get_secret
    from himmy.services.storage.factory import HIMMY_STORE_PATH

    dsn = get_secret("HIMMY_DATABASE_URL")
    if dsn:
        return "postgres (HIMMY_DATABASE_URL)"
    return get_secret("HIMMY_STORE_PATH") or HIMMY_STORE_PATH


def render_service_summary(
    *,
    host: str | None,
    port: int | None,
    agent_path: str | None,
    signing_secret: str | None,
    store_path: str,
    apikey: str | None = None,
) -> str:
    """A boxed "your agent is live" summary, printed after a service binds.

    Shared by ``himmy serve``/``himmy worker`` (and reusable by ``himmy deploy``): shows the
    base URL, the mounted agent endpoint + a ready-to-paste **signed** curl, the docs/health/
    metrics routes, and the durable store path. The curl carries a signature this build's
    verifier accepts — computed with :func:`sign_webhook_body` over the SAME sample payload,
    BOUND to a fresh timestamp (the front door requires one — see the replay guard) and
    accompanied by the timestamp header — so a newcomer can prove the endpoint end to end in
    one paste, and the taught pattern is replay-safe (re-sending needs a fresh timestamp). The
    raw signing SECRET is NEVER included (only a valid signature for the sample body is); when
    no agent is wired the endpoint block is omitted (a bare BFF has no agent surface).

    When ``apikey`` is given (the ``--share`` path minted one and turned auth ON), the sample
    curl also carries the ``x-himmy-internal-key`` header — off-loopback the app requires it IN
    ADDITION to the webhook signature. The live key is NOT baked into the command: the header
    references ``$HIMMY_SHARE_KEY`` (which the operator exports) so the raw admin credential
    never lands in shell history/scrollback from pasting the curl.

    ``host`` is ``None`` for a ``himmy worker`` (no HTTP surface): the URL/endpoint/routes
    blocks are dropped and only the store path (+ agent, if any) is shown, since a worker
    reaches its agent through the run queue / scheduler, not an HTTP endpoint.
    """
    top = "  ┌─ your agent is live ─────────────────────────────"
    bottom = "  └──────────────────────────────────────────────────"
    lines = ["", top]
    http = host is not None and port is not None
    if http:
        base = f"http://{host}:{port}"
        lines.append(f"  │  {base}")
        if agent_path and signing_secret:
            import time as _time

            from himmy.api.connector_inbound import _INBOUND_PATHS
            from himmy.connectors.webhook import (
                DEFAULT_SIGNATURE_HEADER,
                DEFAULT_TIMESTAMP_HEADER,
                sign_webhook_body,
            )

            endpoint = f"{base}{_INBOUND_PATHS[_SERVICE_INBOUND_CONNECTOR]}"
            body = json.dumps(
                {"source": _SERVICE_SAMPLE_SOURCE, "text": "hello"},
                separators=(",", ":"),
            )
            # The auto-wired front door REQUIRES a fresh timestamp (replay guard), so the
            # sample curl computes the signature the SAME way — bound to a live timestamp —
            # and includes the timestamp header, so the taught pattern is replay-safe by
            # construction rather than an eternally-replayable body-only signature.
            timestamp = str(int(_time.time()))
            signature = sign_webhook_body(
                secret=signing_secret,
                body=body.encode("utf-8"),
                timestamp=timestamp,
            )
            lines += [
                "  │",
                f"  │  agent    {endpoint}",
                "  │  try it (signed; refresh the timestamp before re-sending):",
                f"  │    curl -s {endpoint} \\",
            ]
            if apikey:
                # --share off-loopback: auth is ON, so the endpoint needs the apikey header
                # IN ADDITION to the signature. Reference an env var rather than baking the
                # LIVE all-tenants key into a copy-paste command (a pasted literal lands in
                # shell history / scrollback). The header is x-himmy-internal-key (the
                # ApiKeyAuthenticator's header), NOT Authorization: Bearer (which it ignores).
                from himmy.api.auth.apikey import DEFAULT_HEADER as _APIKEY_HEADER

                lines.append(
                    f'  │      -H "{_APIKEY_HEADER}: $HIMMY_SHARE_KEY" \\'
                )
            lines += [
                f"  │      -H '{DEFAULT_TIMESTAMP_HEADER}: {timestamp}' \\",
                f"  │      -H '{DEFAULT_SIGNATURE_HEADER}: {signature}' \\",
                f"  │      -d '{body}'",
                "  │",
                f"  │  docs     {base}/docs",
                f"  │  ready    {base}/readyz",
                f"  │  metrics  {base}/metrics",
            ]
        else:
            lines += [
                "  │",
                f"  │  docs     {base}/docs",
                f"  │  ready    {base}/readyz",
                f"  │  metrics  {base}/metrics",
            ]
    else:
        lines.append("  │  worker (no HTTP endpoint — runs the queue + scheduler)")
        if agent_path:
            lines.append(f"  │  agent    {agent_path}")
    lines += [
        f"  │  store    {store_path}",
        bottom,
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------- boot-time key seeding

#: Platform secret carrying the FULL contents of the keys file as JSON text. Set it in the
#: hosting platform's secret store (fly secrets / render dashboard / railway variables) and
#: :func:`_materialize_api_keys_file` writes it to ``HIMMY_API_KEYS_FILE`` on boot.
API_KEYS_JSON_ENV = "HIMMY_API_KEYS_JSON"

#: Platform secret carrying a SINGLE plaintext key. Simpler than the full JSON: the boot
#: seeder wraps it in one all-tenants record so ``HIMMY_AUTH_MODE=apikey`` boots on 0.0.0.0.
API_KEY_ENV = "HIMMY_API_KEY"


def _materialize_api_keys_file() -> None:
    """Seed the apikey keys file from a platform secret when it is missing (boot self-heal).

    The cloud templates set ``HIMMY_AUTH_MODE=apikey`` + ``HIMMY_API_KEYS_FILE`` and bind
    0.0.0.0, but a hosting platform has no way to ship a pre-written JSON file into the
    container — so without this the file is absent and ``create_app`` fail-closes with a
    ``FileNotFoundError`` (the multi-tenant posture demands a tenant-binding keys file, and a
    shared ``HIMMY_INTERNAL_API_KEY`` alone is refused). This closes that gap: when apikey
    mode is on and the keys file does not yet exist, it is materialized ONCE from a platform
    secret, so the documented one-secret deploy actually boots authenticated.

    Two seed sources (checked in order, both sourced through the secrets layer so a secrets
    manager works too):

    * :data:`API_KEYS_JSON_ENV` — the literal JSON keys-file contents (``{secret: {...}}``);
      validated as a JSON object and written verbatim (full control over tenants/roles/expiry).
    * :data:`API_KEY_ENV` — a single plaintext key, wrapped in one all-tenants record.

    Idempotent + non-clobbering: does nothing when the file already exists (respecting a
    key minted by ``himmy apikey`` or a real mounted secret) or when auth is not apikey mode.
    NEVER logs the raw key material — only that a file was seeded and from which env name.
    """
    if os.environ.get("HIMMY_AUTH_MODE", "").lower() != "apikey":
        return
    keys_file = os.environ.get("HIMMY_API_KEYS_FILE")
    if not keys_file:
        return
    path = Path(keys_file).expanduser()
    if path.exists():
        return  # already provisioned (mounted secret, prior boot, or `himmy apikey mint`).

    from himmy.api.auth.apikey import _fingerprint
    from himmy.cli.apikey_cmd import _write_json_0600
    from himmy.config.secrets import get_secret

    raw_json = get_secret(API_KEYS_JSON_ENV)
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            _eprint(
                f"error: {API_KEYS_JSON_ENV} is not valid JSON ({exc}); cannot seed "
                f"{path} — the authenticated endpoint will not boot."
            )
            return
        if not isinstance(parsed, dict) or not parsed:
            _eprint(
                f"error: {API_KEYS_JSON_ENV} must be a non-empty JSON object "
                "{secret: {...}}; cannot seed the keys file."
            )
            return
        _write_json_0600(path, parsed)
        _eprint(f"seeded {path} from {API_KEYS_JSON_ENV} ({len(parsed)} key record(s)).")
        return

    single = get_secret(API_KEY_ENV)
    if single and single.strip():
        secret = single.strip()
        _write_json_0600(
            path,
            {
                secret: {
                    "subject": f"apikey:{_fingerprint(secret)}",
                    "tenant_ids": [],
                    "roles": [],
                    "all_tenants": True,
                    "disabled": False,
                }
            },
        )
        _eprint(f"seeded {path} from {API_KEY_ENV} (1 key record).")
        return

    # Nothing to seed from: leave the file absent so create_app fails closed with its clear
    # error. Point the operator at the one secret that makes the deploy boot.
    _eprint(
        f"note: HIMMY_AUTH_MODE=apikey but {path} is missing and neither "
        f"{API_KEYS_JSON_ENV} nor {API_KEY_ENV} is set — the authenticated endpoint will "
        "not boot. Set one of those platform secrets, or mint a key with `himmy apikey mint`."
    )


# ----------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> int:
    """Boot the FastAPI BFF via uvicorn (requires the ``api`` extra).

    With ``-f agent.yaml`` (or a discovered nearest one) the agent is exposed as a
    signature-verified HTTP endpoint (POST ``/v1/connectors/webhook``) via the SAME inbound
    machinery ``himmy deploy`` uses — reusing :mod:`himmy.api.connector_inbound`, so its
    default-deny + HMAC verification are untouched. With no agent the BFF boots exactly as
    before (no agent endpoint). Either way a boxed summary prints after bind.
    """
    try:
        import uvicorn

        from himmy.api.app import create_app
    except Exception as exc:  # pragma: no cover - optional extra missing
        _eprint(
            "error: `himmy serve` needs the 'api' extra: "
            f"pip install 'himmy[api]'  ({exc})"
        )
        return 1

    # Expose the agent as a signed webhook endpoint when one is configured/discovered.
    agent_path = _service_agent_path(args)
    signing_secret: str | None = None
    if agent_path is not None:
        _stamp_inbound_provider(args)  # honour --provider on the served agent
        signing_secret = _enable_inbound_webhook(agent_path)

    # Seed the apikey keys file from a platform secret if apikey mode is on but the file is
    # missing (cloud templates have no way to pre-write it) — so create_app boots authenticated
    # instead of fail-closing on a FileNotFoundError. No-op offline / when already provisioned.
    _materialize_api_keys_file()

    # Pass the bind host so create_app can fail closed when an unauthenticated
    # build would be exposed off-loopback (see _enforce_auth_posture).
    app = create_app(bind_host=args.host)
    _eprint(
        render_service_summary(
            host=args.host,
            port=args.port,
            agent_path=agent_path,
            signing_secret=signing_secret,
            store_path=_durable_store_path(),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------- worker


def cmd_worker(args: argparse.Namespace) -> int:
    """Run the durable run-queue dispatcher + routine scheduler with NO API server.

    The routines layer + the leased run dispatcher are normally only started inside the
    FastAPI server (the dispatcher in the app lifespan, the scheduler via a Studio-router
    startup event). A CLI / desktop user with no server therefore gets NEITHER — routines
    silently never fire offline and queued runs never drain. ``himmy worker`` closes that
    gap: in ONE process it constructs the SAME durable container the lifespan builds,
    publishes the store process-wide, enables dispatch, starts the :class:`RunDispatcher`
    (the queue consumer), and starts the :class:`RoutineScheduler` — then blocks until
    SIGINT/SIGTERM and shuts everything down gracefully (scheduler stop → dispatcher drain).

    Modes:
      * default — scheduler + queue worker (the desktop "everything offline" mode);
      * ``--no-scheduler`` — queue worker only (drain runs others enqueue, fire nothing);
      * ``--scheduler-only`` — scheduler only (no dispatcher; useful when a SEPARATE pool of
        workers drains the queue and this node just ticks routines).

    A durable run store is REQUIRED for the dispatcher (the queue's whole value is surviving
    a restart). ``--store`` sets ``HIMMY_STORE_PATH`` for the file-backed SQLite default;
    ``HIMMY_DATABASE_URL`` selects Postgres (and is what makes cross-node single-fire safe —
    see the runbook). ``--concurrency`` maps to ``HIMMY_DISPATCH_CONCURRENCY``.
    """
    import logging

    if args.store:
        os.environ["HIMMY_STORE_PATH"] = str(args.store)
    if args.concurrency is not None:
        os.environ["HIMMY_DISPATCH_CONCURRENCY"] = str(args.concurrency)
    # The dispatcher needs a durable run store; opt it in unless a Postgres DSN already does.
    os.environ.setdefault("HIMMY_DURABLE_STORAGE", "1")

    run_scheduler = not args.no_scheduler
    run_dispatcher = not args.scheduler_only
    if args.no_scheduler and args.scheduler_only:
        _eprint("error: --no-scheduler and --scheduler-only are mutually exclusive")
        return 1

    # ``himmy worker -f agent.yaml`` points the inbound agent at this file so any in-process
    # agent resolution (and a co-located `himmy serve`) uses the SAME spec. The worker has NO
    # HTTP surface, so it exposes no webhook endpoint — the summary reflects that.
    agent_path = _service_agent_path(args)
    if agent_path is not None:
        from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV

        os.environ[INBOUND_AGENT_PATH_ENV] = agent_path
        _stamp_inbound_provider(args)  # honour --provider on the worker's agent

    # Surface the worker's lifecycle log lines on stderr (the CLI default is quiet).
    logging.basicConfig(
        level=os.environ.get("HIMMY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    _eprint(
        render_service_summary(
            host=None,
            port=None,
            agent_path=agent_path,
            signing_secret=None,
            store_path=_durable_store_path(),
        )
    )

    try:
        asyncio.run(_run_worker(run_scheduler=run_scheduler, run_dispatcher=run_dispatcher))
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C before signal wiring
        _eprint("worker interrupted")
        return 130
    return 0


async def _run_worker(*, run_scheduler: bool, run_dispatcher: bool) -> None:
    """Async body of ``himmy worker``: bring up the substrate, block, tear down.

    Reuses the SHARED bootstrap (:mod:`himmy.api.runtime_bootstrap`) so the worker's run
    substrate is byte-identical to the FastAPI lifespan's — the dispatcher is gated on a
    durable store there, so a no-dispatcher fallback (in-memory) is honest about itself.
    The scheduler is the one piece the lifespan does NOT own (a Studio router event handler
    does), so the worker starts/stops it explicitly, gated on the same
    ``HIMMY_ROUTINES_SCHEDULER`` env as the router for parity.
    """
    import asyncio as _asyncio
    import logging
    import signal

    from himmy.api.auth import build_authenticator
    from himmy.api.routines import get_routines_store, get_scheduler
    from himmy.api.runtime_bootstrap import (
        build_durable_container,
        start_run_substrate,
        stop_run_substrate,
        wire_tool_authz,
    )
    from himmy.api.scheduler_leader import acquire_scheduler_leadership
    from himmy.config.flags import env_falsy, env_truthy
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )

    log = logging.getLogger("himmy.worker")

    # Default-ON kill-switch: stay enabled unless the operator wrote a recognised OFF token.
    # Routed through the canonical reader so the worker (the primary surface routines fire on)
    # honors the SAME HIMMY_ROUTINES_SCHEDULER vocabulary as studio_routines.py — e.g.
    # ``=n`` disables in both, never enabled here while disabled there.
    scheduler_enabled = not env_falsy("HIMMY_ROUTINES_SCHEDULER")

    # 1) server context FIRST so the container's spine + any in-process agent resolve the
    #    durable backend (the bootstrap relies on this being set before it builds).
    server_token = set_server_context(True)
    substrate = None
    scheduler = None
    leadership = None
    watchdog: _asyncio.Task[None] | None = None
    try:
        # 1b) multi-tenant fail-closed posture (G2), BEFORE anything executes: a worker
        #     draining a tenant queue / ticking tenant routines must refuse to start on an
        #     authenticator that mints every caller an all-tenants admin, EXACTLY like the
        #     FastAPI server (app._enforce_multi_tenant_posture). Without this a multi-tenant
        #     worker that cannot bind tenants would silently run every claimed run as admin.
        from himmy.api.app import _enforce_multi_tenant_posture

        _enforce_multi_tenant_posture(build_authenticator())

        # 2) build the durable container (Postgres / file-backed SQLite) + bring up the run
        #    substrate (publish store, enable dispatch BEFORE the sweep, sweep + reconcile,
        #    start dispatcher AFTER recovery, start dedup GC, install both providers).
        container, built_durable = await build_durable_container()
        substrate = await start_run_substrate(container, install_providers=True)

        # 2b) P0 tool-capability gate across the process boundary: wire the SAME two lines
        #     create_app's body wires (mark_auth_configured + run_app._access_policy). The
        #     worker dispatches claimed runs / fires routines through this run_app; without
        #     this the chokepoint sentinel stays OFF and a least-privilege tenant's work runs
        #     with the agent's FULL toolset (a cross-process confused-deputy escalation). A
        #     strict NO-OP with no authenticator configured (offline byte-unchanged).
        wire_tool_authz(getattr(substrate.active, "run_app", None))

        backend = substrate.backend
        if run_dispatcher and not substrate.dispatch_on:
            log.warning(
                "queue dispatcher NOT engaged: run store is %r (not durable). "
                "Set HIMMY_DATABASE_URL or HIMMY_DURABLE_STORAGE=1 for a durable queue.",
                backend,
            )
        elif not run_dispatcher and substrate.dispatcher is not None:
            # scheduler-only mode: stop the dispatcher the bootstrap started (the bootstrap
            # always starts it when the store is durable; this mode wants it off).
            await substrate.dispatcher.stop()
            substrate.dispatcher = None
            substrate.dispatch_on = False
            log.info("scheduler-only mode: leased dispatcher stopped")

        # 3) the routine scheduler (the lifespan does not own this — a router event does).
        #    Leader/worker split: bid for cross-node scheduler leadership. Only the LEADER
        #    ticks + runs maintenance sweeps; every other `himmy worker` drains the queue
        #    (worker-only) and promotes itself on the leader's failover. On SQLite the bid is
        #    the topology guard (same-host single-scheduler flock + cross-host warning).
        n_routines = 0
        if run_scheduler and scheduler_enabled:
            # Canonical truthy reader so ``=y`` requires-ack here exactly as it does via
            # env_truthy in studio_routines.py (the divergence the WP exists to kill).
            require_ack = env_truthy("HIMMY_SCHEDULER_REQUIRE_ACK")
            leadership = await acquire_scheduler_leadership(
                substrate.active, require_single_scheduler_ack=require_ack
            )
            if leadership.is_leader:
                scheduler = await _start_scheduler_as_leader(get_scheduler, log)
                log.info("scheduler leader (%s): ticking routines", leadership.mode)
            else:
                # worker-only this cycle; a watchdog will promote us on failover.
                log.info(
                    "scheduler NOT started on this node (%s): %s",
                    leadership.mode,
                    leadership.reason or "another node leads",
                )
            try:
                n_routines = len(get_routines_store().list())
            except Exception:  # noqa: BLE001 - a store read failure is non-fatal here
                n_routines = -1
            # Failover watchdog: a follower re-contends for the lease (promote on the prior
            # leader's death); a leader confirms its lease is still live (demote on drop).
            if leadership.mode in ("postgres-lease", "follower"):
                watchdog = _asyncio.create_task(
                    _scheduler_failover_watchdog(
                        leadership, get_scheduler, log
                    ),
                    name="himmy-scheduler-failover",
                )
        elif run_scheduler and not scheduler_enabled:
            log.info("routine scheduler disabled via HIMMY_ROUTINES_SCHEDULER")

        # 4) startup banner: what store, dispatch on, N routines.
        log.info(
            "himmy worker up: store=%s dispatch=%s scheduler=%s routines=%s "
            "(owner=%s)",
            backend,
            "on" if substrate.dispatch_on else "off",
            "on" if (scheduler is not None) else "off",
            (n_routines if n_routines >= 0 else "?"),
            (substrate.dispatcher.owner_id if substrate.dispatcher else "-"),
        )

        # 5) block until a stop signal arrives, then fall through to graceful shutdown.
        stop = _asyncio.Event()
        loop = _asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):  # pragma: no cover - non-POSIX
                pass
        await stop.wait()
        log.info("himmy worker stopping (signal received)")
    finally:
        # Graceful shutdown: stop the failover watchdog FIRST (so it can't promote/start the
        # scheduler mid-teardown), then the scheduler (stop firing new routine runs), release
        # the leadership lease, then the run substrate (drain in-flight dispatcher workers +
        # inline tasks, close the container, clear providers), then the server-context flag.
        if watchdog is not None:
            watchdog.cancel()
            with contextlib.suppress(_asyncio.CancelledError, Exception):
                await watchdog
        # The watchdog may have promoted a follower and started the scheduler after the local
        # ``scheduler`` var was set, so resolve the live singleton for a clean stop.
        active_scheduler = scheduler
        if active_scheduler is None:
            with contextlib.suppress(Exception):
                live = get_scheduler()
                if live.active:
                    active_scheduler = live
        if active_scheduler is not None:
            try:
                await active_scheduler.stop()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                log.warning("routine scheduler stop failed", exc_info=True)
        if leadership is not None:
            try:
                await leadership.release()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                log.warning("scheduler leadership release failed", exc_info=True)
        if substrate is not None:
            try:
                await stop_run_substrate(substrate, clear_providers=True)
            except Exception:  # noqa: BLE001 - shutdown best-effort
                log.warning("run substrate stop failed", exc_info=True)
        try:
            reset_server_context(server_token)
        except Exception:  # noqa: BLE001 - shutdown best-effort
            pass
        log.info("himmy worker stopped")


async def _start_scheduler_as_leader(get_scheduler: Any, log: Any) -> Any:
    """Run catch-up-on-launch then start the scheduler tick loop; return the scheduler.

    Catch-up reads the AUTHORITATIVE anchor (``last_run_at``) so a missed-while-asleep (or
    missed-while-this-node-was-a-follower) occurrence is coalesced/skipped/backfilled
    deterministically before the first tick — this is also what makes a NEWLY-PROMOTED leader
    fire any fire that fell in the failover gap (no permanently-missed fire).
    """
    scheduler = get_scheduler()
    try:
        actions = await scheduler.catch_up_on_launch()
        if actions:
            log.info("routine catch-up on launch: %s", actions)
    except Exception:  # noqa: BLE001 - catch-up must never block scheduler startup
        log.warning("routine catch-up on launch failed", exc_info=True)
    scheduler.start()
    return scheduler


async def _scheduler_failover_watchdog(
    leadership: Any, get_scheduler: Any, log: Any
) -> None:
    """Periodically refresh leadership: promote a follower / demote a dropped leader.

    A follower re-contends for the Postgres advisory lease; when the previous leader dies its
    lease auto-releases and this node wins the next bid → it runs catch-up (covering the
    failover gap) and starts ticking. A leader whose lease dropped (transient session loss)
    stops its tick loop so it doesn't fire without the lease — the dedup-CAS still guards the
    overlap, but demoting promptly keeps the cluster to one active ticker. Fail-open: an error
    is logged and the loop continues; cancellation (shutdown) propagates cleanly.
    """
    interval = _scheduler_failover_interval_seconds()
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        try:
            was_leader = leadership.is_leader
            now_leader = await leadership.refresh()
            scheduler = get_scheduler()
            if now_leader and not was_leader:
                log.info("scheduler FAILOVER: this node is now the leader")
                await _start_scheduler_as_leader(get_scheduler, log)
            elif not now_leader and was_leader:
                log.warning("scheduler demoted (lease lost); stopping tick loop")
                if scheduler.active:
                    await scheduler.stop()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the watchdog must never crash the worker
            log.warning("scheduler failover watchdog cycle failed", exc_info=True)


def _scheduler_failover_interval_seconds() -> float:
    """Seconds between failover re-contend checks (``HIMMY_SCHEDULER_FAILOVER_INTERVAL``).

    Defaults to 5s — the failover latency upper bound (a dead leader's lease frees instantly,
    so a follower picks it up within one interval). A typo / non-positive value falls back to
    the default so a misconfig never wedges failover.
    """
    raw = os.environ.get("HIMMY_SCHEDULER_FAILOVER_INTERVAL")
    if raw is None or not raw.strip():
        return 5.0
    try:
        value = float(raw.strip())
    except ValueError:
        return 5.0
    return value if value > 0 else 5.0


# ---------------------------------------------------------------------- deploy

#: Tool packs that only DO anything once a credential is configured, mapped to the
#: env/secret key(s) they read. ``himmy deploy`` preflights these BEFORE booting so a
#: newcomer sees exactly which keys are missing up front, instead of a live delivery
#: silently no-op'ing (or erroring) mid-task. Keyless packs (web, utils, data-sources,
#: memory, …) are absent by design — they work offline with nothing configured.
_PACK_REQUIRED_CREDS: dict[str, tuple[str, ...]] = {
    "comms": ("HIMMY_SMTP_HOST",),
    "telegram": ("HIMMY_TELEGRAM_BOT_TOKEN",),
    "google": ("HIMMY_GOOGLE_TOKEN",),
}


def _preflight_pack_credentials(spec: AgentSpec) -> list[str]:
    """Return the env keys the spec's credential-requiring packs need but that are unset.

    Reads through the secrets layer (:func:`himmy.config.secrets.get_secret`) so a
    file/keychain backend counts, not only ``os.environ``. Only packs the spec actually
    declares are checked; a key that resolves to a non-empty value is satisfied. The result
    is a de-duplicated, order-stable list of MISSING keys — the caller prints them as a
    warning and continues (deploy never fails mid-task on a preflight; the operator decides).
    """
    from himmy.config.secrets import get_secret

    missing: list[str] = []
    for pack in spec.tool_packs:
        for key in _PACK_REQUIRED_CREDS.get(pack, ()):  # keyless packs → no entry
            if not (get_secret(key) or "").strip() and key not in missing:
                missing.append(key)
    return missing


def spec_next_steps(agent_yaml: Path, *, spec: AgentSpec | None = None) -> str:
    """A spec-aware ``Next:`` block shown after scaffolding an ``agent.yaml``.

    The generic ``himmy run``/``himmy chat`` lines always come first. Then, when the spec is
    known, three targeted follow-ups the newcomer would otherwise have to discover:

    * a **creds** line when the spec's tool packs need env keys that aren't set yet (so the
      agent doesn't silently no-op its web/email/etc. tools — reuses the SAME
      :func:`_preflight_pack_credentials` deploy uses);
    * ``himmy routines add`` — schedule it to run unattended;
    * ``himmy deploy`` — stand it up as a live, signed HTTP service.

    ``spec`` may be passed by a caller that already has it (the wizard/``himmy new`` hold the
    validated dict); otherwise it is loaded from ``agent_yaml`` best-effort. A load failure
    just drops the spec-specific lines — the generic Next always renders.
    """
    ref = str(agent_yaml)
    lines = [
        "\nNext:",
        f'  himmy run -f {ref} -p "hello"      # one prompt',
        f"  himmy chat -f {ref}                # interactive",
        _deploy_next_tail(agent_yaml, spec=spec),
    ]
    return "\n".join(lines)


def _deploy_next_tail(agent_yaml: Path, *, spec: AgentSpec | None = None) -> str:
    """The creds + routines + deploy tail of a spec-aware ``Next:`` block (no run/chat lines).

    Shared by :func:`spec_next_steps` and the ``himmy init --template`` path (which keeps its
    own template-specific run line, then appends this). Loads ``agent_yaml`` best-effort when no
    ``spec`` is supplied; a load failure just drops the creds line.
    """
    ref = str(agent_yaml)
    if spec is None:
        with contextlib.suppress(Exception):
            spec = load_agent_spec(agent_yaml)
    tail: list[str] = []
    if spec is not None:
        with contextlib.suppress(Exception):
            missing = _preflight_pack_credentials(spec)
            if missing:
                tail.append(
                    f"  set {', '.join(missing)}   # your tool packs need these keys"
                )
    tail += [
        f'  himmy routines add --name daily -f {ref} -p "..." --daily 09:00'
        "   # run it on a schedule",
        f"  himmy deploy -f {ref}              # live, signed HTTP service",
    ]
    return "\n".join(tail)


def _stamp_spec_provider_in_place(
    agent_path: str, choice: Any
) -> tuple[str | None, str | None] | None:
    """Persist the best detected provider into ``agent.yaml`` so the service answers for REAL.

    Called ONLY when the resolved spec would run on the offline stub AND its ``provider`` is
    unset and its ``model`` is ``default`` — i.e. the user has expressed no explicit backend.
    Sets ``provider:`` (and ``model:`` when the detected choice carries one) and returns
    ``(provider, model)``. This is idempotent and NEVER clobbers explicit config: with a
    provider already set, or a model other than ``default``, the caller does not reach here, so
    a deliberate ``provider: stub`` or a pinned model always stands.

    The edit is TEXTUAL and comment-preserving — it mutates only the ``model:``/``provider:``
    lines rather than re-dumping the parsed YAML — so the user's hand-authored comments,
    formatting, and key order survive (a ``deploy`` the user expected only to SERVE must not
    silently reformat their spec). The write is ATOMIC (tmp file + ``os.replace``), so a
    concurrent stamp or a crash mid-write can never leave a truncated/half-written spec — the
    file every subsequent boot depends on. Falls back to a YAML re-dump ONLY when the textual
    anchors are absent (a non-scaffold spec), still writing atomically.

    Returns ``None`` when nothing was written (unreadable/absent file) so the caller can fall
    back to a clearly-labelled stub rather than claim a real backend it did not stamp.
    """
    import yaml

    path = Path(agent_path)
    try:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    # Idempotent + never-clobber: only stamp when the user set neither provider nor model.
    if raw.get("provider") or (raw.get("model") not in (None, "default")):
        return raw.get("provider") or None, raw.get("model") or None

    model = choice.model or "default"
    stamped = _textual_stamp_provider(text, provider=choice.key, model=model)
    if stamped is None:
        # No textual anchors (a spec that didn't come from the scaffold): fall back to a
        # structured re-dump. Comments are lost in this branch only, but the common
        # scaffold path above preserves them; the write is still atomic.
        raw["provider"] = choice.key
        if choice.model:
            raw["model"] = choice.model
        stamped = yaml.safe_dump(raw, sort_keys=False)
    if not _atomic_write_text(path, stamped):
        return None
    return choice.key, choice.model


def _textual_stamp_provider(text: str, *, provider: str, model: str) -> str | None:
    """Set ``provider:``/``model:`` in a scaffold spec by line edit (comments preserved).

    Mirrors :func:`_stamp_scaffold_provider`: uncomment the ``# provider: ...`` template line
    and replace ``model: default``, touching nothing else. Returns the new text, or ``None``
    when the expected anchors are absent (so the caller can fall back to a structured re-dump).
    """
    guard = f"\n{text}"
    if "\nprovider:" in guard or "\nmodel: default" not in guard:
        return None
    stamped = text.replace("model: default", f"model: {model}", 1)
    if "# provider: claude-cli" in stamped:
        return stamped.replace("# provider: claude-cli", f"provider: {provider}", 1)
    # Model anchor present but no commented provider line: append the provider setting.
    return stamped.replace(
        f"model: {model}", f"model: {model}\nprovider: {provider}", 1
    )


def _atomic_write_text(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``). True on success.

    Mirrors the api_keys writer's care: a crash or a concurrent writer can never observe a
    truncated file — the replace is atomic, so a reader sees either the old or the new whole
    contents. Returns ``False`` on an OS error so the caller can fall back cleanly.
    """
    import tempfile

    try:
        directory = path.parent
        fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".stamp-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                if os.path.exists(tmp):
                    os.remove(tmp)
    except OSError:
        return False
    return True


def _deploy_resolve_provider(agent_path: str) -> tuple[str | None, str | None, str | None]:
    """Ensure the deployed agent answers for real: stamp a backend, or name the install line.

    Loads the spec, and when it would resolve to the offline stub runs the SAME machine probe
    the init wizard uses (:func:`detect_provider_choices`). If a real backend is detected it is
    stamped into ``agent.yaml`` (idempotent, never clobbering — see
    :func:`_stamp_spec_provider_in_place`) so the service returns real answers. If nothing is
    detected the spec is left on the stub and the ONE exact install line is returned for the
    caller to print (deploy continues on a clearly-labelled stub rather than refusing to boot).

    Returns ``(provider, model, install_hint)``: ``install_hint`` is non-None only when the
    service is booting on the stub because no real backend was found.
    """
    from himmy.cli.wizard import detect_provider_choices

    spec = from_spec.load_spec_file(agent_path)
    if spec.provider or spec.model not in (None, "default"):
        return spec.provider, (None if spec.model == "default" else spec.model), None
    from himmy.cli.provider import resolves_to_stub

    if not resolves_to_stub(spec.provider, None if spec.model == "default" else spec.model):
        return spec.provider, None, None
    best = detect_provider_choices()[0]
    if best.key == "stub":
        hint = (
            "no real backend detected — the service will answer on the offline stub "
            "(canned output).\n"
            "  install one for real answers:  ollama pull llama3.2   "
            "(then re-run himmy deploy)"
        )
        return None, None, hint
    stamped = _stamp_spec_provider_in_place(agent_path, best)
    if stamped is None:
        return None, None, None
    return stamped[0], stamped[1], None


async def _serve_and_worker(
    app: Any, host: str, port: int, *, run_scheduler: bool, run_dispatcher: bool
) -> None:
    """Run the FastAPI server AND the worker substrate in ONE supervised process group.

    Both live in a single event loop so a SIGINT/SIGTERM (or the server exiting because the
    port is in use) tears BOTH down together — no orphaned worker draining a queue after the
    HTTP surface is gone. The uvicorn ``Server`` is driven programmatically (not
    ``uvicorn.run``) so we own its lifecycle; the worker substrate is the SAME
    :func:`_run_worker` body ``himmy worker`` uses, cancelled on shutdown so it drains
    cleanly. Whichever task finishes first (server stop, or the worker's own signal wait)
    cancels the other, giving one clean joint shutdown.
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    # Let our own task-group own the signal handling / joint shutdown, not uvicorn's
    # per-process installer (which would race the worker's handler for the same signal).
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign,attr-defined]

    server_task = asyncio.create_task(server.serve(), name="himmy-deploy-server")
    worker_task = asyncio.create_task(
        _run_worker(run_scheduler=run_scheduler, run_dispatcher=run_dispatcher),
        name="himmy-deploy-worker",
    )
    try:
        done, pending = await asyncio.wait(
            {server_task, worker_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        # Ask the server to exit, then cancel any still-pending task so neither is orphaned.
        server.should_exit = True
        for task in (server_task, worker_task):
            if not task.done():
                task.cancel()
        for task in (server_task, worker_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    # Surface a boot failure (e.g. port already in use) as an exception the caller reports.
    if server_task in done and not server_task.cancelled():
        exc = server_task.exception()
        if exc is not None:
            raise exc


def cmd_deploy(args: argparse.Namespace) -> int:
    """One command from an ``agent.yaml`` to a live, reachable, signed service.

    The delightful front door over the machinery ``himmy serve``/``himmy worker`` already
    build — it does NOT invent a new runtime. In order it:

    1. resolves + validates the spec (reusing ``himmy validate``'s findings) and preflights
       the credential-requiring tool packs, printing exactly which env keys are missing
       BEFORE booting (never failing mid-task);
    2. ensures the service answers for REAL — when the spec would run on the offline stub it
       stamps the best detected local backend into ``agent.yaml`` (idempotent, never
       clobbering explicit config), or prints the one install line and continues on a
       clearly-labelled stub;
    3. points the inbound webhook at the file and auto-generates + persists a signing secret
       via the secrets layer (the raw secret is never printed — only a valid sample
       signature);
    4. boots ONE supervised process group — ``create_app`` (serve) bound ``127.0.0.1:8000``
       by default (fail-closed off-loopback) PLUS the worker (scheduler + durable run-queue
       dispatcher) on ``.himmy/storage.db`` — with a clean joint shutdown on Ctrl-C/SIGTERM
       and a clear message when the port is in use;
    5. prints the boxed live summary with a working signed curl.

    ``--channel telegram``/``studio`` wrap the existing entrypoints (restart-on-failure);
    ``--host``/``--port`` override the bind; ``--share`` mints real apikey auth (turning it ON)
    and prints a ``cloudflared``/``ngrok`` tunnel command so a friend can reach the service —
    auth is provisioned BEFORE any public command is printed, never an open endpoint;
    ``--docker`` emits a minimal Dockerfile and exits.
    """
    channel = getattr(args, "channel", None) or "http"
    if getattr(args, "docker", False):
        return _emit_deploy_dockerfile(args)
    if channel == "telegram":
        return _deploy_channel_with_restart(commands_telegram_wrapper, args, "telegram")
    if channel == "studio":
        return _deploy_channel_with_restart(cmd_studio, args, "studio")

    agent_path = _service_agent_path(args)
    if agent_path is None:
        _eprint(
            "error: no agent.yaml found here (or above) — run `himmy init` first, or "
            "pass one: himmy deploy -f path/to/agent.yaml"
        )
        return 2

    # 1) validate the spec up front (same findings as `himmy validate`) — a broken spec
    #    should fail here, not after the port is bound.
    from himmy.cli.agents import _findings_for

    findings = _findings_for(Path(agent_path).expanduser())
    if findings:
        _eprint(f"error: {agent_path} has {len(findings)} problem(s):")
        for f in findings:
            _eprint(f"  - {f}")
        _eprint("fix them (see `himmy validate`) and re-run.")
        return 1

    # 1b) preflight credential-requiring packs BEFORE booting (warn, never fail mid-task).
    spec = from_spec.load_spec_file(agent_path)
    missing = _preflight_pack_credentials(spec)
    if missing:
        _eprint("note: some tool packs need credentials that are not set yet:")
        for key in missing:
            _eprint(f"  - {key}")
        _eprint("  those tools will no-op until you set the key(s); the rest deploy fine.\n")

    # 2) make the service answer for REAL (stamp a backend, or name the install line).
    _provider, _model, install_hint = _deploy_resolve_provider(agent_path)
    if install_hint:
        _eprint(f"note: {install_hint}\n")

    # 3) --share: real auth ON BEFORE we print any public tunnel command (fail-closed).
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or 8000
    share_apikey: str | None = None
    if getattr(args, "share", False):
        ok, share_apikey = _deploy_configure_share_auth(host)
        if not ok:
            return 2

    # 4) wire the signed webhook (auto-secret via the secrets layer; never printed raw).
    _stamp_inbound_provider(args)
    signing_secret = _enable_inbound_webhook(agent_path)

    try:
        import uvicorn  # noqa: F401

        from himmy.api.app import create_app
    except Exception as exc:  # pragma: no cover - optional extra missing
        _eprint(
            "error: `himmy deploy` needs the 'api' extra: "
            f"pip install 'himmy[api]'  ({exc})"
        )
        return 1

    # Durable run store for the worker half (survives restarts). Loopback-safe default.
    os.environ.setdefault("HIMMY_DURABLE_STORAGE", "1")

    # Seed the apikey keys file from a platform secret when apikey mode is on but the file is
    # absent (the cloud-template path: HIMMY_AUTH_MODE=apikey set, no way to ship the JSON in).
    # Runs AFTER --share (which mints its own key + file), so it only fires for the env-driven
    # hosted case; no-op when already provisioned.
    _materialize_api_keys_file()

    try:
        app = create_app(bind_host=host)
    except HimmyError as exc:
        # Fail-closed posture: off-loopback with no auth is refused — point at --share.
        # create_app raises HimmyError (subclasses Exception, NOT RuntimeError), so this
        # is the type we must catch for the friendly guidance to fire.
        # The deploy aborted before serving — revoke the share key we minted so it does not
        # linger as a live, never-expiring all-tenants credential on disk.
        _revoke_share_key(share_apikey)
        _eprint(f"error: {exc}")
        _eprint("  add auth before exposing off-loopback:  himmy deploy --share")
        return 2

    _eprint(
        render_service_summary(
            host=host,
            port=port,
            agent_path=agent_path,
            signing_secret=signing_secret,
            store_path=_durable_store_path(),
            apikey=share_apikey,
        )
    )
    if getattr(args, "share", False):
        # Auth is already minted + on (above) — only now do we print a public command.
        _eprint(render_share_tunnel(host=host, port=port))
    _eprint("  (serve + worker running together — Ctrl-C stops both)\n")

    try:
        asyncio.run(
            _serve_and_worker(
                app, host, port, run_scheduler=True, run_dispatcher=True
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C
        _eprint("\n(stopped)")
        return 130
    except OSError as exc:
        # The server never came up (most commonly the port is already in use) — revoke the
        # minted share key so a failed --share attempt doesn't leave an orphaned live key.
        _revoke_share_key(share_apikey)
        if getattr(exc, "errno", None) in (48, 98) or "address already in use" in str(
            exc
        ).lower():
            _eprint(
                f"error: port {port} is already in use — stop the other process or "
                f"pass --port <n>."
            )
            return 1
        _eprint(f"error: failed to bind {host}:{port}: {exc}")
        return 1
    return 0


def commands_telegram_wrapper(args: argparse.Namespace) -> int:
    """Adapter so ``himmy deploy --channel telegram`` reuses ``cmd_telegram`` verbatim."""
    return cmd_telegram(args)


#: Exit code the channel entrypoints (``cmd_telegram``/``cmd_studio``) return for a
#: PERMANENT, deterministic misconfiguration (a missing bot token, a held process lock) —
#: as opposed to a transient crash. The restart supervisor treats this as fatal and gives up
#: immediately rather than thrashing forever on a config error a restart can never heal.
_DEPLOY_FATAL_EXIT = 2

#: A run that stays up at least this long is treated as "healthy" — the restart backoff is
#: reset to its floor afterwards, so a daemon that survives for hours then blips restarts
#: quickly rather than at the last (long) backoff of an unrelated earlier crash-loop.
_DEPLOY_HEALTHY_RUNTIME_S = 60.0


def _deploy_channel_with_restart(
    fn: Any, args: argparse.Namespace, label: str
) -> int:
    """Run a channel entrypoint with restart-on-failure (a long-running daemon should heal).

    ``himmy deploy --channel telegram``/``studio`` is meant to STAY up; a TRANSIENT crash
    (network blip, provider hiccup) should not end the deployment, so this restarts ``fn`` on a
    non-zero return or an exception, backing off up to a cap, and stops cleanly on Ctrl-C. A
    clean exit (``0``) is honoured — the operator stopped it on purpose.

    A PERMANENT misconfiguration is NOT retried: when the entrypoint returns
    :data:`_DEPLOY_FATAL_EXIT` (2) — a missing bot token, a held Telegram lock, any
    deterministic startup refusal — restarting can never heal it, so the supervisor gives up
    immediately and propagates the failure (a wrapping ``systemd``/``docker`` unit then sees a
    real non-zero exit instead of a pinned, silently-looping process). The backoff is RESET
    after a run that stayed up past :data:`_DEPLOY_HEALTHY_RUNTIME_S`, so a long-lived daemon
    that blips restarts fast rather than at a stale long backoff.
    """
    import time

    backoff = 1.0
    while True:
        started = time.monotonic()
        try:
            code = fn(args)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            _eprint(f"\n({label} stopped)")
            return 130
        except Exception as exc:  # noqa: BLE001 - a daemon heals rather than dies
            _eprint(f"⚠ {label} crashed: {exc} — restarting in {backoff:.0f}s")
        else:
            if code == 0:
                return 0
            if code == _DEPLOY_FATAL_EXIT:
                # A permanent config error (missing credential / held lock): a restart can
                # never fix it, so fail fast rather than loop forever pinning the supervisor.
                _eprint(
                    f"error: {label} exited ({code}) — this looks like a permanent "
                    "configuration error, not a transient crash. Fix the config and re-run; "
                    "not restarting."
                )
                return code
            _eprint(f"⚠ {label} exited ({code}) — restarting in {backoff:.0f}s")
        # Reset the backoff after a run that stayed up long enough to be "healthy", so an
        # unrelated later blip doesn't inherit a long backoff from an earlier crash-loop.
        if time.monotonic() - started >= _DEPLOY_HEALTHY_RUNTIME_S:
            backoff = 1.0
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            return 130
        backoff = min(backoff * 2, 30.0)


def _deploy_configure_share_auth(host: str) -> tuple[bool, str | None]:
    """Mint an apikey and turn auth ON — the mandatory pre-step before ``--share`` exposes.

    ``--share`` puts the service on the public internet (a cloudflared/ngrok tunnel to the
    local port), so it MUST be authenticated by construction — an open admin endpoint reachable
    from anywhere is exactly the posture we refuse to ship. This ALWAYS mints a real API key
    (even on a loopback bind, because the TUNNEL is the exposure, not the bind) into the default
    keys file and sets ``HIMMY_AUTH_MODE=apikey`` + ``HIMMY_API_KEYS_FILE`` for THIS process, so
    the exposed surface needs the key. The minted key is printed ONCE (it is the credential the
    operator hands to whoever they share the tunnel URL with).

    Returns ``(ok, apikey)``: ``apikey`` is the minted secret (so the caller can thread it into
    the sample curl — the exposed endpoint needs it in addition to the webhook signature), or
    ``(False, None)`` when auth could NOT be provisioned (a read/write error on the keys file),
    in which case ``--share`` REFUSES rather than exposing an unauthenticated endpoint.
    """
    import secrets as _secrets

    from himmy.api.auth.apikey import _fingerprint
    from himmy.cli.apikey_cmd import _load_keys, _write_json_0600

    keys_file = os.environ.get("HIMMY_API_KEYS_FILE") or ".himmy/api_keys.json"
    path = Path(keys_file)
    try:
        keys = _load_keys(path)
    except (OSError, ValueError) as exc:
        _eprint(f"error: could not read keys file {path}: {exc}")
        _eprint("  --share refuses to expose an unauthenticated endpoint — aborting.")
        return False, None
    secret = f"himmy_{_secrets.token_urlsafe(32)}"
    keys[secret] = {
        "subject": f"apikey:{_fingerprint(secret)}",
        "tenant_ids": [],
        "roles": [],
        "all_tenants": True,
        "disabled": False,
    }
    try:
        _write_json_0600(path, keys)
    except OSError as exc:
        _eprint(f"error: could not write keys file {path}: {exc}")
        _eprint("  --share refuses to expose an unauthenticated endpoint — aborting.")
        return False, None
    os.environ["HIMMY_API_KEYS_FILE"] = str(path)
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    # Keep the shared "friend" key off the observability surface: /metrics authenticates any
    # valid principal, so without its OWN token the shared key could scrape the deployment-wide
    # authz-deny / latency / token-cost map. Provision a SEPARATE metrics token (not the shared
    # key, not printed) so /metrics stays out of the shared credential's reach. Set (not
    # clobber) so an operator-configured token stands.
    os.environ.setdefault("HIMMY_METRICS_TOKEN", _secrets.token_urlsafe(32))
    from himmy.api.auth.apikey import DEFAULT_HEADER as _APIKEY_HEADER

    _eprint(
        "--share: minted an API key and enabled auth (required before the tunnel "
        "exposes this service).\n"
        f"  send it as the {_APIKEY_HEADER} header. This is a LIVE admin credential and is\n"
        "  shown ONCE — save it now (it will not be shown again):\n\n"
        f"    {secret}\n\n"
        "  the sample curl references it as $HIMMY_SHARE_KEY — export it (keeps the raw\n"
        "  key out of your shell history):\n"
        "    export HIMMY_SHARE_KEY=<the key above>\n"
    )
    return True, secret


def _revoke_share_key(secret: str | None) -> None:
    """Remove a just-minted ``--share`` key from the keys file when the deploy aborts.

    ``--share`` mints a live all-tenants key + persists it BEFORE the server binds. If the
    deploy then aborts (fail-closed refusal, port-in-use), that key would otherwise linger on
    disk as a never-expiring admin credential that a later server reading the same keys file
    would honour — and repeated failed attempts would accrete more. Deleting exactly the
    minted secret on abort keeps the file clean without touching any operator-managed key.
    Best-effort and silent on error (a stale key is a hygiene issue, never a boot blocker).
    """
    if not secret:
        return
    from himmy.cli.apikey_cmd import _load_keys, _write_json_0600

    keys_file = os.environ.get("HIMMY_API_KEYS_FILE") or ".himmy/api_keys.json"
    path = Path(keys_file)
    try:
        keys = _load_keys(path)
    except (OSError, ValueError):
        return
    if keys.pop(secret, None) is None:
        return
    with contextlib.suppress(OSError):
        _write_json_0600(path, keys)


def render_share_tunnel(*, host: str, port: int) -> str:
    """A boxed 'share it with a friend' block: copy-paste tunnel commands to the local port.

    Printed after ``--share`` has ALREADY minted an API key + turned auth on (so the endpoint the
    tunnel exposes is authenticated — we never print a public command for an open endpoint). It
    offers two keyless one-liners, cloudflared (no signup) and ngrok, each tunnelling to the
    exact ``host:port`` the service bound. The public HTTPS URL the tool prints is what the
    operator hands out; callers still need the API key + a valid webhook signature to reach the
    agent, so exposure stays fail-closed end to end.
    """
    target = f"http://{host}:{port}"
    return "\n".join(
        [
            "  ┌─ share it (public tunnel to the local port) ─────",
            "  │  auth is ON, so the exposed endpoint needs the API key above.",
            "  │  run ONE of these; it prints a public https URL to hand out:",
            "  │",
            "  │  cloudflared (no signup):",
            f"  │    cloudflared tunnel --url {target}",
            "  │",
            "  │  ngrok (needs a free account):",
            f"  │    ngrok http {port}",
            "  └──────────────────────────────────────────────────",
            "",
        ]
    )


# The published runtime image (built + pushed by .github/workflows/deploy.yml on
# tag/release). It ships himmy[api] already installed, so a user's agent Dockerfile is a
# thin 3-line FROM/COPY/CMD over it — no framework checkout, no pip install at build time.
GHCR_IMAGE = "ghcr.io/nlethetech/himmy"

# Port the emitted agent Dockerfile binds. Pinned to the base image's EXPOSE/HEALTHCHECK port
# (Dockerfile: 8765) so the inherited health probe curls the port the CMD actually serves — a
# mismatch would leave the container permanently `(unhealthy)`.
AGENT_IMAGE_PORT = 8765


def _agent_image_ref() -> str:
    """The pinned base image an emitted agent Dockerfile builds ``FROM``.

    ``ghcr.io/nlethetech/himmy:<version>`` where ``<version>`` is the installed
    :data:`himmy.__version__` — so the generated Dockerfile pins the image that matches the
    CLI that wrote it (reproducible; never a floating ``:latest``).
    """
    from himmy import __version__

    return f"{GHCR_IMAGE}:{__version__}"


def _agent_dockerfile_text(agent_name: str) -> str:
    """Render the 3-line agent Dockerfile that layers an ``agent.yaml`` onto the runtime image.

    The delightful container front door: ``FROM`` the published runtime image (himmy already
    installed), ``COPY`` the user's spec in, ``CMD`` ``himmy deploy`` it — so ``docker build``
    works from a pip-install user's agent folder with NO framework checkout. ``agent_name`` is
    the on-disk spec filename (copied to ``/app/agent.yaml`` inside the image so the CMD path is
    stable regardless of what the user named it).

    Port is pinned to :data:`AGENT_IMAGE_PORT` (8765) so the CMD reuses the base image's already
    -correct ``EXPOSE`` + ``HEALTHCHECK`` verbatim — a mismatched port would probe a closed
    socket and leave the container permanently ``(unhealthy)``.

    Security posture is FAIL-CLOSED by default: this auto-emitted recipe binds loopback and
    does NOT bake an unauthenticated-proxy opt-in. Bound to ``127.0.0.1`` the server stays
    reachable to the in-container healthcheck but is not reachable through a mapped ``-p`` port,
    so ``himmy deploy``'s off-loopback refusal guides the user to add real auth (a COMMENTED
    opt-in block below shows exactly how) rather than shipping an open-by-default container. The
    agent webhook endpoint stays signature-verified + default-deny regardless.
    """
    return (
        "# Container for this himmy agent — layered on the published runtime image so\n"
        "# `docker build` works from this folder with no framework checkout. Build:\n"
        "#   docker build -t my-agent .\n"
        f"FROM {_agent_image_ref()}\n"
        f"COPY {agent_name} /app/agent.yaml\n"
        "# Fail-closed by default: binds 127.0.0.1, so the in-container healthcheck (the base\n"
        "# image probes http://127.0.0.1:8765/readyz) passes while the port is NOT reachable\n"
        "# through `-p`. To expose it, add REAL auth first, then bind 0.0.0.0 — either run\n"
        "# `himmy deploy --share` (mints an api key), or set HIMMY_API_KEYS_FILE /\n"
        "# HIMMY_AUTH_MODE and uncomment the two lines below (0.0.0.0 needs auth or himmy\n"
        "# refuses to boot). Only if auth is terminated at a trusted proxy in front of the\n"
        "# container is the unauthenticated opt-in appropriate:\n"
        "#   ENV HIMMY_ALLOW_UNAUTHENTICATED=1\n"
        f'#   CMD ["himmy", "deploy", "-f", "agent.yaml", "--host", "0.0.0.0", "--port", "{AGENT_IMAGE_PORT}"]\n'  # noqa: E501
        f'CMD ["himmy", "deploy", "-f", "agent.yaml", "--host", "127.0.0.1", "--port", "{AGENT_IMAGE_PORT}"]\n'
    )


def _emit_deploy_dockerfile(args: argparse.Namespace) -> int:
    """Print the agent Dockerfile that runs ``himmy deploy`` for this agent, and exit.

    The 3-line container: ``FROM`` the published runtime image, ``COPY`` the agent, ``CMD``
    ``himmy deploy``. Emitted to stdout so ``himmy deploy --docker > Dockerfile`` just works;
    the agent path is taken from the same resolution the live deploy uses so the image serves
    the SAME spec. Emits WITHOUT booting a server (pure text, no bind).
    """
    agent_path = _service_agent_path(args) or "agent.yaml"
    print(_agent_dockerfile_text(Path(agent_path).name), end="")
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
        # A missing capability prints the EXACT install command for it — no guessing which
        # extra name maps to the feature you just tried and lack.
        hint = (
            f"  → pip install 'himmy[{extra.extra}]'"
            if not extra.ok and extra.extra
            else ""
        )
        print(f"  [{'ok ' if extra.ok else '-- '}] {extra.label}{hint}")

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

    # The runtime section is shown by default (it's the "will my routines actually fire?"
    # answer newcomers most need); ``--runtime`` is accepted as an explicit opt-in too.
    _doctor_runtime_section()

    if getattr(args, "storage", False):
        _doctor_storage_section()
    return 0


def _doctor_runtime_section() -> int:
    """Report the unattended-run substrate: scheduler, durable store, and routine coverage.

    Answers the single question a routine author most needs — "is anything going to FIRE my
    routines?" — from the SAME machinery the runtime uses (no parallel probe):

    * **scheduler** — a live ``himmy worker`` / co-located ``himmy serve`` holds the host
      scheduler ``flock`` (:func:`~himmy.api.scheduler_leader.scheduler_running_on_host`);
    * **durable store** — a Postgres DSN or the file-backed SQLite default is engaged (the
      dispatcher/queue need durability; an in-memory store loses queued runs on exit);
    * **routines** — how many local routines are defined + how many are enabled.

    The load-bearing line is the RED one: when enabled routines EXIST but no scheduler is
    running here, they silently never fire — the most common routines footgun. Returns the
    count of enabled-but-unscheduled routines (0 = healthy) so a caller/test can assert on it.
    """
    from himmy.api.scheduler_leader import scheduler_running_on_host

    print("\nruntime (unattended runs):")
    running = scheduler_running_on_host()
    if running is True:
        print("  [ok ] scheduler: running on this host")
    elif running is False:
        print("  [-- ] scheduler: not running here (start `himmy worker`)")
    else:
        print("  [ ? ] scheduler: undetermined (Postgres lease / non-POSIX host)")

    durable, store_label = _durable_store_engaged()
    flag = "ok " if durable else "-- "
    print(f"  [{flag}] durable store: {store_label}")

    enabled, total = _local_routine_counts()
    print(f"  [{'ok ' if total else '-- '}] routines: {total} defined, {enabled} enabled")

    unscheduled = enabled if (enabled > 0 and running is False) else 0
    if unscheduled:
        print(
            f"  RED: {unscheduled} enabled routine(s) but NO scheduler is running — they "
            "will NEVER fire.\n"
            "       start one:  himmy worker   (or `himmy deploy -f agent.yaml`)"
        )
    return unscheduled


def _durable_store_engaged() -> tuple[bool, str]:
    """Whether a DURABLE run store is engaged, plus a short label (Postgres vs SQLite path).

    A Postgres ``HIMMY_DATABASE_URL`` or an opted-in durable SQLite store counts as durable;
    a bare in-memory default does not (queued runs die with the process). Mirrors the factory's
    selection so the report matches what a worker would actually use.
    """
    from himmy.config.flags import env_truthy
    from himmy.config.secrets import get_secret
    from himmy.services.storage.factory import HIMMY_STORE_PATH, _is_postgres_dsn

    dsn = get_secret("HIMMY_DATABASE_URL")
    if _is_postgres_dsn(dsn):
        return True, "postgres (HIMMY_DATABASE_URL)"
    store_path = get_secret("HIMMY_STORE_PATH") or HIMMY_STORE_PATH
    # The file-backed SQLite store is durable; a worker opts it in via HIMMY_DURABLE_STORAGE,
    # but the file itself persists regardless, so a configured path is the honest "durable" bit.
    durable = bool(store_path) or env_truthy("HIMMY_DURABLE_STORAGE")
    return durable, f"sqlite ({store_path})"


def _local_routine_counts() -> tuple[int, int]:
    """``(enabled, total)`` local-workspace routines, best-effort (``(0, 0)`` on any error)."""
    try:
        from himmy.api import routines as svc

        routines = svc.get_routines_store().list(workspace_id=svc.LOCAL_WORKSPACE)
    except Exception:  # noqa: BLE001 - doctor must never crash on a store read
        return 0, 0
    return sum(1 for r in routines if r.enabled), len(routines)


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
