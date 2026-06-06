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
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from himmy.cli.provider import build_inference_for
from himmy.config.agent_spec import AgentSpec, load_agent_spec


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


def _resolve_register(dotted: str) -> Callable[[Any], Any]:
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


_PROJECT_CACHE: dict[str, Any] | None = None


def _project() -> dict[str, Any]:
    """Load (and cache) the ``himmy.toml`` project config for this invocation."""
    global _PROJECT_CACHE
    if _PROJECT_CACHE is None:
        from himmy.config.project import load_project_config

        _PROJECT_CACHE = load_project_config()
    return _PROJECT_CACHE


def _apply_defaults(spec: AgentSpec) -> AgentSpec:
    """Fill unset spec fields from ``himmy.toml`` ``[defaults]`` (spec/flag still win)."""
    defaults = _project().get("defaults", {})
    if spec.provider is None and defaults.get("provider"):
        spec.provider = defaults["provider"]
    if spec.model == "default" and defaults.get("model"):
        spec.model = defaults["model"]
    if not spec.tool_packs and defaults.get("tool_packs"):
        spec.tool_packs = list(defaults["tool_packs"])
    if not spec.guardrails and defaults.get("guardrails"):
        spec.guardrails = list(defaults["guardrails"])
    return spec


def _spec_from_args(args: argparse.Namespace) -> AgentSpec:
    """Build the AgentSpec from ``-f file`` or ad-hoc ``--name``/``--instruction``."""
    if getattr(args, "file", None):
        spec = load_agent_spec(args.file)
    else:
        spec = AgentSpec(
            name=getattr(args, "name", None) or "himmy-agent",
            description="Ad-hoc agent created from CLI flags.",
            instructions=list(getattr(args, "instruction", None) or []),
        )
    spec = _apply_defaults(spec)
    if spec.skills:
        # Expand declared skills into tools + injected know-how before the runtime is
        # wired, so skill-contributed packs/guardrails flow through the normal path.
        # Project-local skills (./skills/*.yaml, HIMMY_SKILLS_PATH) overlay the builtins.
        from himmy.config.agent_spec import apply_skills
        from himmy.skills import build_skill_registry

        spec = apply_skills(spec, build_skill_registry())
    return spec


def _exec_with_mcp(factory: Any, registry: Any, mcp_servers: Any) -> Any:
    """Run ``factory()`` in one event loop, with MCP servers attached for its duration.

    MCP clients bind their reader task to the running loop, so connect + run + close
    must share a single ``asyncio.run``. With no MCP servers this is a plain run.
    """
    if not mcp_servers:
        return asyncio.run(factory())

    from himmy.config.mcp_spec import attach_mcp_servers, close_mcp_clients

    async def _outer() -> Any:
        clients = await attach_mcp_servers(registry, list(mcp_servers))
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
    route_tools: bool = False,
) -> Any:
    """Produce a final answer, driving the tool loop when the agent has tools.

    ``run_task_detailed`` is a single inference call — correct for a no-tool agent,
    but for a tool-using one the model's first turn is the tool *call*; the answer
    only comes after the runtime feeds the tool result back. So when tools are wired
    we use ``run_agent_loop`` (act → observe → answer) and return its final turn.
    """
    if has_tools:
        loop = await runtime.run_agent_loop(
            persona,
            task,
            thread,
            llm_config=llm_config,
            max_turns=max_turns,
            route_tools=route_tools,
        )
        return loop.final
    return await runtime.run_task_detailed(persona, task, thread, llm_config=llm_config)


_KNOWLEDGE_EXTS = {".md", ".markdown", ".txt", ".rst", ".csv"}


async def _ingest_knowledge(registry: Any, sources: list[str]) -> int:
    """Ingest each declared file/dir of text docs into the KB; returns the count."""
    ingest = registry.handler_for("kb_ingest")
    count = 0
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(
                f for f in p.rglob("*") if f.suffix.lower() in _KNOWLEDGE_EXTS
            )
        else:
            _eprint(f"warning: knowledge source not found: {src}")
            continue
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            await ingest({"text": text, "title": f.stem, "source_uri": f.name})
            count += 1
    return count


def _build_runtime_for(
    spec: AgentSpec,
    args: argparse.Namespace,
    *,
    on_event: Any = None,
    inference: Any = None,
) -> Any:
    """Wire a runtime for ``spec`` honoring CLI provider/model overrides + tools.

    Returns ``(runtime, registry)``. The registry is non-``None`` whenever the agent
    has tools/packs/MCP servers; MCP tools are registered into it later, inside the
    event loop (see :func:`_exec_with_mcp`), since connecting is async. ``inference``
    overrides the provider-derived service (used by record/replay).
    """
    from himmy import build_runtime
    from himmy.services.tools.registry import ToolRegistry

    provider = getattr(args, "provider", None) or spec.provider
    model = getattr(args, "model", None) or (
        spec.model if spec.model != "default" else None
    )
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

        tk = ToolkitConfig.from_sources(_project().get("toolkit"))
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

            tk_config = ToolkitConfig.from_sources(_project().get("toolkit"))
            register_packs(registry, spec.tool_packs, tk_config)
        if spec.knowledge:
            # Auto-ingest the declared docs into a local knowledge base and give the
            # agent kb_search — a grounded doc agent with no driver code.
            from himmy.toolkit import ToolkitConfig, register_packs

            if "knowledge" not in spec.tool_packs:
                register_packs(
                    registry,
                    ["knowledge"],
                    ToolkitConfig.from_sources(_project().get("toolkit")),
                )
            n = asyncio.run(_ingest_knowledge(registry, spec.knowledge))
            _eprint(f"ingested {n} document(s) into the knowledge base")
        if spec.http_tools:
            from himmy.config.http_tool_spec import register_http_tools

            register_http_tools(registry, spec.http_tools)
        if spec.tools_module:
            _resolve_register(spec.tools_module)(registry)
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
    """One-shot: run a single prompt through the agent and print the answer."""
    if not args.prompt:
        _eprint("error: --prompt/-p is required for `himmy run`")
        return 2
    if getattr(args, "record", None) and getattr(args, "replay", None):
        _eprint("error: --record and --replay are mutually exclusive")
        return 2
    spec = _spec_from_args(args)
    tracer = _TraceCollector() if getattr(args, "trace", False) else None
    inference, recorder = _record_replay_inference(spec, args)
    if inference is None:  # record/replay supplies its own (non-stub) manager
        _maybe_hint_stub(spec, args)
    runtime, registry = _build_runtime_for(
        spec, args, on_event=tracer.handle if tracer else None, inference=inference
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

        _exec_with_mcp(_stream, registry, spec.mcp_servers)
        _print_trace()
        return 0

    if getattr(args, "plan", False):
        from himmy.orchestrators import PlannerOrchestrator

        async def _plan() -> Any:
            return await PlannerOrchestrator(runtime).run(
                args.prompt, spec.to_persona(), tool_names=spec.tools or None
            )

        result = _exec_with_mcp(_plan, registry, spec.mcp_servers)
        _eprint(f"plan: {len(result.plan)} step(s)")
        for i, step in enumerate(result.plan, start=1):
            _eprint(f"  {i}. {step}")
        print(result.output_text or "")
        _print_trace()
        return 0

    async def _go() -> Any:
        return await _answer(
            runtime,
            spec.to_persona(),
            spec.make_task(args.prompt),
            llm_config=spec.to_llm_config(),
            has_tools=registry is not None,
            route_tools=spec.tool_router,
        )

    result = _exec_with_mcp(_go, registry, spec.mcp_servers)

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
        print(result.output_text or "")
    _print_trace()
    if recorder is not None:
        path = recorder.dump(args.record)
        _eprint(f"recorded {len(recorder.cassette.entries)} model exchange(s) → {path}")
    elif getattr(args, "replay", None):
        _eprint(f"replayed from {args.replay}")
    return 0 if result.succeeded else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL keeping one thread; `--message` runs a single turn."""
    spec = _spec_from_args(args)
    _maybe_hint_stub(spec, args)
    runtime, registry = _build_runtime_for(spec, args)
    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    has_tools = registry is not None

    async def _turn(thread: Any, text: str) -> Any:
        return await _answer(
            runtime,
            persona,
            spec.make_task(text),
            thread=thread,
            llm_config=llm_config,
            has_tools=has_tools,
            route_tools=spec.tool_router,
        )

    from himmy.agents.base_agent.thread import ChatThread

    # Optional durable session: load the thread + persist after every turn.
    session_id = getattr(args, "session", None)
    store = None
    if session_id:
        from himmy.runtime.session import SqliteSessionStore

        Path(".himmy").mkdir(exist_ok=True)
        store = SqliteSessionStore(str(Path(".himmy") / "sessions.db"))

    def _new_thread() -> Any:
        if store is not None:
            existing = store.load(str(session_id))
            if existing is not None:
                return existing
        return ChatThread(agent_id=persona.agent_id)

    def _persist(thread: Any) -> None:
        if store is not None:
            store.save(str(session_id), thread)

    async def _stream(thread: Any, text: str) -> None:
        """Reply to stdout (appends to ``thread``).

        Streams token-by-token for a no-tool agent; a tool-using agent runs the full
        act→observe→answer loop (which can't stream mid-loop) and prints the answer.
        """
        if has_tools:
            result = await _turn(thread, text)
            sys.stdout.write((result.output_text or "") + "\n")
            _persist(result.thread)
            return
        async for delta in runtime.stream_task(
            persona, spec.make_task(text), thread=thread, llm_config=llm_config
        ):
            if delta.delta:
                sys.stdout.write(delta.delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        _persist(thread)

    if args.message:
        thread = _new_thread()
        result = _exec_with_mcp(
            lambda: _turn(thread, args.message), registry, spec.mcp_servers
        )
        _persist(result.thread)
        print(result.output_text or "")
        return 0 if result.succeeded else 1

    # Interactive REPL. When the agent has MCP servers, connect them ONCE on a
    # persistent loop and reuse it across turns (re-spawning per turn would be slow
    # and the clients' reader tasks are loop-bound).
    loop = asyncio.new_event_loop()
    mcp_clients: list[Any] = []
    if spec.mcp_servers:
        from himmy.config.mcp_spec import attach_mcp_servers

        mcp_clients = loop.run_until_complete(
            attach_mcp_servers(registry, list(spec.mcp_servers))
        )

    _eprint(f"himmy chat — {persona.name} ({persona.role}). /exit, /reset, /help.")
    if session_id:
        _eprint(f"(session: {session_id})")
    thread = _new_thread()
    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                _eprint("")
                break
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/reset":
                thread = ChatThread(agent_id=persona.agent_id)
                _persist(thread)
                _eprint("(thread reset)")
                continue
            if line == "/help":
                _eprint("commands: /exit  /reset  /help")
                continue
            sys.stdout.write("bot> ")
            loop.run_until_complete(_stream(thread, line))
    finally:
        if mcp_clients:
            from himmy.config.mcp_spec import close_mcp_clients

            loop.run_until_complete(close_mcp_clients(mcp_clients))
        loop.close()
    return 0


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

    bot = TelegramBot(TelegramClient(token, timeout=cfg.http_timeout + 30), _handle)
    _eprint(f"himmy telegram — {persona.name} is live. Ctrl-C to stop.")

    async def _serve() -> None:
        try:
            await bot.run()
        except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover
            pass

    try:
        # MCP servers (if any) stay connected for the whole session.
        _exec_with_mcp(_serve, registry, spec.mcp_servers)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _eprint("\n(stopped)")
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
# provider: claude-cli      # stub | claude-cli | ollama | pydantic-ai (default: auto)
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
# provider = "ollama"          # stub | claude-cli | ollama | pydantic-ai
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
        default_suite,
        render_markdown,
        to_json,
    )

    suite = (
        BenchmarkSuite.from_yaml(args.suite)
        if getattr(args, "suite", None)
        else default_suite()
    )
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
    print(render_markdown(cards, suite_name=suite.name))
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
    """Scaffold an agent: a default one, a ``--template`` starter, or a ``--team``."""
    target = Path(args.directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    template = getattr(args, "template", None)
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

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
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
    """Report installed optional extras, local providers, and provider keys."""
    print(f"himmy doctor — Python {sys.version.split()[0]}")

    print("\noptional extras:")
    extras = {
        "providers (pydantic-ai)": "pydantic_ai",
        "api (fastapi)": "fastapi",
        "postgres (asyncpg)": "asyncpg",
        "connectors (feedparser)": "feedparser",
        "connectors (openpyxl)": "openpyxl",
        "observability (logfire)": "logfire",
        "nepal (nepali-datetime)": "nepali_datetime",
        "validation (jsonschema)": "jsonschema",
    }
    for label, module in extras.items():
        mark = "ok " if _can_import(module) else "-- "
        print(f"  [{mark}] {label}")

    print("\nlocal providers (PATH):")
    have_claude = bool(shutil.which("claude"))
    have_ollama = bool(shutil.which("ollama"))
    for binary, found in (("claude", have_claude), ("ollama", have_ollama)):
        path = shutil.which(binary)
        print(
            f"  [{'ok ' if found else '-- '}] {binary}"
            + (f" → {path}" if path else "")
        )

    print("\nprovider keys (env):")
    have_key = False
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PYDANTIC_AI_GATEWAY_API_KEY",
    ):
        present = bool(os.environ.get(key))
        have_key = have_key or present
        print(f"  [{'ok ' if present else '-- '}] {key}")

    from himmy.services.guardrails import BUILTIN_GUARDRAILS

    print(
        f"\nguardrails (agent.yaml `guardrails: [...]`): {', '.join(BUILTIN_GUARDRAILS)}"
    )

    from himmy.config.project import find_project_config

    cfg = find_project_config()
    print(f"\nproject config: {cfg if cfg else '(none — using env + defaults)'}")

    # End on the single most useful next action, not just a status table.
    has_real_model = have_claude or have_ollama or have_key
    has_agent = any(Path(p).exists() for p in ("agent.yaml", "team.yaml"))
    print("\nnext step:")
    if not has_real_model:
        print(
            "  → no real model yet. Install one (free, local):  ollama pull llama3.2\n"
            "    or set OPENAI_API_KEY / ANTHROPIC_API_KEY for a cloud model."
        )
    elif not has_agent:
        print("  → scaffold your first agent:  himmy init my-agent")
    else:
        provider_hint = (
            " --provider ollama" if (have_ollama and not have_key) else ""
        )
        print(
            f'  → run it:  himmy run -f agent.yaml -p "Say hello."{provider_hint}'
        )
    return 0


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
