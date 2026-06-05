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
    return _apply_defaults(spec)


def _build_runtime_for(
    spec: AgentSpec, args: argparse.Namespace, *, on_event: Any = None
) -> Any:
    """Wire a runtime for ``spec`` honoring CLI provider/model overrides + tools."""
    from himmy import build_runtime
    from himmy.services.tools.registry import ToolRegistry

    provider = getattr(args, "provider", None) or spec.provider
    model = getattr(args, "model", None) or (
        spec.model if spec.model != "default" else None
    )
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

    if spec.tool_packs or spec.tools_module:
        registry = ToolRegistry()
        if spec.tool_packs:
            from himmy.toolkit import ToolkitConfig, register_packs

            tk_config = ToolkitConfig.from_sources(_project().get("toolkit"))
            register_packs(registry, spec.tool_packs, tk_config)
        if spec.tools_module:
            _resolve_register(spec.tools_module)(registry)
        if pipeline is not None:
            # Guard tool arguments too (the highest-risk "act" surface).
            from himmy.services.guardrails import build_guardrail_pre_hook
            from himmy.services.tools.service import ToolService

            overrides["tool_service"] = ToolService(
                registry, pre_execution_hook=build_guardrail_pre_hook(pipeline)
            )
        else:
            overrides["tool_registry"] = registry

    runtime, _inference, _tools = build_runtime(**overrides)
    return runtime


# --------------------------------------------------------------------- run/chat


def cmd_run(args: argparse.Namespace) -> int:
    """One-shot: run a single prompt through the agent and print the answer."""
    if not args.prompt:
        _eprint("error: --prompt/-p is required for `himmy run`")
        return 2
    spec = _spec_from_args(args)
    tracer = _TraceCollector() if getattr(args, "trace", False) else None
    runtime = _build_runtime_for(spec, args, on_event=tracer.handle if tracer else None)

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

        asyncio.run(_stream())
        _print_trace()
        return 0

    if getattr(args, "plan", False):
        from himmy.orchestrators import PlannerOrchestrator

        async def _plan() -> Any:
            return await PlannerOrchestrator(runtime).run(
                args.prompt, spec.to_persona(), tool_names=spec.tools or None
            )

        result = asyncio.run(_plan())
        _eprint(f"plan: {len(result.plan)} step(s)")
        for i, step in enumerate(result.plan, start=1):
            _eprint(f"  {i}. {step}")
        print(result.output_text or "")
        _print_trace()
        return 0

    async def _go() -> Any:
        return await runtime.run_task_detailed(
            spec.to_persona(),
            spec.make_task(args.prompt),
            llm_config=spec.to_llm_config(),
        )

    result = asyncio.run(_go())

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
    return 0 if result.succeeded else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL keeping one thread; `--message` runs a single turn."""
    spec = _spec_from_args(args)
    runtime = _build_runtime_for(spec, args)
    persona = spec.to_persona()
    llm_config = spec.to_llm_config()

    async def _turn(thread: Any, text: str) -> Any:
        return await runtime.run_task_detailed(
            persona, spec.make_task(text), thread=thread, llm_config=llm_config
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
        """Stream one reply to stdout token-by-token (appends to ``thread``)."""
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
        result = asyncio.run(_turn(thread, args.message))
        _persist(result.thread)
        print(result.output_text or "")
        return 0 if result.succeeded else 1

    _eprint(f"himmy chat — {persona.name} ({persona.role}). /exit, /reset, /help.")
    if session_id:
        _eprint(f"(session: {session_id})")
    thread = _new_thread()
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
        asyncio.run(_stream(thread, line))
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

    result = asyncio.run(orch.run(args.prompt))

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
        runtime = _build_runtime_for(spec, args)
        harness = AgentEvalHarness(runtime, eval_service)
        run = asyncio.run(
            harness.evaluate_agent(
                suite, spec.to_persona(), llm_config=spec.to_llm_config()
            )
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
# tool_packs: [web, utils]  # built-in tool packs (run `himmy tools` to list them)
tools: []                   # names to bind (from tool_packs and/or tools_module)
# tools_module: tools:register   # uncomment to wire the example tool in tools.py
# output_schema: schema.json     # path to a JSON Schema for structured output
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


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold an ``agent.yaml`` + ``tools.py`` (or a ``team.yaml`` with ``--team``)."""
    target = Path(args.directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    files = (
        {"team.yaml": _TEAM_YAML}
        if args.team
        else {
            "agent.yaml": _AGENT_YAML,
            "tools.py": _TOOLS_PY,
            "himmy.toml": _HIMMY_TOML,
        }
    )

    existing = [name for name in files if (target / name).exists()]
    if existing and not args.force:
        _eprint(
            f"error: {', '.join(existing)} already exist in {target} "
            "(use --force to overwrite)"
        )
        return 1

    for name, content in files.items():
        (target / name).write_text(content)
        print(f"wrote {target / name}")
    if args.team:
        print(f'\nNext: himmy team -f {target / "team.yaml"} -p "your question"')
    else:
        print(f'\nNext: himmy run -f {target / "agent.yaml"} -p "hello"')
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
    for binary in ("claude", "ollama"):
        found = shutil.which(binary)
        print(
            f"  [{'ok ' if found else '-- '}] {binary}"
            + (f" → {found}" if found else "")
        )

    print("\nprovider keys (env):")
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PYDANTIC_AI_GATEWAY_API_KEY",
    ):
        print(f"  [{'ok ' if os.environ.get(key) else '-- '}] {key}")

    from himmy.services.guardrails import BUILTIN_GUARDRAILS

    print(
        f"\nguardrails (agent.yaml `guardrails: [...]`): {', '.join(BUILTIN_GUARDRAILS)}"
    )

    from himmy.config.project import find_project_config

    cfg = find_project_config()
    print(f"\nproject config: {cfg if cfg else '(none — using env + defaults)'}")
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
