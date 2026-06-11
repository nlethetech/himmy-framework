"""The interactive ``himmy chat`` REPL, extracted into a drivable class.

:class:`ChatRepl` holds the spec/args/runtime state the REPL mutates (so ``/model``
can rebuild the backend mid-conversation, the thread survives) and owns the turn
loop. It adds Tier-1 agentic trust + control on top of the original REPL:

* **In-REPL approvals** — on a real terminal, turns run with ``hitl=True``; when a
  turn pauses at an approval-gated tool the REPL renders the pending call(s) and
  prompts ``approve? [y/N/always]`` before resuming. ``always`` also persists the
  tool to ``himmy.toml`` ``[permissions] auto_approve``.
* **Interrupt without losing the session** — Ctrl-C during a turn cancels only the
  in-flight task and returns to the prompt with the thread intact; Ctrl-C at the
  prompt still exits.
* **Plan mode** — ``/plan <task>`` asks the model for a numbered plan, shows it, and
  runs it on approval.

Non-TTY behavior is unchanged: no hitl, gated tools fail closed (``POLICY_BLOCKED``)
so scripts and CI never hang on an approval prompt.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from himmy.cli import permissions
from himmy.cli.ui import LiveRunUI, render_markdown_lite, styles
from himmy.config.agent_spec import AgentSpec

#: ~200-char cap on the faint args echoed beside a pending tool name.
_ARG_PREVIEW_CAP = 200


def _truncate(text: str, cap: int = _ARG_PREVIEW_CAP) -> str:
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


async def cancel_inflight_task(task: asyncio.Task[Any]) -> None:
    """Cancel an in-flight turn task and drain it without leaking a pending warning.

    The REPL turns a turn coroutine into a task so a Ctrl-C can cancel ONLY that
    task (not the whole REPL). After ``task.cancel()`` the task must still be
    awaited so its cancellation actually unwinds — otherwise asyncio logs
    ``Task was destroyed but it is pending``. ``gather(..., return_exceptions=True)``
    swallows the resulting :class:`asyncio.CancelledError` (and any error raised
    while unwinding) so the caller returns cleanly to the prompt.
    """
    if task.done():
        # Already finished (or already cancelled) — still retrieve any exception
        # so it isn't reported as never-retrieved.
        await asyncio.gather(task, return_exceptions=True)
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


class ChatRepl:
    """Drives one interactive chat session over a single (surviving) thread.

    Construct with the resolved :class:`AgentSpec` and the parsed CLI args, then
    call :meth:`run`. The runtime/registry/live trio lives in :attr:`rt` so
    ``/model`` can rebuild it in place. ``input_fn`` / ``loop`` are injectable for
    tests (default to :func:`input` and a fresh event loop).
    """

    def __init__(
        self,
        spec: AgentSpec,
        args: argparse.Namespace,
        *,
        input_fn: Callable[[str], str] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        tty: bool | None = None,
    ) -> None:
        self.spec = spec
        self.args = args
        self._input = input_fn or input
        self.rt: dict[str, Any] = {}
        self.persona = spec.to_persona()
        self.llm_config = spec.to_llm_config()
        self._owns_loop = loop is None
        self.loop = loop or asyncio.new_event_loop()
        import sys

        self.tty = tty if tty is not None else sys.stdout.isatty()
        # In-run approvals only when BOTH stdin and stderr are real TTYs: a script
        # or CI (no interactive stdin) must never block on an approval prompt.
        self.interactive = bool(sys.stdin.isatty() and sys.stderr.isatty())
        self.yolo = bool(getattr(args, "yolo", False))
        self.safe = bool(getattr(args, "safe", False))
        # Streamed final synthesis for tool turns is OPT-IN (/stream on): it costs
        # one extra model call per tool turn and records the synthesis exchange in
        # the thread, so it must be a deliberate choice, not a silent default.
        self.stream_final = False
        self._c = styles(sys.stdout)
        # Session continuity. An explicit --session <id> keeps the historical
        # behavior (persist + auto-load that id). Otherwise the REPL auto-persists
        # to the implicit "last" session after every turn — but a fresh REPL does
        # NOT auto-load it (a fresh `himmy` feels fresh) unless `-c`/`--continue`.
        explicit = getattr(args, "session", None)
        self._explicit_session = bool(explicit)
        self._session_id = str(explicit) if explicit else "last"
        self._continue_last = bool(getattr(args, "continue_last", False))

    # ------------------------------------------------------------------ wiring

    def rebuild(self) -> None:
        """(Re)build the runtime + registry + live UI for the current spec/args.

        Wires a durable checkpoint store so an approval-gated tool can pause/resume,
        loads the ``[permissions] auto_approve`` allowlist (unless ``--safe``), and —
        under ``--yolo`` — grants every gated tool. ``/model`` calls this to swap the
        backend mid-conversation.
        """
        from himmy.cli.commands import _build_runtime_for
        from himmy.runtime.checkpoint import SqliteCheckpointStore

        self.rt["live"] = LiveRunUI(model_label=self._model_label())
        store = SqliteCheckpointStore(_cli_checkpoint_db())
        runtime, registry = _build_runtime_for(
            self.spec,
            self.args,
            on_event=self.rt["live"].handle if self.rt["live"].enabled else None,
            checkpoint_store=store,
        )
        self.rt["runtime"] = runtime
        self.rt["registry"] = registry
        self.rt["checkpoint_store"] = store
        self._apply_permissions(registry)

    def _apply_permissions(self, registry: Any) -> None:
        """Resolve the permission profile onto ``registry`` (allowlist / yolo / safe)."""
        if registry is None:
            return
        if self.yolo:
            permissions.grant_all_approvals(registry)
            return
        if self.safe:
            return  # ignore the allowlist: everything gated prompts
        permissions.apply_allowlist(registry, permissions.load_auto_approve())

    def _model_label(self) -> str:
        from himmy.cli.commands import _model_label

        return _model_label(self.spec, self.args)

    # ------------------------------------------------------------------ a turn

    async def _answer_turn(self, thread: Any, text: str) -> Any:
        """Run one tool-using turn under hitl (interactive) and return the loop result."""
        return await self.rt["runtime"].run_agent_loop(
            self.persona,
            self.spec.make_task(text),
            thread,
            llm_config=self.llm_config,
            max_turns=8,
            route_tools=self.spec.tool_router,
            hitl=self.interactive,
        )

    async def _drive_turn_with_approvals(self, thread: Any, text: str) -> Any:
        """Run a turn, then service any approval pauses, looping until the run ends.

        Returns the terminal :class:`~himmy.runtime.single_agent.AgentLoopResult`.
        When a turn pauses (``awaiting_approval``) the live spinner is stopped, the
        pending tool call(s) rendered, and the human prompted; the run resumes with
        the decision and may pause again — so this loops until ``checkpoint_id`` is
        ``None`` (the run finished). Non-interactive sessions never reach the
        approval branch (hitl is off → gated tools fail closed).
        """
        loop_result = await self._answer_turn(thread, text)
        while getattr(loop_result, "checkpoint_id", None):
            self.rt["live"].finish()  # stop the spinner before prompting
            approved = self._prompt_for_checkpoint(loop_result.checkpoint_id)
            loop_result = await self.rt["runtime"].resume_agent_loop(
                loop_result.checkpoint_id,
                approved=approved,
                llm_config=self.llm_config,
            )
        return loop_result

    def _prompt_for_checkpoint(self, checkpoint_id: str) -> bool:
        """Render the pending tool call(s) and ask approve? [y/N/always].

        ``always`` approves now AND persists the tool name to
        ``himmy.toml`` ``[permissions] auto_approve`` so it won't prompt again.
        Returns the approve/deny decision for :meth:`resume_agent_loop`.
        """
        return prompt_approval(
            self.rt["checkpoint_store"],
            checkpoint_id,
            input_fn=self._input,
            on_line=self._eprint,
        )

    # ----------------------------------------------------------------- plan mode

    async def _plan_and_maybe_run(self, thread: Any, goal: str) -> Any:
        """``/plan <task>``: model-authored numbered plan → confirm → execute.

        The plan text comes from :class:`PlannerOrchestrator` (it plans on any
        provider, stub included) — never fabricated locally. On ``y`` the task runs
        with the approved plan injected into the prompt and the normal live event
        rendering + approval loop; on ``n`` the plan is discarded. Returns the
        (possibly new) thread.
        """
        from himmy.orchestrators import PlannerOrchestrator

        c = self._c
        planner = PlannerOrchestrator(self.rt["runtime"])
        plan = await planner.plan_only(
            goal,
            self.persona,
            model_key=str(self.llm_config.model_key if self.llm_config else "default"),
        )
        self.rt["live"].finish()
        if not plan:
            self._eprint(f"{c['dim']}(the model produced no plan){c['reset']}")
            return thread
        self._eprint(f"{c['snow']}plan:{c['reset']}")
        for i, step in enumerate(plan, start=1):
            self._eprint(f"  {c['snow']}{i}.{c['reset']} {c['ink']}{step}{c['reset']}")
        try:
            answer = self._input("run this plan? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            self._eprint(f"{c['dim']}— plan discarded{c['reset']}")
            return thread
        plan_block = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, start=1))
        prompt = f"{goal}\n\nFollow this approved plan, step by step:\n{plan_block}"
        loop_result = await self._drive_turn_with_approvals(thread, prompt)
        self.rt["live"].finish()
        self._print_reply(loop_result.final.output_text or "")
        self._persist(loop_result.thread)
        return loop_result.thread

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _eprint(*args: Any) -> None:
        import sys

        print(*args, file=sys.stderr)

    # ------------------------------------------------------ input affordances

    def _expand_at_files(self, text: str) -> str:
        """Inline ``@path`` tokens that name a real file (warning on a miss)."""
        from himmy.cli.input_affordances import expand_at_files

        c = self._c

        def _warn(msg: str) -> None:
            self._eprint(f"{c['dim']}{msg}{c['reset']}")

        return expand_at_files(text, on_warn=_warn)

    def _handle_bang(self, line: str) -> str | None:
        """Run a ``!cmd`` / ``!!cmd`` shell escape the user typed.

        Returns ``None`` for ``!cmd`` (output shown locally, agent NOT involved) and
        the framed command+output string for ``!!cmd`` (to be sent to the agent). A
        bare ``!`` with no command is a no-op (returns ``None``). This is a local
        shell escape the human typed verbatim — never agent-initiated — so it carries
        no approval gate.
        """
        from himmy.cli.input_affordances import (
            classify_bang,
            frame_for_agent,
            run_shell,
        )

        c = self._c
        parsed = classify_bang(line)
        if parsed is None:
            self._eprint(f"{c['dim']}usage: !<cmd>  or  !!<cmd>{c['reset']}")
            return None
        result = run_shell(parsed.command)
        body = result.output or "(no output)"
        self._eprint(f"{c['dim']}{body}{c['reset']}")
        if result.exit_code != 0 and not result.timed_out:
            self._eprint(f"{c['dim']}(exit {result.exit_code}){c['reset']}")
        if parsed.send_to_agent:
            return frame_for_agent(result)
        return None

    def _print_reply(self, text: str) -> None:
        import sys

        prefix = "" if self.tty else "bot> "
        sys.stdout.write(prefix + render_markdown_lite(text, stream=sys.stdout) + "\n")
        if self.tty:
            sys.stdout.write("\n")
        sys.stdout.flush()

    # --------------------------------------------------------------- a chat turn

    async def _chat_turn(self, thread: Any, text: str) -> Any:
        """One user turn: tool-using agents run the approval loop; others stream.

        Returns the (possibly new) thread. A tool-using agent runs the act→observe
        loop under hitl (interactive only) with the approval prompt loop; a no-tool
        agent streams token-by-token exactly as before. When a tool-using turn
        finishes WITHOUT a pending approval, the final answer is re-issued as a
        streamed synthesis turn (see :meth:`_stream_final_answer`) so the user sees
        it arrive token-by-token instead of all at once.
        """
        import sys

        if self.rt["registry"] is not None:
            tools_before = self._count_tool_messages(thread)
            loop_result = await self._drive_turn_with_approvals(thread, text)
            self.rt["live"].finish()
            # Stream the final answer only when this turn actually exercised tools
            # AND all approvals are resolved (no pending checkpoint). We detect tool
            # use by the TOOL messages the turn added to the thread — which also
            # catches a tool executed across an approval/resume (where the tool call
            # is in the pre-checkpoint segment, not in the final result's turns).
            used_tools = self._count_tool_messages(loop_result.thread) > tools_before
            if (
                self.stream_final
                and used_tools
                and getattr(loop_result, "checkpoint_id", None) is None
            ):
                await self._stream_final_answer(loop_result.thread)
                return loop_result.thread
            self._print_reply(loop_result.final.output_text or "")
            return loop_result.thread
        # No tools: stream the reply token-by-token (the original behavior).
        prefix = "" if self.tty else "bot> "
        sys.stdout.write(prefix)
        async for delta in self.rt["runtime"].stream_task(
            self.persona,
            self.spec.make_task(text),
            thread=thread,
            llm_config=self.llm_config,
        ):
            if delta.delta:
                sys.stdout.write(delta.delta)
                sys.stdout.flush()
        sys.stdout.write("\n\n" if self.tty else "\n")
        return thread

    @staticmethod
    def _count_tool_messages(thread: Any) -> int:
        """How many TOOL-role messages a thread holds (tool-use detector)."""
        from himmy.agents.base_agent.thread import MessageRole

        return sum(
            1 for m in getattr(thread, "messages", []) if m.role == MessageRole.TOOL
        )

    #: The synthesis prompt for the streamed final-answer turn (tools unbound).
    _STREAM_SYNTHESIS_PROMPT = (
        "Using only the tool results above, answer the user's question directly and "
        "completely. Do not call any tools."
    )

    async def _stream_final_answer(self, thread: Any) -> None:
        """Stream a final synthesis turn over a thread whose tool loop already ran.

        The act→observe loop buffers its whole answer ("can't stream mid-loop"). Once
        the loop has finished and every approval is resolved, the thread carries the
        full tool history, so we re-issue ONE final no-tool turn through
        :meth:`SingleAgentRuntime.stream_task` (tools unbound via ``tool_names: []``)
        and write its tokens live. This is an additional, real, streamed synthesis
        turn — it does not touch ``run_agent_loop`` — so the loop's bounding and the
        approval flow are unchanged; streaming only kicks in after all of that.
        """
        import sys

        from himmy.agents.base_agent.task import Task

        task = Task(
            title=f"{self.spec.name}-synthesis",
            prompt=self._STREAM_SYNTHESIS_PROMPT,
            context={"tool_names": []},  # unbind tools: force a streamed text answer
        )
        prefix = "" if self.tty else "bot> "
        sys.stdout.write(prefix)
        async for delta in self.rt["runtime"].stream_task(
            self.persona,
            task,
            thread=thread,
            llm_config=self.llm_config,
        ):
            if delta.delta:
                sys.stdout.write(delta.delta)
                sys.stdout.flush()
        sys.stdout.write("\n\n" if self.tty else "\n")

    # --------------------------------------------------------------- slash cmds

    def _slash(self, line: str, thread: Any) -> Any | None:
        """Handle a /command; returns the (possibly new) thread, or None to exit.

        ``/plan <task>`` is async (it calls the model), so it is dispatched onto the
        REPL's event loop here; the rest are synchronous.
        """
        from himmy.cli.commands import _model_label

        c = self._c
        cmd, *rest = line.split()
        from himmy.agents.base_agent.thread import ChatThread

        if cmd in ("/exit", "/quit"):
            return None
        if cmd == "/plan":
            goal = (
                line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            )
            if not goal:
                self._eprint(f"{c['dim']}usage: /plan <task>{c['reset']}")
                return thread
            return self._run_turn_coro(self._plan_and_maybe_run(thread, goal), thread)
        if cmd == "/resume":
            return self._resume_picker(thread)
        if cmd == "/reset":
            thread = ChatThread(agent_id=self.persona.agent_id)
            self._persist(thread)
            self._eprint(f"{c['dim']}(thread reset){c['reset']}")
        elif cmd == "/model":
            if rest:
                self.args.model = rest[0]
                if len(rest) > 1:
                    self.args.provider = rest[1]
                self.rebuild()
            self._eprint(
                f"{c['dim']}model: {_model_label(self.spec, self.args) or 'auto'}"
                f"{c['reset']}"
            )
        elif cmd == "/tools":
            reg = self.rt["registry"]
            names = sorted(d.name for d in reg.list()) if reg is not None else []
            self._eprint(
                f"{c['dim']}{', '.join(names) or '(no tools bound)'}{c['reset']}"
            )
        elif cmd == "/agents":
            from himmy.cli.agents import cmd_agents

            cmd_agents(argparse.Namespace(directory=".", json=False))
        elif cmd == "/stream":
            if rest and rest[0] in ("on", "off"):
                self.stream_final = rest[0] == "on"
            state = "on" if self.stream_final else "off"
            self._eprint(
                f"{c['dim']}streamed final answers: {state} — when on, tool turns "
                f"re-issue the final synthesis as a live stream (one extra model "
                f"call per turn){c['reset']}"
            )
        elif cmd == "/cost":
            live = self.rt["live"]
            self._eprint(
                f"{c['dim']}this session: {live.events} events · "
                f"{live.tool_calls} tool call(s) · ${live.cost:.4f}{c['reset']}"
            )
        elif cmd == "/help":
            self._eprint(
                f"{c['dim']}/model [name [provider]]  switch the model\n"
                f"/plan <task>              plan a task, then run it on approval\n"
                f"/resume                   pick a recent session to continue\n"
                f"/tools                    what this agent can call\n"
                f"/agents                   specs in this directory\n"
                f"/stream on|off            stream tool-turn answers (extra call)\n"
                f"/cost                     session events + spend\n"
                f"/reset                    start a fresh thread\n"
                f"/exit                     leave\n"
                f"\n"
                f"@path                     inline a file's contents into your message\n"
                f"!cmd                      run a shell command, show its output\n"
                f"!!cmd                     run a shell command, send it to the agent"
                f"{c['reset']}"
            )
        else:
            self._eprint(
                f"{c['dim']}unknown command {cmd} — /help lists them{c['reset']}"
            )
        return thread

    # ------------------------------------------------------------------ resume

    def _resume_picker(self, thread: Any) -> Any:
        """``/resume``: list recent sessions and swap the live thread to a chosen one.

        Renders a numbered picker (id · last-active · message count) from the session
        store, reads a choice, and on a valid pick swaps the live thread to that
        session AND repoints the active session id so subsequent turns persist there.
        An empty/invalid choice leaves the current thread untouched.
        """
        c = self._c
        store = self.rt.get("session_store")
        if store is None or not hasattr(store, "list_sessions"):
            self._eprint(f"{c['dim']}(no session store){c['reset']}")
            return thread
        sessions = store.list_sessions(limit=10)
        if not sessions:
            self._eprint(f"{c['dim']}(no saved sessions yet){c['reset']}")
            return thread
        self._eprint(f"{c['snow']}recent sessions:{c['reset']}")
        for i, info in enumerate(sessions, start=1):
            self._eprint(
                f"  {c['snow']}{i}.{c['reset']} {c['ink']}{info.session_id}{c['reset']}"
                f"{c['faint']}  {info.updated_at}  "
                f"{info.message_count} message(s){c['reset']}"
            )
        try:
            answer = self._input("resume which? [number, or blank to cancel] ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if not answer:
            self._eprint(f"{c['dim']}— cancelled{c['reset']}")
            return thread
        try:
            choice = int(answer)
        except ValueError:
            self._eprint(f"{c['dim']}— not a number, cancelled{c['reset']}")
            return thread
        if not 1 <= choice <= len(sessions):
            self._eprint(f"{c['dim']}— out of range, cancelled{c['reset']}")
            return thread
        picked = sessions[choice - 1]
        loaded = store.load(picked.session_id)
        if loaded is None:
            self._eprint(f"{c['dim']}— could not load that session{c['reset']}")
            return thread
        self._session_id = picked.session_id
        self._explicit_session = True
        count = len(getattr(loaded, "messages", []))
        self._eprint(
            f"{c['dim']}continuing session '{picked.session_id}' "
            f"({count} messages){c['reset']}"
        )
        return loaded

    # ----------------------------------------------------------- loop + persist

    def _run_turn_coro(self, coro: Any, thread: Any) -> Any:
        """Run a turn coroutine, making Ctrl-C cancel ONLY the in-flight task.

        The coroutine is scheduled as a task; on KeyboardInterrupt the task is
        cancelled and drained (no leaked-pending warning), the live spinner stopped,
        and a dim ``— interrupted`` line printed before returning to the prompt with
        the thread intact. On normal completion the (possibly new) thread is returned.
        """
        c = self._c
        task = self.loop.create_task(coro)
        try:
            return self.loop.run_until_complete(task)
        except KeyboardInterrupt:
            self.loop.run_until_complete(cancel_inflight_task(task))
            live = self.rt.get("live")
            if live is not None:
                live.finish()
            self._eprint(f"{c['dim']}— interrupted{c['reset']}")
            return thread

    def _new_thread(self) -> Any:
        """The thread the REPL opens with.

        Loads the active session's saved thread ONLY when the user asked to
        continue it — an explicit ``--session <id>`` (resume that id) or
        ``-c``/``--continue`` (continue the implicit ``last`` session). A bare
        ``himmy chat`` starts fresh even though it auto-persists to ``last``.
        """
        from himmy.agents.base_agent.thread import ChatThread

        store = self.rt.get("session_store")
        should_load = self._explicit_session or self._continue_last
        if store is not None and should_load:
            existing = store.load(self._session_id)
            if existing is not None:
                count = len(getattr(existing, "messages", []))
                self._eprint(
                    f"{self._c['dim']}continuing session '{self._session_id}' "
                    f"({count} messages){self._c['reset']}"
                )
                return existing
        return ChatThread(agent_id=self.persona.agent_id)

    def _persist(self, thread: Any) -> None:
        """Auto-persist ``thread`` to the active session id (``last`` by default)."""
        store = self.rt.get("session_store")
        if store is not None:
            store.save(self._session_id, thread)

    def run(self) -> int:
        """Run the interactive REPL until the user exits. Returns a process code."""

        self.rebuild()
        # Durable session continuity. Always wire the store: a bare REPL still
        # auto-persists to the implicit "last" session after every turn (only an
        # explicit --session or -c/--continue LOADS a prior thread at startup).
        from himmy.runtime.session import SqliteSessionStore

        Path(".himmy").mkdir(exist_ok=True)
        self.rt["session_store"] = SqliteSessionStore(
            str(Path(".himmy") / "sessions.db")
        )

        # Connect MCP servers ONCE on the persistent loop (reused across turns).
        mcp_clients: list[Any] = []
        if self.spec.mcp_servers:
            from himmy.config.mcp_spec import attach_mcp_servers

            mcp_clients = self.loop.run_until_complete(
                attach_mcp_servers(self.rt["registry"], list(self.spec.mcp_servers))
            )

        try:  # arrow-key history + line editing, when the platform has it
            import readline  # noqa: F401
        except ImportError:
            pass

        c = self._c
        prompt = f"{c['crimson']}›{c['reset']} " if self.tty else "you> "
        if self.yolo:
            self._eprint(
                f"{c['crimson']}⚠ --yolo: every tool runs without approval "
                f"this session{c['reset']}"
            )
        self._eprint(f"{c['dim']}{self._status()} — /help for commands{c['reset']}")
        if self._explicit_session:
            self._eprint(f"{c['faint']}session: {self._session_id}{c['reset']}")
        self._eprint("")

        thread = self._new_thread()
        try:
            while True:
                try:
                    line = self._input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    self._eprint("")
                    break
                if not line:
                    continue
                if line.startswith("/"):
                    handled = self._slash(line, thread)
                    if handled is None:
                        break
                    thread = handled
                    continue
                if line.startswith("!"):
                    sent = self._handle_bang(line)
                    if sent is None:
                        continue  # `!cmd`: shown locally, agent not involved
                    line = sent  # `!!cmd`: framed command+output goes to the agent
                else:
                    line = self._expand_at_files(line)
                thread = self._run_turn_coro(self._chat_turn(thread, line), thread)
                self._persist(thread)
        finally:
            if mcp_clients:
                from himmy.config.mcp_spec import close_mcp_clients

                self.loop.run_until_complete(close_mcp_clients(mcp_clients))
            if self._owns_loop:
                self.loop.close()
        return 0

    def _status(self) -> str:
        label = self._model_label() or "auto"
        packs = ", ".join(self.spec.tool_packs) or "no tools"
        return f"{self.persona.name} · {label} · {packs}"


def _render_args(args: dict[str, Any]) -> str:
    """Compact JSON of tool args for the faint approval preview."""
    import json

    try:
        return json.dumps(args or {}, default=str)
    except Exception:  # noqa: BLE001 - preview must never raise
        return str(args)


def prompt_approval(
    checkpoint_store: Any,
    checkpoint_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    on_line: Callable[[str], None] | None = None,
) -> bool:
    """Render a checkpoint's pending tool call(s) and ask approve? [y/N/always].

    Shared by the chat REPL and one-shot ``himmy run``. ``on_line`` receives each
    rendered (already-styled) diagnostic line — defaults to printing to stderr.
    ``always`` approves AND persists every pending tool name to ``himmy.toml``
    ``[permissions] auto_approve``. Returns the approve/deny decision.
    """
    import sys

    def _emit(text: str) -> None:
        if on_line is not None:
            on_line(text)
        else:
            print(text, file=sys.stderr)

    c = styles(sys.stderr)
    checkpoint = checkpoint_store.load(checkpoint_id)
    pending = list(checkpoint.pending_tool_calls) if checkpoint else []
    for call in pending:
        args_text = _truncate(_render_args(call.args))
        _emit(
            f"  {c['gold']}⏸ approval required{c['reset']} "
            f"{c['gold']}{call.tool_name}{c['reset']}"
            f"{c['faint']} {args_text}{c['reset']}"
        )
    try:
        answer = input_fn("approve? [y/N/always] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _emit(f"{c['dim']}— denied{c['reset']}")
        return False
    if answer in ("a", "always"):
        for call in pending:
            if permissions.persist_auto_approve(call.tool_name):
                _emit(
                    f"{c['dim']}(added {call.tool_name} to "
                    f"himmy.toml auto_approve){c['reset']}"
                )
        return True
    return answer in ("y", "yes")


def _cli_checkpoint_db() -> str:
    """Path to the CLI's durable approval checkpoint store (``.himmy/approvals.db``).

    Mirrors Himmy Studio's store location so the CLI and Studio share the same
    durable approvals DB under the project root.
    """
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "approvals.db")


__all__ = ["ChatRepl", "cancel_inflight_task", "prompt_approval"]
