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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from himmy.cli import permissions
from himmy.cli.ui import (
    LiveRunUI,
    TurnRecorder,
    compose_event_handlers,
    render_markdown_lite,
    styles,
)
from himmy.config.agent_spec import AgentSpec

#: ~200-char cap on the faint args echoed beside a pending tool name.
_ARG_PREVIEW_CAP = 200


def _truncate(text: str, cap: int = _ARG_PREVIEW_CAP) -> str:
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _kfmt(tokens: int) -> str:
    """Compact token count: ``980`` → ``0.9k``, ``3210`` → ``3.2k``."""
    return f"{tokens / 1000:.1f}k"


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


@dataclass
class _Job:
    """One backgrounded ``/spawn`` sub-agent turn and its (eventual) result."""

    job_id: str
    task: str
    status: str = "running"  # running | done | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: str = ""
    notified: bool = False  # has the "✓ done" line been shown yet?


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
        # Forensics + off-TTY cost accounting: a TurnRecorder keeps the last turn's
        # raw events (for /why) AND a cumulative session cost (for budgets) even when
        # LiveRunUI is silent (pipes/CI). Wired in rebuild() alongside LiveRunUI.
        self.recorder = TurnRecorder()
        # Budget guardrail: --budget flag wins over himmy.toml [limits] session_budget.
        flag_budget = getattr(args, "budget", None)
        self.budget: float | None = (
            float(flag_budget)
            if flag_budget is not None
            else permissions.load_session_budget()
        )
        self._budget_warned = False  # the one-time 80% warning fired?
        # Background /spawn jobs, oldest first; guarded for the daemon threads.
        self.jobs: list[_Job] = []
        self._jobs_lock = threading.Lock()
        self._job_seq = 0

    # ------------------------------------------------------------------ wiring

    def rebuild(self) -> None:
        """(Re)build the runtime + registry + live UI for the current spec/args.

        Wires a durable checkpoint store so an approval-gated tool can pause/resume,
        loads the ``[permissions] auto_approve`` allowlist (unless ``--safe``), and —
        under ``--yolo`` — grants every gated tool. ``/model`` calls this to swap the
        backend mid-conversation.
        """
        from himmy.cli.commands import _apply_cli_overrides, _build_runtime_for
        from himmy.runtime.checkpoint import SqliteCheckpointStore

        # /model mutates args; fold the override into the spec AND re-derive the
        # llm_config so the per-request model_key matches the rebuilt manager
        # (a stale model_key sends the old model string to the new provider).
        self.spec = _apply_cli_overrides(self.spec, self.args)
        self.llm_config = self.spec.to_llm_config()
        self.rt["live"] = LiveRunUI(model_label=self._model_label())
        store = SqliteCheckpointStore(_cli_checkpoint_db())
        # Always record events (TurnRecorder) for /why + budget accounting; the live
        # overlay is only added when it has a TTY to draw on.
        runtime, registry = _build_runtime_for(
            self.spec,
            self.args,
            on_event=compose_event_handlers(
                self.recorder.handle,
                self.rt["live"].handle if self.rt["live"].enabled else None,
            ),
            checkpoint_store=store,
        )
        self.rt["runtime"] = runtime
        self.rt["registry"] = registry
        self.rt["checkpoint_store"] = store
        self._apply_permissions(registry)

    def _apply_permissions(self, registry: Any) -> None:
        """Resolve the permission profile onto ``registry`` (allowlist / yolo / safe).

        After the profile, a restricted RBAC role re-gates any gated tool it may not
        run (fail closed); the default ``admin`` role is a no-op so existing sessions
        are unaffected.
        """
        if registry is None:
            return
        from himmy.cli.rbac_cmd import born_gated_names, enforce_role_on_registry

        # Snapshot born-gated tools BEFORE the profile ungates any, so RBAC can tell
        # a born-gated tool (tool_gated:execute) from a plain util/read (tool:execute).
        born_gated = born_gated_names(registry)
        if self.yolo:
            permissions.grant_all_approvals(registry)
        elif self.safe:
            pass  # ignore the allowlist: everything gated prompts
        else:
            permissions.apply_allowlist(registry, permissions.load_auto_approve())

        enforce_role_on_registry(
            registry,
            role=getattr(self.args, "role", None),
            on_line=self._eprint,
            born_gated=born_gated,
        )

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
            role=getattr(self.args, "role", None),
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
        self.recorder.start_turn()
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

        self.recorder.start_turn()
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

    # ------------------------------------------------------------------ /why

    def _why(self) -> None:
        """``/why``: full forensic detail of the LAST turn's recorded events.

        Renders :func:`format_timeline` over the recorder's last-turn events, then —
        per tool/inference event — the UNtruncated args/results and per-inference
        latency/cost/tokens. When a persistent ``.himmy/trace.db`` exists, points the
        user at it for cross-turn digging. Nothing recorded yet → a dim placeholder.
        """
        from himmy.services.observability.trace import format_timeline

        c = self._c
        events = list(self.recorder.events)
        if not events:
            self._eprint(f"{c['dim']}nothing to explain yet{c['reset']}")
            return
        self._eprint(f"{c['snow']}why — last turn ({len(events)} events):{c['reset']}")
        for raw_line in format_timeline(events).splitlines():
            self._eprint(f"{c['faint']}{raw_line}{c['reset']}")
        for ev in sorted(events, key=lambda e: e.timestamp):
            kind = getattr(ev.event_type, "value", str(ev.event_type))
            p = ev.payload or {}
            if kind in ("TOOL_CALLED", "TOOL_COMPLETED", "TOOL_FAILED"):
                name = p.get("tool_name", "")
                self._eprint(
                    f"  {c['gold']}{kind} {name}{c['reset']}"
                    + (f"{c['crimson']} {ev.error}{c['reset']}" if ev.error else "")
                )
                if "tool_args" in p:
                    for ln in _render_forensic_block(p.get("tool_args")):
                        self._eprint(f"    {c['faint']}args: {ln}{c['reset']}")
                for key in ("tool_result", "result", "output"):
                    if key in p:
                        block = _render_forensic_block(p.get(key))
                        for ln in block:
                            self._eprint(f"    {c['faint']}{key}: {ln}{c['reset']}")
            elif kind in ("INFERENCE_SUCCEEDED", "INFERENCE_FAILED"):
                ms = f"{ev.latency_ms:.0f}ms" if ev.latency_ms else "?ms"
                cost = f"${ev.cost:.4f}" if ev.cost else "$0"
                toks = ""
                it, ot = p.get("input_tokens"), p.get("output_tokens")
                if it is not None or ot is not None:
                    toks = f" · {it or 0}→{ot or 0} tok"
                self._eprint(
                    f"  {c['faint']}inference: {ms} · {cost}{toks}{c['reset']}"
                    + (f"{c['crimson']} {ev.error}{c['reset']}" if ev.error else "")
                )
        if Path(".himmy/trace.db").is_file():
            self._eprint(
                f"{c['faint']}(deeper history: .himmy/trace.db — "
                f"`himmy trace`){c['reset']}"
            )

    # ------------------------------------------------------------ /good · /bad

    def _feedback(self, verdict: str, note: str | None) -> None:
        """``/good`` | ``/bad`` [note]: rate the LAST turn (human feedback → spine).

        The cheapest, least-noisy learning signal there is. Records against the
        last turn's run id (``trace_id``) on the durable entity spine
        (``.himmy/entities.db``, shared with Studio) — no Studio run row needed.
        Re-rating appends a new version, so the audit trail keeps every change.
        """
        from himmy.api.studio_feedback import record_feedback

        c = self._c
        last = next(
            (e for e in reversed(self.recorder.events) if getattr(e, "trace_id", None)),
            None,
        )
        if last is None:
            self._eprint(
                f"{c['dim']}nothing to rate yet — ask something first{c['reset']}"
            )
            return
        extra = {"thread_id": last.thread_id} if last.thread_id else None
        try:
            fb = record_feedback(
                last.trace_id,
                verdict=verdict,
                note=note,
                source="cli",
                require_studio_run=False,
                extra_metadata=extra,
            )
        except Exception as exc:  # never let feedback crash the REPL
            self._eprint(f"{c['crimson']}couldn't record feedback: {exc}{c['reset']}")
            return
        assert fb is not None  # require_studio_run=False never returns None
        label = "👍 good" if verdict == "up" else "👎 bad"
        suffix = f" · “{note}”" if note else ""
        self._eprint(
            f"{c['snow']}{label} recorded{c['reset']}"
            f"{c['faint']}{suffix} (v{fb.version}){c['reset']}"
        )

    # --------------------------------------------------------------- /lineage

    def _lineage(self, thread: Any, *, as_dot: bool = False) -> None:
        """/lineage: provenance tree of the current thread's run (--dot for Graphviz)."""
        import sys

        from himmy.cli.lineage_cmd import render_run_lineage
        from himmy.services.observability.trace import SqliteEventStore

        c = self._c
        db = Path(".himmy/trace.db")
        if not db.is_file():
            self._eprint(
                f"{c['dim']}no persisted trace yet (run with --trace, or this "
                f"session hasn't written one){c['reset']}"
            )
            return
        store = SqliteEventStore(str(db))
        try:
            events = store.list_events(thread_id=thread.thread_id)
        finally:
            store.close()
        if not events:
            self._eprint(f"{c['dim']}no lineage for this turn yet{c['reset']}")
            return
        self._eprint(
            render_run_lineage(
                events, thread.thread_id, as_dot=as_dot, c=styles(sys.stderr)
            )
        )

    # --------------------------------------------------------------- /compact

    def _compact(self, thread: Any) -> None:
        """``/compact``: run the runtime's summarize-compaction against this thread NOW.

        Invokes the SAME machinery the runtime uses mid-loop
        (:meth:`SingleAgentRuntime._maybe_compact`) on-demand, with a one-shot
        ``compaction_spec`` ctx — so the live thread is summarized in place and the
        real ``CONTEXT_COMPACTED`` event fires. Prints measured before/after counts
        (never fabricated). A no-op (honest line) when the thread is already small.
        """
        from himmy.runtime.compaction import estimate_tokens

        c = self._c
        runtime = self.rt.get("runtime")
        if runtime is None:
            self._eprint(f"{c['dim']}(no runtime to compact with){c['reset']}")
            return
        msgs = getattr(thread, "messages", [])
        before_msgs = len(msgs)
        before_tokens = sum(estimate_tokens(m.content) for m in msgs)
        # Force a low threshold so an on-demand /compact actually compacts (the user
        # asked for it explicitly), keeping the spec's keep_recent.
        ctx = {
            "compaction_spec": {
                "max_tokens": min(before_tokens, self.spec.compact_after_tokens),
                "keep_recent": self.spec.compact_keep_recent,
            },
            "model_key": str(
                self.llm_config.model_key if self.llm_config else "default"
            ),
        }

        async def _run() -> None:
            await runtime._maybe_compact(
                self.persona, thread, ctx, "compact-on-demand", self.llm_config
            )

        try:
            self.loop.run_until_complete(_run())
        except Exception as exc:  # noqa: BLE001 - on-demand compaction is best-effort
            self._eprint(f"{c['dim']}(compaction failed: {exc}){c['reset']}")
            return
        after_msgs = len(getattr(thread, "messages", []))
        after_tokens = sum(
            estimate_tokens(m.content) for m in getattr(thread, "messages", [])
        )
        if after_msgs >= before_msgs:
            self._eprint(
                f"{c['dim']}nothing to compact "
                f"(~{before_tokens // 1000}.{(before_tokens % 1000) // 100}k tokens, "
                f"{before_msgs} messages){c['reset']}"
            )
            return
        self._persist(thread)
        self._eprint(
            f"{c['gold']}compacted: {before_msgs} → {after_msgs} messages "
            f"(~{_kfmt(before_tokens)} → ~{_kfmt(after_tokens)} tokens){c['reset']}"
        )

    def _context_meter(self, thread: Any) -> str:
        """A faint context gauge string: ``~3.2k tokens`` (vs budget if known)."""
        from himmy.runtime.compaction import estimate_tokens

        tokens = sum(
            estimate_tokens(m.content) for m in getattr(thread, "messages", [])
        )
        maximum = self.spec.compact_after_tokens if self.spec.compact_context else None
        if maximum:
            pct = min(100, int(tokens * 100 / maximum)) if maximum else 0
            return f"~{_kfmt(tokens)} / ~{_kfmt(maximum)} tokens ({pct}%)"
        return f"~{_kfmt(tokens)} tokens"

    def _footer(self, thread: Any) -> None:
        """The faint post-turn footer: context gauge + budget spend if a budget is set."""
        c = self._c
        gauge = self._context_meter(thread)
        spent = self.recorder.total_cost
        budget = ""
        if self.budget:
            budget = f" · ${spent:.4f} / ${self.budget:.2f}"
        elif spent:
            budget = f" · ${spent:.4f}"
        self._eprint(f"{c['faint']}  {gauge}{budget}{c['reset']}")
        self._maybe_warn_budget()

    # -------------------------------------------------------------- budgets

    def _maybe_warn_budget(self) -> None:
        """Print ONE gold warning when cumulative spend first crosses 80% of budget."""
        if not self.budget or self._budget_warned:
            return
        if self.recorder.total_cost >= 0.8 * self.budget:
            self._budget_warned = True
            c = self._c
            self._eprint(
                f"{c['gold']}⚠ budget: ${self.recorder.total_cost:.4f} of "
                f"${self.budget:.2f} spent (≥80%){c['reset']}"
            )

    def _budget_blocks_turn(self) -> bool:
        """At/over budget before a turn starts → prompt continue? [y/N]; True = block."""
        if not self.budget or self.recorder.total_cost < self.budget:
            return False
        c = self._c
        try:
            answer = (
                self._input(
                    f"budget reached (${self.recorder.total_cost:.4f} spent of "
                    f"${self.budget:.2f}) — continue? [y/N] "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes"):
            return False
        self._eprint(f"{c['dim']}— skipped (budget){c['reset']}")
        return True

    # ----------------------------------------------------------- spawn / jobs

    def _spawn(self, task_text: str) -> None:
        """``/spawn <task>``: run ONE sub-agent turn on a daemon thread, silently.

        Builds a FRESH runtime/registry + thread (never shares the live runtime
        across threads) with NO hitl — so any approval-gated tool in a background
        job fails closed (background jobs cannot prompt for approval). The job's
        events are silent (``on_event=None``), so nothing interleaves with the REPL.
        """
        c = self._c
        with self._jobs_lock:
            self._job_seq += 1
            job = _Job(job_id=f"j{self._job_seq}", task=task_text)
            self.jobs.append(job)
        self._eprint(f"{c['dim']}↗ {job.job_id} spawned — /jobs to check{c['reset']}")

        def _worker() -> None:
            try:
                output = self._run_job(task_text)
                with self._jobs_lock:
                    job.result = output
                    job.status = "done"
                    job.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001 - surface failure, don't crash REPL
                with self._jobs_lock:
                    job.result = f"{type(exc).__name__}: {exc}"
                    job.status = "failed"
                    job.finished_at = time.time()

        threading.Thread(target=_worker, daemon=True).start()

    def _run_job(self, task_text: str) -> str:
        """Run one isolated sub-agent turn to completion; return its final text."""
        from himmy.agents.base_agent.thread import ChatThread

        # Fresh runtime/registry on a fresh loop in THIS thread (no shared state, no
        # checkpoint store → gated tools fail closed; no live UI → no stderr writes).
        from himmy.cli.commands import _build_runtime_for

        runtime, registry = _build_runtime_for(self.spec, self.args, on_event=None)
        self._apply_permissions(registry)
        thread = ChatThread(agent_id=self.persona.agent_id)

        async def _go() -> Any:
            return await runtime.run_agent_loop(
                self.persona,
                self.spec.make_task(task_text),
                thread,
                llm_config=self.llm_config,
                max_turns=8,
                route_tools=self.spec.tool_router,
                hitl=False,
            )

        result = asyncio.run(_go())
        return result.final.output_text or ""

    def _notify_finished_jobs(self) -> None:
        """Before a prompt: one dim line per job that finished since the last check."""
        c = self._c
        with self._jobs_lock:
            newly = [j for j in self.jobs if j.status != "running" and not j.notified]
            for j in newly:
                j.notified = True
        for j in newly:
            mark = "✓" if j.status == "done" else "✗"
            self._eprint(
                f"{c['dim']}{mark} {j.job_id} {j.status} — "
                f"/jobs {j.job_id} to read{c['reset']}"
            )

    def _jobs(self, rest: list[str]) -> None:
        """``/jobs`` lists jobs; ``/jobs <id>`` prints that job's full answer."""
        c = self._c
        with self._jobs_lock:
            jobs = list(self.jobs)
        if rest:
            target = rest[0]
            match = next((j for j in jobs if j.job_id == target), None)
            if match is None:
                self._eprint(f"{c['dim']}no such job {target}{c['reset']}")
                return
            if match.status == "running":
                self._eprint(f"{c['dim']}{target} still running…{c['reset']}")
                return
            self._eprint(
                f"{c['snow']}{target} ({match.status}) — {match.task}{c['reset']}"
            )
            import sys

            sys.stdout.write(
                render_markdown_lite(match.result, stream=sys.stdout) + "\n"
            )
            sys.stdout.flush()
            return
        if not jobs:
            self._eprint(f"{c['dim']}(no background jobs){c['reset']}")
            return
        self._eprint(f"{c['snow']}background jobs:{c['reset']}")
        for j in jobs:
            self._eprint(
                f"  {c['dim']}{j.job_id}  {j.status:<8}{c['reset']}"
                f"{c['faint']}{_truncate(j.task, 60)}{c['reset']}"
            )

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
        if cmd == "/why":
            self._why()
            return thread
        if cmd in ("/good", "/bad"):
            parts = line.split(None, 1)
            note = parts[1].strip() if len(parts) > 1 else None
            self._feedback("up" if cmd == "/good" else "down", note)
            return thread
        if cmd == "/spawn":
            task_text = (
                line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            )
            if not task_text:
                self._eprint(f"{c['dim']}usage: /spawn <task>{c['reset']}")
                return thread
            self._spawn(task_text)
            return thread
        if cmd == "/jobs":
            self._jobs(rest)
            return thread
        if cmd == "/workflow":
            from himmy.cli.workflow_cmd import slash_workflow

            parts = line.split(None, 1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            slash_workflow(self, arg)
            return thread
        if cmd == "/orchestrate":
            from himmy.cli.orchestrate import slash_orchestrate

            slash_orchestrate(self, rest)
            return thread
        if cmd == "/guardrails":
            from himmy.cli.guardrails_view import slash_guardrails

            slash_guardrails(self)
            return thread
        if cmd == "/lineage":
            self._lineage(thread, as_dot=bool(rest and rest[0] == "--dot"))
            return thread
        if cmd == "/seclog":
            from himmy.cli.security_audit_cmd import cli_security_log, render_seclog

            # Read the SAME durable spine the deny/approval sites write to, so a
            # denial recorded this session shows up here (and across processes).
            want_json = bool(rest) and rest[0] == "--json"
            render_seclog(cli_security_log(), limit=20, as_json=want_json)
            return thread
        if cmd == "/whoami":
            from himmy.cli.rbac_cmd import cmd_whoami as _whoami

            _whoami(argparse.Namespace(role=getattr(self.args, "role", None)))
            return thread
        if cmd == "/mcp":
            from himmy.cli.mcp_cmd import cmd_mcp

            action = rest[0] if rest else "list"
            name = rest[1] if len(rest) > 1 else None
            cmd_mcp(
                argparse.Namespace(
                    action=action,
                    name=name,
                    file=getattr(self.args, "file", None),
                    command=None,
                    arg=None,
                    env=None,
                    cwd=None,
                    prefix=None,
                    requires_approval=False,
                    tool=None,
                )
            )
            return thread
        if cmd == "/skill":
            from himmy.cli.skill_dispatch import slash_skill

            slash_skill(self, rest)
            return thread
        if cmd == "/compact":
            self._compact(thread)
            return thread
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
            budget = f" / ${self.budget:.2f} budget" if self.budget else ""
            self._eprint(
                f"{c['dim']}this session: {self.recorder.total_events} events · "
                f"{self.recorder.total_tool_calls} tool call(s) · "
                f"${self.recorder.total_cost:.4f}{budget}{c['reset']}"
            )
        elif cmd == "/help":
            self._eprint(
                f"{c['dim']}/model [name [provider]]  switch the model\n"
                f"/plan <task>              plan a task, then run it on approval\n"
                f"/resume                   pick a recent session to continue\n"
                f"/tools                    what this agent can call\n"
                f"/agents                   specs in this directory\n"
                f"/stream on|off            stream tool-turn answers (extra call)\n"
                f"/cost                     session events + spend (vs budget)\n"
                f"/why                      full forensics on the last turn\n"
                f"/good [note] · /bad [note] rate the last answer (teaches the "
                f"agent over time)\n"
                f"/lineage [--dot]          provenance tree of this turn "
                f"(run -> tools -> results)\n"
                f"/spawn <task>             run a task in the background (can't "
                f"approve gated tools)\n"
                f"/jobs [id]                list background jobs, or read one\n"
                f"/workflow <file>          run a workflow.yaml on the live model\n"
                f"/orchestrate <mode> <p>   run team.yaml through a multi-agent mode\n"
                f"/skill <name> <task>      run a named skill as a focused sub-agent\n"
                f"/mcp [list|test] [name]   inspect the agent's MCP servers\n"
                f"/guardrails               active guardrails + how many fired "
                f"this turn\n"
                f"/seclog [--json]          recent security events "
                f"(denied/approved/blocked)\n"
                f"/whoami                   your RBAC role and what it grants\n"
                f"/compact                  summarize old history to free up context\n"
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
        from himmy.config.project import conversations_db_path
        from himmy.runtime.session import SqliteSessionStore

        Path(".himmy").mkdir(exist_ok=True)
        self.rt["session_store"] = SqliteSessionStore(conversations_db_path())

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

        if self.budget:
            self._eprint(
                f"{c['faint']}budget: ${self.budget:.2f} this session{c['reset']}"
            )

        thread = self._new_thread()
        try:
            while True:
                self._notify_finished_jobs()
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
                if self._budget_blocks_turn():
                    continue  # at/over budget and the user declined to continue
                thread = self._run_turn_coro(self._chat_turn(thread, line), thread)
                self._persist(thread)
                self._footer(thread)
        finally:
            with self._jobs_lock:
                running = sum(1 for j in self.jobs if j.status == "running")
            if running:
                self._eprint(
                    f"{c['dim']}⚠ {running} background job(s) still running — "
                    f"abandoned on exit{c['reset']}"
                )
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


def _render_args(args: Any) -> str:
    """Compact JSON of tool args/results for the faint approval/forensics preview."""
    import json

    try:
        return json.dumps(args or {}, default=str)
    except Exception:  # noqa: BLE001 - preview must never raise
        return str(args)


#: How many lines a single /why forensic block (args/result) may show before truncating.
_WHY_BLOCK_LINES = 6


def _render_forensic_block(
    value: Any, *, max_lines: int = _WHY_BLOCK_LINES
) -> list[str]:
    """Render one /why args/result value as readable, line-bounded text.

    The fix for the /why double-escape bug: tool results often arrive as an *already
    serialized* JSON string. The old path re-``json.dumps``ed it with
    ``ensure_ascii=True``, double-escaping non-ASCII into ``\\uXXXX`` sprawl (e.g.
    काठमाडौं → ``\\u0915\\u093e…``). Here a string that parses as JSON is decoded and
    re-rendered with ``ensure_ascii=False, indent=2`` so Devanagari (and any unicode)
    renders as itself; non-JSON strings pass through verbatim; other values pretty-print.
    The block is truncated to ``max_lines`` with a ``… +N more lines`` marker so a huge
    result never floods the forensics view. Returns a list of lines (never raises).
    """
    import json

    text: str
    try:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "{[":
                try:
                    parsed = json.loads(stripped)
                    text = json.dumps(parsed, ensure_ascii=False, indent=2)
                except (ValueError, TypeError):
                    text = value
            else:
                text = value
        else:
            text = json.dumps(value or {}, ensure_ascii=False, indent=2, default=str)
    except Exception:  # noqa: BLE001 - forensics must never raise
        text = str(value)
    lines = text.splitlines() or [""]
    if len(lines) <= max_lines:
        return lines
    hidden = len(lines) - max_lines
    return [*lines[:max_lines], f"… +{hidden} more line{'s' if hidden != 1 else ''}"]


def prompt_approval(
    checkpoint_store: Any,
    checkpoint_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    on_line: Callable[[str], None] | None = None,
    role: str | None = None,
) -> bool:
    """Render a checkpoint's pending tool call(s) and ask approve? [y/N/always].

    Shared by the chat REPL and one-shot ``himmy run``. ``on_line`` receives each
    rendered (already-styled) diagnostic line — defaults to printing to stderr.
    ``always`` approves AND persists every pending tool name to ``himmy.toml``
    ``[permissions] auto_approve``. Returns the approve/deny decision.

    RBAC enforcement runs FIRST and only ever RESTRICTS: the caller's ``role``
    (resolved once into a :class:`Principal` against the active
    :class:`AccessPolicy`) gates every pending gated tool. If any is DENIED for the
    role (a viewer hitting a gated tool), the denial is emitted and the call fails
    closed (returns ``False``) with no prompt. Otherwise the normal HITL approve?
    prompt is shown unchanged — RBAC never auto-approves away the human approval
    gate. The default role is ``admin`` (unrestricted, nothing denied), so
    unconfigured sessions behave exactly as before.
    """
    import sys

    from himmy.cli.rbac_cmd import (
        cli_access_policy,
        cli_principal,
        deny_message,
        gate_tool_for_role,
    )
    from himmy.cli.security_audit_cmd import record_security_event

    def _emit(text: str) -> None:
        if on_line is not None:
            on_line(text)
        else:
            print(text, file=sys.stderr)

    c = styles(sys.stderr)
    role_label = next(iter(sorted(cli_principal(argparse.Namespace(role=role)).roles)))

    def _seclog(outcome: str, tool: str, detail: str) -> None:
        record_security_event(
            event_type="authz_denied" if outcome == "deny" else "access",
            outcome=outcome,
            actor={"subject": role_label, "auth_method": "cli"},
            action="tool_call",
            resource=tool,
            detail=detail,
        )

    checkpoint = checkpoint_store.load(checkpoint_id)
    pending = list(checkpoint.pending_tool_calls) if checkpoint else []
    # RBAC gate: resolve the principal/policy once, then decide every pending call.
    principal = cli_principal(argparse.Namespace(role=role))
    policy = cli_access_policy()
    decisions = [
        gate_tool_for_role(principal, policy, call.tool_name, True) for call in pending
    ]
    if any(d == "deny" for d in decisions):
        for call, decision in zip(pending, decisions, strict=False):
            if decision == "deny":
                _emit(deny_message(principal, call.tool_name, stream=sys.stderr))
                _seclog("deny", call.tool_name, "rbac: role may not run gated tool")
        return False  # fail closed: a restricted role may not run a gated tool
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
        for call in pending:
            _seclog("deny", call.tool_name, "approval rejected at prompt")
        return False
    if answer in ("a", "always"):
        for call in pending:
            if permissions.persist_auto_approve(call.tool_name):
                _emit(
                    f"{c['dim']}(added {call.tool_name} to "
                    f"himmy.toml auto_approve){c['reset']}"
                )
            _seclog("allow", call.tool_name, "approval granted (always)")
        return True
    approved = answer in ("y", "yes")
    for call in pending:
        _seclog(
            "allow" if approved else "deny",
            call.tool_name,
            "approval granted" if approved else "approval rejected at prompt",
        )
    return approved


def _cli_checkpoint_db() -> str:
    """Path to the CLI's durable approval checkpoint store (``.himmy/approvals.db``).

    Mirrors Himmy Studio's store location so the CLI and Studio share the same
    durable approvals DB under the project root.
    """
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "approvals.db")


__all__ = ["ChatRepl", "cancel_inflight_task", "prompt_approval"]
