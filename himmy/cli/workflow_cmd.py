"""``himmy workflow`` — run a declarative WORKFLOW through the real orchestrator.

A workflow is an ordered list of steps that share one chat thread and an
accumulating ``state`` dict (each step's ``output_key`` threads into the next
step's ``{placeholder}`` subtask). This surfaces the deterministic
:class:`~himmy.orchestrators.workflow.WorkflowOrchestrator` in the CLI:

* ``himmy workflow run -f research.workflow.yaml`` loads the spec, builds a
  runtime on the CLI-selected provider (default: the offline stub), and drives
  every step through the **real** orchestrator — rendering each step live
  (name · status · timing) and printing the final result + accumulated state.
* ``himmy workflow init`` (or ``--scaffold``) writes a minimal working
  ``workflow.yaml`` you can run immediately on the stub.
* ``/workflow <file>`` runs one inside the chat REPL on the live runtime.

Everything is offline-first: no spec/flags ⇒ the stub provider, so a bare
``himmy workflow run -f workflow.yaml`` works with no keys and no network. All
live decoration goes to stderr (TTY only); stdout stays clean for scripts, and
``--json`` emits the machine-readable :class:`WorkflowResult`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from himmy.cli.commands import (
    _build_runtime_for,
    _eprint,
    _model_label,
    _spec_from_args,
)
from himmy.cli.ui import LiveRunUI, compose_event_handlers, styles

# A minimal, immediately-runnable workflow: two steps, the first threads its
# output into the second via {topic}/{draft}. Runs on the stub out of the box.
_WORKFLOW_YAML = """\
# A linear workflow: ordered steps share one thread + an accumulating state dict.
# Each step's `output_key` is substituted into a later step's `{placeholder}`.
# Run it:  himmy workflow run -f workflow.yaml
name: research-brief
description: Draft notes on a topic, then tighten them into a short brief.
steps:
  - name: draft
    subtask: "Draft a few rough notes about {topic}."
    output_key: draft
  - name: brief
    subtask: "Rewrite these notes as a tight 3-bullet brief:\\n{draft}"
    output_key: brief
"""


def _scaffold(target: Path, *, force: bool) -> int:
    """Write a minimal working ``workflow.yaml`` at ``target`` (a file or dir)."""
    dest = target / "workflow.yaml" if target.is_dir() else target
    if dest.exists() and not force:
        _eprint(f"error: {dest} already exists (use --force to overwrite)")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_WORKFLOW_YAML, encoding="utf-8")
    print(f"wrote {dest}")
    print(f'\nNext: himmy workflow run -f {dest} --state topic="permaculture"')
    return 0


def _parse_state(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``--state key=value`` pairs into the workflow's initial state dict.

    A value that parses as JSON (number, bool, object, list, quoted string) is
    used as-is; anything else is kept as the raw string. So ``topic=ACME`` and
    ``count=3`` and ``tags=["a","b"]`` all do the obvious thing.
    """
    state: dict[str, Any] = {}
    for raw in pairs or []:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"--state expects key=value, got {raw!r}")
        key = key.strip()
        if not key:
            raise ValueError(f"--state has an empty key in {raw!r}")
        try:
            state[key] = json.loads(value)
        except (ValueError, TypeError):
            state[key] = value
    return state


class _StepReporter:
    """Render each workflow step's start/finish live (name · status · timing).

    Wired as an ``on_event`` listener alongside :class:`LiveRunUI`: it watches the
    orchestrator's ``WORKFLOW_STEP_COMPLETED`` markers and prints a green ✓ / crimson
    ✗ line with the elapsed wall-clock since the prior step. Decoration only — it
    writes to stderr (TTY-styled, plain off-TTY) and never raises into the run.
    """

    def __init__(self, total: int, *, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._c = styles(self._stream)
        self._total = total
        self._done = 0
        self._last = time.monotonic()

    def _line(self, text: str) -> None:
        self._stream.write(text + "\n")
        self._stream.flush()

    async def handle(self, event: Any) -> None:
        try:
            kind = getattr(event.event_type, "value", str(event.event_type))
            if kind != "WORKFLOW_STEP_COMPLETED":
                return
            c = self._c
            p = event.payload or {}
            self._done += 1
            elapsed = time.monotonic() - self._last
            self._last = time.monotonic()
            name = p.get("step_name", "")
            status = p.get("status", "")
            if status == "completed":
                mark = f"{c['green']}✓{c['reset']}"
            else:
                mark = f"{c['crimson']}✗{c['reset']}"
            err = ""
            if status != "completed" and getattr(event, "error", None):
                err = f"{c['faint']} — {event.error}{c['reset']}"
            self._line(
                f"  {mark} {c['dim']}step {self._done}/{self._total}{c['reset']} "
                f"{c['snow']}{name}{c['reset']} "
                f"{c['faint']}({status}, {elapsed * 1000:.0f}ms){c['reset']}{err}"
            )
        except Exception:  # noqa: BLE001 - never let decoration break a run
            pass


def _run_workflow_core(
    workflow: Any,
    persona: Any,
    runtime: Any,
    *,
    initial_state: dict[str, Any],
    on_event: Any,
    total_timeout_seconds: float | None,
) -> Any:
    """Drive ``workflow`` through the REAL orchestrator and return the result.

    Constructs a :class:`~himmy.orchestrators.workflow.WorkflowOrchestrator` over the
    given runtime (not a mock) and runs it under ``asyncio.run``. Kept separate from
    argument plumbing so it is unit-testable on its own with a stub-backed runtime.
    """
    from himmy.orchestrators.workflow import WorkflowOrchestrator

    orch = WorkflowOrchestrator(
        runtime,
        on_event=on_event,
        total_timeout_seconds=total_timeout_seconds,
    )

    async def _go() -> Any:
        return await orch.run(workflow, persona, initial_state=initial_state)

    return asyncio.run(_go())


def _print_result(result: Any, *, as_json: bool) -> None:
    """Print the workflow's final result: ``--json`` dump or a styled summary."""
    if as_json:
        print(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str)
        )
        return

    c = styles(sys.stdout)
    status = result.status
    if status == "completed":
        head = f"{c['green']}workflow completed{c['reset']}"
    elif status == "partial":
        head = f"{c['gold']}workflow partial{c['reset']}"
    elif status == "cancelled":
        head = f"{c['crimson']}workflow cancelled{c['reset']}"
    else:
        head = f"{c['crimson']}workflow failed{c['reset']}"
    done = sum(1 for r in result.step_results if r.status == "completed")
    print(f"{head} {c['faint']}({done}/{len(result.step_results)} steps){c['reset']}")

    for r in result.step_results:
        mark = "✓" if r.status == "completed" else "✗"
        colour = c["green"] if r.status == "completed" else c["crimson"]
        line = f"  {colour}{mark}{c['reset']} {c['snow']}{r.step_name}{c['reset']}"
        if r.error:
            line += f"  {c['faint']}{r.error}{c['reset']}"
        print(line)

    # The final accumulated output: prefer the last step's text, else the state.
    final_text = ""
    for r in reversed(result.step_results):
        if r.status == "completed" and r.output_text:
            final_text = r.output_text
            break
    if final_text:
        from himmy.cli.ui import render_markdown_lite

        print()
        print(render_markdown_lite(final_text, stream=sys.stdout))
    elif result.final_state:
        print(
            f"\n{c['faint']}final state: "
            f"{json.dumps(result.final_state, default=str)[:200]}{c['reset']}"
        )


def cmd_workflow(args: argparse.Namespace) -> int:
    """``himmy workflow``: scaffold or run a declarative workflow.

    ``himmy workflow init`` (or ``--scaffold``) writes a starter ``workflow.yaml``.
    Otherwise loads the ``-f`` spec, builds a runtime on the CLI-selected provider
    (offline stub by default), and runs every step through the real orchestrator —
    rendering step progress live and printing the final result.
    """
    action = getattr(args, "action", None)

    # Scaffold path: `himmy workflow init [dir]` or `--scaffold`.
    if action == "init" or getattr(args, "scaffold", False):
        directory = getattr(args, "directory", None) or "."
        return _scaffold(
            Path(directory).expanduser(), force=getattr(args, "force", False)
        )

    spec_path = getattr(args, "file", None)
    if not spec_path:
        _eprint(
            "error: `himmy workflow run` needs -f <workflow.yaml> "
            "(or `himmy workflow init` to scaffold one)"
        )
        return 2

    path = Path(spec_path).expanduser()
    if not path.is_file():
        _eprint(f"error: workflow file not found: {path}")
        return 2

    from himmy.config.workflow_spec import load_workflow_spec

    try:
        workflow = load_workflow_spec(str(path))
    except Exception as exc:  # noqa: BLE001 - report a bad spec cleanly, don't crash
        _eprint(f"error: could not load workflow {path}: {exc}")
        return 2

    if not workflow.steps:
        _eprint(f"error: workflow {workflow.name!r} has no steps")
        return 2

    try:
        initial_state = _parse_state(getattr(args, "state", None))
    except ValueError as exc:
        _eprint(f"error: {exc}")
        return 2

    # Build the agent spec (honours -f-style flags, --provider/--model) and the
    # runtime — defaulting to the offline stub when nothing is configured.
    spec = _spec_from_args(args)

    from himmy.cli.commands import _maybe_hint_stub

    _maybe_hint_stub(spec, args)

    c = styles(sys.stderr)
    _eprint(
        f"{c['snow']}{workflow.name}{c['reset']} "
        f"{c['faint']}— {len(workflow.steps)} step(s) "
        f"· {_model_label(spec, args) or 'auto'}{c['reset']}"
    )

    live = LiveRunUI(model_label=_model_label(spec, args))
    reporter = _StepReporter(len(workflow.steps))
    on_event = compose_event_handlers(
        live.handle if live.enabled else None,
        reporter.handle,
    )
    runtime, registry = _build_runtime_for(spec, args, on_event=on_event)
    # RBAC: a restricted --role re-gates any gated tool it may not run (fail closed),
    # consistent with run/chat — so a viewer can't slip a gated tool via workflow.
    from himmy.cli.rbac_cmd import enforce_role_on_registry

    enforce_role_on_registry(
        registry, role=getattr(args, "role", None), on_line=_eprint
    )

    timeout = getattr(args, "timeout", None)
    try:
        result = _run_workflow_core(
            workflow,
            spec.to_persona(),
            runtime,
            initial_state=initial_state,
            on_event=on_event,
            total_timeout_seconds=float(timeout) if timeout else None,
        )
    except asyncio.CancelledError:
        live.finish()
        _eprint(f"{c['crimson']}workflow cancelled (timeout){c['reset']}")
        return 1

    live.finish()
    _print_result(result, as_json=bool(getattr(args, "json", False)))
    live.footer()
    return 0 if result.status == "completed" else 1


def slash_workflow(repl: Any, arg: str) -> Any:
    """``/workflow <file>`` REPL handler: run a workflow on the live runtime.

    Loads the spec at ``arg``, runs it through the real orchestrator on the REPL's
    already-built runtime/persona (sharing the live model), and prints the result.
    ``repl`` is a :class:`~himmy.cli.repl.ChatRepl`; only ``rt['runtime']``,
    ``persona``, and ``_eprint``/``_c`` are used, so a light fake works in tests.
    Returns nothing meaningful — callers keep their current thread.
    """
    c = repl._c
    spec_path = (arg or "").strip()
    if not spec_path:
        repl._eprint(f"{c['dim']}usage: /workflow <file.yaml>{c['reset']}")
        return

    path = Path(spec_path).expanduser()
    if not path.is_file():
        repl._eprint(f"{c['dim']}workflow file not found: {path}{c['reset']}")
        return

    from himmy.config.workflow_spec import load_workflow_spec

    try:
        workflow = load_workflow_spec(str(path))
    except Exception as exc:  # noqa: BLE001 - report cleanly inside the REPL
        repl._eprint(f"{c['dim']}could not load workflow: {exc}{c['reset']}")
        return

    if not workflow.steps:
        repl._eprint(f"{c['dim']}workflow {workflow.name!r} has no steps{c['reset']}")
        return

    runtime = repl.rt.get("runtime")
    if runtime is None:
        repl._eprint(f"{c['dim']}(no runtime to run the workflow){c['reset']}")
        return

    repl._eprint(
        f"{c['snow']}{workflow.name}{c['reset']} "
        f"{c['faint']}— running {len(workflow.steps)} step(s){c['reset']}"
    )
    reporter = _StepReporter(len(workflow.steps))
    from himmy.orchestrators.workflow import WorkflowOrchestrator

    orch = WorkflowOrchestrator(runtime, on_event=reporter.handle)

    async def _go() -> Any:
        return await orch.run(workflow, repl.persona, initial_state={})

    live = repl.rt.get("live")
    try:
        # Reuse the REPL's loop so we don't spin a second event loop.
        loop = getattr(repl, "loop", None)
        if loop is not None:
            result = loop.run_until_complete(_go())
        else:
            result = asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001 - a failed workflow shouldn't kill the REPL
        repl._eprint(f"{c['dim']}workflow error: {exc}{c['reset']}")
        return
    finally:
        if live is not None:
            live.finish()

    done = sum(1 for r in result.step_results if r.status == "completed")
    status_colour = c["green"] if result.status == "completed" else c["gold"]
    repl._eprint(
        f"{status_colour}workflow {result.status}{c['reset']} "
        f"{c['faint']}({done}/{len(result.step_results)} steps){c['reset']}"
    )
    final_text = ""
    for r in reversed(result.step_results):
        if r.status == "completed" and r.output_text:
            final_text = r.output_text
            break
    if final_text:
        from himmy.cli.ui import render_markdown_lite

        print(render_markdown_lite(final_text, stream=sys.stdout))


__all__ = ["cmd_workflow", "slash_workflow"]
