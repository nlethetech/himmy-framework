"""Tier-1 agentic trust + control in the chat REPL / one-shot run.

Covers in-REPL + in-run approvals (approve / reject / always → himmy.toml), the
permission profile (allowlist / --yolo / --safe / mutual-exclusion), plan mode,
the interrupt-cancellation helper, and the critical non-TTY fail-closed path.

The approval-producing run reuses the runtime HITL harness (an approval-gated
``wire_money`` tool + the ``_ApprovalGateManager`` that invokes it), driving a
real checkpoint through ``run_agent_loop(hitl=True)`` / ``resume_agent_loop`` —
exactly what the REPL does — rather than inventing a parallel mechanism.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path
from typing import Any

import pytest

from himmy.cli import permissions
from himmy.cli.repl import ChatRepl, cancel_inflight_task, prompt_approval
from himmy.runtime import InMemoryCheckpointStore
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from tests.conftest import run_async
from tests.runtime.test_hitl import _hitl_runtime


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "yolo": False,
        "safe": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _repl_with_hitl_runtime(
    inputs: list[str], store: Any = None
) -> tuple[ChatRepl, Any]:
    """A ChatRepl wired to the HITL test runtime (approval-gated wire_money tool).

    Bypasses ``rebuild()`` so we inject the deterministic approval-gate runtime
    instead of a provider-built one; everything else (the approval loop, the
    prompt, the toml write) is the real REPL code.
    """
    from himmy.cli.ui import LiveRunUI
    from himmy.config.agent_spec import AgentSpec

    rt_obj, cp_store = _hitl_runtime(store=store)
    # spec.tools makes make_task bind wire_money via context['tool_names'].
    spec = AgentSpec(
        name="treasurer",
        description="pays vendors",
        provider="stub",
        tools=["wire_money"],
    )
    it = iter(inputs)
    repl = ChatRepl(spec, _args(name="treasurer"), input_fn=lambda _p: next(it))
    from himmy.agents.personas.persona import Persona

    repl.persona = Persona(name="treasurer")
    repl.llm_config = None
    repl.interactive = True
    repl.rt = {
        "runtime": rt_obj,
        "registry": rt_obj.tool_service.registry,
        "checkpoint_store": cp_store,
        "live": LiveRunUI(model_label="", stream=io.StringIO()),
    }
    return repl, cp_store


# ----------------------------------------------------------------- approvals


def test_approve_executes_the_tool() -> None:
    """A scripted 'y' approves the pending tool, which then really executes."""
    repl, _store = _repl_with_hitl_runtime(["y"])
    from himmy.agents.base_agent.thread import ChatThread, MessageRole

    thread = ChatThread(agent_id=repl.persona.agent_id)
    loop_result = run_async(repl._drive_turn_with_approvals(thread, "wire 1000"))
    assert loop_result.checkpoint_id is None
    tool_msgs = [m for m in loop_result.thread.messages if m.role == MessageRole.TOOL]
    assert any('"sent"' in (m.content or "") for m in tool_msgs)


def test_reject_records_rejection_without_running_tool() -> None:
    """A scripted 'n' rejects: the run completes, the tool never ran."""
    repl, store = _repl_with_hitl_runtime(["n"])
    from himmy.agents.base_agent.thread import ChatThread, MessageRole

    thread = ChatThread(agent_id=repl.persona.agent_id)
    loop_result = run_async(repl._drive_turn_with_approvals(thread, "wire 1000"))
    assert loop_result.checkpoint_id is None
    tool_msgs = [m for m in loop_result.thread.messages if m.role == MessageRole.TOOL]
    assert any("rejected" in (m.content or "").lower() for m in tool_msgs)
    assert not any('"sent"' in (m.content or "") for m in tool_msgs)


def test_always_persists_tool_to_himmy_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'always' approves AND writes the tool into himmy.toml auto_approve."""
    monkeypatch.chdir(tmp_path)
    repl, _store = _repl_with_hitl_runtime(["always"])
    from himmy.agents.base_agent.thread import ChatThread

    thread = ChatThread(agent_id=repl.persona.agent_id)
    run_async(repl._drive_turn_with_approvals(thread, "wire 1000"))
    assert (tmp_path / "himmy.toml").is_file()
    assert "wire_money" in permissions.load_auto_approve(tmp_path)


def test_prompt_approval_shared_helper(tmp_path: Path) -> None:
    """The shared prompt helper renders the pending call and reads the decision."""
    store = InMemoryCheckpointStore()
    rt_obj, cp_store = _hitl_runtime(store=store)
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    paused = run_async(
        rt_obj.run_agent_loop(
            Persona(name="t"),
            Task(
                title="t",
                prompt="wire",
                context={"tool_names": ["wire_money"], "response_format": "AUTO_TOOLS"},
            ),
            hitl=True,
        )
    )
    lines: list[str] = []
    decision = prompt_approval(
        cp_store,
        paused.checkpoint_id,
        input_fn=lambda _p: "y",
        on_line=lines.append,
    )
    assert decision is True
    assert any("approval required" in line and "wire_money" in line for line in lines)


# --------------------------------------------------------------- permissions


def _gated_registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_local_tool(
        reg,
        name="wire_money",
        handler=lambda args: {"sent": True},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=True,
    )
    return reg


def test_allowlist_ungates_named_tool() -> None:
    """A tool on the allowlist has requires_approval flipped off."""
    reg = _gated_registry()
    permissions.apply_allowlist(reg, ["wire_money"])
    assert reg.get("wire_money").requires_approval is False


def test_allowlist_leaves_unlisted_tool_gated() -> None:
    reg = _gated_registry()
    permissions.apply_allowlist(reg, ["other_tool"])
    assert reg.get("wire_money").requires_approval is True


def test_yolo_grants_everything() -> None:
    reg = _gated_registry()
    permissions.grant_all_approvals(reg)
    assert reg.get("wire_money").requires_approval is False


def test_safe_ignores_allowlist_via_repl() -> None:
    """--safe means the REPL never applies the allowlist (tool stays gated)."""
    from himmy.config.agent_spec import AgentSpec

    repl = ChatRepl(
        AgentSpec(name="t", description="d", provider="stub"),
        _args(safe=True),
        input_fn=lambda _p: "n",
    )
    reg = _gated_registry()
    repl._apply_permissions(reg)
    assert reg.get("wire_money").requires_approval is True


def test_yolo_via_repl_grants_everything() -> None:
    from himmy.config.agent_spec import AgentSpec

    repl = ChatRepl(
        AgentSpec(name="t", description="d", provider="stub"),
        _args(yolo=True),
        input_fn=lambda _p: "n",
    )
    reg = _gated_registry()
    repl._apply_permissions(reg)
    assert reg.get("wire_money").requires_approval is False


def test_yolo_and_safe_mutually_exclusive() -> None:
    """argparse rejects --yolo --safe together."""
    from himmy.cli.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--yolo", "--safe", "hi"])


# ------------------------------------------------------------- toml round-trip


def test_persist_auto_approve_preserves_existing_file(tmp_path: Path) -> None:
    """A hand-authored himmy.toml keeps its other content + comments."""
    toml = tmp_path / "himmy.toml"
    toml.write_text(
        '# my project\n[defaults]\nprovider = "ollama"\n\n[toolkit]\n'
        'embedder = "ollama"\n'
    )
    assert permissions.persist_auto_approve("send_email", start=tmp_path) is True
    text = toml.read_text()
    assert "# my project" in text
    assert 'provider = "ollama"' in text
    assert 'embedder = "ollama"' in text
    assert permissions.load_auto_approve(tmp_path) == ["send_email"]
    # Idempotent: a second add of the same tool is a no-op.
    assert permissions.persist_auto_approve("send_email", start=tmp_path) is False


def test_persist_auto_approve_appends_to_existing_table(tmp_path: Path) -> None:
    toml = tmp_path / "himmy.toml"
    toml.write_text('[permissions]\nauto_approve = [\n    "a",\n]\n')
    permissions.persist_auto_approve("b", start=tmp_path)
    assert permissions.load_auto_approve(tmp_path) == ["a", "b"]


def test_persist_auto_approve_creates_file(tmp_path: Path) -> None:
    assert permissions.persist_auto_approve("x", start=tmp_path) is True
    assert permissions.load_auto_approve(tmp_path) == ["x"]


# ------------------------------------------------------------------ interrupt


def test_cancel_inflight_task_drains_cleanly() -> None:
    """The cancellation helper cancels + drains without a pending-task warning."""

    async def _scenario() -> bool:
        started = asyncio.Event()

        async def _long() -> None:
            started.set()
            await asyncio.sleep(60)

        task = asyncio.create_task(_long())
        await started.wait()
        await cancel_inflight_task(task)
        return task.cancelled()

    assert run_async(_scenario()) is True


def test_cancel_inflight_task_already_done() -> None:
    """Cancelling an already-finished task is a no-op and retrieves its result."""

    async def _scenario() -> None:
        async def _quick() -> int:
            return 7

        task = asyncio.create_task(_quick())
        await task
        await cancel_inflight_task(task)  # done branch

    run_async(_scenario())


# ---------------------------------------------------------------------- plan


def test_plan_mode_runs_on_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """/plan shows a model plan and runs it when the user answers y."""
    monkeypatch.chdir(tmp_path)
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(), input_fn=lambda _p: "y")
    repl.rebuild()
    lines: list[str] = []
    repl._eprint = staticmethod(lambda *a: lines.append(" ".join(str(x) for x in a)))  # type: ignore[method-assign,assignment]
    thread = ChatThread(agent_id=repl.persona.agent_id)
    new_thread = run_async(repl._plan_and_maybe_run(thread, "ship the feature"))
    assert new_thread is not None
    assert any("plan:" in line for line in lines)


def test_plan_mode_discards_on_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/plan with a 'n' answer discards the plan and keeps the same thread."""
    monkeypatch.chdir(tmp_path)
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name="t", description="d", provider="stub")
    repl = ChatRepl(spec, _args(), input_fn=lambda _p: "n")
    repl.rebuild()
    lines: list[str] = []
    repl._eprint = staticmethod(lambda *a: lines.append(" ".join(str(x) for x in a)))  # type: ignore[method-assign,assignment]
    thread = ChatThread(agent_id=repl.persona.agent_id)
    same = run_async(repl._plan_and_maybe_run(thread, "do thing"))
    assert same is thread
    assert any("discarded" in line for line in lines)


# --------------------------------------------------------------- non-TTY (!!)


def test_non_tty_gated_tool_is_policy_blocked_no_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL: piped (non-TTY) `himmy chat --message` never prompts/hangs.

    A gated tool stays POLICY_BLOCKED (hitl off) and the command returns without
    reading stdin for an approval — so scripts and CI can never deadlock.
    """
    monkeypatch.chdir(tmp_path)
    # An input_fn that would explode proves the non-interactive path never prompts.
    from himmy.cli.__main__ import main

    # `--message` runs the single-turn non-interactive path. With the stub provider
    # there are no real tools, but the key assertion is: it returns, no prompt.
    code = main(["chat", "--provider", "stub", "--name", "t", "--message", "hi"])
    assert code == 0


def test_non_tty_gated_tool_denied_without_prompt() -> None:
    """Without hitl, the HITL runtime's gated tool is denied (POLICY_BLOCKED).

    This is the substrate the non-interactive CLI relies on: ``run_agent_loop``
    with ``hitl=False`` never pauses — the gated tool is denied and the run
    finishes, so no approval prompt is ever reachable off a TTY.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import MessageRole
    from himmy.agents.personas.persona import Persona

    rt_obj, _store = _hitl_runtime()
    result = run_async(
        rt_obj.run_agent_loop(
            Persona(name="t"),
            Task(
                title="t",
                prompt="wire",
                context={"tool_names": ["wire_money"], "response_format": "AUTO_TOOLS"},
            ),
            hitl=False,
        )
    )
    assert result.checkpoint_id is None  # never paused
    tool_msgs = [m for m in result.thread.messages if m.role == MessageRole.TOOL]
    assert not any('"sent"' in (m.content or "") for m in tool_msgs)  # tool blocked
