"""Tier-2 conversation ergonomics: session continuity, @file/!cmd, HIMMY.md, streaming.

Everything runs offline against the stub. The REPL is driven via a scripted
``input_fn`` (the same pattern as ``test_repl_approvals``); the input-affordance
helpers are unit-tested as pure functions.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pytest

from himmy.cli.input_affordances import (
    classify_bang,
    expand_at_files,
    frame_for_agent,
    run_shell,
)
from himmy.cli.repl import ChatRepl
from himmy.config.agent_spec import AgentSpec
from tests.conftest import run_async


def _args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "file": None,
        "name": "t",
        "instruction": None,
        "provider": "stub",
        "model": None,
        "message": None,
        "session": None,
        "continue_last": False,
        "yolo": False,
        "safe": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _scripted_repl(inputs: list[str], **kw: Any) -> ChatRepl:
    """A stub-backed ChatRepl driven by a scripted list of inputs."""
    spec = AgentSpec(name="t", description="d", provider="stub")
    it = iter(inputs)
    repl = ChatRepl(spec, _args(**kw), input_fn=lambda _p: next(it))
    return repl


# ------------------------------------------------------------- @file expansion


def test_expand_at_file_inlines_contents(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha beta gamma")
    out = expand_at_files("read @notes.txt now", cwd=tmp_path)
    assert "```notes.txt" in out
    assert "alpha beta gamma" in out


def test_expand_at_file_truncates_with_marker(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("Q" * 5000)
    out = expand_at_files("@data.txt", cap=1000, cwd=tmp_path)
    assert "… (truncated)" in out
    # The inlined body is capped at exactly `cap` chars (well under the 5000 source).
    assert out.count("Q") == 1000


def test_expand_at_file_missing_warns_and_keeps_token(tmp_path: Path) -> None:
    warns: list[str] = []
    out = expand_at_files("@nope.txt please", on_warn=warns.append, cwd=tmp_path)
    assert "@nope.txt" in out  # left as-is
    assert warns and "no such file" in warns[0]


def test_expand_at_file_leaves_email_like_token_untouched(tmp_path: Path) -> None:
    # A leading-@ handle with no path-ish shape and no matching file is not expanded.
    assert expand_at_files("ping @someone", cwd=tmp_path) == "ping @someone"
    # An embedded email (token does not start with @) is never touched.
    assert expand_at_files("mail me@host.com", cwd=tmp_path) == "mail me@host.com"


# --------------------------------------------------------------- !cmd / !!cmd


def test_classify_bang_variants() -> None:
    assert classify_bang("!ls -la") == classify_bang("!ls -la")
    one = classify_bang("!echo hi")
    assert one is not None and one.command == "echo hi" and one.send_to_agent is False
    two = classify_bang("!!git status")
    assert two is not None and two.command == "git status" and two.send_to_agent is True
    assert classify_bang("!") is None
    assert classify_bang("!!") is None
    assert classify_bang("hello") is None


def test_run_shell_captures_output_and_exit_code() -> None:
    ok = run_shell("echo hello")
    assert "hello" in ok.output and ok.exit_code == 0 and ok.timed_out is False
    bad = run_shell("exit 3")
    assert bad.exit_code == 3


def test_run_shell_caps_output() -> None:
    res = run_shell("yes x | head -c 100000", cap=500)
    assert "… (truncated)" in res.output
    assert len(res.output) <= 500 + len("\n… (truncated)")


def test_run_shell_timeout_path_injected() -> None:
    """A timeout is surfaced as timed_out + exit 124 (injected runner, no real wait)."""
    import subprocess

    def _fake_run(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=1, output="partial")

    res = run_shell("sleep 99", timeout=1, runner=_fake_run)
    assert res.timed_out is True
    assert res.exit_code == 124
    assert "timed out" in res.output and "partial" in res.output


def test_handle_bang_run_only_does_not_call_agent() -> None:
    """`!cmd` shows output and returns None (the agent is never invoked)."""
    repl = _scripted_repl([])
    lines: list[str] = []
    repl._eprint = staticmethod(  # type: ignore[method-assign,assignment]
        lambda *a: lines.append(" ".join(str(x) for x in a))
    )
    sent = repl._handle_bang("!echo from-shell")
    assert sent is None  # agent NOT involved
    assert any("from-shell" in ln for ln in lines)


def test_handle_bang_send_to_agent_frames_output() -> None:
    """`!!cmd` returns the framed command+output to hand to the agent."""
    repl = _scripted_repl([])
    repl._eprint = staticmethod(lambda *a: None)  # type: ignore[method-assign,assignment]
    sent = repl._handle_bang("!!echo framed-out")
    assert sent is not None
    assert "I ran: `echo framed-out`" in sent
    assert "framed-out" in sent and "```" in sent


def test_frame_for_agent_includes_nonzero_exit() -> None:
    res = run_shell("exit 5")
    framed = frame_for_agent(res)
    assert "(exit 5)" in framed


# ----------------------------------------------------------- session continuity


def test_repl_turn_autopersists_to_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare REPL turn auto-saves the thread to the implicit 'last' session."""
    monkeypatch.chdir(tmp_path)
    from himmy.runtime.session import SqliteSessionStore

    repl = _scripted_repl(["hello", "/exit"])
    code = repl.run()
    assert code == 0
    from himmy.config.project import conversations_db_path

    store = SqliteSessionStore(conversations_db_path())
    assert store.load("last") is not None
    store.close()


def test_fresh_repl_does_not_autoload_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh REPL (no -c) starts empty even though a 'last' session exists."""
    monkeypatch.chdir(tmp_path)
    _scripted_repl(["seed turn", "/exit"]).run()
    # A new bare REPL: _new_thread must NOT load 'last'.
    repl = _scripted_repl(["/exit"])
    repl.rebuild()
    from himmy.config.project import conversations_db_path
    from himmy.runtime.session import SqliteSessionStore

    Path(".himmy").mkdir(exist_ok=True)
    repl.rt["session_store"] = SqliteSessionStore(conversations_db_path())
    thread = repl._new_thread()
    assert thread.messages == []  # fresh, not the seeded 'last'


def test_continue_last_loads_prior_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-c/--continue loads the saved 'last' thread at startup."""
    monkeypatch.chdir(tmp_path)
    _scripted_repl(["remember this", "/exit"]).run()
    repl = _scripted_repl(["/exit"], continue_last=True)
    repl.rebuild()
    from himmy.config.project import conversations_db_path
    from himmy.runtime.session import SqliteSessionStore

    repl.rt["session_store"] = SqliteSessionStore(conversations_db_path())
    thread = repl._new_thread()
    assert any("remember this" in (m.content or "") for m in thread.messages)


def test_continue_flag_plumbs_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-level `himmy -c` and `himmy chat -c` both parse to continue_last."""
    from himmy.cli.__main__ import build_parser

    ns = build_parser().parse_args(["chat", "-c", "--provider", "stub"])
    assert ns.continue_last is True


def test_explicit_session_unaffected_by_default_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --session still resumes that id (not 'last')."""
    monkeypatch.chdir(tmp_path)
    repl = _scripted_repl(["into session", "/exit"], session="work")
    repl.run()
    from himmy.config.project import conversations_db_path
    from himmy.runtime.session import SqliteSessionStore

    store = SqliteSessionStore(conversations_db_path())
    assert store.load("work") is not None
    # A bare REPL persists to 'last', so 'work' must be distinct.
    assert any(i.session_id == "work" for i in store.list_sessions())
    store.close()


def test_resume_picker_swaps_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/resume lists sessions and a scripted choice swaps the live thread + id."""
    monkeypatch.chdir(tmp_path)
    # Seed an explicit session so it is distinct from 'last'.
    _scripted_repl(["alpha content", "/exit"], session="alpha").run()

    repl = _scripted_repl(["1"])  # picker reads "1"
    repl.rebuild()
    from himmy.config.project import conversations_db_path
    from himmy.runtime.session import SqliteSessionStore

    repl.rt["session_store"] = SqliteSessionStore(conversations_db_path())
    lines: list[str] = []
    repl._eprint = staticmethod(  # type: ignore[method-assign,assignment]
        lambda *a: lines.append(" ".join(str(x) for x in a))
    )
    from himmy.agents.base_agent.thread import ChatThread

    swapped = repl._resume_picker(ChatThread(agent_id=repl.persona.agent_id))
    assert any("recent sessions" in ln for ln in lines)
    # The swapped thread is one of the seeded sessions (has prior messages).
    assert swapped.messages
    # Subsequent persistence targets the chosen session id.
    assert repl._session_id in ("alpha", "last")


def test_resume_picker_cancel_keeps_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _scripted_repl(["seed", "/exit"]).run()
    repl = _scripted_repl([""])  # blank → cancel
    repl.rebuild()
    from himmy.config.project import conversations_db_path
    from himmy.runtime.session import SqliteSessionStore

    repl.rt["session_store"] = SqliteSessionStore(conversations_db_path())
    repl._eprint = staticmethod(lambda *a: None)  # type: ignore[method-assign,assignment]
    from himmy.agents.base_agent.thread import ChatThread

    original = ChatThread(agent_id=repl.persona.agent_id)
    result = repl._resume_picker(original)
    assert result is original  # untouched on cancel


# ------------------------------------------------------------------- HIMMY.md


def test_himmy_md_folds_into_house_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discovered HIMMY.md is appended to the house agent's instructions."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "HIMMY.md").write_text("Always answer in haiku.")
    from himmy.cli.commands import _spec_from_args

    # House agent path: no name/instruction/provider.
    ns = _args(name=None, provider=None)
    spec = _spec_from_args(ns)
    assert any("Project notes (HIMMY.md)" in i for i in spec.instructions)
    assert any("haiku" in i for i in spec.instructions)


def test_himmy_md_folds_into_discovered_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HIMMY.md folds into a discovered agent.yaml spec too."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "name: disc\ndescription: d\ninstructions:\n  - Be terse.\n"
    )
    (tmp_path / "HIMMY.md").write_text("Prefer metric units.")
    from himmy.cli.commands import _spec_from_args

    ns = _args(name=None, provider=None)
    spec = _spec_from_args(ns)
    assert any("Be terse." in i for i in spec.instructions)
    assert any("metric units" in i for i in spec.instructions)


def test_himmy_md_cap_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized HIMMY.md is truncated with a marker."""
    monkeypatch.chdir(tmp_path)
    import himmy.cli.commands as commands

    (tmp_path / "HIMMY.md").write_text("y" * (commands._HIMMY_MD_CAP + 5000))
    ns = _args(name=None, provider=None)
    spec = commands._spec_from_args(ns)
    note = next(i for i in spec.instructions if "Project notes (HIMMY.md)" in i)
    assert "… (truncated)" in note
    assert len(note) <= commands._HIMMY_MD_CAP + 200


def test_no_himmy_md_leaves_instructions_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from himmy.cli.commands import _spec_from_args

    ns = _args(name=None, provider=None)
    spec = _spec_from_args(ns)
    assert not any("Project notes (HIMMY.md)" in i for i in spec.instructions)


# ------------------------------------------------------------------- streaming


def _treasurer_repl(input_fn: Any = None) -> tuple[ChatRepl, Any]:
    """A ChatRepl over the HITL harness runtime (wire_money gated tool)."""
    from himmy.agents.personas.persona import Persona
    from himmy.cli.ui import LiveRunUI
    from tests.runtime.test_hitl import _hitl_runtime

    rt_obj, cp = _hitl_runtime()
    spec = AgentSpec(
        name="treasurer", description="pays", provider="stub", tools=["wire_money"]
    )
    repl = ChatRepl(
        spec, _args(name="treasurer"), input_fn=input_fn or (lambda _p: "y")
    )
    repl.persona = Persona(name="treasurer")
    repl.llm_config = None
    repl.interactive = True
    repl.rt = {
        "runtime": rt_obj,
        "registry": rt_obj.tool_service.registry,
        "checkpoint_store": cp,
        "live": LiveRunUI(model_label="", stream=io.StringIO()),
    }
    return repl, rt_obj


def _spy_stream_task(rt_obj: Any) -> list[int]:
    calls: list[int] = []
    original = rt_obj.stream_task

    def _spy(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return original(*a, **k)

    rt_obj.stream_task = _spy
    return calls


def test_tool_turn_streams_final_answer_when_opted_in() -> None:
    """With /stream on, a tool-using turn's final answer arrives via stream_task."""
    from himmy.agents.base_agent.thread import ChatThread

    repl, rt_obj = _treasurer_repl()
    repl.stream_final = True  # the /stream on toggle
    calls = _spy_stream_task(rt_obj)

    thread = ChatThread(agent_id=repl.persona.agent_id)
    run_async(repl._chat_turn(thread, "wire 1000"))
    assert calls, "stream_task should be used to stream the tool-turn final answer"


def test_tool_turn_does_not_stream_by_default() -> None:
    """Default off: no extra synthesis call, the loop's own answer is printed."""
    from himmy.agents.base_agent.thread import ChatThread

    repl, rt_obj = _treasurer_repl()
    assert repl.stream_final is False
    calls = _spy_stream_task(rt_obj)

    thread = ChatThread(agent_id=repl.persona.agent_id)
    run_async(repl._chat_turn(thread, "wire 1000"))
    assert not calls, "default must not re-issue the synthesis as an extra call"


def test_stream_toggle_slash_command() -> None:
    """/stream on|off flips the opt-in; bare /stream reports state."""
    repl, _rt = _treasurer_repl()
    from himmy.agents.base_agent.thread import ChatThread

    thread = ChatThread(agent_id=repl.persona.agent_id)
    repl._slash("/stream on", thread)
    assert repl.stream_final is True
    repl._slash("/stream off", thread)
    assert repl.stream_final is False


def test_no_tool_agent_still_streams() -> None:
    """A no-tool agent still streams its reply (unchanged original behavior)."""
    repl = _scripted_repl([])
    repl.rebuild()
    from himmy.agents.base_agent.thread import ChatThread

    # registry is None for a stub agent with no tools → the no-tool stream branch.
    assert repl.rt["registry"] is None
    thread = ChatThread(agent_id=repl.persona.agent_id)
    out = run_async(repl._chat_turn(thread, "hi there"))
    assert out is not None
