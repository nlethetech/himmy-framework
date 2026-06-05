"""Tests for the `agentic` pack: ask_human, scratchpad, todo."""

from __future__ import annotations

from typing import Any

from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from himmy.toolkit import ToolkitConfig, register_packs, set_human_responder
from himmy.toolkit.agentic import AGENTIC_TOOL_NAMES
from tests.conftest import run_async


def _service() -> ToolService:
    registry = ToolRegistry()
    register_packs(registry, ["agentic"], ToolkitConfig())
    return ToolService(registry)


def _call(service: ToolService, name: str, **args: Any) -> Any:
    res = run_async(service.execute(ToolInvocation(tool_name=name, args=args)))
    assert res.outcome == "success", res
    return res.result


def test_pack_registers_all_tools() -> None:
    registry = ToolRegistry()
    register_packs(registry, ["agentic"], ToolkitConfig())
    assert set(AGENTIC_TOOL_NAMES) <= {t.name for t in registry.list()}


def test_ask_human_uses_injected_responder() -> None:
    set_human_responder(lambda q: f"answer to: {q}")
    try:
        out = _call(_service(), "ask_human", question="proceed?")
    finally:
        set_human_responder(None)
    assert out["answered"] is True
    assert out["answer"] == "answer to: proceed?"


def test_ask_human_async_responder() -> None:
    async def responder(q: str) -> str:
        return "async ok"

    set_human_responder(responder)
    try:
        out = _call(_service(), "ask_human", question="x")
    finally:
        set_human_responder(None)
    assert out["answer"] == "async ok"


def test_ask_human_non_interactive_reports_unanswered() -> None:
    # No responder + no TTY (pytest captures stdin) → answered: false, not a hang.
    set_human_responder(None)
    out = _call(_service(), "ask_human", question="anyone?")
    assert out["answered"] is False
    assert out["answer"] == ""


def test_scratchpad_set_and_get_roundtrip() -> None:
    service = _service()
    _call(service, "scratchpad_set", key="plan", value="step 1, step 2")
    one = _call(service, "scratchpad_get", key="plan")
    assert one["value"] == "step 1, step 2"
    _call(service, "scratchpad_set", key="risks", value="none")
    every = _call(service, "scratchpad_get")
    assert every["notes"] == {"plan": "step 1, step 2", "risks": "none"}


def test_scratchpad_get_missing_key_is_none() -> None:
    assert _call(_service(), "scratchpad_get", key="nope")["value"] is None


def test_todo_write_takes_flat_strings_then_complete_tracks_status() -> None:
    service = _service()
    written = _call(service, "todo_write", items=["research", "draft", "review"])
    assert written["count"] == 3
    assert written["completed"] == 0
    assert all(it["status"] == "pending" for it in written["items"])
    done = _call(service, "todo_complete", item="research")
    assert done["matched"] is True
    read = _call(service, "todo_read")
    assert read["completed"] == 1
    assert read["items"][0]["status"] == "completed"


def test_todo_complete_unmatched_is_reported() -> None:
    service = _service()
    _call(service, "todo_write", items=["alpha"])
    assert _call(service, "todo_complete", item="zzz")["matched"] is False


def test_todo_write_replaces_the_list() -> None:
    service = _service()
    _call(service, "todo_write", items=["old"])
    _call(service, "todo_write", items=["new1", "new2"])
    assert _call(service, "todo_read")["count"] == 2


def test_state_is_isolated_per_registry() -> None:
    a, b = _service(), _service()
    _call(a, "scratchpad_set", key="k", value="in-a")
    # b has its own scratchpad — a's note must not leak.
    assert _call(b, "scratchpad_get", key="k")["value"] is None
