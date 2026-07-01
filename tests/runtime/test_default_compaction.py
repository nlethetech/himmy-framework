"""eff-p0 #3: auto-compaction is DEFAULT-ON with a conservative high budget.

A multi-turn agent re-sends its whole history every turn (O(turns^2) cumulative
tokens). Compaction used to run only when a caller opted in via
``ctx['compaction_spec']`` / ``AgentSpec.compact_context`` — which almost nobody set —
so long runs shipped the full uncompressed history forever. The runtime now falls back
to a default policy: an explicit spec still wins, but with NO spec a conservative
default budget kicks in, fully disable-able / tunable via ``HIMMY_AUTO_COMPACT*`` env.

These tests cover: the default triggers on a large thread and keeps it bounded; a
typical short thread is byte-unchanged (never crosses the budget); disabling via env
restores the old opt-in-only behaviour; the summary-only-if-smaller guard still holds
on the default path.
"""

from __future__ import annotations

import pytest

from himmy.agents.base_agent.thread import ChatThread, MessageRole
from himmy.agents.personas.persona import Persona
from himmy.runtime import single_agent as sa
from himmy.runtime.compaction import estimate_tokens
from himmy.runtime.single_agent import (
    _AUTO_COMPACT_TOKENS_DEFAULT,
    SingleAgentRuntime,
    _auto_compact_default_spec,
)
from himmy.services.inference.service import InferenceService
from tests.conftest import run_async
from tests.runtime.test_compaction import (
    _big,
    _FixedSummaryManager,
    _m,
    _over_budget_thread,
)


def _runtime(summary: str, events: list) -> SingleAgentRuntime:
    async def on_event(e):
        events.append(e)

    return SingleAgentRuntime(
        inference_service=InferenceService(_FixedSummaryManager(summary)),
        on_event=on_event,
    )


def _huge_thread() -> ChatThread:
    """A thread comfortably over the DEFAULT budget (~24k tokens), no explicit spec."""
    t = ChatThread()
    t.append_message(_m(MessageRole.SYSTEM, _big(200)))  # protected head
    # Four fat middle turns (~10k tokens each) → well over the default budget.
    for _ in range(4):
        t.append_message(_m(MessageRole.USER, _big(10_000)))
        t.append_message(_m(MessageRole.ASSISTANT, _big(10_000)))
    # A couple of small recent turns.
    t.append_message(_m(MessageRole.USER, _big(20)))
    t.append_message(_m(MessageRole.ASSISTANT, _big(20)))
    return t


# --- the default spec itself --------------------------------------------------------


def test_default_spec_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_AUTO_COMPACT", raising=False)
    monkeypatch.delenv("HIMMY_AUTO_COMPACT_TOKENS", raising=False)
    monkeypatch.delenv("HIMMY_AUTO_COMPACT_KEEP", raising=False)
    spec = _auto_compact_default_spec()
    assert spec == {"max_tokens": _AUTO_COMPACT_TOKENS_DEFAULT, "keep_recent": 8}


@pytest.mark.parametrize("val", ["0", "false", "no", "OFF", "False"])
def test_default_spec_disabled_via_env(val: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_AUTO_COMPACT", val)
    assert _auto_compact_default_spec() is None


def test_default_spec_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_AUTO_COMPACT_TOKENS", "5000")
    monkeypatch.setenv("HIMMY_AUTO_COMPACT_KEEP", "3")
    assert _auto_compact_default_spec() == {"max_tokens": 5000, "keep_recent": 3}


def test_default_spec_bad_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIMMY_AUTO_COMPACT_TOKENS", "not-an-int")
    monkeypatch.setenv("HIMMY_AUTO_COMPACT_KEEP", "also-bad")
    assert _auto_compact_default_spec() == {
        "max_tokens": _AUTO_COMPACT_TOKENS_DEFAULT,
        "keep_recent": 8,
    }


# --- the runtime apply path with the default (no explicit compaction_spec) -----------


def test_default_compacts_a_huge_thread_and_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIMMY_AUTO_COMPACT", raising=False)
    events: list = []
    rt = _runtime("distilled briefing", events)
    thread = _huge_thread()
    before_tokens = sum(estimate_tokens(m.content) for m in thread.messages)

    # No compaction_spec in ctx → the default policy must kick in.
    did = run_async(rt._maybe_compact(Persona(name="a"), thread, {}, "trace", None))

    assert did is True
    assert any(m.metadata.get("compacted") for m in thread.messages)
    after_tokens = sum(estimate_tokens(m.content) for m in thread.messages)
    # Compaction actually shrank the re-sent context (bounded, not O(turns^2) growth).
    # A single pass keeps `keep_recent` recent turns verbatim, so it need not drop below
    # the budget in one shot — but it MUST net-shrink, replacing summarized middle turns
    # with a compact summary.
    assert after_tokens < before_tokens
    assert any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)
    # The system head is never dropped.
    assert thread.messages[0].role == MessageRole.SYSTEM


def test_default_leaves_a_short_thread_byte_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIMMY_AUTO_COMPACT", raising=False)
    events: list = []
    rt = _runtime("summary", events)
    # A typical short run (~640 tokens) is far under the conservative default budget.
    thread = _over_budget_thread()
    before = list(thread.messages)

    did = run_async(rt._maybe_compact(Persona(name="a"), thread, {}, "trace", None))

    assert did is False
    assert thread.messages == before  # byte-identical: no summary injected
    assert not any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)


def test_disabling_env_restores_old_noop_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_AUTO_COMPACT", "0")
    events: list = []
    rt = _runtime("summary", events)
    thread = _huge_thread()  # would compact if the default were on
    before = list(thread.messages)

    did = run_async(rt._maybe_compact(Persona(name="a"), thread, {}, "trace", None))

    assert did is False
    assert thread.messages == before  # untouched — default disabled, no explicit spec
    assert not any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)


def test_explicit_spec_still_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with the default disabled, an explicit spec runs (explicit always wins).
    monkeypatch.setenv("HIMMY_AUTO_COMPACT", "0")
    events: list = []
    rt = _runtime("short summary", events)
    thread = _over_budget_thread()
    ctx = {"compaction_spec": {"max_tokens": 100, "keep_recent": 2}}

    did = run_async(rt._maybe_compact(Persona(name="a"), thread, ctx, "trace", None))

    assert did is True
    assert any(m.metadata.get("compacted") for m in thread.messages)


def test_default_path_honours_summary_only_if_smaller_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A verbose summary bigger than the span it replaces must be refused, even on the
    # default path — compaction must never GROW the context.
    monkeypatch.setenv("HIMMY_AUTO_COMPACT_TOKENS", "300")  # low budget → default triggers
    monkeypatch.delenv("HIMMY_AUTO_COMPACT", raising=False)
    events: list = []
    rt = _runtime(_big(5000), events)  # summary far larger than the span
    thread = _over_budget_thread()
    before = list(thread.messages)

    did = run_async(rt._maybe_compact(Persona(name="a"), thread, {}, "trace", None))

    assert did is False
    assert thread.messages == before  # guard refused the net-negative rewrite
    assert not any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)


def test_module_default_constant_is_conservative() -> None:
    # A guardrail on the DECISION: the default budget must stay high enough that ordinary
    # short runs never trip it (correctness > shaving tokens off runs that won't overflow).
    assert _AUTO_COMPACT_TOKENS_DEFAULT >= 16000
    assert sa._AUTO_COMPACT_KEEP_DEFAULT >= 4
