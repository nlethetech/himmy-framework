"""ContextCompactor planning: budget trigger + the three invariants."""

from __future__ import annotations

from himmy.agents.base_agent.thread import Message, MessageRole
from himmy.runtime.compaction import ContextCompactor, estimate_tokens


def _m(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


def _big(n: int) -> str:
    return "x" * (n * 4)  # ~n tokens


def test_under_budget_is_a_noop() -> None:
    msgs = [_m(MessageRole.SYSTEM, "p"), _m(MessageRole.USER, "hi")]
    plan = ContextCompactor(max_tokens=1000).plan(msgs)
    assert plan.should_compact is False
    assert plan.reason == "under budget"


def test_compacts_the_middle_keeping_head_and_tail() -> None:
    msgs = [
        _m(MessageRole.SYSTEM, _big(50)),  # head — protected
        _m(MessageRole.USER, _big(200)),  # ┐ summarized
        _m(MessageRole.ASSISTANT, _big(200)),  # │
        _m(MessageRole.USER, _big(200)),  # ┘
        _m(MessageRole.ASSISTANT, _big(10)),  # ┐ kept (recent 2)
        _m(MessageRole.USER, _big(10)),  # ┘
    ]
    plan = ContextCompactor(max_tokens=300, keep_recent=2).plan(msgs)
    assert plan.should_compact is True
    assert plan.head_count == 1  # the system message
    assert plan.summarize_start == 1
    assert plan.summarize_end == 4  # tail starts at index 4 (last 2 kept)
    assert len(plan.summarize) == 3


def test_never_summarizes_the_system_message() -> None:
    msgs = [_m(MessageRole.SYSTEM, _big(500))] + [
        _m(MessageRole.USER, _big(500)) for _ in range(4)
    ]
    plan = ContextCompactor(max_tokens=300, keep_recent=1).plan(msgs)
    assert plan.should_compact is True
    assert plan.head_count == 1
    assert all(m.role == MessageRole.USER for m in plan.summarize)


def test_tail_boundary_never_orphans_a_tool_result() -> None:
    # If keep_recent would make the tail start with a TOOL message (whose ASSISTANT
    # tool_call is in the summarized span), the boundary snaps back to keep the pair.
    msgs = [
        _m(MessageRole.SYSTEM, _big(50)),
        _m(MessageRole.USER, _big(200)),
        _m(MessageRole.ASSISTANT, _big(200)),  # made a tool call
        _m(MessageRole.ASSISTANT, _big(200)),  # tool call
        _m(MessageRole.TOOL, _big(200)),  # its result  ← naive tail would start here
        _m(MessageRole.ASSISTANT, _big(10)),
    ]
    # keep_recent=2 → naive split=4 (a TOOL message); must snap back to 3.
    plan = ContextCompactor(max_tokens=300, keep_recent=2).plan(msgs)
    assert plan.should_compact is True
    assert plan.summarize_end == 3  # snapped back off the tool message
    # The kept tail starts with the assistant that owns the tool result, not the orphan.
    assert msgs[plan.tail_start].role == MessageRole.ASSISTANT


def test_too_little_to_summarize_is_a_noop() -> None:
    # Over budget but only one non-head, non-recent message → nothing worth summarizing.
    msgs = [
        _m(MessageRole.SYSTEM, _big(500)),
        _m(MessageRole.USER, _big(500)),
        _m(MessageRole.ASSISTANT, _big(500)),
    ]
    plan = ContextCompactor(max_tokens=300, keep_recent=2, min_summarize=2).plan(msgs)
    assert plan.should_compact is False
    assert plan.reason == "too little to summarize"


def test_render_span_flattens_roles_and_content() -> None:
    span = [_m(MessageRole.USER, "how many ducks?"), _m(MessageRole.TOOL, "12")]
    text = ContextCompactor().render_span(span)
    assert "user: how many ducks?" in text
    assert "tool: 12" in text


def test_estimate_tokens_is_length_based() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


# --- runtime apply path (deterministic, via a fake summarizer) -----------------------

from himmy.agents.base_agent.thread import ChatThread  # noqa: E402
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.runtime.single_agent import SingleAgentRuntime  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.service import InferenceService  # noqa: E402
from tests.conftest import run_async  # noqa: E402


class _FixedSummaryManager:
    """Returns a fixed summary text for every request (a stand-in summarizer)."""

    def __init__(self, summary: str) -> None:
        self._summary = summary
        self.provider_name = "fake"

    def resolve(self, model_key: str) -> str:
        return "fake:model"

    async def generate(self, request: object) -> InferenceResponse:
        return InferenceResponse(
            request_id="r",
            status=InferenceStatus.SUCCESS,
            output_text=self._summary,
        )


def _over_budget_thread() -> ChatThread:
    t = ChatThread()
    t.append_message(_m(MessageRole.SYSTEM, _big(20)))
    t.append_message(_m(MessageRole.USER, _big(300)))
    t.append_message(_m(MessageRole.ASSISTANT, _big(300)))
    t.append_message(_m(MessageRole.USER, _big(10)))
    t.append_message(_m(MessageRole.ASSISTANT, _big(10)))
    return t


def _runtime(summary: str, events: list) -> SingleAgentRuntime:
    async def on_event(e):
        events.append(e)

    return SingleAgentRuntime(
        inference_service=InferenceService(_FixedSummaryManager(summary)),
        on_event=on_event,
    )


def test_runtime_applies_compaction_and_emits_event() -> None:
    events: list = []
    rt = _runtime("short summary", events)
    thread = _over_budget_thread()
    ctx = {"compaction_spec": {"max_tokens": 100, "keep_recent": 2}}
    run_async(rt._maybe_compact(Persona(name="a"), thread, ctx, "trace", None))
    # The two big middle messages collapsed into one system summary.
    assert any(m.metadata.get("compacted") for m in thread.messages)
    assert "short summary" in thread.messages[1].content
    assert len(thread.messages) == 4  # system + summary + 2 recent
    assert any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)


def test_runtime_skips_when_summary_would_not_shrink() -> None:
    events: list = []
    rt = _runtime(_big(5000), events)  # a summary far larger than the span
    thread = _over_budget_thread()
    before = list(thread.messages)
    ctx = {"compaction_spec": {"max_tokens": 100, "keep_recent": 2}}
    run_async(rt._maybe_compact(Persona(name="a"), thread, ctx, "trace", None))
    assert thread.messages == before  # untouched — the guard refused a net-negative
    assert not any(e.event_type.value == "CONTEXT_COMPACTED" for e in events)


def test_runtime_noop_without_compaction_spec() -> None:
    events: list = []
    rt = _runtime("x", events)
    thread = _over_budget_thread()
    before = list(thread.messages)
    run_async(rt._maybe_compact(Persona(name="a"), thread, {}, "trace", None))
    assert thread.messages == before
