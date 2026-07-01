"""Model-facing tool-result cap (eff-p0 #2).

A tool that returns a large blob (a scraped page, a file dump, wide DB rows) used
to write its ENTIRE serialized output into the thread; because the whole thread is
re-sent every turn, that blob was re-billed on every later turn. ``_append_tool_messages``
now caps the MODEL-FACING copy at ``_TOOL_RESULT_MODEL_MAX`` chars (env
``HIMMY_TOOL_RESULT_MODEL_MAX``, default 6000) with a ``…[truncated N chars]`` marker,
while the FULL result still reaches the tool-return record + the TOOL_COMPLETED event
so observers lose nothing. These tests pin:

1. a large result is capped in the thread with the marker;
2. the full result still reaches the TOOL_COMPLETED event (bounded only by the
   separate, larger observability cap);
3. a per-tool opt-out (metadata flag or env allowlist) preserves the full output on
   the thread;
4. the cap disabled (``_TOOL_RESULT_MODEL_MAX == 0``) restores the old uncapped
   behaviour.

Offline-first: a scripted manager + an in-memory thread; nothing to skip.
"""

from __future__ import annotations

from typing import Any

import himmy.runtime.single_agent as single_agent
from himmy.agents.base_agent.thread import ChatThread, MessageRole
from himmy.core.events import EventType, RunEvent
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.tools.models import ToolBackendKind, ToolDefinition
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async


class _NullManager:
    """A no-op manager; the runtime is built only to reach ``_append_tool_messages``."""

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:  # pragma: no cover
        return InferenceResponse(
            request_id=request.request_id, status=InferenceStatus.SUCCESS
        )


class _StubToolService:
    """Just enough surface for ``_tool_result_uncapped`` to reach a registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry


def _runtime(
    *, tool_service: _StubToolService | None = None
) -> tuple[SingleAgentRuntime, list[RunEvent]]:
    events: list[RunEvent] = []
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_NullManager()),
        tool_service=tool_service,
        on_event=lambda e: events.append(e),
    )
    return runtime, events


def _response(tool_name: str, content: Any) -> InferenceResponse:
    return InferenceResponse(
        request_id="r1",
        status=InferenceStatus.SUCCESS,
        tool_calls=[ToolCallRecord(tool_call_id="tc1", tool_name=tool_name, args={})],
        tool_returns=[
            ToolReturnRecord(tool_call_id="tc1", tool_name=tool_name, content=content)
        ],
    )


def _append(runtime: SingleAgentRuntime, response: InferenceResponse) -> ChatThread:
    thread = ChatThread()
    run_async(
        runtime._append_tool_messages(
            thread,
            response,
            request_id="r1",
            trace_id="t1",
            agent_id=None,
        )
    )
    return thread


def _tool_message_content(thread: ChatThread) -> str:
    tool_msgs = [m for m in thread.messages if m.role == MessageRole.TOOL]
    assert len(tool_msgs) == 1
    return tool_msgs[0].content


def _completed_result(events: list[RunEvent]) -> str:
    done = [e for e in events if e.event_type == EventType.TOOL_COMPLETED]
    assert len(done) == 1
    return done[0].payload["result"]


def test_large_result_is_capped_in_the_thread_with_marker() -> None:
    runtime, _events = _runtime()
    blob = "A" * (single_agent._TOOL_RESULT_MODEL_MAX + 5000)
    thread = _append(runtime, _response("scrape", blob))

    content = _tool_message_content(thread)
    # The model sees only the capped prefix plus an explicit, self-describing marker.
    assert len(content) < len(blob)
    assert content.startswith("A" * single_agent._TOOL_RESULT_MODEL_MAX)
    assert "…[truncated 5000 chars]" in content


def test_full_result_preserved_on_record_and_event_not_the_model_copy() -> None:
    # The FULL result must survive for observers. The tool-return RECORD keeps every
    # byte (uncapped), and the TOOL_COMPLETED event carries the result derived from
    # the FULL content bounded ONLY by its own (separate, unchanged) observability
    # cap — crucially NOT bounded by the smaller/other model cap.
    runtime, events = _runtime()
    blob = "B" * (single_agent._TOOL_RESULT_MODEL_MAX + 5000)
    response = _response("scrape", blob)
    thread = _append(runtime, response)

    # Thread (model-facing) is capped with the marker.
    assert len(_tool_message_content(thread)) < len(blob)
    assert "truncated" in _tool_message_content(thread)

    # Record: full result, untouched.
    assert response.tool_returns[0].content == blob

    # Event: derived from the FULL blob, bounded only by the event cap — i.e. it
    # reflects the original content, not the model-truncated copy (no "truncated"
    # marker leaks in, and it carries the leading event-cap window of the blob).
    result = _completed_result(events)
    assert "truncated" not in result
    assert result.startswith("B" * (single_agent._TOOL_RESULT_EVENT_MAX - 1))


def test_metadata_opt_out_preserves_full_output() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="signed_artifact",
            kind=ToolBackendKind.LOCAL,
            metadata={"model_result_uncapped": True},
        )
    )
    runtime, _events = _runtime(tool_service=_StubToolService(registry))
    blob = "C" * (single_agent._TOOL_RESULT_MODEL_MAX + 5000)
    thread = _append(runtime, _response("signed_artifact", blob))

    # Opt-out tool: the model sees every byte, no marker.
    content = _tool_message_content(thread)
    assert content == blob
    assert "truncated" not in content


def test_env_allowlist_opt_out_preserves_full_output(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_TOOL_RESULT_UNCAPPED", "  other , essential ")
    runtime, _events = _runtime()
    blob = "D" * (single_agent._TOOL_RESULT_MODEL_MAX + 5000)
    thread = _append(runtime, _response("essential", blob))

    assert _tool_message_content(thread) == blob


def test_cap_disabled_restores_old_behaviour(monkeypatch: Any) -> None:
    # env HIMMY_TOOL_RESULT_MODEL_MAX=0 disables the cap; the whole blob rides the
    # thread again, exactly like the pre-C5 runtime.
    monkeypatch.setattr(single_agent, "_TOOL_RESULT_MODEL_MAX", 0)
    runtime, _events = _runtime()
    blob = "E" * 20000
    thread = _append(runtime, _response("scrape", blob))

    content = _tool_message_content(thread)
    assert content == blob
    assert "truncated" not in content


def test_small_result_is_left_untouched() -> None:
    runtime, _events = _runtime()
    thread = _append(runtime, _response("probe", {"ok": True}))
    content = _tool_message_content(thread)
    assert "truncated" not in content
    assert "ok" in content
