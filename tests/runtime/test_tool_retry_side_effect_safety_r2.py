"""Red-team round-2: timeout-retry must not re-fire more side-effecting shapes (eff-p1 safe-r2).

Round 1 closed read-verb-PREFIXED writers whose suffix was already in ``_WRITE_VERBS``
(``fetch_and_delete``). Round 2 closes two more confirmed holes:

* a read verb paired with an action verb the old deny-list MISSED
  (``get_and_charge``/``list_and_publish``/``describe_and_deploy``) — now barriered by the
  expanded ``_WRITE_VERBS`` so a TIMEOUT never re-fires the charge/publish/deploy;
* an HTTP GET tool with a server-side side effect (an analytics beacon, ``GET /trigger``):
  its method-derived read-only is a parallelism hint only, NOT an authoritative assertion,
  so a TIMEOUT does NOT re-fire it. Only an EXPLICIT ``read_only=True`` re-fires a GET.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.models import HttpToolConfig
from himmy.services.tools.registry import (
    ToolRegistry,
    register_http_tool,
    register_local_tool,
)
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


def _runtime(
    registry: ToolRegistry, *, transport: httpx.MockTransport | None = None
) -> tuple[SingleAgentRuntime, list[RunEvent]]:
    events: list[RunEvent] = []

    async def collect(event: RunEvent) -> None:
        events.append(event)

    tool_service = (
        ToolService(registry, http_client=httpx.AsyncClient(transport=transport))
        if transport is not None
        else ToolService(registry)
    )
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=StorageService(),
        tool_service=tool_service,
        on_event=collect,
    )
    return rt, events


def _timeout_handler() -> tuple[Any, dict[str, int]]:
    calls = {"count": 0}

    async def _h(args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        await asyncio.sleep(0.5)  # > the 0.05s tool timeout
        return {"ok": True}

    return _h, calls


def _task(tool: str) -> Task:
    return Task(
        title="t",
        prompt="do it",
        context={
            "tool_names": [tool],
            "response_format": "AUTO_TOOLS",
            "tool_retry_backoff_seconds": 0.0,
        },
    )


def _retries(events: list[RunEvent]) -> list[RunEvent]:
    return [
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and "transient_retry" in e.payload
    ]


def test_read_prefixed_action_verb_not_retried_on_timeout() -> None:
    """A read verb + an action verb the deny-list previously missed fires exactly once."""
    for name in ("get_and_charge", "list_and_publish", "describe_and_deploy"):
        registry = ToolRegistry()
        handler, calls = _timeout_handler()
        register_local_tool(
            registry,
            name=name,
            handler=handler,
            args_json_schema={"type": "object", "properties": {}},
            timeout_seconds=0.05,
        )
        rt, events = _runtime(registry)
        run_async(rt.run_task_detailed(Persona(name="a"), _task(name)))
        assert calls["count"] == 1, name  # never re-fired on the timeout
        assert _retries(events) == [], name


def test_side_effecting_get_not_retried_on_timeout() -> None:
    """A GET beacon (method-derived read-only, unset flag) is NOT re-fired on TIMEOUT."""
    calls = {"n": 0}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="ping_beacon",  # GET -> read_only=True (parallel hint) but NOT authoritative
        http_config=HttpToolConfig(base_url="https://svc", method="GET", path_template="/track"),
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
    )
    rt, events = _runtime(registry, transport=httpx.MockTransport(transport_handler))
    run_async(rt.run_task_detailed(Persona(name="a"), _task("ping_beacon")))

    assert calls["n"] == 1  # the timed-out GET was not duplicated
    assert _retries(events) == []


def test_authoritative_read_only_get_is_retried_on_timeout() -> None:
    """An EXPLICIT read_only=True GET (author asserts no side effect) IS retried on timeout."""
    calls = {"n": 0}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="get_weather",
        http_config=HttpToolConfig(base_url="https://svc", method="GET", path_template="/w"),
        args_json_schema={"type": "object", "properties": {}},
        timeout_seconds=0.05,
        read_only=True,  # AUTHORITATIVE assertion of no side effect
    )
    rt, events = _runtime(registry, transport=httpx.MockTransport(transport_handler))
    run_async(rt.run_task_detailed(Persona(name="a"), _task("get_weather")))

    # Both retry layers permit a re-fire: initial + DEFAULT_TOOL_RETRY_ATTEMPTS.
    assert calls["n"] > 1  # re-fired because the author asserted read-only
    assert len(_retries(events)) >= 1
