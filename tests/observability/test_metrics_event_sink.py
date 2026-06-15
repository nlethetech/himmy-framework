"""MetricsEventSink + service-layer Prometheus instruments.

Asserts that the run-event stream is translated into bounded-cardinality
counters/histograms (inference calls/failures, tokens, cost, tool calls,
guardrail blocks, run outcomes), that the dispatcher records its own
queue/latency/reap series, and that ``GET /metrics`` exposes the new families.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.api.app import create_app
from himmy.application import RecommendationAppService, RunAppService
from himmy.application.dispatcher import LocalProviderProbe, RunDispatcher
from himmy.core.events import EventType, RunEvent
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.observability import metrics as metrics_mod
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    """A fresh metrics registry per test so counts are deterministic."""
    metrics_mod.reset_registry()


def test_event_sink_translates_run_events_to_counters() -> None:
    sink = metrics_mod.MetricsEventSink()
    reg = metrics_mod.get_registry()

    async def feed() -> None:
        await sink.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED))
        await sink.append_event(
            RunEvent(event_type=EventType.INFERENCE_REQUESTED, payload={})
        )
        await sink.append_event(
            RunEvent(
                event_type=EventType.INFERENCE_SUCCEEDED,
                latency_ms=12.0,
                cost=0.0025,
                payload={"input_tokens": 100, "output_tokens": 40},
            )
        )
        await sink.append_event(RunEvent(event_type=EventType.TOOL_CALLED))
        await sink.append_event(RunEvent(event_type=EventType.TOOL_COMPLETED))
        await sink.append_event(
            RunEvent(
                event_type=EventType.GUARDRAIL_APPLIED,
                payload={"stage": "output", "blocked": True, "redacted": False},
            )
        )
        await sink.append_event(
            RunEvent(
                event_type=EventType.AGENT_RUN_FINISHED, payload={"status": "success"}
            )
        )

    run_async(feed())

    assert reg.inference_requests_total.value(("false",)) == 1.0
    assert reg.inference_input_tokens_total.value() == 100.0
    assert reg.inference_output_tokens_total.value() == 40.0
    assert reg.inference_cost_usd_total.value() == pytest.approx(0.0025)
    assert reg.inference_tokens.count() == 1.0
    assert reg.inference_duration_seconds.count() == 1.0
    assert reg.tool_calls_total.value(("called",)) == 1.0
    assert reg.tool_calls_total.value(("completed",)) == 1.0
    assert reg.guardrail_blocks_total.value(("output", "blocked")) == 1.0
    assert reg.agent_runs_total.value() == 1.0
    assert reg.agent_run_outcomes_total.value(("success",)) == 1.0


def test_inference_failure_increments_failure_counter() -> None:
    sink = metrics_mod.MetricsEventSink()
    reg = metrics_mod.get_registry()

    async def feed() -> None:
        await sink.append_event(
            RunEvent(
                event_type=EventType.INFERENCE_FAILED,
                latency_ms=5.0,
                error="boom",
                payload={"error_code": "TIMEOUT"},
            )
        )

    run_async(feed())
    assert reg.inference_failures_total.value() == 1.0
    assert reg.inference_duration_seconds.count() == 1.0


def test_unknown_run_status_clamps_to_other_label() -> None:
    """A forged/garbage status must not create an unbounded label series."""
    sink = metrics_mod.MetricsEventSink()
    reg = metrics_mod.get_registry()

    async def feed() -> None:
        await sink.append_event(
            RunEvent(
                event_type=EventType.AGENT_RUN_FINISHED,
                payload={"status": "definitely-not-a-real-status-xyz"},
            )
        )

    run_async(feed())
    assert reg.agent_run_outcomes_total.value(("other",)) == 1.0
    # The raw forged value never became a label.
    assert reg.agent_run_outcomes_total.value(("definitely-not-a-real-status-xyz",)) == 0.0


def test_full_run_through_runtime_increments_metrics() -> None:
    """A real (stub) agent run feeds the process-wide sink via runtime _emit."""
    reg = metrics_mod.get_registry()
    registry = EntityRegistry()
    storage = StorageService()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )

    async def go() -> None:
        await runtime.run_agent_loop(
            persona=Persona(name="A"),
            task=Task(title="t", prompt="hello"),
        )

    run_async(go())

    assert reg.agent_runs_total.value() == 1.0
    # The run drove at least one inference and stamped a terminal outcome.
    assert reg.inference_requests_total.value(("false",)) >= 1.0
    assert (
        reg.agent_run_outcomes_total.value(("success",))
        + reg.agent_run_outcomes_total.value(("failed",))
        >= 1.0
    )


def _durable_stack(db_path: str) -> tuple[SqliteStorageService, RunAppService]:
    storage = SqliteStorageService(db_path)
    typed = cast(StorageService, storage)
    registry = EntityRegistry()
    context = ContextService(storage_service=typed, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    run_app = RunAppService(
        runtime=runtime,
        storage=typed,
        entity_registry=registry,
        recommendation_app=RecommendationAppService(storage=typed),
    )
    return storage, run_app


def test_dispatcher_records_claim_and_run_duration() -> None:
    """Claiming + executing a run bumps the dispatcher claim + duration series."""
    reg = metrics_mod.get_registry()
    db = str(Path(tempfile.mkdtemp()) / "q.db")
    storage, run_app = _durable_stack(db)

    async def go() -> RunRecord | None:
        run_app.enable_dispatch()  # switch to queue mode so the create enqueues
        # Enqueue a fresh queued run so the dispatcher has something to claim.
        run2 = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="A"),
            task=Task(title="t", prompt="hi again"),
        )
        assert run2.status == RunStatus.QUEUED
        dispatcher = RunDispatcher(
            run_app,
            max_concurrency=2,
            probe=LocalProviderProbe(probe=lambda: True),
        )
        dispatcher.start()
        # Poll until the queued run reaches a terminal state.
        for _ in range(200):
            cur = await storage.get_run(run2.run_id)
            if cur is not None and cur.status in (
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.PARKED,
            ):
                break
            import asyncio

            await asyncio.sleep(0.02)
        await dispatcher.stop()
        return await storage.get_run(run2.run_id)

    run_async(go())

    assert reg.dispatcher_claims_total.value() >= 1.0
    assert reg.dispatcher_run_duration_seconds.count() >= 1.0
    # The in-flight gauge returns to zero after the worker finishes.
    assert reg.dispatcher_in_flight.value() == 0.0


def _clean_app_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for var in (
        "HIMMY_MULTI_TENANT",
        "HIMMY_AUTH_MODE",
        "HIMMY_INTERNAL_API_KEY",
        "HIMMY_DATABASE_URL",
        "HIMMY_DURABLE_STORAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_metrics_endpoint_exposes_new_service_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _clean_app_env(monkeypatch, tmp_path)
    metrics_mod.reset_registry()
    app = create_app()
    client = TestClient(app)

    body = client.get("/metrics").text
    for family in (
        "himmy_inference_requests_total",
        "himmy_inference_failures_total",
        "himmy_inference_retries_total",
        "himmy_inference_input_tokens_total",
        "himmy_inference_output_tokens_total",
        "himmy_inference_cost_usd_total",
        "himmy_inference_tokens",
        "himmy_tool_calls_total",
        "himmy_guardrail_blocks_total",
        "himmy_agent_runs_total",
        "himmy_agent_run_outcomes_total",
        "himmy_dispatcher_in_flight_runs",
        "himmy_dispatcher_claims_total",
        "himmy_dispatcher_reaps_total",
        "himmy_dispatcher_run_duration_seconds",
    ):
        assert f"# HELP {family}" in body, family
