"""Tests for the inference kernel: stub generation, formats, retries, batching."""

from __future__ import annotations

import asyncio
import time

import pytest

from himmy.services.inference import (
    BatchInferenceRequest,
    BoundTool,
    InferenceError,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    InferenceStatus,
    InMemoryTTLCache,
    LLMConfig,
    ResponseFormat,
    StubClientManager,
    ToolReturnRecord,
    WorkflowDefinition,
    WorkflowState,
    synthesize_from_schema,
)
from tests.conftest import executor_from, run_async


def _service() -> InferenceService:
    return InferenceService(StubClientManager())


def test_text_echo_round_trip() -> None:
    """A plain text request returns a deterministic SUCCESS echo with tokens."""
    svc = _service()
    req = InferenceRequest(
        messages=[
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="hello world"),
        ]
    )
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text and "hello world" in resp.output_text
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.latency_ms >= 0.0


def test_structured_output_from_schema() -> None:
    """STRUCTURED_OUTPUT synthesizes a schema-valid instance with seeded prose."""
    svc = _service()
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
        "required": ["title", "count"],
    }
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="ACME outlook")],
        output_json_schema=schema,
    )
    # response_format auto-derives to STRUCTURED_OUTPUT.
    assert req.response_format == ResponseFormat.STRUCTURED_OUTPUT
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert isinstance(resp.output_structured, dict)
    assert resp.output_structured["count"] == 0
    assert "ACME outlook" in resp.output_structured["title"]


def test_auto_tools_invokes_bound_tools() -> None:
    """AUTO_TOOLS calls each bound tool once and records call/return pairs."""

    async def _handler(args: dict) -> ToolReturnRecord:
        return ToolReturnRecord(tool_call_id="", tool_name="adder", content={"sum": 42})

    tool = BoundTool(
        name="adder",
        args_json_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="add")],
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[tool],
        tool_executor=executor_from({"adder": _handler}),
    )
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert len(resp.tool_calls) == 1
    assert len(resp.tool_returns) == 1
    assert resp.tool_returns[0].content == {"sum": 42}
    # The synthesized call id should be propagated onto the return for pairing.
    assert resp.tool_calls[0].tool_call_id == resp.tool_returns[0].tool_call_id


def test_workflow_runs_current_step_only_and_returns_state() -> None:
    """WORKFLOW calls only the current step's tool and returns state unchanged."""

    async def _handler(args: dict) -> ToolReturnRecord:
        return ToolReturnRecord(tool_call_id="", tool_name="step_one", content="done")

    tool = BoundTool(name="step_one")
    state = WorkflowState(definition=WorkflowDefinition(steps=["step_one", "step_two"]))
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="go")],
        workflow=state,
        bound_tools=[tool],
        tool_executor=executor_from({"step_one": _handler}),
    )
    assert req.response_format == ResponseFormat.WORKFLOW
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "step_one"
    # The caller advances; the returned pointer is unchanged.
    assert resp.workflow is not None
    assert resp.workflow.current_step == 0


def test_tool_format_without_a_tool_is_failed() -> None:
    """ResponseFormat.TOOL with nothing to force is a typed FAILED (not a raise)."""
    mgr = StubClientManager()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="x")],
        response_format=ResponseFormat.TOOL,
    )
    resp = run_async(mgr.generate(req))
    assert resp.status == InferenceStatus.FAILED


def test_batch_preserves_order_and_counts() -> None:
    """run_batch preserves input order and tallies successes."""
    svc = _service()
    reqs = [
        InferenceRequest(messages=[InferenceMessage(role="user", content=str(i))])
        for i in range(5)
    ]
    batch = BatchInferenceRequest(requests=reqs, max_concurrency=2)
    out = run_async(svc.run_batch(batch))
    assert out.success_count == 5
    assert out.failure_count == 0
    assert len(out.responses) == 5
    for i, resp in enumerate(out.responses):
        assert str(i) in (resp.output_text or "")


def test_service_retries_only_retryable_failures() -> None:
    """The service retries RATE_LIMITED then succeeds; non-retryable is not retried."""

    class FlakyManager:
        """Fails with a retryable error N times, then succeeds."""

        def __init__(self, fail_times: int) -> None:
            self.calls = 0
            self.fail_times = fail_times

        def resolve(self, model_key: str) -> str:
            return "flaky"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            if self.calls <= self.fail_times:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.RATE_LIMITED,
                        message="slow down",
                        retryable=True,
                    ),
                )
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="recovered",
            )

    mgr = FlakyManager(fail_times=1)
    svc = InferenceService(mgr, retry_base_delay_seconds=0.0, retry_jitter_seconds=0.0)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.SUCCESS
    assert mgr.calls == 2

    class AuthManager:
        """Always fails with a non-retryable AUTH error."""

        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, model_key: str) -> str:
            return "auth"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=InferenceError(
                    code=InferenceErrorCode.AUTH, message="bad key", retryable=False
                ),
            )

    auth = AuthManager()
    svc2 = InferenceService(
        auth, retry_base_delay_seconds=0.0, retry_jitter_seconds=0.0
    )
    resp2 = run_async(svc2.run(InferenceRequest()))
    assert resp2.status == InferenceStatus.FAILED
    assert auth.calls == 1


def test_llmconfig_conflicting_format_raises() -> None:
    """An explicit non-structured format with an output schema is rejected."""
    with pytest.raises(ValueError):
        LLMConfig(
            response_format=ResponseFormat.TEXT,
            output_json_schema={"type": "object"},
        )


def test_synthesize_from_schema_enum_and_array() -> None:
    """The schema synthesizer honors enums, defaults, and array minItems."""
    schema = {
        "type": "object",
        "properties": {
            "status": {"enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "n": {"type": "integer", "default": 7},
        },
    }
    result = synthesize_from_schema(schema)
    assert result["status"] == "a"
    assert len(result["tags"]) == 2
    assert result["n"] == 7


# ----------------------------------------------------------------- INF-4 synth
def test_synthesize_const_enum_default_and_constraints() -> None:
    """const wins, enum is checked before a bad default, size constraints clear."""
    # const
    assert synthesize_from_schema({"const": "X"}) == "X"
    # enum-violating default must NOT leak; the enum value is used.
    assert synthesize_from_schema({"enum": ["a", "b"], "default": "z"}) == "a"
    # enum-satisfying default is honored.
    assert synthesize_from_schema({"enum": ["a", "b"], "default": "b"}) == "b"
    # minLength padding.
    assert len(synthesize_from_schema({"type": "string", "minLength": 10})) >= 10
    # minimum / exclusiveMinimum.
    assert synthesize_from_schema({"type": "integer", "minimum": 10}) == 10
    assert synthesize_from_schema({"type": "integer", "exclusiveMinimum": 5}) > 5
    assert synthesize_from_schema({"type": "number", "minimum": 2.5}) == 2.5
    # dict-typed object with minProperties synthesizes a key.
    out = synthesize_from_schema(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "minProperties": 1,
        }
    )
    assert isinstance(out, dict) and len(out) >= 1


def test_synthesize_output_is_jsonschema_valid() -> None:
    """The synthesized instance validates against the schema (INF-10)."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "ratio": {"type": "number", "minimum": 0.5},
            "status": {"enum": ["ok", "bad"]},
            "kind": {"const": "report"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
        "required": ["title", "score", "status", "kind", "tags"],
    }
    instance = synthesize_from_schema(schema, seed_text="ACME outlook")
    # Should not raise.
    jsonschema.validate(instance=instance, schema=schema)


# ------------------------------------------------------- INF-12 request guards
def test_request_rejects_contradictory_response_format() -> None:
    """An explicit format that contradicts related fields raises at construction."""
    with pytest.raises(ValueError):
        InferenceRequest(
            response_format=ResponseFormat.JSON_OBJECT,
            output_json_schema={"type": "object"},
        )
    with pytest.raises(ValueError):
        InferenceRequest(response_format=ResponseFormat.WORKFLOW)
    with pytest.raises(ValueError):
        InferenceRequest(response_format=ResponseFormat.STRUCTURED_OUTPUT)


# ------------------------------------------------ INF-1/INF-10 raising manager
class _RaisingManager:
    """A client manager whose generate() RAISES (not returns FAILED)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self._exc = exc or ValueError("boom")

    def resolve(self, model_key: str) -> str:
        return "raising"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        raise self._exc


def test_run_never_raises_normalizes_manager_exception() -> None:
    """run() returns a FAILED response (never raises) when the manager raises."""
    mgr = _RaisingManager()
    svc = InferenceService(mgr, retry_base_delay_seconds=0.0, retry_jitter_seconds=0.0)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.UNKNOWN
    assert resp.latency_ms >= 0.0
    # Non-retryable UNKNOWN: only one call.
    assert mgr.calls == 1


def test_run_batch_failure_isolated_with_raising_request() -> None:
    """One raising request does not kill the batch; it tallies as a failure."""

    class _MixedManager:
        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            if "boom" in (request.messages[0].content if request.messages else ""):
                raise RuntimeError("kaboom")
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="ok",
            )

        def resolve(self, model_key: str) -> str:
            return "mixed"

    svc = InferenceService(
        _MixedManager(), retry_base_delay_seconds=0.0, retry_jitter_seconds=0.0
    )
    reqs = [
        InferenceRequest(messages=[InferenceMessage(role="user", content="fine")]),
        InferenceRequest(messages=[InferenceMessage(role="user", content="boom")]),
        InferenceRequest(messages=[InferenceMessage(role="user", content="fine2")]),
    ]
    out = run_async(svc.run_batch(BatchInferenceRequest(requests=reqs)))
    assert len(out.responses) == 3
    assert out.success_count == 2
    assert out.failure_count == 1
    # Order preserved.
    assert out.responses[1].status == InferenceStatus.FAILED


def test_tool_format_via_service_is_failed_not_raised() -> None:
    """NotImplementedError from the manager becomes a FAILED INVALID_REQUEST."""
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="x")],
        response_format=ResponseFormat.TOOL,
    )
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.INVALID_REQUEST


# ------------------------------------------------------------- INF-3/11 timeout
def test_timeout_path_fires_with_proportional_ceiling() -> None:
    """A slow manager is cancelled at the proportional ceiling, yielding TIMEOUT."""

    class _SlowManager:
        def resolve(self, model_key: str) -> str:
            return "slow"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            await asyncio.sleep(5.0)
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    # timeout 0.05s, factor 1.0, grace 0.05 -> ceiling ~0.10s (NOT +1.0s).
    svc = InferenceService(
        _SlowManager(),
        max_retries=0,
        retry_base_delay_seconds=0.0,
        retry_jitter_seconds=0.0,
        timeout_grace_factor=1.0,
        timeout_grace_seconds=0.05,
    )
    req = InferenceRequest()
    req.timeout_seconds = 0.05
    started = time.perf_counter()
    resp = run_async(svc.run(req))
    elapsed = time.perf_counter() - started
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.TIMEOUT
    # Should be well under the old +1.0s floor.
    assert elapsed < 0.8


# ------------------------------------------------------------------ INF-9 cache
def test_cache_serves_warm_and_skips_provider() -> None:
    """An opted-in identical request is served from cache without re-calling."""

    class _CountingStub(StubClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return await super().generate(request)

    mgr = _CountingStub()
    cache = InMemoryTTLCache(ttl_seconds=60.0)
    svc = InferenceService(mgr, cache=cache)

    def _req() -> InferenceRequest:
        return InferenceRequest(
            messages=[InferenceMessage(role="user", content="cache me")],
            generation_params={"use_cache": True},
        )

    first = run_async(svc.run(_req()))
    second = run_async(svc.run(_req()))
    assert first.status == InferenceStatus.SUCCESS
    assert second.status == InferenceStatus.SUCCESS
    assert mgr.calls == 1  # second served from cache
    assert second.metadata.get("cache_hit") is True
    assert first.metadata.get("cache_hit") is not True


def test_cache_off_by_default() -> None:
    """Without use_cache the provider is hit every time (no caching)."""

    class _CountingStub(StubClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return await super().generate(request)

    mgr = _CountingStub()
    svc = InferenceService(mgr, cache=InMemoryTTLCache())
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="x")])
    run_async(svc.run(req))
    run_async(svc.run(req))
    assert mgr.calls == 2


# -------------------------------------------------------------- INF-7 streaming
def test_run_stream_emits_deltas_and_final_response() -> None:
    """run_stream chunks the stub output and ends with a done frame + response."""
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="stream this please")]
    )

    async def _collect() -> list:
        deltas = []
        async for d in svc.run_stream(req, chunk_size=8):
            deltas.append(d)
        return deltas

    deltas = run_async(_collect())
    assert deltas, "expected at least one delta"
    assert deltas[-1].done is True
    assert deltas[-1].response is not None
    assert deltas[-1].response.status == InferenceStatus.SUCCESS
    reassembled = "".join(d.delta for d in deltas)
    assert reassembled == (deltas[-1].response.output_text or "")


def test_run_stream_uses_manager_stream_when_present() -> None:
    """The stub exposes generate_stream, so run_stream consumes it directly."""
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="hello streaming world")]
    )

    async def _collect() -> list:
        out = []
        async for d in svc.run_stream(req):
            out.append(d)
        return out

    deltas = run_async(_collect())
    assert deltas[-1].done is True
    text = deltas[-1].response.output_text or ""
    assert "".join(d.delta for d in deltas) == text


# --------------------------------------------------------- INF-3 stub accounting
def test_stub_tokens_include_tool_and_structured_payloads() -> None:
    """Output tokens reflect tool calls/returns, not just the final text."""

    async def _handler(args: dict) -> ToolReturnRecord:
        return ToolReturnRecord(
            tool_call_id="", tool_name="big", content={"data": "x" * 400}
        )

    tool = BoundTool(name="big")
    svc = _service()
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="go")],
        response_format=ResponseFormat.AUTO_TOOLS,
        bound_tools=[tool],
        tool_executor=executor_from({"big": _handler}),
    )
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS
    # The large tool return should inflate output tokens beyond a tiny echo.
    assert resp.output_tokens > 20


def test_pydantic_ai_path_is_providers_gated() -> None:
    """The pydantic-ai manager raises a clear error when the extra is missing.

    This is the offline assertion of the gated provider path: with pydantic_ai
    absent, generate() surfaces an HimmyError rather than importing at module
    top. When the [providers] extra IS installed, this test is skipped because the
    real path requires provider credentials.
    """
    from himmy.core.errors import HimmyError
    from himmy.services.inference import PydanticAIClientManager

    try:
        import pydantic_ai  # type: ignore  # noqa: F401

        pytest.skip("pydantic-ai installed; real path needs provider credentials")
    except ImportError:
        pass

    mgr = PydanticAIClientManager()
    with pytest.raises(HimmyError):
        run_async(
            mgr.generate(
                InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
            )
        )
