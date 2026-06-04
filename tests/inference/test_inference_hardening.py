"""Expanded INF hardening tests — lock in the production inference contract.

These complement ``test_inference_service.py`` by pinning the harder edges of the
hardened kernel (IMPROVEMENTS.md INF-1..INF-12):

* ``run`` NEVER raises for any manager exception, and normalizes by *type*
  (``HimmyError`` -> non-retryable INVALID_REQUEST; transport-named exceptions
  -> retryable transient codes; everything else -> non-retryable UNKNOWN), while
  still firing latency stamping and the ``INFERENCE_FAILED`` lifecycle event.
* ``run_batch`` is failure-isolated for *both* raising and FAILED-returning
  requests, preserves order, and tallies success/failure correctly.
* the proportional timeout ceiling (``timeout * factor + grace``) replaces the
  old fixed +1.0s floor and actually fires.
* ``synthesize_from_schema`` emits JSON-schema-VALID instances across
  const/enum/minLength/minimum/minItems (validated with ``jsonschema`` when it
  is installed; otherwise structurally asserted).
* explicit ``response_format`` contradictions are rejected at construction for
  both :class:`InferenceRequest` and :class:`LLMConfig`, while valid combinations
  construct cleanly.
* streaming deltas (stub path) are monotonically indexed, reassemble to the
  buffered text, and end with a single ``done`` frame carrying the response.
* the response cache serves identical opted-in requests warm (one provider call),
  never caches FAILED responses, keys stably, and respects TTL expiry.

Real-provider / gateway paths are gated: they SKIP when pydantic-ai is absent (the
offline default), asserting only the import-safe error contract.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from himmy.core.errors import HimmyError
from himmy.core.events import EventType
from himmy.services.inference import (
    BatchInferenceRequest,
    GatewayClientManager,
    GatewayModelConfig,
    GatewayRuntimeConfig,
    InferenceError,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    InferenceStatus,
    InMemoryTTLCache,
    LLMConfig,
    ModelPrice,
    NoopInferenceCache,
    ResponseFormat,
    StubClientManager,
    WorkflowDefinition,
    WorkflowState,
    compute_cache_key,
    synthesize_from_schema,
)
from himmy.services.inference.cache import compute_cache_key as _ck
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# ----------------------------------------------------------------- local helpers
def _svc(manager=None, **kw) -> InferenceService:
    """Build an InferenceService with retries instant (no real sleeps)."""
    kw.setdefault("retry_base_delay_seconds", 0.0)
    kw.setdefault("retry_jitter_seconds", 0.0)
    return InferenceService(manager or StubClientManager(), **kw)


class _ExcManager:
    """A manager whose ``generate`` raises a configurable exception every call."""

    def __init__(self, exc: BaseException) -> None:
        self.calls = 0
        self._exc = exc

    def resolve(self, model_key: str) -> str:  # noqa: D401 - protocol stub
        return "exc"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        raise self._exc


# NOTE: the service normalizer keys on the exception CLASS NAME (so it never has
# to import provider SDKs). These stand-ins must therefore be named EXACTLY like
# the transport exceptions in ``service._ERROR_NAME_MAP`` to exercise that path.
class ConnectError(Exception):
    """A transport-ish exception named like an httpx/provider connect failure."""


class ReadTimeout(Exception):
    """A transport-ish exception named like a provider read timeout."""


# ============================================================ INF-1/4/10: run()
def test_run_normalizes_himmy_error_to_invalid_request_non_retryable() -> None:
    """An HimmyError from the manager -> FAILED INVALID_REQUEST, not retried."""
    mgr = _ExcManager(HimmyError("bad request shape"))
    svc = _svc(mgr)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.INVALID_REQUEST
    assert resp.error.retryable is False
    # Non-retryable: exactly one attempt.
    assert mgr.calls == 1


def test_run_normalizes_transport_named_exception_to_retryable_transient() -> None:
    """A connect-named exception maps to retryable PROVIDER_UNAVAILABLE and retries."""
    mgr = _ExcManager(ConnectError("upstream down"))
    svc = _svc(mgr, max_retries=2)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.PROVIDER_UNAVAILABLE
    assert resp.error.retryable is True
    # Retried until attempts exhausted: 1 + max_retries.
    assert mgr.calls == 3


def test_run_normalizes_read_timeout_named_exception_to_timeout() -> None:
    """A ReadTimeout-named exception maps to the retryable TIMEOUT code."""
    mgr = _ExcManager(ReadTimeout("slow upstream"))
    svc = _svc(mgr, max_retries=1)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.TIMEOUT
    assert resp.error.retryable is True
    assert mgr.calls == 2


def test_run_unknown_exception_is_non_retryable_unknown() -> None:
    """An arbitrary (non-transport) exception -> non-retryable UNKNOWN, one call."""
    mgr = _ExcManager(KeyError("surprise"))
    svc = _svc(mgr, max_retries=2)
    resp = run_async(svc.run(InferenceRequest()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.UNKNOWN
    assert resp.error.retryable is False
    assert mgr.calls == 1


def test_run_failure_emits_inference_failed_event_with_latency() -> None:
    """A raising manager still fires INFERENCE_FAILED (with latency) via the sink."""
    storage = StorageService()
    mgr = _ExcManager(ValueError("boom"))
    svc = _svc(mgr, event_sink=storage)
    req = InferenceRequest()
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.FAILED
    assert resp.latency_ms >= 0.0

    events = run_async(storage.list_events())
    types = [e.event_type for e in events if e.request_id == req.request_id]
    assert EventType.INFERENCE_REQUESTED in types
    assert EventType.INFERENCE_FAILED in types
    assert EventType.INFERENCE_SUCCEEDED not in types


def test_run_success_emits_succeeded_event_with_token_payload() -> None:
    """A successful run fires INFERENCE_SUCCEEDED carrying token accounting."""
    storage = StorageService()
    svc = _svc(StubClientManager(), event_sink=storage)
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="hello")])
    resp = run_async(svc.run(req))
    assert resp.status == InferenceStatus.SUCCESS

    events = run_async(storage.list_events())
    succeeded = [
        e
        for e in events
        if e.request_id == req.request_id
        and e.event_type == EventType.INFERENCE_SUCCEEDED
    ]
    assert len(succeeded) == 1
    payload = succeeded[0].payload
    assert payload.get("input_tokens", 0) > 0
    assert payload.get("output_tokens", 0) > 0


def test_run_event_sink_failure_never_breaks_inference() -> None:
    """A sink that raises on append_event must not break the run (best-effort)."""

    class _BadSink:
        async def append_event(self, event):  # noqa: ANN001
            raise RuntimeError("sink down")

    svc = _svc(StubClientManager(), event_sink=_BadSink())
    resp = run_async(
        svc.run(InferenceRequest(messages=[InferenceMessage(role="user", content="x")]))
    )
    # Observability failures are swallowed; the call still succeeds.
    assert resp.status == InferenceStatus.SUCCESS


# =============================================================== INF-5: run_batch
def test_run_batch_isolated_across_raising_and_failed_returning() -> None:
    """A batch mixing raising, FAILED-returning, and OK requests tallies correctly."""

    class _Mixed:
        def resolve(self, model_key: str) -> str:
            return "mixed"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            content = request.messages[0].content if request.messages else ""
            if content == "raise":
                raise RuntimeError("kaboom")
            if content == "fail":
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.AUTH, message="nope", retryable=False
                    ),
                )
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text=content,
            )

    contents = ["ok0", "raise", "ok1", "fail", "ok2"]
    reqs = [
        InferenceRequest(messages=[InferenceMessage(role="user", content=c)])
        for c in contents
    ]
    svc = _svc(_Mixed())
    out = run_async(svc.run_batch(BatchInferenceRequest(requests=reqs)))

    assert len(out.responses) == 5
    assert out.success_count == 3
    assert out.failure_count == 2
    # Order preserved 1:1 with input.
    assert out.responses[0].status == InferenceStatus.SUCCESS
    assert out.responses[1].status == InferenceStatus.FAILED  # raised
    assert out.responses[2].status == InferenceStatus.SUCCESS
    assert out.responses[3].status == InferenceStatus.FAILED  # returned FAILED
    assert out.responses[4].status == InferenceStatus.SUCCESS
    assert out.elapsed_ms >= 0.0


def test_run_batch_respects_concurrency_bound() -> None:
    """max_concurrency caps simultaneous in-flight requests."""

    class _Tracker:
        def __init__(self) -> None:
            self.live = 0
            self.peak = 0

        def resolve(self, model_key: str) -> str:
            return "track"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0.01)
            self.live -= 1
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    tracker = _Tracker()
    svc = _svc(tracker)
    reqs = [InferenceRequest() for _ in range(8)]
    out = run_async(
        svc.run_batch(BatchInferenceRequest(requests=reqs, max_concurrency=3))
    )
    assert out.success_count == 8
    assert tracker.peak <= 3


def test_run_batch_empty_is_zeroed() -> None:
    """An empty batch returns zero counts and no responses (no crash)."""
    svc = _svc()
    out = run_async(svc.run_batch(BatchInferenceRequest(requests=[])))
    assert out.responses == []
    assert out.success_count == 0
    assert out.failure_count == 0


# ====================================================== INF-3/11: timeout grace
def test_ceiling_is_proportional_not_fixed_one_second() -> None:
    """The hard ceiling is timeout*factor + grace, NOT a fixed +1.0s floor."""
    svc = _svc(timeout_grace_factor=1.05, timeout_grace_seconds=0.25)
    # 0.1 * 1.05 + 0.25 = 0.355, well below the old 1.1s.
    ceiling = svc._ceiling(0.1)
    assert abs(ceiling - 0.355) < 1e-9
    assert ceiling < 1.0


def test_timeout_fires_below_old_floor_and_reports_ceiling() -> None:
    """A slow manager is cancelled near the proportional ceiling with TIMEOUT."""

    class _Slow:
        def resolve(self, model_key: str) -> str:
            return "slow"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            await asyncio.sleep(5.0)
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    svc = _svc(
        _Slow(),
        max_retries=0,
        timeout_grace_factor=1.0,
        timeout_grace_seconds=0.05,
    )
    req = InferenceRequest(timeout_seconds=0.05)  # ceiling ~0.10s
    started = time.perf_counter()
    resp = run_async(svc.run(req))
    elapsed = time.perf_counter() - started
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None
    assert resp.error.code == InferenceErrorCode.TIMEOUT
    assert resp.error.retryable is True
    # Far under the legacy +1.0s floor (1.1s).
    assert elapsed < 0.7
    # The message reports the actual ceiling, not a hardcoded constant.
    assert "0.10" in resp.error.message or "ceiling" in resp.error.message


def test_default_timeout_used_when_request_timeout_is_zero() -> None:
    """timeout_seconds=0 falls back to the service default (does not become a no-op)."""

    class _Slow:
        def resolve(self, model_key: str) -> str:
            return "slow"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            await asyncio.sleep(5.0)
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    svc = _svc(
        _Slow(),
        max_retries=0,
        default_timeout_seconds=0.05,
        timeout_grace_factor=1.0,
        timeout_grace_seconds=0.05,
    )
    req = InferenceRequest(timeout_seconds=0.0)
    started = time.perf_counter()
    resp = run_async(svc.run(req))
    elapsed = time.perf_counter() - started
    assert resp.status == InferenceStatus.FAILED
    assert resp.error.code == InferenceErrorCode.TIMEOUT
    assert elapsed < 0.7


# ==================================================== INF-8: schema synthesis
def test_synthesize_const_inside_required_object() -> None:
    """A const property inside a required object yields exactly that value."""
    schema = {
        "type": "object",
        "properties": {"kind": {"const": "report"}, "title": {"type": "string"}},
        "required": ["kind", "title"],
    }
    out = synthesize_from_schema(schema)
    assert out["kind"] == "report"
    assert isinstance(out["title"], str) and out["title"]


def test_synthesize_minlength_minimum_minitems_together() -> None:
    """minLength padding, minimum flooring, and nested minItems all hold at once."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 8},
            "qty": {"type": "integer", "minimum": 3},
            "rate": {"type": "number", "minimum": 1.5},
            "tags": {
                "type": "array",
                "minItems": 3,
                "items": {"type": "string", "minLength": 2},
            },
        },
        "required": ["name", "qty", "rate", "tags"],
    }
    out = synthesize_from_schema(schema)
    assert len(out["name"]) >= 8
    assert out["qty"] >= 3
    assert out["rate"] >= 1.5
    assert len(out["tags"]) >= 3
    assert all(len(t) >= 2 for t in out["tags"])


def test_synthesize_enum_default_violation_uses_enum_member() -> None:
    """{enum:[a,b], default:z} must NOT leak the enum-violating default."""
    assert synthesize_from_schema({"enum": ["a", "b"], "default": "z"}) == "a"
    # A default that DOES satisfy the enum is honored.
    assert synthesize_from_schema({"enum": ["a", "b"], "default": "b"}) == "b"


def test_synthesize_dict_object_minproperties_synthesizes_keys() -> None:
    """A free-form object with minProperties>0 gets enough synthesized entries."""
    out = synthesize_from_schema(
        {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 5},
            "minProperties": 2,
        }
    )
    assert isinstance(out, dict)
    assert len(out) >= 2
    assert all(isinstance(v, int) and v >= 5 for v in out.values())


def test_synthesize_ref_and_defs_round_trip() -> None:
    """Local $ref/$defs resolve so nested pydantic-style models synthesize."""
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Child"}},
        "required": ["child"],
        "$defs": {
            "Child": {
                "type": "object",
                "properties": {"label": {"const": "L"}},
                "required": ["label"],
            }
        },
    }
    out = synthesize_from_schema(schema)
    assert out["child"]["label"] == "L"


def test_synthesize_is_jsonschema_valid_across_constructs() -> None:
    """The synthesized instance validates against a schema mixing many constructs."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 5},
            "kind": {"const": "brief"},
            "status": {"enum": ["draft", "final"]},
            "score": {"type": "integer", "minimum": 2, "maximum": 9},
            "weight": {"type": "number", "minimum": 0.25},
            "tags": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "attrs": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "minProperties": 1,
            },
        },
        "required": ["title", "kind", "status", "score", "weight", "tags", "attrs"],
    }
    instance = synthesize_from_schema(schema, seed_text="ACME quarterly outlook")
    jsonschema.validate(instance=instance, schema=schema)  # must not raise


# ============================================= INF-12: response_format contracts
def test_request_rejects_text_with_output_schema() -> None:
    """TEXT + an output_json_schema is a contradiction rejected at construction."""
    with pytest.raises(ValueError):
        InferenceRequest(
            response_format=ResponseFormat.TEXT,
            output_json_schema={"type": "object"},
        )


def test_request_rejects_json_object_with_output_schema() -> None:
    """JSON_OBJECT + an output_json_schema is rejected (must be STRUCTURED_OUTPUT)."""
    with pytest.raises(ValueError):
        InferenceRequest(
            response_format=ResponseFormat.JSON_OBJECT,
            output_json_schema={"type": "object"},
        )


def test_request_rejects_workflow_without_workflow_and_struct_without_schema() -> None:
    """Formats that hard-require a field raise when it is missing."""
    with pytest.raises(ValueError):
        InferenceRequest(response_format=ResponseFormat.WORKFLOW)
    with pytest.raises(ValueError):
        InferenceRequest(response_format=ResponseFormat.STRUCTURED_OUTPUT)


def test_request_rejects_explicit_text_when_workflow_provided() -> None:
    """An explicit TEXT format with a workflow set is a contradiction."""
    state = WorkflowState(definition=WorkflowDefinition(steps=["a"]))
    with pytest.raises(ValueError):
        InferenceRequest(response_format=ResponseFormat.TEXT, workflow=state)


def test_request_valid_combinations_construct_cleanly() -> None:
    """Auto-derivation succeeds for the legal field combinations."""
    # schema -> STRUCTURED_OUTPUT
    r1 = InferenceRequest(output_json_schema={"type": "object"})
    assert r1.response_format == ResponseFormat.STRUCTURED_OUTPUT
    # workflow -> WORKFLOW
    state = WorkflowState(definition=WorkflowDefinition(steps=["a"]))
    r2 = InferenceRequest(workflow=state)
    assert r2.response_format == ResponseFormat.WORKFLOW
    # explicit STRUCTURED_OUTPUT + schema agrees
    r3 = InferenceRequest(
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema={"type": "object"},
    )
    assert r3.response_format == ResponseFormat.STRUCTURED_OUTPUT
    # plain TEXT with nothing else
    r4 = InferenceRequest(response_format=ResponseFormat.TEXT)
    assert r4.response_format == ResponseFormat.TEXT


def test_llmconfig_rejects_workflow_with_nonworkflow_format() -> None:
    """LLMConfig mirrors the request guards for workflow conflicts."""
    state = WorkflowState(definition=WorkflowDefinition(steps=["a"]))
    with pytest.raises(ValueError):
        LLMConfig(response_format=ResponseFormat.TEXT, workflow=state)


def test_llmconfig_auto_derives_structured_and_workflow() -> None:
    """LLMConfig auto-derives the format from related fields when omitted."""
    assert (
        LLMConfig(output_json_schema={"type": "object"}).response_format
        == ResponseFormat.STRUCTURED_OUTPUT
    )
    state = WorkflowState(definition=WorkflowDefinition(steps=["a"]))
    assert LLMConfig(workflow=state).response_format == ResponseFormat.WORKFLOW


# ================================================================ INF-7: streaming
def test_stream_deltas_monotonic_and_reassemble_via_fallback() -> None:
    """The buffered fallback path yields ordered deltas ending in one done frame."""

    # A manager WITHOUT generate_stream forces the buffered fallback in run_stream.
    class _NoStream:
        def resolve(self, model_key: str) -> str:
            return "nostream"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="abcdefghijklmnop",
            )

    svc = _svc(_NoStream())
    req = InferenceRequest()

    async def _collect() -> list:
        return [d async for d in svc.run_stream(req, chunk_size=5)]

    deltas = run_async(_collect())
    # Exactly one terminal frame, and it is last.
    done = [d for d in deltas if d.done]
    assert len(done) == 1
    assert deltas[-1].done is True
    # Indices are strictly increasing.
    indices = [d.index for d in deltas]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    # Reassembly equals the buffered text, carried on the done frame.
    assert deltas[-1].response is not None
    assert "".join(d.delta for d in deltas) == "abcdefghijklmnop"
    assert deltas[-1].response.output_text == "abcdefghijklmnop"


def test_stream_prefers_manager_stream_and_carries_final_response() -> None:
    """The stub exposes generate_stream, so run_stream consumes it (not the fallback)."""
    svc = _svc(StubClientManager())
    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="hello stream world")]
    )

    async def _collect() -> list:
        return [d async for d in svc.run_stream(req)]

    deltas = run_async(_collect())
    assert deltas[-1].done is True
    final = deltas[-1].response
    assert final is not None and final.status == InferenceStatus.SUCCESS
    # Non-terminal deltas are not 'done'.
    assert all(not d.done for d in deltas[:-1])
    assert "".join(d.delta for d in deltas) == (final.output_text or "")


def test_stream_request_id_is_propagated_on_every_delta() -> None:
    """Each delta carries the request's id so consumers can correlate the stream."""
    svc = _svc(StubClientManager())
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="id check")])

    async def _collect() -> list:
        return [d async for d in svc.run_stream(req, chunk_size=4)]

    deltas = run_async(_collect())
    assert deltas
    assert all(d.request_id == req.request_id for d in deltas)


# ================================================================== INF-9: cache
def test_cache_hit_serves_warm_and_skips_provider() -> None:
    """An opted-in identical request is served from cache; provider hit only once."""

    class _Counting(StubClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return await super().generate(request)

    mgr = _Counting()
    svc = _svc(mgr, cache=InMemoryTTLCache(ttl_seconds=60.0))

    def _req() -> InferenceRequest:
        return InferenceRequest(
            messages=[InferenceMessage(role="user", content="warm me")],
            generation_params={"use_cache": True},
        )

    first = run_async(svc.run(_req()))
    second = run_async(svc.run(_req()))
    assert mgr.calls == 1
    assert first.metadata.get("cache_hit") is not True
    assert second.metadata.get("cache_hit") is True
    # The hit retains the original content but is stamped with this request's id.
    assert second.output_text == first.output_text


def test_cache_hit_distinct_request_id_same_content() -> None:
    """A cache hit carries the NEW request_id but the cached content/tokens."""

    mgr = StubClientManager()
    svc = _svc(mgr, cache=InMemoryTTLCache())
    r1 = InferenceRequest(
        messages=[InferenceMessage(role="user", content="abc")],
        generation_params={"use_cache": True},
    )
    r2 = InferenceRequest(
        messages=[InferenceMessage(role="user", content="abc")],
        generation_params={"use_cache": True},
    )
    first = run_async(svc.run(r1))
    second = run_async(svc.run(r2))
    assert second.metadata.get("cache_hit") is True
    assert second.request_id == r2.request_id
    assert second.request_id != first.request_id
    assert second.output_text == first.output_text
    assert second.output_tokens == first.output_tokens


def test_cache_emits_succeeded_event_on_hit() -> None:
    """A cache hit still fires INFERENCE_SUCCEEDED tagged cache_hit=True."""
    storage = StorageService()
    svc = _svc(StubClientManager(), cache=InMemoryTTLCache(), event_sink=storage)

    def _req() -> InferenceRequest:
        return InferenceRequest(
            messages=[InferenceMessage(role="user", content="hit")],
            generation_params={"use_cache": True},
        )

    run_async(svc.run(_req()))
    r2 = _req()
    run_async(svc.run(r2))
    events = run_async(storage.list_events())
    hit_events = [
        e
        for e in events
        if e.request_id == r2.request_id
        and e.event_type == EventType.INFERENCE_SUCCEEDED
        and e.payload.get("cache_hit") is True
    ]
    assert len(hit_events) == 1


def test_cache_does_not_store_failed_responses() -> None:
    """A FAILED response is never cached; the next call re-hits the provider."""

    class _FailOnceThenOk:
        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, model_key: str) -> str:
            return "fail-once"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            if self.calls == 1:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.AUTH, message="x", retryable=False
                    ),
                )
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="ok now",
            )

    mgr = _FailOnceThenOk()
    svc = _svc(mgr, cache=InMemoryTTLCache())

    def _req() -> InferenceRequest:
        return InferenceRequest(
            messages=[InferenceMessage(role="user", content="same")],
            generation_params={"use_cache": True},
        )

    first = run_async(svc.run(_req()))
    second = run_async(svc.run(_req()))
    assert first.status == InferenceStatus.FAILED
    assert second.status == InferenceStatus.SUCCESS
    # The failure was NOT cached: provider was hit both times.
    assert mgr.calls == 2


def test_cache_off_by_default_without_opt_in() -> None:
    """Without use_cache the provider is hit each time even with a cache wired."""

    class _Counting(StubClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return await super().generate(request)

    mgr = _Counting()
    svc = _svc(mgr, cache=InMemoryTTLCache())
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="x")])
    run_async(svc.run(req))
    run_async(svc.run(req))
    assert mgr.calls == 2


def test_cache_ttl_expiry_misses() -> None:
    """An expired TTL entry is a miss (re-hits the provider)."""

    class _Counting(StubClientManager):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            self.calls += 1
            return await super().generate(request)

    mgr = _Counting()
    svc = _svc(mgr, cache=InMemoryTTLCache(ttl_seconds=0.0))  # everything expires

    def _req() -> InferenceRequest:
        return InferenceRequest(
            messages=[InferenceMessage(role="user", content="ttl")],
            generation_params={"use_cache": True},
        )

    run_async(svc.run(_req()))
    run_async(svc.run(_req()))
    # ttl=0 means the stored entry is immediately expired -> miss -> re-call.
    assert mgr.calls == 2


def test_compute_cache_key_stable_and_content_sensitive() -> None:
    """The cache key ignores request_id/use_cache but tracks content + format."""
    a = InferenceRequest(
        messages=[InferenceMessage(role="user", content="same")],
        generation_params={"use_cache": True, "temperature": 0.0},
    )
    b = InferenceRequest(
        messages=[InferenceMessage(role="user", content="same")],
        generation_params={"use_cache": False, "temperature": 0.0},
    )
    c = InferenceRequest(
        messages=[InferenceMessage(role="user", content="different")],
        generation_params={"use_cache": True, "temperature": 0.0},
    )
    # request_id differs (random) and use_cache differs, yet a == b.
    assert compute_cache_key(a) == compute_cache_key(b) == _ck(a)
    # Content change flips the key.
    assert compute_cache_key(a) != compute_cache_key(c)


def test_noop_cache_never_stores() -> None:
    """The default NoopInferenceCache always misses (behavior unchanged)."""
    cache = NoopInferenceCache()
    req = InferenceRequest(messages=[InferenceMessage(role="user", content="x")])
    key = cache.key_for(req)
    resp = InferenceResponse(request_id=req.request_id, status=InferenceStatus.SUCCESS)
    run_async(cache.set(key, resp))
    assert run_async(cache.get(key)) is None


# ============================================ INF-2: gated gateway / provider path
def _pydantic_ai_absent() -> bool:
    try:
        import pydantic_ai  # type: ignore  # noqa: F401

        return False
    except ImportError:
        return True


@pytest.mark.skipif(
    not _pydantic_ai_absent(),
    reason="pydantic-ai installed; gateway production path needs real credentials",
)
def test_gateway_raises_clear_error_without_extra_or_key() -> None:
    """Offline: GatewayClientManager surfaces an actionable HimmyError."""
    os.environ.pop("PYDANTIC_AI_GATEWAY_API_KEY", None)
    os.environ.pop("HIMMY_GATEWAY_STUB_FALLBACK", None)
    mgr = GatewayClientManager(GatewayRuntimeConfig())
    with pytest.raises(HimmyError):
        run_async(mgr.generate(InferenceRequest()))


@pytest.mark.skipif(
    not _pydantic_ai_absent(),
    reason="pydantic-ai installed; gateway production path needs real credentials",
)
def test_gateway_stub_fallback_when_env_opted_in() -> None:
    """With the explicit stub-fallback env, the gateway degrades to the stub offline."""
    os.environ.pop("PYDANTIC_AI_GATEWAY_API_KEY", None)
    os.environ["HIMMY_GATEWAY_STUB_FALLBACK"] = "1"
    try:
        mgr = GatewayClientManager(GatewayRuntimeConfig())
        resp = run_async(
            mgr.generate(
                InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
            )
        )
        assert resp.status == InferenceStatus.SUCCESS
    finally:
        os.environ.pop("HIMMY_GATEWAY_STUB_FALLBACK", None)


def test_gateway_resolve_uses_registry_then_falls_back_to_key() -> None:
    """resolve() maps registered keys via the registry and passes unknown keys through.

    Pure resolution touches no provider SDK, so this is safe offline (ungated).
    """
    cfg = GatewayRuntimeConfig(
        region="eu",
        model_registry={
            "fast": GatewayModelConfig(api_format="openai", model_name="gpt-4o-mini")
        },
    )
    mgr = GatewayClientManager(cfg)
    assert mgr.resolve("fast") == "eu:gpt-4o-mini"
    # Unknown key: treated as a model name under the region.
    assert mgr.resolve("unknown-model") == "eu:unknown-model"


def test_model_price_table_computes_cost() -> None:
    """GatewayRuntimeConfig price lookup + ModelPrice.cost produce real USD numbers.

    This pins the cost-accounting plumbing (INF-3) without needing a live provider.
    """
    cfg = GatewayRuntimeConfig(
        model_prices={"premium": ModelPrice(input_per_1k=0.01, output_per_1k=0.03)}
    )
    price = cfg.price_for(model_key="premium")
    assert price.cost(input_tokens=1000, output_tokens=2000) == pytest.approx(
        0.01 + 0.06
    )
    # Unknown key -> zero price (no crash).
    assert (
        cfg.price_for(model_key="missing").cost(input_tokens=1000, output_tokens=1000)
        == 0.0
    )


@pytest.mark.skipif(
    not _pydantic_ai_absent(),
    reason="pydantic-ai installed; real provider path needs credentials",
)
def test_pydantic_ai_manager_is_import_safe_but_errors_on_generate() -> None:
    """The pydantic-ai manager imports cleanly offline and errors only on generate."""
    from himmy.services.inference import PydanticAIClientManager

    mgr = PydanticAIClientManager()
    with pytest.raises(HimmyError):
        run_async(
            mgr.generate(
                InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])
            )
        )
