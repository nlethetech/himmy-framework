"""C5 — prompt-cache ACTIVATION: capability opt-in, default-on loop, both emit sites.

These pin the C5 wiring that turns universal prompt caching on safely:

* capability is resolved at the CALL SITE via ``getattr(manager,'cache_capability',NONE)``
  so every non-supporting manager (Stub/local/custom) no-ops a ``cache_policy``; the
  wrapping managers (multi-provider/routing) expose ``cache_capability_for`` so the
  runtime sees the BACKEND that will actually serve a key;
* the shared ``cache_metrics_payload`` is the SINGLE event-schema source both emit sites
  use, including the ``cache_read==0``-across-identical-prefix audit signal
  (``prefix_cache_miss``) so a silently-busted prefix is caught;
* ``InferenceService`` surfaces the cache fields on the real-provider INFERENCE_SUCCEEDED;
* ``SingleAgentRuntime`` is default-ON (``enable_prompt_cache`` / ``HIMMY_PROMPT_CACHE``),
  attaches ``CachePolicy()`` only when the manager supports caching, and BUSTS the cache
  on a compaction turn (so a stale prefix is not marked).

All offline: injected fake managers, no provider key, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.models import (
    CachePolicy,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.multi_provider import MultiProviderClientManager
from himmy.services.inference.prompt_cache import (
    CacheCapability,
    cache_metrics_payload,
    resolve_cache_capability,
)
from himmy.services.inference.routing import RoutingClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# ====================================================== fake managers (offline)
class _StubNoCap:
    """A bare manager with NO cache_capability attribute (the local/offline default)."""

    provider_name = "stub"

    def __init__(self) -> None:
        self.seen_policies: list[CachePolicy | None] = []

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        # Record whether a policy was attached so a test can prove the no-op contract.
        self.seen_policies.append(request.cache_policy)
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="ok",
            input_tokens=1,
            output_tokens=1,
            metadata={"saw_cache_policy": request.cache_policy is not None},
        )


class _FakeAnthropic:
    """A manager that DECLARES ANTHROPIC_EXPLICIT and stamps cache metadata on hits.

    The first turn reports a cache *creation* (a cold write); every later turn reports a
    cache *read* — mirroring how a real prefix cache behaves across identical-prefix
    turns. A turn whose request carried NO active policy (e.g. a compaction-busted turn)
    reports zero cache activity, like a request that never opted in.
    """

    cache_capability = CacheCapability.ANTHROPIC_EXPLICIT
    provider_name = "anthropic"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_policies: list[CachePolicy | None] = []

    def resolve(self, model_key: str) -> str:
        return f"anthropic:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        self.seen_policies.append(request.cache_policy)
        active = request.cache_policy is not None and request.cache_policy.enabled
        meta: dict[str, Any] = {}
        if active:
            if self.calls == 1:
                meta = {
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 40,
                    "cache_savings_usd": -0.001,
                }
            else:
                meta = {
                    "cache_read_tokens": 40,
                    "cache_creation_tokens": 0,
                    "cache_savings_usd": 0.01,
                }
        # A continuation turn keeps tool-calling so the loop drives multiple turns; the
        # very last scripted call answers (no tools) to end the loop deterministically.
        has_tools = self.calls < 3
        cid = f"c{self.calls}"
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=None if has_tools else "done",
            tool_calls=(
                [ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})]
                if has_tools
                else []
            ),
            tool_returns=(
                [
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ]
                if has_tools
                else []
            ),
            input_tokens=10,
            output_tokens=2,
            metadata=meta,
        )


# ============================================ capability resolution at the call site
def test_resolve_capability_getattr_default_none_for_bare_manager() -> None:
    """A manager with no attribute resolves to NONE (fail-safe no-op)."""
    assert resolve_cache_capability(_StubNoCap()) is CacheCapability.NONE


def test_resolve_capability_reads_declared_attribute() -> None:
    """A manager that sets the attribute is resolved to its declared capability."""
    assert (
        resolve_cache_capability(_FakeAnthropic())
        is CacheCapability.ANTHROPIC_EXPLICIT
    )


def test_resolve_capability_non_enum_attribute_degrades_to_none() -> None:
    """A manager whose attribute is not a CacheCapability degrades to NONE, not a crash."""

    class _Weird:
        cache_capability = "not-an-enum"

    assert resolve_cache_capability(_Weird()) is CacheCapability.NONE


def test_multi_provider_resolves_per_key_backend_capability() -> None:
    """MultiProvider returns the picked backend's capability per model_key."""
    mp = MultiProviderClientManager(
        {"brain": _FakeAnthropic(), "worker": _StubNoCap()},
        default_key="worker",
    )
    assert mp.cache_capability_for("brain") is CacheCapability.ANTHROPIC_EXPLICIT
    assert mp.cache_capability_for("worker") is CacheCapability.NONE
    # Unknown key falls back to the default backend's capability.
    assert mp.cache_capability_for("nope") is CacheCapability.NONE


def test_routing_resolves_primary_route_capability() -> None:
    """Routing keys the decision on the PRIMARY route's manager."""
    caching = RoutingClientManager.from_managers(_FakeAnthropic(), _StubNoCap())
    assert caching.cache_capability_for("x") is CacheCapability.ANTHROPIC_EXPLICIT
    local = RoutingClientManager.from_managers(_StubNoCap(), _FakeAnthropic())
    assert local.cache_capability_for("x") is CacheCapability.NONE


def test_service_exposes_manager_capability() -> None:
    """InferenceService.cache_capability_for delegates to the wrapped manager."""
    assert (
        InferenceService(_FakeAnthropic()).cache_capability_for("default")
        is CacheCapability.ANTHROPIC_EXPLICIT
    )
    assert (
        InferenceService(_StubNoCap()).cache_capability_for("default")
        is CacheCapability.NONE
    )


# ====================================================== cache_metrics_payload (shared)
#: A system prompt large enough to clear the per-model cacheable-prefix floor.
_BIG_SYSTEM = "You are a meticulous assistant. " * 200


def _req(active: bool, *, over_threshold: bool = True) -> InferenceRequest:
    messages = [InferenceMessage(role="user", content="hi")]
    if over_threshold:
        messages.insert(0, InferenceMessage(role="system", content=_BIG_SYSTEM))
    return InferenceRequest(
        model_key="claude-3-5-sonnet-latest",
        cache_policy=CachePolicy() if active else None,
        messages=messages,
    )


def test_metrics_payload_empty_when_caching_not_requested() -> None:
    """No policy + no cache metadata → empty payload (byte-identical no-cache event)."""
    resp = InferenceResponse(request_id="r", status=InferenceStatus.SUCCESS)
    assert cache_metrics_payload(_req(active=False), resp) == {}


def test_metrics_payload_surfaces_read_creation_savings() -> None:
    """The adapter-stamped breakdown is surfaced 1:1 for the event."""
    resp = InferenceResponse(
        request_id="r",
        status=InferenceStatus.SUCCESS,
        metadata={
            "cache_read_tokens": 40,
            "cache_creation_tokens": 0,
            "cache_savings_usd": 0.01,
        },
    )
    payload = cache_metrics_payload(_req(active=True), resp)
    assert payload["cache_read_tokens"] == 40
    assert payload["cache_creation_tokens"] == 0
    assert payload["cache_savings_usd"] == 0.01
    assert payload["prefix_cache_requested"] is True
    assert "prefix_cache_miss" not in payload  # there WAS cache activity


def test_metrics_payload_flags_busted_prefix_audit_signal() -> None:
    """Over-threshold + requested + zero activity → prefix_cache_miss (the audit signal)."""
    resp = InferenceResponse(
        request_id="r",
        status=InferenceStatus.SUCCESS,
        model_path="claude-3-5-sonnet-latest",  # resolved model → the audit floor
    )
    payload = cache_metrics_payload(_req(active=True, over_threshold=True), resp)
    assert payload["prefix_cache_requested"] is True
    assert payload["prefix_cache_miss"] is True


def test_metrics_payload_sub_threshold_prefix_is_not_flagged() -> None:
    """A legitimately sub-threshold prefix produces no miss flag (not noise)."""
    resp = InferenceResponse(request_id="r", status=InferenceStatus.SUCCESS)
    payload = cache_metrics_payload(_req(active=True, over_threshold=False), resp)
    assert payload.get("prefix_cache_requested") is True
    assert "prefix_cache_miss" not in payload


def test_metrics_payload_warm_response_cache_hit_is_not_a_busted_prefix() -> None:
    """A warm response-cache hit (cache_hit) is never flagged as a busted prefix."""
    resp = InferenceResponse(
        request_id="r",
        status=InferenceStatus.SUCCESS,
        metadata={"cache_hit": True},
    )
    payload = cache_metrics_payload(_req(active=True), resp)
    assert payload.get("prefix_cache_requested") is True
    assert "prefix_cache_miss" not in payload


def test_metrics_payload_floor_uses_resolved_model_path_not_aliased_key() -> None:
    """An aliased/'default' model_key misses the family floor and falls to 4096; the
    audit floor must come from response.model_path (the model the adapter served, and
    gated should_cache_prefix on) so a genuine busted prefix over the RESOLVED floor is
    still flagged. Prefix ~1600 tokens: below the 4096 alias default, above the resolved
    'claude' floor of 1024."""
    req = InferenceRequest(
        model_key="default",  # alias: _family_min_tokens misses → 4096 default
        cache_policy=CachePolicy(),
        messages=[
            InferenceMessage(role="system", content=_BIG_SYSTEM),
            InferenceMessage(role="user", content="hi"),
        ],
    )
    resp = InferenceResponse(
        request_id="r",
        status=InferenceStatus.SUCCESS,
        model_path="claude-3-5-sonnet-latest",  # resolved model → floor 1024
    )
    payload = cache_metrics_payload(req, resp)
    assert payload["prefix_cache_requested"] is True
    assert payload["prefix_cache_miss"] is True  # flagged via resolved-model floor


def test_metrics_payload_empty_model_path_falls_to_conservative_default() -> None:
    """A local manager leaves model_path empty → floor falls to the 4096 default; a
    ~1600-token prefix is below it, so no busted-prefix flag (conservative, no noise)."""
    req = InferenceRequest(
        model_key="default",
        cache_policy=CachePolicy(),
        messages=[
            InferenceMessage(role="system", content=_BIG_SYSTEM),
            InferenceMessage(role="user", content="hi"),
        ],
    )
    resp = InferenceResponse(request_id="r", status=InferenceStatus.SUCCESS)
    payload = cache_metrics_payload(req, resp)
    assert payload["prefix_cache_requested"] is True
    assert "prefix_cache_miss" not in payload


# ====================================================== service emit-site wiring
def test_service_emits_cache_fields_on_real_provider_success() -> None:
    """A real-provider success surfaces the cache breakdown on INFERENCE_SUCCEEDED."""
    storage = StorageService()

    class _M:
        provider_name = "anthropic"
        cache_capability = CacheCapability.ANTHROPIC_EXPLICIT

        def resolve(self, model_key: str) -> str:
            return "anthropic:x"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="ok",
                input_tokens=50,
                output_tokens=2,
                metadata={
                    "cache_read_tokens": 40,
                    "cache_creation_tokens": 0,
                    "cache_savings_usd": 0.02,
                },
            )

    svc = InferenceService(_M(), event_sink=storage)
    req = InferenceRequest(
        cache_policy=CachePolicy(),
        messages=[InferenceMessage(role="user", content="hi")],
    )
    run_async(svc.run(req))
    events = run_async(storage.list_events())
    succeeded = [
        e
        for e in events
        if e.request_id == req.request_id
        and e.event_type == EventType.INFERENCE_SUCCEEDED
    ]
    assert len(succeeded) == 1
    payload = succeeded[0].payload
    assert payload["cache_read_tokens"] == 40
    assert payload["cache_savings_usd"] == 0.02
    assert payload["prefix_cache_requested"] is True


# ====================================================== runtime default-on / opt-out
def _persona_task() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


def test_runtime_attaches_policy_for_a_caching_manager_by_default() -> None:
    """Default-ON: a caching-capable manager sees CachePolicy() on every turn."""
    mgr = _FakeAnthropic()
    rt = SingleAgentRuntime(inference_service=InferenceService(mgr))
    persona, task = _persona_task()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=3))
    assert result.turn_count >= 1
    # Every turn's request carried an enabled policy.
    assert mgr.seen_policies
    assert all(p is not None and p.enabled for p in mgr.seen_policies)


def test_runtime_no_op_for_non_supporting_manager() -> None:
    """A NONE-capability manager NEVER gets a policy (byte-identical offline payload)."""
    mgr = _StubNoCap()
    rt = SingleAgentRuntime(inference_service=InferenceService(mgr))
    persona, task = _persona_task()
    result = run_async(rt.run_task_detailed(persona, task))
    assert result.succeeded
    # The stub recorded whether it saw a policy: it must not have on any turn.
    assert mgr.seen_policies == [None]


def test_runtime_env_opt_out_disables_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIMMY_PROMPT_CACHE=0 restores the no-cache path even for a caching manager."""
    monkeypatch.setenv("HIMMY_PROMPT_CACHE", "0")
    mgr = _FakeAnthropic()
    rt = SingleAgentRuntime(inference_service=InferenceService(mgr))
    persona, task = _persona_task()
    run_async(rt.run_task(persona, task))
    assert mgr.seen_policies == [None]


def test_runtime_explicit_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit enable_prompt_cache=False wins even with the env unset/on."""
    monkeypatch.setenv("HIMMY_PROMPT_CACHE", "1")
    mgr = _FakeAnthropic()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(mgr), enable_prompt_cache=False
    )
    persona, task = _persona_task()
    run_async(rt.run_task(persona, task))
    assert mgr.seen_policies == [None]


# ====================================================== compaction cache-buster
def test_compaction_turn_busts_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn where compaction rewrote the prefix carries NO policy (skip the breakpoint)."""
    mgr = _FakeAnthropic()
    rt = SingleAgentRuntime(inference_service=InferenceService(mgr))

    # Force _maybe_compact to report it fired on the next continuation turn.
    fired = {"count": 0}

    async def _fake_compact(*_a: Any, **_k: Any) -> bool:
        fired["count"] += 1
        return True  # pretend compaction rewrote the system prefix

    monkeypatch.setattr(rt, "_maybe_compact", _fake_compact)
    persona, task = _persona_task()
    run_async(rt.run_agent_loop(persona, task, max_turns=3))

    # Turn 1 (first turn) does NOT call _maybe_compact, so it caches; every continuation
    # turn is compaction-busted → no policy.
    assert mgr.seen_policies[0] is not None and mgr.seen_policies[0].enabled
    assert all(p is None for p in mgr.seen_policies[1:])
    assert fired["count"] >= 1


def test_runtime_metrics_reach_the_event_log_from_both_turn_kinds() -> None:
    """cache_read accrues turn-over-turn in INFERENCE_SUCCEEDED events (BOTH emit sites).

    Asserts the cache breakdown reaches the log from the FIRST-turn path
    (``_run_task_pipeline``, payload carries ``model_path``) AND the CONTINUATION path
    (``_continue_turn``, agent-tagged, no ``model_path``) — the two runtime emit sites
    C5 wires — and that the service-level emit surfaces it too.
    """
    storage = StorageService()
    mgr = _FakeAnthropic()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(mgr, event_sink=storage),
        memory_store=storage,
    )
    persona, task = _persona_task()
    run_async(rt.run_agent_loop(persona, task, max_turns=3))
    events = run_async(storage.list_events())
    succeeded = [e for e in events if e.event_type == EventType.INFERENCE_SUCCEEDED]

    # Runtime FIRST-turn emit (carries model_path) → cold cache creation.
    first_turn = [
        e for e in succeeded if e.agent_id and "model_path" in e.payload
    ]
    assert first_turn, "no first-turn runtime INFERENCE_SUCCEEDED found"
    assert any(e.payload.get("cache_creation_tokens", 0) > 0 for e in first_turn)
    assert all(
        e.payload.get("prefix_cache_requested") is True for e in first_turn
    )

    # Runtime CONTINUATION emit (agent-tagged, no model_path) → warm cache read.
    continuation = [
        e for e in succeeded if e.agent_id and "model_path" not in e.payload
    ]
    assert continuation, "no continuation-turn runtime INFERENCE_SUCCEEDED found"
    assert any(e.payload.get("cache_read_tokens", 0) > 0 for e in continuation)

    # Service-level emit (no agent_id) also surfaces the breakdown.
    service_emits = [e for e in succeeded if not e.agent_id]
    assert service_emits
    assert any(
        e.payload.get("cache_read_tokens", 0) > 0
        or e.payload.get("cache_creation_tokens", 0) > 0
        for e in service_emits
    )
