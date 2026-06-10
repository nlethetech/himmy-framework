"""Tenant-scoped response cache — no cross-tenant cache leaks.

The response cache key carries a *cache scope* derived from the request's
``metadata`` (``cache_scope`` / ``tenant_id`` / ``workspace_id`` / ``subject_id``),
so two principals issuing identical prompts can never share a cached response.
These tests pin the isolation contract:

* same prompt + different scope -> distinct keys, NO cross-scope cache hit;
* same prompt + same scope -> cache hit (one provider call);
* unscoped requests keep the original single-tenant behavior AND byte-identical
  keys (so previously recorded replay cassettes still match);
* every scope dimension partitions independently, and combinations never alias.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from himmy.services.inference import (
    InferenceMessage,
    InferenceRequest,
    InferenceService,
    InMemoryTTLCache,
    StubClientManager,
    compute_cache_key,
    derive_cache_scope,
)
from tests.conftest import run_async


# ----------------------------------------------------------------- local helpers
class _CountingManager(StubClientManager):
    """A stub manager that counts provider hits."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate(self, request: InferenceRequest) -> Any:
        self.calls += 1
        return await super().generate(request)


def _svc(manager: _CountingManager) -> InferenceService:
    """Build an InferenceService with a real cache and instant retries."""
    return InferenceService(
        manager,
        cache=InMemoryTTLCache(ttl_seconds=60.0),
        retry_base_delay_seconds=0.0,
        retry_jitter_seconds=0.0,
    )


def _req(content: str = "same prompt", **metadata: Any) -> InferenceRequest:
    """An opted-in cacheable request carrying the given scope metadata."""
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content=content)],
        generation_params={"use_cache": True},
        metadata=dict(metadata),
    )


# ------------------------------------------------------------------- key derivation
def test_different_scope_yields_distinct_keys() -> None:
    """Identical prompts under different tenants must NOT share a cache key."""
    a = _req(tenant_id="acme")
    b = _req(tenant_id="globex")
    assert compute_cache_key(a) != compute_cache_key(b)


def test_each_scope_dimension_partitions_the_key() -> None:
    """cache_scope / tenant_id / workspace_id / subject_id each flip the key."""
    base = compute_cache_key(_req())
    keys = {base}
    for meta in (
        {"cache_scope": "s1"},
        {"tenant_id": "t1"},
        {"workspace_id": "w1"},
        {"subject_id": "u1"},
    ):
        keys.add(compute_cache_key(_req(**meta)))
    # base + four distinctly scoped variants = five distinct keys.
    assert len(keys) == 5


def test_scope_combinations_never_alias() -> None:
    """tenant+workspace combos are distinct from either dimension alone."""
    combo = compute_cache_key(_req(tenant_id="t1", workspace_id="w1"))
    assert combo != compute_cache_key(_req(tenant_id="t1"))
    assert combo != compute_cache_key(_req(workspace_id="w1"))
    # The pair (t1, w1) must not collide with (t1w, 1) style smearing either.
    assert combo != compute_cache_key(_req(tenant_id="t1w", workspace_id="1"))


def test_unscoped_key_unchanged_and_collides_as_before() -> None:
    """No scope metadata -> the pre-scope key (single-tenant default preserved)."""
    a = _req()
    b = _req()
    assert compute_cache_key(a) == compute_cache_key(b)
    # The scope dimension is OMITTED when absent, so the key is byte-identical
    # to the historical derivation (replay cassettes keep matching).
    payload = {
        "model_key": a.model_key,
        "route_override": a.route_override,
        "response_format": None,
        "output_json_schema": a.output_json_schema,
        "workflow": None,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
            for m in a.messages
        ],
        "tool_names": [],
        "generation_params": {},
    }
    legacy = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert compute_cache_key(a) == legacy


def test_non_scope_metadata_does_not_partition() -> None:
    """Unrelated metadata keys (e.g. trace ids) do not fragment the cache."""
    assert compute_cache_key(_req(trace_id="abc")) == compute_cache_key(_req())


def test_empty_or_none_scope_values_are_unscoped() -> None:
    """Empty-string / None scope values are treated as 'no scope', not a scope."""
    assert derive_cache_scope(_req(tenant_id="")) is None
    assert derive_cache_scope(_req(tenant_id=None)) is None
    assert compute_cache_key(_req(tenant_id="")) == compute_cache_key(_req())


def test_derive_cache_scope_folds_all_present_keys() -> None:
    """All present scope keys contribute, in stable precedence order."""
    scope = derive_cache_scope(
        _req(cache_scope="s", tenant_id="t", workspace_id="w", subject_id="u")
    )
    assert scope == "cache_scope=s|tenant_id=t|workspace_id=w|subject_id=u"
    assert derive_cache_scope(_req()) is None


# ------------------------------------------------------------------ service behavior
def test_no_cross_scope_cache_hit_through_service() -> None:
    """Tenant B never receives tenant A's cached response for the same prompt."""
    mgr = _CountingManager()
    svc = _svc(mgr)
    first = run_async(svc.run(_req(tenant_id="acme")))
    second = run_async(svc.run(_req(tenant_id="globex")))
    # Two provider calls: the second tenant was NOT served tenant A's entry.
    assert mgr.calls == 2
    assert first.metadata.get("cache_hit") is not True
    assert second.metadata.get("cache_hit") is not True


def test_same_scope_hits_warm_cache() -> None:
    """The same tenant repeating the same prompt is served from cache."""
    mgr = _CountingManager()
    svc = _svc(mgr)
    first = run_async(svc.run(_req(tenant_id="acme")))
    second = run_async(svc.run(_req(tenant_id="acme")))
    assert mgr.calls == 1
    assert second.metadata.get("cache_hit") is True
    assert second.output_text == first.output_text


def test_unscoped_default_behavior_unchanged() -> None:
    """Requests without scope metadata still share the cache (zero-config path)."""
    mgr = _CountingManager()
    svc = _svc(mgr)
    run_async(svc.run(_req()))
    second = run_async(svc.run(_req()))
    assert mgr.calls == 1
    assert second.metadata.get("cache_hit") is True


def test_scoped_and_unscoped_do_not_share() -> None:
    """An unscoped request never reads a scoped entry, and vice versa."""
    mgr = _CountingManager()
    svc = _svc(mgr)
    run_async(svc.run(_req(tenant_id="acme")))
    unscoped = run_async(svc.run(_req()))
    assert mgr.calls == 2
    assert unscoped.metadata.get("cache_hit") is not True
