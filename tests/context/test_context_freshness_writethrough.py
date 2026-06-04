"""Context kernel hardening: freshness re-fetch (CK-4) + write-through copy (CK-9).

These lock the production behavior the IMPROVEMENTS audit flagged:

* CK-4 — ``STORAGE_FIRST`` must honor ``freshness_seconds``: a stored field past
  its TTL falls through to the adapter and is re-cached, while a fresh value (or a
  value with no TTL, or a TTL but no ``cached_at`` stamp) is served from cache.
* CK-9 — write-through must persist a *copy*: the ``ContextField`` the adapter
  returned (which is also the object held in ``snapshot.fields``) must never be
  mutated, and the snapshot's field metadata must stay free of the internal
  ``subject_id`` smuggling key while the *stored* copy carries it.

All tests are plain ``def test_*`` driving async via the local ``run_async`` shim;
the whole module is offline (in-memory storage, no provider/DB).
"""

from __future__ import annotations

import time

from opensims.services.context.adapters import ContextAdapter
from opensims.services.context.models import (
    ContextBuildSpec,
    ContextField,
    ContextSourcePreference,
    ContextSpecKey,
    EvidenceRef,
)
from opensims.services.context.service import ContextService
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


class _CountingAdapter(ContextAdapter):
    """An adapter that returns a fresh value each call, tagged with the call count.

    ``freshness_seconds`` is configurable so a test can make the cached value
    expire quickly (CK-4) or never (default).
    """

    name = "counter"

    def __init__(self, *, freshness_seconds: float | None = None) -> None:
        self.calls = 0
        self._freshness = freshness_seconds

    async def fetch(self, key: str, scope: dict) -> ContextField | None:
        self.calls += 1
        return ContextField(
            key=key,
            value=f"v{self.calls}",
            source="counter",
            confidence=0.9,
            freshness_seconds=self._freshness,
            evidence_refs=[
                EvidenceRef(source_type="counter", source_id=f"r{self.calls}")
            ],
        )


class _SharedFieldAdapter(ContextAdapter):
    """An adapter that returns the *same* cached field instance on every call.

    This is the realistic pattern CK-9 protects against: if write-through mutated
    the returned object, the adapter's shared cache would be silently corrupted.
    """

    name = "shared"

    def __init__(self) -> None:
        self.field = ContextField(
            key="shared",
            value="cached",
            source="shared",
            metadata={"origin": "adapter-cache"},
            evidence_refs=[EvidenceRef(source_type="shared", source_id="row")],
        )

    async def fetch(self, key: str, scope: dict) -> ContextField | None:
        return self.field


def _spec(key: str, *, adapter: str | None) -> ContextBuildSpec:
    return ContextBuildSpec(keys=[ContextSpecKey(key=key, adapter_name=adapter)])


# --------------------------------------------------------------------- CK-4
def test_fresh_value_within_ttl_is_served_from_cache() -> None:
    """A stored field still inside its TTL is NOT re-fetched from the adapter."""
    storage = StorageService()
    adapter = _CountingAdapter(freshness_seconds=3600.0)  # 1h TTL — stays fresh
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = _spec("k", adapter="counter")

    s1 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert s1.fields["k"].value == "v1"
    s2 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    # Within TTL: the cache is reused, adapter not consulted a second time.
    assert adapter.calls == 1
    assert s2.fields["k"].value == "v1"


def test_stale_value_past_ttl_triggers_refetch_and_recache() -> None:
    """A stored field past its TTL falls through to the adapter and is re-cached."""
    storage = StorageService()
    adapter = _CountingAdapter(freshness_seconds=0.0001)  # expires almost instantly
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = _spec("k", adapter="counter")

    s1 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert s1.fields["k"].value == "v1"
    time.sleep(0.01)  # exceed the 0.0001s TTL
    s2 = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert adapter.calls == 2
    assert s2.fields["k"].value == "v2"
    # The fresh value was re-cached: it round-trips from storage.
    cached = run_async(storage.get_context_field("s1", "k"))
    assert cached is not None and cached.value == "v2"


def test_field_without_ttl_is_never_treated_as_stale() -> None:
    """A stored field with no ``freshness_seconds`` is always served from cache."""
    storage = StorageService()
    adapter = _CountingAdapter(freshness_seconds=None)  # no TTL
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = _spec("k", adapter="counter")

    run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    time.sleep(0.01)
    run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    # No TTL -> never expires -> adapter consulted exactly once.
    assert adapter.calls == 1


def test_ttl_without_cached_at_is_treated_as_fresh() -> None:
    """A hand-seeded field with a TTL but no ``cached_at`` is not re-fetched forever."""
    storage = StorageService()
    adapter = _CountingAdapter(freshness_seconds=10.0)
    svc = ContextService(storage_service=storage, adapters=[adapter])

    # Seed storage directly (no cached_at stamp), as a hand-written cache would be.
    seeded = ContextField(
        key="k",
        value="seeded",
        source="storage",
        freshness_seconds=10.0,
        metadata={"subject_id": "s1"},  # no cached_at
    )
    run_async(storage.save_context_field(seeded))

    spec = _spec("k", adapter="counter")
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    # Treated as fresh (defensive): cache served, adapter not consulted.
    assert snap.fields["k"].value == "seeded"
    assert adapter.calls == 0


def test_storage_first_no_adapter_returns_stale_rather_than_nothing() -> None:
    """With no adapter to refresh, an expired cached value still beats nothing."""
    storage = StorageService()
    svc = ContextService(storage_service=storage)  # no adapters

    # Seed a field that is already stale (TTL elapsed) with a cached_at in the past.
    from opensims.core.ids import utc_now_iso

    stale = ContextField(
        key="k",
        value="old",
        source="storage",
        freshness_seconds=0.0001,
        metadata={"subject_id": "s1", "cached_at": "2000-01-01T00:00:00+00:00"},
    )
    # Sanity: cached_at far in the past so the field is unambiguously stale now.
    assert stale.metadata["cached_at"] < utc_now_iso()
    run_async(storage.save_context_field(stale))

    spec = ContextBuildSpec(
        keys=[ContextSpecKey(key="k")]  # STORAGE_FIRST, no adapter
    )
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    # No way to refresh -> the stale cache is returned (not dropped/missing).
    assert "k" in snap.fields
    assert snap.fields["k"].value == "old"


# --------------------------------------------------------------------- CK-9
def test_write_through_does_not_mutate_adapter_returned_field() -> None:
    """The adapter's shared field object is never mutated by write-through."""
    storage = StorageService()
    adapter = _SharedFieldAdapter()
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = _spec("shared", adapter="shared")

    before = dict(adapter.field.metadata)
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))

    # The adapter's own field instance is untouched (no subject_id smuggled in).
    assert adapter.field.metadata == before
    assert "subject_id" not in adapter.field.metadata
    assert "cached_at" not in adapter.field.metadata
    # The snapshot holds the same un-mutated field shape.
    assert snap.fields["shared"].metadata == before
    assert "subject_id" not in snap.fields["shared"].metadata


def test_write_through_stores_a_copy_with_subject_and_cached_at() -> None:
    """The *stored* copy carries subject_id + cached_at; the live field does not."""
    storage = StorageService()
    adapter = _SharedFieldAdapter()
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = _spec("shared", adapter="shared")

    run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    cached = run_async(storage.get_context_field("s1", "shared"))
    assert cached is not None
    # The stored copy is scoped + timestamped...
    assert cached.metadata.get("subject_id") == "s1"
    assert cached.metadata.get("cached_at")
    # ...but it is a distinct object from the adapter's shared field.
    assert cached is not adapter.field
    # The original adapter metadata keys are preserved in the copy.
    assert cached.metadata.get("origin") == "adapter-cache"


def test_tool_only_field_is_not_mutated_and_not_persisted() -> None:
    """TOOL_ONLY keeps the field pristine and never writes it through."""
    storage = StorageService()
    adapter = _SharedFieldAdapter()
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = ContextBuildSpec(
        keys=[
            ContextSpecKey(
                key="shared",
                adapter_name="shared",
                source_preference=ContextSourcePreference.TOOL_ONLY,
            )
        ]
    )
    before = dict(adapter.field.metadata)
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert snap.fields["shared"].value == "cached"
    # Not written through, and the shared field is untouched.
    assert run_async(storage.get_context_field("s1", "shared")) is None
    assert adapter.field.metadata == before
