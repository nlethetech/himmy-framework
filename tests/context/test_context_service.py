"""Tests for the context kernel: snapshot assembly, source preference, write-through."""

from __future__ import annotations

from opensims.entities.registry import EntityRegistry
from opensims.services.context import (
    ContextAdapter,
    ContextBuildSpec,
    ContextField,
    ContextService,
    ContextSnapshot,
    ContextSourcePreference,
    ContextSpecKey,
    EvidenceRef,
)
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


class _StaticAdapter(ContextAdapter):
    """A test adapter that returns a fixed field with one evidence ref."""

    name = "static"

    def __init__(self, value: str) -> None:
        self._value = value
        self.calls = 0

    async def fetch(self, key: str, scope: dict) -> ContextField | None:
        self.calls += 1
        return ContextField(
            key=key,
            value=self._value,
            source="static",
            confidence=0.9,
            evidence_refs=[EvidenceRef(source_type="static", source_id="row-1")],
        )


def test_storage_first_reads_storage_then_falls_back_to_adapter() -> None:
    """STORAGE_FIRST returns stored value when present, else the adapter result."""
    storage = StorageService()
    adapter = _StaticAdapter("from-adapter")
    svc = ContextService(storage_service=storage, adapters=[adapter])

    # Pre-seed storage for 'cached' so the adapter is not consulted.
    stored = ContextField(
        key="cached", value="from-storage", metadata={"subject_id": "s1"}
    )
    run_async(storage.save_context_field(stored))

    spec = ContextBuildSpec(
        keys=[
            ContextSpecKey(key="cached", adapter_name="static"),
            ContextSpecKey(key="fresh", adapter_name="static"),
        ]
    )
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert snap.fields["cached"].value == "from-storage"
    assert snap.fields["fresh"].value == "from-adapter"
    # Adapter was only consulted for the un-cached key.
    assert adapter.calls == 1


def test_adapter_result_is_written_through_to_storage() -> None:
    """A STORAGE_FIRST adapter hit is cached back to storage (subject-scoped)."""
    storage = StorageService()
    adapter = _StaticAdapter("v")
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="k", adapter_name="static")])
    run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    cached = run_async(storage.get_context_field("s1", "k"))
    assert cached is not None and cached.value == "v"


def test_tool_only_never_writes_through() -> None:
    """TOOL_ONLY uses the adapter but never persists the field."""
    storage = StorageService()
    adapter = _StaticAdapter("ephemeral")
    svc = ContextService(storage_service=storage, adapters=[adapter])
    spec = ContextBuildSpec(
        keys=[
            ContextSpecKey(
                key="k",
                adapter_name="static",
                source_preference=ContextSourcePreference.TOOL_ONLY,
            )
        ]
    )
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert snap.fields["k"].value == "ephemeral"
    # Not written through.
    assert run_async(storage.get_context_field("s1", "k")) is None


def test_missing_required_key_is_reported() -> None:
    """A required key with no resolvable value lands in missing_required_keys."""
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="needed", required=True)])
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert "needed" in snap.missing_required_keys
    assert "needed" not in snap.fields


def test_snapshot_persisted_and_registered() -> None:
    """The snapshot is saved to storage and projected into the registry."""
    storage = StorageService()
    registry = EntityRegistry()
    adapter = _StaticAdapter("v")
    svc = ContextService(
        storage_service=storage, adapters=[adapter], entity_registry=registry
    )
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="k", adapter_name="static")])
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))

    assert isinstance(snap, ContextSnapshot)
    assert run_async(storage.load_snapshot(snap.snapshot_id)) is not None
    # The snapshot entity record exists.
    records = registry.list_by_kind("context_snapshot")
    assert len(records) == 1
    # Evidence is projected and linked from the snapshot.
    evidence = registry.list_by_kind("context_evidence")
    assert len(evidence) == 1


def test_build_spec_accepts_raw_dict() -> None:
    """build_snapshot accepts a raw dict build spec (model_validate path)."""
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    snap = run_async(
        svc.build_snapshot(
            subject_id="s1",
            build_spec={"keys": [{"key": "x", "required": True}]},
        )
    )
    assert "x" in snap.missing_required_keys
