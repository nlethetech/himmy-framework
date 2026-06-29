"""Tests for the context kernel: snapshot assembly, source preference, write-through."""

from __future__ import annotations

from himmy.entities.registry import EntityRegistry
from himmy.services.context import (
    ContextAdapter,
    ContextBuildSpec,
    ContextField,
    ContextService,
    ContextSnapshot,
    ContextSourcePreference,
    ContextSpecKey,
    EvidenceRef,
)
from himmy.services.storage.service import StorageService
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


# --------------------------------------------------------------------------- #
# red-team reattack-r6 — cross-tenant context-field IDOR on the BUILD path.
#
# ``context_fields`` are keyed globally by ``(subject_id, key)`` with no workspace
# column, and the per-run ``subject_id`` defaults to the SHARED ``persona.agent_id``
# across tenants running the same agent spec. A tenant-scoped ``build_snapshot`` must
# therefore resolve ONLY fields stamped with the caller's workspace — matching the HTTP
# list/read paths' exact ``metadata.workspace_id`` filter. Two legs were leaking:
#   * LEG 1 — an UNSTAMPED stored field was admitted even WITH a workspace filter (the
#     old lenient-when-unstamped rule), so tenant B's value surfaced to tenant A.
#   * LEG 2 — adapter-cached fields were written through with NO workspace stamp, so
#     tenant B's adapter value became an unstamped (leg-1-eligible) cache row.
# --------------------------------------------------------------------------- #


def test_build_snapshot_drops_unstamped_field_under_workspace_filter() -> None:
    """LEG 1: an unstamped stored field is NOT surfaced to a tenant-scoped build.

    Tenant B writes a value under the shared ``subject_id`` with NO workspace stamp (as a
    legacy / ambiguous row). Tenant A builds a snapshot for the SAME subject scoped to
    workspace_A — the unstamped value must be treated as absent (dropped), never leaked.
    """
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    # Tenant B's unstamped secret under the shared subject_id.
    run_async(
        storage.save_context_field(
            ContextField(
                key="api_key",
                value="TENANT_B_SECRET",
                metadata={"subject_id": "shared_subject"},
            )
        )
    )
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="api_key")])
    snap = run_async(
        svc.build_snapshot(
            subject_id="shared_subject", build_spec=spec, workspace_id="workspace_A"
        )
    )
    assert "api_key" not in snap.fields  # unstamped → dropped, not leaked


def test_build_snapshot_drops_foreign_stamped_field_under_workspace_filter() -> None:
    """A field stamped with a DIFFERENT workspace is never surfaced (defense in depth)."""
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    run_async(
        storage.save_context_field(
            ContextField(
                key="api_key",
                value="TENANT_B_STAMPED",
                metadata={"subject_id": "shared_subject", "workspace_id": "workspace_B"},
            )
        )
    )
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="api_key")])
    snap = run_async(
        svc.build_snapshot(
            subject_id="shared_subject", build_spec=spec, workspace_id="workspace_A"
        )
    )
    assert "api_key" not in snap.fields


def test_build_snapshot_surfaces_own_workspace_field_no_false_lockout() -> None:
    """No false lockout: a field stamped with the caller's own workspace IS surfaced."""
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    run_async(
        storage.save_context_field(
            ContextField(
                key="api_key",
                value="TENANT_A_OWN",
                metadata={"subject_id": "shared_subject", "workspace_id": "workspace_A"},
            )
        )
    )
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="api_key")])
    snap = run_async(
        svc.build_snapshot(
            subject_id="shared_subject", build_spec=spec, workspace_id="workspace_A"
        )
    )
    assert snap.fields["api_key"].value == "TENANT_A_OWN"


def test_write_through_stamps_workspace_so_cache_is_tenant_scoped() -> None:
    """LEG 2: an adapter value cached for tenant B is NOT resolvable for tenant A.

    Tenant B runs a STORAGE_FIRST key whose value comes from an adapter; write-through must
    stamp ``workspace_id=workspace_B`` so a later tenant-A build for the same subject/key
    (no adapter to refresh) sees the cache as foreign and drops it — closing the leg where
    an unstamped cache row was admitted to any workspace.
    """
    storage = StorageService()
    adapter_b = _StaticAdapter("TENANT_B_ADAPTER")
    svc_b = ContextService(storage_service=storage, adapters=[adapter_b])
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="k", adapter_name="static")])
    # Tenant B builds → adapter fires → write-through caches the value.
    snap_b = run_async(
        svc_b.build_snapshot(
            subject_id="shared_subject", build_spec=spec, workspace_id="workspace_B"
        )
    )
    assert snap_b.fields["k"].value == "TENANT_B_ADAPTER"
    cached = run_async(storage.get_context_field("shared_subject", "k"))
    assert cached is not None
    # The cache row is tenant-attributable now.
    assert (cached.metadata or {}).get("workspace_id") == "workspace_B"
    # Tenant A builds for the SAME subject/key with NO adapter (cannot refresh) → must NOT
    # surface tenant B's cached value.
    svc_a = ContextService(storage_service=storage)
    snap_a = run_async(
        svc_a.build_snapshot(
            subject_id="shared_subject",
            build_spec=ContextBuildSpec(keys=[ContextSpecKey(key="k")]),
            workspace_id="workspace_A",
        )
    )
    assert "k" not in snap_a.fields


def test_build_snapshot_unscoped_resolution_byte_unchanged() -> None:
    """INVARIANT: with no workspace_id (offline / single-tenant) every stored field resolves.

    An unstamped storage value is resolved exactly as before — the tightened exact-match
    only engages when a ``workspace_id`` is supplied, so the offline / CLI path is unchanged.
    """
    storage = StorageService()
    svc = ContextService(storage_service=storage)
    run_async(
        storage.save_context_field(
            ContextField(
                key="api_key", value="any", metadata={"subject_id": "s1"}
            )
        )
    )
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="api_key")])
    snap = run_async(svc.build_snapshot(subject_id="s1", build_spec=spec))
    assert snap.fields["api_key"].value == "any"
