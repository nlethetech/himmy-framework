"""Context kernel: ContextService.build_snapshot — the snapshot assembly engine.

Builds an immutable :class:`ContextSnapshot` by resolving each declared key from
storage and/or registered adapters per its source preference, writing adapter
results through to storage (except ``TOOL_ONLY``), surfacing missing required keys,
and persisting the snapshot + every :class:`EvidenceRef`. When an entity registry
is wired, the snapshot and its evidence are projected into the lineage graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from himmy.core.ids import utc_now_iso
from himmy.services.context.adapters import ContextAdapter
from himmy.services.context.models import (
    ContextBuildSpec,
    ContextField,
    ContextSnapshot,
    ContextSourcePreference,
    ContextSpecKey,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.services.storage.service import StorageService


class ContextService:
    """Assembles evidenced context snapshots from storage + adapters.

    Drop any optional dependency and degrade cleanly: no registry -> no lineage;
    no adapters -> storage-only resolution.
    """

    def __init__(
        self,
        *,
        storage_service: StorageService,
        adapters: list[ContextAdapter] | None = None,
        entity_registry: EntityRegistryProtocol | None = None,
    ) -> None:
        """Wire the service with a storage backend and optional adapters/registry."""
        self._storage = storage_service
        self._adapters: dict[str, ContextAdapter] = {
            a.name: a for a in (adapters or [])
        }
        self._registry = entity_registry

    async def build_snapshot(
        self,
        *,
        subject_id: str,
        task_id: str | None = None,
        build_spec: ContextBuildSpec | dict[str, Any],
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> ContextSnapshot:
        """Build, persist, and return an immutable context snapshot.

        ``build_spec`` is accepted as a :class:`ContextBuildSpec` or a raw dict
        (validated). For each key, source preference is honored:
        ``STORAGE_FIRST`` reads storage then falls back to the adapter;
        ``TOOL_FIRST`` calls the adapter then falls back to storage;
        ``TOOL_ONLY`` never reads storage. Adapter-sourced fields are written
        through to storage except under ``TOOL_ONLY``. Required keys with no value
        land in ``missing_required_keys``.

        Tenant isolation (red-team reattack-r1/r6): ``context_fields`` are stored
        globally by ``(subject_id, key)`` with NO workspace column, so a stored field
        resolved on the build path could leak ACROSS tenants that share a free-form
        ``subject_id`` (notably the default ``persona.agent_id``, identical across tenants
        running the same agent spec). When ``workspace_id`` is supplied, a STORAGE-sourced
        field is eligible ONLY when its ``metadata.workspace_id`` EXACTLY matches — an
        unstamped or differently-stamped field is treated as not present, matching
        :meth:`ApplicationServices.list_fields`' exact metadata filter, so a snapshot built
        in workspace A can never surface workspace B's (or an ambiguous unstamped) stored
        value. Adapter-sourced fields are stamped with ``workspace_id`` on write-through so
        they too are tenant-attributable. ``workspace_id`` of ``None`` (offline / zero-config
        / all-tenants) keeps resolution byte-for-byte unchanged — every stored field is
        eligible, exactly as before.
        """
        spec = (
            build_spec
            if isinstance(build_spec, ContextBuildSpec)
            else ContextBuildSpec.model_validate(build_spec)
        )
        run_metadata = dict(metadata or {})
        scope: dict[str, Any] = {
            "subject_id": subject_id,
            "task_id": task_id,
            **run_metadata,
        }

        fields: dict[str, ContextField] = {}
        missing_required_keys: list[str] = []

        for spec_key in spec.keys:
            field = await self._resolve_key(
                spec_key, subject_id, scope, workspace_id=workspace_id
            )
            if field is not None:
                fields[spec_key.key] = field
            elif spec_key.required:
                missing_required_keys.append(spec_key.key)

        snapshot = ContextSnapshot(
            subject_id=subject_id,
            task_id=task_id,
            fields=fields,
            missing_required_keys=missing_required_keys,
            metadata=run_metadata,
        )

        await self._storage.save_snapshot(snapshot)
        await self._persist_evidence(snapshot)
        await self._register_entities(snapshot)
        return snapshot

    async def _resolve_key(
        self,
        spec_key: ContextSpecKey,
        subject_id: str,
        scope: dict[str, Any],
        *,
        workspace_id: str | None = None,
    ) -> ContextField | None:
        """Resolve one spec key honoring its source preference; write through.

        ``workspace_id`` (when supplied) tenant-scopes every STORAGE read: a stored
        field stamped with a different workspace is treated as absent, so a cross-tenant
        ``(subject_id, key)`` collision in the global ``context_fields`` store can never
        be surfaced here. ``None`` keeps resolution unscoped (offline/all-tenants).
        """
        pref = spec_key.source_preference
        key = spec_key.key
        adapter = (
            self._adapters.get(spec_key.adapter_name) if spec_key.adapter_name else None
        )
        # Per-key scope carries the spec metadata so adapters can read kb_name, etc.
        key_scope = {**scope, "spec_metadata": dict(spec_key.metadata)}

        if pref == ContextSourcePreference.TOOL_ONLY:
            # Never read storage; never write through.
            if adapter is None:
                return None
            return await adapter.fetch(key, key_scope)

        if pref == ContextSourcePreference.STORAGE_FIRST:
            stored = await self._stored_field(subject_id, key, workspace_id)
            if stored is not None and not self._is_stale(stored):
                return stored
            if adapter is None:
                # No way to refresh — a stale value still beats nothing.
                return stored
            field = await adapter.fetch(key, key_scope)
            if field is not None:
                await self._write_through(field, subject_id, workspace_id)
                return field
            # Adapter could not produce a fresh value; fall back to the stale cache.
            return stored

        # TOOL_FIRST: adapter, then storage fallback.
        if adapter is not None:
            field = await adapter.fetch(key, key_scope)
            if field is not None:
                await self._write_through(field, subject_id, workspace_id)
                return field
        return await self._stored_field(subject_id, key, workspace_id)

    async def _stored_field(
        self, subject_id: str, key: str, workspace_id: str | None
    ) -> ContextField | None:
        """Fetch a stored field, tenant-scoping it when ``workspace_id`` is supplied.

        The ``context_fields`` store is keyed globally by ``(subject_id, key)`` with no
        workspace column, so a field written by tenant B under a shared ``subject_id`` is
        physically retrievable by tenant A. When ``workspace_id`` is supplied, a stored
        field is eligible ONLY when its ``metadata.workspace_id`` EXACTLY matches (an
        unstamped or differently-stamped field is treated as absent) — closing the
        cross-tenant IDOR on the snapshot-build resolution path with the SAME exact-match
        semantics the read/list paths already apply. ``None`` is byte-unchanged (offline /
        all-tenants): every stored field is eligible.
        """
        stored = await self._storage.get_context_field(
            subject_id, key, workspace_id=workspace_id
        )
        if stored is None or workspace_id is None:
            return stored
        stamped = (getattr(stored, "metadata", {}) or {}).get("workspace_id")
        # Tenant isolation (red-team reattack-r6): when a ``workspace_id`` IS supplied, a
        # stored field must carry the SAME stamp to be eligible — an UNSTAMPED field is
        # dropped, not admitted. This matches :meth:`ApplicationServices.list_fields`' exact
        # ``metadata.workspace_id == workspace_id`` filter (the HTTP read path) so the
        # snapshot-build resolution path can never surface a field the list path would hide.
        # The previous lenient-when-unstamped rule (mirroring ``_snapshot_in_workspace``)
        # was a cross-tenant IDOR: ``_write_through`` historically cached adapter-sourced
        # fields WITHOUT a workspace stamp, so an unstamped cached value written by tenant B
        # was admitted to tenant A's snapshot even WITH a workspace filter. Cached fields are
        # now stamped (see :meth:`_write_through`), and any genuinely workspace-less field is
        # a foreign/ambiguous value a tenant-scoped build must not surface. ``workspace_id``
        # of ``None`` (offline / all-tenants) is byte-unchanged — every stored field eligible.
        if stamped != workspace_id:
            return None
        return stored

    async def _write_through(
        self, field: ContextField, subject_id: str, workspace_id: str | None = None
    ) -> None:
        """Cache an adapter-sourced field back to storage under (subject_id, key).

        Persists a *copy* with ``subject_id`` (and, when the build is tenant-scoped, the
        run's ``workspace_id``) stamped into its metadata rather than mutating the field the
        adapter returned (which is also the object stored in ``snapshot.fields`` and may be a
        cached/shared instance). This keeps the snapshot's field metadata free of the
        internal ``subject_id`` smuggling key and prevents an adapter's shared field from
        being silently rewritten.

        Tenant isolation (red-team reattack-r6): a tenant-scoped build stamps
        ``workspace_id`` so the cached field is tenant-ATTRIBUTABLE on every later read —
        :meth:`_stored_field` and :meth:`ApplicationServices.list_fields` both require an
        exact ``metadata.workspace_id`` match, so an adapter value cached by tenant B can
        never be resolved into tenant A's snapshot. ``workspace_id`` of ``None`` (offline /
        single-tenant) writes no workspace stamp — byte-unchanged.
        """
        # Stamp the cache time so STORAGE_FIRST freshness checks have a reference,
        # alongside the subject scope the storage key is derived from. Always a copy
        # so the adapter's returned (and snapshot-held) field is never mutated.
        new_metadata = {
            **field.metadata,
            "subject_id": subject_id,
            "cached_at": utc_now_iso(),
        }
        if workspace_id is not None:
            new_metadata["workspace_id"] = workspace_id
        to_store = field.model_copy(update={"metadata": new_metadata})
        await self._storage.save_context_field(to_store)

    @staticmethod
    def _is_stale(field: ContextField) -> bool:
        """True when a stored field has a freshness TTL it has outlived.

        Freshness is measured from the ``cached_at`` timestamp stamped at
        write-through. A field without ``freshness_seconds`` never expires; a field
        with a TTL but no ``cached_at`` (e.g. a hand-seeded storage value) is treated
        as fresh rather than perpetually re-fetched.
        """
        ttl = field.freshness_seconds
        if ttl is None or ttl <= 0:
            return False
        cached_at = field.metadata.get("cached_at")
        if not cached_at:
            return False
        try:
            stamped = datetime.fromisoformat(str(cached_at))
        except ValueError:  # pragma: no cover - defensive against bad timestamps
            return False
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - stamped).total_seconds()
        return age > ttl

    async def _persist_evidence(self, snapshot: ContextSnapshot) -> None:
        """Persist each field's EvidenceRef to the storage evidence stream."""
        from himmy.services.storage.models import ContextEvidenceRecord

        for key, field in snapshot.fields.items():
            for ref in field.evidence_refs:
                record = ContextEvidenceRecord(
                    evidence_id=ref.evidence_id,
                    subject_id=snapshot.subject_id,
                    snapshot_id=snapshot.snapshot_id,
                    key=key,
                    payload=ref.model_dump(mode="json"),
                    metadata=ref.metadata,
                )
                await self._storage.save_context_evidence(record)

    async def _register_entities(self, snapshot: ContextSnapshot) -> None:
        """Project the snapshot + its evidence into the entity registry, if present."""
        if self._registry is None:
            return
        snapshot_record = snapshot.to_record()
        self._registry.register(snapshot_record)
        for key, field in snapshot.fields.items():
            for ref in field.evidence_refs:
                from himmy.entities.projection import project

                evidence_record = project(
                    ref,
                    stable_value=ref.evidence_id,
                    namespace="context_evidence",
                    kind="context_evidence",
                    version=1,
                    metadata={"key": key, "snapshot_id": snapshot.snapshot_id},
                )
                self._registry.register(evidence_record)
                self._registry.link(
                    from_record_id=snapshot_record.record_id,
                    to_record_id=evidence_record.record_id,
                    relation="built_from",
                    metadata={"key": key},
                )


__all__ = ["ContextService"]
