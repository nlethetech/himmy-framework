"""Entities kernel: immutable records, links, queries, and deterministic id helpers."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from himmy.core.ids import new_uuid, utc_now_iso

# Fixed UUID5 namespace so derived ids are stable across processes and machines.
_HIMMY_NAMESPACE = uuid.UUID("6f1d3a2e-7c4b-5a9e-8d12-0a1b2c3d4e5f")


def _is_uuid(value: str) -> bool:
    """Return True when ``value`` parses as a UUID of any version."""
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def stable_id_for(
    value: str, *, namespace: str, fallback_key: str | None = None
) -> str:
    """Derive a deterministic stable id for a semantic key.

    If ``value`` already parses as a UUID it is returned unchanged so external
    ids round-trip. Otherwise the result is ``UUID5(namespace, value)``. When
    ``value`` is falsy, ``fallback_key`` is used instead.
    """
    key = value or fallback_key or ""
    if _is_uuid(key):
        return str(key)
    return str(uuid.uuid5(_HIMMY_NAMESPACE, f"{namespace}:{key}"))


def record_id_for(*, stable_id: str, version: int, kind: str) -> str:
    """Derive the deterministic physical record id from (kind, stable_id, version)."""
    return str(uuid.uuid5(_HIMMY_NAMESPACE, f"{kind}:{stable_id}:{version}"))


def metadata_contains(haystack: Any, needle: Any) -> bool:
    """Return True when ``haystack`` JSONB-contains ``needle`` (mirrors PG ``@>``).

    Dicts match when every key in ``needle`` is present in ``haystack`` with a
    containing value; lists match when every element of ``needle`` is contained by
    some element of ``haystack``; scalars match on equality. This is the in-memory
    equivalent of the Postgres ``metadata @> $1::jsonb`` containment operator so
    both registry backends evaluate ``EntityQuery.metadata_filters`` identically.
    """
    if isinstance(needle, dict):
        if not isinstance(haystack, dict):
            return False
        return all(
            k in haystack and metadata_contains(haystack[k], v)
            for k, v in needle.items()
        )
    if isinstance(needle, list):
        if not isinstance(haystack, list):
            return False
        return all(
            any(metadata_contains(item, sub) for item in haystack) for sub in needle
        )
    return haystack == needle


class EntityRecord(BaseModel):
    """An immutable, content-addressed, versioned projection of a domain artefact.

    ``record_id`` is computed deterministically from ``(kind, stable_id, version)``
    so re-registering identical content is idempotent. The model is ``frozen`` so
    that ``record_id`` can never drift out of sync with the triple it was derived
    from; create a new version (``EntityRegistry.new_version``) to evolve content.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = ""
    stable_id: str
    version: int = 1
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    def model_post_init(self, __context: Any) -> None:
        """Fill in a deterministic ``record_id`` when one was not provided.

        Frozen models block normal attribute assignment, but ``object.__setattr__``
        bypasses the freeze during construction, which is the documented way to
        finalise a derived field on an otherwise immutable model.
        """
        if not self.record_id:
            object.__setattr__(
                self,
                "record_id",
                record_id_for(
                    stable_id=self.stable_id, version=self.version, kind=self.kind
                ),
            )

    @classmethod
    def create(
        cls,
        *,
        stable_id: str,
        version: int,
        kind: str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EntityRecord:
        """Build a record with a deterministically computed ``record_id``."""
        return cls(
            record_id=record_id_for(stable_id=stable_id, version=version, kind=kind),
            stable_id=stable_id,
            version=version,
            kind=kind,
            payload=payload or {},
            metadata=metadata or {},
        )


class EntityLink(BaseModel):
    """A typed directed relationship between two ``EntityRecord``s."""

    link_id: str = Field(default_factory=new_uuid)
    from_record_id: str
    to_record_id: str
    relation: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class EntityQuery(BaseModel):
    """A simple filter over the registry by kind, metadata, and/or stable id.

    ``metadata_filters`` is JSONB-containment semantics: a record matches when its
    ``metadata`` contains every supplied key with an equal value (nested dicts are
    compared by containment, mirroring the Postgres ``@>`` operator). ``kind`` and
    ``stable_id`` are optional, so metadata-only and stable_id-only queries are
    valid (no ``kind`` is required).
    """

    kind: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    stable_id: str | None = None
