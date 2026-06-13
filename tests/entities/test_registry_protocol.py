"""T0.1 — :class:`EntityRegistryProtocol` is satisfied by BOTH registry backends.

These tests pin the structural-typing refactor that unblocks the durable-spine swap
(plan item T0.1 → T2.1). The headline guarantee: a single Protocol type now describes
the registry surface every consumer depends on, so a durable
:class:`SqliteEntityRegistry` can be injected anywhere the in-memory
:class:`EntityRegistry` was previously required — with NO code change at the swap point.

The load-bearing regression is :func:`test_pydantic_wall_now_accepts_sqlite_registry`:
before this refactor, :class:`RecordedDataContext` annotated its ``entity_registry`` field
with the concrete :class:`EntityRegistry`, so Pydantic's ``arbitrary_types_allowed``
validation ran ``isinstance(value, EntityRegistry)`` and REJECTED a durable
:class:`SqliteEntityRegistry` at runtime (a ``ValidationError``). That is the exact
"nominal-type wall" this unit tears down — the assertion below FAILED before the refactor.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from himmy.entities.protocol import EntityRegistryProtocol
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.entities.sqlite_registry import SqliteEntityRegistry

#: The full method surface the Protocol promises (every consumer is allowed to call
#: only these). Kept explicit so a future widening of the Protocol that an impl forgets
#: to satisfy is caught here, not deep in a consumer.
_PROTOCOL_METHODS = (
    "register",
    "new_version",
    "link",
    "get",
    "get_latest",
    "get_history",
    "list_by_kind",
    "query",
    "links_from",
    "links_to",
    "neighbors",
    "trace",
)


@pytest.fixture
def sqlite_registry() -> Iterator[SqliteEntityRegistry]:
    """A durable (in-file ``:memory:``) SQLite registry, closed after the test."""
    reg = SqliteEntityRegistry(":memory:")
    yield reg
    reg.close()


def test_inmemory_registry_satisfies_protocol() -> None:
    """The volatile :class:`EntityRegistry` is a structural member of the Protocol."""
    reg = EntityRegistry()
    assert isinstance(reg, EntityRegistryProtocol)
    assert issubclass(EntityRegistry, EntityRegistryProtocol)


def test_sqlite_registry_satisfies_protocol(
    sqlite_registry: SqliteEntityRegistry,
) -> None:
    """The durable :class:`SqliteEntityRegistry` is a structural member of the Protocol."""
    assert isinstance(sqlite_registry, EntityRegistryProtocol)
    assert issubclass(SqliteEntityRegistry, EntityRegistryProtocol)


@pytest.mark.parametrize("impl", [EntityRegistry, SqliteEntityRegistry])
def test_both_impls_expose_every_protocol_method(impl: type) -> None:
    """Every method the Protocol promises is present (callable) on each backend."""
    for name in _PROTOCOL_METHODS:
        assert callable(getattr(impl, name, None)), f"{impl.__name__} missing {name!r}"


def test_pydantic_wall_now_accepts_sqlite_registry(
    sqlite_registry: SqliteEntityRegistry,
) -> None:
    """The previously-failing assertion: a durable registry validates into the model.

    ``RecordedDataContext.entity_registry`` is now typed ``EntityRegistryProtocol | None``;
    because the Protocol is ``runtime_checkable``, Pydantic's ``arbitrary_types_allowed``
    membership check passes for BOTH backends. With the old concrete ``EntityRegistry``
    annotation this raised ``ValidationError`` for the SQLite registry.
    """
    from himmy.services.evaluation.privacy.models import RecordedDataContext

    # Both backends are accepted; the SQLite one is the regression that used to fail.
    for reg in (EntityRegistry(), sqlite_registry):
        ctx = RecordedDataContext(entity_registry=reg)
        assert ctx.entity_registry is reg


def test_pydantic_wall_still_rejects_a_non_registry() -> None:
    """The structural gate is real: an object lacking the surface is still rejected.

    A guard against the Protocol degenerating into "accepts anything" — a plain object
    that does not implement the registry methods must still fail validation.
    """
    from himmy.services.evaluation.privacy.models import RecordedDataContext

    with pytest.raises(ValidationError):
        # Intentionally invalid: a bare object is not a registry. mypy flags the type
        # mismatch (which is correct — that is the static half of the same guarantee);
        # this test asserts the RUNTIME Pydantic validation also rejects it.
        RecordedDataContext(entity_registry=object())  # type: ignore[arg-type]


@pytest.mark.parametrize("registry_kind", ["inmemory", "sqlite"])
def test_protocol_typed_consumers_accept_either_backend(registry_kind: str) -> None:
    """A consumer typed against the Protocol works identically on either backend.

    Exercises the real consumer seams retyped in this unit (``SecurityAuditLog`` over the
    audit spine; ``ConsentLedger`` over the consent spine) with BOTH backends, proving the
    durable registry drops in where the in-memory one was required — no cast, no branch.
    """
    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.audit.models import SecurityEvent
    from himmy.services.governance.consent import (
        ConsentPolicy,
        ConsentState,
        Purpose,
    )
    from himmy.services.governance.consent_ledger import ConsentLedger

    registry: EntityRegistryProtocol
    if registry_kind == "inmemory":
        registry = EntityRegistry()
    else:
        registry = SqliteEntityRegistry(":memory:")

    try:
        # Audit spine: write-then-read a security event through the Protocol surface.
        audit = SecurityAuditLog(registry)
        audit.record(
            SecurityEvent(event_type="authz_denied", outcome="deny", actor={})
        )
        recent = audit.recent(limit=10)
        assert len(recent) == 1
        assert recent[0].event_type == "authz_denied"

        # Consent spine: append a version and read it back (new_version/get_latest path).
        ledger = ConsentLedger(registry, policy=ConsentPolicy(governed=True))
        ledger.set("subject-1", Purpose.RETAIN, ConsentState.GRANTED, actor="t")
        assert ledger.state("subject-1", Purpose.RETAIN) is ConsentState.GRANTED
    finally:
        if isinstance(registry, SqliteEntityRegistry):
            registry.close()


def test_protocol_round_trips_records_on_both_backends(
    sqlite_registry: SqliteEntityRegistry,
) -> None:
    """A record registered via the Protocol surface reads back on either backend."""
    record = EntityRecord.create(
        stable_id="art-1", version=1, kind="demo", payload={"k": "v"}
    )
    for reg in (EntityRegistry(), sqlite_registry):
        registry: EntityRegistryProtocol = reg
        stored = registry.register(record)
        assert stored.record_id == record.record_id
        assert registry.get(record.record_id) is not None
        assert [r.record_id for r in registry.list_by_kind("demo")] == [
            record.record_id
        ]
