"""Property tests: deterministic entity id derivation (entities kernel).

UUID5 derivation must be a pure function — same inputs always give the same id,
distinct inputs give distinct ids, external UUIDs round-trip unchanged, and
unicode / empty-string keys never crash. Prefix components (namespaces, kinds,
stable ids) are escaped before joining, so the derivation string is unambiguous
even for inputs containing ``:`` or ``\\`` — while colon/backslash-free prefixes
(all real usage) derive byte-identical ids to the pre-escape formula (pinned).
Skips gracefully when ``hypothesis`` is not installed (dev-only dependency).
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from himmy.entities.records import (  # noqa: E402
    EntityRecord,
    record_id_for,
    stable_id_for,
)

# Moderate example counts, no deadline, derandomized => fast and seed-stable in CI.
_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True, database=None)


def _parses_as_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


#: Code-controlled identifiers (namespaces/kinds): arbitrary unicode INCLUDING
#: ``:`` and ``\\`` — prefix escaping keeps the joined derivation unambiguous.
_IDENTIFIERS = st.text(min_size=1, max_size=20)
#: Semantic keys: arbitrary unicode, excluding strings that already parse as a
#: UUID (those intentionally bypass derivation and round-trip unchanged).
_KEYS = st.text(min_size=1, max_size=60).filter(lambda s: not _parses_as_uuid(s))


@_SETTINGS
@given(value=_KEYS, namespace=_IDENTIFIERS)
def test_stable_id_is_deterministic_and_canonical(value: str, namespace: str) -> None:
    """Same (namespace, value) always derives the same canonical UUID5."""
    derived = stable_id_for(value, namespace=namespace)
    assert derived == stable_id_for(value, namespace=namespace)
    parsed = uuid.UUID(derived)
    assert str(parsed) == derived
    assert parsed.version == 5


@_SETTINGS
@given(external=st.uuids(), namespace=_IDENTIFIERS)
def test_external_uuid_round_trips_unchanged(
    external: uuid.UUID, namespace: str
) -> None:
    """A value that already parses as a UUID is returned verbatim."""
    assert stable_id_for(str(external), namespace=namespace) == str(external)


@_SETTINGS
@given(a=st.tuples(_IDENTIFIERS, _KEYS), b=st.tuples(_IDENTIFIERS, _KEYS))
def test_distinct_inputs_derive_distinct_stable_ids(
    a: tuple[str, str], b: tuple[str, str]
) -> None:
    """Different (namespace, value) pairs never collide."""
    assume(a != b)
    assert stable_id_for(a[1], namespace=a[0]) != stable_id_for(b[1], namespace=b[0])


@_SETTINGS
@given(namespace=_IDENTIFIERS, fallback=_KEYS)
def test_empty_value_is_robust_and_uses_fallback_key(
    namespace: str, fallback: str
) -> None:
    """A falsy value falls back to ``fallback_key`` (or derives from '')."""
    assert stable_id_for("", namespace=namespace, fallback_key=fallback) == (
        stable_id_for(fallback, namespace=namespace)
    )
    bare = stable_id_for("", namespace=namespace)
    assert bare == stable_id_for("", namespace=namespace)
    assert uuid.UUID(bare).version == 5


@_SETTINGS
@given(
    a=st.tuples(_IDENTIFIERS, st.text(min_size=1, max_size=60), st.integers(1, 10**6)),
    b=st.tuples(_IDENTIFIERS, st.text(min_size=1, max_size=60), st.integers(1, 10**6)),
)
def test_record_id_is_deterministic_and_collision_free(
    a: tuple[str, str, int], b: tuple[str, str, int]
) -> None:
    """Same (kind, stable_id, version) → same id; different triples → different."""
    kind_a, stable_a, version_a = a
    kind_b, stable_b, version_b = b
    id_a = record_id_for(stable_id=stable_a, version=version_a, kind=kind_a)
    assert id_a == record_id_for(stable_id=stable_a, version=version_a, kind=kind_a)
    assert uuid.UUID(id_a).version == 5
    assume(a != b)
    assert id_a != record_id_for(stable_id=stable_b, version=version_b, kind=kind_b)


def test_crafted_colon_inputs_no_longer_collide() -> None:
    """Regression: the exact boundary-ambiguity collisions found by the audit.

    Pre-escape, both pairs below joined to the same derivation string
    (``x:a:b`` and ``k:s:7:1``) and silently derived the SAME id.
    """
    assert stable_id_for("b", namespace="x:a") != stable_id_for("a:b", namespace="x")
    assert record_id_for(kind="k", stable_id="s:7", version=1) != record_id_for(
        kind="k:s", stable_id="7", version=1
    )


@_SETTINGS
@given(value=_KEYS, namespace=st.text(min_size=1, max_size=20))
def test_clean_prefixes_pin_pre_escape_derivation(value: str, namespace: str) -> None:
    """Backward compat: colon/backslash-free prefixes derive the HISTORICAL id.

    Every id ever derived before prefix escaping existed used the raw
    ``f"{namespace}:{key}"`` formula; this pins that ids for all real-world
    (clean-prefix) inputs are unchanged, so stored registries stay consistent.
    """
    assume(":" not in namespace and "\\" not in namespace)
    historical = str(
        uuid.uuid5(
            uuid.UUID("6f1d3a2e-7c4b-5a9e-8d12-0a1b2c3d4e5f"), f"{namespace}:{value}"
        )
    )
    assert stable_id_for(value, namespace=namespace) == historical


@_SETTINGS
@given(
    kind=_IDENTIFIERS,
    stable_id=st.text(min_size=1, max_size=60),
    version=st.integers(min_value=1, max_value=1000),
)
def test_entity_record_id_is_derived_consistently(
    kind: str, stable_id: str, version: int
) -> None:
    """``create`` and implicit post-init both fill the same deterministic id."""
    expected = record_id_for(stable_id=stable_id, version=version, kind=kind)
    created = EntityRecord.create(stable_id=stable_id, version=version, kind=kind)
    implicit = EntityRecord(stable_id=stable_id, version=version, kind=kind)
    assert created.record_id == expected
    assert implicit.record_id == expected
