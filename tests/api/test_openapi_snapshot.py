"""Deterministic ``/v1`` OpenAPI snapshot — the served API stability contract.

The public, versioned contract is the ``/v1`` REST surface (see ``docs/STABILITY.md``).
This test builds the app's OpenAPI document, FILTERS it to ``/v1*`` paths only (the
``/api/studio`` routers are Studio's internal BFF and are explicitly NOT part of the
stable contract — including them would red this test on every Studio change), strips
the volatile ``info.version`` (single-sourced from ``himmy.__version__`` and bumped
independently of the schema), normalizes ordering, and diffs against a committed
snapshot.

A breaking ``/v1`` change (a removed path/operation, a renamed field, a tightened
schema) reds this test. To accept an intentional ``/v1`` change, regenerate the
snapshot::

    UPDATE_OPENAPI_SNAPSHOT=1 .venv/bin/python -m pytest \
        tests/api/test_openapi_snapshot.py -q

Determinism: the snapshot is captured on the NO-AUTH baseline (no ``HIMMY_AUTH_MODE``
/ ``HIMMY_INTERNAL_API_KEY`` / ``HIMMY_API_KEYS_FILE`` in the environment). That is the
offline-first invariant path: with no authenticator configured, ``create_app`` does
NOT install ``_custom_openapi``, so no ``securitySchemes`` are injected and the
document is identical on every machine regardless of the operator's auth env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

SNAPSHOT_PATH = Path(__file__).with_name("openapi_v1_snapshot.json")

# Auth-mode env vars that would otherwise inject provider-specific securitySchemes
# (and so make the snapshot machine/operator dependent). Cleared for the baseline.
_AUTH_ENV = (
    "HIMMY_AUTH_MODE",
    "HIMMY_INTERNAL_API_KEY",
    "HIMMY_API_KEYS_FILE",
    "HIMMY_OIDC_ISSUER",
    "HIMMY_OIDC_AUDIENCE",
    "HIMMY_OIDC_JWKS_URL",
)


def _iter_schema_refs(node: Any) -> list[str]:
    """Collect every ``#/components/schemas/<name>`` ref name under ``node``."""
    found: list[str] = []
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.append(ref.rsplit("/", 1)[-1])
        for value in node.values():
            found.extend(_iter_schema_refs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_schema_refs(item))
    return found


def _reachable_schemas(
    paths: dict[str, Any], schemas: dict[str, Any]
) -> dict[str, Any]:
    """Transitively prune ``components.schemas`` to only those the /v1 paths use.

    FastAPI emits ONE shared component pool keyed by model name, so a Studio-only
    model otherwise leaks into the snapshot and reds this test on a pure-Studio
    change. We walk the /v1 paths, follow ``$ref`` edges, and keep only the
    closure — so the contract is exactly the /v1 surface + its referenced schemas.
    """
    seen: set[str] = set()
    queue = _iter_schema_refs(paths)
    while queue:
        name = queue.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        queue.extend(_iter_schema_refs(schemas[name]))
    return {name: schemas[name] for name in sorted(seen)}


def _build_v1_openapi() -> dict[str, Any]:
    """Build the app's OpenAPI doc, scoped to ``/v1`` and made deterministic.

    Imported lazily so the auth env is neutralized BEFORE ``create_app`` runs.
    """
    from himmy.api.app import create_app

    app = create_app()
    spec = app.openapi()

    paths = {p: spec["paths"][p] for p in spec.get("paths", {}) if p.startswith("/v1")}
    all_schemas = spec.get("components", {}).get("schemas", {})
    schemas = _reachable_schemas(paths, all_schemas)
    contract = {
        "openapi": spec.get("openapi"),
        "paths": paths,
        # Only the schemas the /v1 surface references — a Studio-only model change
        # cannot red this snapshot; info.version (volatile) is excluded entirely.
        "components": {"schemas": schemas},
    }
    # json round-trip with sorted keys = a canonical, machine-independent form.
    return json.loads(json.dumps(contract, sort_keys=True))


@pytest.fixture(autouse=True)
def _neutralize_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUTH_ENV:
        monkeypatch.delenv(name, raising=False)


def test_v1_openapi_matches_snapshot() -> None:
    current = _build_v1_openapi()

    if os.environ.get("UPDATE_OPENAPI_SNAPSHOT") == "1":
        SNAPSHOT_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip("regenerated tests/api/openapi_v1_snapshot.json")

    assert SNAPSHOT_PATH.exists(), (
        "missing tests/api/openapi_v1_snapshot.json — regenerate with "
        "UPDATE_OPENAPI_SNAPSHOT=1"
    )
    committed = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    if current != committed:
        cur_paths = set(current["paths"])
        comm_paths = set(committed["paths"])
        added = sorted(cur_paths - comm_paths)
        removed = sorted(comm_paths - cur_paths)
        changed = sorted(
            p
            for p in cur_paths & comm_paths
            if current["paths"][p] != committed["paths"][p]
        )
        raise AssertionError(
            "The /v1 OpenAPI contract changed.\n"
            f"  added paths:   {added}\n"
            f"  removed paths: {removed}\n"
            f"  changed paths: {changed}\n"
            "If this change is intentional, regenerate the snapshot with "
            "UPDATE_OPENAPI_SNAPSHOT=1 and review the diff in the PR."
        )


def test_snapshot_is_v1_only_and_versionless() -> None:
    """Guard the snapshot itself: only /v1 paths, no volatile version leaked in."""
    assert SNAPSHOT_PATH.exists()
    committed = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert committed["paths"], "snapshot has no /v1 paths"
    assert all(p.startswith("/v1") for p in committed["paths"]), (
        "snapshot leaked a non-/v1 path"
    )
    # info.version (and the whole info block) must not be part of the contract.
    assert "info" not in committed
