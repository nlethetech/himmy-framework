"""``_authorize_run`` is the BOLA gate for the Studio by-id run readers.

Red-team r5: it previously failed OPEN for a cache-only run — a run whose canonical
``RunRecord`` is absent (erased/crypto-shredded or aged out) while the process-wide
studio.db presentation cache row (feedback/lineage) lingers. The presentation cache
carries no ``workspace_id``, so a tenant-bound reader served such a row would read
another tenant's feedback/lineage by id (a cross-tenant IDOR). The gate must now mirror
``get_studio_run_unified`` and fold a cache-only row to not-found (404) for a
tenant-bound reader, while the offline / ``all_tenants`` path stays a byte-unchanged
no-op (allow).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.routers.studio import _authorize_run


class _Storage:
    """A canonical storage stub whose ``get_run`` returns a preset record (or None)."""

    def __init__(self, rec: Any) -> None:
        self._rec = rec

    async def get_run(self, run_id: str) -> Any:  # noqa: ARG002
        return self._rec


def _request(principal: Principal, *, storage: Any) -> Any:
    """A minimal request whose app.state exposes a container with ``storage``."""
    container = SimpleNamespace(storage=storage)
    app = SimpleNamespace(state=SimpleNamespace(container=container))
    return SimpleNamespace(app=app, state=SimpleNamespace(principal=principal))


@pytest.mark.asyncio
async def test_cache_only_run_denied_to_tenant_bound_reader() -> None:
    """Canonical record absent + tenant-bound principal → DENY (folds to 404)."""
    principal = Principal(subject="acme-user", tenant_ids=frozenset({"acme"}))
    request = _request(principal, storage=_Storage(rec=None))
    assert await _authorize_run(request, "cache-only-1") is False


@pytest.mark.asyncio
async def test_all_tenants_principal_still_allowed_for_cache_only() -> None:
    """The offline / all_tenants path is a NO-OP (allow), byte-unchanged."""
    request = _request(ANONYMOUS, storage=_Storage(rec=None))
    assert await _authorize_run(request, "cache-only-1") is True


@pytest.mark.asyncio
async def test_no_canonical_storage_is_allowed() -> None:
    """The bare/single-box app (no canonical storage) stays a no-op (allow)."""
    principal = Principal(subject="acme-user", tenant_ids=frozenset({"acme"}))
    container = SimpleNamespace(storage=None)
    app = SimpleNamespace(state=SimpleNamespace(container=container))
    request = SimpleNamespace(
        app=app, state=SimpleNamespace(principal=principal)
    )
    assert await _authorize_run(request, "any") is True


@pytest.mark.asyncio
async def test_own_tenant_run_allowed() -> None:
    """A canonically-known run in the caller's own workspace is allowed."""
    principal = Principal(subject="acme-user", tenant_ids=frozenset({"acme"}))
    rec = SimpleNamespace(workspace_id="acme")
    request = _request(principal, storage=_Storage(rec=rec))
    assert await _authorize_run(request, "r1") is True


@pytest.mark.asyncio
async def test_other_tenant_run_denied() -> None:
    """A canonically-known run in ANOTHER workspace is denied."""
    principal = Principal(subject="acme-user", tenant_ids=frozenset({"acme"}))
    rec = SimpleNamespace(workspace_id="evilcorp")
    request = _request(principal, storage=_Storage(rec=rec))
    assert await _authorize_run(request, "r1") is False


# ----------------------------------------------- subject axis (BOLA, scope-r1#3)
@pytest.mark.asyncio
async def test_other_subject_run_denied_to_subject_scoped_reader() -> None:
    """A subject_scoped reader is denied another subject's run even in its OWN tenant.

    The Studio by-id run gate now ANDs the subject/BOLA axis (``authorize_object`` on
    ``rec.subject_id``) onto the tenant axis, mirroring ``/v1/runs`` and missions — so a
    subject_scoped principal cannot read/poison another subject's run feedback via Studio.
    """
    principal = Principal(
        subject="alice", tenant_ids=frozenset({"t"}), subject_scoped=True
    )
    rec = SimpleNamespace(workspace_id="t", subject_id="bob")
    request = _request(principal, storage=_Storage(rec=rec))
    assert await _authorize_run(request, "bob-run") is False


@pytest.mark.asyncio
async def test_own_subject_run_allowed_to_subject_scoped_reader() -> None:
    """A subject_scoped reader CAN read its OWN subject's run in its tenant."""
    principal = Principal(
        subject="alice", tenant_ids=frozenset({"t"}), subject_scoped=True
    )
    rec = SimpleNamespace(workspace_id="t", subject_id="alice")
    request = _request(principal, storage=_Storage(rec=rec))
    assert await _authorize_run(request, "alice-run") is True
