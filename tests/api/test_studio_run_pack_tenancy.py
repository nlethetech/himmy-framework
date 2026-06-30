"""rbac-harden(tool-store-tenancy): the Studio/Missions run surface tenant-scopes its
memory + knowledge tool packs.

The ``/v1`` run service was already scoped (``services.py: subject=workspace_id``, covered by
``tests/runtime/test_from_spec_pack_tenancy.py``), but the Studio/Missions run path
(``studio_service.stream_agent_run``) is a SECOND verified multi-tenant surface that built
its memory/knowledge packs WITHOUT threading the launching tenant — so on a shared durable
``.himmy/memory.db`` tenant t1's ``remember`` was recallable by tenant t2's ``recall``.

Two assertions:

1. :func:`test_stream_agent_run_threads_owner_into_pack_scope` — the wiring proof: the
   verified ``owner_workspace_id`` reaches ``build_runtime_for_spec(subject=...)`` (it is the
   single load-bearing argument that scopes the packs). Fails before the fix.

2. :func:`test_studio_path_scope_isolates_memory` — the store proof, at the exact seam the
   Studio path now uses (``build_runtime_for_spec(subject=owner_workspace_id)``): t1's
   ``remember`` on a shared db is NOT visible to a t2-scoped run, and ``subject=None``
   (offline / single-box / all_tenants) keeps the historical shared store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec
from tests.conftest import run_async


class _Stop(Exception):
    """Short-circuit the run once we have captured the build call."""


def test_stream_agent_run_threads_owner_into_pack_scope(monkeypatch: Any) -> None:
    """The verified launching tenant reaches ``build_runtime_for_spec(subject=...)``.

    ``owner_workspace_id`` is resolved from the authenticated principal by every
    multi-tenant caller (Missions stamps ``mission.workspace_id``; Studio routers resolve
    ``_run_owner(request)``). Before this fix the Studio run path omitted ``subject=``
    entirely, so every Mission/Studio run keyed its memory/knowledge packs off the static
    shared scope. This asserts the owner is now the ``subject``.
    """
    import himmy.runtime.from_spec as from_spec_mod

    captured: dict[str, Any] = {}

    def _capture(spec: Any, **kwargs: Any) -> Any:
        captured["subject"] = kwargs.get("subject")
        raise _Stop

    monkeypatch.setattr(from_spec_mod, "build_runtime_for_spec", _capture)

    from himmy.api import studio_service

    spec = AgentSpec(name="m", provider="stub", memory=True, tool_packs=["memory"])

    async def _drive() -> None:
        agen = studio_service.stream_agent_run(
            spec, "hello", owner_workspace_id="tenant-1"
        )
        async for _ in agen:  # pragma: no cover - we expect _Stop before any frame
            pass

    with pytest.raises(_Stop):
        run_async(_drive())

    assert captured["subject"] == "tenant-1", (
        "Studio/Missions run path did not thread the verified owner workspace into the "
        "memory/knowledge pack scope"
    )


def _memory_registry(subject: str | None, db: str) -> Any:
    """Build the run's tool registry exactly as the Studio path does (``subject=owner``)."""
    import os

    spec = AgentSpec(name="m", provider="stub", memory=True, tool_packs=["memory"])
    os.environ["HIMMY_MEMORY_PATH"] = db
    try:
        _runtime, registry = build_runtime_for_spec(spec, subject=subject)
    finally:
        os.environ.pop("HIMMY_MEMORY_PATH", None)
    return registry


def test_studio_path_scope_isolates_memory(tmp_path: Path) -> None:
    """A t2 Studio/Mission run cannot ``recall`` a t1 run's ``remember`` (shared db)."""
    db = str(tmp_path / "shared_studio_mem.db")
    t1 = _memory_registry("tenant-1", db)
    t1.handler_for("remember")({"text": "mission t1 secret: launch code is bluejay-7"})

    t2 = _memory_registry("tenant-2", db)
    found = run_async(
        t2.handler_for("recall")({"query": "launch code bluejay-7", "top_k": 5})
    )
    assert not any("bluejay-7" in r["text"] for r in found["results"]), (
        "tenant-2's Studio/Mission run recalled tenant-1's memory through the tool path"
    )

    own = run_async(
        t1.handler_for("recall")({"query": "launch code bluejay-7", "top_k": 5})
    )
    assert any("bluejay-7" in r["text"] for r in own["results"])


def test_studio_path_offline_subject_none_shares_store(tmp_path: Path) -> None:
    """Offline invariant: ``subject=None`` (single-box / all_tenants) shares the store."""
    db = str(tmp_path / "shared_studio_mem.db")
    a = _memory_registry(None, db)
    a.handler_for("remember")({"text": "a plain single-box note about the harvest"})
    b = _memory_registry(None, db)
    found = run_async(b.handler_for("recall")({"query": "harvest note", "top_k": 3}))
    assert any("harvest" in r["text"] for r in found["results"])
