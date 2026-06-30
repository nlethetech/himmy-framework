"""Round-8: ``build_runtime_for_spec(subject=...)`` tenant-scopes the memory/knowledge packs.

The HTTP write surfaces were hardened earlier, but the TOOL-execution path was a separate
cross-tenant confused deputy: the memory pack keyed every run off the static
``memory_subject`` (``"default"``) and the knowledge pack pinned the durable KB to a fixed
``(local, local, default)`` scope. On a shared server process — every tenant's runs landing
in ONE ``.himmy/memory.db`` / one durable KB — tenant t1's ``remember``/``kb_ingest`` was
recallable by t2's ``recall``/``kb_search``.

These assert the run's ``subject`` (its owning ``workspace_id``) is threaded into the pack
scope (``ToolkitConfig.tenant_scope``), so t2 cannot read t1's rows; and that the offline
``subject=None`` path is byte-for-byte the historical shared store.
"""

from __future__ import annotations

from pathlib import Path

from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec
from tests.conftest import run_async


def _memory_registry(subject: str | None, db: str):
    spec = AgentSpec(name="m", provider="stub", memory=True, tool_packs=["memory"])
    import os

    os.environ["HIMMY_MEMORY_PATH"] = db
    try:
        _runtime, registry = build_runtime_for_spec(spec, subject=subject)
    finally:
        os.environ.pop("HIMMY_MEMORY_PATH", None)
    return registry


def test_memory_pack_tenant_scoped_via_subject(tmp_path: Path) -> None:
    """t2's ``recall`` does NOT surface t1's ``remember`` on the shared durable store."""
    db = str(tmp_path / "shared_mem.db")
    t1 = _memory_registry("tenant-1", db)
    t1.handler_for("remember")({"text": "the wire transfer pin is tango-9 for t1"})

    t2 = _memory_registry("tenant-2", db)
    found = run_async(
        t2.handler_for("recall")({"query": "wire transfer pin tango-9", "top_k": 5})
    )
    assert not any("tango-9" in r["text"] for r in found["results"]), (
        "tenant-2 recalled tenant-1's memory through the tool path"
    )

    own = run_async(
        t1.handler_for("recall")({"query": "wire transfer pin tango-9", "top_k": 5})
    )
    assert any("tango-9" in r["text"] for r in own["results"])


def test_memory_pack_offline_subject_none_shares_store(tmp_path: Path) -> None:
    """Offline invariant: ``subject=None`` (CLI / no authenticator) shares the store as before."""
    db = str(tmp_path / "shared_mem.db")
    a = _memory_registry(None, db)
    a.handler_for("remember")({"text": "an ordinary shared note about the orchard"})
    b = _memory_registry(None, db)
    found = run_async(b.handler_for("recall")({"query": "orchard note", "top_k": 3}))
    assert any("orchard" in r["text"] for r in found["results"])
