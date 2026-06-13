"""T2.1 — the durable shared spine via :class:`SpineFactory`.

These cover the plan's acceptance for the SpineFactory unit and every reviewer must_fix:

* ``deps.build_default`` constructs NO bare in-memory ``EntityRegistry()`` — its spine is
  the durable :class:`SqliteEntityRegistry` at the canonical ``.himmy/spine.db``.
* a run's lineage survives a server restart with ZERO DSN configured (the build-order
  trap: ``build_default`` runs before the lifespan sets server context, so the lifespan
  ALWAYS rebinds the spine).
* the CLI ``seclog`` reads the SAME ``.himmy/spine.db`` a server writes (one spine).
* in-memory vs SQLite bundle parity: a record's ``content_hash`` is byte-identical across
  the two backends (so a tamper-evidence bundle is backend-agnostic).
* the Pydantic NOMINAL-type wall: a durable ``SqliteEntityRegistry`` is now accepted where
  a registry is required (``RecordedDataContext``) — the prior ``isinstance`` rejection
  against the concrete ``EntityRegistry`` is gone.
* the deferred-commit buffered flush (the hot-path perf escape hatch) is correct and a real
  speed-up.
* ``SpineFactory`` resolution: for_cli/for_server/for_context, ``HIMMY_SPINE_PATH``
  override, project-root resolution, and the opt-in Postgres-spine policy.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from himmy.config.project import find_project_root, spine_db_path
from himmy.entities.protocol import EntityRegistryProtocol
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.entities.spine_factory import SpineFactory
from himmy.entities.sqlite_registry import SqliteEntityRegistry


@pytest.fixture
def spine_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin the canonical spine to an isolated file via ``HIMMY_SPINE_PATH``."""
    target = tmp_path / "spine.db"
    monkeypatch.setenv("HIMMY_SPINE_PATH", str(target))
    # Never let a stray ``HIMMY_SPINE_DATABASE_URL`` flip us onto the Postgres branch.
    monkeypatch.delenv("HIMMY_SPINE_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_SPINE_BUFFER", raising=False)
    return target


def _run_event(i: int, *, run: str = "run-1") -> EntityRecord:
    """A representative ``run_event`` record."""
    return EntityRecord.create(
        stable_id=f"{run}-evt-{i}",
        version=1,
        kind="run_event",
        payload={"seq": i, "type": "tool_call", "tool": "echo"},
        metadata={"run_id": run},
    )


# --------------------------------------------------------------------------- path resolution
def test_spine_db_path_honours_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HIMMY_SPINE_PATH`` is honoured verbatim and its parent dir is created."""
    target = tmp_path / "nested" / "custom-spine.db"
    monkeypatch.setenv("HIMMY_SPINE_PATH", str(target))
    assert spine_db_path() == str(target)
    assert target.parent.is_dir()


def test_project_root_walks_up_to_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project root is the nearest ancestor carrying a ``himmy.toml`` / ``.himmy/``.

    A server launched from a SUBDIR of the project must resolve the SAME ``.himmy/``
    directory the CLI at the project root uses — that is what shares one spine.
    """
    monkeypatch.delenv("HIMMY_SPINE_PATH", raising=False)
    (tmp_path / "himmy.toml").write_text("[defaults]\n")
    subdir = tmp_path / "a" / "b"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)
    assert find_project_root() == tmp_path.resolve()
    assert spine_db_path() == str(tmp_path.resolve() / ".himmy" / "spine.db")


def test_project_root_defaults_to_cwd_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no marker anywhere up-tree, the root is the CWD (predictable, no surprises)."""
    monkeypatch.delenv("HIMMY_SPINE_PATH", raising=False)
    leaf = tmp_path / "fresh-checkout"
    leaf.mkdir()
    monkeypatch.chdir(leaf)
    assert find_project_root() == leaf.resolve()


# --------------------------------------------------------------------------- factory shape
def test_for_cli_is_durable_sqlite_at_canonical_path(spine_path: Path) -> None:
    """``for_cli`` returns the DURABLE SQLite spine — unlike the in-memory run store."""
    spine = SpineFactory.for_cli()
    try:
        assert isinstance(spine, SqliteEntityRegistry)
        rec = spine.register(_run_event(0))
        spine.flush()
        assert spine_path.exists(), "the CLI spine must persist to the canonical file"
        assert spine.get(rec.record_id) is not None
    finally:
        spine.close()


def test_for_context_cli_and_server_share_one_sqlite_file(spine_path: Path) -> None:
    """for_context(server=False/True) both resolve the SAME on-disk spine by default."""
    cli_spine = SpineFactory.for_context(server=False)
    rec = cli_spine.register(_run_event(1))
    cli_spine.flush()
    cli_spine.close()

    server_spine = SpineFactory.for_context(server=True)
    try:
        assert isinstance(server_spine, SqliteEntityRegistry)
        # The server reads the record the CLI wrote — one shared spine.
        assert server_spine.get(rec.record_id) is not None
    finally:
        server_spine.close()


def test_for_server_keeps_sqlite_spine_with_postgres_storage_dsn(
    spine_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Postgres STORAGE dsn does NOT flip the spine to Postgres (opt-in is separate)."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    monkeypatch.delenv("HIMMY_SPINE_DATABASE_URL", raising=False)
    spine = SpineFactory.for_server()
    try:
        assert isinstance(spine, SqliteEntityRegistry)
    finally:
        spine.close()


# --------------------------------------------------------------------------- deps.py wiring
def test_build_default_constructs_no_bare_entity_registry(spine_path: Path) -> None:
    """ACCEPTANCE: ``deps.build_default`` no longer builds a bare in-memory registry."""
    from himmy.api.deps import ApiContainer

    container = ApiContainer.build_default()
    try:
        assert isinstance(container.entity_registry, SqliteEntityRegistry)
        assert not isinstance(container.entity_registry, EntityRegistry)
    finally:
        import asyncio

        asyncio.run(container.aclose())


def test_deps_module_has_no_bare_entity_registry_construction() -> None:
    """Static (AST) guard: ``deps.py`` never *constructs* a bare ``EntityRegistry``.

    Uses the AST (not a substring) so a comment/docstring naming the old construct cannot
    fool it: it scans for an actual ``EntityRegistry(...)`` call node.
    """
    import ast

    import himmy.api.deps as deps_mod

    tree = ast.parse(Path(deps_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            assert name != "EntityRegistry", (
                "deps.py must resolve its spine through SpineFactory, never construct a "
                "bare EntityRegistry() (T2.1)"
            )


# --------------------------------------------------------------------------- restart survival
def test_run_lineage_survives_server_restart_zero_dsn(
    spine_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's lineage survives a server restart with NO DSN configured.

    Reproduces the build-order trap: ``create_app`` builds the default container BEFORE
    the lifespan sets server context. The lifespan ALWAYS rebinds the spine, so the spine
    every request-time reader uses is the durable canonical ``.himmy/spine.db``. A second
    app process (new container) over the same file sees the first run's lineage.
    """
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)

    from fastapi.testclient import TestClient

    from himmy.api.app import create_app

    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "scribe", "description": "d"},
        "task": {"title": "brief", "prompt": "say hi"},
    }

    # --- first "server process": run something through the live app ---
    app1 = create_app()
    constructed = app1.state.container
    with TestClient(app1) as client:
        # The lifespan ALWAYS rebinds the spine under server context (the build-order
        # fix): the container request-time readers see is rebuilt, with a durable spine.
        assert app1.state.container is not constructed
        assert isinstance(app1.state.container.entity_registry, SqliteEntityRegistry)
        resp = client.post("/v1/runs", json=body)
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        # Let the background run finish so its events land on the spine.
        for _ in range(100):
            got = client.get(f"/v1/runs/{run_id}")
            if got.json().get("status") in {"SUCCEEDED", "FAILED"}:
                break
            time.sleep(0.05)

    # The spine file persists to disk after the first process shuts down.
    assert spine_path.exists()
    after_first = SqliteEntityRegistry(spine_path.as_posix())
    try:
        events_first = after_first.list_by_kind("run_event")
        assert events_first, "the first run must have projected run_event lineage"
        n_first = len(events_first)
    finally:
        after_first.close()

    # --- restart: a brand-new app/container over the SAME canonical spine ---
    app2 = create_app()
    with TestClient(app2):
        spine2 = app2.state.container.entity_registry
        assert isinstance(spine2, SqliteEntityRegistry)
        # The restarted server's spine still holds the prior run's lineage.
        assert len(spine2.list_by_kind("run_event")) >= n_first


# --------------------------------------------------------------------------- CLI == server
def test_cli_seclog_reads_same_spine_as_server(
    spine_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACCEPTANCE: CLI ``seclog`` rows == the server's spine rows (one ``.himmy/spine.db``).

    The server records a security event into its container spine; the CLI ``seclog`` adapter
    (a wholly separate object, resolved through ``SpineFactory.for_cli``) reads the same
    on-disk database and sees that exact event.
    """
    from himmy.api.deps import ApiContainer
    from himmy.cli.security_audit_cmd import cli_security_log
    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.audit.models import SecurityEvent

    container = ApiContainer.build_default()
    try:
        server_audit = SecurityAuditLog(container.entity_registry)
        server_audit.record(
            SecurityEvent(
                event_type="authz_denied",
                outcome="deny",
                actor={"subject": "tenant-x"},
                resource="POST /v1/runs",
                action="run.create",
            )
        )
        container.entity_registry.flush()  # type: ignore[attr-defined]
    finally:
        import asyncio

        asyncio.run(container.aclose())

    # A fresh CLI adapter reads the SAME canonical spine.db and sees the server's event.
    cli_log = cli_security_log()
    events = cli_log.recent(limit=10, event_type="authz_denied")
    assert any(e.resource == "POST /v1/runs" for e in events)


# --------------------------------------------------------------------------- bundle parity
def test_in_memory_vs_sqlite_content_hash_parity(spine_path: Path) -> None:
    """A record's ``content_hash`` is byte-identical across in-memory + SQLite backends.

    So a tamper-evidence bundle exported from either spine is the same — the durable swap
    cannot change the integrity surface.
    """
    from himmy.entities.integrity import content_hash

    mem = EntityRegistry()
    sql = SpineFactory.for_cli()
    try:
        records = [_run_event(i) for i in range(5)]
        for rec in records:
            mem.register(rec)
            sql.register(rec)
        sql.flush()
        for rec in records:
            mem_rec = mem.get(rec.record_id)
            sql_rec = sql.get(rec.record_id)
            assert mem_rec is not None and sql_rec is not None
            assert content_hash(mem_rec) == content_hash(sql_rec)
            # The stored bytes round-trip identically too.
            assert mem_rec.payload == sql_rec.payload
            assert mem_rec.metadata == sql_rec.metadata
    finally:
        sql.close()


# --------------------------------------------------------------------------- Pydantic wall
def test_sqlite_registry_satisfies_protocol_isinstance() -> None:
    """The durable spine structurally satisfies the runtime-checkable Protocol."""
    spine = SqliteEntityRegistry(":memory:")
    try:
        assert isinstance(spine, EntityRegistryProtocol)
    finally:
        spine.close()


def test_recorded_data_context_accepts_durable_sqlite_registry() -> None:
    """The Pydantic NOMINAL-type wall is gone: a SqliteEntityRegistry validates now.

    Before the Protocol retype, ``RecordedDataContext.entity_registry`` was annotated with
    the concrete ``EntityRegistry``; ``arbitrary_types_allowed`` made Pydantic validate via
    ``isinstance(value, EntityRegistry)`` and REJECT a durable ``SqliteEntityRegistry``.
    We first reproduce that exact failure, then prove the real model now accepts it.
    """
    from pydantic import BaseModel, ConfigDict, ValidationError

    spine = SqliteEntityRegistry(":memory:")
    try:
        # 1) Reproduce the prior breakage: validating against the CONCRETE class rejects it.
        class _OldNominalContext(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            entity_registry: EntityRegistry | None = None

        with pytest.raises(ValidationError):
            _OldNominalContext(entity_registry=spine)  # type: ignore[arg-type]

        # 2) The real model (now typed against the Protocol) accepts the durable spine.
        from himmy.services.evaluation.privacy.models import RecordedDataContext

        ctx = RecordedDataContext(entity_registry=spine)
        assert ctx.entity_registry is spine
    finally:
        spine.close()


# --------------------------------------------------------------------------- buffered flush
def test_buffered_register_defers_then_flushes(tmp_path: Path) -> None:
    """A buffered spine still reads back every record (same-connection visibility) and
    only persists across connections after a flush."""
    path = tmp_path / "buffered.db"
    spine = SqliteEntityRegistry(path.as_posix(), register_buffer=50)
    try:
        recs = [_run_event(i) for i in range(10)]
        for r in recs:
            spine.register(r)
        # Same-connection reads see the uncommitted batch.
        assert all(spine.get(r.record_id) is not None for r in recs)
        assert len(spine.list_by_kind("run_event")) == 10

        # A SECOND connection does not see the uncommitted tail until flush.
        other = SqliteEntityRegistry(path.as_posix())
        try:
            assert other.list_by_kind("run_event") == []
            spine.flush()
            assert len(other.list_by_kind("run_event")) == 10
        finally:
            other.close()
    finally:
        spine.close()


def test_buffered_new_version_flushes_pending(tmp_path: Path) -> None:
    """``new_version`` flushes the deferred register batch before its IMMEDIATE txn."""
    path = tmp_path / "buffered2.db"
    spine = SqliteEntityRegistry(path.as_posix(), register_buffer=100)
    try:
        for i in range(5):
            spine.register(_run_event(i))
        # new_version opens BEGIN IMMEDIATE; it must not error on the open implicit txn.
        v = spine.new_version(stable_id="artifact", kind="agent", payload={"v": 1})
        assert v.version == 1
        # The pending register batch was committed by the flush new_version forced.
        other = SqliteEntityRegistry(path.as_posix())
        try:
            assert len(other.list_by_kind("run_event")) == 5
        finally:
            other.close()
    finally:
        spine.close()


def test_buffered_close_persists_tail(tmp_path: Path) -> None:
    """``close`` flushes the not-yet-committed tail so nothing is silently lost."""
    path = tmp_path / "buffered3.db"
    spine = SqliteEntityRegistry(path.as_posix(), register_buffer=1000)
    for i in range(7):
        spine.register(_run_event(i))
    spine.close()  # must flush on the way out
    reopened = SqliteEntityRegistry(path.as_posix())
    try:
        assert len(reopened.list_by_kind("run_event")) == 7
    finally:
        reopened.close()


def test_buffered_register_is_faster(tmp_path: Path) -> None:
    """The buffered hot path is a measurable speed-up over per-event commits.

    Guards the perf must_fix: if a future change makes the deferred-commit path no faster,
    the buffer has stopped doing its job. A generous bound keeps it non-flaky on slow CI.
    """
    n = 1500

    def _elapsed(buffer: int) -> float:
        db = tmp_path / f"perf-{buffer}.db"
        reg = SqliteEntityRegistry(db.as_posix(), register_buffer=buffer)
        recs = [_run_event(i) for i in range(n)]
        start = time.perf_counter()
        for r in recs:
            reg.register(r)
        reg.flush()
        out = time.perf_counter() - start
        reg.close()
        return out

    per_event = _elapsed(1)
    batched = _elapsed(n)
    assert batched < per_event, (per_event, batched)


def test_spine_buffer_env_drives_factory(
    spine_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HIMMY_SPINE_BUFFER`` selects the deferred-commit batch size on factory spines."""
    monkeypatch.setenv("HIMMY_SPINE_BUFFER", "64")
    spine = SpineFactory.for_cli()
    try:
        assert isinstance(spine, SqliteEntityRegistry)
        assert spine._register_buffer == 64  # noqa: SLF001 - asserting the wired knob
    finally:
        spine.close()


def test_sync_facade_bridges_async_registry_without_db() -> None:
    """The sync Postgres facade drives an async registry on a background loop (no DB).

    Proves the bridge (and so the opt-in Postgres spine path) works without standing up
    Postgres: a stub async registry is wrapped and called through the synchronous Protocol
    surface, and the blocking calls return the awaited results in order.
    """
    from himmy.entities.sync_facade import SyncPostgresEntityRegistry, _LoopThread

    class _StubAsync:
        def __init__(self) -> None:
            self.store: dict[str, EntityRecord] = {}
            self.closed = False

        async def register(self, record: EntityRecord) -> EntityRecord:
            self.store[record.record_id] = record
            return record

        async def get(self, record_id: str) -> EntityRecord | None:
            return self.store.get(record_id)

        async def list_by_kind(self, kind: str) -> list[EntityRecord]:
            return [r for r in self.store.values() if r.kind == kind]

        async def close(self) -> None:
            self.closed = True

    stub = _StubAsync()
    facade = SyncPostgresEntityRegistry(stub, _LoopThread())  # type: ignore[arg-type]
    try:
        assert isinstance(facade, EntityRegistryProtocol)
        rec = facade.register(_run_event(0))
        assert facade.get(rec.record_id) is rec
        assert facade.list_by_kind("run_event") == [rec]
    finally:
        facade.close()
    assert stub.closed is True


def test_from_spec_runtime_spine_is_durable_on_cli(
    spine_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI-built per-spec runtime's spine is the durable canonical spine (not in-RAM)."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.runtime import from_spec

    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    spec = AgentSpec(name="t", provider="stub")
    runtime, _registry = from_spec.build_runtime_for_spec(spec, durable_defaults=False)
    spine = runtime.entity_registry
    assert isinstance(spine, SqliteEntityRegistry)
    # It is the canonical project spine — a record written here is visible to the CLI log.
    rec = spine.register(_run_event(99, run="cli-run"))
    spine.flush()
    cli_spine = SpineFactory.for_cli()
    try:
        assert cli_spine.get(rec.record_id) is not None
    finally:
        cli_spine.close()
