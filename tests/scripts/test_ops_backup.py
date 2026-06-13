"""Round-trip tests for ``scripts/ops_backup.py`` (real WAL SQLite dbs on tmp)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "ops_backup.py"
    spec = importlib.util.spec_from_file_location("ops_backup", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load()


def _make_db(path: Path, *, wal: bool, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        if wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO t (id) VALUES (?)", [(i,) for i in range(rows)])
        conn.commit()
    finally:
        conn.close()


def _count(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT count(*) FROM t").fetchone()[0])
    finally:
        conn.close()


def _backup_args(
    store: Path, secrets: Path, out: Path, *, include_secrets: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(
        store_path=str(store / "storage.db"),
        secrets_dir=str(secrets),
        out=str(out),
        dsn=None,
        include_secrets=include_secrets,
        func=ops.cmd_backup,
    )


def _restore_args(
    archive: Path, store: Path, secrets: Path, *, force: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(
        archive=str(archive),
        store_path=str(store / "storage.db"),
        secrets_dir=str(secrets),
        force=force,
        dsn=None,
        func=ops.cmd_restore,
    )


def test_backup_restore_round_trip(tmp_path: Path) -> None:
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=True, rows=5)
    _make_db(store / "memory.db", wal=False, rows=3)
    secrets.mkdir(parents=True)
    (secrets / "postgres_password").write_text("hunter2\n", encoding="utf-8")

    assert ops.cmd_backup(_backup_args(store, secrets, out, include_secrets=True)) == 0
    archives = list(out.glob("himmy-backup-*.tar.gz"))
    assert len(archives) == 1
    archive = archives[0]

    # manifest is present + checksums recorded for every db + secret (opted in)
    with tarfile.open(archive) as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
    names = {f["name"] for f in manifest["files"]}
    assert {"storage.db", "memory.db", "secrets/postgres_password"} <= names
    # The manifest stamps the live storage schema version (max of the forward
    # migration chain), not a frozen constant — assert it tracks ops_backup's own
    # resolver so the round-trip stays correct as new migrations are appended.
    assert manifest["schema_version"] == ops._schema_version()
    assert manifest["secrets_included"] is True

    # wipe and restore into a fresh store dir
    restore_store = tmp_path / "restored"
    restore_secrets = restore_store / "secrets"
    assert ops.cmd_restore(_restore_args(archive, restore_store, restore_secrets)) == 0
    assert _count(restore_store / "storage.db") == 5
    assert _count(restore_store / "memory.db") == 3
    assert (restore_secrets / "postgres_password").read_text() == "hunter2\n"


def test_restore_refuses_live_wal_without_force(tmp_path: Path) -> None:
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=True, rows=2)
    secrets.mkdir(parents=True)

    ops.cmd_backup(_backup_args(store, secrets, out))
    archive = next(out.glob("himmy-backup-*.tar.gz"))

    # simulate a live server: a WAL side file in the target store dir
    (store / "storage.db-wal").write_bytes(b"\x00")

    with pytest.raises(SystemExit) as exc:
        ops.cmd_restore(_restore_args(archive, store, secrets, force=False))
    assert "WAL" in str(exc.value)

    # with --force it proceeds
    assert ops.cmd_restore(_restore_args(archive, store, secrets, force=True)) == 0


def test_backup_skips_live_wal_shm_side_files(tmp_path: Path) -> None:
    """A writer holding the db open (wal_autocheckpoint=0) leaves live -wal/-shm; the
    archive must capture only the consistent .db snapshot, never the side files."""
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    store.mkdir(parents=True)
    secrets.mkdir(parents=True)

    db = store / "storage.db"
    # Open a writer that keeps the WAL un-checkpointed (so -wal/-shm stay on disk).
    writer = sqlite3.connect(db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        writer.executemany("INSERT INTO t (id) VALUES (?)", [(i,) for i in range(7)])
        writer.commit()
        # The side files exist while the connection is open and uncheckpointed.
        assert (store / "storage.db-wal").exists()

        assert ops.cmd_backup(_backup_args(store, secrets, out)) == 0
        archive = next(out.glob("himmy-backup-*.tar.gz"))

        with tarfile.open(archive) as tar:
            members = set(tar.getnames())
            manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
        names = {f["name"] for f in manifest["files"]}
    finally:
        writer.close()

    # Neither the tar members nor the manifest may contain any side file.
    assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in members), members
    assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in names), names
    assert "storage.db" in names  # the consistent snapshot IS captured


def test_restore_removes_stale_wal_shm_adjacent_to_db(tmp_path: Path) -> None:
    """Restoring over a dir that still has a stale <db>-wal/-shm must delete them so
    SQLite cannot replay a torn WAL over the clean backup snapshot."""
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=False, rows=4)
    secrets.mkdir(parents=True)

    assert ops.cmd_backup(_backup_args(store, secrets, out)) == 0
    archive = next(out.glob("himmy-backup-*.tar.gz"))

    # Restore into a target dir that has a STALE -wal/-shm but no live db open. Use a
    # fresh dir (no live WAL guard) and plant stale side files there.
    restore_store = tmp_path / "restored"
    restore_store.mkdir(parents=True)
    stale_wal = restore_store / "storage.db-wal"
    stale_shm = restore_store / "storage.db-shm"
    stale_wal.write_bytes(b"\x00stale-wal")
    stale_shm.write_bytes(b"\x00stale-shm")

    # --force because the planted side files trip the live-WAL guard (intentional).
    assert (
        ops.cmd_restore(
            _restore_args(archive, restore_store, restore_store / "secrets", force=True)
        )
        == 0
    )
    # The stale side files must be gone, and the restored db readable + correct.
    assert not stale_wal.exists()
    assert not stale_shm.exists()
    assert _count(restore_store / "storage.db") == 4


def test_restore_rejects_unsafe_manifest_paths(tmp_path: Path) -> None:
    """A manifest name that is absolute or contains ``..`` is rejected before any
    write, so a hostile archive cannot escape the target dirs."""
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=False, rows=1)
    secrets.mkdir(parents=True)

    ops.cmd_backup(_backup_args(store, secrets, out))
    archive = next(out.glob("himmy-backup-*.tar.gz"))

    # Rewrite the manifest with a traversal name pointing at the real db member.
    work = tmp_path / "work"
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    manifest = json.loads((work / "manifest.json").read_text())
    for meta in manifest["files"]:
        if meta["name"] == "storage.db":
            meta["name"] = "../escape.db"
    (work / "manifest.json").write_text(json.dumps(manifest))
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        for item in sorted(work.rglob("*")):
            if item.is_file():
                tar.add(item, arcname=str(item.relative_to(work)))

    restore_store = tmp_path / "restored"
    with pytest.raises(SystemExit) as exc:
        ops.cmd_restore(_restore_args(evil, restore_store, restore_store / "secrets"))
    assert "unsafe manifest path" in str(exc.value)
    # Nothing should have been written outside the target.
    assert not (tmp_path / "escape.db").exists()


def test_backup_excludes_secrets_by_default(tmp_path: Path) -> None:
    """The KEK / DB password are NOT bundled unless --include-secrets is passed."""
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=False, rows=2)
    secrets.mkdir(parents=True)
    (secrets / "HIMMY_ENCRYPTION_KEY").write_text("kek\n", encoding="utf-8")
    (secrets / "postgres_password").write_text("pw\n", encoding="utf-8")

    assert ops.cmd_backup(_backup_args(store, secrets, out)) == 0
    archive = next(out.glob("himmy-backup-*.tar.gz"))
    with tarfile.open(archive) as tar:
        members = set(tar.getnames())
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
    # No secret bytes anywhere in the archive (members or manifest), and the data IS.
    assert not any(n.startswith("secrets/") for n in members), members
    names = {f["name"] for f in manifest["files"]}
    assert not any(n.startswith("secrets/") for n in names), names
    assert "storage.db" in names
    assert manifest["secrets_included"] is False


def test_backup_archive_is_owner_only(tmp_path: Path) -> None:
    """The archive is created 0600 so it isn't world-readable on a shared host."""
    import stat

    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=False, rows=1)
    secrets.mkdir(parents=True)

    assert ops.cmd_backup(_backup_args(store, secrets, out)) == 0
    archive = next(out.glob("himmy-backup-*.tar.gz"))
    mode = stat.S_IMODE(archive.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_restore_rejects_checksum_tamper(tmp_path: Path) -> None:
    store = tmp_path / ".himmy"
    secrets = store / "secrets"
    out = tmp_path / "out"
    _make_db(store / "storage.db", wal=False, rows=2)
    secrets.mkdir(parents=True)

    ops.cmd_backup(_backup_args(store, secrets, out))
    archive = next(out.glob("himmy-backup-*.tar.gz"))

    # rebuild the archive with a tampered storage.db but the ORIGINAL manifest,
    # so the recorded sha256 no longer matches the member bytes.
    work = tmp_path / "work"
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    (work / "storage.db").write_bytes(b"corrupted-bytes-not-a-db")
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as tar:
        for item in sorted(work.rglob("*")):
            if item.is_file():
                tar.add(item, arcname=str(item.relative_to(work)))

    restore_store = tmp_path / "restored"
    with pytest.raises(SystemExit) as exc:
        ops.cmd_restore(
            _restore_args(tampered, restore_store, restore_store / "secrets")
        )
    assert "checksum mismatch" in str(exc.value)
