#!/usr/bin/env python3
"""Back up and restore a himmy deployment's durable state.

A himmy site's durable state is a directory of SQLite stores (``.himmy/*.db``) plus
the secrets files that decrypt/authenticate them. Copying a live ``.db`` with the
filesystem is unsafe — a WAL-mode database has uncommitted pages in a side file —
so this tool uses the SQLite online-backup API (``src.backup(dst)``) to snapshot
each database consistently while the server keeps running, then bundles the
snapshots + a checksummed ``manifest.json`` into one ``tar.gz`` (mode 0600).

The secrets directory (the encryption KEK + Postgres password) is EXCLUDED by
default: bundling it next to the data would let anyone who reads the archive decrypt
every encrypted-at-rest field, so the KEK must be backed up SEPARATELY (store it in a
secrets manager / offline, not on the same /backups mount). ``--include-secrets`` opts
back in for an air-gapped, separately-encrypted archive only.

``restore`` is deliberately paranoid: it verifies every file's SHA-256 against the
manifest before writing anything, and REFUSES to overwrite a store directory that
still has live WAL/SHM side files (a running server) unless ``--force`` — restoring
over a live database corrupts it.

Postgres (when ``--dsn``/``HIMMY_DATABASE_URL`` is set) is dumped with ``pg_dump
-Fc`` if the client tool is on PATH; it is OPTIONAL — a SQLite-only site backs up
fine without postgres tooling, and the manifest records whether a dump was
included.

Usage:
    python scripts/ops_backup.py backup                      # -> himmy-backup-*.tar.gz
    python scripts/ops_backup.py backup --out /backups
    python scripts/ops_backup.py restore himmy-backup-*.tar.gz
    python scripts/ops_backup.py restore archive.tar.gz --force   # over live WAL
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_STORE_PATH = ".himmy/storage.db"
_DEFAULT_SECRETS_DIR = ".himmy/secrets"
_MANIFEST_NAME = "manifest.json"
_PG_DUMP_NAME = "postgres.dump"

#: SQLite side files that belong to a live/open database. They MUST NOT be copied
#: byte-for-byte into a backup: the WAL-safe snapshot is taken via the online-backup
#: API, and replaying a torn, point-in-time-mismatched WAL/SHM over that consistent
#: snapshot at restore time would corrupt it.
_SQLITE_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


def _is_sqlite_side_file(name: str) -> bool:
    """True for an SQLite WAL/SHM/journal side file (never backed up verbatim)."""
    return any(name.endswith(suffix) for suffix in _SQLITE_SIDE_SUFFIXES)


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_version() -> int | None:
    try:
        from himmy.services.storage.postgres import STORAGE_MIGRATIONS

        return max(v for v, _, _ in STORAGE_MIGRATIONS)
    except Exception:  # noqa: BLE001 - himmy import optional for a backup
        return None


def _himmy_version() -> str:
    try:
        from himmy import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _sqlite_backup(src: Path, dst: Path) -> None:
    """WAL-safe online snapshot of ``src`` into a fresh ``dst`` database file."""
    src_conn = sqlite3.connect(src)
    try:
        dst_conn = sqlite3.connect(dst)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _pg_dump(dsn: str, out: Path) -> tuple[bool, str]:
    """Run ``pg_dump -Fc`` into ``out``; return (included, reason)."""
    if shutil.which("pg_dump") is None:
        return False, "pg_dump not on PATH"
    try:
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["pg_dump", "-Fc", "--dbname", dsn, "--file", str(out)],  # noqa: S607
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"pg_dump failed ({exc.stderr.decode('utf-8', 'replace')[:200]})"
    return True, "included"


def cmd_backup(args: argparse.Namespace) -> int:
    store_dir = Path(args.store_path).parent
    secrets_dir = Path(args.secrets_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not store_dir.is_dir():
        raise SystemExit(f"error: store dir {store_dir} does not exist")

    staging = Path(tempfile.mkdtemp(prefix="himmy-backup-"))
    files_meta: list[dict[str, Any]] = []
    try:
        # 1. SQLite databases via the online-backup API (WAL-safe).
        for db in sorted(store_dir.glob("*.db")):
            dst = staging / db.name
            _sqlite_backup(db, dst)
            files_meta.append(
                {"name": db.name, "sha256": _sha256(dst), "bytes": dst.stat().st_size}
            )
            _eprint(f"  backed up {db.name}")

        # 2. Non-db files in the store dir (plain byte copy + checksum). SQLite WAL/
        #    SHM/journal side files are deliberately skipped — the consistent .db
        #    snapshot above already captured their committed contents, and bundling a
        #    live side file would let restore replay a torn WAL over the clean snapshot.
        for extra in sorted(store_dir.iterdir()):
            if (
                extra.is_file()
                and extra.suffix != ".db"
                and not _is_sqlite_side_file(extra.name)
            ):
                dst = staging / extra.name
                shutil.copy2(extra, dst)
                files_meta.append(
                    {
                        "name": extra.name,
                        "sha256": _sha256(dst),
                        "bytes": dst.stat().st_size,
                    }
                )

        # 3. Secrets directory (preserve under a secrets/ subtree).
        #    EXCLUDED BY DEFAULT: this dir holds the encryption KEK and the Postgres
        #    password. Bundling them next to the data means anyone who reads the backup
        #    (a shared /backups mount, a backup server operator) gets the key that
        #    decrypts every encrypted-at-rest field — encryption-at-rest is undone by
        #    possession of one tar.gz. Store the KEK separately and only opt in with
        #    --include-secrets for an air-gapped, separately-protected archive.
        if args.include_secrets and secrets_dir.is_dir():
            for secret in sorted(secrets_dir.iterdir()):
                if secret.is_file():
                    rel = Path("secrets") / secret.name
                    dst = staging / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(secret, dst)
                    files_meta.append(
                        {
                            "name": str(rel),
                            "sha256": _sha256(dst),
                            "bytes": dst.stat().st_size,
                        }
                    )
            _eprint(
                "  WARNING: --include-secrets bundled the KEK/DB password into the "
                "archive — store it encrypted with restricted access (it is as "
                "sensitive as the KEK itself)."
            )
        elif secrets_dir.is_dir():
            _eprint(
                "  secrets EXCLUDED (default) — back up .himmy/secrets separately; "
                "the data archive is useless without the KEK, which is the point. "
                "Pass --include-secrets to bundle them anyway."
            )

        # 4. Optional Postgres dump.
        pg: dict[str, Any] = {"included": False, "reason": "no DSN configured"}
        if args.dsn:
            dump_path = staging / _PG_DUMP_NAME
            included, reason = _pg_dump(args.dsn, dump_path)
            pg = {"included": included, "reason": reason}
            if included:
                files_meta.append(
                    {
                        "name": _PG_DUMP_NAME,
                        "sha256": _sha256(dump_path),
                        "bytes": dump_path.stat().st_size,
                    }
                )
            else:
                _eprint(f"  postgres dump skipped: {reason}")

        manifest = {
            "himmy_version": _himmy_version(),
            "created_utc": _utc_stamp(),
            "schema_version": _schema_version(),
            "files": files_meta,
            "postgres": pg,
            "secrets_included": bool(args.include_secrets),
        }
        (staging / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        archive = out_dir / f"himmy-backup-{manifest['created_utc']}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(staging.rglob("*")):
                if item.is_file():
                    tar.add(item, arcname=str(item.relative_to(staging)))
        # The archive can hold the entire database (and, with --include-secrets, the
        # KEK). Restrict it to the owner so it isn't world-readable on a shared host.
        os.chmod(archive, 0o600)
        print(str(archive))
        _eprint(f"backup complete: {len(files_meta)} file(s) -> {archive}")
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _has_live_wal(store_dir: Path) -> list[str]:
    """Return names of any live WAL/SHM side files under ``store_dir``."""
    return sorted(
        p.name
        for p in store_dir.iterdir()
        if p.is_file() and (p.name.endswith("-wal") or p.name.endswith("-shm"))
    )


def cmd_restore(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    if not archive.is_file():
        raise SystemExit(f"error: archive {archive} not found")
    store_dir = Path(args.store_path).parent
    secrets_dir = Path(args.secrets_dir)

    if store_dir.is_dir():
        live = _has_live_wal(store_dir)
        if live and not args.force:
            raise SystemExit(
                f"error: live WAL/SHM files present in {store_dir} ({', '.join(live)}) "
                "— a server appears to be running. Stop it first, or pass --force to "
                "overwrite (this WILL corrupt a live database)."
            )

    staging = Path(tempfile.mkdtemp(prefix="himmy-restore-"))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")  # noqa: S202 - filter guards traversal

        manifest_path = staging / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise SystemExit("error: archive has no manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Reject any manifest name that could escape the staging/store/secrets dirs
        # when later joined (absolute paths or ``..`` components) BEFORE we read or
        # write a single member — a hostile archive must never write outside target.
        for meta in manifest.get("files", []):
            name = meta["name"]
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise SystemExit(f"error: unsafe manifest path {name!r}")

        # Verify EVERY file's checksum before writing anything.
        for meta in manifest.get("files", []):
            member = staging / meta["name"]
            if not member.is_file():
                raise SystemExit(f"error: manifest lists missing file {meta['name']}")
            actual = _sha256(member)
            if actual != meta["sha256"]:
                raise SystemExit(
                    f"error: checksum mismatch for {meta['name']} "
                    f"(manifest {meta['sha256'][:12]}…, got {actual[:12]}…)"
                )
        _eprint(f"verified {len(manifest.get('files', []))} checksum(s)")

        store_dir.mkdir(parents=True, exist_ok=True)
        secrets_dir.mkdir(parents=True, exist_ok=True)
        for meta in manifest.get("files", []):
            name = meta["name"]
            src = staging / name
            if name == _PG_DUMP_NAME:
                continue  # handled below
            if name.startswith("secrets/"):
                dst = secrets_dir / Path(name).name
            else:
                dst = store_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            # The snapshot is a fully checkpointed, side-file-free database. If a STALE
            # ``<db>-wal``/``<db>-shm`` from a previous server lingers next to it, SQLite
            # would replay it over the clean snapshot on first open — corrupting it.
            # Remove the adjacent side files so the restored .db opens standalone.
            if dst.suffix == ".db":
                for suffix in _SQLITE_SIDE_SUFFIXES:
                    side = dst.with_name(dst.name + suffix)
                    if side.exists():
                        side.unlink()

        # Optional Postgres restore.
        pg = manifest.get("postgres", {})
        if pg.get("included") and args.dsn:
            dump = staging / _PG_DUMP_NAME
            if shutil.which("pg_restore") is None:
                _eprint("warning: pg_restore not on PATH — skipping Postgres restore")
            else:
                subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [  # noqa: S607 - bare pg_restore resolved via PATH
                        "pg_restore",
                        "--clean",
                        "--if-exists",
                        "--dbname",
                        args.dsn,
                        str(dump),
                    ],
                    check=True,
                    capture_output=True,
                )
                _eprint("postgres restore complete")

        print(
            "restore complete — verify with `himmy doctor --storage` and "
            "`PRAGMA integrity_check`"
        )
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_backup = sub.add_parser("backup", help="snapshot the durable state to a tar.gz")
    p_backup.add_argument("--store-path", default=_DEFAULT_STORE_PATH)
    p_backup.add_argument("--secrets-dir", default=_DEFAULT_SECRETS_DIR)
    p_backup.add_argument(
        "--include-secrets",
        action="store_true",
        help="bundle .himmy/secrets (KEK + DB password) into the archive. OFF by "
        "default — the KEK must be backed up separately so the data archive alone "
        "cannot decrypt the data. Only use for a separately-protected, encrypted archive.",
    )
    p_backup.add_argument("--out", default=".", help="output directory for the archive")
    p_backup.add_argument(
        "--dsn",
        default=os.environ.get("HIMMY_DATABASE_URL"),
        help="Postgres DSN to pg_dump (default $HIMMY_DATABASE_URL; optional)",
    )
    p_backup.set_defaults(func=cmd_backup)

    p_restore = sub.add_parser("restore", help="restore a tar.gz produced by backup")
    p_restore.add_argument("archive", help="path to the himmy-backup-*.tar.gz")
    p_restore.add_argument("--store-path", default=_DEFAULT_STORE_PATH)
    p_restore.add_argument("--secrets-dir", default=_DEFAULT_SECRETS_DIR)
    p_restore.add_argument(
        "--force",
        action="store_true",
        help="overwrite even if live WAL/SHM files are present (corrupts a live db)",
    )
    p_restore.add_argument(
        "--dsn",
        default=os.environ.get("HIMMY_DATABASE_URL"),
        help="Postgres DSN to pg_restore into (default $HIMMY_DATABASE_URL; optional)",
    )
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
