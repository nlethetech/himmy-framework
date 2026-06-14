# Backup & disaster recovery runbook

Operator procedures for backing up and recovering a himmy deployment's durable
state. This is the focused companion to the [deployment runbook](deployment.md);
it documents only artifacts that exist today and the flags they actually take.

A himmy site's durable state is three things, and a recoverable backup needs all
three:

1. **The SQLite stores** — a directory of single-writer `.himmy/*.db` files
   (run/event store, approvals, sessions, the CLI trace, and any graph
   checkpoint DB). Always present.
2. **Postgres** (optional) — the durable entity store, used only when
   `HIMMY_DATABASE_URL` points at a `postgresql://…` DSN. The compose stack
   always sets it; a bare SQLite-only site does not have it.
3. **The secrets directory** — `.himmy/secrets`, holding the encryption KEK
   (`HIMMY_ENCRYPTION_KEY`) and the Postgres password. This is backed up
   **separately** from the data and is required to read encrypted-at-rest fields.
   See [Encryption-key escrow](#encryption-key-kek-escrow--the-data-is-useless-without-it).

> **NEVER `cp`, `tar`, or `rsync` a live `.himmy/*.db` file.** A WAL-mode SQLite
> database has uncommitted pages in a `-wal` side file, so a byte copy captures a
> torn, point-in-time-mismatched database. Use `scripts/ops_backup.py`, which
> snapshots each store through the SQLite online-backup API and **intentionally
> skips** the `-wal`/`-shm`/`-journal` side files.

## What `scripts/ops_backup.py` does

`scripts/ops_backup.py` is the supported tool for the SQLite state (and,
optionally, Postgres). It has two subcommands, `backup` and `restore`.

- **`backup`** snapshots every `.himmy/*.db` through the SQLite **online-backup
  API** (`sqlite3.Connection.backup`) into a fresh, fully-checkpointed file — safe
  to run while the server is live. It copies the non-`.db` store files verbatim,
  skips the `-wal`/`-shm`/`-journal` side files, writes a checksummed
  `manifest.json` (recording himmy version, schema version, and a SHA-256 per
  file), and bundles everything into a single `himmy-backup-<utc>.tar.gz` created
  mode `0600`.
- **`restore`** verifies every file's SHA-256 against the manifest **before
  writing anything**, refuses to overwrite a store directory that still has live
  `-wal`/`-shm` side files (a running server) unless `--force`, and deletes any
  stale side files next to a restored `.db` so SQLite opens it standalone.

Flags (defaults shown):

| Command | Flag | Default | Meaning |
|---|---|---|---|
| `backup` | `--store-path` | `.himmy/storage.db` | Its parent dir is the store directory that gets snapshotted. |
| `backup` | `--secrets-dir` | `.himmy/secrets` | Secrets source; **excluded** unless `--include-secrets`. |
| `backup` | `--include-secrets` | off | Bundle the KEK + DB password into the archive. See the warning below. |
| `backup` | `--out` | `.` | Output directory for the `.tar.gz`. |
| `backup` | `--dsn` | `$HIMMY_DATABASE_URL` | Postgres DSN to `pg_dump`; optional. |
| `restore` | `archive` | (positional) | Path to a `himmy-backup-*.tar.gz`. |
| `restore` | `--store-path` | `.himmy/storage.db` | Target store directory (parent). |
| `restore` | `--secrets-dir` | `.himmy/secrets` | Where `secrets/` members are restored, if present. |
| `restore` | `--force` | off | Overwrite even with live WAL/SHM present (**corrupts a live db**). |
| `restore` | `--dsn` | `$HIMMY_DATABASE_URL` | Postgres DSN to `pg_restore` into; optional. |

The `Makefile` target `make ops-backup` wraps `python scripts/ops_backup.py
backup` (writing to the repo root).

## Back up — SQLite (always)

```bash
# Snapshot every .himmy/*.db (WAL-safe) into /backups/himmy-backup-<utc>.tar.gz
python scripts/ops_backup.py backup --out /backups
```

The resulting archive contains a consistent snapshot of every `.himmy/*.db`, the
non-db store files, and `manifest.json`. The secrets directory is **excluded by
default** (see escrow section). This is safe to run against a live server — the
online-backup API takes a consistent snapshot without stopping writers.

## Back up — Postgres (when configured)

When `HIMMY_DATABASE_URL` is set and `pg_dump` is on `PATH`, `ops_backup.py`
automatically includes a `pg_dump -Fc` (custom-format) dump in the same archive;
if `pg_dump` is absent the SQLite-only backup still succeeds and the manifest
records that the Postgres dump was skipped. You can also dump Postgres directly:

```bash
# Custom-format dump (same as what ops_backup.py embeds)
pg_dump -Fc --dbname "$HIMMY_DATABASE_URL" --file himmy-pg.dump
```

For a higher-RPO Postgres setup, layer on one of:

- **Continuous WAL archiving / PITR.** Configure `archive_mode = on` and an
  `archive_command` (or a managed service's continuous backup) so you can replay
  the write-ahead log to a chosen point in time. This bounds data loss to seconds
  rather than the dump interval. Restore is base backup + WAL replay (managed by
  your Postgres tooling), not `pg_restore`.
- **Volume / disk snapshots.** A filesystem or cloud block-storage snapshot of the
  Postgres data volume is a coarse-grained alternative. Snapshot a **quiesced or
  crash-consistent** volume; for compose this is the `himmy_pgdata_full` named
  volume. A volume snapshot captures Postgres only — you still need the SQLite
  `.himmy/*.db` backup (and the secrets) for a complete recovery.

These are complementary to (not a replacement for) the `.himmy/*.db` SQLite
backup: even with Postgres, the sidecar SQLite stores (traces, run store,
sessions, approvals) are single-writer and live only on the `.himmy` volume.

## Restore — step by step

Restore is destructive and must run against a **stopped** server (no live WAL).

### SQLite (and an embedded Postgres dump)

```bash
# 1. Stop the deployment so no live -wal/-shm side files exist.
docker compose -f deploy/compose/docker-compose.yml down     # or: helm scale ... --replicas=0

# 2. Restore the .himmy state (checksum-verified before any write).
python scripts/ops_backup.py restore himmy-backup-<utc>.tar.gz
```

`restore` verifies every member's SHA-256 against the manifest first, then writes
the stores and removes any stale side files. If the archive embeds a Postgres
dump **and** `HIMMY_DATABASE_URL` is set **and** `pg_restore` is on `PATH`, it
also runs `pg_restore --clean --if-exists` into that DSN automatically.

If you see `live WAL/SHM files present … a server appears to be running`, the
server is still up — stop it. Use `--force` only if you are certain nothing is
writing; it will corrupt a live database.

### Postgres from a standalone dump

If you took a separate `pg_dump -Fc` (not embedded in the archive):

```bash
pg_restore --clean --if-exists --dbname "$HIMMY_DATABASE_URL" himmy-pg.dump
```

For a **PITR** setup, follow your Postgres tooling's base-backup-restore + WAL
replay procedure instead of `pg_restore`; for a **volume snapshot**, restore the
underlying volume and start Postgres against it.

### Full disaster-recovery sequence

1. **Stop** the deployment (`compose down`, or scale the Helm release to 0) so no
   live WAL exists.
2. **Restore the SQLite state** with `ops_backup.py restore` (into the
   `himmy_state` volume for compose; the RWO PVC for Helm).
3. **Restore Postgres** — let `restore` `pg_restore` the embedded dump, or restore
   from your own `pg_dump`/PITR/volume snapshot.
4. **Restore secrets** from your **separate** secrets backup — the Postgres
   password and `HIMMY_ENCRYPTION_KEY` (the data archive excludes them by
   default). Encrypted fields are unreadable without the original KEK, so this
   step is **mandatory**. If you backed up with `--include-secrets`, `restore`
   writes the `secrets/` subtree from the archive instead.
5. **Start** the deployment.
6. **Verify** — `himmy doctor --storage` and `python scripts/ops_health.py`
   (expect exit `0`).

## Encryption-key (KEK) escrow — the data is useless without it

Encryption-at-rest uses an **envelope** scheme: each value is encrypted with a
fresh AES-GCM data key (DEK) that is itself wrapped by the long-lived
key-encryption key (KEK). The KEK comes from `HIMMY_ENCRYPTION_KEY` (a base64
32-byte value) for the default `local` `HIMMY_KEK_PROVIDER`, or from a cloud KMS
CMK (`HIMMY_AWS_KMS_KEY_ID`) when `HIMMY_KEK_PROVIDER=aws-kms`.

> ## ⚠️ Lose the KEK and the encrypted data is gone — permanently, by design
>
> Every encrypted field can only be decrypted with the same KEK that wrapped its
> DEK. **If you lose `HIMMY_ENCRYPTION_KEY` (local) or the cloud KMS key
> (`aws-kms`), all encrypted-at-rest data is permanently unrecoverable.** There is
> no recovery path, no backdoor, and no master key — that irreversibility is the
> security property, not a bug. A data backup taken **without** the KEK is, for
> encrypted fields, deliberately undecryptable.

Escrow rules:

- **`scripts/ops_backup.py` excludes `.himmy/secrets` by default.** Bundling the
  KEK next to the data would let anyone who reads the backup decrypt every
  encrypted field, and backups routinely travel to lower-trust storage — that one
  `tar.gz` would undo encryption-at-rest entirely.
- **Back up the KEK separately**, on its own lifecycle: a secrets manager (Vault /
  AWS / GCP / Azure — the same backends `HIMMY_SECRETS` supports), an offline copy
  in a safe, or your existing key-distribution channel. Store at least two
  independent copies in different locations.
- `--include-secrets` opts the KEK + DB password **into** the archive. Use it
  **only** for an air-gapped, separately-encrypted archive whose access you fully
  control; that archive is then **as sensitive as the KEK itself** and must be
  stored encrypted with restricted access.
- For `HIMMY_KEK_PROVIDER=aws-kms`, the KEK never leaves KMS — "escrow" means
  protecting the **CMK** (key policy, deletion protection, cross-region replica
  for DR). Losing or scheduling deletion of that CMK is equally terminal for the
  data.

**Key rotation.** The envelope design means rotating the KEK **never requires
re-encrypting data** — only re-wrapping DEKs. See the encryption details in
[Storage Service → Optional payload encryption](../services/storage.md#optional-payload-encryption-encryptionpy)
and the encryption-at-rest variables in the
[deployment runbook](deployment.md#encryption-at-rest). When you rotate, keep the
**previous** KEK escrowed until you are certain no ciphertext was wrapped under
it.

## RPO / RTO guidance

**RPO (recovery point objective — how much data you can lose):**

| Approach | Effective RPO |
|---|---|
| Scheduled `ops_backup.py backup` (cron) | The backup interval — e.g. an hourly cron ⇒ up to ~1 h of lost runs/sessions. |
| Postgres continuous WAL archiving / PITR | Seconds (last archived WAL segment). Applies to the Postgres entity store only; the SQLite sidecars still follow the snapshot interval. |
| Volume snapshots | The snapshot interval, like scheduled backups. |

The dominant grower is `.himmy/storage.db` (several rows per agent turn), so a
busy site favours a tighter SQLite backup interval. Run `ops_backup.py backup`
from cron at a cadence that matches your tolerance for lost audit/run history, and
**keep the KEK backup on its own (less frequent) lifecycle** — it changes only on
rotation.

**RTO (recovery time objective — how long recovery takes):** dominated by
(1) provisioning the host/cluster, (2) `ops_backup.py restore` (checksum-verify +
copy, roughly proportional to total `.himmy` size), and (3) any `pg_restore` or
WAL replay (proportional to dump size / WAL volume). For a single-host compose
deployment this is typically minutes. Rehearse the
[full DR sequence](#full-disaster-recovery-sequence) on a scratch host so the RTO
is measured, not assumed — and confirm the **separately-escrowed KEK** is part of
the rehearsal, since a restore that can't decrypt is not a successful recovery.

## Related docs

- [Deployment runbook](deployment.md) — install, configure, monitor; the broader backup/restore context and the `HIMMY_*` configuration reference (including [Encryption-at-rest](deployment.md#encryption-at-rest)).
- [Storage Service](../services/storage.md) — the envelope encryption scheme and the SQLite/Postgres stores.
- [Upgrades](upgrades.md) — schema migrations and version upgrades (run before restoring across versions).
- [Air-gapped installs](airgap.md) — the `himmy_ollama` model-volume contract referenced in DR.
