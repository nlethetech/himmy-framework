# Upgrades & schema migrations

himmy's durable storage evolves by **forward-only, additive migrations** that the
server applies for you at startup. There is no down-migration: rolling *back* a
version means restoring a pre-upgrade backup, not reversing a schema change. This
page is the operator runbook for upgrading a real deployment safely.

## How migrations work

The two storage backends version their schema differently:

- **Postgres** carries a versioned `schema_migrations` table. Every migration in
  `STORAGE_MIGRATIONS` (`himmy/services/storage/postgres.py`) has an integer version,
  a name, and a list of idempotent DDL statements. On startup the server applies each
  migration whose version is not yet recorded, at most once, inside a transaction, and
  records it in `schema_migrations`. It takes a **session advisory lock** first, so
  when several workers boot at once exactly one applies the pending set and the others
  wait, then re-read the table — safe across a multi-worker rollout. It is
  **idempotent**: an already-migrated store is a no-op, so restarts are free.

- **SQLite** stores have **no migration table**. Each store creates its schema
  idempotently when it's opened (`CREATE TABLE IF NOT EXISTS …`), so there is nothing
  to "migrate" and nothing to record — opening a store always brings its schema up to
  the running binary's expected shape. (This is why `himmy doctor --storage` shows a
  `schema_migrations` report for Postgres but only size + journal mode for SQLite.)

Because every Postgres migration only *adds* (new tables/columns/indexes), an older
binary can still read a newer schema's pre-existing columns — but do not rely on that
as a rollback strategy; restore from backup instead.

## Version notes

| Version | Schema version | Notes |
|---|---|---|
| 0.1.0 | 1 (`base_schema`) | Initial release. On Postgres, creates `schema_migrations` and the base storage schema; SQLite stores create their schema idempotently at open (no migration table). |

When you ship a schema change, add the migration to `STORAGE_MIGRATIONS` and a row
here recording the himmy version and the new schema version.

## Upgrade procedure

1. **Back up first** (WAL-safe; do this before anything else):

   ```
   python scripts/ops_backup.py backup --out /backups
   ```

   This snapshots each SQLite store via the online-backup API and bundles a
   checksummed `manifest.json`. With Postgres, pass `--dsn "$HIMMY_DATABASE_URL"` to
   also include a `pg_dump`.

2. **Install the new version.** Pick whichever matches your topology:

   ```
   pip install -U himmy          # pip/venv install
   docker compose -f deploy/compose/docker-compose.yml build studio   # compose image
   helm upgrade himmy deploy/helm/himmy-studio                        # Kubernetes
   ```

3. **Start the server.** Migrations apply automatically on boot — no manual step.
   On Postgres the advisory lock makes a multi-worker / multi-replica start safe.

4. **Verify the schema is current:**

   ```
   himmy doctor --storage
   ```

   On Postgres this prints `applied: [...]`, `code max: N`, and either `up to date`
   or a `PENDING: [...]` line. On SQLite it lists each `.himmy/*.db` with its size and
   journal mode. For a deeper SQLite integrity check, run `PRAGMA integrity_check`
   against the store.

5. **Confirm the deployment is healthy:**

   ```
   python scripts/ops_health.py
   ```

   Exit `0` = all clear, `1` = warnings, `2` = a failed check.

## Rollback (restore-from-backup, not down-migration)

There is **no schema rollback**. To revert to a prior version:

1. Stop the server (so no live WAL/SHM files remain on the SQLite stores).
2. Restore the backup taken before the upgrade:

   ```
   python scripts/ops_backup.py restore /backups/himmy-backup-<stamp>.tar.gz
   ```

   `restore` verifies every file's SHA-256 against the manifest before writing, and
   refuses to overwrite a store dir that still has live WAL/SHM side files unless you
   pass `--force` — restoring over a running database corrupts it.
3. Reinstall the prior himmy version.

## Gotchas

- **Forward-only.** Migrations never reverse a change; recovery is restore-from-backup.
- **Back up *before* upgrading**, every time. A migration that adds a column is cheap
  to roll forward and impossible to roll back without a snapshot.
- **SQLite is single-writer.** Stop the server before restoring; never restore over a
  store with live `*-wal`/`*-shm` files.
- **Additive only.** Don't author destructive DDL (dropping/renaming columns) as a
  migration — it breaks the "older binary can still read" property and makes
  partial-rollout rollbacks unrecoverable without downtime.
- **Table-rewriting columns lock the table.** Schema v8 adds a `BIGSERIAL` `seq` column
  to `run_events`; because a serial default is non-constant, Postgres takes an `ACCESS
  EXCLUSIVE` lock and rewrites the whole table, blocking reads/writes for the duration.
  On a large, long-lived `run_events` audit stream, run this upgrade in a **maintenance
  window** (the rewrite is proportional to row count). Fresh installs are unaffected.

## Self-learning is a cross-tenant aggregate signal (multi-tenant deployments)

The opt-in **self-learning** feature (`self_learning: true` on an agent) mines the
`TOOL_FAILED` / `TOOL_COMPLETED` audit stream into a per-tool reliability score that
reorders the bound toolset and injects a short reliability hint into the prompt.

On a shared server store — which is what `StoreFactory.for_context(server=True)` returns:
**one** SQLite file or **one** Postgres pool for every workspace — the `run_events` audit
stream is process-wide, and the reputation read is scoped only by tool name (the
`run_events` table has no workspace/subject column, only `thread_id` / `trace_id`). So a
tool's reputation is an **aggregate across all tenants** that used a same-named tool: a
built-in pack tool (`kb_search`, `web_fetch`, …) collides by name across workspaces, so
one workspace's flaky usage can nudge another workspace's tool ordering and reliability
hint.

What does **not** cross the boundary: only tool names and integer counts are ever read or
rendered — never prompts, tool arguments, error text, or PII. So this is an aggregate
reliability *signal* shared across tenants, not a content leak.

If your deployment requires strict per-tenant isolation of this signal, **leave
`self_learning` off** (it defaults off) until a tenant-scoped reputation read is available;
turning it on is a deliberate decision to share that aggregate signal across workspaces.

## Related docs

- [Deployment runbook](deployment.md) — the compose / Helm topologies these commands
  upgrade, plus configuration and reverse-proxy guidance.
- [Air-gapped install bundle](airgap.md) — across the gap you rebuild and reinstall
  rather than patch in place; the same forward-only migration rules apply.
- [Sandbox backends](sandbox_backends.md) — the code-execution isolation an upgraded
  served deployment should keep configured.
