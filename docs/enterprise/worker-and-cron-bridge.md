# Worker & OS-cron bridge runbook

How to run himmy's scheduled routines and durable run-queue **without a web
server**, and how to bridge an OS-cron / systemd-timer line into the same durable
pipeline. This is the operator reference for the `himmy worker` command and the
hardened `himmy routines run-now` bridge.

## Why this exists

The routines layer (`himmy/api/routines.py`) and the leased run-queue dispatcher
(`himmy/application/dispatcher.py`) are normally only started **inside the FastAPI
server**:

- the **dispatcher** is started in the app lifespan (`himmy/api/app.py`), and
- the **routine scheduler** is started by a Studio-router startup event
  (`himmy/api/routers/studio_routines.py`).

So a CLI / desktop user with **no server** got *neither*: routines silently never
fired offline, and any run that was enqueued never drained. `himmy worker` closes
that gap by standing up the **same durable substrate the server lifespan builds**,
in one process, with no HTTP listener.

Both `himmy worker` and the FastAPI lifespan now call **one shared bootstrap**
(`himmy/api/runtime_bootstrap.py`), so the wiring can never drift between them.

## What "durable" actually requires

A queue is worthless if it loses runs on exit. `create_run` **defaults to inline
fire-and-forget** — durability is engaged only when *all four* of these are true:

1. a **durable run store** is active (SQLite file or Postgres),
2. `run_app.enable_dispatch(...)` has been called,
3. a `RunDispatcher` has been **started**, and
4. (for routines) the **routine container provider** is installed so `agent_id`
   routines resolve a real run/agent service offline.

`himmy worker` does all four. If you point it at an in-memory store it will still
run, but it logs a warning and stays in inline mode — it is honest about not being
durable.

Select the durable store with **either**:

- `HIMMY_DATABASE_URL=postgres://…` → Postgres (the multi-node story, below), or
- `HIMMY_DURABLE_STORAGE=1` → file-backed SQLite at `HIMMY_STORE_PATH`
  (default `.himmy/storage.db`). `himmy worker` sets `HIMMY_DURABLE_STORAGE=1` for
  you if neither is set, so a bare `himmy worker` is durable on SQLite out of the box.

## Start the worker

```bash
# Scheduler + queue worker (the desktop "everything offline" mode).
himmy worker

# Point the durable SQLite store somewhere explicit.
himmy worker --store /var/lib/himmy/storage.db

# Size the dispatcher fan-out.
himmy worker --concurrency 4

# Queue worker only — drain runs others enqueue, fire NO routines.
himmy worker --no-scheduler

# Scheduler only — tick routines, but let a SEPARATE worker pool drain the queue.
himmy worker --scheduler-only
```

On startup the worker logs exactly what it wired, e.g.:

```
himmy worker up: store=sqlite dispatch=on scheduler=on routines=3 (owner=12345-ab12cd34)
```

It blocks until **SIGINT / SIGTERM**, then shuts down gracefully: it stops the
scheduler first (no new routine runs), then drains the dispatcher's in-flight
runs, drains inline tasks, closes the store, and clears the process-wide providers.

### Knobs

| Flag / env | Effect |
|---|---|
| `--store PATH` / `HIMMY_STORE_PATH` | File-backed SQLite run store path (default `.himmy/storage.db`). Ignored when `HIMMY_DATABASE_URL` selects Postgres. |
| `--concurrency N` / `HIMMY_DISPATCH_CONCURRENCY` | Max concurrent dispatched runs (default 8). |
| `HIMMY_DATABASE_URL` | `postgres://…` → Postgres run store (and the only path to cross-node single-fire). |
| `HIMMY_DURABLE_STORAGE=1` | Opt into the durable SQLite store (set automatically by `himmy worker`). |
| `HIMMY_RUN_TIMEOUT_SECONDS` | Per-run wall clock + lease TTL basis. |
| `HIMMY_DISPATCH_MAX_ATTEMPTS` | Per-run retry ceiling. |
| `HIMMY_ROUTINES_SCHEDULER` | `off`/`0`/`false`/`no` disables the scheduler (parity with the server router gate). |
| `HIMMY_LOG_LEVEL` | Worker log verbosity (default `INFO`). |

## Run as a systemd service

```ini
# /etc/systemd/system/himmy-worker.service
[Unit]
Description=himmy routine scheduler + run-queue worker
After=network-online.target

[Service]
Type=simple
User=himmy
WorkingDirectory=/var/lib/himmy
Environment=HIMMY_DURABLE_STORAGE=1
Environment=HIMMY_STORE_PATH=/var/lib/himmy/storage.db
# For multi-node single-fire, use Postgres instead:
# Environment=HIMMY_DATABASE_URL=postgres://himmy:...@db/himmy
ExecStart=/usr/local/bin/himmy worker --concurrency 4
Restart=on-failure
RestartSec=5
# systemd sends SIGTERM on stop; the worker drains and exits cleanly.
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now himmy-worker
journalctl -u himmy-worker -f   # watch the startup banner + run lifecycle
```

## The OS-cron bridge: `himmy routines run-now`

himmy's built-in schedule kinds are `daily at HH:MM` and `every N hours`. When you
want **OS-native scheduling** (a crontab, a systemd timer, an external scheduler)
to drive a routine, point it at `himmy routines run-now <id>`. This is the
**bridge**: a cron line gets the full durable pipeline — the run is recorded in the
same canonical store `himmy runs` / `GET /v1/runs` / Studio read, carries `RUN_*`
lineage and an `actor.routine_id`, and is **HITL-safe** (an approval-gated tool
*pauses* the run; run-now never auto-approves).

```cron
# Every weekday at 06:30, run a local routine through the durable pipeline.
30 6 * * 1-5  cd /var/lib/himmy && HIMMY_DURABLE_STORAGE=1 /usr/local/bin/himmy routines run-now rtn_abc123 >> /var/log/himmy/run-now.log 2>&1
```

systemd-timer equivalent:

```ini
# himmy-routine.timer
[Timer]
OnCalendar=Mon..Fri 06:30
Persistent=true
[Install]
WantedBy=timers.target
```
```ini
# himmy-routine.service
[Service]
Type=oneshot
WorkingDirectory=/var/lib/himmy
Environment=HIMMY_DURABLE_STORAGE=1
ExecStart=/usr/local/bin/himmy routines run-now rtn_abc123
```

### Two routine seams, handled automatically

`run-now` mirrors the scheduler's two execution seams, picked by how the routine
binds its agent:

- **`agent_path`** (the single-user-local CLI/Studio routine): executes inline
  through the Studio stream pipeline and is mirrored into the canonical store.
  Durable + lineage-tracked; the `actor.routine_id` is stamped on the record.
- **`agent_id`** (a workspace-scoped `/v1` routine — reachable only against a
  **shared durable routines store**): dispatched through `RunAppService.create_run`
  on a full durable container (the same one `himmy worker` builds). Without a
  worker draining the queue the run executes **inline here**; with a worker it is
  dispatcher-routed. Either way it is durable. An `agent_id` `run-now` with **no
  durable store** is **refused** with a clear error rather than silently lost.

### `run-now` vs. a running worker

- `run-now` is a **one-shot** that settles the run before exiting (it polls to
  completion). It is complete on its own for `agent_path` routines and for
  `agent_id` routines on a durable store.
- A long-running `himmy worker` is what you want when you have **many** routines on
  himmy's own schedules, or when you enqueue runs from elsewhere and need a
  consumer to **drain the queue** continuously. Pair the two freely: a cron
  `run-now` and a running worker can never double-execute the same routine (see
  single-fire below).

## Single-fire honesty — read before you scale

A routine must not double-execute. himmy enforces this in three layers:

1. **in-process** — the scheduler's running-task registry (one process won't
   launch a routine that is already running in it);
2. **host-local** — a cross-process `flock` (`routine-<id>`) inside
   `execute_routine`, so a cron `run-now` and a co-located worker on the **same
   box** can't both run it; and
3. **cluster-wide** — an atomic `mark_started` compare-and-set gated on the
   `last_run_at` the tick observed.

The boundaries that matter:

- **One worker on one box (SQLite): SAFE.** Layers 1–3 all hold because there is a
  single store and a single host.
- **Multiple workers / multiple boxes: SAFE ONLY ON SHARED POSTGRES.** The `flock`
  in layer 2 is **host-local** — it does *not* coordinate across nodes. The only
  cross-node guard is layer 3's compare-and-set, and that holds **only when every
  node shares the same durable routines store *and* run store** — i.e.
  `HIMMY_DATABASE_URL` Postgres for both. **Per-node SQLite does NOT coordinate
  across nodes.**

> Do not run more than one `himmy worker` (or worker + cron `run-now`) across
> multiple hosts on per-node SQLite stores and expect single-fire. Move to a
> shared Postgres (`HIMMY_DATABASE_URL`) for multi-node, or keep it to one box.

This mirrors the rest of himmy's deployment story: the durable state is
single-writer SQLite unless you opt into Postgres. See
[`deployment.md`](deployment.md) for the broader single-replica constraints.

## Verifying it works

```bash
# 1) add a trivial local routine bound to a project agent.yaml
himmy routines add --name ping -f agent.yaml -p "say hello" --every 1

# 2) bridge it once through the durable pipeline
HIMMY_DURABLE_STORAGE=1 himmy routines run-now <id>

# 3) confirm the run is durable + attributable
himmy runs list            # the run appears here (canonical store)
# its metadata carries source=routine, actor.routine_id=<id>
```

To watch the long-running path instead:

```bash
himmy worker            # leaves the banner, then ticks the routine on its cadence
```
