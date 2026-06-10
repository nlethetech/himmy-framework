# Deployment runbook

How to install, configure, expose, monitor, back up, and recover a himmy Studio
deployment. This is the operator's reference for the artifacts under `deploy/`
and `scripts/ops_*.py`. It documents what those artifacts actually do today —
including the sharp edges — rather than an idealized topology.

himmy is offline-first and single-process at its core: the durable state is a
directory of **single-writer** SQLite stores (`.himmy/*.db`) plus an *optional*
external Postgres for the main entity store. That single-writer reality drives
almost every constraint below (single replica, RWO volumes, WAL-safe backups).
Read the [Gotchas](#gotchas) before you scale anything.

## Overview — three deployment shapes

Pick the smallest shape that fits. They share the same image, the same `.himmy`
state contract, and the same `HIMMY_*` configuration surface.

| Shape | What it is | Use it when |
|---|---|---|
| **Bare `pip` + `himmy studio`** | `pip install "himmy[studio]"` and run `himmy studio` directly on a host (no containers). State in a cwd-relative `.himmy/`. | A single trusted operator on a laptop/VM, evaluation, or a CLI-first workflow. No Postgres, no proxy. |
| **Compose full stack** | `deploy/compose/docker-compose.yml` — Studio + Postgres (+ optional bundled Ollama). Durable Postgres entity store, file-delivered secrets, named volumes. | A real, durable single-host deployment. The default recommendation for most installs. |
| **Helm chart** | `deploy/helm/himmy-studio` — a deliberately minimal single-replica chart against an **external** Postgres and a Kubernetes Secret. | You already run Kubernetes and want Studio managed alongside the rest of your platform. Still single-replica (see below). |

All three serve the same Studio SPA on port **8765** and pass the loopback guard
for `localhost`. None of them is multi-replica: the SQLite sidecar stores are
single-writer even when Postgres is configured.

The Makefile (owned by the wire-up lane) exposes convenience targets that wrap
the commands below: `docker-build`, `compose-up`, `compose-up-ollama`,
`compose-down`, `ops-health`, `ops-backup`, `helm-lint`, `airgap-bundle`.

## Install

### Build the image

The `Dockerfile` is a two-stage build: stage 1 compiles the Studio React SPA into
the Python package, stage 2 installs it with the production extras
(`.[studio,knowledge,postgres,encryption,auth]`) and runs as the non-root uid
`10001`. Postgres/encryption/auth are installed because the compose stack
*mandates* `HIMMY_DATABASE_URL` (`[postgres]`) and `.env.example` documents
encryption-at-rest and OIDC, so those features must actually be present.

```bash
# from the repo root
docker build -t himmy-studio .
# or via the Makefile
make docker-build
```

The image declares `HEALTHCHECK CMD curl -fsS http://127.0.0.1:8765/health`,
exposes 8765, and pre-creates `/app/.himmy` owned by uid 10001 so a mounted named
volume inherits the right owner (see [Gotchas](#gotchas)).

### Compose up

First provision `.env` + the secret files. The Postgres password lives in **two
places that cannot share one source** (an env-interpolated DSN and a Docker
secret file); the provisioner generates a strong value and writes both
consistently:

```bash
# generates deploy/compose/.env + secrets/postgres_password (apikey mode + key)
python scripts/ops_provision.py
# add --encryption-key to also generate secrets/HIMMY_ENCRYPTION_KEY
```

If you hand-edit instead, set the **same** `POSTGRES_PASSWORD` in `.env` *and* in
`deploy/compose/secrets/postgres_password`, and use a **URL-safe (alphanumeric)**
password — it is interpolated raw into the DSN, so `@ : / # ? $` break parsing.

Bring it up (compose resolves the build context and `./secrets` relative to the compose file, so any working directory works):

```bash
# Studio + Postgres
docker compose -f deploy/compose/docker-compose.yml up -d        # or: make compose-up

# ...plus a bundled, model-pulling Ollama (several GB on first start)
docker compose -f deploy/compose/docker-compose.yml --profile ollama up -d   # or: make compose-up-ollama

docker compose -f deploy/compose/docker-compose.yml down          # or: make compose-down
```

The `ollama` profile adds two services: `ollama` (pinned `0.13.5`, matching the
CI pin) and a one-shot `ollama-init` that pulls `qwen2.5:3b-instruct`,
`nomic-embed-text`, and `qwen3-embedding`. `ollama pull` is idempotent, so
re-running `up` is safe. Without the profile, neither starts and Studio expects
an Ollama reachable at `HIMMY_OLLAMA_URL` (or none at all).

### First-boot checklist

```bash
# 1. Storage + migrations are healthy (run inside the container/host)
docker compose -f deploy/compose/docker-compose.yml exec studio himmy doctor --storage

# 2. The server answers and reports ok (Host=localhost passes the guard)
curl -s http://localhost:8765/health        # -> {"status":"ok"}

# 3. Full operational probe (disk, sqlite quick_check, postgres migrations, ...)
python scripts/ops_health.py                 # or: make ops-health
```

Then open <http://localhost:8765>. `Host: localhost` passes the loopback guard;
to reach it any other way, see [Reverse proxy / TLS](#reverse-proxy--tls).

## Configuration reference

Every variable below is read through himmy's secret provider or `os.environ` and
was verified against source. Unset means "off / default behavior" unless noted.
Secrets (`HIMMY_DATABASE_URL`, `HIMMY_INTERNAL_API_KEY`, `HIMMY_ENCRYPTION_KEY`,
…) can be sourced from files or a vault instead of raw env — see
[Secrets](#secrets-himmy_secrets).

### Serving & Studio guard

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_STUDIO_GUARD` | `1` (on) | Loopback guard for `/api/studio` routes. **Never set to `0`** in a real deployment. |
| `HIMMY_STUDIO_ALLOW_HOSTS` | empty | Comma-separated Host values allowed past the guard (e.g. your proxy domain). loopback (`localhost`/`127.0.0.1`/`::1`/`testserver`) is always allowed. |
| `HIMMY_CORS_ORIGINS` | empty (deny) | Comma-separated allowed CORS origins. Leave unset to keep CORS same-origin/closed. |
| `HIMMY_HSTS` | `1` (on) | Emit the HSTS response header. Other security headers (nosniff, frame-deny, referrer) are always on. |

`/health` is **unguarded** by design so probes and load balancers can reach it
without being on the allow-list.

### Auth

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_AUTH_MODE` | empty (open) | `apikey` or `oidc`. Empty = no auth enforcement (loopback-only use). |
| `HIMMY_INTERNAL_API_KEY` | empty | Valid API key(s) for `apikey` mode, comma-separated for rotation. |
| `HIMMY_INTERNAL_HEADER` | `x-himmy-internal-key` | Override the header name the API key is read from. |
| `HIMMY_API_KEYS_FILE` | unset | Path to a keys file mapping keys to tenant-bound principals (multi-tenant). |
| `HIMMY_RBAC_FILE` | unset | Data-driven role → permission policy file. |
| `HIMMY_OIDC_ISSUER` | — | OIDC issuer (required for `oidc`). |
| `HIMMY_OIDC_AUDIENCE` | — | Expected token audience (required for `oidc`). |
| `HIMMY_OIDC_JWKS_URL` | derived from issuer | JWKS endpoint; fetched + cached. |
| `HIMMY_OIDC_ALGORITHMS` | `RS256` | Allowed signing algorithms (comma-separated). |
| `HIMMY_OIDC_SUBJECT_CLAIM` | `sub` | Claim used as the principal subject. |
| `HIMMY_OIDC_TENANT_CLAIM` | unset | Claim mapped to tenant. |
| `HIMMY_OIDC_ROLES_CLAIM` | `roles` | Claim carrying roles. |
| `HIMMY_OIDC_SCOPES_CLAIM` | `scope` | Claim carrying scopes. |
| `HIMMY_OIDC_ADMIN_ROLES` | empty | Roles treated as admin (comma-separated). |

### Rate limits

Off entirely unless `HIMMY_RATE_LIMIT` is set. The compose file injects explicit
defaults for window/burst because compose passes *empty strings* for unset vars,
and `float("")` would crash startup — so don't rely on the bare-process defaults
when running under compose.

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_RATE_LIMIT` | unset (off) | Sustained requests per window per caller. |
| `HIMMY_RATE_WINDOW` | `1` | Window length in seconds. |
| `HIMMY_RATE_BURST` | = `HIMMY_RATE_LIMIT` | Token-bucket burst capacity. |

The limiter is in-memory per process. It is per-caller (authenticated principal,
falling back to client IP) — adequate for the single-replica topology this
runbook targets; a multi-replica limiter would need a shared store (out of scope).

### Storage

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_DATABASE_URL` | unset | A `postgresql://…` DSN selects the durable Postgres entity store. Unset → file-backed SQLite. The compose stack always sets it. |
| `HIMMY_STORE_PATH` | `.himmy/storage.db` | Path of the file-backed SQLite store (cwd-relative). |

Note: the **server** path is durable; a one-shot `himmy run`/`himmy chat` (CLI)
*ignores* `HIMMY_DATABASE_URL` and always uses an ephemeral in-memory store, so a
developer's exported DSN never corrupts a quick run. Even with Postgres
configured, the `.himmy/*.db` sidecar stores (traces, run store, sessions) stay
SQLite and single-writer.

### Secrets (`HIMMY_SECRETS`)

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_SECRETS` | `env` | Secret backend: `env`, `file`, `vault`, `aws`, `gcp`, `azure` (plus `keychain` for macOS desktop use). All non-`env` backends chain an env fallback. |
| `HIMMY_SECRETS_DIR` | — | For `file` mode: directory of `<NAME>` files (compose mounts `/run/himmy-secrets` read-only). Also honors a `<NAME>_FILE` env pointer. |
| `HIMMY_VAULT_PATH` / `HIMMY_VAULT_MOUNT` | — / `secret` | Vault KV v2 path + mount (with `VAULT_ADDR`/`VAULT_TOKEN`). |
| `HIMMY_AZURE_VAULT_URL` | — | Azure Key Vault URL for the `azure` backend. |

The compose stack runs `HIMMY_SECRETS=file` against `/run/himmy-secrets`; the
Dockerfile also defaults it to `file`.

### Encryption-at-rest

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_KEK_PROVIDER` | `local` | KEK backend: `local` (raw key) or `aws-kms`. |
| `HIMMY_ENCRYPTION_KEY` | unset | Base64 32-byte KEK for `local`. Absent ⇒ field encryption is off. Deliver it as a file secret (`secrets/HIMMY_ENCRYPTION_KEY`), never an image layer. |
| `HIMMY_AWS_KMS_KEY_ID` | — | CMK id for `aws-kms` (needs the `kms-aws` extra). |

### Ollama (local inference)

| Variable | Default | Meaning |
|---|---|---|
| `HIMMY_OLLAMA_URL` | `http://localhost:11434` | Base URL of the Ollama server. Compose sets `http://ollama:11434`. Only reachable when the `ollama` profile is up. |
| `HIMMY_OLLAMA_TIMEOUT` | `120` (seconds) | Per-request inference timeout. |

## Reverse proxy / TLS

TLS **terminates at the proxy**; Studio speaks plain HTTP on 8765 behind it. The
single hard requirement is that the proxy forwards the **original `Host`** header
and that host is in `HIMMY_STUDIO_ALLOW_HOSTS` — otherwise the guard returns
**403** for `/api/studio` routes (the SPA loads, then API calls fail).

> **Never set `HIMMY_STUDIO_GUARD=0`.** Disabling the guard is not the fix for a
> 403 — adding your host to `HIMMY_STUDIO_ALLOW_HOSTS` is. The guard is what keeps
> a misconfigured proxy from exposing Studio to arbitrary `Host` headers.

Set `HIMMY_STUDIO_ALLOW_HOSTS=studio.example.com` (matching the public hostname),
then:

### nginx

```nginx
server {
    listen 443 ssl;
    server_name studio.example.com;

    ssl_certificate     /etc/ssl/certs/studio.crt;
    ssl_certificate_key /etc/ssl/private/studio.key;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;            # REQUIRED — must match HIMMY_STUDIO_ALLOW_HOSTS
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # WebSocket upgrade (Studio streams):
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Caddy

```caddy
studio.example.com {
    reverse_proxy 127.0.0.1:8765 {
        header_up Host {host}                  # REQUIRED — must match HIMMY_STUDIO_ALLOW_HOSTS
    }
}
```

Caddy provisions TLS automatically and forwards `Host` by default, but the
`header_up Host {host}` line is shown to make the contract explicit.

**403 symptom:** `/health` works, the SPA loads, but every `/api/studio/*` call
returns 403. Cause: the proxy rewrote `Host` (e.g. to `127.0.0.1:8765`) or the
real public host isn't in `HIMMY_STUDIO_ALLOW_HOSTS`. Fix the header / allow-list,
not the guard. A third variant: the guard also 403s ("cross-origin blocked") when
the browser sends a non-allow-listed `Origin`/`Referer` — same fix, the page's
host belongs in the allow-list.

## Auth hardening

The default (no `HIMMY_AUTH_MODE`) is **open** and only safe behind the loopback
guard on a trusted host. **The minimum for any non-loopback exposure is `apikey`
mode:**

```bash
HIMMY_AUTH_MODE=apikey
HIMMY_INTERNAL_API_KEY=<strong-random-key>      # comma-separate to rotate
```

`scripts/ops_provision.py` writes exactly this (apikey + a generated key) into
the compose `.env`, so a provisioned compose stack is non-open by default.

For an IdP-integrated deployment use `HIMMY_AUTH_MODE=oidc` with the
`HIMMY_OIDC_*` vars in the [configuration reference](#auth). mTLS is on the
roadmap (see HARDENING_PLAN.md) but is **not** a shipped `HIMMY_AUTH_MODE` today —
front Studio with an authenticated proxy if you need client-cert auth now. Full
threat-model and hardening guidance lives in
[HARDENING_PLAN.md](HARDENING_PLAN.md).

## Health monitoring

Three layers, all hitting the **unguarded** `/health`:

- **`GET /health`** — returns `{"status":"ok"}`. Liveness/readiness target.
- **Container `HEALTHCHECK`** — baked into the image (`curl -fsS .../health`),
  30s interval. Docker marks the container unhealthy on failure.
- **`scripts/ops_health.py`** — a deeper, dependency-free probe (HTTP `/health`,
  free disk, SQLite `quick_check`, Postgres connectivity + pending migrations,
  optional Ollama). Exit code is the **worst** result:

  | Exit | Meaning | Action |
  |---|---|---|
  | `0` | all `ok` / `skip` / `info` | healthy |
  | `1` | at least one `warn` (e.g. low disk, pending migrations, Ollama unreachable) | investigate soon; not yet down |
  | `2` | at least one `fail` (`/health` unreachable, SQLite corrupt, Postgres unreachable) | page / alert |

  ```bash
  python scripts/ops_health.py            # text report
  python scripts/ops_health.py --json     # machine-readable for a monitoring cron
  ```

  The Ollama check is **skipped** unless `HIMMY_OLLAMA_URL`/`--ollama-url` is set,
  so a non-Ollama deployment never WARNs forever (and a HEALTHCHECK that maps
  warn→unhealthy doesn't flap).

**Alert on:** exit `2` from `ops_health` (or a `down` container), `/health`
non-200, SQLite `quick_check` failure, Postgres unreachable. **Warn (ticket, not
page):** free disk below ~1 GiB, pending Postgres migrations after a deploy.

## Backup / restore & DR

> **NEVER `cp` (or `tar`/`rsync`) a live `.himmy/*.db` file.** A WAL-mode SQLite
> database has uncommitted pages in a `-wal` side file; a byte copy captures a
> torn, point-in-time-mismatched database. Use `scripts/ops_backup.py`, which
> snapshots each store through the SQLite online-backup API and **intentionally
> skips** the `-wal`/`-shm`/`-journal` side files. Restoring also deletes any
> stale side files next to a restored `.db` so SQLite opens it standalone.

### Back up

```bash
python scripts/ops_backup.py backup --out /backups      # `make ops-backup` writes to the repo root
```

Produces `himmy-backup-<utc>.tar.gz` containing a WAL-safe snapshot of every
`.himmy/*.db`, the non-db store files, the `secrets/` directory, and a
checksummed `manifest.json`. If `HIMMY_DATABASE_URL` is set and `pg_dump` is on
PATH, a `pg_dump -Fc` of Postgres is included automatically; if `pg_dump` is
absent the SQLite-only backup still succeeds and the manifest records that the
dump was skipped.

For Postgres specifically, you can also dump it directly:

```bash
pg_dump -Fc --dbname "$HIMMY_DATABASE_URL" --file himmy-pg.dump
```

### Restore

```bash
python scripts/ops_backup.py restore himmy-backup-<utc>.tar.gz
```

Restore is paranoid: it verifies every file's SHA-256 against the manifest
*before writing anything*, and **refuses** to overwrite a store directory that
still has live `-wal`/`-shm` side files (a running server) unless you pass
`--force` — which will corrupt a live database. **Stop the server first.** A
bundled Postgres dump is restored via `pg_restore --clean --if-exists` when
`HIMMY_DATABASE_URL` is set and `pg_restore` is on PATH.

### Full DR sequence

1. **Stop** the deployment (`compose down`, or scale the Helm release to 0) so no
   live WAL exists.
2. **Restore volumes** — run `ops_backup.py restore` against the `.himmy` state
   (and let it `pg_restore` the Postgres dump, or restore Postgres from your own
   `pg_dump`). For compose, restore into the `himmy_state` volume; for an
   air-gapped Ollama, restore models into the `himmy_ollama` volume.
3. **Restore secrets** — the `secrets/` subtree from the archive (Postgres
   password, `HIMMY_ENCRYPTION_KEY`). Encrypted fields are unreadable without the
   original KEK, so this step is mandatory.
4. **Start** the deployment.
5. **Verify** — `himmy doctor --storage` and `python scripts/ops_health.py`
   (expect exit 0).

## Kubernetes

`deploy/helm/himmy-studio` is a deliberately minimal chart for an existing
cluster. See its `values.yaml` and `templates/NOTES.txt` for the full surface.
Key constraints:

- **Single replica, hard-pinned.** `replicaCount: 1` with a `Recreate` strategy
  and an RWO PVC for `/app/.himmy`. The SQLite sidecar stores are single-writer —
  **raising the replica count corrupts state**, even with external Postgres.
- **External Postgres only.** No Postgres subchart. Set `database.url` inline or
  reference `database.existingSecret` (the latter wins). Leave both empty to fall
  back to the PVC SQLite store.
- **File secrets via `secrets.existingSecret`.** Each Secret key is mounted
  one-file-per-key under `/run/himmy-secrets` with `HIMMY_SECRETS=file`.
- **Allow-hosts auto-union.** Every enabled `ingress` host is automatically
  unioned into `HIMMY_STUDIO_ALLOW_HOSTS`, so an ingress can never 403 itself. Add
  extra proxy/CDN hostnames to `studio.allowHosts`. Probes hit the unguarded
  `/health` with a `Host: localhost` header defensively.

Out of scope (wire in yourself if needed): HPA, PDB, NetworkPolicy,
ServiceMonitor, bundled Postgres. There is no published registry image — build
from the Dockerfile and push to your own registry. Lint with `make helm-lint`.

## Gotchas

- **Single-writer SQLite — keep replicas at 1.** Even on Postgres, the
  `.himmy/*.db` sidecar stores (traces, run store, sessions) are SQLite and
  single-writer. The Helm chart pins `replicaCount: 1` for this reason; compose is
  single-container. Do not scale horizontally.
- **`.himmy` is cwd-relative.** The state dir resolves against the process working
  directory (`/app/.himmy` in the image). If you run `himmy studio` from a
  different cwd, you point at (or create) a *different* state dir — a common cause
  of "my data disappeared."
- **Named-volume uid-10001 ownership.** A freshly mounted named volume is created
  root-owned, which breaks every SQLite store for the non-root uid 10001. The
  Dockerfile **pre-creates and chowns `/app/.himmy`** so the volume inherits the
  `himmy` user; in Helm the pod `fsGroup: 10001` does the equivalent. Don't bind a
  host directory there without fixing ownership.
- **Dev stack vs full stack coexist.** `docker/docker-compose.yml` (the dev/test
  stack) publishes a bare Postgres on host **5433** and uses distinct volume names
  (`himmy_pgdata` vs `himmy_pgdata_full`). The full stack publishes **no** Postgres
  host port, so both can run concurrently without colliding.
- **`POSTGRES_PASSWORD` two-place mirroring.** The password must be identical in
  `deploy/compose/.env` (interpolated into the DSN) **and** in
  `deploy/compose/secrets/postgres_password` (read by the Postgres server). Diverge
  them and Studio can't authenticate. `scripts/ops_provision.py` writes both
  atomically; changing it later also requires resetting the pg volume.
- **`himmy_ollama` is an explicit named volume** for the air-gap contract — the
  offline installer restores models into a volume literally named `himmy_ollama`.
  Don't rename it. See [airgap.md](airgap.md).
- **Compose injects empty strings for unset vars.** That's why the rate-limit
  window/burst have explicit `:-` defaults in the compose file — `float("")` would
  crash startup. Mirror that pattern if you add numeric env passthroughs.

## Related docs

- [HARDENING_PLAN.md](HARDENING_PLAN.md) — security hardening roadmap and threat model
- [upgrades.md](upgrades.md) — schema migrations and version upgrades
- [airgap.md](airgap.md) — offline / air-gapped installs and the `himmy_ollama` contract
- [../architecture/config.md](../architecture/config.md) — configuration model
- [../architecture/local-first.md](../architecture/local-first.md) — the offline-first design that drives these constraints
