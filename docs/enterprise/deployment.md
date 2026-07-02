# Deployment runbook

> **Two different things can be deployed — pick the right one first.**
>
> * **Deploy MY agent (a service).** You have an `agent.yaml` and want it running
>   as a reachable, signed HTTP service. Use
>   [`deploy/compose/agent-compose.yml`](../../deploy/compose/agent-compose.yml) or the
>   [`deploy/helm/himmy-agent`](../../deploy/helm/himmy-agent) chart. See
>   [Deploy MY agent](#deploy-my-agent-a-service).
> * **Deploy Himmy Studio (the admin GUI).** You want the local web app for
>   building/inspecting agents. Use
>   [`deploy/compose/docker-compose.yml`](../../deploy/compose/docker-compose.yml) or the
>   [`deploy/helm/himmy-studio`](../../deploy/helm/himmy-studio) chart. That is the rest
>   of this runbook (below).
>
> These are NOT the same: the Studio artifacts stand up the GUI and expose **zero
> agent endpoints**. If you followed the Studio compose/helm expecting your agent
> to answer HTTP, you wanted the agent path above.

How to install, configure, expose, monitor, back up, and recover a himmy Studio
deployment. This is the operator's reference for the artifacts under `deploy/`
and `scripts/ops_*.py`. It documents what those artifacts actually do today —
including the sharp edges — rather than an idealized topology.

## Deploy MY agent (a service)

This path deploys the agent in your `agent.yaml` as a durable, signature-verified
service — distinct from Himmy Studio. The single-command front door is:

```bash
himmy deploy -f agent.yaml     # serve + worker together, bound 127.0.0.1, signed webhook
```

For a durable multi-process topology (an HTTP `api` process and a background
`worker` process over one store), use one of the two artifacts below.

### Agent — Docker Compose

[`deploy/compose/agent-compose.yml`](../../deploy/compose/agent-compose.yml) runs two
services over ONE durable named volume:

* **api** — `himmy serve -f /app/agent.yaml`: the FastAPI BFF exposing your agent as a
  **signature-verified**, **default-deny** webhook at `POST /v1/connectors/webhook`.
* **worker** — `himmy worker -f /app/agent.yaml`: the routine scheduler + durable
  run-queue dispatcher on the SAME `.himmy` store, so routines fire and queued runs drain
  even when no one is calling the endpoint.
* **postgres** (optional, `postgres` profile) — a durable entity store. Omit it and both
  services fall back to the offline single-writer SQLite default. Zero-config works with
  no Postgres.

```bash
# point AGENT_DIR at the FOLDER holding your agent.yaml (defaults to ./)
export AGENT_DIR=/abs/path/to/my-agent
docker compose -f deploy/compose/agent-compose.yml up -d
# ...with a durable Postgres entity store:
docker compose -f deploy/compose/agent-compose.yml --profile postgres up -d
```

Optionally drop a plain `KEY=value` file at `$AGENT_DIR/agent.env` with the provider keys
your tool packs need — it is read by both services and skipped when absent.

**Security posture (fail-closed).** The api binds `127.0.0.1` inside the container by
default: the webhook stays signature-verified + default-deny, and the port is deliberately
NOT reachable through a published `-p` mapping until you add real auth. To expose it, in
order: (1) set `HIMMY_AUTH_MODE=apikey` + provide a key (via `agent.env`), (2) set
`AGENT_BIND=0.0.0.0`, (3) uncomment the api `ports:` line. Binding `0.0.0.0` with no auth is
refused by `create_app` — himmy will not boot an open admin surface. The signing secret is
file-delivered (RO-mounted `./secrets`), never baked into an image.

### Agent — Helm

[`deploy/helm/himmy-agent`](../../deploy/helm/himmy-agent) is a distinct chart (NOT
`himmy-studio`) with an `api` Deployment and a `worker` Deployment that both mount your
`agent.yaml` (inline `agent.spec` rendered into a ConfigMap, or an existing ConfigMap via
`agent.existingConfigMap`) and share one RWO state PVC.

```bash
helm install my-agent deploy/helm/himmy-agent \
  --set image.repository=ghcr.io/nlethetech/himmy \
  --set-file agent.spec=./agent.yaml
```

Fail-closed by default: `api.bindHost` is `127.0.0.1`, so the endpoint is reachable only
in-pod. To expose it via the Service/ingress you MUST set `auth.mode` (e.g. `apikey`) AND
`api.bindHost=0.0.0.0` — the chart refuses to render an ingress otherwise, and himmy
refuses to boot an unauthenticated off-loopback bind. Each component is single-replica by
design (the shared `.himmy` SQLite run store is single-WRITER but opened WAL +
busy_timeout, so the api + worker coordinate safely as two processes on ONE node); a
Postgres `HIMMY_DATABASE_URL` moves the entity store off SQLite, but the run store
stays on the shared PVC. Because that PVC is `ReadWriteOnce` (one node at a time), the
worker pod is auto-scheduled onto the api pod's node via a hard `podAffinity` — leave
`affinity` empty to keep it, or supply your own only with an RWX volume. The api's
health/readiness use `exec` (in-pod `curl`) probes, not `httpGet`, so they stay correct
at the loopback `bindHost` default. Both containers pin `command: ["himmy"]` (the image
has no ENTRYPOINT).

<a id="agent-over-http"></a>
### Agent over HTTP — the signed webhook by hand

`himmy deploy` / `himmy serve` wire the inbound webhook for you and print a ready-to-paste
signed `curl`. This is the same wiring done by hand, so you can reproduce it in a container,
a systemd unit, or any process that constructs the FastAPI app with
[`create_app`](../../himmy/api/app.py) — the connector is mounted by
[`mount_inbound_connectors`](../../himmy/api/connector_inbound.py) at app startup.

Four config keys turn a bare BFF into an agent endpoint. All are read through the secrets
layer, so the process env is the zero-config path (a file/keychain backend also works):

| Key | Purpose |
| --- | --- |
| `HIMMY_INBOUND_AGENT_PATH` | the `agent.yaml` an inbound delivery runs (absent → nothing mounts) |
| `HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED` | enable the `webhook` connector for the `inbound` surface |
| `HIMMY_WEBHOOK_SIGNING_SECRET` | the shared HMAC secret every delivery must be signed with |
| `HIMMY_WEBHOOK_ALLOWED_SOURCES` | allow-list for the payload `source` field (default-deny; empty allow-list rejects all) |

Default-deny is preserved end to end: the connector refuses to mount without a signing
secret (an unsigned public trigger is a forgeable agent trigger), and every delivery is
HMAC-verified over the raw body before it reaches your agent.

```bash
# 1) point the inbound webhook at your agent + enable + allow the sample source
export HIMMY_INBOUND_AGENT_PATH="$PWD/agent.yaml"
export HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED=1
export HIMMY_WEBHOOK_ALLOWED_SOURCES=local

# 2) generate + persist a signing secret (NEVER print the raw secret; store it, don't echo)
export HIMMY_WEBHOOK_SIGNING_SECRET="whsec_$(python -c 'import secrets;print(secrets.token_hex(24))')"

# 3) serve it — the agent mounts at POST /v1/connectors/webhook (bound 127.0.0.1)
himmy serve -f agent.yaml
```

The endpoint is `POST /v1/connectors/webhook` (under the guarded `/v1` prefix). Each request
carries the HMAC of the *raw* body in the `X-Himmy-Signature` header, GitHub-style
`sha256=<hex>`. Compute a valid signature and call it — printing the signature, **never** the
secret:

```bash
BODY='{"source":"local","text":"hello"}'
SIG="sha256=$(printf '%s' "$BODY" | \
  openssl dgst -sha256 -hmac "$HIMMY_WEBHOOK_SIGNING_SECRET" | awk '{print $2}')"
curl -s http://127.0.0.1:8000/v1/connectors/webhook \
  -H "X-Himmy-Signature: $SIG" \
  -d "$BODY"
```

`himmy serve`/`himmy deploy` print exactly this `curl` (a valid signature over the sample
body, never the secret) in their live summary — so a newcomer can prove the endpoint end to
end in one paste. To expose it beyond loopback, add real auth first (`--share`, or
`HIMMY_AUTH_MODE=apikey` + a key) — an off-loopback bind with no auth is refused by
`create_app`.

---

The remainder of this runbook covers the **Himmy Studio** deployment (the admin GUI).

himmy is offline-first and single-process at its core: the durable state is a
directory of **single-writer** SQLite stores (`.himmy/*.db`) plus an *optional*
external Postgres for the main entity store. That single-writer reality drives
almost every constraint below (single replica, RWO volumes, WAL-safe backups).
Read the [Gotchas](#gotchas) before you scale anything.

## Overview — three Studio deployment shapes

These three shapes deploy **Himmy Studio (the GUI)**, not an agent endpoint — for
an agent service see [Deploy MY agent](#deploy-my-agent-a-service) above. Pick the
smallest shape that fits. They share the same image, the same `.himmy` state
contract, and the same `HIMMY_*` configuration surface.

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

### Metrics & structured logs

Beyond liveness, the served app exposes two operational signals (both honest
about their defaults):

- **`GET /metrics`** — Prometheus **text exposition** with request count
  (`http_requests_total`), latency histogram (`http_request_duration_seconds`),
  and in-flight gauge (`http_requests_in_flight`). Labels are bounded to method,
  the route *template* (never the filled-in path), and status class, so series
  can't explode. It is in-process and dependency-free (no extra needed) and
  exposes no secrets — point a Prometheus scrape at it. The endpoint is unguarded
  for scraping but reveals only these aggregate counters.
- **`HIMMY_LOG_FORMAT=json`** — off by default (human-readable logs unchanged).
  Set it to emit one JSON object per log line (`timestamp`, `level`, `logger`,
  `message`, plus `request_id` matching the `X-Request-ID` response header) for a
  log shipper. See [observability.md](../services/observability.md) for both.

## Backup / restore & DR

> The dedicated **[backup & disaster-recovery runbook](backup-recovery.md)** has
> the full procedures — `ops_backup.py` flag reference, Postgres WAL archiving /
> volume snapshots, step-by-step restore, KEK escrow, and RPO/RTO guidance. The
> summary below is the quick path.

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

Produces `himmy-backup-<utc>.tar.gz` (created `0600`) containing a WAL-safe
snapshot of every `.himmy/*.db`, the non-db store files, and a checksummed
`manifest.json`. If `HIMMY_DATABASE_URL` is set and `pg_dump` is on PATH, a
`pg_dump -Fc` of Postgres is included automatically; if `pg_dump` is absent the
SQLite-only backup still succeeds and the manifest records that the dump was
skipped.

> **The `secrets/` directory (encryption KEK + Postgres password) is EXCLUDED by
> default.** Bundling the KEK next to the data would let anyone who reads the
> backup decrypt every encrypted-at-rest field — backups routinely travel to
> lower-trust storage, so that one `tar.gz` would undo encryption-at-rest
> entirely. **Back up the KEK separately** (a secrets manager, an offline copy,
> or your existing secret-distribution channel) — without it the data archive is
> deliberately un-decryptable. Pass `--include-secrets` ONLY for an air-gapped,
> separately-encrypted archive whose access you fully control; the archive is then
> as sensitive as the KEK itself and must be stored encrypted with restricted
> access.

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
3. **Restore secrets** — from your SEPARATE secrets backup (Postgres password,
   `HIMMY_ENCRYPTION_KEY`), since the data archive excludes them by default.
   Encrypted fields are unreadable without the original KEK, so this step is
   mandatory. (If you used `--include-secrets`, restore also writes the `secrets/`
   subtree from the archive.)
4. **Start** the deployment.
5. **Verify** — `himmy doctor --storage` and `python scripts/ops_health.py`
   (expect exit 0).

## Retention / pruning

himmy **never deletes durable history automatically** — there is no background
GC. The durable SQLite stores grow monotonically with usage:

| Store | Default path | Grows per | Eligible for prune |
|---|---|---|---|
| Run events | `.himmy/storage.db` | several rows per agent turn (the audit stream) | events older than the cutoff, and/or all but the most recent N |
| Approval checkpoints | `.himmy/approvals.db` | one full thread+persona+ctx snapshot per HITL pause | only `approved`/`rejected` rows older than the cutoff (live `awaiting_approval`/`resolving` are always kept) |
| Sessions | `.himmy/sessions.db` | the whole thread re-upserted every REPL turn | sessions not touched since the cutoff, and/or all but the most recent N |
| Graph checkpoints | (caller-chosen) | one snapshot per interrupted graph | only `completed`/`failed` rows older than the cutoff (resumable `running`/`interrupted` always kept) |

On a busy server `.himmy/storage.db` is the dominant grower — every agent turn
appends multiple events, and `list_events`/timeline reads get linearly slower as
the stream lengthens — followed by `approvals.db` (each pause is a full snapshot).

Run **`scripts/ops_prune.py`** from cron / a scheduled job to bound this. The
recommended starting policy for a long-lived deployment is **`--older-than-days
90`** (90 days of audit/approval/session history retained, everything older
reclaimed); live and unresolved work is preserved at any age.

```bash
# 90-day retention across run store, approvals, and sessions (recommended baseline)
python scripts/ops_prune.py --older-than-days 90

# also cap the event stream and session count regardless of age
python scripts/ops_prune.py --older-than-days 90 \
    --keep-last-events 1000000 --keep-last-sessions 500

# include a graph checkpoint DB (no default path — pass it explicitly)
python scripts/ops_prune.py --older-than-days 30 --graph-path .himmy/graphs.db
```

It prunes in place through the same WAL-aware connection the server uses, so it is
safe to run while the server is up (writes serialize on the SQLite write lock).
Pruning deletes rows but does **not** shrink the file; follow a large first prune
with `VACUUM` (against a quiescent database) to actually return the space to the
filesystem. The `prune_events` / `prune_resolved` / `prune_terminal` / session
`prune` methods chunk their deletes, so a first-ever prune of a multi-hundred-
thousand-row store does not trip SQLite's bound-parameter limit.

## Kubernetes

`deploy/helm/himmy-studio` is a deliberately minimal chart for an existing
cluster. See its `values.yaml` and `templates/NOTES.txt` for the full surface.
Key constraints:

- **Single replica, hard-pinned.** `replicaCount: 1` with a `Recreate` strategy
  and an RWO PVC for `/app/.himmy`. The SQLite sidecar stores are single-writer —
  **raising the replica count corrupts state**, even with external Postgres.
- **State PVC is retained by default.** The chart-managed PVC carries
  `helm.sh/resource-policy: keep` (`persistence.retain: true`), so a routine
  `helm uninstall` or a failed `helm upgrade --install` will **not** delete the only
  copy of your chats/runs/traces. Delete it explicitly (`kubectl delete pvc …`) after
  a backup, or set `persistence.retain=false` to let Helm GC it with the release.
- **Default resource requests/limits.** The pod ships `requests cpu 250m / mem 512Mi`
  and `limits cpu 1 / mem 1Gi` (Burstable, not BestEffort) so it isn't first-evicted
  under node memory pressure and has an OOM ceiling. Override `resources` for heavier
  in-pod local models, or set it to `{}` to opt out.
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

- [backup-recovery.md](backup-recovery.md) — backup & disaster-recovery runbook (SQLite + Postgres backup/restore, KEK escrow, RPO/RTO)
- [HARDENING_PLAN.md](HARDENING_PLAN.md) — security hardening roadmap and threat model
- [upgrades.md](upgrades.md) — schema migrations and version upgrades
- [airgap.md](airgap.md) — offline / air-gapped installs and the `himmy_ollama` contract
- [../architecture/config.md](../architecture/config.md) — configuration model
- [../architecture/local-first.md](../architecture/local-first.md) — the offline-first design that drives these constraints
