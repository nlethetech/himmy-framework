# Air-gapped install bundle

himmy is offline-first at runtime, but a *fresh* install on an isolated network
still needs the bits carried across the gap: container images, a Python
wheelhouse, and the Ollama models. `scripts/airgap_bundle.py` assembles all of
it into one uncompressed outer tar that the bundled POSIX installer unpacks on a
machine with only `docker` + `docker compose` — no pip, no network, no `jq`.

## What's in the bundle

`himmy-airgap-<version>-linux-amd64.tar` (uncompressed outer tar) contains:

| Path | Contents |
|---|---|
| `images/*.tar.gz` | gzipped `docker save` of `himmy-studio`, `pgvector/pgvector:pg16`, `ollama/ollama:0.13.5` (pinned to match compose), and `busybox:1.36` (~2 MB — the installer needs it to untar the models volume offline). The three upstream images are pinned **by digest** (`tag@sha256:…`, tag kept for legibility) in both `scripts/airgap_bundle.py` and `deploy/compose/docker-compose.yml`, so a re-pull resolves byte-identical bits. |
| `wheels/` | a `manylinux2014_x86_64` / cpython-3.12 wheelhouse for `himmy[studio,knowledge,postgres,encryption,auth]` |
| `models/ollama-models.tar` | the Ollama models volume, tarred via busybox |
| `compose/` | a copy of `deploy/compose/` |
| `airgap_install.sh` | the installer |
| `manifest.json` | schema/version/platform, image refs + digests, model list, per-artifact sha256 |
| `SHA256SUMS` | `sha256sum -c`-compatible sums so the installer verifies with no JSON parser |

## Building the bundle (connected host)

```sh
# Plan + size estimate only — no docker, no downloads, no side effects (CI runs this):
python scripts/airgap_bundle.py build --dry-run

# Real build (linux/amd64 host strongly recommended — see "Limits"):
python scripts/airgap_bundle.py build --out dist/

# Trim size: bundle only the chat model:
python scripts/airgap_bundle.py build --out dist/ --models qwen2.5:3b-instruct
```

The default model set is `qwen2.5:3b-instruct,nomic-embed-text,qwen3-embedding`
(chat + two embedders). `qwen3-embedding` is multi-GB; drop it with `--models`
if your deployment doesn't use it.

> **Size warning — roughly 6–12 GB.** Three models dominate the total
> (~1–4 GB each). Images add ~2–4 GB gzipped and the wheelhouse ~0.2–0.6 GB.
> The `--dry-run` plan prints a coarse estimate before you commit the disk and
> the transfer.

## Reproducible installs — the dependency lock (`uv.lock`)

himmy is a **library**, so its `pyproject.toml` runtime requirements stay
floating (`>=`) — that is correct for a consumer that pins on its own side. For
the parts that need a *reproducible install* (the air-gap wheelhouse and CI), the
exact resolution is captured once in **`uv.lock`** (generated with [`uv`](https://docs.astral.sh/uv/)):

```sh
uv lock            # generate / refresh the lock from pyproject (run after any dep change)
uv lock --check    # CI gate: fail if pyproject drifted from the lock
```

The lock pins every transitive dependency to an exact version **with sha256
hashes**, across the whole extras universe. It never alters the floating ranges
in `pyproject.toml`.

**Where the lock is used:**

- **Air-gap wheelhouse.** When `uv.lock` is present, `scripts/airgap_bundle.py`
  exports the wheelhouse extras from the lock to a hash-pinned requirements file
  (`uv export --frozen …`) and downloads with `pip download --require-hashes`, so
  two runs produce a **bit-reproducible** `wheels/` set and a swapped artifact
  fails the download. Without the lock it falls back to resolving `.[extras]`
  live (older behaviour).
- **CI.** The Deploy workflow runs `uv lock --check` so a dependency change that
  forgets to refresh the lock fails the build. An air-gapped/CI install can sync
  straight from the lock with `uv sync --frozen --extra studio --extra knowledge
  --extra postgres --extra encryption --extra auth`.

> **Why `uv` (not pip-tools)?** `uv` is the resolver available in this
> environment and produces a single cross-extra, hash-pinned lock with a
> first-class `--check` consistency gate and a `pip download`-compatible
> `uv export`. If you must reproduce this without `uv`, the equivalent pip-tools
> command is `pip-compile --generate-hashes --all-extras -o requirements.lock
> pyproject.toml` — but `uv.lock` is the source of truth here.

## Installing (air-gapped host)

Copy the tar across, unpack it, and run the installer from inside:

```sh
tar -xf himmy-airgap-<version>-linux-amd64.tar
./airgap_install.sh
```

The installer verifies `SHA256SUMS`, `docker load`s each image, restores the
models into the `himmy_ollama` volume (the same volume name the compose stack
reads), and drops the compose files at `./deploy/compose`. Then:

```sh
cd deploy/compose
cp .env.example .env            # edit it
# place POSTGRES_PASSWORD etc. under deploy/compose/secrets/
# Start the NAMED services only — this deliberately skips the ollama-init
# one-shot (see "Air-gapped startup" below):
docker compose --profile ollama up -d ollama studio postgres
curl -fsS http://localhost:8765/health
```

### Upgrade in place (re-running the installer)

The installer is **idempotent**: re-running it (unpack a newer bundle in the same
directory and run `./airgap_install.sh`) refreshes the bundled compose files but
**preserves your provisioned `deploy/compose/.env` and `deploy/compose/secrets/`** —
it never overwrites an existing `.env`/`secrets/`, and it copies an existing
`deploy/compose` to `deploy/compose.bak-<utc>` first. This protects the Postgres
password and `HIMMY_ENCRYPTION_KEY`: losing the KEK would make every
encrypted-at-rest field permanently unreadable. Even so, **keep an independent copy
of the KEK** off-host (see the backup section in [deployment.md](deployment.md)) —
the data is unrecoverable without it.

### Air-gapped startup — skip the `ollama-init` puller

The compose stack ships an `ollama-init` one-shot whose only job is
`ollama pull <models>` on first start. On a normal (connected) host this is
idempotent and convenient. **On an air-gapped host it is harmful:** the models
are already restored into the `himmy_ollama` volume by the installer, but
`ollama pull` on an *already-present* model **still contacts the registry** to
check the manifest — with no network that fails.

The fix needs no compose edit: start the **named** services only and omit
`ollama-init`. Because nothing `depends_on` `ollama-init`, naming the services
explicitly leaves it un-started:

```sh
docker compose --profile ollama up -d ollama studio postgres
```

`studio`'s `depends_on: postgres` is still satisfied (postgres is named), and
`ollama` serves the pre-loaded models directly. Do **not** run a bare
`docker compose --profile ollama up -d` on an air-gapped host — that brings up
`ollama-init` and its pull will error out.

**Bare-metal** (no containers) installs use the wheelhouse with no network:

```sh
pip install --no-index --find-links wheels/ \
  "himmy[studio,knowledge,postgres,encryption,auth]"
```

## Model licensing notes

These are factual pointers, not legal advice — confirm the upstream terms for
your use before redistributing model weights inside your organization.

- **Ollama** (the runtime) is **MIT**-licensed.
- **`nomic-embed-text`** is released under **Apache-2.0**.
- **Qwen models** (`qwen2.5:3b-instruct`, `qwen3-embedding`) ship under their own
  Qwen license terms published by the model authors; review them — they are not
  a standard OSI license and may carry use restrictions. If your air-gap policy
  forbids carrying restricted weights, drop those models with `--models` and pull
  an approved alternative on the target.

The bundle carries weights as opaque blobs inside `models/ollama-models.tar`; it
does not alter or relicense them.

## Limits

- **linux/amd64 only.** The wheelhouse pins `manylinux2014_x86_64` / cpython-3.12
  and the images are pulled with `docker pull --platform linux/amd64` (so an
  arm64 build host still bundles amd64 images, and each image's platform is
  recorded in `manifest.json`); `--only-binary=:all:` makes a missing wheel
  fail loudly rather than smuggle an incompatible source build. Build on (or
  cross-build for) that target. arm64 air-gap is not supported by this script.
- **Image tags are pinned to compose.** The bundled `ollama/ollama` tag is
  `0.13.5` — byte-identical to the pin in `deploy/compose/docker-compose.yml`.
  A unit test (`tests/airgap/`) parses the compose file and fails the build if
  any `image:` tag it references is missing from the bundle, so a `docker load`
  of one tag can never be followed by a `compose up` that pulls a different one.
- **`pip download` of the local project + `--only-binary`.** Some pip versions
  refuse `pip download .[...] --platform … --only-binary=:all:` because the
  local project directory itself has no wheel to satisfy the binary-only
  constraint. The builder detects that specific failure and prints the
  workaround: build himmy's own wheel first, then download its dependencies as
  wheels —

  ```sh
  python3 -m build --wheel                       # produces dist/himmy-*.whl
  pip download --no-deps --dest wheels/ dist/himmy-*.whl
  pip download --platform manylinux2014_x86_64 --python-version 312 \
      --only-binary=:all: --dest wheels/ "himmy[studio,knowledge,postgres,encryption,auth]"
  ```

  The resulting `wheels/` directory is what the bundle ships and the installer
  consumes with `pip install --no-index --find-links wheels/`.
- **Build host needs docker + network + pip.** The `--dry-run` path needs none of
  these; the real build pulls/saves images, runs `pip download`, and pulls models.
- **The bundle is a snapshot.** Image tags are pinned (no `:latest`) and the
  resolved digest is recorded per image in `manifest.json`. Rebuild to pick up
  upstream updates; there is no in-place patch path across the gap.
- **No GPU assumptions.** Models run on whatever the target's Ollama supports;
  CPU-only inference of the larger embedder is slow.
- **`deploy/compose/` must exist at build time** (it is produced by the compose
  artifacts lane). A real build hard-errors without it; `--dry-run` only NOTEs it.

## Related docs

- [Deployment runbook](deployment.md) — the compose / Helm topologies the bundle
  installs, plus configuration and reverse-proxy guidance.
- [Upgrades](upgrades.md) — migrations are forward-only and auto-applied; across
  the gap you rebuild and reinstall rather than patch in place.
- [Sandbox backends](sandbox_backends.md) — the container runtime the air-gapped
  host already needs for code execution.
