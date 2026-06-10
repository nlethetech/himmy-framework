#!/bin/sh
# Air-gapped installer for a himmy bundle (built by scripts/airgap_bundle.py).
#
# Runs on a machine with ONLY docker + docker compose available — no network,
# no python, no jq. Run it from inside the unpacked bundle directory (the one
# that contains manifest.json, SHA256SUMS, images/, compose/).
#
# Steps: verify checksums -> docker load every image -> create the himmy_ollama
# volume and untar the models into it -> drop the compose files in place ->
# print next steps (including the bare-metal pip wheelhouse path).
set -eu

# Volume name MUST match deploy/compose/docker-compose.yml so the studio stack
# reads the models we load here.
OLLAMA_VOLUME="himmy_ollama"
BUSYBOX_IMAGE="busybox:1.36"

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -f manifest.json ] || die "run this from inside the unpacked bundle (manifest.json not found)"
[ -f SHA256SUMS ] || die "SHA256SUMS not found — bundle is incomplete"

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"

# --- 1. verify checksums ------------------------------------------------------
log "==> verifying checksums (SHA256SUMS)"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c SHA256SUMS || die "checksum verification failed"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c SHA256SUMS || die "checksum verification failed"
else
    die "no sha256sum/shasum available to verify the bundle"
fi

# --- 2. docker load every image ----------------------------------------------
log "==> loading container images"
for img in images/*.tar.gz; do
    [ -e "$img" ] || die "no images found under images/"
    log "    loading $img"
    docker load -i "$img"
done

# --- 3. ollama models -> himmy_ollama volume ---------------------------------
if [ -f models/ollama-models.tar ]; then
    log "==> restoring Ollama models into volume '$OLLAMA_VOLUME'"
    docker volume create "$OLLAMA_VOLUME" >/dev/null
    # Stream the host tar into the volume via busybox. busybox:1.36 is ALWAYS
    # bundled (saved under images/ and loaded in step 2), so this needs no
    # network — guard anyway with a clear message if the image is somehow absent.
    docker image inspect "$BUSYBOX_IMAGE" >/dev/null 2>&1 \
        || die "$BUSYBOX_IMAGE not loaded (bundle incomplete) — cannot restore models offline"
    docker run --rm -i \
        -v "$OLLAMA_VOLUME:/data" \
        "$BUSYBOX_IMAGE" \
        tar -xf - -C /data < models/ollama-models.tar
    log "    models restored"
else
    log "==> no models bundled (models/ollama-models.tar absent) — skipping"
fi

# --- 4. drop compose files in place ------------------------------------------
log "==> installing compose files"
mkdir -p deploy
rm -rf deploy/compose
cp -R compose deploy/compose
log "    compose files at ./deploy/compose"

# --- 5. next steps -----------------------------------------------------------
cat <<'EOF'

==> done. Next steps:

  1. Provide secrets and config:
       cd deploy/compose
       cp .env.example .env            # then edit
       # put POSTGRES_PASSWORD etc. under deploy/compose/secrets/

  2. Start the stack using the models loaded above. Start the named services
     ONLY, so the `ollama-init` one-shot does NOT run — its `ollama pull` would
     hit the registry (pull is a no-op for an existing model but STILL contacts
     the registry, which fails air-gapped):
       docker compose --profile ollama up -d ollama studio postgres
     (ollama-init is intentionally omitted — the models are already in the
     volume, so nothing needs pulling.)

  3. Check health:
       curl -fsS http://localhost:8765/health

  Bare-metal (no containers) install of the Python package uses the bundled
  wheelhouse with NO network:
       pip install --no-index --find-links wheels/ "himmy[studio,knowledge,postgres,encryption,auth]"
EOF
