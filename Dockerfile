# syntax=docker/dockerfile:1
#
# Himmy runtime image — published to ghcr.io/nlethetech/himmy by
# .github/workflows/deploy.yml on tag/release. It ships himmy[api,studio,...] already
# installed, so it serves two audiences from ONE image:
#
#   1. as-is it runs `himmy studio` (the local web GUI, the default CMD below);
#   2. as a BASE image for a user's agent: `himmy init` / `himmy deploy --docker` emit a
#      3-line Dockerfile (`FROM ghcr.io/nlethetech/himmy:<version>` + COPY agent.yaml +
#      CMD ["himmy","deploy",...]) so `docker build` works from the user's agent folder
#      with no framework checkout. The CMD is overridden there to `himmy deploy`.
#
# Stage 1 builds the React SPA into the Python package (himmy/api/_studio_static);
# stage 2 installs the package and serves it. Studio is accessed via a mapped localhost
# port (Host=localhost passes the Studio loopback guard). To reach it over a domain set
# HIMMY_STUDIO_ALLOW_HOSTS=<your-host> (or run it behind an authenticated proxy).
#
#   docker build -t himmy-studio .
#   docker run --rm -p 8765:8765 himmy-studio   # → http://localhost:8765

# ---- build the Studio frontend ----
FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS web   # node:20-slim
WORKDIR /app
COPY studio/package.json studio/package-lock.json studio/
RUN cd studio && npm ci
COPY studio/ studio/
RUN cd studio && npm run build   # emits to /app/himmy/api/_studio_static

# ---- runtime ----
FROM python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9 AS runtime   # python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIMMY_SECRETS=file
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY himmy/ himmy/
# the SPA built in stage 1 (package data served by `himmy studio`)
COPY --from=web /app/himmy/api/_studio_static/ himmy/api/_studio_static/
# postgres+encryption+auth match the air-gap wheelhouse extras: the compose stack
# mandates HIMMY_DATABASE_URL (PostgresStorageService._require_asyncpg() raises
# without [postgres]), and .env.example documents HIMMY_ENCRYPTION_KEY (encryption)
# and HIMMY_AUTH_MODE=oidc (auth), so those features must actually be installed.
RUN pip install ".[studio,knowledge,postgres,encryption,auth]"

# run as a non-root user
RUN useradd --create-home --uid 10001 himmy && chown -R himmy /app
# `.himmy/` is cwd-relative (/app/.himmy). When a named volume is mounted there
# it is created root-owned, so every SQLite store fails on first boot for uid
# 10001. Pre-create + chown the mount point so the volume inherits himmy.
RUN mkdir -p /app/.himmy && chown himmy /app/.himmy
USER himmy

EXPOSE 8765
# Probe /readyz (G5): unlike /health (liveness, process-up), /readyz returns 503 when
# the pod cannot serve durable traffic (durable storage requested-but-unwired, Postgres
# unreachable, or migrations behind code). Caveat: a Docker HEALTHCHECK marks the
# CONTAINER unhealthy (it does not pull it from a load balancer the way a K8s readiness
# probe removes a Service endpoint) — under an orchestrator, prefer the Helm readiness
# probe for traffic removal. The longer start-period covers durable-backend wiring.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/readyz || exit 1

CMD ["himmy", "studio", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
