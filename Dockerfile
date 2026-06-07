# syntax=docker/dockerfile:1
#
# Himmy Studio — a deployable image of the local web GUI.
#
# Stage 1 builds the React SPA into the Python package (himmy/api/_studio_static);
# stage 2 installs the package and serves it. Accessed via a mapped localhost port
# (Host=localhost passes the Studio loopback guard). To reach it over a domain set
# HIMMY_STUDIO_ALLOW_HOSTS=<your-host> (or run it behind an authenticated proxy).
#
#   docker build -t himmy-studio .
#   docker run --rm -p 8765:8765 himmy-studio   # → http://localhost:8765

# ---- build the Studio frontend ----
FROM node:20-slim AS web
WORKDIR /app
COPY studio/package.json studio/package-lock.json studio/
RUN cd studio && npm ci
COPY studio/ studio/
RUN cd studio && npm run build   # emits to /app/himmy/api/_studio_static

# ---- runtime ----
FROM python:3.12-slim AS runtime
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
RUN pip install ".[studio,knowledge]"

# run as a non-root user
RUN useradd --create-home --uid 10001 himmy && chown -R himmy /app
USER himmy

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/health || exit 1

CMD ["himmy", "studio", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
