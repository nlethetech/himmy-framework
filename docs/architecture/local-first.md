# Local-first architecture

> Offline-by-default: the whole framework runs with three dependencies, no keys, and no network — every enterprise control degrades to a safe local default.

## Overview

Himmy is an **offline-first** agent framework. The default install pulls only three
runtime dependencies (`pydantic`, `pyyaml`, `httpx`) and runs the entire stack —
agents, tools, skills, memory, RAG, orchestration, multi-agent teams — against a
deterministic stub with no keys and no network. Everything heavier (real model
providers, Postgres, embeddings, auth, DLP, encryption, managed secrets) is an
**opt-in extra**, never a requirement.

The guiding invariant (from the [hardening plan](../enterprise/HARDENING_PLAN.md)):
*every feature degrades to a working, keyless, in-memory default; enterprise backends
are opt-in via config, never required for `himmy run` to work.* The corollary is
"secure by default where it counts" — auth, tenant isolation, and code-exec policy
default to the safe setting in any *configured* (non-default) deployment.

## Module map

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Core deps (3) + every optional extra; package data; lint/type/test config. |
| `.env.example` | Every env var is optional; documents the keyed/extra paths. |
| `Makefile` | Local mirror of the CI gate (lint/format/types/test) + security (SBOM, pip-audit). |
| `Dockerfile` | Two-stage build of the Studio image (non-root, healthcheck). |
| `docker/docker-compose.yml` | Local Postgres + pgvector for the `postgres`/`knowledge` extras. |
| `README.md` | The offline-default vs. opt-in capability table (canonical). |

## Key abstractions (the offline defaults)

- **Deterministic stub inference** — `StubClientManager` is the default
  `ClientManager`: canned, deterministic, $0 output. Real models are opt-in (Ollama /
  Claude CLI locally; `pydantic-ai` cloud/gateway via the `providers` extra). See
  [provider selection](cli-and-studio.md#provider-selection-providerpy).
- **`DeterministicEmbedder`** — the default embedder (sha256 hashing-trick, dim 64):
  reproducible lexical-overlap vectors with no model. Real semantic embedders are
  opt-in (`fastembed` / Ollama / OpenAI).
- **In-memory volatile storage** — the default entity registry + storage + memory
  store are in-memory (lost on restart). Durable SQLite/Postgres backends are passed
  explicitly.
- **Keyless web search** — the `web` pack defaults to keyless DuckDuckGo;
  Tavily/Brave need a key.
- **No auth by default** — the BFF runs as an ANONYMOUS, all-tenants principal; RBAC,
  rate-limiting, tenant isolation, and the security audit log only *enforce* once an
  authenticator is configured.
- **Subprocess sandbox** — code execution defaults to the portable `SubprocessSandbox`
  (resource/fault isolation). The hardened container backend and `off` are opt-in via
  `HIMMY_CODE_EXEC`. See [sandbox](../services/sandbox.md).

## How it works / data flow

The same code path serves both the offline default and a configured deployment; the
difference is purely which backend/secret/extra is wired:

```
himmy run -p "..."            # 3 deps, stub model, deterministic embedder, in-memory
        │                       registry, no auth, subprocess sandbox — works keyless
        ▼
(configure)  HIMMY_DATABASE_URL ─> Postgres storage + entity registry (durable)
             --provider ollama  ─> real local model
             HIMMY_EMBEDDER=...  ─> real embeddings
             HIMMY_AUTH_MODE=... ─> OIDC/API-key auth → RBAC/rate-limit/audit activate
             HIMMY_CODE_EXEC=container ─> hardened sandbox
             HIMMY_ENCRYPTION_KEY ─> field-level encryption at rest
```

No code change is required to move along that axis — extras and env vars select the
backend behind a stable seam (`ClientManager`, the `Sandbox` protocol, the
`SecretProvider`, the `Authenticator`, the entity registry).

## Configuration

### Optional extras (`pyproject.toml [project.optional-dependencies]`)

| Extra | Packages | Unlocks |
| --- | --- | --- |
| `api` | fastapi, uvicorn, starlette | The FastAPI BFF (`himmy serve`). |
| `studio` | fastapi, uvicorn, starlette | Himmy Studio, the local web GUI (`himmy studio`). |
| `providers` | pydantic-ai | Cloud/gateway model providers (`--provider pydantic-ai` / `openrouter`). |
| `postgres` | asyncpg | Postgres-backed storage / entity registry / durable runs. |
| `knowledge` | pgvector, openai, pypdf | Durable pgvector KB, OpenAI embeddings, PDF reading. |
| `embeddings` | fastembed | Self-contained local (ONNX) embedding model for real semantic recall. |
| `auth` | pyjwt[crypto] | OIDC/JWT bearer verification for the BFF. |
| `dlp` | presidio-analyzer, presidio-anonymizer | ML-based PII/DLP detection beyond regex. |
| `encryption` | cryptography | Field-level envelope (AES-GCM) encryption at rest. |
| `secrets-aws` | boto3 | AWS Secrets Manager backend. |
| `secrets-gcp` | google-cloud-secret-manager | GCP Secret Manager backend. |
| `secrets-azure` | azure-identity, azure-keyvault-secrets | Azure Key Vault backend. |
| `observability` | logfire | Logfire / OpenTelemetry instrumentation. |
| `connectors` | feedparser, openpyxl | Nepali-news RSS + NRB forex/macro (Excel) connectors. |
| `nepal` | nepali-datetime | Bikram Sambat calendar primitives. |
| `validation` | jsonschema | Full-fidelity JSON Schema validation of tool args/output (else a built-in offline subset). |
| `toolkit` | beautifulsoup4 | Better `web_fetch` readability (else stdlib HTML parsing). |
| `dev` | pytest, ruff, mypy, … | The dev/test toolchain (CI gate). |
| `all` | (aggregate) | A broad bundle of the above for convenience. |

> HashiCorp Vault needs no extra (it uses plain HTTP via `httpx`). The internal-API
> key, gateway routing, and embedding/search keys are all env/secret-driven, not
> extras.

### Key environment toggles

All env vars in `.env.example` are optional. Notable ones: `OPENROUTER_API_KEY` +
`OPENROUTER_MODEL` (providers), `PYDANTIC_AI_GATEWAY_API_KEY` + `HIMMY_GATEWAY_REGION`
(gateway), `HIMMY_INTERNAL_API_KEY` (BFF trusted-boundary header),
`HIMMY_DATABASE_URL` / `HIMMY_TEST_POSTGRES_DSN` (Postgres), `HIMMY_SECRETS`
(secret provider backend, e.g. `file`), `HIMMY_LOGFIRE_ENABLED` (observability).

### Tooling

- `make install` — editable install with the common extras.
- `make gate` — the CI quality gate locally: ruff check + `ruff format --check` +
  mypy + pytest.
- `make security` — `pip-audit` (known-CVE scan) + CycloneDX SBOM.
- `make integration` — real-provider integration tests (needs Ollama).
- `docker compose -f docker/docker-compose.yml up -d` — local Postgres + pgvector.
- `Dockerfile` — builds the Studio SPA (Node stage) into the Python package, installs
  `.[studio,knowledge]`, runs as non-root uid 10001 with a `/health` healthcheck.

## Enterprise hardening (WS1–WS5) and safe-default degradation

The [enterprise hardening plan](../enterprise/HARDENING_PLAN.md) organizes the
production-grade controls into workstreams, each degrading to a safe local default:

| WS | Theme | Control | Safe local default |
| --- | --- | --- | --- |
| **WS1** | Identity, authn & access control | OIDC/JWT or API-key `Authenticator`, RBAC (`require_permission`), actor stamping on runs, security audit log. | No auth → ANONYMOUS all-tenants principal; RBAC/audit are no-ops until auth is configured. |
| **WS2** | Code-execution isolation | `HIMMY_CODE_EXEC` ∈ off/subprocess/container; hardened `ContainerSandbox`; gVisor/Kata seam; `run_python` always approval-gated. | `subprocess` backend (resource/fault isolation, portable, keyless). |
| **WS3** | Secrets, network & runtime controls | `SecretProvider` (env/file/Vault/AWS/GCP/Azure); `TokenBucketRateLimiter`; egress allow-list; security headers + strict CORS; Studio loopback guard. | `EnvSecrets` (plain env reads); rate limiter off; CORS deny; headers safe-on. |
| **WS4** | Data governance & compliance | DLP policy (block/redact/tokenize) + Presidio; right-to-erasure via crypto-shred + tombstones; region pinning; field-level encryption; Ed25519-signed audit bundles. | Regex PII redaction; in-memory key vault; no encryption unless `HIMMY_ENCRYPTION_KEY`; HMAC bundles on demand. |
| **WS5** | Supply chain & secure SDLC | CycloneDX SBOM; `pip-audit` + Dependabot; Ruff flake8-bandit SAST + gitleaks/Trivy; pinned actions; real-provider integration CI lane. | The `make`/CI targets run; the offline build stays reproducible with the 3 core deps. |

(WS6 — compliance posture & operations — is the buyer-facing wrapper: control
mapping, threat model, Helm/Terraform, SIEM, HA/DR; see the plan.)

The pattern is uniform: the enterprise backend plugs into an existing seam behind an
extra + config flag, and the absence of that config is itself the safe, offline
default.

## Extension points

- Add an extra in `pyproject.toml` and gate its import behind a clear
  "needs the X extra" error (the pattern used across the codebase).
- Implement a seam (`ClientManager`, `Sandbox`, `SecretProvider`, `Authenticator`,
  the entity registry) to add a backend without touching callers.

## Gotchas & invariants

- The offline default path must always work keyless — never make an extra/key
  mandatory for `himmy run` / `create_app`.
- In-memory storage is **volatile** (lost on restart); durability is an explicit
  opt-in (SQLite/Postgres).
- The deterministic embedder reflects lexical overlap, not real semantics — fine for
  wiring/tests, swap in a real embedder for production recall.
- Enterprise controls only *enforce* once configured; an unconfigured deployment is
  intentionally permissive (anonymous, all-tenants) — secure-by-default applies to
  *configured* deployments.
- Optional third-party deps have `# type: ignore` guards that look "unused" with
  extras installed but are required for no-extras type-checking (hence
  `warn_unused_ignores` is off in mypy).

## Related docs

- [CLI & Studio](cli-and-studio.md)
- [Sandbox](../services/sandbox.md) · [Guardrails](../services/guardrails.md) · [Governance](../services/governance.md) · [Audit](../services/audit.md)
- [Enterprise hardening plan (WS1–WS6)](../enterprise/HARDENING_PLAN.md)
- [Sandbox backends](../enterprise/sandbox_backends.md)
