# Himmy — Enterprise Hardening Plan

**Goal:** take Himmy from a strong offline-first agent framework (late-alpha) to an
**enterprise- and government-grade** platform that a company or public-sector body can
deploy on-prem, air-gapped, or in a regulated cloud — with the security, identity, audit,
and compliance posture that buy-off requires.

**Status:** planning. This document is the authoritative, code-grounded backlog. Every
item names the real integration point in the current codebase, a target state, the files
to add/change, acceptance criteria, and how it is verified. It is sequenced into phases so
value lands incrementally without a big-bang rewrite.

---

## 0. Guiding principles (non-negotiable invariants)

1. **Offline-first is preserved.** Every feature degrades to a working, keyless, in-memory
   default. Enterprise backends (OIDC, Vault, containers, Postgres) are *opt-in* via
   config — never required for `himmy run` to work.
2. **Additive & seam-based.** We extend existing seams (`Sandbox` Protocol,
   `set_rate_limiter`, `GuardrailPipeline`, `ClientManager`) rather than rewrite. New
   capabilities ship behind extras and config flags.
3. **Secure by default where it counts.** Auth, tenant isolation, and code-exec policy
   default to the *safe* setting in any non-default (i.e. configured) deployment.
4. **Every change is gated.** Local gate + CI-mirror (ruff + format + mypy + pytest) green
   before merge; security workstreams add their own CI gates (SBOM, scanners, integration).
5. **Auditable end-to-end.** Security-relevant actions become first-class, tamper-evident
   `EntityRecord`s. We already have the spine (`himmy/entities/integrity.py`); we wire it.

---

## 1. WS1 — Identity, Authentication & Access Control

The biggest enterprise gap. Today the BFF has an *optional* single shared internal key
(`himmy/api/app.py::_internal_key_dependency`, env `HIMMY_INTERNAL_API_KEY`), constant-time
compared, with key rotation. There is **no user identity, no roles, and no binding between
the caller and the tenant they may access.**

### 1.0 — [P0 SECURITY] Close the tenant-isolation hole (IDOR) — ✅ DONE
*Shipped: `himmy/api/auth/` (`Principal`, `Authenticator`, `ApiKeyAuthenticator`,
`resolve_workspace`/`require_workspace`); all 5 routers derive `workspace_id` from the
verified principal; mapped API keys bind a caller to tenants; offline default is an
ANONYMOUS all-tenants principal (unchanged). Tests: `tests/api/test_tenant_isolation.py`
(cross-tenant read/create → 403), `tests/api/test_auth.py`.*

- **Current:** `workspace_id` is a **client-supplied query parameter** threaded into
  `RunAppService.get_run(run_id, workspace_id=...)` (`himmy/api/routers/runs.py:115`,
  `himmy/application/services.py`). The 404-on-mismatch (AAEO-4) only triggers if the
  caller *chooses* to pass a non-matching id. With the shared key, **a caller can read any
  tenant's data by passing that tenant's `workspace_id`.**
- **Target:** the set of workspaces a request may touch is derived from the **authenticated
  principal**, server-side. A `workspace_id` that the principal is not entitled to → 403
  (not 404 leak), enforced in a single choke point, not per-router.
- **Files:** new `himmy/api/principal.py` (`Principal{subject, tenant_ids, roles, scopes}`);
  a FastAPI dependency `require_principal()` that replaces the bare guard and is the *only*
  source of `workspace_id`; update all 5 routers to take `workspace_id` from the principal,
  not the query string.
- **Acceptance:** a request authenticated for tenant A returns 403 for any tenant-B
  resource; no router reads `workspace_id` from the query; regression test proves the IDOR
  is closed.
- **Verify:** `tests/api/test_tenant_isolation.py` — cross-tenant read/write denied.

### 1.1 — Pluggable authentication (OIDC/JWT) — ✅ DONE
*Shipped: `himmy/api/auth/oidc.py::OidcAuthenticator` — verifies a Bearer JWT against the
provider's JWKS (RSA/EC signature, `iss`, `aud`, required `exp`), maps claims →
`Principal` (subject/tenant/roles/scopes, dotted claim paths for Keycloak-style
`realm_access.roles`, `all_tenants_roles` for platform admins). `JwksCache` (TTL + rotation
refresh on unknown `kid`) / `StaticJwks` (injectable, so the path is tested offline with a
self-signed key). `HIMMY_AUTH_MODE=oidc` + `HIMMY_OIDC_*` config; new `auth` extra
(`pyjwt[crypto]`, also added to `dev` so CI runs it). OpenAPI advertises the bearer scheme.
Tests: `tests/api/test_oidc.py` (17, incl. expired/wrong-aud/wrong-iss/bad-sig/missing-exp/
kid-rotation + live BFF RBAC + cross-tenant denial under OIDC). Works with Entra ID /
Keycloak / Okta / Auth0 / Google. **WS1 (identity & access control) is complete.**

### 1.1-orig — Pluggable authentication (`AuthN` provider)
- **Current:** static shared key only.
- **Target:** an `Authenticator` Protocol resolving a request → `Principal`. Ship three
  impls: (a) `ApiKeyAuthenticator` (today's behavior, now mapping a key → a principal with
  fixed tenant/roles), (b) `OidcJwtAuthenticator` (validate a Bearer JWT against a JWKS:
  issuer/audience/exp/signature; map claims → principal; supports Entra ID / Keycloak /
  Okta / Google), (c) `MutualTlsAuthenticator` (client-cert subject → principal, for
  service-to-service / gov mTLS).
- **Files:** `himmy/api/auth/` (`base.py`, `apikey.py`, `oidc.py`, `mtls.py`); config via
  `HIMMY_AUTH_MODE=apikey|oidc|mtls`, `HIMMY_OIDC_ISSUER`, `HIMMY_OIDC_AUDIENCE`,
  `HIMMY_OIDC_JWKS_URL`. New optional extra `auth = ["pyjwt[crypto]>=2.8"]`.
- **Acceptance:** valid OIDC token → principal with claims; expired/wrong-aud/bad-sig →
  401; offline default unchanged (no auth unless configured).
- **Verify:** `tests/api/test_oidc_auth.py` with a locally-signed JWT + a fixture JWKS.

### 1.2 — Authorization (RBAC + policy enforcement point) — ✅ DONE
*Shipped: `himmy/api/auth/rbac.py` (`AccessPolicy` over `resource:action` perms with `*`
wildcards; default `viewer`/`operator`/`auditor`/`admin` roles; `HIMMY_RBAC_FILE` for
custom policy; `require_permission(resource, action)` route dependency). Every BFF route
is guarded; deny-by-default for a role-less principal; bypassed when no auth is configured
(offline-first). Tests: `tests/api/test_rbac.py` (policy matrix + route enforcement).*

### 1.2-orig — Authorization (RBAC + policy enforcement point)
- **Target:** roles → permissions over (resource, action) pairs. A single
  `AccessPolicy.authorize(principal, resource, action)` checked by a `require_permission(...)`
  dependency on every mutating/reading route. Default roles: `viewer`, `operator`, `admin`,
  `auditor`. Permissions are data (a policy file), so customers can customize without code.
- **Files:** `himmy/api/auth/rbac.py` (roles/permissions + default policy),
  `himmy/api/auth/policy.py` (the PEP/PDP), `docs/enterprise/rbac_default.yaml`.
- **Acceptance:** a `viewer` cannot create or cancel a run (403); an `auditor` can read
  audit bundles but not mutate; admin-only routes reject operators.
- **Verify:** `tests/api/test_rbac.py` — a permission matrix test.

### 1.3 — Actor identity on runs ("who did what") — ✅ DONE
*Shipped: `RunAppService.create_run(..., actor=)` stamps the verified principal's
`actor_metadata()` (subject/auth_method/roles/source_ip) into the run's durable
`metadata["actor"]` (round-trips on in-memory + Postgres, no migration). The runs router
passes the principal; offline runs record the `anonymous` actor. Test:
`tests/api/test_actor_stamping.py`.*

### 1.4 — Security audit log (auth/authz/access) — ✅ DONE
*Shipped: `himmy/services/audit/` (`SecurityEvent`, `SecurityAuditLog`) records events as
tamper-evident `security_event` EntityRecords; `himmy/api/security_audit.py::audit_event`
emits from `principal_dependency` (auth_failure), `require_permission` (authz_denied), and
run-create (access) — no-op when auth is off. New `GET /v1/audit/events` router gated by
`audit:read` (auditor/admin), tenant-scoped. Tests: `tests/audit/test_security_log.py`
(incl. a signed-bundle tamper-evidence proof), `tests/api/test_security_audit.py`.*

### 1.3-orig — Actor identity on runs & entities ("who did what") - ✅ DONE
- **Current:** `RunRecord` (`himmy/services/storage/models.py:34`) and `EntityRecord`
  (`himmy/entities/records.py:70`) carry **no actor field** — only `created_at`.
- **Target:** stamp the authenticated `principal.subject` (+ auth method, source IP, request
  id) as `actor` on every run and on the entity metadata of records a request produces.
  Thread an optional `actor` through `RunAppService.create_run(...)` and the runtime's
  entity registration path (`SingleAgentRuntime` lineage writes).
- **Files:** add `actor_id`/`actor_meta` to `RunRecord`; a conventional
  `metadata["actor"]` on `EntityRecord` writes; migration for the Postgres `runs` table.
- **Acceptance:** every run/record created via the API records the caller; air-gapped CLI
  records a configured local actor; reading a run shows who launched it.
- **Verify:** `tests/api/test_actor_stamping.py`.

### 1.4 — Security audit log (access & admin events) - ✅ DONE
- **Target:** authn decisions, authz denials, data-access, and admin actions become
  first-class **`EntityRecord`s** (`kind="security_event"`) so they inherit immutability +
  lineage + the tamper-evident bundle (WS4.5). Shippable to a SIEM (WS6.4).
- **Files:** `himmy/services/audit/security_events.py`; emit from the auth dependencies and
  the run/recommendation services.
- **Acceptance:** a denied cross-tenant read produces a signed `security_event`; the event
  is included in `export_audit_bundle`.
- **Verify:** `tests/audit/test_security_events.py`.

---

## 2. WS2 — Code-Execution Isolation (the `code`/`run_python` sandbox) — ✅ DONE

*Shipped: **2.1** backend selection by policy — `HIMMY_CODE_EXEC` ∈ `off`/`subprocess`/
`container`, `build_sandbox()` + `DisabledSandbox` (refuses), default stays `subprocess`
(non-breaking) but served deployments set `container`/`off`; `run_python` always
approval-gated. **2.2** `ContainerSandbox` (`himmy/services/sandbox/container_sandbox.py`)
— hardened Docker/Podman isolate: `--network none`, `--read-only` + `/tmp` tmpfs,
`--cap-drop ALL`, `no-new-privileges`, non-root `--user`, pids/mem/cpu limits, hard
in-container `timeout` kill + outer watchdog. **2.3** gVisor/microVM seam (`runtime=`) +
`docs/enterprise/sandbox_backends.md` (threat model + per-tenant quotas). **Live-verified**
against real Docker: `tests/sandbox/test_container_sandbox.py` proves egress-denied,
read-only-rootfs, non-root, input-files, and wall-clock-kill (6 tests, skip when no
engine); `tests/sandbox/test_sandbox_factory.py` covers selection + off-mode refusal.*

### 2-orig — Code-Execution Isolation (the `code`/`run_python` sandbox)

Running model-authored code for a customer is the highest-severity surface.

- **Current:** the only `Sandbox` impl is `SubprocessSandbox` (`setrlimit` + wall-clock kill
  + env allow-list + fs jail). Network isolation is **advisory, not enforced**
  (`himmy/services/sandbox/models.py` documents `network` as unenforced for subprocess).
  The `Sandbox` Protocol (`himmy/services/sandbox/base.py`) is the seam for stronger isolates.

### 2.1 — Policy: code-exec off-by-default in served deployments
- **Target:** when the API is configured (non-default), `run_python` / `register_sandbox_tool`
  is **disabled unless explicitly enabled** (`HIMMY_CODE_EXEC=on`) and is always approval-
  gated. A clear, documented "we do not run untrusted code on the subprocess backend" stance.
- **Files:** gate in `himmy/toolkit/code.py` + the BFF container build.
- **Acceptance:** default served app refuses `run_python` with a clear error; CLI unchanged.

### 2.2 — Container sandbox backend
- **Target:** `ContainerSandbox` implementing `Sandbox`: rootless Podman/Docker, read-only
  rootfs, dropped capabilities, `no-new-privileges`, seccomp profile, **network namespace
  with egress denied by default**, per-run CPU/mem/pids cgroup limits, ephemeral tmpfs
  workdir, non-root UID. Files/stdin marshalled in; result marshalled out.
- **Files:** `himmy/services/sandbox/container_sandbox.py`, a hardened
  `docker/sandbox.Dockerfile`, a seccomp json; new extra `sandbox = []` (docs only) — the
  runtime shells out to the container engine.
- **Acceptance:** a snippet attempting network egress / fork-bomb / fs-escape is contained
  and reported as a failed `SandboxResult`, not an incident.
- **Verify:** `tests/sandbox/test_container_sandbox.py` (skipped when no engine), incl.
  egress-denied and capability-drop assertions.

### 2.3 — gVisor / microVM option + per-tenant quotas
- **Target:** document and seam in `runsc` (gVisor) and Firecracker/Kata as drop-in `Sandbox`
  backends for hostile-multi-tenant; per-tenant execution quotas (count/CPU-seconds) wired
  to WS3.2 rate limiting.
- **Files:** `docs/enterprise/sandbox_backends.md`; quota hook in the sandbox tool.

---

## 3. WS3 — Secrets, Network & Runtime Controls — ✅ DONE

*Shipped: **3.1** `himmy/config/secrets.py` — `SecretProvider` (env/file/Vault/AWS/GCP/Azure,
env-fallback chain), `get_secret()` routes every secret read (DSNs, SMTP/Telegram/search
keys, internal + gateway keys); default `EnvSecrets` = unchanged. **3.2**
`himmy/api/ratelimit.py::TokenBucketRateLimiter` behind the existing hook (per-principal/IP,
429 + Retry-After, `HIMMY_RATE_LIMIT`), off by default; principal now resolves before the
limiter. **3.3** egress allow-list in `guard_url(..., allow_hosts=)` + `egress_allow_hosts`
(`HIMMY_EGRESS_ALLOW`) threaded through web/comms tools. **3.4** security headers
(HSTS/nosniff/frame-deny/referrer) + opt-in strict CORS (`HIMMY_CORS_ORIGINS`, deny by
default) in `create_app`. Tests: `tests/config/test_secrets.py`, `tests/api/test_rate_limit.py`,
`tests/toolkit/test_egress.py`, `tests/api/test_security_headers.py`.*

### 3-orig — Secrets, Network & Runtime Controls

### 3.1 — Secret provider abstraction
- **Current:** every secret is a **direct `os.environ` read** with no abstraction —
  `HIMMY_SQL_DSN`, `HIMMY_KB_DSN`, `HIMMY_DATABASE_URL`, `HIMMY_SMTP_PASSWORD`,
  `HIMMY_TELEGRAM_BOT_TOKEN`, `HIMMY_SEARCH_API_KEY`, `PYDANTIC_AI_GATEWAY_API_KEY`,
  `HIMMY_INTERNAL_API_KEY` (`himmy/toolkit/config.py:85`, `himmy/api/deps.py:85,167`,
  `himmy/api/app.py:46`).
- **Target:** a `SecretProvider` Protocol with `get(name) -> str | None`. Impls: `EnvSecrets`
  (default, today's behavior), `FileSecrets` (Docker/K8s secret files, `*_FILE` convention),
  `VaultSecrets` (HashiCorp), `AwsSecretsManager`, `GcpSecretManager`, `AzureKeyVault`. All
  secret reads route through the provider. Secrets never logged (extend the existing
  redaction in `himmy/services/tools/security.py`).
- **Files:** `himmy/config/secrets.py`; refactor the reads above to `secrets.get(...)`; extra
  `secrets-aws`/`secrets-gcp`/`secrets-vault` per backend.
- **Acceptance:** with `HIMMY_SECRETS=file`, the DB password is read from a mounted file,
  not env; a fixture proves Vault lookup; default env path unchanged.
- **Verify:** `tests/config/test_secret_provider.py`.

### 3.2 — Rate limiting & quotas (real limiter behind the existing hook)
- **Current:** `set_rate_limiter` hook exists (`himmy/api/app.py:73`) but the default is a
  **no-op** — no throttling.
- **Target:** a `TokenBucketRateLimiter` (per-principal + per-tenant + global), pluggable
  store (in-memory default, Redis for multi-replica), `429` with `Retry-After`. Plus
  **inference/cost quotas** per tenant (cap tokens/$/day), surfaced from `InferenceService`.
- **Files:** `himmy/api/ratelimit.py`; optional `ratelimit-redis = ["redis>=5"]`.
- **Acceptance:** N+1 requests in a window → 429; per-tenant quota exhaustion → 429; limits
  configurable; default app still admits all (offline-first).
- **Verify:** `tests/api/test_rate_limit.py`.

### 3.3 — Egress allow-listing & network policy
- **Current:** SSRF guard (`himmy/toolkit/_net.py::guard_url`) blocks private/loopback hosts.
- **Target:** an explicit **egress allow-list** policy (per-tenant) for web/http/comms tools
  + MCP servers; deny-by-default mode for air-gapped installs. Document network policies for
  K8s (WS6.3).
- **Files:** extend `_net.py` with an allow-list; config `HIMMY_EGRESS_ALLOW`.

### 3.4 — Transport security & headers
- **Target:** TLS termination + optional mTLS guidance; security headers (HSTS, CSP for any
  served UI, `X-Content-Type-Options`), strict CORS policy (default deny), request body size
  limits, request-id propagation.
- **Files:** `himmy/api/security_headers.py` middleware; docs.

---

## 4. WS4 — Data Governance & Compliance — ✅ DONE

*Shipped: **4.1** `himmy/services/guardrails/dlp.py` — DLP as policy (allow/redact/tokenize/
block per class, reversible `TokenVault`, audited counts, optional Presidio); **4.2**
`himmy/services/governance/retention.py` — right-to-erasure via crypto-shred
(`SubjectKeyVault`) + immutable erasure tombstone, age-based `expired`; **4.3**
`himmy/config/residency.py` — region pinning (`enforce_region`); **4.4**
`himmy/services/storage/encryption.py` — envelope AES-GCM `FieldEncryptor`/`RecordCipher`
(AAD-bound, opt-in via `HIMMY_ENCRYPTION_KEY`); **4.5** Ed25519-signed audit bundles
(`himmy/entities/integrity.py`) + `GET /v1/audit/bundle` (auditor-gated). New extras
`dlp`/`encryption`. Tests across `tests/guardrails/test_dlp.py`,
`tests/storage/test_encryption.py`, `tests/config/test_residency.py`,
`tests/entities/test_audit_signing.py`, `tests/governance/test_retention.py`,
`tests/api/test_audit_bundle.py`. NOTE (4.4): ships the encryption capability + helpers;
transparent whole-store wiring is the documented opt-in deployment step.*

### 4-orig — Data Governance & Compliance

### 4.1 — PII as a compliance control (not just a filter)
- **Current:** `PIIGuardrail`/`NepalPIIGuardrail` are regex redactors at 3 surfaces
  (`himmy/services/guardrails/`). Good filter, not a governed control.
- **Target:** (a) a **classification → policy-action** model (`block` | `redact` | `tokenize`
  | `allow`) configurable per data class and per surface; (b) **audit every redaction** as a
  `security_event` (what class, where, count — never the value); (c) an optional **ML backend**
  (Microsoft Presidio) behind extra `dlp = ["presidio-analyzer","presidio-anonymizer"]` for
  recall beyond regex; (d) reversible **tokenization** with a vaulted map for workflows that
  must round-trip.
- **Files:** `himmy/services/guardrails/dlp.py`, `himmy/services/guardrails/presidio.py`.
- **Acceptance:** a configured policy blocks (not just redacts) a card number on output;
  redactions are counted in the audit log; Presidio backend detects a name regex misses.
- **Verify:** `tests/guardrails/test_dlp_policy.py`.

### 4.2 — Retention, deletion & right-to-erasure
- **Target:** retention policies per `kind`/tenant; a purge job; **crypto-shredding** for the
  right-to-erasure (delete the per-subject key, leaving immutable audit **tombstones** that
  prove a record existed and was erased — reconciles GDPR erasure with the append-only spine).
- **Files:** `himmy/services/governance/retention.py`; `himmy serve`/CLI `himmy gc` command;
  Postgres migration for per-subject encryption keys (ties to 4.4).
- **Acceptance:** an erasure request renders a subject's payloads unrecoverable while audit
  bundle verification still passes (tombstone accounted for).
- **Verify:** `tests/governance/test_erasure.py`.

### 4.3 — Data residency / region pinning
- **Target:** pin storage + inference routing to a region; refuse cross-region calls when
  `HIMMY_REGION` is set; document a per-region deployment topology.
- **Files:** region guard in `himmy/api/deps.py` storage/inference builders + routing.

### 4.4 — Encryption at rest (field-level for sensitive payloads)
- **Target:** envelope encryption of sensitive `EntityRecord.payload`/run outputs at the
  storage layer (per-tenant or per-subject data keys via the SecretProvider/KMS), plus DB
  TLS. Transparent to callers; the audit hash covers ciphertext deterministically.
- **Files:** `himmy/services/storage/encryption.py`; integrate in `PostgresStorageService`.
- **Acceptance:** a DB dump shows ciphertext for sensitive fields; decryption requires the KMS
  key; audit verification unaffected.
- **Verify:** `tests/storage/test_field_encryption.py`.

### 4.5 — Continuous, exportable audit
- **Current:** tamper-evident bundles exist (`export_audit_bundle`/`verify_audit_bundle`,
  HMAC-SHA256 + Merkle root, `himmy/entities/integrity.py`) but are invoked ad-hoc; the HMAC
  secret is a plain arg.
- **Target:** scheduled/append signing; signing key from the SecretProvider/HSM; a BFF route
  `GET /v1/audit/bundle` (auditor-role only) to export + verify; optional public-key
  signatures (Ed25519) so a verifier needs no shared secret.
- **Files:** `himmy/services/audit/exporter.py`, an `audit` router; extend integrity with an
  asymmetric option.
- **Acceptance:** an auditor exports a signed bundle and verifies it offline; tampering with a
  row is detected with the exact record id.
- **Verify:** extend `tests/entities/test_integrity.py`.

---

## 5. WS5 — Supply Chain & Secure SDLC (CI/CD)

Current CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest (stub-only). No security
scanning, no SBOM, no real-provider integration lane.

- **5.1 SBOM** — generate CycloneDX (`cyclonedx-py`) per build, attach as a release artifact.
- **5.2 Dependency scanning** — `pip-audit` gate in CI; Dependabot/Renovate; pin transitive
  deps with hashes for reproducible, audited builds.
- **5.3 SAST + secret scanning** — `bandit` + Ruff security rules (`S`), `semgrep` ruleset,
  `gitleaks`/`trufflehog` on every PR (catches a committed key before it ships).
- **5.4 Container scanning** — `trivy` on the BFF + sandbox images.
- **5.5 Provenance & signed releases** — pin GitHub Actions by SHA, `cosign`/SLSA provenance,
  signed tags.
- **5.6 [carries last session's lesson] Real-provider integration lane** — a separate CI job
  that runs the agent **against a real local model** (Ollama in a service container) so the
  stub can never again hide a broken core path; plus a security test suite (authz, IDOR,
  rate-limit, sandbox-escape). This job is required-to-merge.
- **Files:** new `.github/workflows/security.yml` + `integration.yml`; `Makefile`/`noxfile.py`
  targets so the same checks run locally.
- **Acceptance:** a PR introducing a known-vuln dep or a hardcoded secret fails CI; the
  integration lane catches a tool-loop regression.

---

## 6. WS6 — Compliance Posture & Operations (the buyer-facing wrapper)

- **6.1 Control mapping** — `docs/enterprise/compliance/` mapping Himmy features to **SOC 2
  CC**, **ISO 27001 Annex A**, and **NIST 800-53 / FedRAMP** controls (and which are
  product vs. operational responsibility). This is what unblocks procurement.
- **6.2 Threat model** — STRIDE + data-flow diagrams (`docs/enterprise/threat_model.md`);
  documented trust boundaries (the BFF, the sandbox, MCP servers, providers).
- **6.3 Deployment artifacts** — a **Helm chart** + **Terraform module** + hardened images,
  K8s `NetworkPolicy` (default-deny egress), secrets via CSI/External-Secrets, non-root
  pods, pod security standards, HPA. Air-gapped install bundle (vendored wheels + model).
- **6.4 Observability & SIEM** — Prometheus metrics + SLOs + alert rules; ship audit/security
  events to Splunk/Elastic/Sentinel; structured JSON logs with request/trace ids.
- **6.5 HA / DR** — multi-replica BFF (stateless once rate-limit/session state is in Redis),
  Postgres HA guidance, backup/restore runbook with RPO/RTO targets.
- **6.6 Process & docs** — `SECURITY.md` (vuln disclosure), incident-response runbook,
  access-review cadence, a `docs/enterprise/operations.md` runbook, versioned API + upgrade/
  migration guide.

---

## 7. Phasing & sequencing

Each phase is independently shippable and CI-gated. Dependencies flow downward.

| Phase | Theme | Items | Outcome |
|-------|-------|-------|---------|
| **P0 — Security stop-the-bleed** | close the known holes | WS1.0 (IDOR), WS2.1 (code-exec off-by-default), WS5.3 (secret scanning) | No known critical vuln in a configured deployment. |
| **P1 — Enterprise identity MVP** | who + what + limits | WS1.1 OIDC, WS1.2 RBAC, WS1.3 actor stamping, WS3.1 secret provider, WS3.2 rate limiting, WS5.1/5.2 SBOM+pip-audit | A technical team can run a **trusted-internal** pilot with real identity, RBAC, secrets, throttling. |
| **P2 — Isolation & data governance** | safe to run untrusted work / handle regulated data | WS2.2 container sandbox, WS4.1 DLP, WS4.2 erasure, WS4.4 encryption-at-rest, WS1.4 + WS4.5 audit, WS5.6 integration+security CI | Handles customer data + model-authored code defensibly. |
| **P3 — Compliance & operations** | procurement-ready | WS3.3/3.4 network/TLS, WS4.3 residency, WS6.1–6.6 (control mapping, threat model, Helm/Terraform, SIEM, HA/DR, process) | A company/government can run a **production** deployment; audit-ready. |
| **P4 — Continuous** | keep it certified | renovate, periodic pen-test, access reviews, SLSA provenance, cert renewals | Sustained posture. |

**Honest effort estimate:** P0 is days–weeks. P1 is the core enterprise lift (1–2 focused
months). P2 (isolation + data governance) is the deepest (2–3 months). P3 is as much
documentation/process/ops as code, and certification (SOC 2 Type II, FedRAMP) is a
multi-quarter organizational effort requiring an external auditor and, realistically, more
than one engineer. Total to credible "enterprise/government production": **a team, 2–4
quarters** — but the framework becomes *pilot-deployable* much earlier (end of P1).

---

## 8. Cross-cutting acceptance bar (applies to every item)

- Offline default path unchanged (a no-config `himmy run`/`create_app` still works keyless).
- New deps are isolated behind extras; CI mirror stays green without them.
- Unit tests for the happy path **and** the denial/abuse path (authz, limits, escape).
- A short `docs/enterprise/<feature>.md` for operators (config, defaults, threat notes).
- Security workstreams add a required CI gate so regressions can't merge.

---

*This plan is the backlog of record. Update item status inline as phases land; keep each
entry's "Current" grounded in the code so the plan never drifts from reality.*
