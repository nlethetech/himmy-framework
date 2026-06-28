# Himmy RBAC — Production-Hardening Report

**Posture grade:** B-: a genuinely strong single-box / trusted-team foundation (deny-by-default, clean tenant choke point, demoted shared key), with real, concentrated gaps before it's safe for untrusted multi-tenant traffic — authorization stops at the HTTP door and doesn't follow the request into the agent's actual work.

## Executive summary

Here is the plain-English picture of where Himmy's permission system (RBAC = the rules for "who is allowed to do what") stands today.

WHAT'S GOOD: The foundation is genuinely solid for a single-box or trusted-team setup. By default, if Himmy isn't sure someone is allowed to do something, it says NO ("deny-by-default"). Every customer's data is kept in its own walled-off space ("tenant"), and there is a single, well-guarded doorway that enforces that separation — so one customer cannot see another customer's data. The team also already fixed the most dangerous historical footgun (a shared master key that used to make everyone an admin) and the server now refuses to start in some unsafe configurations. The login/identity checks are careful and well above average for a young product.

WHAT'S RISKY once you let untrusted, paying customers share the same server: The permission rules only protect the "front door" (the web API). Once an agent starts running, it can use ALL of its tools with no further permission check tied to who asked for it — so a lower-privilege user could set up an agent that does things their own account should never be allowed to do. Worse, two side doors (incoming webhooks and scheduled/automated tasks) run agents with NO identity attached at all. There's also a powerful internal admin console ("Studio") protected by a single all-or-nothing switch — you either get the keys to everything across all customers, or nothing; there's no "read-only support person" option. And within one customer's space, any user who can read runs can read EVERY user's runs — there's no per-person ownership. Finally, a typo in a customer's hand-edited permissions file can silently widen access instead of failing safely, and there's no admin screen to manage any of this without restarting the server.

WHAT TO DO FIRST: (1) Make the permission-file parser fail SAFE on typos and add validation so a bad file can't quietly grant or revoke access. (2) Carry the user's identity all the way into the agent's actual work, and put a real identity on the webhook and scheduled-task paths. (3) Lock down the Studio admin console so it can't be accidentally exposed in a multi-customer setup. The deeper "enterprise-grade" features (role hierarchies, time-limited admin access, separation-of-duties) are honestly NOT needed yet — don't build them until a customer asks.

## Strengths (what is already production-quality)

- Offline-first, deny-by-default core: an unknown or role-less caller is denied, and the wildcard engine (_covers in rbac.py) handles resource:*, *:action and *:* exactly as intended — 34 passing tests confirm the matrix.
- Single tenant choke point: tenant/workspace isolation runs through one resolve_workspace/require_workspace path that is INDEPENDENT of the RBAC permission set, so even a permission bug cannot cross tenants. This is the load-bearing security boundary and it's clean.
- The historically dangerous shared-key-to-all-tenants-admin footgun is actively defused: demoted to a tenant-less operator plus a hard startup refusal of a shared-key-only multi-tenant deploy.
- Strong /v1 tenant API coverage: every endpoint carries an explicit read/write/audit guard, no method-level holes (POST/PUT/DELETE all write-gated), and SSE/streaming routes inherit guards via router-level dependencies rather than bypassing them.
- Disciplined authn: constant-time API-key comparison (hmac.compare_digest), X-Forwarded-For only trusted from configured proxies, and solid OIDC verification (signature against JWKS, iss/aud/required-exp, DoS-hardened JWKS refresh).
- Fail-closed config posture: a malformed RBAC file raises inside create_app so the server won't boot half-open, and the shipped Docker/Helm prod path binds 0.0.0.0 so the off-loopback no-auth startup guard actually fires.
- Honest architectural restraint: ABAC/ReBAC engines (Zanzibar/Cedar/OPA) are correctly NOT adopted — the workspace-partitioned model doesn't need them yet.

## P0-now

_Close the genuine holes that bite under untrusted multi-tenant traffic: authorization must follow the request into the agent's work, the unauthenticated side-doors need an identity, and policy authoring must fail safe._

### Thread the Principal/roles into tool execution — stop RBAC at the HTTP door no longer (confused-deputy fix)  ([large])

**Why:** Today the permission check only gates the route (POST /v1/runs needs run:write); once a run starts, the agent executes EVERY tool its spec binds with NO check against the caller's roles. A mid-privilege operator can stand up an agent whose tools reach data/side-effects their own role could never touch directly. This is the single largest gap between Himmy and production multi-tenant RBAC.

**What:** Add a Principal/roles parameter through create_run -> build_runtime_for_spec -> ToolService, and add a deny-by-default, audited authorization check in _execute_invocation that authorizes (tool-resource, action) against the run's principal before dispatch. Model tool capability as first-class policy (e.g. tool:<name>:invoke or a capability tag on ToolDefinition mapped to a permission), seeding from the existing read_only classification in services/tools/access.py. Propagate the same context (capability attenuation, never amplification) into spawned sub-agents and multi-agent orchestration.

**Files:** `himmy/services/tools/service.py`, `himmy/application/services.py`, `himmy/api/auth/rbac.py`, `himmy/runtime/from_spec.py`, `himmy/services/tools/registry.py`, `himmy/toolkit/spawn.py`, `himmy/orchestrators/multi_agent.py`

### Give the inbound webhook and scheduled-routine paths a real identity  ([large])

**Why:** The inbound webhook builds a runtime and runs the full agent loop directly, bypassing create_run, RBAC, actor-stamping, run quota AND resolve_workspace — runs/events land unscoped on the shared store. Scheduled routines fire across all workspaces with a literal actor={'source':'routine'}, not a verified Principal. These are the highest-trust execution paths running with no authorization subject. (Channel-level HMAC auth exists, but it authenticates the channel, not a subject, and gates nothing about which tools fire.)

**What:** Route inbound deliveries and scheduled ticks through RunAppService.create_run under a dedicated least-privilege SERVICE principal: explicit roles, a fixed workspace_id, all_tenants=False, and a tool allow-list independent of interactive agents. Stamp the connector/routine identity as the audit actor. For routines, persist the authorizing principal's roles (or a derived capability set) on the routine/run record at create time and re-evaluate at fire time so revocation between scheduling and firing takes effect.

**Files:** `himmy/api/connector_inbound.py`, `himmy/api/app.py`, `himmy/application/services.py`, `himmy/api/routines.py`, `himmy/api/routers/routines.py`

### Make permission-file parsing fail CLOSED and validate the policy at load  ([small])

**Why:** _parse_perm does `resource.strip() or '*'` so a typo or trailing colon ('run:', ':read', '') silently widens a constrained role toward a wildcard — fail-OPEN parsing in a system whose whole value prop is 'permissions are data an operator edits'. Separately, an empty {} silently locks everyone out and a bad shape leaks a raw TypeError as a 500. (Bounded by tenant isolation so it's intra-tenant, not cross-tenant — hence P0 for correctness, not catastrophe.)

**What:** Make _parse_perm strict: raise on empty/colon-less/empty-half tokens; require a wildcard to be the literal '*'. In load_policy/from_mapping, reject non-list perm values and non-string perms with a clear HimmyError naming the offending role, cross-check resources/actions against the catalogue the routers actually use (warn on unknowns to catch typos), and warn loudly on an empty policy or a non-admin role granted *:* / *:action. Ship a `himmy rbac validate <file>` CLI lint.

**Files:** `himmy/api/auth/rbac.py`, `himmy/cli/rbac_cmd.py`

### Lock down the Studio admin console so it can't be accidentally cross-tenant exposed  ([medium])

**Why:** ~138-150 Studio routes (filesystem download, MCP-server CRUD with live subprocess launch, provider API-key writes, security logs, privacy data, global run/thread readers) are gated by ONE coarse studio:use permission with no read/write split and no tenant scoping. Default is fail-safe (admin-only), but HIMMY_STUDIO_AUTH=off is a kill-switch that _enforce_multi_tenant_posture does NOT reject, and granting studio:use to any tenant-bound role via HIMMY_RBAC_FILE hands them every tenant's data.

**What:** As the immediate P0 step: posture-gate HIMMY_STUDIO_AUTH=off so it cannot be set under a multi-tenant deploy (reject in _enforce_multi_tenant_posture), and formally document Studio as a network-isolated single-tenant operator console. The granular read/write/manage permission split is the P1 follow-up below.

**Files:** `himmy/api/app.py`, `himmy/api/routers/studio_common.py`

## P1-near

_Important hardening: per-object isolation inside a tenant, the dead scopes field, granular Studio permissions, tighter tool-provisioning, credential lifecycle, and audit/observability._

### Add intra-tenant object-level (BOLA/IDOR) authorization  ([large])

**Why:** Object authz is workspace-only everywhere except consent.py. In a tenant shared by many of a customer's users (the normal B2B2C shape), anyone with run:read/knowledge:read can read EVERY user's runs, threads, lineage and knowledge — tenant isolation is present, intra-tenant per-user isolation is absent. This is a classic Broken Object Level Authorization gap.

**What:** Decide and DOCUMENT the isolation contract first (is a workspace single-user or multi-user?). If multi-user, add an ownership check (resource subject_id vs principal.subject, or a relationship/ABAC-lite layer) at the service chokepoints — get_run, load_owned_thread, knowledge readers — with a tenant_admin role allowed to cross subjects within its own tenant. Add tests asserting subject A cannot read subject B's run within the same workspace.

**Files:** `himmy/api/routers/runs.py`, `himmy/api/routers/threads.py`, `himmy/application/services.py`, `himmy/api/routers/knowledge.py`

### Resolve the dead Principal.scopes field (merged across 4 dimensions)  ([medium])

**Why:** scopes is populated by both OIDC and mapped keys but read in ZERO authorization decisions. An integrator will reasonably assume a token minted with scope=run:read is constrained — but it silently gets the full reach of its roles claim. It's either a least-privilege footgun or security-theater. (Tenant isolation is enforced independently, so this is not an IDOR boundary — hence P1, not P0.)

**What:** Pick one and document it: (a) make AccessPolicy.authorize intersect role-derived permissions with token scopes when scopes are non-empty (scopes can only NARROW, never widen) — preferred for OAuth2/machine-client deployments; OR (b) delete scopes from Principal and the OIDC/apikey parsing so the contract stops implying enforcement that doesn't exist. Add a test that a scoped token cannot exceed its scopes.

**Files:** `himmy/api/auth/principal.py`, `himmy/api/auth/rbac.py`, `himmy/api/auth/oidc.py`, `himmy/api/auth/apikey.py`

### Split Studio's coarse studio:use into granular read/write/manage permissions  ([large])

**Why:** Follow-up to the P0 Studio gate: a binary surface-wide permission means a read-only support engineer cannot be given Studio read without also getting MCP-server write, provider-key write and connection management. No studio:read vs studio:write/manage split exists.

**What:** Replace the single studio:use with per-surface/action permissions mirroring the /v1 pattern (studio.runs:read, studio.files:read, studio.mcp:manage, studio.connections:write, studio.models:write), default non-admin roles to read-only Studio, and thread resolve_workspace/principal tenant filtering through the global run/thread/file readers. Keep build_studio_router as the choke point but parameterize the required permission per route.

**Files:** `himmy/api/routers/studio_common.py`, `himmy/api/routers/studio.py`, `himmy/api/routers/studio_mcp.py`, `himmy/api/routers/studio_models.py`, `himmy/api/routers/studio_seclog.py`, `himmy/api/routers/studio_privacy.py`, `himmy/api/auth/rbac.py`

### Make tool/connector capability part of the policy instead of the binary all_tenants flag  ([large])

**Why:** sanitize_tenant_spec keys entirely on operator_provisioned = bool(principal.all_tenants) and only screens the RCE/SSRF fields (tools_module/http_tools/mcp_servers). Every other surface — built-in tool_packs (comms/telegram/files/google/spawn), connectors, knowledge, allow_spawn, allow_skill_dispatch — passes ungated for any tenant, and the capability is process-global, so once an operator enables sending/writing for any reason it's shared by every agent:write tenant. (Default side-effecting tools are HITL-approval-blocked, so this is a least-privilege granularity gap, not a default-config exploit.)

**What:** Tag each tool_pack/connector with a required permission; authorize a stored agent's declared tools against the provisioning principal's roles at write time AND the running principal's roles at execute time. Introduce explicit permissions (tool:comms:use, connector:slack:use) so an operator role can be granted exactly the side-effecting tools intended, instead of overloading all_tenants as the sole 'is privileged' signal.

**Files:** `himmy/config/spec_sanitizer.py`, `himmy/api/routers/runs.py`, `himmy/api/auth/rbac.py`, `himmy/runtime/from_spec.py`

### Hash API keys at rest and add lifecycle (expiry, revocation, rotation)  ([large])

**Why:** API keys live in plaintext in memory and in the keys file — a single read of HIMMY_API_KEYS_FILE exposes every tenant's LIVE credential, a leaked key can only be killed by editing the file and restarting (drains in-flight runs), and keys never expire. This is the biggest credential-lifecycle gap feeding RBAC for paying multi-tenant SaaS.

**What:** Store a salted hash (argon2 or hmac-sha256 with a server pepper) plus a non-secret key_id/prefix for O(1) lookup, compared constant-time. Add per-key metadata (key_id, tenant_ids, roles, created_at, expires_at, disabled) enforced at authenticate() time, a live-consulted revocation list/generation counter so a key dies without restart, and a mint/rotate/revoke CLI. Until then, document the keys file as live cleartext requiring 0600 + a secrets manager.

**Files:** `himmy/api/auth/apikey.py`, `himmy/api/auth/context.py`

### Add policy-lifecycle observability and a route-coverage CI gate  ([medium])

**Why:** Only the DENY branch is audited — successful grants, policy loads, and role/key assignments are never logged, and there are zero authz metrics, so a SOC2/incident-response auditor can't answer 'what policy loaded, who could access what, when'. Separately, a new /v1 or Studio route added without a require_permission guard ships silently with no test catching it (the existing AST guard only covers /v1 GET workspace-scoping).

**What:** Emit a startup policy_loaded event capturing source path + content hash (never perms verbatim) and role count. Add authz_denied (and sampled authz_granted for privileged resources) Prometheus counters labelled resource/action/outcome. Add a CI test that walks app.routes and asserts every non-public route carries a require_permission dependency, failing the build on a new unguarded route.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/app.py`, `himmy/cli/rbac_cmd.py`, `himmy/services/observability/metrics.py`, `tests/api/test_rbac.py`

### Make RBAC-off observable; harden the loopback no-auth startup gap  ([medium])

**Why:** If a deploy binds 127.0.0.1 (behind an in-pod proxy), allow-lists the proxy host, and forgets the auth env var, it reaches an RBAC-off all-tenants-admin /v1 surface with NO startup error — and RBAC being a no-op is invisible (no log, no readiness signal). The shipped prod path binds 0.0.0.0 so this needs a deliberate misconfiguration (hence low severity), but invisibility is the real problem.

**What:** Emit a loud startup WARN whenever authenticator is None, add an auth-posture field to /readyz and diagnostics, and require an explicit HIMMY_ALLOW_UNAUTHENTICATED=1 to start with no authenticator on any non-CLI server entrypoint. Consider defaulting is_multi_tenant() to fail-closed once a production marker (DATABASE_URL / non-memory backend) is present.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/auth/context.py`, `himmy/api/app.py`

### Make the approval gate authorization-aware (separation-of-duties on HITL)  ([medium])

**Why:** requires_approval is a one-size, principal-blind gate: an operator can approve its own gated tool call, and the CLI auto_approve list can silently DISARM a gate a multi-tenant deploy wants enforced. Combined with the missing tool-layer authz, it's the only standing tool control and it's both coarse and weakenable.

**What:** Require a distinct run:approve permission (separate from run:write), optionally enforce approver != initiator, audit who approved with their roles, and ensure the CLI auto-approve loosening cannot apply in an authenticated/multi-tenant deployment (it's a single-user-local trust decision).

**Files:** `himmy/services/tools/service.py`, `himmy/api/routers/runs.py`, `himmy/cli/permissions.py`

### Add studio:use (or a read-only Studio variant) to the default roles, or document the gap  ([small])

**Why:** DEFAULT_RBAC grants studio:use to nobody but admin, so a customer who turns auth on with built-in roles finds operators/auditors 403'd on the entire console with only an opaque error. This is an operability trap. (Note: for multi-tenant, granting full studio:use to viewer is the WRONG fix since Studio holds creds/approvals — pair this with the granular split above.)

**What:** Once the granular Studio permissions land, grant studio.*:read to viewer/operator/auditor by default. Until then, ship a sample policy file and have `himmy rbac validate` warn when a policy defines viewer/operator/auditor but omits Studio access. Keep mutating Studio perms admin-only.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/routers/studio_common.py`

### Gate or disable OpenAPI docs in a multi-tenant posture  ([small])

**Why:** /docs, /redoc and /openapi.json bypass the app-level auth dependency entirely (FastAPI auto-docs routes), so an unauthenticated caller can enumerate every route and the auth scheme. Low blast radius (no data), easy to close.

**What:** Set docs_url=None/redoc_url=None/openapi_url=None (or a dependency-protected custom docs route) when an authenticator is configured / in a multi-tenant posture. Keep /health and /readyz open for probes. Root cause to fix is the FastAPI auto-docs-route dependency bypass, not the rebinding guard.

**Files:** `himmy/api/app.py`

### Add small OIDC token-validation completeness fixes  ([small])

**Why:** Clock-skew leeway defaults to 0 (a few seconds of IdP/server drift spuriously rejects fresh tokens), and a kid-less token is verified against the first JWKS key during rotation instead of being rejected as ambiguous. Both are robustness/correctness, not exploits.

**What:** Default leeway to ~30-60s, overridable via a HIMMY_OIDC_LEEWAY env wired through from_env(). When kid is None and the JWKS has >1 key, raise AuthError('token missing kid'); only fall back to the sole key when exactly one is present. Add tests for small-skew tolerance and multi-key kid-less rejection. Do NOT require nbf (it's optional in RFC 7519 and many IdPs omit it — requiring it would cause spurious 401s).

**Files:** `himmy/api/auth/oidc.py`

## P2-later

_Maturity and enterprise features — several are honestly YAGNI today; build only when a concrete customer requirement appears. Do not gold-plate._

### Role hierarchy / inheritance (NIST Hierarchical RBAC Level 2)  ([medium])

**Why:** Operator/auditor hand-copy viewer's reads inline, inviting drift at SaaS scale. But the worst drift outcome is FAIL-CLOSED (a role loses a read), not unauthorized access — so this is maintainability hygiene, not a security hole.

**What:** Add optional `inherits: [role]` to the policy schema, expanded transitively with cycle detection at load into the existing flat frozensets so the hot-path _covers check is unchanged. Refactor DEFAULT_RBAC so operator/auditor inherit viewer's reads. Also replace the hardcoded {operator,admin,auditor} privileged-role literals in consent.py with a policy-driven/capability check so renamed custom roles behave correctly.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/routers/consent.py`

### Reusable ABAC-lite ownership helper (don't adopt OpenFGA/Cedar/OPA)  ([small])

**Why:** Per-object ownership rules are hand-rolled per router (consent's _enforce_self_scope), with no shared primitive, so a future 'owner edits, others read' feature will reinvent it. A full relationship/policy engine (Zanzibar/Cedar) is genuine over-engineering for the current workspace-partitioned model.

**What:** IF object-ownership rules proliferate, extract a small reusable, testable ownership/attribute-check helper (principal attribute vs resource attribute) invoked from a single dependency, rather than per-router literals. Revisit a real policy engine ONLY if customers demand cross-workspace sharing or fine-grained per-record ACLs.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/routers/consent.py`, `himmy/api/auth/context.py`

### Auth-failure rate-limiting / lockout (anti-automation)  ([medium])

**Why:** There's no per-IP/per-key failure throttle or lockout on credential verification, and no alert on auth_failure bursts. High-entropy hmac-compared keys make online brute force infeasible and the JWKS-refresh flood is already throttled, so this is defense-in-depth (ASVS V2.2), not a credential-compromise risk.

**What:** Add an auth-failure rate limit keyed on client_ip (already trusted-proxy-aware) and on key_id, with exponential backoff/temporary lockout after N failures in a window, run before/inside principal_dependency so failures are counted. Wire the existing audit_event to a metric/alert on sustained bursts.

**Files:** `himmy/api/auth/apikey.py`, `himmy/api/auth/context.py`, `himmy/api/app.py`

### Runtime RBAC administration API + hot-reload  ([large])

**Why:** Policy and key->role mappings are hand-edited JSON loaded once at startup; onboarding a tenant or rotating a leaked key needs a restart (a blast-radius event). Real, but partly subsumed by the API-key-lifecycle P1 work, and a full admin console risks reinventing the IdP.

**What:** Add an admin-gated (rbac:manage) surface to read the effective policy and CRUD key->tenant/role mappings, persisted and audited, plus a safe hot-reload (atomic swap of app.state.access_policy after re-validation). Keep off/admin-only by default so the offline single-box path is unchanged.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/auth/context.py`, `himmy/api/auth/apikey.py`, `himmy/api/app.py`

### Explicit-deny, Separation-of-Duties, JIT / time-bound roles (Constrained RBAC) — YAGNI  ([large])

**Why:** Mature platforms offer deny-overrides, mutually-exclusive roles, and time-limited break-glass elevation. Himmy has none today. Honestly YAGNI for the current use: build speculatively and you gold-plate. JWT exp already gives time-bounding for free, and 'role assignment lives in the IdP' is a defensible answer to admin-of-admins.

**What:** Do NOT build speculatively. IF a customer needs auditor independence or maker-checker: add an optional explicit-deny tier (deny beats allow), static SoD constraints (mutually-exclusive role sets validated at policy load/principal assembly), and time-bound grants evaluated at authorize-time. Meanwhile, document that roles union with no SoD, document the shared-key admin as the audited break-glass path, and recommend short-lived OIDC tokens.

**Files:** `himmy/api/auth/rbac.py`, `himmy/api/auth/principal.py`, `himmy/api/auth/apikey.py`, `himmy/api/auth/oidc.py`
