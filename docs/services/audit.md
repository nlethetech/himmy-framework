# Audit

> Security-relevant actions recorded as tamper-evident entities, exportable as a signed bundle for SIEM / auditors.

## Overview

The audit service captures who-did-what (auth decisions, authorization denials, data
access) as first-class `EntityRecord`s of kind `security_event`. Because they live in
the append-only entity registry, they inherit immutability, content-addressed
lineage, and the **signed audit bundle** integrity check — so the log can later be
proven un-altered, down to the exact tampered record id.

Note on file layout: the [hardening plan](../enterprise/HARDENING_PLAN.md) (WS1.4)
sketched `security_events.py` and `exporter.py`; the shipped code is
`himmy/services/audit/models.py` (the `SecurityEvent` model) + `log.py` (the
`SecurityAuditLog`), and the signing/export primitives live in
`himmy/entities/integrity.py`. The export route is `GET /v1/audit/bundle`.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/services/audit/models.py` | `SecurityEvent` Pydantic model. |
| `himmy/services/audit/log.py` | `SecurityAuditLog` — append + query events; `SECURITY_EVENT_KIND`. |
| `himmy/services/audit/__init__.py` | Public surface re-export. |
| `himmy/entities/integrity.py` | `content_hash`, `AuditBundle`, `export_audit_bundle*` / `verify_audit_bundle*` (HMAC + Ed25519). |
| `himmy/api/security_audit.py` | `audit_event(request, ...)` — request-aware emit helper (no-op when auth off). |
| `himmy/api/routers/audit.py` | `/v1/audit/events` (list) + `/v1/audit/bundle` (signed export), `audit:read`-gated. |

## Key abstractions

### `SecurityEvent` (`models.py`)

```python
class SecurityEvent(BaseModel):
    event_id: str          # default new_uuid
    event_type: str        # auth_failure | authz_denied | access | admin
    outcome: str = "deny"  # allow | deny
    created_at: str        # default utc_now_iso
    actor: dict[str, Any]  # compact principal descriptor (subject/auth_method/roles/ip)
    resource: str | None
    action: str | None
    workspace_id: str | None
    method: str | None     # HTTP method
    path: str | None       # request path
    detail: str = ""
```

The docstring also references `auth_success` and `access` as the kinds of action a
security event captures; the emitted `event_type` values in the codebase are
`auth_failure`, `authz_denied`, and `access` (run-create).

### `SecurityAuditLog` (`log.py`)

An append-only trail backed by the `EntityRegistry`:

- `record(event)` — registers the event as
  `EntityRecord.create(stable_id=event.event_id, version=1,
  kind="security_event", payload=event.model_dump(), metadata={workspace_id, actor,
  outcome})`. The append-only registry gives it immutability + the bundle's
  integrity check "for free".
- `recent(*, limit=100, workspace_id=None, event_type=None)` — reads back via
  `registry.list_by_kind(SECURITY_EVENT_KIND)`, validates each payload back into a
  `SecurityEvent`, filters by tenant / type, and returns newest-first.

Durability follows the registry backend: in-memory by default, Postgres when a
`PostgresEntityRegistry` is wired.

### Signed audit bundles (`himmy/entities/integrity.py`)

`record_id` is derived only from `(kind, stable_id, version)` — the payload is *not*
in the identity, so an in-place edit of a stored row would otherwise be undetectable.
The integrity layer closes that without changing record identity:

- `content_hash(record)` — SHA-256 over the record's full content
  (kind/stable_id/version/payload/metadata; `created_at` excluded so re-projections
  are stable). `link_hash(link)` does the same for lineage links.
- `AuditBundle` — `records`/`links` maps of id → content hash, a `merkle_root` over
  them (sorted leaves → order-independent), and a `signature`. `algorithm` is
  `HMAC-SHA256` or `Ed25519`.
- `export_audit_bundle(records, links, *, secret)` — HMAC-SHA256 over the Merkle root
  with a shared secret. `export_audit_bundle_ed25519(..., private_pem)` — asymmetric
  variant (auditors verify with the public key alone; stronger non-repudiation).
- `verify_audit_bundle*` — re-derives hashes from a (possibly tampered) live graph
  and reports `tampered`/`missing`/`added` record + link ids and whether the
  signature is intact. `ok` is True only when the signature is valid *and* nothing
  diverged.

This is the **tamper-evidence** guarantee: altering a `security_event` row after the
fact changes its content hash, which the bundle's Merkle root + signature detect.

## How it works / data flow

### Emitting events (`himmy/api/security_audit.py`)

`audit_event(request, *, event_type, outcome, resource=None, action=None,
workspace_id=None, detail="")` builds a `SecurityEvent` from the request + its
principal (`get_principal(request).actor_metadata()`) and records it into
`app.state.security_audit`. It is a **no-op when no authenticator is configured**, so
the offline / zero-config path records nothing and is unchanged. In a configured
deployment, events are emitted for auth failures, authz denials, and data access
(run-create).

```
request ──> principal_dependency (auth_failure on bad creds)
        ──> require_permission   (authz_denied on RBAC miss)
        ──> run create           (access)
                       │
                       ▼
            audit_event(...) ──> SecurityAuditLog.record(...) ──> EntityRecord(kind="security_event")
```

### Reading + exporting (`himmy/api/routers/audit.py`)

Both routes are read-only and gated by the `audit:read` permission
(`auditor`/`admin` roles), tenant-scoped via the principal:

- `GET /v1/audit/events` — recent events, newest first, optionally filtered by
  `workspace_id` / `event_type`.
- `GET /v1/audit/bundle` — exports a signed, tamper-evident bundle over every
  `security_event` entity. Uses **Ed25519** when `HIMMY_AUDIT_PRIVATE_KEY` (PEM) is
  set (an auditor then verifies offline with the public key alone), else **HMAC**
  with `HIMMY_AUDIT_SECRET`; returns `503` if neither is configured. This is the SIEM
  / external-auditor export path.

### API-layer security wiring (brief)

The audit service is one layer of the BFF's security stack (configured in
`himmy/api/app.py::create_app`):

- `himmy/api/auth/` — `Authenticator` resolves a request → `Principal` (API key
  today, OIDC/JWT in `oidc.py`); `rbac.py::require_permission(resource, action)`
  enforces role→permission policy per route (`HIMMY_RBAC_FILE`); both bypassed when
  no auth is configured (offline-first). These dependencies are where `audit_event`
  fires `auth_failure` / `authz_denied`.
- `himmy/api/ratelimit.py` — `TokenBucketRateLimiter` behind the `set_rate_limiter`
  hook, keyed per-principal (falling back to client IP), `429` + `Retry-After`. Off
  unless `HIMMY_RATE_LIMIT` is set.
- `himmy/api/security_audit.py` — the `audit_event` emit helper described above.

`create_app` also installs security response headers (HSTS/nosniff/frame-deny), a
loopback/same-site guard on the Studio API (`HIMMY_STUDIO_GUARD`), and request-id
propagation.

## Configuration

| Var | Effect |
| --- | --- |
| `HIMMY_AUDIT_PRIVATE_KEY` | Ed25519 private key (PEM) → asymmetric signed bundles. |
| `HIMMY_AUDIT_SECRET` | Shared secret → HMAC-SHA256 signed bundles (fallback). |
| `HIMMY_RBAC_FILE` | Custom role→permission policy (gates `audit:read`). |
| `HIMMY_RATE_LIMIT` / `HIMMY_RATE_WINDOW` / `HIMMY_RATE_BURST` | Token-bucket rate limiter. |

Audit emission only happens when an authenticator is configured
(`build_authenticator()` returns non-None).

## Extension points

- Add new `event_type` values + `audit_event(...)` calls at new choke points.
- Swap the registry backend (Postgres) for durable, multi-process audit trails.
- Use `export_audit_bundle_ed25519` for public-key-verifiable bundles shipped to a
  SIEM / external auditor.

## Gotchas & invariants

- Security events are append-only `EntityRecord`s — never mutate them; immutability
  is what makes the tamper-evidence meaningful.
- Emission is a no-op without a configured authenticator (offline default records
  nothing).
- `created_at` is deliberately excluded from `content_hash`, so a bundle is stable
  across re-projections of identical content.
- The bundle export route returns `503` until a signing key/secret is configured.
- Shipped filenames differ from the WS1.4 plan sketch (`log.py`/`models.py` vs
  `security_events.py`; `entities/integrity.py` vs an `audit/exporter.py`).

## Related docs

- [Governance](governance.md) — how erasure tombstones reconcile with this append-only spine.
- [Guardrails / DLP](guardrails.md) — DLP detection counts feed the audit log.
- [Enterprise hardening plan (WS1.4 / WS4.5)](../enterprise/HARDENING_PLAN.md)
