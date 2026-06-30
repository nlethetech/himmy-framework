# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
suspected vulnerability.

- Use GitHub's **private vulnerability reporting** (Security → *Report a vulnerability*)
  on this repository, or
- email the maintainers (see the repository profile).

Include a description, affected version/commit, reproduction steps, and impact. We aim to
acknowledge within a few business days and to coordinate a fix and disclosure timeline
with you.

## Supported versions

Himmy is pre-1.0; security fixes target the `main` branch. Pin a commit for reproducible
builds and update regularly.

## What we run on every change

- **SAST** — Ruff's flake8-bandit (`S`) rules in CI (`exec`, `pickle`, weak hashes,
  `shell=True`, `os.system`, `verify=False` … stay active).
- **Dependency audit** — `pip-audit` against known CVEs (+ weekly schedule and Dependabot).
- **Secret scanning** — gitleaks over the full history.
- **SBOM** — a CycloneDX SBOM is generated per build.
- **Filesystem vuln/misconfig scan** — Trivy.
- **Type + test gates** — mypy and the full pytest suite, plus a real-provider
  integration lane.

## Hardening posture

The framework is offline-first and secure-by-default once configured: see
`docs/enterprise/HARDENING_PLAN.md` for the identity/RBAC, tenant isolation, code-exec
isolation, secrets, rate-limiting, DLP, encryption, and tamper-evident audit controls.

## Multi-tenancy maturity (beta)

The **default, supported posture is single-box / single-user** (offline-first, no
authenticator ⇒ anonymous all-tenants principal). In that posture there is no cross-tenant
surface to leak, and behavior is unchanged.

**Multi-tenant RBAC (multiple untrusted tenants on one deployment) is BETA — hardened but
not yet certified.** It has been through a substantial hardening program: authorization is
enforced *by construction* at central chokepoints for (1) tool/capability execution, (2)
HTTP request data-scoping, (3) tool-store tenancy (memory/knowledge), and (4) the Studio
singleton stores (tenant columns + migrations), each backed by a structural coverage test
that fails the build if a new path skips the gate. An extensive internal adversarial
red-team found and fixed a large number of cross-tenant/cross-user isolation issues.

However, the internal red-team did **not** reach a formal "two consecutive clean rounds"
bar, and one recurring class remains a watch-item: the **within-tenant subject (per-user)
axis can be dropped on newly-added side-door paths** (background missions, team/group
orchestration, approval-resume, inbound connectors, the notify ring) until a structural
subject-axis gate lands. Treat untrusted multi-tenant as **opt-in and gated**, keep
`HIMMY_MULTI_TENANT` deployments behind your own review, and commission an **independent
external audit before GA** for untrusted multi-tenant traffic.
