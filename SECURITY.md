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
