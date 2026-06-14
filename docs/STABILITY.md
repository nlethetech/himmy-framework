# API Stability Contract

Himmy follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). This
document defines exactly what is covered by that promise, so a consumer can pin
`himmy` and know when an upgrade is safe.

The version is single-sourced from `himmy.__version__` (read dynamically by
`pyproject.toml`, the FastAPI app, the MCP client/server, and the compose/Helm
`appVersion`). A release is cut by tagging `vX.Y.Z` where `X.Y.Z == himmy.__version__`;
the release workflow asserts that equality before publishing.

## What IS public (covered by SemVer)

1. **The top-level Python API: `himmy.__all__`.** Everything re-exported at the package
   root — `Agent`, `Task`, `Persona`, `AgentSpec`, `load_agent_spec`, the orchestrator
   primitives, `build_runtime` / `build_inference` / `build_storage`, `MemoryService`,
   the skills + typed-agent helpers, and `__version__`. These names, their import
   location (`from himmy import X`), and their documented signatures are stable within a
   major version. The frozen set is enforced by `tests/test_public_api.py` — dropping or
   renaming any name reds the build.

2. **The served `/v1` REST API.** Every path under `/v1` in the OpenAPI document, plus
   the request/response schemas those paths reference. This is the contract for HTTP
   consumers (NEPSE, Samriddha, yetidai, …). It is enforced by
   `tests/api/test_openapi_snapshot.py`, which diffs a committed, `/v1`-scoped,
   version-stripped OpenAPI snapshot — a breaking route or schema change reds the build.

## What is NOT public (may change in any release)

- **`himmy.*` internals** — any module, class, or function NOT re-exported through
  `himmy.__all__` (e.g. `himmy.application.services`, `himmy.services.storage.*`,
  `himmy.api.*` internals). Import these at your own risk; they can move or change
  without a major bump.
- **The `/api/studio` surface.** Studio's BFF is an internal, co-versioned API for the
  bundled GUI, not a stability contract. It is deliberately excluded from the OpenAPI
  snapshot so Studio can evolve freely.
- **On-disk schemas and storage internals.** The SQLite/Postgres table layouts evolve
  through forward-only migrations (see `docs/enterprise/upgrades.md`); they are an
  implementation detail, not a public API. Upgrades are forward-only (no downgrade
  path).
- **CLI output formatting and log lines.** The `himmy` CLI commands and their flags are
  stable, but human-readable output text is not a contract — parse `--json` where
  offered.
- **Environment-variable *defaults* and experimental flags** explicitly marked as such
  in the docs.

## Versioning rules

- **PATCH** (`0.2.0 → 0.2.1`): bug fixes; no public API or `/v1` contract change.
- **MINOR** (`0.2.0 → 0.3.0`): backward-compatible additions — new `himmy.__all__`
  exports (added to the frozen set in the same PR), new `/v1` paths/optional fields, new
  CLI commands/flags.
- **MAJOR** (`0.x → 1.0`, `1.x → 2.0`): any breaking change to a public symbol or the
  `/v1` contract — a removed/renamed export, a removed `/v1` route, a removed or
  newly-required field, a tightened response schema.

### Deprecation window

A public symbol or `/v1` element slated for removal is **deprecated for at least one
minor release** before it is removed in a major release: it keeps working, emits a
`DeprecationWarning` (Python) or is annotated `deprecated: true` (OpenAPI), and the
CHANGELOG records the removal target. Consumers get at least one minor cycle to migrate.

## How the contract is enforced in CI

| Surface | Test | A breaking change… |
| --- | --- | --- |
| `himmy.__all__` | `tests/test_public_api.py` | reds the build (dropped/renamed symbol, or an export that no longer imports). |
| `/v1` OpenAPI | `tests/api/test_openapi_snapshot.py` | reds the build (changed `/v1` path or referenced schema). Studio-only changes do **not** red it. |

To accept an intentional `/v1` change, regenerate the snapshot and review the diff in
the PR:

```bash
UPDATE_OPENAPI_SNAPSHOT=1 .venv/bin/python -m pytest tests/api/test_openapi_snapshot.py -q
```
