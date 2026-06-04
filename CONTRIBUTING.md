# Contributing to OpenSims

Thanks for working on OpenSims. The project keeps an **offline-first** invariant:
everything must run end-to-end with no network and no API keys via the
deterministic `StubClientManager`. Optional extras layer in real providers,
Postgres/pgvector, and observability.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12+
pip install -e ".[api,postgres,knowledge,dev]"      # offline-capable extras + tooling
pre-commit install                                  # optional: run the gate on commit
```

## The quality gate

CI runs exactly these four checks; run them locally before opening a PR:

```bash
ruff check .              # lint (E/W/F/I/UP/B)
ruff format --check .     # formatting
mypy opensims             # static types
pytest -q                 # the offline suite must stay green
```

`ruff check . --fix` and `ruff format .` auto-resolve most lint/format findings.

## Conventions

- **Offline-first.** Any new capability must have a deterministic, network-free
  path that the test suite exercises against the stub. Real-provider, Postgres,
  and observability tests must `skip` cleanly when their deps/DB/keys are absent.
- **Typed boundaries.** Public surfaces should be typed; avoid widening `Any`.
  When a `# type: ignore[...]` is unavoidable, scope it to the specific code and
  add a one-line justification.
- **Lineage is the point.** Prefer modeling new artifacts as versioned
  `EntityRecord`s with typed links so they stay traceable.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.

## Tests

The suite uses `asyncio.run` inside synchronous tests, so `pytest-asyncio` is not
required. Run `pytest -q -rs` to see why each skip fired.
