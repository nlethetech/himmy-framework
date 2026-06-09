# Load / concurrency / profiling harness (`tests/load`)

A deterministic, **offline**, CI-friendly harness for measuring Himmy's behaviour
under concurrency, plus lightweight profiling of the hot paths. It needs **no API
keys and no network** — everything runs against the in-process FastAPI app
(httpx `AsyncClient` over an `ASGITransport`) and the in-memory `EntityRegistry`,
driven on the default offline stack (in-memory storage + stub inference).

## Running

```bash
# Fast subset only (the default `make test` already excludes slow + this dir):
make test-load                       # = pytest tests/load -v  (includes slow)

# Just the fast load/concurrency tests (no profiling, no Postgres):
python3 -m pytest tests/load -m "not slow" -q

# Just the heavy ones (profiling + the Postgres probe, which self-skips offline):
python3 -m pytest tests/load -m slow -q
```

Use the project interpreter
(`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`) if `python3` on
your PATH is a different version.

`make test` stays fast: it runs `pytest -m "not slow"`, so the profiling and
Postgres tests never run in the default gate.

## What is measured

| Test file | Marker | What it does |
| --- | --- | --- |
| `test_concurrent_runs.py` | (fast) | Fans out N = 10 / 50 / 100 concurrent requests against `POST /v1/runs`, the lineage read endpoint, and paginated list reads (plus a mixed workload). Asserts `error_rate == 0` and the expected counts; **records** p50/p95/p99 + throughput. |
| `test_profiling.py` | `slow` | `cProfile` around concurrent run creates and a deep lineage traversal over a ~500-node synthetic graph; prints the top-20 functions by cumulative time via `pstats`. |
| `test_pgvector_latency.py` | `slow` | Times lineage traversals against a real Postgres registry. **Skips cleanly** unless `asyncpg` is importable *and* `HIMMY_DATABASE_URL` is set. |

### Determinism

Assertions are **only** on deterministic quantities — request counts, error rates,
and synthetic-graph node/edge counts. Latency percentiles and throughput are
*recorded and printed*, never asserted, so the suite cannot flake on a slow or busy
CI box.

The synthetic data (`fixtures.py::SyntheticDataFactory`) is fully seeded: every
record id is derived from a fixed seed + an incrementing per-kind counter (never the
wall clock or unseeded randomness), so the same seed reproduces a byte-identical
graph across processes and machines.

### Helpers

* `fixtures.py` — `SyntheticDataFactory` builds `persona -> prompt -> snapshot` DAGs
  with typed `uses_persona` / `built_from` links and returns an in-memory
  `EntityRegistry` + the records created.
* `metrics.py` — `LoadTestMetrics` / `ProfileResult` pydantic models plus
  `compute_percentile`, `throughput`, and `render_profile_top_n` helpers.
* `conftest.py` — function-scoped `app`, `client_factory`, and `factory` fixtures so
  each test is isolated (safe under `pytest-xdist -n auto`).

## Offline note

Nothing here requires a key, a server, or a database. The Postgres latency probe is
the only test that *can* reach external infrastructure, and it self-skips with a
clear reason when the optional `asyncpg` driver or `HIMMY_DATABASE_URL` is absent —
so the whole harness is green on a fresh, disconnected checkout.
