# Learning service

## Overview

The learning service is himmy's **self-learning loop (P1)**: it turns the recorded
tool-execution audit stream into *behavioural* change. The P0 layer records every
`TOOL_COMPLETED` / `TOOL_FAILED` run-event (denormalised onto indexed `event_type` /
`tool_name` / `workspace_id` columns); this service **mines** those events into a
per-tool **reputation** and feeds it back two ways:

1. a learned **hint** injected into the prompt/context (so the model is told a tool has
   been flaky lately), and
2. a **reorder** of the bound tools (a more-reliable tool sorts ahead of a flaky one).

It measures **operational reliability** — did a tool *execute* cleanly — not outcome
quality (whether the resulting answer was right). It is **opt-in** (`self_learning`
defaults `False`) and **100% best-effort**: any storage/data error is swallowed to the
neutral result and logged at `debug`. Learning must never break or noticeably slow a run.

## Module map

| File | Role |
|---|---|
| `service.py` | `LearningService` (mines reputation), `ToolReputation` (the result), `ToolReputationProvider` (sync hot-path snapshot) |
| `adapter.py` | `LearnedHintsContextAdapter` — async; registered on the `ContextService`, injects reliability notes when a context snapshot is built |
| `__init__.py` | Public exports |

## Key abstractions

### `ToolReputation` (`service.py`)
Frozen dataclass: `tool_name`, `completed`, `failed`, `score` (`[0,1]`), `has_min_samples`,
and a `total` property (`completed + failed`). `score = completed / (completed + failed)`
over the recent window — but only **trusted below neutral** when `has_min_samples` is True.

### `LearningService` (`service.py`)
Mines reputation from the `EventLog` the runtime already uses (no second backend opened).
`get_tool_reputation(tool_names) -> dict[str, ToolReputation]`. For each tool it reads up
to `window` of its most-recent `TOOL_COMPLETED` and `TOOL_FAILED` events (a bounded index
seek on the `event_type`/`tool_name` columns), **merges them newest-first and keeps only
the most-recent `window` across both types**, then counts. Constructor: `window`
(default `200`), `min_samples` (default `3`), `workspace_id` (tenancy scope).

### `ToolReputationProvider` (`service.py`)
A **sync snapshot** for `ToolService.bound_tools`, which is on the per-turn inference hot
path and cannot `await`. `refresh(tool_names)` (async, run **out-of-band** once when the
runtime is built) populates an in-memory snapshot; `score_for()` / `is_unreliable()` read
it with no I/O. `floor` (default `0.2`) is the score below which a sufficiently-sampled
tool is annotated as unreliable (annotate, not drop). Emits a `LEARNING_APPLIED` event
**only when the stable score-sort actually moves a tool** (so the reorder is auditable
without false positives).

### `LearnedHintsContextAdapter` (`adapter.py`)
Async context adapter registered on the `ContextService`; reads reputation directly when
building a snapshot and injects a learned reliability hint. Emits its own
`LEARNING_APPLIED` inside the run when a hint is injected.

## How it works / data flow

```
P0 (elsewhere): tool runs → TOOL_COMPLETED / TOOL_FAILED recorded
                (denormalised event_type / tool_name / workspace_id columns)
        │
        ▼
LearningService.get_tool_reputation(tools)   ← bounded, workspace-scoped index reads
        │  completed/(completed+failed) over the recent `window`; neutral until min_samples
        ├─────────────► LearnedHintsContextAdapter  → hint in the prompt (per-run, async)
        └─────────────► ToolReputationProvider.refresh(tools)  → sync snapshot
                                 │   (once at runtime build, off the hot path)
                                 ▼
                        ToolService.bound_tools reorder (sync, no await)
                                 │   stable sort by score; annotate < floor
                                 ▼
                        LEARNING_APPLIED event (only on a real move)
```

### Recency window (why combined, not per-type)
`_recent_counts` reads up to `window` of *each* outcome type, then merges and keeps only
the most-recent `window` **across both**. Windowing the combined stream is what makes the
score reflect *recent* behaviour — a long history of completions can no longer dilute a
recent burst of failures.

### Neutral prior (cold-start safety)
An unseen tool, or one with fewer than `min_samples` scored calls, reports `score = 1.0`
(`NEUTRAL_SCORE`) with `has_min_samples = False`. A stable sort on all-`1.0` is a no-op, so
a brand-new tool is **never** deprioritised, and a single early failure can't bury an
otherwise-fine tool.

## Configuration

- `self_learning` — feature flag; **defaults off** (zero behaviour change until enabled).
- `LearningService(window=200, min_samples=3, workspace_id=None)`.
- `ToolReputationProvider(floor=0.2, event_sink=None)`.

## Tenancy

When `LearningService` is built with a `workspace_id`, every reputation read is scoped to
that tenant via the denormalised `workspace_id` column on `run_events` (captured
pre-encryption, at parity across in-memory / SQLite / Postgres). So on a **shared** event
store a tool's reputation reflects **only the run's own workspace** — one tenant's failures
never pollute another's. The scope is threaded from the run's `workspace_id`
(`build_runtime_for_spec(subject=...)`). `workspace_id=None` (the one-shot CLI / offline
path, isolated in-memory store per process) reads the whole stream unscoped — byte-identical
to the pre-tenancy behaviour.

## Extension points

- New consumers read `LearningService.get_tool_reputation(...)` directly (async) or hold a
  `ToolReputationProvider` snapshot (sync).
- The reputation math is deterministic given the same events; swap `window` / `min_samples`
  / `floor` to tune sensitivity without touching the miner.

## Gotchas & invariants

- **Best-effort, always.** Every public path swallows exceptions to the neutral result and
  logs at `debug`. Learning never raises into a run.
- **Hot-path reads are sync + snapshot-backed.** `bound_tools` never `await`s; the snapshot
  is refreshed out-of-band. An un-refreshed snapshot reports every tool neutral → reorder is
  a no-op (the zero-behaviour-change default).
- **Reliability ≠ outcome.** This is operational reliability (did the tool execute), not
  whether the recommendation was correct — keep it distinct from evaluation/calibration.
- **Annotate, don't drop.** Below `floor`, a flaky tool is cautioned, not removed.
- **`LEARNING_APPLIED` fires only on a real move** — a stable sort that doesn't change order
  emits nothing.
- **±1 boundary skew.** The cross-type merge orders by microsecond timestamp (not storage
  `seq`); at an exact-microsecond tie straddling the window edge the count can be off by ±1
  — accepted, since it cannot move a score across the min-sample / floor gates.

## Related docs

- [`tools.md`](tools.md) — emits `TOOL_COMPLETED` / `TOOL_FAILED`; `bound_tools` consumes the reorder.
- [`context.md`](context.md) — hosts the `LearnedHintsContextAdapter`.
- [`observability.md`](observability.md) — `LEARNING_APPLIED` event.
- [`storage.md`](storage.md) — the `EventLog` + denormalised `workspace_id` column the reads scope on.
