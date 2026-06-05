# Tier 3 — Record-and-replay + automatic context compaction

Two frontier features that deepen the moat (the immutable lineage spine + trace), built
on substrate that already exists. Each tier is CI-gated and verified live on Ollama
`qwen2.5:3b-instruct`.

## A. Deterministic record-and-replay

**Goal:** re-run a failed agent run *exactly*, by replaying the recorded model responses —
no provider, no network — so debugging becomes "step through the exact trace."

**Enabler:** `compute_cache_key(request)` (`services/inference/cache.py`) is a deterministic
SHA-256 over a request's output-determining surface (messages, model_key, response_format,
tool names, generation params) that **excludes** the random `request_id`. So the same
logical call always hashes the same — the key to matching a recorded request to its
recorded response on re-run.

**Hook point:** the `ClientManager` protocol (`generate(request) -> response`). Recording
and replay are both *managers*, swapped in via `InferenceService(client_manager=…)` — the
runtime, loop, and tools are untouched.

- `RecordingClientManager(inner)` — delegates to a real manager, appends
  `(cache_key, response)` to an ordered cassette per call (failures included), returns the
  response unchanged. The recorded response already carries `tool_calls` + `tool_returns`,
  so tool *outputs* are captured.
- `ReplayClientManager(cassette)` — on `generate`, computes the request's cache key and
  returns the **next** recorded response for that key (FIFO, so duplicate/retry calls
  replay in order). A miss is a strict `ReplayError` (deterministic replay must be exact);
  an optional fallback manager enables partial replay.
- **Tools are not re-executed on replay** — the replay manager just returns the recorded
  response, so `bound_tools` handlers (side effects) never fire. Determinism preserved.

**Cassette:** a portable JSON artifact (`InferenceCassette`: meta + ordered entries), each
entry `{cache_key, model_key, response}` (+ an optional request snapshot for debugging).
Written by recording, loaded by replay.

**CLI (T3.2):** `himmy run --record FILE` (wrap the chosen provider, dump the cassette
after) and `himmy run --replay FILE` / `himmy replay FILE` (run against the cassette with
no provider). Replay uses a Noop cache so the cassette is the single source of truth.

**Tiers:** T3.1 cassette + managers (pure, stub-tested). T3.2 CLI + live: record a qwen
run, replay it with Ollama *off*, assert byte-identical final output.

## B. Automatic context compaction

**Goal:** long-horizon runs that don't blow the context — summarize old turns when history
grows past a budget, the way Claude Code compacts long sessions.

**Finding:** the runtime sends the **entire** `thread.messages` to every inference call
(`_build_request`, `single_agent.py:~1435`) with zero filtering. That single function is
the hook. `_estimate_tokens` (~4 chars/token) already exists for a pre-call size estimate;
`self.inference_service` is available mid-loop to produce a summary; threads persist via
`save_thread` (so a compacted thread is saved automatically).

- `ContextCompactor` (pure, T3.3): given the messages, a token budget, and `keep_recent`,
  decides **whether** to compact and **which span** to summarize. Invariants it must hold:
  - never compact the leading **system** message(s);
  - always keep the most recent `keep_recent` messages verbatim;
  - **never split a tool_call from its tool_return** — the summarizable span is trimmed to
    a tool-pairing-safe boundary, so the message list stays provider-valid;
  - no-op when under budget or when there's nothing safe to compact.
  Returns a `CompactionPlan` (the span to summarize + the kept head/tail) — fully testable
  without a model.
- Runtime apply (T3.4): when a plan fires, call `inference_service.run` with a
  summarization prompt over the span, replace the span with one synthetic summary message
  (`[Earlier conversation summary: …]`), mutate the thread, and emit a `CONTEXT_COMPACTED`
  event (what was summarized → audit trail on the lineage spine). Config via `AgentSpec`
  (`compact_context`, `compact_after_tokens`, `compact_keep_recent`) plumbed through
  `make_task` into `task.context`, mirroring the memory/skill specs.

**Tiers:** T3.3 compactor + invariants (unit-tested, incl. the tool-pairing guard). T3.4
runtime hook + config + event + live: a long multi-turn qwen run that crosses the budget,
compacts, and still answers correctly from the summarized context.

## Cross-cutting
- Additive and opt-in; default behavior unchanged (no `--record/--replay`, no
  `compact_context` → identical to today). Offline-first preserved.
- New `CONTEXT_COMPACTED` (and recording is event-free — the cassette is the artifact).
- Each tier: local gate + CI-mirror, committed as `nlethetech`, verified on qwen 3b.
