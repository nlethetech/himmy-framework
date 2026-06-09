# Consent & purpose limitation (WS4.6)

Himmy can enforce, per data subject, *what their data may be used for* — and **prove** it
afterward. The motivating scenario (DeepGyan): **a teacher who did not opt in must have their
data neither retained nor used for training.** Consent is a generic, `subject_id`-agnostic
capability; "teacher" is just a `context_subject_id`. A runnable, fully-asserted walkthrough
lives at `examples/deepgyan_consent/main.py` (tested by `tests/examples/test_deepgyan_consent.py`).

> **Offline-first.** This entire layer is **opt-in** and **off by default**. With
> `HIMMY_CONSENT` unset, `himmy run` / `create_app()` wires a bare `StorageService`, an
> ungoverned runtime (`consent_decider is None`), and no governance singletons — persistence
> is byte-for-byte what it was before WS4.6. It mirrors how RBAC/encryption stay inert until
> configured. *(verified: `tests/governance/test_consent_wiring.py::test_zero_config_is_bare_and_ungoverned` and the example's `zero_config_persists_verbatim`.)*

## Data model

| Concept | Values | Meaning |
|---|---|---|
| **`Purpose`** | `RETAIN`, `TRAIN`, `INFER` (+ `ANALYTICS`/`SHARE`/`IMPROVE`) | The axis a subject's data may be used along. `RETAIN` = write to storage/event-log/memory/spine; `TRAIN` = capture raw I/O / include in a dataset/replay/derived corpus; `INFER` = use in a live model call (no persistence implied). |
| **`ConsentState`** | `GRANTED`, `DENIED`, `WITHDRAWN`, `UNKNOWN` | The latest recorded state for one `(subject, purpose)`. `UNKNOWN` = no record on file. |
| **`Effect`** | `ALLOW`, `EPHEMERAL`, `DENY` | What a gate does. `ALLOW` = persist/use freely; `EPHEMERAL` = use for the live request only, never write; `DENY` = refuse. |
| **`Decision`** | `{subject_id, purpose, effect, reason, allowed}` | The PDP verdict. `allowed` is `True` only for `ALLOW`. |

Consent is recorded as immutable, content-addressed `EntityRecord`s (kind `consent`): one
append-only version chain per `(subject_id, purpose)` keyed by `consent_stable_id` (a UUID5
derivation — no raw `subject_id` survives in cleartext beyond a tombstone). History is
therefore tamper-evident and covered by the signed audit bundle for free.

### Default-decision table (governed mode only)

| Recorded state | `RETAIN` | `TRAIN` | `INFER` |
|---|---|---|---|
| `GRANTED` | ALLOW | ALLOW | ALLOW |
| `DENIED` / `WITHDRAWN` | DENY | DENY | DENY |
| `UNKNOWN` (no record) | **DENY** | **DENY** | **EPHEMERAL** |

Secure-by-default *where it counts*: no opt-in means no retention and no training, but a live
inference may still run ephemerally (used, never written). When **ungoverned**, `decide()`
returns `ALLOW` unconditionally and the gates are never even constructed. Override the
`UNKNOWN` defaults with `HIMMY_CONSENT_FILE` (a JSON `{purpose: effect}` map, mirrors
`HIMMY_RBAC_FILE`).

## The four enforcement points

All consult the one pure `ConsentPolicy.decide` (via `ConsentLedger.decision`, the runtime's
`consent_decider`). They are constructed **only** in the governed branch of
`ApiContainer._assemble`.

1. **`ConsentGatedStorage`** — a transparent decorator over the `StorageService` facade
   (covers Postgres too). Gates every subject-bearing `save_*` (`save_run`,
   `save_run_if_absent_by_idempotency`, `save_snapshot`, `save_memory`,
   `save_episodic_memory`, `save_recommendation`) at `RETAIN`. `ALLOW` ⇒ delegate;
   `EPHEMERAL`/`DENY` ⇒ skip the write and emit a `consent_denied_persist` security event. A
   subject-bearing record whose `subject_id` can't be resolved **fails closed**.
2. **`ConsentAwareRegistry`** — *the linchpin.* The runtime writes transcripts to the
   immutable `EntityRegistry` spine **in parallel** with storage, so gating storage alone
   would leave a cleartext copy on the spine. This wrapper gates the subject-bearing spine
   kinds (`run_event`, `message`, `chat_thread`, `context_snapshot`, `recommendation`) at
   `RETAIN`: non-`ALLOW` ⇒ the record never reaches the spine; `ALLOW` ⇒ register it with its
   subject-bearing fields **encrypted under the subject's shreddable key**. Infrastructure
   kinds (`persona`/`prompt`/`agent_state`/`env_state`) and the audit kinds
   (`consent`/`security_event`/`erasure_tombstone`) pass through untouched, so the audit trail
   is never starved.
3. **Runtime `TRAIN` gate** — a `consent_decider` param on `SingleAgentRuntime` (default
   `None` ⇒ unchanged offline). For a participating human subject lacking `TRAIN` consent it
   forces raw-I/O capture **off** and **strips the verbatim `rendered_prompt`** from the run
   events. The human subject is `ctx['context_subject_id']` — `persona.agent_id` is **not** a
   data subject.
4. **Export / replay `TRAIN` gate** — `ConsentFilteredExporter.filter(records)` is the single
   funnel every dataset/SFT/replay-corpus builder routes through; it drops every subject
   without an `ALLOW` for `TRAIN` (unresolved ⇒ fail closed). A subject who never opted in to
   training can never appear in an exported corpus — even if their run was retained (`RETAIN`
   and `TRAIN` are independent).

## Withdrawal ⇒ crypto-shred (right to erasure)

`ConsentLedger.withdraw(subject)` appends a `WITHDRAWN` version (for one purpose, or every
purpose on file) then — when a `RetentionService` is wired — calls `erase_subject`, which
**destroys the subject's per-subject key** (`SubjectKeyVault.destroy`) and writes a signed
**erasure tombstone**. Because gates 1 & 2 encrypt subject content under that key on write,
the on-spine ciphertext becomes permanently undecryptable while the append-only records and
the tombstone remain — so the signed audit bundle still verifies and you can prove the subject
existed and was erased. *(verified: the example's `withdraw_and_verify_erasure`,
`tests/governance/test_consent_wiring.py::test_governed_withdraw_crypto_shreds_consented_message`.)*

## Surfaces

| Surface | What | Gate |
|---|---|---|
| **CLI** `himmy consent {grant,deny,status,history,revoke}` | Record/inspect consent over a durable `.himmy/consent.db`; always governed; `revoke` withdraws + writes a tombstone. No server needed. | n/a (operator-local) |
| **HTTP** `POST /v1/consent/{grant,deny,withdraw}`, `GET /v1/consent/{decision,latest,history}` | The same surface over the BFF, tenant-scoped. | `consent:write` (grant/deny), `consent:read` (reads + withdraw self) |

### RBAC

`consent:read` / `consent:write` plus a self-scoped **`data_subject`** role (holds only
`consent:read`; the router additionally restricts it to its **own** `subject_id`, so a person
can read their decision/history and exercise withdrawal but cannot touch anyone else's data or
operational surfaces). `operator` and `auditor` get `consent:read`; `operator` and `admin` get
`consent:write`.

## Configuration

| Env | Effect |
|---|---|
| `HIMMY_CONSENT` (`on`/`1`/`true`) | Turn governance **on**. Unset ⇒ fully offline (the default). |
| `HIMMY_CONSENT_FILE` | JSON `{purpose: effect}` to override the `UNKNOWN` defaults. |
| `HIMMY_ENCRYPTION_KEY` | Optional; field encryption degrades to an in-process per-subject key when absent, so zero-config still works keylessly. |

## Known gaps

- **No first-class `subject_id`** on `ContextField`/`RunEvent`/`ChatThread`/`MemoryObject`:
  the runtime stamps `metadata['subject_id']` onto governed records and the gates **fail
  closed** when it is unresolved in governed mode. A subject-bearing sink with no resolvable
  subject is treated as a denial (skipped), never silently persisted.
- **KnowledgeBase ingestion** carries no `subject_id` (only the `client_id == subject_id`
  convention), so `erase_subject` cannot reach KB chunks today — gate at the ingest adapter.
- **Multi-subject artifacts** (one record naming several subjects) are not yet
  most-restrictive-wins; resolve a single subject per record for v1.
- **Replay cassettes** persist to disk outside `StorageService`; the `RecordingClientManager`
  TRAIN shim refuses to record a non-`TRAIN`-consented subject's I/O, but existing cassettes
  predating consent are out of `erase_subject`'s reach.
