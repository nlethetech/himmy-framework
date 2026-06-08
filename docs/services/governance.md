# Governance

> Retention and right-to-erasure for an append-only audit spine — via crypto-shredding plus immutable erasure tombstones.

## Overview

The entity registry is an immutable, content-addressed spine: records are
append-only and a signed audit bundle commits to every one of them (see
[audit](audit.md)). That collides with the GDPR/CCPA "right to be forgotten" — you
cannot simply *delete* a person's rows, because the whole point of the audit trail is
that rows are never altered. The governance service resolves this **erasure paradox**
with two moves:

1. **Crypto-shredding** — each subject's sensitive data is encrypted under a
   per-subject key; erasing the subject destroys that key, rendering their ciphertext
   permanently unrecoverable. The (now-undecryptable) records stay in place.
2. **An immutable erasure tombstone** — a new `EntityRecord` (kind
   `erasure_tombstone`) is appended recording *that* the subject was erased. The
   audit bundle still verifies (nothing was removed; one record was added), and you
   can prove the subject existed and was erased.

It also provides an age-based retention helper for time-bound expiry of operational
records.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/services/governance/retention.py` | `SubjectKeyVault`, `RetentionService`, `ERASURE_KIND`. |
| `himmy/services/governance/__init__.py` | Package marker (empty). |

> The module covers WS4.2 of the [hardening plan](../enterprise/HARDENING_PLAN.md).
> The per-subject *encryption* primitives it leans on
> (`himmy/services/storage/encryption.py::FieldEncryptor`) are WS4.4.

## Key abstractions

### `SubjectKeyVault` (`retention.py`)

Holds a per-subject 32-byte data key. Destroying a key crypto-shreds that subject's
data (any ciphertext encrypted under it becomes unrecoverable).

```python
class SubjectKeyVault:
    def key_for(self, subject_id: str) -> bytes: ...       # create-on-first-use
    def encryptor_for(self, subject_id: str) -> FieldEncryptor: ...
    def has(self, subject_id: str) -> bool: ...            # i.e. not yet erased
    def destroy(self, subject_id: str) -> bool: ...        # crypto-shred; True if a key existed
```

`encryptor_for` returns a `FieldEncryptor` (from
`himmy/services/storage/encryption.py`) bound to the subject's key — this is how a
subject's sensitive payloads get encrypted in the first place. The default vault is
**in-memory**; a production deployment backs it with a KMS / HSM-held key store.

### `RetentionService` (`retention.py`)

```python
class RetentionService:
    def __init__(self, entity_registry, *, key_vault=None, clock=None): ...
    def erase_subject(self, subject_id: str, *, reason: str = "") -> EntityRecord: ...
    @staticmethod
    def expired(records, *, max_age_seconds, now_epoch) -> list: ...
```

- `erase_subject` — calls `key_vault.destroy(subject_id)` (crypto-shred), then
  appends an `erasure_tombstone` `EntityRecord` whose payload records
  `subject_id`, `reason`, `crypto_shredded` (whether a key actually existed), and
  `erased_at` (from the injectable `clock`, default `utc_now_iso`). Returns the
  registered tombstone. If no `key_vault` is wired, the tombstone still records the
  erasure intent (`crypto_shredded=False`).
- `expired` — a stateless helper that returns the records whose `created_at` is older
  than `max_age_seconds` relative to `now_epoch`. Parses `created_at` ISO strings
  (treating naive timestamps as UTC); records without a parseable `created_at` are
  skipped. This is the building block for a purge job; the service does not itself
  schedule or run deletions.

`ERASURE_KIND = "erasure_tombstone"` is the `EntityRecord.kind` for tombstones.

## How it works / data flow

### Erasure paradox resolution

```
subject's sensitive payloads ── encrypted under ──> SubjectKeyVault.key_for(subject)
                                                          │
   erase_subject(subject) ────────────────────────────── ▼
        1. key_vault.destroy(subject)   → ciphertext now permanently unrecoverable
        2. append EntityRecord(kind="erasure_tombstone", payload={subject_id, reason,
                                crypto_shredded, erased_at})

audit bundle re-verify:
   - no record removed  → no "missing"
   - one record added   → the tombstone (expected, accounted for)
   - signature intact   → trail still proves integrity
```

The subject's original records remain in the spine, but their encrypted fields are
just opaque bytes no one can decrypt. The immutability of the audit trail is
preserved, the erasure is provable, and GDPR erasure is satisfied — without a
destructive `DELETE` against an append-only store.

### Retention (time-bound)

For operational records that simply age out, `RetentionService.expired(...)` selects
the over-age records; a caller (a `himmy gc`-style job / scheduled task) decides what
to do with them. This is intentionally a pure selector, not a built-in deleter — the
deletion policy is a deployment decision.

## Configuration

There are no governance-specific env vars in `retention.py` itself. The capability is
wired by a deployment:

- Encryption at rest is opt-in via `HIMMY_ENCRYPTION_KEY` (see
  `himmy/services/storage/encryption.py`); without it, `FieldEncryptor` has no key to
  bind and crypto-shredding has nothing to shred.
- The default `SubjectKeyVault` is in-memory; back it with a managed key store for
  durable, multi-process erasure guarantees.

## Extension points

- Back `SubjectKeyVault` with a KMS/HSM (override `key_for` / `destroy`) for durable
  per-subject keys.
- Build a purge job around `RetentionService.expired(...)` with per-`kind`/per-tenant
  retention windows.
- Inject a custom `clock` into `RetentionService` for deterministic tombstone
  timestamps in tests.

## Gotchas & invariants

- Never mutate or delete `EntityRecord` rows — erasure is *crypto-shred + append a
  tombstone*, never an in-place delete.
- Crypto-shredding only works if the subject's data was actually encrypted under the
  subject's key; without `FieldEncryptor` wiring, `erase_subject` records intent
  (`crypto_shredded=False`) but cannot make plaintext unrecoverable.
- The default key vault is in-memory: keys (and thus the ability to shred durably) do
  not survive a process restart unless a persistent backend is wired.
- `expired` is a selector only — it does not delete anything.

## Related docs

- [Audit](audit.md) — the append-only spine + signed bundles the tombstones reconcile with.
- [Guardrails / DLP](guardrails.md) — DLP is the sibling WS4 data-governance control.
- [Enterprise hardening plan (WS4.2 / WS4.4)](../enterprise/HARDENING_PLAN.md)
