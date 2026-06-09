# Privacy & compliance audit (WS4.7)

Himmy can *actively assess* its own privacy posture — not just enforce controls, but
**measure and prove** they held. The audit is a Ragas-inspired "experiment" with two
halves that roll into one signed scorecard:

* a **probe half** — a seeded, adversarial `EvaluationSuite` run against the *live* agent
  through the existing `AgentEvalHarness` (does the agent disclose, retain, or train on
  data it shouldn't?), and
* a **recorded-data half** — deterministic metrics over what is **already persisted** (is
  there PII in stored outputs? anything past retention? every erasure complete? the audit
  log untampered? — plus consent-derived metrics that light up once WS4.6 is wired).

Both roll into one `PrivacyAuditReport`, registered as its own
`EntityRecord(kind="privacy_audit_report")` so it inherits the tamper-evident signed
bundle for free. A runnable end-to-end walkthrough (governed mode) lives in
`examples/deepgyan_consent/main.py`.

> **Offline-first.** The audit runs no model and needs no keys. A zero-config `himmy run`
> / `create_app()` is **unchanged** by WS4.7 — the `PrivacyAuditService` is a passive,
> read-only auditor you *invoke*, never on the hot path. Over an empty store it is a
> clean recorded-only scan that registers a passing report; it lights up extra metrics
> only as the matching seams (a real provider, a retention service, a consent ledger) are
> configured, mirroring how RBAC/encryption stay inert until set. *(verified:
> `tests/services/evaluation/privacy/test_privacy_audit_service.py`.)*

## The metrics

Each metric scores in `0.0..1.0` (1.0 = fully compliant) and is one of two kinds:
**deterministic** recorded-data scans, or **LLM** aspect-critics. A metric whose seam is
absent reports `skipped=True` (excluded from the posture roll-up) rather than a misleading
`score=0` — so the scorecard distinguishes "compliant" from "couldn't check".

| Metric | Kind | Measures | Lights up when |
|---|---|---|---|
| `pii_leakage` | deterministic | `1 − leak-rate` over recorded run outputs/errors/structured, DLP-scanned with the combined `_PII_RULES + _NEPAL_PII_RULES` rule set. Findings carry labels + counts + `run_id` only — **never the matched value**. | always |
| `audit_integrity` | deterministic | Re-derives each first-class record's `content_hash` against the last signed bundle baseline; any in-place mutation drags the score. **First run has no baseline ⇒ `skipped`** (no false fail). | a signed baseline exists |
| `retention_compliance` | deterministic | No pre-fetched run/memory/episodic record lives past `max_age_seconds` (via `RetentionService.expired`). | a `RetentionService` + a max-age window are wired |
| `erasure_completeness` | deterministic | Per `erasure_tombstone`: subject key shredded (`key_vault.has(subject) is False`), tombstone present, **no** post-erasure reference survives, and (optionally) the signed bundle still verifies. Skips when no subject has been erased. | a `SubjectKeyVault` is wired |
| `consent_coverage`, `purpose_limitation`, `denial_enforcement` | deterministic, **consent-gated** | Join recorded runs against the consent ledger / denial events. **Defined but None-guarded ⇒ `skipped` and excluded from posture until Part A (WS4.6) is wired** — never scored `0`. | a consent ledger (WS4.6) is injected |
| `safety` | probe | Fraction of adversarial probe cases that did **not** elicit a disallowed disclosure (graded by the existing deterministic `SafetyMetric`). | a runtime + persona are wired (the probe half) |
| `*_critic` / `privacy_rubric` | LLM aspect-critic | Ragas aspect-critic & rubric scorers (discloses-3rd-party-PII / uses-unconsented-data / grounded-in-consented-sources). | a **real** provider is wired (`provider not in (None, 'stub')`) |

**Posture** is the weighted aggregate over the *non-skipped* metrics; the overall verdict
`passed` is `True` only when every (non-skipped) veto metric passed **and** the aggregate
cleared its bar. The default veto set is `('safety',)` — kept opt-in so a recorded-only
scan is never vetoed by a probe metric that didn't run. An audit with only skipped metrics
is vacuously passing (there is nothing to substantiate a failure).

> **The report can never become a PII sink.** A `PrivacyFinding` forbids extra fields and
> exposes only `code`/`severity`/`count` plus *reference ids* (`record_refs` /
> `subject_refs`) — there is deliberately no `value`/`text`/`match` field. To inspect the
> offending data you pull the referenced record through the normal access-controlled path;
> the signed report itself stays free of cleartext PII. *(verified:
> `tests/services/evaluation/privacy/test_privacy_models.py`.)*

## Skipped until Part A (the consent caveat)

`consent_coverage`, `purpose_limitation`, and `denial_enforcement` need WS4.6's
`ConsentService` (the ledger seam) and the `denied_*` enforcement events it emits. Until
that seam is injected (`RecordedDataContext.consent_service is None`) those three metrics
report `skipped=True` and are excluded from the posture — **not** scored `0`. This is by
design: Part B ships and tests green before Part A exists, and the consent-derived metrics
plus the `from_consent_ledger` probe refresh light up *automatically* once consent is
wired. So on an ungoverned deployment a clean scorecard shows them as `SKIP`, which is
honest ("not checked"), not a silent pass.

## CLI — `himmy audit privacy`

A CI gate that scans the durable spine and **exits non-zero when the posture fails** (a
planted PII leak / retention gap / erasure gap fails the build). It runs no model and needs
no keys.

```console
$ himmy audit privacy
privacy audit: PASS   posture=1.000   (recorded-only)
  [ok  ] pii_leakage            1.00 (>= 0.90) no recorded outputs to scan
  [SKIP] audit_integrity        0.00 (>= 0.90) no signed baseline yet (first run) — integrity check skipped
  [SKIP] consent_coverage       0.00 (>= 0.90) consent service not wired (Part A) — metric skipped
  [SKIP] purpose_limitation     0.00 (>= 0.90) consent service not wired (Part A) — metric skipped
  [SKIP] denial_enforcement     0.00 (>= 0.90) consent service not wired (Part A) — metric skipped
```

The scorecard prints one line per metric (`ok` / `FAIL` / `SKIP`, the score vs its
threshold, and a PII-free detail), with any findings indented beneath as `severity:code`
+ counts + `refs=`/`subjects=` (reference counts only). `--json` emits the same report as
a machine-readable `PrivacyAuditReport`:

```console
$ himmy audit privacy --json | jq '{passed, posture_score, metrics: [.metrics[] | {metric, score, skipped}]}'
```

| Flag | Effect |
|---|---|
| `--lookback N` | Bound the recorded-data scan to the last `N` seconds (default: unbounded). |
| `--suite NAME` | Run the named adversarial probe suite against a live agent (default: a recorded-only scan; the offline path). |
| `--export-bundle PATH` | Also write a tamper-evident signed bundle over the registered report to `PATH`. |
| `--json` | Emit the machine-readable JSON scorecard instead of the human table. |

Exporting a signed bundle needs a signing key (Ed25519 when `HIMMY_AUDIT_PRIVATE_KEY` is
set, else HMAC with `HIMMY_AUDIT_SECRET`); without one, `--export-bundle` errors:

```console
$ HIMMY_AUDIT_SECRET=$SIGNING_SECRET himmy audit privacy --export-bundle audit.json
signed audit bundle written to audit.json (algorithm=HMAC-SHA256, records=1)
privacy audit: PASS   posture=1.000   (recorded-only)
  ...
```

> The CLI scan currently wires an **in-memory** registry/store (an empty store is the
> clean posture); a deployment plants real data by handing the auditor its own wired
> registry — which is exactly what the BFF does, so the **HTTP surface scans the live
> store**. A durable-spine CLI scan is a documented follow-up.

## HTTP — `/v1/audit/privacy`

The same auditor over the BFF, tenant-scoped exactly like the rest of himmy. RBAC: `auditor`
and `admin` hold `audit:run` and `audit:read`; a `viewer`/`operator` without `audit:run`
gets a 403.

```console
# Run an audit over the caller's authorized workspace (auditor/admin).
$ curl -fsS -X POST "$HIMMY/v1/audit/privacy?lookback=86400" \
       -H "Authorization: Bearer $AUDITOR_TOKEN" | jq '{passed, posture_score}'

# ...and also export a signed bundle over the registered report in one call.
# 503 when no signing key is configured (mirroring GET /v1/audit/bundle).
$ curl -fsS -X POST "$HIMMY/v1/audit/privacy?bundle=true" \
       -H "Authorization: Bearer $AUDITOR_TOKEN" | jq '{report: .report.report_id, algo: .bundle.algorithm}'

# The posture trend (registered reports, newest-first) and one report by id.
$ curl -fsS "$HIMMY/v1/audit/privacy?limit=20" -H "Authorization: Bearer $AUDITOR_TOKEN"
$ curl -fsS "$HIMMY/v1/audit/privacy/$REPORT_ID" -H "Authorization: Bearer $AUDITOR_TOKEN"
```

## The report is covered by the signed audit bundle

Because the report is its own `EntityRecord(kind="privacy_audit_report")`, the existing
audit-bundle export at `GET /v1/audit/bundle` was **widened to union**
`SECURITY_EVENT_KIND + PRIVACY_AUDIT_REPORT_KIND` — so a single signed bundle now genuinely
covers both the security-event trail *and* every registered privacy audit report.
Tampering with any cited record after the fact flips
`verify_audit_bundle_ed25519().ok` (or its HMAC sibling) to `False`.

```console
# An auditor exports the union bundle and verifies it offline with the public key alone.
$ curl -fsS "$HIMMY/v1/audit/bundle" -H "Authorization: Bearer $AUDITOR_TOKEN" > bundle.json
```

A per-report bundle is also available directly from the service
(`PrivacyAuditService.export_signed_bundle(report)` / `verify_report(report, bundle)`),
which commits to the report's own record plus the first-class audit kinds it cites
(`security_event` / `erasure_tombstone` / `consent`).

## Refreshing the probe corpus (the Ragas "refresh regularly")

The adversarial probe suite is **offline-deterministic by default** — given a `seed` the
`DeterministicProbeSynthesizer` emits a byte-identical suite every run (CI-stable, no
model). Its PII attack/grading corpus is synthesized from `_PII_RULES + _NEPAL_PII_RULES`
with fake-but-valid tokens (no real PII ever). An optional `LLMProbeSynthesizer` rewrites
those prompts into more natural phrasings, but is constructed **only when a real provider
is wired** and **fails closed** to the deterministic prompt on any malformed output.

Ragas recommends refreshing your test set against live data. `PrivacyProbeGenerator.from_consent_ledger(registry)`
does exactly that: it reads the deployment's own `kind="consent"` + `erasure_tombstone`
records and turns opted-out subjects into `INDUCE_TRAINING_USE` probes and erased subjects
into `CROSS_SUBJECT_LEAK` probes — degrading to the catalog default (tagged
`derived_from="catalog_default"`) when the store has neither, so you always get an explicit
signal, never a silent empty suite.

**Operational cadence.** Schedule `himmy audit privacy` as a periodic CI / cron gate
(e.g. nightly, plus on every deploy) so a regression — a new PII leak, a retention drift,
an incomplete erasure — fails fast with a findings reference but no raw value. Re-baseline
`audit_integrity` against the latest signed bundle on each run, and once WS4.6 is wired,
regenerate the probe corpus from the live ledger (`from_consent_ledger`) so newly opted-out
or erased subjects are probed on the next cycle.

## See also

* `docs/enterprise/consent.md` — WS4.6 consent & purpose-limitation enforcement (Part A),
  the layer that lights up the consent-derived metrics here.
* `docs/enterprise/HARDENING_PLAN.md` — the WS4.6 / WS4.7 plan entries.
* `examples/deepgyan_consent/main.py` — the end-to-end DeepGyan walkthrough that runs an
  audit and verifies the bundle.
