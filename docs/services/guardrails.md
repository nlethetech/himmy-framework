# Guardrails

> Composable, dependency-free text inspectors that redact, flag, or block prompts, model output, and tool arguments.

## Overview

A guardrail inspects a piece of text (a user prompt, a model reply, a tool argument)
and returns a `GuardrailVerdict` — possibly *redacted* text, an `allowed` flag
(`False` blocks), and human-readable `reasons`/`flags`. Guardrails are **synchronous
and dependency-free** (regex/string checks), so they sit on the hot path without
adding latency or I/O. A `GuardrailPipeline` chains several: redactions accumulate as
text flows through, and the pipeline blocks if *any* guardrail blocks.

DLP (`DlpGuardrail`) is the compliance-grade variant: instead of always redacting, it
applies a per-class **policy action** (`allow` / `redact` / `tokenize` / `block`),
counts detections for audit (never the values), and can optionally use a Microsoft
Presidio ML backend for recall beyond regex.

Guardrails bind to **three surfaces**: the input prompt, the model output, and tool
arguments (the highest-risk "act" path).

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/services/guardrails/base.py` | `GuardrailVerdict`, the `Guardrail` Protocol, and `GuardrailPipeline`. |
| `himmy/services/guardrails/builtins.py` | Built-in guardrails + `BUILTIN_GUARDRAILS` registry + `build_guardrail_pipeline`. |
| `himmy/services/guardrails/dlp.py` | `DlpAction`, `DlpPolicy`, `TokenVault`, `DlpGuardrail`, Presidio adapter, `build_dlp_guardrail`. |
| `himmy/services/guardrails/tool_hook.py` | `build_guardrail_pre_hook` — adapts a pipeline into a `ToolService` pre-execution hook. |
| `himmy/services/guardrails/__init__.py` | Public surface re-export. |

## Key abstractions

### `Guardrail` protocol + `GuardrailVerdict` (`base.py`)

```python
@dataclass
class GuardrailVerdict:
    allowed: bool = True
    text: str = ""
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

@runtime_checkable
class Guardrail(Protocol):
    name: str
    def inspect(self, text: str, *, context: dict[str, Any]) -> GuardrailVerdict: ...
```

`context` carries a `stage` key (`"input"`, `"output"`, or `"tool_arg"`), plus
`tool`/`arg` for tool-arg inspection — some guardrails are stage-aware (see
`GroundingGuardrail`).

### `GuardrailPipeline` (`base.py`)

Runs an ordered list of guardrails over the text. Each guardrail sees the *current*
(possibly already-redacted) text; redactions chain, `reasons`/`flags` accumulate, and
`allowed` becomes `False` if any guardrail denies. **Note:** the pipeline runs every
guardrail (so all redactions/flags are collected) — the "short-circuit" is logical
(one block blocks the whole verdict), not an early return. `.names` exposes the
pipeline's guardrail names.

### Built-in guardrails (`builtins.py`)

| Class | `name` | Behavior |
| --- | --- | --- |
| `PIIGuardrail` | `pii` | Redacts API keys, JWTs, URL credentials, emails, IBANs, Luhn-valid cards, SSNs, IPv4, MACs, phones. Never blocks. Rules ordered most-specific → loosest. |
| `SecretsGuardrail` | `secrets` | The **credential-only** subset of `PIIGuardrail` (API keys, JWTs, URL credentials). Safe to enforce by default because no legitimate agent output contains a raw credential — so himmy redacts secrets on every outbound vector for every spec-built agent unless `redact_secrets: false`. Never blocks. |
| `InjectionGuardrail` | `injection` | Flags common prompt-injection phrasings ("ignore previous instructions", "reveal your system prompt", …); denies when `block=True` (default). |
| `BlocklistGuardrail` | `blocklist` | Denies when any configured (case-insensitive regex) pattern matches. Custom `name` per instance. |
| `NepalPIIGuardrail` | `nepal_pii` | Redacts Nepal-specific PII: +977 / domestic mobiles, citizenship numbers, PAN (PAN requires the `PAN` label so a bare 9-digit number isn't wrongly redacted). Never blocks. |
| `GroundingGuardrail` | `grounding` | **Output-stage only.** Blocks + replaces a reply that answered from stale parametric memory ("based on my training data", "as of my knowledge cutoff", …) with a refusal pointing the user to a live tool. No-op on input / already-grounded answers. |
| `DlpGuardrail` | `dlp` | Registered into `BUILTIN_GUARDRAILS` from `dlp.py` (default = redact-all PII-like). See below. |

`PIIRule` is the unit of redaction: a `label`, a `placeholder`, a compiled `pattern`,
and an optional `validator` (e.g. cards are only redacted if they pass the Luhn
checksum; IPv4 only if each octet ≤ 255 — this cuts regex false positives).

`build_guardrail_pipeline(names)` resolves built-in names against
`BUILTIN_GUARDRAILS` into a `GuardrailPipeline`; an unknown name raises `HimmyError`.

### DLP (`dlp.py`)

- `DlpAction` — `ALLOW` / `REDACT` / `TOKENIZE` / `BLOCK`.
- `DlpPolicy` — maps a data class (`email`, `card`, …) to an action, with a default
  (default `REDACT`). `DlpPolicy.parse("card:block,email:tokenize,*:redact")` parses
  a spec string (`*`/`default` sets the fallback).
- `TokenVault` — a reversible token↔value map for `tokenize`: `tokenize(label,
  value)` returns a stable `[LABEL:<sha256[:12]>]` token (same value → same token);
  `detokenize(text)` restores originals. A downstream workflow round-trips the value
  while the model only ever sees the token. In-memory by default.
- `DlpGuardrail` — applies a `DlpPolicy` over the PII rule set. Per class: `allow`
  skips, `redact` substitutes the placeholder, `tokenize` vaults, `block` denies the
  whole verdict (and returns the *original* text untouched). Detections are counted
  per label and, if an `audit_sink` callable is set, reported as
  `{label: count}` (never the values) — wired to the security audit log in a
  configured deployment. Flags are emitted as `dlp:<label>`.
- `PresidioAnalyzerAdapter` — optional ML backend (extra `dlp`); lazily imports
  `presidio_analyzer.AnalyzerEngine`, maps Presidio entity types → data-class labels,
  returns `(label, start, end)` spans that `DlpGuardrail` applies (mutating from the
  end so offsets stay valid).
- `build_dlp_guardrail(*, audit_sink=None)` — builds from env: `HIMMY_DLP_POLICY`
  (spec string), `HIMMY_DLP_DEFAULT` (default action), `HIMMY_DLP_BACKEND=presidio`.

## How it works / data flow

### The three surfaces

The runtime (`SingleAgentRuntime`, `himmy/runtime/single_agent.py`) holds an
`input_guardrail` and an `output_guardrail` pipeline. `_apply_guardrail` runs the
pipeline with `context={"stage": ...}` and emits a `GUARDRAIL_APPLIED` event when it
redacts or blocks:

- **Input** — applied to the user prompt before the model sees it (`stage="input"`,
  redact).
- **Output** — applied to the assistant reply (`stage="output"`, redact/block).
- **Tool args** — bound separately on the tool surface via `build_guardrail_pre_hook`
  (see below), `stage="tool_arg"`.

### `build_guardrail_pre_hook` (`tool_hook.py`)

Adapts a `GuardrailPipeline` into a `PreExecutionHook` for the `ToolService`. For
each **string** tool argument it runs `pipeline.inspect(value, context={"stage":
"tool_arg", "tool": ..., "arg": ...})`:

- A blocked arg → `ToolPolicyDecision(allow=False, reason=...)` — the call is denied.
- A redacted arg → passed through transformed via `transformed_args`.

This means an agent cannot exfiltrate PII through `send_email`/`http_request` or be
steered into a blocked action via a tool argument.

### Wiring from an agent spec (`himmy/runtime/from_spec.py`)

When building a runtime from a spec:

1. `build_guardrail_pipeline(spec.guardrails)` becomes the **input** pipeline — only what
   the spec declares, so the user's own prompt is never silently mangled.
2. The **tool-arg** pre-hook pipeline is the secrets default + the spec guardrails:
   `["secrets", *spec.guardrails]` (just `spec.guardrails` when `redact_secrets: false`).
   This stops a model emitting a raw credential into a tool call.
3. The **output** pipeline always prepends `grounding` and the secrets default:
   `build_guardrail_pipeline(["grounding", "secrets", *spec.guardrails])`. Two institutional
   defaults apply to **every** spec-built agent (including Studio-created ones):
   - **`grounding`** blocks any answer given from stale built-in knowledge.
   - **`secrets`** redacts credentials (API keys / JWTs / URL credentials) from tool args,
     tool results, and the final answer. Opt out with `redact_secrets: false`; add `pii` to
     `guardrails` for personal-data (email/phone/…) redaction. Unlike full `pii`, the secrets
     subset never appears in legitimate output, so it is safe to enforce by default.

## Configuration

| Var | Effect |
| --- | --- |
| `HIMMY_DLP_POLICY` | DLP policy spec, e.g. `card:block,email:tokenize,*:redact`. |
| `HIMMY_DLP_DEFAULT` | DLP default action when a class isn't listed (default `redact`). |
| `HIMMY_DLP_BACKEND` | Set to `presidio` to enable the ML backend (needs the `dlp` extra). |

Guardrails on an agent are declared in `agent.yaml` (`guardrails: [pii, injection,
...]`); the `grounding` output guardrail is added automatically.

## Extension points

- Implement the `Guardrail` protocol (a class with a `name` and `inspect`) and add it
  to `BUILTIN_GUARDRAILS`, or pass instances directly to `GuardrailPipeline`.
- Add a `PIIRule` (with a `validator` to suppress false positives) to extend
  PII/DLP coverage.
- Provide a custom `audit_sink` to `DlpGuardrail` to route detection counts to your
  audit pipeline.
- Inject a custom analyzer into `PresidioAnalyzerAdapter` / `DlpGuardrail` to swap the
  ML backend.

## Gotchas & invariants

- Guardrails must stay synchronous + dependency-free (hot path).
- `PIIGuardrail` / `NepalPIIGuardrail` never block — they only redact.
- A `BLOCK` action / a blocking guardrail returns the **original** text (not the
  partially-redacted text) alongside `allowed=False`.
- `GroundingGuardrail` only acts at `stage="output"`; it is a no-op elsewhere.
- DLP detection counts are audited; the sensitive *values* never are.
- Built-in `dlp` is registered into `BUILTIN_GUARDRAILS` at import time from `dlp.py`
  (deferred to avoid a circular import with `builtins.py`).
- The output pipeline always includes `grounding` even when the spec lists no
  guardrails.

## Related docs

- [Sandbox](sandbox.md) — the other safety control on the tool surface.
- [Audit](audit.md) — where DLP detection counts and security events land.
- [Governance](governance.md) — DLP's sibling under WS4 data governance.
- [Enterprise hardening plan (WS4.1)](../enterprise/HARDENING_PLAN.md)
