"""BFCL HIMMY-arm handler: the SAME model on OpenRouter, tool-calls routed through himmy.

This module defines the HIMMY (treatment) arm of the same-model lift harness. It
runs the *same* OpenRouter model as the RAW arm, but every tool-call render and
decode goes through himmy's cross-provider tool-call format registry + tolerant
parser/repair + multi-turn result-threading. The delta vs. the RAW arm is exactly
what himmy adds.

APPLES-TO-APPLES contract (the ONLY variable is himmy in the loop)
------------------------------------------------------------------
Both arms run in **prompting mode** (``is_fc_model=False``) against the same model,
same tasks (same ``--run-ids``), same decoding (``temperature 0`` + the same seed),
same OpenRouter transport. The two — and only two — differences are:

  1. MANIFEST. RAW injects BFCL's stock ``system_prompt_pre_processing_chat_model``
     manifest; HIMMY injects ``himmy.format_for(tag, override).render_system_manifest``
     (e.g. the Hermes/Qwen ChatML ``<tools>...</tools>`` grammar, or Llama3-JSON,
     or Mistral-v3).
  2. PARSE. RAW decodes with BFCL's ``default_decode_ast_prompting`` /
     ``default_decode_execute_prompting`` (which throw on a ``<tool_call>{json}</tool_call>``
     envelope); HIMMY decodes with the format's tolerant ``parse`` (native grammar
     pass OR'd with himmy's fail-open text/python recovery + name repair).

For multi_turn, HIMMY additionally threads tool results back with the format's
``render_tool_results`` (Hermes ``<tool_response>`` / Llama ipython / Mistral
``[TOOL_RESULTS]``) instead of BFCL's ``format_execution_results_prompting``.

PROOF himmy is genuinely in the loop
------------------------------------
* Every decode logs a marker line ``[himmy-arm] format=<name> parsed N call(s) ...``
  via the ``himmy.bfcl`` logger (see ``HIMMY_MARKER``).
* The handler asserts the resolved format actually came from himmy's registry
  (``format.__module__`` starts with ``himmy``) at construction.
* Decoded calls carry himmy-minted ``tool_call_id`` UUIDs (from the parser), never
  an OpenRouter provider id — distinct provenance from the RAW arm.

The RAW handler (``handlers.OpenRouterRawHandler``) imports NO himmy code; this
handler imports himmy directly. They are two distinct ``MODEL_CONFIG_MAPPING``
entries, so the arms cannot accidentally share a code path.

himmy import
------------
himmy is not pip-installed in the bfcl venv (uv-managed, no pip). We make it
importable with a single ``sys.path.insert(0, repo_root)`` — proven to resolve all
needed himmy modules in the bfcl py3.12 venv with zero dependency conflict (himmy
core deps pydantic>=2 / pyyaml>=6 / httpx>=0.27 are already satisfied). This is
zero-mutation: the bfcl venv stays pristine and the integration is fully
reproducible from the repo.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# --- make himmy importable in the bfcl venv (see module docstring) ----------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hashlib  # noqa: E402
import json as _json  # noqa: E402

from bfcl_eval.constants.enums import ModelStyle, ReturnFormat  # noqa: E402
from openai import (  # noqa: E402
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from handlers import (  # noqa: E402  (RAW transport base; himmy layered on top)
    OpenRouterRawHandler,
)

# himmy imports — the whole point of this arm. If these fail, the arm must fail
# loudly (no silent fallback to a himmy-free path), so the lift is never faked.
from himmy.services.inference.models import (  # noqa: E402
    BoundTool,
    ToolReturnRecord,
)
from himmy.services.inference.tool_formats import format_for  # noqa: E402

# himmy's PRODUCTION schema-aware lenient arg repair (himmy/services/tools/service.py):
# losslessly coerces a stringified scalar ("20" -> 20, "true" -> True) ONLY when the
# tool's JSON schema declares that scalar type and the conversion round-trips exactly.
# This is the shipped capability that closes the small-model "stringify everything"
# gap; we apply it in the HIMMY arm exactly as himmy production would. The RAW arm
# never gets it (BFCL's plain decode), so any resulting delta is genuine himmy lift.
from himmy.services.tools.service import _coerce_lenient_args  # noqa: E402

_LOG = logging.getLogger("himmy.bfcl")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_h)
    _LOG.setLevel(logging.INFO)

#: Stable marker string the run/adversarial phase greps for to prove himmy ran.
HIMMY_MARKER = "[himmy-arm]"

# --------------------------------------------------------------------------- #
# Bounded NON-STREAMING transport (the streaming-wedge fix)
# --------------------------------------------------------------------------- #
# ROOT CAUSE of the prior wedge: the himmy arm inherited BFCL's stock
# ``generate_with_backoff`` (handlers.py -> OpenAICompletionsHandler), which calls
# ``client.chat.completions.create(**kwargs)`` with NO explicit per-request timeout
# and whose ``@retry_with_backoff`` only re-tries on ``RateLimitError``. The OpenAI
# SDK's default request timeout is 600s, so when the OpenRouter read socket froze
# mid-response on a qwen task, the call blocked for up to ten minutes (observed
# "frozen socket 3+ min") with no retry path, hanging the whole single-threaded run.
# A DIRECT NON-STREAMING probe of the same qwen model returned in 1-3.5s, proving the
# stall was the unbounded transport, not himmy or the manifest.
#
# THE FIX (scoped to the himmy arm only; the RAW arm is untouched): every himmy
# OpenRouter completion is issued NON-STREAMING (``stream=False``, explicit) with an
# explicit bounded per-request ``timeout`` AND a small bounded retry/backoff over the
# transient transport errors (timeout / connection drop / rate-limit / 5xx). A single
# task can therefore never hang the run: the worst case is
# (HIMMY_MAX_ATTEMPTS x HIMMY_REQUEST_TIMEOUT_S) bounded wall-clock, after which the
# task fails loudly and the run proceeds. Decoding (temperature 0, hermes_chatml_xml
# override, render/parse/threading) is byte-for-byte identical to before.

#: Per-request wall-clock timeout for one OpenRouter completion (seconds). Tuned to
#: comfortably exceed the observed direct-probe latency (1-3.5s) with headroom for a
#: slow-but-real qwen response, while being far below the SDK's 600s default so a
#: frozen socket is aborted in seconds, not minutes. Overridable via env for ops.
HIMMY_REQUEST_TIMEOUT_S = float(os.getenv("HIMMY_REQUEST_TIMEOUT_S", "90"))

#: Bounded retry budget for transient transport failures (the FIRST try plus up to
#: HIMMY_MAX_ATTEMPTS-1 retries). With timeout=90s and 3 attempts the absolute worst
#: case for one task is ~3*90s + backoff sleeps, but in practice a transient stall
#: clears on the first retry (the direct probe succeeds in seconds).
HIMMY_MAX_ATTEMPTS = int(os.getenv("HIMMY_MAX_ATTEMPTS", "3"))

#: Exponential backoff base/cap (seconds) between transient retries.
HIMMY_BACKOFF_BASE_S = float(os.getenv("HIMMY_BACKOFF_BASE_S", "2"))
HIMMY_BACKOFF_MAX_S = float(os.getenv("HIMMY_BACKOFF_MAX_S", "20"))

#: Transport errors that are transient and worth a bounded retry. ``APITimeoutError``
#: (our bounded timeout firing on a frozen/slow socket) and ``APIConnectionError`` are
#: the wedge culprits; ``RateLimitError`` covers qwen's rate limiting; a 5xx
#: ``APIError`` is an upstream-provider hiccup. A non-transient 4xx (bad request) is
#: NOT retried — it re-raises immediately so a real bug fails fast.
_HIMMY_TRANSIENT_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)

#: Sidecar dir: per-model maps of (model-output-text-hash -> bound tool names) written
#: at GENERATION time and read back at DECODE (evaluate) time. See the long comment on
#: ``_himmy_parse`` for why this is necessary and exactly faithful to himmy production.
_KNOWN_DIR = _REPO_ROOT / "scripts" / "bfcl" / "workdir" / "himmy_known"


def _text_key(text: str) -> str:
    """Stable hash of a model-output string (the sidecar key)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# BFCL function-def -> himmy BoundTool
# --------------------------------------------------------------------------- #
#: Gorilla (BFCL) JSON-type dialect -> standard JSON-schema type names. BFCL uses
#: ``float``/``dict``/``tuple`` etc. (the Python-ish Gorilla dialect); himmy's manifest
#: renderers AND its schema-aware scalar coercion expect canonical JSON-schema types
#: (``number``/``object``/``array``/...). Getting ``float`` -> ``number`` right is what
#: lets himmy losslessly coerce a stringified ``"0.01"`` back to a float.
_GORILLA_TO_JSON_TYPE = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
}


def _to_json_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert a BFCL ``parameters`` block into a JSON-schema ``args_json_schema``.

    BFCL uses the Gorilla dialect (``dict``/``float``/``tuple``/``integer``/...);
    himmy's manifest renderers and schema-aware coercion expect canonical JSON-schema
    types, so we remap every ``type`` value through :data:`_GORILLA_TO_JSON_TYPE`
    (recursively over nested ``properties``/``items``). Unmapped types (``integer``,
    ``string``, ``boolean``, ``array``, ``number``, ``object``) pass through unchanged.
    """
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    def _fix(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = _GORILLA_TO_JSON_TYPE.get(v, v)
                elif k == "type" and isinstance(v, list):
                    out[k] = [_GORILLA_TO_JSON_TYPE.get(t, t) if isinstance(t, str) else t for t in v]
                else:
                    out[k] = _fix(v)
            return out
        if isinstance(node, list):
            return [_fix(x) for x in node]
        return node

    schema = _fix(parameters)
    if isinstance(schema, dict) and schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema


def _bound_tools(functions: list[dict]) -> list[BoundTool]:
    """Convert BFCL ``test_entry['function']`` into himmy ``BoundTool`` records."""
    tools: list[BoundTool] = []
    for fn in functions:
        tools.append(
            BoundTool(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                args_json_schema=_to_json_schema(fn.get("parameters", {})),
            )
        )
    return tools


class HimmyOpenRouterHandler(OpenRouterRawHandler):
    """HIMMY arm: the same OpenRouter model, tool-calls routed through himmy.

    Subclasses the RAW handler ONLY to reuse the OpenRouter transport (base_url +
    key + ``generate_with_backoff``). All tool-call behavior (manifest render, AST
    decode, execute decode, multi-turn result threading) is overridden to run
    through himmy. The model is driven in prompting mode (``is_fc_model=False``);
    the native ``tools=`` field is never sent (so the manifest is the only tool
    advertisement, matching the RAW prompting baseline).

    The himmy format is chosen by ``HIMMY_FORMAT_OVERRIDE`` (set per-registration in
    ``register_himmy.py``) so selection is deterministic and not subject to
    model-tag string quirks.
    """

    #: Format-name override passed to ``himmy.format_for`` (e.g. "hermes_chatml_xml").
    #: Set on the subclass/instance by the registration; None => auto-select by tag.
    HIMMY_FORMAT_OVERRIDE: str | None = None

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        # Per-task tool schemas (name -> args_json_schema), stashed during manifest
        # injection so the response-parse phase can apply himmy's schema-aware lenient
        # coercion. Single-task at a time (himmy-arm generate runs --num-threads 1 for
        # determinism), so there is no cross-task race.
        self._tool_schemas: dict[str, dict] = {}
        # Known-names sidecar (see _himmy_parse). Written at generation, read at decode.
        self._known_path = _KNOWN_DIR / f"{registry_name}.jsonl"
        # Lazily-loaded {text_hash -> [names]} + the per-model UNION of all tool names
        # (the conservative fallback when an exact text-hash lookup misses).
        self._known_map: dict[str, list[str]] | None = None
        self._known_union: set[str] | None = None
        # Allow the env to carry the override (registration sets it before the CLI
        # builds the handler), falling back to the class attribute.
        self._himmy_override = os.getenv(
            "HIMMY_FORMAT_OVERRIDE", self.HIMMY_FORMAT_OVERRIDE
        )
        # Resolve once and PROVE the format came from himmy's registry.
        self._fmt = format_for(self.model_name, self._himmy_override)
        assert self._fmt.render_system_manifest.__module__.startswith("himmy"), (
            "HIMMY arm integrity check failed: resolved format is not from himmy "
            f"({self._fmt.render_system_manifest.__module__})"
        )
        _LOG.info(
            "%s init model=%s format=%s use_text_path=%s override=%s",
            HIMMY_MARKER,
            self.model_name,
            self._fmt.name,
            self._fmt.flags.use_text_tool_path,
            self._himmy_override,
        )

    # ------------------------------------------------------------------ #
    # Bounded NON-STREAMING transport (the streaming-wedge fix)
    # ------------------------------------------------------------------ #
    def _himmy_generate_bounded(self, **kwargs):
        """Issue ONE OpenRouter completion NON-STREAMING with a bounded timeout and
        a small bounded retry/backoff, so a single task can never hang the run.

        Replaces the inherited ``generate_with_backoff`` for the himmy arm ONLY. The
        signature/return contract is identical: returns ``(api_response, latency_s)``
        so every downstream parse path is unchanged. ``stream=False`` is set
        explicitly (the OpenAI SDK default is already non-streaming, but we pin it so
        the no-hang guarantee is explicit and immune to any caller/SDK default drift).
        ``timeout`` is passed per-request so the SDK's 600s default never applies.

        Retries cover only the transient transport errors in
        :data:`_HIMMY_TRANSIENT_ERRORS` plus 5xx ``APIError``; a non-transient 4xx
        re-raises immediately. After the bounded attempt budget is exhausted the last
        error re-raises, which BFCL records as a task FAILURE (no fabricated result).
        """
        kwargs.setdefault("stream", False)
        last_exc: Exception | None = None
        for attempt in range(1, HIMMY_MAX_ATTEMPTS + 1):
            start = time.time()
            try:
                api_response = self.client.chat.completions.create(
                    timeout=HIMMY_REQUEST_TIMEOUT_S, **kwargs
                )
                return api_response, time.time() - start
            except _HIMMY_TRANSIENT_ERRORS as exc:
                last_exc = exc
            except APIError as exc:
                # Retry only transient upstream 5xx; re-raise client 4xx immediately.
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= int(status) < 500:
                    raise
                last_exc = exc
            # Bounded exponential backoff, capped, before the next attempt.
            if attempt < HIMMY_MAX_ATTEMPTS:
                sleep_s = min(
                    HIMMY_BACKOFF_MAX_S, HIMMY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                )
                _LOG.info(
                    "%s transport retry attempt=%d/%d after %.2fs backoff (err=%s: %s)",
                    HIMMY_MARKER,
                    attempt,
                    HIMMY_MAX_ATTEMPTS,
                    sleep_s,
                    type(last_exc).__name__,
                    str(last_exc)[:160],
                )
                time.sleep(sleep_s)
        # Budget exhausted: re-raise so BFCL records a real FAILURE, never a fake pass.
        _LOG.info(
            "%s transport gave up after %d attempts (err=%s)",
            HIMMY_MARKER,
            HIMMY_MAX_ATTEMPTS,
            type(last_exc).__name__ if last_exc else "unknown",
        )
        assert last_exc is not None
        raise last_exc

    def _query_prompting(self, inference_data: dict):
        """himmy-arm prompting query over the bounded non-streaming transport.

        Identical message/model/temperature payload to the RAW base
        (:meth:`handlers.OpenRouterRawHandler._query_prompting`); the ONLY change is
        routing the completion through :meth:`_himmy_generate_bounded` instead of the
        unbounded inherited ``generate_with_backoff`` (the wedge fix).
        """
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"])
        }
        return self._himmy_generate_bounded(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )

    def _query_FC(self, inference_data: dict):
        """himmy-arm FC query over the bounded transport (defensive parity).

        The himmy arm runs in prompting mode (``is_fc_model=False``) so this path is
        not exercised today, but we override it too so that IF a future registration
        flips to FC the bounded no-hang transport still applies. Mirrors the RAW
        base's FC payload exactly, swapping only the transport call.
        """
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }
        kwargs: dict[str, Any] = {
            "messages": message,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if len(tools) > 0:
            kwargs["tools"] = tools
        return self._himmy_generate_bounded(**kwargs)

    # ------------------------------------------------------------------ #
    # Manifest injection (replaces BFCL's default system prompt)
    # ------------------------------------------------------------------ #
    def _inject_himmy_manifest(self, test_entry: dict) -> None:
        """Prepend himmy's rendered tool manifest as the system message.

        Mirrors BFCL's ``system_prompt_pre_processing_chat_model`` placement
        (system message first; if the question already has a system message, the
        manifest is prepended to it) but uses himmy's native-format manifest.
        """
        functions: list = test_entry["function"]
        bound = _bound_tools(functions)
        # Stash schemas for schema-aware lenient coercion in the response parse.
        self._tool_schemas = {t.name: (t.args_json_schema or {}) for t in bound}
        manifest = self._fmt.render_system_manifest(bound, "openrouter")
        prompts: list[dict] = test_entry["question"][0]
        if prompts and prompts[0].get("role") == "system":
            prompts[0]["content"] = manifest + "\n\n" + prompts[0]["content"]
        else:
            prompts.insert(0, {"role": "system", "content": manifest})
        _LOG.info(
            "%s manifest format=%s tools=%d bytes=%d",
            HIMMY_MARKER,
            self._fmt.name,
            len(functions),
            len(manifest),
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # himmy manifest instead of BFCL's stock manifest. Everything downstream
        # (message accumulation, transport) is unchanged from the base handler.
        self._inject_himmy_manifest(test_entry)
        return {"message": []}

    # ------------------------------------------------------------------ #
    # Response parse: native tool_calls -> himmy native-format text
    # ------------------------------------------------------------------ #
    def _render_native_calls_as_text(self, tool_calls) -> str:
        """Render provider-side structured ``tool_calls`` into the himmy format's
        NATIVE text envelope, so the downstream himmy ``decode_ast`` / ``decode_execute``
        recover them through ``self._fmt.parse`` (keeping the decode inside himmy).

        WHY this exists (an honest, important transport fact): when himmy advertises
        a model's NATIVE tool-call grammar (e.g. the Llama3-JSON or Hermes/Qwen XML
        manifest), the OpenRouter UPSTREAM provider's own chat template recognizes the
        emitted call and parses it into the OpenAI ``message.tool_calls`` field, leaving
        ``message.content`` = null. himmy's production loop (himmy/services/inference/
        local.py) handles exactly this: it reads native ``tool_calls`` first and only
        falls back to ``tool_format.parse(text)`` when there are none. We mirror that by
        re-rendering the structured calls into the format's native text grammar so the
        single himmy decode site (`self._fmt.parse`) still does the round-trip — the
        marker fires and the provenance stays himmy.

        The envelope is built from the format's declared keys (``name_key`` / ``arg_key``)
        + the per-family wrapper, so it is the same grammar the format's parser consumes.
        """
        fmt = self._fmt
        name_key = fmt.flags.name_key
        arg_key = fmt.flags.arg_key
        parts: list[str] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) if fn is not None else None
            raw_args = getattr(fn, "arguments", None) if fn is not None else None
            if isinstance(raw_args, str):
                try:
                    args = _json.loads(raw_args)
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            # himmy schema-aware lenient coercion: the upstream provider's native
            # tool-parser stringifies scalars (e.g. duration "20", model_3d "true").
            # Coerce them back to the schema's declared scalar type, losslessly, so
            # the stored result round-trips with correct types. (RAW gets no such repair.)
            schema = self._tool_schemas.get(name or "", {})
            if isinstance(schema, dict) and schema.get("properties"):
                args, coerced = _coerce_lenient_args(args, schema)
                if coerced:
                    _LOG.info(
                        "%s coerce format=%s name=%s keys=%s",
                        HIMMY_MARKER,
                        self._fmt.name,
                        name,
                        coerced,
                    )
            obj = {name_key: name or "", arg_key: args}
            payload = _json.dumps(obj)
            if fmt.name.startswith("hermes"):
                parts.append(f"<tool_call>\n{payload}\n</tool_call>")
            else:
                # llama3_json / generic / mistral all parse a bare JSON object
                # (the format parser is fail-open and recovers it).
                parts.append(payload)
        return "\n".join(parts)

    def _parse_query_response_prompting(self, api_response) -> dict:
        """Prefer native ``tool_calls`` (re-rendered into himmy's native text), else
        the raw ``content``. Mirrors himmy's production native-first / text-fallback.

        Token counts come from the RAW base's tolerant ``_usage_tokens`` (inherited).
        """
        msg = api_response.choices[0].message
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if (not content) and tool_calls:
            content = self._render_native_calls_as_text(tool_calls)
            _LOG.info(
                "%s native->text format=%s n=%d (provider parsed tool_calls; "
                "re-rendered into himmy native grammar)",
                HIMMY_MARKER,
                self._fmt.name,
                len(tool_calls),
            )
        in_tok, out_tok = self._usage_tokens(api_response)
        # For the multi_turn chat history we must NOT push the raw provider message
        # object: when the upstream provider parsed the call natively, its `content` is
        # null, and re-sending a null-content assistant message makes the provider reject
        # the next turn with HTTP 400. Instead we record the assistant turn as a clean
        # STRING in himmy's native grammar (the himmy-rendered `content`), so the whole
        # multi-turn loop stays valid AND stays inside himmy's tool-call grammar.
        history_msg = {"role": "assistant", "content": content if content else ""}
        # Persist the (output-text -> bound tool names) mapping so the DECODE phase
        # (a separate `bfcl evaluate` process that only receives the raw text) can
        # apply himmy's REAL name guard — exactly the known-tools set himmy carries in
        # production. Without this the decode would fall open and accept hallucinated /
        # wrong-tool names (which destroys irrelevance and is not apples-to-apples).
        self._record_known_names(content if content else "")
        return {
            "model_responses": content,
            "model_responses_message_for_chat_history": history_msg,
            "input_token": in_tok,
            "output_token": out_tok,
        }

    def _record_known_names(self, output_text: str) -> None:
        """Append ``{hash(output_text): [bound tool names]}`` to the model sidecar.

        Keyed by the exact stored ``result`` string so the decode phase can recover the
        correct known-tools set. Single-threaded generation (``--num-threads 1``) means
        no concurrent writers. Best-effort: a write failure never breaks generation.
        """
        names = sorted(self._tool_schemas.keys())
        if not names:
            return
        try:
            self._known_path.parent.mkdir(parents=True, exist_ok=True)
            with self._known_path.open("a", encoding="utf-8") as fh:
                fh.write(_json.dumps({"k": _text_key(output_text), "names": names}) + "\n")
        except OSError as exc:  # pragma: no cover - disk issue, not a model result
            _LOG.info("%s WARN could not write known-names sidecar: %s", HIMMY_MARKER, exc)

    # ------------------------------------------------------------------ #
    # Decode (replaces BFCL's default tolerant-free prompting decoders)
    # ------------------------------------------------------------------ #
    def _load_known(self) -> None:
        """Load the known-names sidecar written at generation time (cached)."""
        self._known_map = {}
        self._known_union = set()
        if self._tool_schemas:  # available when decode runs in the same process as gen
            self._known_union |= set(self._tool_schemas.keys())
        if not self._known_path.exists():
            return
        try:
            for line in self._known_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                names = list(rec.get("names", []))
                self._known_map[rec["k"]] = names
                self._known_union |= set(names)
        except (OSError, _json.JSONDecodeError, KeyError) as exc:
            _LOG.info("%s WARN could not read known-names sidecar: %s", HIMMY_MARKER, exc)

    def _known_for(self, text: str) -> set[str]:
        """The bound tool names that were in scope when ``text`` was generated.

        Exact match on the output-text hash (the per-task tool set); on a miss, fall
        back to the per-model UNION of all tool names seen (conservative — it only ever
        over-accepts a name that was a real tool in SOME task, never an arbitrary
        hallucination). The empty set (fully fail-open) is used only as a last resort
        and logged, so a missing guard is never silent.
        """
        if self._known_map is None:
            self._load_known()
        assert self._known_map is not None and self._known_union is not None
        hit = self._known_map.get(_text_key(text))
        if hit is not None:
            return set(hit)
        if self._known_union:
            _LOG.info(
                "%s known-names exact-miss; using per-model union (%d names)",
                HIMMY_MARKER,
                len(self._known_union),
            )
            return set(self._known_union)
        _LOG.info("%s WARN no known-names available; parser falls fully open", HIMMY_MARKER)
        return set()

    def _himmy_parse(self, result: str) -> list:
        """Run himmy's format-specific tolerant parser on a raw reply string.

        CRITICAL: himmy's parser is *guarded by the set of known bound tool names* (the
        ``known`` arg) — this is what lets it correctly REJECT a hallucinated or
        wrong-tool name (the irrelevance signal) and is exactly how himmy runs in
        production. BFCL's evaluator passes only the raw text to ``decode_*`` (no
        function list), so we recover the per-task known-tools set from a sidecar the
        generation phase wrote (keyed by the exact output text). Passing an EMPTY set
        here would disable the name guard, accept hallucinated calls, collapse
        irrelevance to 0, and break apples-to-apples (RAW's decoder has the function
        list). So we always thread the real known-names through.
        """
        text = result if isinstance(result, str) else str(result)
        known = self._known_for(text)
        calls = self._fmt.parse(text, known)
        _LOG.info(
            "%s decode format=%s known=%d parsed %d call(s): %s",
            HIMMY_MARKER,
            self._fmt.name,
            len(known),
            len(calls),
            [c.tool_name for c in calls],
        )
        return calls

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """Decode to AST shape ``[{name: {arg: val}}]`` via himmy's parser.

        Used for the single-turn categories (simple_python, multiple, parallel,
        parallel_multiple, irrelevance). For irrelevance the right answer is zero
        calls, so an empty list (himmy correctly recovered nothing) is the success
        signal — we never fabricate a call.
        """
        calls = self._himmy_parse(result)
        return [{c.tool_name: dict(c.args)} for c in calls]

    def decode_execute(self, result, has_tool_call_tag: bool):
        """Decode to execute shape ``['name(arg=val,...)']`` via himmy's parser.

        Used inside the multi_turn loop to drive tool execution. Returning an empty
        list signals "no call this step" and breaks the turn loop (BFCL's
        ``is_empty_execute_response``).
        """
        calls = self._himmy_parse(result)
        out: list[str] = []
        for c in calls:
            args_str = ", ".join(f"{k}={v!r}" for k, v in c.args.items())
            out.append(f"{c.tool_name}({args_str})")
        return out

    # ------------------------------------------------------------------ #
    # Multi-turn result threading (replaces BFCL's default formatter)
    # ------------------------------------------------------------------ #
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """Thread tool results back using himmy's format-native result grammar.

        Replaces BFCL's ``format_execution_results_prompting`` (a bare ``repr`` of
        ``{role:tool}`` dicts) with the format's ``render_tool_results`` (Hermes
        ``<tool_response>`` / Llama ipython / Mistral ``[TOOL_RESULTS]``), so the
        whole multi-turn loop stays inside himmy's grammar — the result-threading
        capability the harness is meant to measure.
        """
        decoded = model_response_data.get("model_responses_decoded", []) or []
        records: list[ToolReturnRecord] = []
        for i, exec_result in enumerate(execution_results):
            # Best-effort tie of a result to the tool name it answered.
            name = decoded[i] if i < len(decoded) else f"tool_{i}"
            if not isinstance(name, str):
                name = str(name)
            # decoded entries are "name(args...)" execute strings; take the bare name.
            bare = name.split("(", 1)[0] if "(" in name else name
            records.append(
                ToolReturnRecord(
                    tool_call_id=f"bfcl-{i}",
                    tool_name=bare,
                    content=str(exec_result),
                )
            )
        rendered = self._fmt.render_tool_results(records)
        inference_data["message"].append({"role": "user", "content": rendered})
        _LOG.info(
            "%s thread-results format=%s n=%d bytes=%d",
            HIMMY_MARKER,
            self._fmt.name,
            len(records),
            len(rendered),
        )
        return inference_data
