"""Tool-call FORMAT registry: per-model rendering/parsing of the tool-call grammar.

Open-weight chat models do not share one tool-call convention. Hermes/Qwen emit
``<tool_call>{json}</tool_call>`` XML; the himmy text protocol uses ``TOOL_CALL
<name> {json}``. A :class:`ToolCallFormat` packages the three sites that differ per
model family into one declarative unit:

* :meth:`render_system_manifest` — how the available tools are advertised to the model.
* :meth:`parse` — how the model's reply is read back into :class:`ToolCallRecord`s.
* :meth:`render_tool_results` — how a tool's output is fed back to the model.

A :class:`ToolCallFormatRegistry` maps a *resolved model tag* (plus an optional
per-manager override) to a format, falling back to :data:`GENERIC` — the format that
reproduces today's behavior byte-for-byte. Selection NEVER raises: an unknown tag or
a bad override resolves to GENERIC, so a misconfiguration can never take the inference
path down.

This module deliberately mirrors :mod:`himmy.services.tools.schema_normalize`'s
``ProviderProfile`` pattern (a frozen table keyed by name + a safe-default resolver),
so the two registries read the same way.

The :data:`GENERIC` format is a verbatim wrapper over ``_react_tool_manifest`` /
``parse_text_tool_calls`` / the ``[Tool result]`` label, so threading the GENERIC
format through the managers is a strict no-op. Native open-weight formats register on
top: :data:`HERMES_CHATML_XML` covers Hermes 2 Pro / Hermes 3 / Qwen2.5-Instruct, whose
``<tool_call>{json}</tool_call>`` ChatML-XML grammar the GENERIC tolerant parser misses
entirely. Its parser is FAIL-OPEN — it OR-s in :func:`parse_text_tool_calls` as a
secondary pass, so a native format can only ADD tool-call hits, never regress.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid an import cycle at module load (local.py imports this module)
    from himmy.services.inference.models import (
        BoundTool,
        ToolCallRecord,
        ToolReturnRecord,
    )

#: Render a tool manifest (system-prompt body) from the bound tools + provider key.
ManifestRenderer = Callable[[Sequence["BoundTool"], str], str]
#: Parse a text reply into tool-call records, guarded by the set of known tool names.
ReplyParser = Callable[[str, set[str]], list["ToolCallRecord"]]
#: Render tool-result records into the text fed back to the model.
ResultRenderer = Callable[[Sequence["ToolReturnRecord"]], str]
#: Render few-shot exemplars as REAL prior ``(role, content)`` turns (opt-in; empty list
#: means "no multi-turn few-shot" so the manager renders single-turn as before).
FewshotTurnRenderer = Callable[[Sequence["BoundTool"]], list[tuple[str, str]]]


@dataclass(frozen=True)
class ToolCallGrammarFlags:
    """Data-only knobs describing how a model family diverges on the wire.

    These are pure data (no behavior, no ``if model ==`` branching): the shared
    Hermes/Qwen helpers read these flags so a new family in the same grammar family
    is one frozen row, not a new code path.
    """

    #: JSON key the model uses for a call's arguments object (Hermes/Qwen: ``arguments``).
    arg_key: str = "arguments"
    #: JSON key the model uses for a call's tool name (Hermes/Qwen: ``name``).
    name_key: str = "name"
    #: ChatML role tool results are fed back under. Hermes uses ``tool``; Qwen2.5
    #: renders them as a ``user`` turn (the local template / HF agree on the user turn).
    result_role: str = "user"
    #: Batch consecutive tool results into ONE turn (HF-canonical Qwen2.5 behavior).
    batch_consecutive_results: bool = True
    #: The family emits N back-to-back ``<tool_call>`` blocks for parallel calls.
    parallel_supported: bool = True
    #: Drive Ollama through the prompt/text path (suppress the native ``tools=`` field
    #: and inject the rendered manifest) instead of Ollama's native function-tool API.
    #: GENERIC keeps the native path (today's behavior); the ChatML-XML grammar uses
    #: the text path so the FORMAT is the single independent variable in an A/B.
    use_text_tool_path: bool = False
    #: Number of synthesized few-shot exemplars to APPEND to the manifest (0 = none).
    #: Small open-weight models (e.g. HimalayaGPT-0.5b) do NOT emit ``<tool_call>``
    #: zero-shot — the few-shot exemplar is the trigger (NousResearch recipe,
    #: DEFAULT_FEWSHOT=2). Data-only + opt-in: 0 keeps the manifest byte-identical so
    #: existing Hermes/Qwen users are unaffected and GENERIC byte-parity is untouched.
    fewshot: int = 0
    #: Append the explicit "output ONLY the ``<tool_call>`` line, do not write code /
    #: prose" instruction to the manifest. The 0.5b otherwise answers in prose or
    #: writes a ```python``` block; this nudge (paired with few-shot) keeps it on the
    #: grammar. Opt-in so the canonical Hermes/Qwen manifest stays byte-for-byte.
    no_code_instruction: bool = False
    #: OR-in :func:`parse_python_tool_calls` as a FINAL fallback pass: recover a call
    #: the model emitted as PYTHON syntax (``get_weather(city="Paris")``, a
    #: ```python``` fenced block, or a ``<|python_start|>...<|python_end|>`` span)
    #: instead of a ``<tool_call>`` block. Add-only + name-guarded (a bound tool must
    #: match), so it never false-fires on prose and never changes a successful
    #: ``<tool_call>`` parse. Opt-in (False by default) so GENERIC stays byte-for-byte
    #: ``parse_text_tool_calls`` and canonical Hermes/Qwen recall is unchanged.
    python_call_fallback: bool = False


@dataclass(frozen=True)
class ToolCallFormat:
    """A declarative tool-call grammar bound to one or more model families.

    The three callables are the only sites that vary per model family; everything
    else (validation, repair, execution) is handled by the tools kernel and is the
    same for every format. ``model_tags`` is an allow-list of normalized substrings
    matched against a resolved model tag by :class:`ToolCallFormatRegistry`.
    """

    #: Canonical format key (``"generic"``, ``"hermes_chatml_xml"`` ...).
    name: str
    #: Render the tool manifest into the system-prompt body for this format.
    render_system_manifest: ManifestRenderer
    #: Parse a text reply into tool-call records (fail-open: only adds hits).
    parse: ReplyParser
    #: Render tool-result records into the text fed back to the model.
    render_tool_results: ResultRenderer
    #: Normalized substrings matched against a resolved model tag for auto-selection.
    model_tags: frozenset[str] = field(default_factory=frozenset)
    #: Normalized substrings that VETO auto-selection even if ``model_tags`` matched
    #: (e.g. ``coder`` keeps ``qwen2.5-coder`` on GENERIC). An explicit override still
    #: wins over an exclude — the operator is trusted to know their model.
    exclude_tags: frozenset[str] = field(default_factory=frozenset)
    #: Data-only wire-divergence knobs (see :class:`ToolCallGrammarFlags`).
    flags: ToolCallGrammarFlags = field(default_factory=ToolCallGrammarFlags)
    #: OPT-IN renderer that exposes few-shot exemplars as REAL prior ``(role, content)``
    #: turns. ``None`` (the default for GENERIC + canonical Hermes/Qwen + every other
    #: format) means a manager renders single-turn exactly as before — only a format that
    #: sets this (the himalaya few-shot format) can be composed multi-turn by a manager
    #: that supports it. A manager that ignores it is unaffected.
    fewshot_turns: FewshotTurnRenderer | None = None
    #: OPT-IN manifest renderer that OMITS the flattened ``# Examples`` block, used by a
    #: manager that renders ``fewshot_turns`` as real turns (so exemplars aren't doubled).
    #: ``None`` means "same as :attr:`render_system_manifest`".
    render_system_manifest_base: ManifestRenderer | None = None


def _normalize(value: str | None) -> str:
    """Lower/trim a tag or override for tolerant matching (mirrors ``profile_for``)."""
    return (value or "").strip().lower()


# --------------------------------------------------------------------------- #
# GENERIC format — wraps today's primitives VERBATIM so threading a format is a
# strict no-op. The wrappers delegate (never re-implement) to guarantee byte parity.
# --------------------------------------------------------------------------- #
def _generic_render_system_manifest(tools: Sequence[BoundTool], provider: str) -> str:
    """Today's text manifest, byte-for-byte (delegates to ``_react_tool_manifest``)."""
    from himmy.services.inference.local import _react_tool_manifest

    return _react_tool_manifest(list(tools), provider=provider)


def _generic_parse(text: str, known: set[str]) -> list[ToolCallRecord]:
    """Today's tolerant parser, unchanged (delegates to ``parse_text_tool_calls``)."""
    from himmy.services.inference.tool_protocol import parse_text_tool_calls

    return parse_text_tool_calls(text, known)


def _generic_render_tool_results(results: Sequence[ToolReturnRecord]) -> str:
    """Today's ``[Tool result]`` label for fed-back tool output, verbatim.

    The runtime replays tool returns onto the thread and ``_compose_prompt`` labels
    each ``tool`` message ``[Tool result]``; this reproduces that body so the GENERIC
    format is a faithful no-op if a manager ever renders results itself.
    """
    return "\n\n".join(f"[Tool result]\n{r.content}" for r in results)


GENERIC = ToolCallFormat(
    name="generic",
    render_system_manifest=_generic_render_system_manifest,
    parse=_generic_parse,
    render_tool_results=_generic_render_tool_results,
    model_tags=frozenset(),
)


# --------------------------------------------------------------------------- #
# HERMES_CHATML_XML — Hermes 2 Pro / Hermes 3 / Qwen2.5-Instruct.
#
# These families share one ChatML-XML grammar: tools advertised inside a
# ``<tools>...</tools>`` block of OpenAI-style ``{"type":"function","function":{...}}``
# objects (one per line), calls emitted as ``<tool_call>\n{json}\n</tool_call>``
# blocks (one JSON object each; parallel = N back-to-back blocks, NOT an array),
# tool results fed back inside ``<tool_response>...</tool_response>``.
#
# The grammar text below is verbatim from the local Ollama qwen2.5:*-instruct chat
# template (byte-identical across 0.5b/3b/7b) and agrees with the HF canonical
# Qwen2.5/Hermes tokenizer template on envelope, call, and result shapes.
# --------------------------------------------------------------------------- #

#: The fixed preamble that introduces the ``<tools>`` block (verbatim local template).
_HERMES_TOOLS_PREAMBLE = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)

#: The fixed emission instruction that closes the manifest (verbatim local template).
_HERMES_TOOLS_INSTRUCTION = (
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

#: Match one ``<tool_call> {json} </tool_call>`` block (DOTALL, non-greedy on the JSON).
#: Hermes-3 may wrap reasoning in ``<scratch_pad>...</scratch_pad>`` — that is NOT a
#: ``<tool_call>`` tag so this regex skips it for free.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

#: General format-native tool-use guidance appended to the SHIPPING Hermes/Qwen manifest
#: (NOT the byte-for-byte A/B baseline body, which uses its own ``_baseline_manifest_body``).
#: Covers two cases the canonical local template is silent on: emitting several independent
#: calls in one turn (Hermes's native multi-call form is N back-to-back ``<tool_call>``
#: blocks, not an array) and declining when no advertised function fits (do not fabricate
#: a call). General best practice, not benchmark-specific syntax.
_HERMES_TOOLS_GUIDANCE = (
    "\n\n"
    "When several independent calls are needed, emit multiple "
    "<tool_call></tool_call> blocks back-to-back in the same reply (one JSON object "
    "per block). If no available function fits the request, do not emit a "
    "<tool_call>; answer the user directly."
)

#: The explicit emission discipline a small open-weight model needs (HimalayaGPT-0.5b
#: otherwise answers in prose or writes a ```python``` block). Appended to the manifest
#: only when ``flags.no_code_instruction`` is set — opt-in, so the canonical Hermes/Qwen
#: manifest stays byte-for-byte for every other user.
_HERMES_NO_CODE_INSTRUCTION = (
    "\n\n"
    "When a tool is needed, output ONLY the <tool_call>...</tool_call> line and "
    "NOTHING else — do not write Python, do not use code fences, do not explain. "
    "If no tool is needed, answer the user directly in prose."
)


#: The one-line framing instruction that precedes the exemplars: it tells the model the
#: examples teach the SHAPE only, and that argument values must come from the user's
#: request — the anti-parrot guard so the 0.5b doesn't copy the exemplar literal.
_FEWSHOT_FRAMING = (
    "Below are examples of the FORMAT only. Fill the arguments from the USER's "
    "request, NOT from the examples."
)

#: Obviously-example-only placeholder VALUES, chosen so they cannot collide with (and
#: thus be parroted into) a real eval/user input. Deliberately NOT a real city / a small
#: round int the model parrots (99 / 2 / 5): the string sentinels are bracketed
#: ``<...>`` markers that no natural user phrase contains, and the int sentinel is a
#: distinctive 4242 (not a value any plausible "add a and b" / "set a timer for N"
#: request would carry). If the model copies one verbatim it is unmistakably a parrot
#: (the anti-parrot test asserts these are disjoint from the eval-suite inputs).
#:
#: A small POOL of distinct string sentinels (cycled per property) so a multi-arg tool's
#: exemplar shows a DIFFERENT value in each slot — a 2-arg tool must demonstrate both
#: args carrying distinct, type-correct, obviously-fake values, not one repeated literal.
_PLACEHOLDER_STRINGS = ("<example-text>", "<example-target>", "<example-value>")
_PLACEHOLDER_INT = 4242
_PLACEHOLDER_NUMBER = 42.42
_PLACEHOLDER_BOOL = True


def _example_args_for(tool: BoundTool) -> dict[str, Any]:
    """Fill EVERY required property of the tool with an obviously-fake, type-correct value.

    Filling all required props (not just the first) is what lets a 2-arg exemplar
    (``add_numbers`` a+b, ``translate_text`` text+target_language) actually DEMONSTRATE
    both arguments — a one-arg exemplar can never pass the args check for such a tool, so
    the few-shot recipe capped at ~50%. Each value is a clearly example-only sentinel
    (see :data:`_PLACEHOLDER_STRINGS` et al.) and string slots cycle through a small pool
    so a multi-arg call shows DISTINCT per-slot values, not one repeated literal — a small
    model that copies any of them instead of extracting from the user query is plainly
    parroting. Required props are filled (falling back to every declared prop when the
    schema omits ``required``) so the exemplar always satisfies the tool's arg contract.
    """
    schema = tool.args_json_schema or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    required = schema.get("required")
    # Fill the REQUIRED props (in declared order); fall back to all declared props when a
    # schema omits `required` so the exemplar still demonstrates real arguments.
    if isinstance(required, list) and required:
        keys = [k for k in props if k in required]
    else:
        keys = list(props)

    example_args: dict[str, Any] = {}
    string_slot = 0
    for key in keys:
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t == "integer":
            example_args[key] = _PLACEHOLDER_INT
        elif t == "number":
            example_args[key] = _PLACEHOLDER_NUMBER
        elif t == "boolean":
            example_args[key] = _PLACEHOLDER_BOOL
        elif t == "array":
            example_args[key] = []
        elif t == "object":
            example_args[key] = {}
        else:  # string / unknown — cycle the pool so each slot differs
            example_args[key] = _PLACEHOLDER_STRINGS[
                string_slot % len(_PLACEHOLDER_STRINGS)
            ]
            string_slot += 1
    return example_args


def _required_prop_count(tool: BoundTool) -> int:
    """How many required args the tool declares (falls back to declared-prop count)."""
    schema = tool.args_json_schema or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return 0
    required = schema.get("required")
    if isinstance(required, list) and required:
        return sum(1 for k in props if k in required)
    return len(props)


def _fewshot_tools(tools: Sequence[BoundTool], flags: ToolCallGrammarFlags) -> list[BoundTool]:
    """The (up to ``flags.fewshot``) DISTINCT tools to build exemplars over.

    Picks the first N distinct tools so the exemplars cover DIFFERENT tools (anti-parrot:
    diversity over one tool's value), capped at the available tool count. When at least
    one MULTI-ARG tool exists but the leading-N slice would be all single-arg, the first
    multi-arg tool is swapped into the last slot so the exemplar set always demonstrates a
    2-arg call when one is available — a single-arg-only exemplar set is what lets a model
    pass single-arg tasks while failing every 2-arg tool (the ~50% ceiling).
    """
    if flags.fewshot <= 0 or not tools:
        return []
    n = min(flags.fewshot, len(tools))
    chosen = list(tools[:n])
    if n >= 1 and not any(_required_prop_count(t) >= 2 for t in chosen):
        multi = next((t for t in tools if _required_prop_count(t) >= 2), None)
        if multi is not None:
            chosen[-1] = multi
    return chosen


def _exemplar_call_json(tool: BoundTool, flags: ToolCallGrammarFlags) -> str:
    """The ``<tool_call>`` JSON body for one exemplar tool (placeholder-filled args)."""
    call = {flags.name_key: tool.name, flags.arg_key: _example_args_for(tool)}
    return json.dumps(call, separators=(", ", ": "), ensure_ascii=False)


def _synthesize_fewshot_example(tool: BoundTool, flags: ToolCallGrammarFlags) -> str:
    """Synthesize ONE flattened exemplar: a user line + the ideal ``<tool_call>`` reply.

    Used for the single-turn (in-manifest) rendering. The exemplar is tied to the real
    manifest (not a hard-coded weather demo) and uses an obviously-fake placeholder value
    (anti-parrot). It teaches the *envelope*, not a domain.
    """
    call_json = _exemplar_call_json(tool, flags)
    return (
        f"Example — user: Use the {tool.name} tool.\n"
        f"Example — assistant:\n<tool_call>\n{call_json}\n</tool_call>"
    )


def render_fewshot_turns(
    tools: Sequence[BoundTool], flags: ToolCallGrammarFlags
) -> list[tuple[str, str]]:
    """Render the few-shot exemplars as REAL prior conversation turns.

    Returns a list of ``(role, content)`` pairs — per exemplar a ``user`` turn (the
    example query) followed by an ``assistant`` turn (the ``<tool_call>`` answer) — so a
    manager that supports multi-turn (the himalaya path) can present few-shot as genuine
    PRIOR TURNS (the vendor recipe) instead of one flattened block. Empty when
    ``flags.fewshot<=0`` or there are no tools. This is data-only and OPT-IN: a manager
    that does not call it (canonical Hermes/Qwen single-turn, GENERIC, every other
    manager) renders byte-identically to before.
    """
    turns: list[tuple[str, str]] = []
    for tool in _fewshot_tools(tools, flags):
        call_json = _exemplar_call_json(tool, flags)
        turns.append(("user", f"Use the {tool.name} tool."))
        turns.append(("assistant", f"<tool_call>\n{call_json}\n</tool_call>"))
    return turns


def _render_fewshot_block(tools: Sequence[BoundTool], flags: ToolCallGrammarFlags) -> str:
    """Render up to ``flags.fewshot`` exemplars (empty when fewshot<=0 or no tools)."""
    chosen = _fewshot_tools(tools, flags)
    if not chosen:
        return ""
    examples = [_synthesize_fewshot_example(tool, flags) for tool in chosen]
    return (
        "\n\n# Examples\n\n"
        + _FEWSHOT_FRAMING
        + "\n\n"
        + "\n\n".join(examples)
    )


def _hermes_manifest_body(
    tools: Sequence[BoundTool],
    provider: str,
    flags: ToolCallGrammarFlags,
    *,
    include_fewshot: bool,
) -> str:
    """The Hermes/Qwen ``<tools>`` manifest body (the inner template the chat wrapper
    wraps). ``include_fewshot`` gates ONLY the flattened ``# Examples`` block — set False
    when a manager presents the exemplars as real prior turns instead (so they are not
    duplicated). The canonical body + the no-code nudge are unaffected by this flag.
    """
    from himmy.services.tools.access import describe_for_model
    from himmy.services.tools.schema_normalize import normalize_tool_schema

    lines = [_HERMES_TOOLS_PREAMBLE]
    for tool in tools:
        function = {
            "name": tool.name,
            "description": describe_for_model(
                tool.name, tool.description, tool.read_only
            ),
            "parameters": normalize_tool_schema(
                tool.args_json_schema or {"type": "object"}, provider
            ),
        }
        # Compact separators match the local template's single-line per-tool object;
        # ensure_ascii=False keeps non-ASCII (e.g. Nepali tool text) literal on the
        # wire instead of \uXXXX-escaping it, matching the HF/Ollama template output.
        lines.append(
            json.dumps(
                {"type": "function", "function": function},
                separators=(", ", ": "),
                ensure_ascii=False,
            )
            + "\n"
        )
    lines.append(_HERMES_TOOLS_INSTRUCTION)
    lines.append(_HERMES_TOOLS_GUIDANCE)
    if flags.no_code_instruction:
        lines.append(_HERMES_NO_CODE_INSTRUCTION)
    if include_fewshot:
        lines.append(_render_fewshot_block(tools, flags))
    return "".join(lines)


def _make_hermes_render_system_manifest(flags: ToolCallGrammarFlags) -> ManifestRenderer:
    """Build the Hermes/Qwen ``<tools>`` manifest renderer bound to ``flags``.

    The ``<|im_start|>system`` / ``<|im_end|>`` ChatML envelope is added by the chat
    wrapper; this emits the inner body the model template expects. Each tool is one
    line ``{"type": "function", "function": {name, description, parameters}}`` where
    ``parameters`` is the schema run through :func:`normalize_tool_schema` per tool
    (so a text-only model sees the same ``$ref``-inlined, nullable-collapsed shape
    every other backend gets) and ``description`` carries himmy's reader/writer intent
    hint (:func:`describe_for_model`) for tool-disambiguation consistency.

    When ``flags.fewshot``/``flags.no_code_instruction`` are set the manifest gains the
    synthesized exemplars + the "output only <tool_call>" nudge — APPENDED after the
    canonical body so the default (fewshot=0, no_code=False) manifest is byte-identical.
    """

    def _render(tools: Sequence[BoundTool], provider: str) -> str:
        return _hermes_manifest_body(tools, provider, flags, include_fewshot=True)

    return _render


def _make_hermes_render_system_manifest_base(
    flags: ToolCallGrammarFlags,
) -> ManifestRenderer:
    """Build the manifest renderer that OMITS the flattened ``# Examples`` block.

    A manager composing few-shot as REAL turns uses this for the system block so the
    exemplars are not duplicated (once as turns, once flattened). The no-code nudge and
    canonical body are identical to :func:`_make_hermes_render_system_manifest`.
    """

    def _render(tools: Sequence[BoundTool], provider: str) -> str:
        return _hermes_manifest_body(tools, provider, flags, include_fewshot=False)

    return _render


#: The canonical (fewshot=0, no-nudge) Hermes manifest renderer — kept as a module-level
#: name so the GENERIC byte-parity wrappers and existing imports stay stable.
_hermes_render_system_manifest = _make_hermes_render_system_manifest(
    ToolCallGrammarFlags()
)


def _hermes_native_calls(
    text: str, known: set[str], flags: ToolCallGrammarFlags
) -> list[ToolCallRecord]:
    """Extract every ``<tool_call>{json}</tool_call>`` block into ToolCallRecords.

    Order-insensitive on the JSON keys (``name``/``arguments`` in either position).
    A malformed JSON block or one missing the name key is skipped, never fatal.
    """
    from himmy.core.ids import new_uuid
    from himmy.services.inference.models import ToolCallRecord

    calls: list[ToolCallRecord] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get(flags.name_key)
        if not isinstance(name, str) or not name:
            continue
        args: Any = obj.get(flags.arg_key, {})
        if isinstance(args, str):  # arguments delivered as a JSON string
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(
            ToolCallRecord(tool_call_id=new_uuid(), tool_name=name, args=args)
        )
    return calls


def _dedup_key(name: str, args: dict[str, Any]) -> str:
    """The de-dup key matching ``parse_text_tool_calls`` (name + sorted-args json)."""
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _make_hermes_parse(flags: ToolCallGrammarFlags) -> ReplyParser:
    """Build the FAIL-OPEN Hermes/Qwen parser bound to ``flags``.

    Native ``<tool_call>`` blocks are extracted first, then
    :func:`parse_text_tool_calls` is OR-ed in as a secondary pass and any call it
    finds whose ``(name, args)`` de-dup key is not already present is appended. When
    ``flags.python_call_fallback`` is set, :func:`parse_python_tool_calls` is OR-ed in
    as a FINAL pass too (recovers a call the model wrote as python syntax). Every pass
    can only ADD hits the prior passes missed — never replace or drop one — so recall
    strictly improves and the generic path is never regressed. Any unexpected error in
    a pass is swallowed so the earlier passes' result still stands.
    """

    def _parse(text: str, known: set[str]) -> list[ToolCallRecord]:
        from himmy.services.inference.tool_protocol import (
            parse_python_tool_calls,
            parse_text_tool_calls,
        )

        try:
            calls = _hermes_native_calls(text, known, flags)
        except Exception:  # noqa: BLE001 - native pass is best-effort, never fatal
            calls = []
        seen = {_dedup_key(c.tool_name, c.args) for c in calls}

        def _merge(extra: list[ToolCallRecord]) -> None:
            for call in extra:
                key = _dedup_key(call.tool_name, call.args)
                if key not in seen:
                    seen.add(key)
                    calls.append(call)

        # OR-in the tolerant generic parser (TOOL_CALL markers / fenced / bare JSON).
        try:
            _merge(parse_text_tool_calls(text, known))
        except Exception:  # noqa: BLE001 - keep whatever native already found
            pass
        # OR-in the python-call fallback LAST (opt-in, add-only, name-guarded).
        if flags.python_call_fallback:
            try:
                _merge(parse_python_tool_calls(text, known))
            except Exception:  # noqa: BLE001 - fallback is best-effort, never fatal
                pass
        return calls

    return _parse


def _make_hermes_render_tool_results(flags: ToolCallGrammarFlags) -> ResultRenderer:
    """Build the Hermes/Qwen ``<tool_response>`` result renderer bound to ``flags``.

    Each result is wrapped ``\\n<tool_response>\\n{content}\\n</tool_response>``.
    When ``batch_consecutive_results`` is set (HF-canonical Qwen2.5), all the results
    share ONE ``<|im_start|>{role}`` ... ``<|im_end|>`` turn; otherwise each result
    gets its own turn (matches the local Ollama template's per-message wrapping).
    """

    def _render(results: Sequence[ToolReturnRecord]) -> str:
        if not results:
            return ""
        bodies = [
            f"\n<tool_response>\n{r.content}\n</tool_response>" for r in results
        ]
        open_tag = f"<|im_start|>{flags.result_role}"
        if flags.batch_consecutive_results:
            return f"{open_tag}{''.join(bodies)}<|im_end|>\n"
        return "".join(f"{open_tag}{body}<|im_end|>\n" for body in bodies)

    return _render


#: The shared Hermes/Qwen ChatML-XML divergence flags.
_HERMES_FLAGS = ToolCallGrammarFlags(
    arg_key="arguments",
    name_key="name",
    result_role="user",
    batch_consecutive_results=True,
    parallel_supported=True,
    use_text_tool_path=True,
)

HERMES_CHATML_XML = ToolCallFormat(
    name="hermes_chatml_xml",
    render_system_manifest=_make_hermes_render_system_manifest(_HERMES_FLAGS),
    parse=_make_hermes_parse(_HERMES_FLAGS),
    render_tool_results=_make_hermes_render_tool_results(_HERMES_FLAGS),
    # Auto-select for Qwen2.5-Instruct + the Hermes families. ``qwen2.5-coder`` is
    # EXCLUDED below in selection (it does not use this grammar reliably).
    model_tags=frozenset({"qwen2.5", "qwen2_5", "hermes"}),
    exclude_tags=frozenset({"coder"}),
    flags=_HERMES_FLAGS,
)


# --------------------------------------------------------------------------- #
# HERMES_CHATML_XML_FEWSHOT — same wire grammar, but the manifest carries 2
# synthesized few-shot exemplars + the "output only <tool_call>, no code" nudge.
#
# This is the format that makes a SMALL open-weight model (HimalayaGPT-0.5b, nanochat
# arch) actually emit <tool_call> blocks: the 0.5b does not do so zero-shot — it
# answers in prose, loops, or writes ```python```. Per the NousResearch / Himalaya AI
# recipe the FEW-SHOT exemplar (DEFAULT_FEWSHOT=2) is the trigger. It shares the Hermes
# parser/result-renderer byte-for-byte; only the manifest differs (data-only flags), so
# the canonical HERMES_CHATML_XML and GENERIC paths are completely unaffected. It does
# NOT auto-select by model tag — a manager opts in via ``tool_call_format`` (the
# HimalayaGPT manager defaults to it), keeping larger Hermes/Qwen users on the lean
# zero-shot manifest.
# --------------------------------------------------------------------------- #
_HERMES_FEWSHOT_FLAGS = ToolCallGrammarFlags(
    arg_key="arguments",
    name_key="name",
    result_role="user",
    batch_consecutive_results=True,
    parallel_supported=True,
    use_text_tool_path=True,
    fewshot=2,
    no_code_instruction=True,
    # The 0.5b frequently emits the call as python (a ```python``` block or a bare
    # get_weather(city=...) line) which the <tool_call>/JSON parsers miss — recover it.
    python_call_fallback=True,
)

def _fewshot_turns_renderer(tools: Sequence[BoundTool]) -> list[tuple[str, str]]:
    """Bind :func:`render_fewshot_turns` to the few-shot flags (FewshotTurnRenderer)."""
    return render_fewshot_turns(tools, _HERMES_FEWSHOT_FLAGS)


HERMES_CHATML_XML_FEWSHOT = ToolCallFormat(
    name="hermes_chatml_xml_fewshot",
    render_system_manifest=_make_hermes_render_system_manifest(_HERMES_FEWSHOT_FLAGS),
    parse=_make_hermes_parse(_HERMES_FEWSHOT_FLAGS),
    render_tool_results=_make_hermes_render_tool_results(_HERMES_FEWSHOT_FLAGS),
    # Opt-in only (no auto-select tags): a manager pins it via tool_call_format.
    model_tags=frozenset(),
    flags=_HERMES_FEWSHOT_FLAGS,
    # Multi-turn few-shot: a manager that supports it (HimalayaGptClientManager) renders
    # these exemplars as REAL prior turns + uses the base (no flattened # Examples)
    # manifest. A single-turn caller (Ollama on this format) ignores both and gets the
    # flattened in-manifest exemplars exactly as before.
    fewshot_turns=_fewshot_turns_renderer,
    render_system_manifest_base=_make_hermes_render_system_manifest_base(
        _HERMES_FEWSHOT_FLAGS
    ),
)


# --------------------------------------------------------------------------- #
# HERMES_CHATML_XML_FEWSHOT_BASELINE — the PRE-BOOST few-shot format, reconstructed.
#
# This format reproduces main's ORIGINAL ``hermes_chatml_xml_fewshot`` behavior
# byte-for-byte so the A/B's BASELINE arm is the genuine pre-boost recipe and the only
# difference vs the boosted format is the three+one prompting boosts:
#   * exemplar fills ONLY the FIRST property (original ``_example_args_for`` w/ ``break``),
#     so a 2-arg tool's exemplar can never demonstrate both args (the ~50% ceiling);
#   * placeholders are the non-distinct, parrotable originals (``"example"`` / ``1``),
#     with NO anti-parrot framing line and NO multi-arg swap;
#   * single-turn ONLY (no ``fewshot_turns`` / no ``render_system_manifest_base``);
#   * the canonical Hermes parser (NO ``python_call_fallback``).
# It keeps ``fewshot=2`` + ``no_code_instruction`` (those existed pre-boost too), so the
# boost is the SOLE independent variable. Opt-in only (no auto-select tags); used by the
# A/B harness as the baseline arm. The shipping default stays the boosted format.
# --------------------------------------------------------------------------- #
_HERMES_FEWSHOT_BASELINE_FLAGS = ToolCallGrammarFlags(
    arg_key="arguments",
    name_key="name",
    result_role="user",
    batch_consecutive_results=True,
    parallel_supported=True,
    use_text_tool_path=True,
    fewshot=2,
    no_code_instruction=True,
    # No python_call_fallback, no multi-arg-aware exemplar tooling: PRE-BOOST behavior.
    python_call_fallback=False,
)

#: The original (pre-boost) placeholder for a string slot — a parrotable literal.
_BASELINE_PLACEHOLDER_STRING = "example"


def _baseline_example_args_for(tool: BoundTool) -> dict[str, Any]:
    """Pre-boost exemplar args: fill ONLY the FIRST declared property (main's behavior).

    Mirrors main's original ``_example_args_for``: it filled a single property (the first
    one, then ``break``-ed) with a non-distinct placeholder. A 2-arg tool's exemplar thus
    shows only ONE argument, so the exemplar can never satisfy the tool's full arg
    contract — the documented ~50% ceiling this baseline exists to demonstrate.
    """
    schema = tool.args_json_schema or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    example_args: dict[str, Any] = {}
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t == "integer":
            example_args[key] = 1
        elif t == "number":
            example_args[key] = 1.0
        elif t == "boolean":
            example_args[key] = True
        elif t == "array":
            example_args[key] = []
        elif t == "object":
            example_args[key] = {}
        else:  # string / unknown
            example_args[key] = _BASELINE_PLACEHOLDER_STRING
        break  # one filled property is enough to teach the envelope (pre-boost)
    return example_args


def _baseline_fewshot_tools(
    tools: Sequence[BoundTool], flags: ToolCallGrammarFlags
) -> list[BoundTool]:
    """Pre-boost exemplar tool pick: rotate the lead over the first N (no multi-arg swap)."""
    if flags.fewshot <= 0 or not tools:
        return []
    n = min(flags.fewshot, max(1, len(tools)))
    return [tools[i % len(tools)] for i in range(n)]


def _baseline_synthesize_example(tool: BoundTool, flags: ToolCallGrammarFlags) -> str:
    """One pre-boost flattened exemplar (first-prop-only args, no anti-parrot framing)."""
    call = {flags.name_key: tool.name, flags.arg_key: _baseline_example_args_for(tool)}
    call_json = json.dumps(call, separators=(", ", ": "), ensure_ascii=False)
    return (
        f"Example — user: Use the {tool.name} tool.\n"
        f"Example — assistant:\n<tool_call>\n{call_json}\n</tool_call>"
    )


def _baseline_render_fewshot_block(
    tools: Sequence[BoundTool], flags: ToolCallGrammarFlags
) -> str:
    """The pre-boost ``# Examples`` block: NO framing line, first-prop-only exemplars."""
    chosen = _baseline_fewshot_tools(tools, flags)
    if not chosen:
        return ""
    examples = [_baseline_synthesize_example(t, flags) for t in chosen]
    return "\n\n# Examples\n\n" + "\n\n".join(examples)


def _baseline_manifest_body(
    tools: Sequence[BoundTool], provider: str, flags: ToolCallGrammarFlags
) -> str:
    """The pre-boost Hermes manifest body: canonical body + no-code nudge + pre-boost
    flattened ``# Examples`` block (no anti-parrot framing, first-prop-only exemplars)."""
    from himmy.services.tools.access import describe_for_model
    from himmy.services.tools.schema_normalize import normalize_tool_schema

    lines = [_HERMES_TOOLS_PREAMBLE]
    for tool in tools:
        function = {
            "name": tool.name,
            "description": describe_for_model(
                tool.name, tool.description, tool.read_only
            ),
            "parameters": normalize_tool_schema(
                tool.args_json_schema or {"type": "object"}, provider
            ),
        }
        lines.append(
            json.dumps(
                {"type": "function", "function": function},
                separators=(", ", ": "),
                ensure_ascii=False,
            )
            + "\n"
        )
    lines.append(_HERMES_TOOLS_INSTRUCTION)
    if flags.no_code_instruction:
        lines.append(_HERMES_NO_CODE_INSTRUCTION)
    lines.append(_baseline_render_fewshot_block(tools, flags))
    return "".join(lines)


def _baseline_render_system_manifest(tools: Sequence[BoundTool], provider: str) -> str:
    return _baseline_manifest_body(tools, provider, _HERMES_FEWSHOT_BASELINE_FLAGS)


HERMES_CHATML_XML_FEWSHOT_BASELINE = ToolCallFormat(
    name="hermes_chatml_xml_fewshot_baseline",
    render_system_manifest=_baseline_render_system_manifest,
    # Canonical Hermes parser (NO python fallback) — pre-boost recall.
    parse=_make_hermes_parse(_HERMES_FEWSHOT_BASELINE_FLAGS),
    render_tool_results=_make_hermes_render_tool_results(_HERMES_FEWSHOT_BASELINE_FLAGS),
    model_tags=frozenset(),  # opt-in only — never auto-selected.
    flags=_HERMES_FEWSHOT_BASELINE_FLAGS,
    # Single-turn ONLY: no fewshot_turns, no base manifest -> the manager renders the
    # flattened in-manifest exemplars exactly as main did pre-boost.
)


# --------------------------------------------------------------------------- #
# Shared OpenAI-style tool-schema serialization for the JSON-list family.
#
# Llama-3.1 and Mistral both advertise tools as a JSON LIST of OpenAI-style
# ``{"type":"function","function":{name, description, parameters}}`` objects (NOT the
# per-line ``<tools>`` block Hermes uses). The single helper below builds that list of
# dicts once (normalized + read/write-hint-tagged, exactly like the Hermes body) so both
# families render from one source of truth — no ``if model ==`` and no duplicated schema
# plumbing. Each family then frames the list differently (Llama: a first-user-message
# manifest sentence; Mistral: an ``[AVAILABLE_TOOLS]`` envelope).
# --------------------------------------------------------------------------- #
def _openai_function_specs(
    tools: Sequence[BoundTool], provider: str
) -> list[dict[str, Any]]:
    """Build the list of OpenAI-style ``{"type":"function","function":{...}}`` dicts.

    Mirrors the Hermes manifest body's per-tool object EXACTLY (same
    :func:`describe_for_model` read/write hint, same per-tool
    :func:`normalize_tool_schema` of ``parameters``) so the only difference between the
    families is the envelope/framing, never the advertised schema.
    """
    from himmy.services.tools.access import describe_for_model
    from himmy.services.tools.schema_normalize import normalize_tool_schema

    specs: list[dict[str, Any]] = []
    for tool in tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": describe_for_model(
                        tool.name, tool.description, tool.read_only
                    ),
                    "parameters": normalize_tool_schema(
                        tool.args_json_schema or {"type": "object"}, provider
                    ),
                },
            }
        )
    return specs


# --------------------------------------------------------------------------- #
# LLAMA3_JSON — Llama-3.1 / Llama-3.2 / Llama-3.3 Instruct (JSON-function calling).
#
# Llama-3.1's "JSON based tool calling" convention (Meta's model card + the HF llama-3.1
# chat template):
#   * MANIFEST: the tool schemas are advertised as a JSON LIST in the FIRST USER message,
#     prefixed by Meta's literal instruction. The model card's wording is
#     "Given the following functions, please respond with a JSON for a function call with
#      its proper arguments that best answers the given prompt." — reproduced verbatim.
#   * CALL: a BARE single JSON object ``{"name": ..., "parameters": {...}}`` — note the
#     arg key is ``parameters`` (NOT ``arguments``), and there is no XML wrapper.
#   * RESULT: tool output is fed back in an ``ipython``-role turn
#     (``<|start_header_id|>ipython<|end_header_id|>`` ... ``<|eot_id|>``), the Llama-3.1
#     role reserved for tool/function responses.
# The parser is FAIL-OPEN: the bare-JSON ``{"name","parameters"}`` pass is OR-ed with the
# generic tolerant pass AND a ``<tool_call>``-style pass, so a Llama that drifts onto the
# Hermes envelope (some fine-tunes do) is still recovered. ``parameters``-keyed args are
# read by setting ``flags.arg_key="parameters"``.
# --------------------------------------------------------------------------- #

#: Meta's literal first-user-message manifest instruction (verbatim from the Llama-3.1
#: model card / HF template). ``{tools_json}`` is the JSON list of function specs.
_LLAMA3_MANIFEST_PREAMBLE = (
    "You have access to the following functions. To call a function, respond with a "
    "JSON for a function call with its proper arguments that best answers the given "
    "prompt.\n\n"
    "Respond in the format {\"name\": function name, \"parameters\": dictionary of "
    "argument name and its value}. Do not use variables.\n\n"
)

#: General format-native guidance appended AFTER the JSON spec list (outside the array).
#: Llama's native parallel form is a JSON list of such call objects, and a non-call answer
#: is plain text. Both are documented Llama-3.1 behaviours, not benchmark-specific syntax.
_LLAMA3_MANIFEST_GUIDANCE = (
    "\n\n"
    "If multiple independent calls are needed, respond with a JSON list of such "
    "objects. If none of the functions are relevant, respond in plain text instead "
    "of a function call."
)


def _llama3_render_system_manifest(tools: Sequence[BoundTool], provider: str) -> str:
    """Render the Llama-3.1 JSON-function manifest (preamble + JSON list of specs).

    The manager injects this as the leading system/first-user block. The function
    specs are the shared OpenAI-style list (one source of truth with Mistral); the
    JSON list is pretty-printed with 4-space indent the way the Llama template shows it,
    ``ensure_ascii=False`` so Nepali tool text stays literal on the wire.
    """
    specs = _openai_function_specs(tools, provider)
    tools_json = json.dumps(specs, indent=4, ensure_ascii=False)
    return f"{_LLAMA3_MANIFEST_PREAMBLE}{tools_json}{_LLAMA3_MANIFEST_GUIDANCE}"


#: Match a BARE ``{"name": ..., "parameters": {...}}`` JSON object anywhere in the reply.
#: Non-greedy + DOTALL on the body; the real JSON shape is validated by ``json.loads`` in
#: the parser (this regex only finds candidate object spans, balanced via the decoder).
#: We anchor on a ``{`` that is followed (in any key order) by a ``"name"`` key so a plain
#: structured-output JSON answer that happens to be an object is not mis-read as a call.
_LLAMA3_BARE_CALL_HINT = re.compile(r'"name"\s*:', re.DOTALL)


def _llama3_native_calls(
    text: str, known: set[str], flags: ToolCallGrammarFlags
) -> list[ToolCallRecord]:
    """Extract bare ``{"name": ..., "parameters": {...}}`` call object(s) from the reply.

    Llama-3.1 emits the call as a bare JSON object (no XML/markers). It may emit ONE
    object, or a JSON ARRAY of objects for parallel calls. We scan every balanced JSON
    value in the text (via the shared :func:`_iter_json_objects`), accept dicts that
    carry the name key, and read args from ``flags.arg_key`` (``parameters``). A name is
    accepted only when it is a plausible bound tool (``_name_is_plausible``) so a generic
    JSON answer is never mis-parsed as a call. Malformed spans are skipped, never fatal.
    """
    from himmy.core.ids import new_uuid
    from himmy.services.inference.models import ToolCallRecord
    from himmy.services.inference.tool_protocol import (
        _iter_json_objects,
        _name_is_plausible,
    )

    if not _LLAMA3_BARE_CALL_HINT.search(text):
        return []  # no object carrying a "name" key — nothing bare to parse
    calls: list[ToolCallRecord] = []
    for obj in _iter_json_objects(text):
        candidates = obj if isinstance(obj, list) else [obj]
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            name = cand.get(flags.name_key)
            if not isinstance(name, str) or not name:
                continue
            if not _name_is_plausible(name, known):
                continue
            args: Any = cand.get(flags.arg_key, {})
            if isinstance(args, str):  # parameters delivered as a JSON string
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCallRecord(tool_call_id=new_uuid(), tool_name=name, args=args)
            )
    return calls


def _make_llama3_parse(flags: ToolCallGrammarFlags) -> ReplyParser:
    """Build the FAIL-OPEN Llama-3.1 parser bound to ``flags``.

    Pass order (each pass can only ADD a ``(name, args)`` the prior passes missed):
      1. bare ``{"name","parameters"}`` object(s) — the native Llama-3.1 shape;
      2. :func:`parse_text_tool_calls` — the generic tolerant pass (TOOL_CALL markers /
         fenced blocks / the ``{"name","arguments"}`` shape some Llama fine-tunes use);
      3. :func:`_hermes_native_calls` — a ``<tool_call>`` envelope a Hermes-flavored
         Llama fine-tune might emit (read with the SAME ``parameters``/``name`` keys).
    Every pass is wrapped so an unexpected error leaves the earlier passes' result.
    """

    def _parse(text: str, known: set[str]) -> list[ToolCallRecord]:
        from himmy.services.inference.tool_protocol import parse_text_tool_calls

        try:
            calls = _llama3_native_calls(text, known, flags)
        except Exception:  # noqa: BLE001 - native pass is best-effort, never fatal
            calls = []
        seen = {_dedup_key(c.tool_name, c.args) for c in calls}

        def _merge(extra: list[ToolCallRecord]) -> None:
            for call in extra:
                key = _dedup_key(call.tool_name, call.args)
                if key not in seen:
                    seen.add(key)
                    calls.append(call)

        try:
            _merge(parse_text_tool_calls(text, known))
        except Exception:  # noqa: BLE001 - keep whatever native already found
            pass
        # OR-in a Hermes-style <tool_call> envelope (read with name/parameters keys).
        try:
            _merge(_hermes_native_calls(text, known, flags))
        except Exception:  # noqa: BLE001 - fallback is best-effort, never fatal
            pass
        return calls

    return _parse


def _make_llama3_render_tool_results(flags: ToolCallGrammarFlags) -> ResultRenderer:
    """Build the Llama-3.1 ``ipython``-role result renderer bound to ``flags``.

    Tool output is fed back in the ``ipython`` role Llama-3.1 reserves for function
    responses: ``<|start_header_id|>ipython<|end_header_id|>\\n\\n{content}<|eot_id|>``.
    When ``batch_consecutive_results`` is set the results share ONE ipython turn (joined
    by blank lines); otherwise each result gets its own turn. Empty -> "" (no-op).
    """

    def _render(results: Sequence[ToolReturnRecord]) -> str:
        if not results:
            return ""
        header = f"<|start_header_id|>{flags.result_role}<|end_header_id|>\n\n"
        if flags.batch_consecutive_results:
            body = "\n\n".join(str(r.content) for r in results)
            return f"{header}{body}<|eot_id|>"
        return "".join(
            f"{header}{r.content}<|eot_id|>" for r in results
        )

    return _render


#: Llama-3.1 wire-divergence flags: BARE ``{"name","parameters"}`` calls (arg_key=
#: ``parameters``), tool results in the ``ipython`` role, parallel = a JSON array.
_LLAMA3_FLAGS = ToolCallGrammarFlags(
    arg_key="parameters",
    name_key="name",
    result_role="ipython",
    batch_consecutive_results=True,
    parallel_supported=True,
    use_text_tool_path=True,
)


def _make_llama3_render_system_manifest(
    flags: ToolCallGrammarFlags,
) -> ManifestRenderer:
    """Build the Llama-3.1 manifest renderer (kept arity-compatible with the registry)."""

    def _render(tools: Sequence[BoundTool], provider: str) -> str:
        return _llama3_render_system_manifest(tools, provider)

    return _render


LLAMA3_JSON = ToolCallFormat(
    name="llama3_json",
    render_system_manifest=_make_llama3_render_system_manifest(_LLAMA3_FLAGS),
    parse=_make_llama3_parse(_LLAMA3_FLAGS),
    render_tool_results=_make_llama3_render_tool_results(_LLAMA3_FLAGS),
    # Auto-select for Llama-3.x instruct tags (``llama3`` / ``llama-3`` / ``llama_3``).
    # ``code``/``guard`` variants are EXCLUDED (Code Llama / Llama Guard do not use this
    # JSON-function grammar); a non-instruct base ``llama`` (no "3") never matches.
    model_tags=frozenset({"llama3", "llama-3", "llama_3", "llama 3"}),
    exclude_tags=frozenset({"code", "guard", "vision"}),
    flags=_LLAMA3_FLAGS,
)


# --------------------------------------------------------------------------- #
# MISTRAL_V3 — Mistral / Mixtral / Mistral-Nemo Instruct (v3 tokenizer tool calling).
#
# Mistral's v3 (and v3-tekken) instruct tool-calling convention (mistral-common /
# the HF mistralai chat template):
#   * MANIFEST: tools are advertised inside ``[AVAILABLE_TOOLS]`` ... ``[/AVAILABLE_TOOLS]``
#     as a JSON LIST of OpenAI-style function specs, placed NEAR THE LAST user turn (the
#     manager injects the manifest as the leading text block; the envelope is what marks
#     it as Mistral's available-tools section).
#   * CALL: ``[TOOL_CALLS]`` followed by a JSON ARRAY of ``{"name", "arguments"}`` objects
#     — natively PARALLEL (one array, N entries; arg key is ``arguments``).
#   * RESULT: ``[TOOL_RESULTS]`` ... ``[/TOOL_RESULTS]`` carrying ``{"call_id", "content"}``.
#     Mistral pairs a result to its call by a 9-char ``[a-zA-Z0-9]`` ``call_id`` (a.k.a.
#     ``tool_call_id``). That 9-char id is a WIRE artifact ONLY: himmy's own UUID stays on
#     the record; the 9-char id is MINTED AT RENDER TIME (deterministically from the UUID)
#     so a call/result pair lines up within one serialization and the himmy UUID never
#     leaves the record.
# The parser is FAIL-OPEN: the ``[TOOL_CALLS]`` array pass is OR-ed with the generic
# tolerant pass and a ``<tool_call>`` pass.
# --------------------------------------------------------------------------- #

#: The Mistral v3 9-char tool-call id alphabet (mistral-common: ``[a-zA-Z0-9]{9}``).
_MISTRAL_ID_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_MISTRAL_ID_LEN = 9


def mint_mistral_tool_call_id(himmy_id: str) -> str:
    """Mint Mistral's 9-char ``[a-zA-Z0-9]`` wire id DETERMINISTICALLY from a himmy id.

    Mistral requires a 9-char alphanumeric ``tool_call_id`` to pair a ``[TOOL_RESULTS]``
    block back to its ``[TOOL_CALLS]`` entry. himmy's own ids are UUIDs (longer, with
    hyphens) which Mistral rejects — so the 9-char id is a RENDER-TIME-ONLY artifact: we
    derive it from the himmy id (a stable hash projected onto the 9-char alphabet) so the
    same himmy id always maps to the same wire id WITHIN a render (call ↔ result pair up)
    while the himmy UUID itself never leaves the :class:`ToolCallRecord` /
    :class:`ToolReturnRecord`. Deterministic (golden-pinnable) and collision-resistant
    enough for the handful of calls in one turn.
    """
    import hashlib

    digest = hashlib.sha256((himmy_id or "").encode("utf-8")).digest()
    # Project the digest bytes onto the 9-char alphabet (base-62-ish, fixed length).
    out: list[str] = []
    acc = int.from_bytes(digest[:12], "big")
    for _ in range(_MISTRAL_ID_LEN):
        acc, rem = divmod(acc, len(_MISTRAL_ID_ALPHABET))
        out.append(_MISTRAL_ID_ALPHABET[rem])
    return "".join(out)


#: Match the ``[TOOL_CALLS]`` marker followed by its JSON array body. DOTALL so the array
#: may span lines; the decoder validates the real (balanced) JSON shape.
_MISTRAL_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]\s*", re.DOTALL)


def _mistral_native_calls(
    text: str, known: set[str], flags: ToolCallGrammarFlags
) -> list[ToolCallRecord]:
    """Extract ``[TOOL_CALLS]`` JSON-array call(s) into ToolCallRecords.

    Mistral emits ``[TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]`` — a single
    marker then a JSON ARRAY (native parallel). We locate the marker, decode the array
    that follows it (tolerating a leading object too), and read each entry's name +
    ``flags.arg_key`` (``arguments``). A bare array with no marker is also accepted as a
    fallback. Malformed entries are skipped; never fatal.
    """
    from himmy.core.ids import new_uuid
    from himmy.services.inference.models import ToolCallRecord

    calls: list[ToolCallRecord] = []

    def _emit(entries: Any) -> None:
        items = entries if isinstance(entries, list) else [entries]
        for entry in items:
            if not isinstance(entry, dict):
                continue
            name = entry.get(flags.name_key)
            if not isinstance(name, str) or not name:
                continue
            args: Any = entry.get(flags.arg_key, {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCallRecord(tool_call_id=new_uuid(), tool_name=name, args=args)
            )

    decoder = json.JSONDecoder()
    for marker in _MISTRAL_TOOL_CALLS_RE.finditer(text):
        rest = text[marker.end() :]
        start = rest.find("[")
        obj_start = rest.find("{")
        # Prefer the array; fall back to a single object if that comes first.
        if start == -1 or (obj_start != -1 and obj_start < start):
            start = obj_start
        if start == -1:
            continue
        try:
            decoded, _ = decoder.raw_decode(rest[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        _emit(decoded)
    return calls


def _make_mistral_parse(flags: ToolCallGrammarFlags) -> ReplyParser:
    """Build the FAIL-OPEN Mistral parser bound to ``flags``.

    Pass order (add-only; each can only ADD a ``(name, args)`` the prior passes missed):
      1. ``[TOOL_CALLS]`` JSON-array pass — the native Mistral shape (parallel);
      2. :func:`parse_text_tool_calls` — the generic tolerant pass;
      3. :func:`_hermes_native_calls` — a ``<tool_call>`` envelope a Hermes-flavored
         Mistral fine-tune might emit (read with the same name/arguments keys).
    """

    def _parse(text: str, known: set[str]) -> list[ToolCallRecord]:
        from himmy.services.inference.tool_protocol import parse_text_tool_calls

        try:
            calls = _mistral_native_calls(text, known, flags)
        except Exception:  # noqa: BLE001 - native pass is best-effort, never fatal
            calls = []
        seen = {_dedup_key(c.tool_name, c.args) for c in calls}

        def _merge(extra: list[ToolCallRecord]) -> None:
            for call in extra:
                key = _dedup_key(call.tool_name, call.args)
                if key not in seen:
                    seen.add(key)
                    calls.append(call)

        try:
            _merge(parse_text_tool_calls(text, known))
        except Exception:  # noqa: BLE001 - keep whatever native already found
            pass
        try:
            _merge(_hermes_native_calls(text, known, flags))
        except Exception:  # noqa: BLE001 - fallback is best-effort, never fatal
            pass
        return calls

    return _parse


#: General format-native guidance appended AFTER the ``[/AVAILABLE_TOOLS]`` envelope.
#: Mistral's native parallel form is multiple entries in one ``[TOOL_CALLS]`` JSON array,
#: and a non-call answer is plain text. Documented mistral-common behaviour, not
#: benchmark-specific syntax.
_MISTRAL_MANIFEST_GUIDANCE = (
    " When multiple independent calls are needed, include them as multiple entries "
    "in a single [TOOL_CALLS] array. If no available tool fits, reply in plain text "
    "without a [TOOL_CALLS] block."
)


def _make_mistral_render_system_manifest(
    flags: ToolCallGrammarFlags,
) -> ManifestRenderer:
    """Build the Mistral ``[AVAILABLE_TOOLS]`` manifest renderer.

    The function specs are the shared OpenAI-style list (one source of truth with
    Llama-3.1), wrapped in Mistral's ``[AVAILABLE_TOOLS]`` ... ``[/AVAILABLE_TOOLS]``
    envelope. The JSON is compact (mistral-common renders the array without indentation)
    and ``ensure_ascii=False`` keeps non-ASCII literal.
    """

    def _render(tools: Sequence[BoundTool], provider: str) -> str:
        specs = _openai_function_specs(tools, provider)
        tools_json = json.dumps(specs, separators=(", ", ": "), ensure_ascii=False)
        return (
            f"[AVAILABLE_TOOLS] {tools_json} [/AVAILABLE_TOOLS]"
            f"{_MISTRAL_MANIFEST_GUIDANCE}"
        )

    return _render


def _make_mistral_render_tool_results(flags: ToolCallGrammarFlags) -> ResultRenderer:
    """Build the Mistral ``[TOOL_RESULTS]`` result renderer bound to ``flags``.

    Each result is rendered ``[TOOL_RESULTS] {"call_id": <9char>, "content": ...}
    [/TOOL_RESULTS]`` where ``call_id`` is the 9-char wire id minted AT RENDER TIME from
    the result's himmy ``tool_call_id`` (see :func:`mint_mistral_tool_call_id`) — the
    himmy UUID itself never appears on the wire. Each result is its own ``[TOOL_RESULTS]``
    block (mistral-common emits one block per result). Empty -> "" (no-op).
    """

    def _render(results: Sequence[ToolReturnRecord]) -> str:
        if not results:
            return ""
        blocks: list[str] = []
        for r in results:
            payload = {
                "call_id": mint_mistral_tool_call_id(r.tool_call_id),
                "content": r.content,
            }
            body = json.dumps(payload, separators=(", ", ": "), ensure_ascii=False)
            blocks.append(f"[TOOL_RESULTS] {body} [/TOOL_RESULTS]")
        return "".join(blocks)

    return _render


#: Mistral v3 wire-divergence flags: ``[TOOL_CALLS]`` JSON array (arg_key=``arguments``,
#: native parallel), results in ``[TOOL_RESULTS]`` blocks keyed by a 9-char wire id.
_MISTRAL_FLAGS = ToolCallGrammarFlags(
    arg_key="arguments",
    name_key="name",
    result_role="tool",
    batch_consecutive_results=False,
    parallel_supported=True,
    use_text_tool_path=True,
)


MISTRAL_V3 = ToolCallFormat(
    name="mistral_v3",
    render_system_manifest=_make_mistral_render_system_manifest(_MISTRAL_FLAGS),
    parse=_make_mistral_parse(_MISTRAL_FLAGS),
    render_tool_results=_make_mistral_render_tool_results(_MISTRAL_FLAGS),
    # Auto-select for Mistral / Mixtral / Mistral-Nemo / Ministral instruct tags.
    # ``embed`` (Mistral embedding models) and ``codestral`` (a CODE model that does not
    # use this tool grammar reliably — the coder-exclude rule, like qwen2.5-coder for
    # Hermes) are EXCLUDED. An explicit override still wins over an exclude.
    model_tags=frozenset({"mistral", "mixtral", "ministral", "magistral"}),
    exclude_tags=frozenset({"embed", "codestral"}),
    flags=_MISTRAL_FLAGS,
)


class ToolCallFormatRegistry:
    """Resolve a :class:`ToolCallFormat` from a model tag + optional override.

    Resolution order (never raises — any miss/error falls to GENERIC):

    1. ``override`` — a per-manager format name, when it names a registered format.
    2. ``model_tag`` — auto-select by matching the format's ``model_tags`` substrings.
    3. :data:`GENERIC` — the safe default that reproduces today's behavior.
    """

    def __init__(self, *, default: ToolCallFormat = GENERIC) -> None:
        self._default = default
        self._formats: dict[str, ToolCallFormat] = {}
        self.register_format(default)

    def register_format(self, fmt: ToolCallFormat) -> ToolCallFormat:
        """Register (or override) a format by name; returns it for chaining."""
        self._formats[_normalize(fmt.name)] = fmt
        return fmt

    def get_format(self, name: str | None) -> ToolCallFormat | None:
        """Look up a format by name (case-insensitive); ``None`` if unregistered."""
        return self._formats.get(_normalize(name))

    def list_formats(self) -> dict[str, ToolCallFormat]:
        """A copy of the current format table (introspection / tests)."""
        return dict(self._formats)

    def format_for(
        self, model_tag: str | None, override: str | None = None
    ) -> ToolCallFormat:
        """Resolve the format for a resolved ``model_tag`` + per-manager ``override``.

        A non-empty ``override`` that names a registered format wins. Otherwise the
        first format whose ``model_tags`` substring is contained in the normalized
        ``model_tag`` is chosen. Anything else (unknown override, no tag match, or an
        unexpected error) resolves to the GENERIC default — selection is total.
        """
        try:
            if override:
                chosen = self.get_format(override)
                if chosen is not None:
                    return chosen
            tag = _normalize(model_tag)
            if tag:
                for fmt in self._formats.values():
                    if fmt is self._default:
                        continue
                    if any(t and t in tag for t in fmt.exclude_tags):
                        continue  # vetoed (e.g. qwen2.5-coder) → keep looking / default
                    if any(t and t in tag for t in fmt.model_tags):
                        return fmt
        except Exception:  # noqa: BLE001 - selection is fail-open, never fatal
            return self._default
        return self._default


#: Process-wide registry the managers route through. GENERIC is the safe default;
#: HERMES_CHATML_XML auto-selects for Hermes / Qwen2.5-Instruct resolved tags.
_REGISTRY = ToolCallFormatRegistry()
_REGISTRY.register_format(HERMES_CHATML_XML)
_REGISTRY.register_format(HERMES_CHATML_XML_FEWSHOT)
_REGISTRY.register_format(HERMES_CHATML_XML_FEWSHOT_BASELINE)
_REGISTRY.register_format(LLAMA3_JSON)
_REGISTRY.register_format(MISTRAL_V3)


def register_format(fmt: ToolCallFormat) -> ToolCallFormat:
    """Register a format on the process-wide registry (returns it for chaining)."""
    return _REGISTRY.register_format(fmt)


def get_format(name: str | None) -> ToolCallFormat | None:
    """Look up a format by name on the process-wide registry."""
    return _REGISTRY.get_format(name)


def list_formats() -> dict[str, ToolCallFormat]:
    """A copy of the process-wide format table."""
    return _REGISTRY.list_formats()


def format_for(
    model_tag: str | None, override: str | None = None
) -> ToolCallFormat:
    """Resolve a format from the process-wide registry (defaults to GENERIC)."""
    return _REGISTRY.format_for(model_tag, override)


__all__ = [
    "ToolCallFormat",
    "ToolCallGrammarFlags",
    "ToolCallFormatRegistry",
    "GENERIC",
    "HERMES_CHATML_XML",
    "HERMES_CHATML_XML_FEWSHOT",
    "HERMES_CHATML_XML_FEWSHOT_BASELINE",
    "LLAMA3_JSON",
    "MISTRAL_V3",
    "mint_mistral_tool_call_id",
    "register_format",
    "get_format",
    "list_formats",
    "format_for",
    "render_fewshot_turns",
    "ManifestRenderer",
    "ReplyParser",
    "ResultRenderer",
    "FewshotTurnRenderer",
]
