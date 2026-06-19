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


def _hermes_render_system_manifest(tools: Sequence[BoundTool], provider: str) -> str:
    """Render the Hermes/Qwen ``<tools>`` manifest body (system-prompt text).

    The ``<|im_start|>system`` / ``<|im_end|>`` ChatML envelope is added by the chat
    wrapper; this emits the inner body the model template expects. Each tool is one
    line ``{"type": "function", "function": {name, description, parameters}}`` where
    ``parameters`` is the schema run through :func:`normalize_tool_schema` per tool
    (so a text-only model sees the same ``$ref``-inlined, nullable-collapsed shape
    every other backend gets) and ``description`` carries himmy's reader/writer intent
    hint (:func:`describe_for_model`) for tool-disambiguation consistency.
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
    return "".join(lines)


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
    finds whose ``(name, args)`` de-dup key is not already present is appended. The
    native pass can therefore only ADD hits the generic parser misses — never
    replace or drop one — so recall strictly improves and the generic path is
    never regressed. Any unexpected error falls back to the generic-only result.
    """

    def _parse(text: str, known: set[str]) -> list[ToolCallRecord]:
        from himmy.services.inference.tool_protocol import parse_text_tool_calls

        try:
            calls = _hermes_native_calls(text, known, flags)
        except Exception:  # noqa: BLE001 - native pass is best-effort, never fatal
            calls = []
        seen = {_dedup_key(c.tool_name, c.args) for c in calls}
        # OR-in the tolerant generic parser (TOOL_CALL markers / fenced / bare JSON).
        try:
            generic = parse_text_tool_calls(text, known)
        except Exception:  # noqa: BLE001 - keep whatever native already found
            generic = []
        for call in generic:
            key = _dedup_key(call.tool_name, call.args)
            if key not in seen:
                seen.add(key)
                calls.append(call)
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
    render_system_manifest=_hermes_render_system_manifest,
    parse=_make_hermes_parse(_HERMES_FLAGS),
    render_tool_results=_make_hermes_render_tool_results(_HERMES_FLAGS),
    # Auto-select for Qwen2.5-Instruct + the Hermes families. ``qwen2.5-coder`` is
    # EXCLUDED below in selection (it does not use this grammar reliably).
    model_tags=frozenset({"qwen2.5", "qwen2_5", "hermes"}),
    exclude_tags=frozenset({"coder"}),
    flags=_HERMES_FLAGS,
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
    "register_format",
    "get_format",
    "list_formats",
    "format_for",
    "ManifestRenderer",
    "ReplyParser",
    "ResultRenderer",
]
