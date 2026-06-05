"""Inference kernel: typed request/response envelopes, configs, and schema synthesis.

These are the provider-agnostic data types that flow through ``InferenceService``.
The offline ``StubClientManager`` relies on :func:`synthesize_from_schema` to make
structured output and tool calling work without a model or network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from himmy.core.ids import new_uuid


class InferenceStatus(str, Enum):
    """Terminal status of a single inference attempt."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ResponseFormat(str, Enum):
    """The explicit lever for the shape of a model response.

    ``TOOL`` forces the model to call one named tool: the inference service rewrites it
    into a single-tool ``AUTO_TOOLS`` call (bind only the forced tool + instruct it),
    so it works uniformly across providers. The forced tool is ``tool_names_override[0]``
    when set, else the first bound tool.
    """

    TEXT = "TEXT"
    JSON_OBJECT = "JSON_OBJECT"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"
    AUTO_TOOLS = "AUTO_TOOLS"
    WORKFLOW = "WORKFLOW"
    TOOL = "TOOL"


class InferenceErrorCode(str, Enum):
    """Normalized, provider-agnostic error taxonomy."""

    AUTH = "AUTH"
    QUOTA = "QUOTA"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


#: Error codes the service is permitted to retry (transient failures only).
RETRYABLE_ERROR_CODES: frozenset[InferenceErrorCode] = frozenset(
    {
        InferenceErrorCode.RATE_LIMITED,
        InferenceErrorCode.TIMEOUT,
        InferenceErrorCode.PROVIDER_UNAVAILABLE,
    }
)


class InferenceError(BaseModel):
    """A normalized inference failure with an explicit ``retryable`` flag."""

    code: InferenceErrorCode
    message: str
    retryable: bool = False


class InferenceMessage(BaseModel):
    """One message in the conversation handed to the model."""

    role: str  # system | user | assistant | tool
    content: str = ""
    metadata: dict[str, Any] = {}
    tool_call_id: str | None = None
    name: str | None = None

    def to_model_message(self) -> dict[str, Any]:
        """Return a plain-dict representation for the pydantic-ai message rebuild.

        The real provider path reconstructs ``pydantic_ai.messages`` objects from
        these fields (see :meth:`PydanticAIClientManager._to_message_history`);
        here we return an import-safe dict so the core path never depends on
        pydantic-ai being installed. Kind is normalized so the adapter can route
        each role to the correct ``ModelRequest``/``ModelResponse`` part.
        """
        role = (self.role or "user").lower()
        kind = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
        }.get(role, "user")
        return {
            "role": role,
            "kind": kind,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": self.metadata,
        }


class ToolCallRecord(BaseModel):
    """A single tool invocation the model requested, paired by ``tool_call_id``."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = {}


class ToolReturnRecord(BaseModel):
    """The result a tool returned for a given ``tool_call_id``."""

    tool_call_id: str
    tool_name: str
    content: Any = None
    outcome: str = "success"
    metadata: dict[str, Any] = {}


class BoundTool(BaseModel):
    """Internal tool binding consumed by the offline stub path.

    The runtime/tool kernel produces these so the stub can synthesize arguments
    from ``args_json_schema`` and (optionally) execute the bound ``handler``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    args_json_schema: dict[str, Any] = {}
    output_json_schema: dict[str, Any] | None = None
    handler: Callable[[dict[str, Any]], Awaitable[ToolReturnRecord]] | None = None


class WorkflowDefinition(BaseModel):
    """An ordered list of tool names to call, one per inference step."""

    steps: list[str] = []


class WorkflowState(BaseModel):
    """A pointer into a :class:`WorkflowDefinition` for forced sequential calling."""

    definition: WorkflowDefinition
    current_step: int = 0

    @property
    def is_complete(self) -> bool:
        """True once the pointer has advanced past the final step."""
        return self.current_step >= len(self.definition.steps)

    @property
    def current_tool_name(self) -> str | None:
        """The tool name for the current step, or ``None`` when complete."""
        if self.is_complete:
            return None
        return self.definition.steps[self.current_step]

    def advance(self) -> WorkflowState:
        """Return a new state with the step pointer moved forward by one."""
        return WorkflowState(
            definition=self.definition, current_step=self.current_step + 1
        )


class InferenceRequest(BaseModel):
    """The single typed request envelope for every model call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=new_uuid)
    model_key: str = "default"
    messages: list[InferenceMessage] = []
    response_format: ResponseFormat | None = None
    output_json_schema: dict[str, Any] | None = None
    workflow: WorkflowState | None = None
    generation_params: dict[str, Any] = {}
    timeout_seconds: float = 30.0
    route_override: str | None = None
    metadata: dict[str, Any] = {}
    toolsets: list[Any] = []
    bound_tools: list[BoundTool] = []
    tool_names_override: list[str] | None = None

    @model_validator(mode="after")
    def _derive_response_format(self) -> InferenceRequest:
        """Derive ``response_format`` when omitted; reject contradictions otherwise.

        Mirrors :class:`LLMConfig` so direct request authors (batch/eval) get the
        same construction-time guardrails: an explicit format must agree with the
        related fields, and the formats that hard-require a field (WORKFLOW,
        STRUCTURED_OUTPUT) must have it so the stub never raises at runtime.
        """
        if self.response_format is None:
            if self.workflow is not None:
                self.response_format = ResponseFormat.WORKFLOW
            elif self.output_json_schema is not None:
                self.response_format = ResponseFormat.STRUCTURED_OUTPUT
            return self

        fmt = self.response_format
        # An explicit format must not contradict the related fields.
        if self.workflow is not None and fmt != ResponseFormat.WORKFLOW:
            raise ValueError(
                "response_format must be WORKFLOW when a workflow is provided"
            )
        if (
            self.output_json_schema is not None
            and fmt != ResponseFormat.STRUCTURED_OUTPUT
        ):
            raise ValueError(
                "response_format must be STRUCTURED_OUTPUT when an "
                "output_json_schema is provided"
            )
        # Formats that hard-require a field must have it (fail at construction,
        # not as a runtime HimmyError inside the manager).
        if fmt == ResponseFormat.WORKFLOW and self.workflow is None:
            raise ValueError("response_format WORKFLOW requires a workflow")
        if fmt == ResponseFormat.STRUCTURED_OUTPUT and self.output_json_schema is None:
            raise ValueError(
                "response_format STRUCTURED_OUTPUT requires an output_json_schema"
            )
        return self


class InferenceResponse(BaseModel):
    """The single typed response envelope returned for every model call."""

    request_id: str
    status: InferenceStatus
    output_text: str | None = None
    output_structured: Any = None
    tool_calls: list[ToolCallRecord] = []
    tool_returns: list[ToolReturnRecord] = []
    workflow: WorkflowState | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    model_path: str = ""
    provider_name: str = ""
    error: InferenceError | None = None
    #: Free-form observability annotations (e.g. {"cache_hit": True}). Additive.
    metadata: dict[str, Any] = {}


class LLMConfig(BaseModel):
    """Typed alternative to packing knobs into ``generation_params``.

    The runtime maps each field into the right slot on :class:`InferenceRequest`.
    Validators auto-derive ``response_format`` from related fields; conflicts
    (e.g. ``TEXT`` with an ``output_json_schema``) raise ``ValueError``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_key: str = "default"
    response_format: ResponseFormat | None = None
    output_json_schema: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout_seconds: float | None = None
    use_cache: bool | None = None
    route_override: str | None = None
    workflow: WorkflowState | None = None
    extra_params: dict[str, Any] = {}

    @model_validator(mode="after")
    def _derive_and_check_response_format(self) -> LLMConfig:
        """Auto-derive ``response_format`` and reject contradictory combinations."""
        if self.response_format is None:
            if self.workflow is not None:
                self.response_format = ResponseFormat.WORKFLOW
            elif self.output_json_schema is not None:
                self.response_format = ResponseFormat.STRUCTURED_OUTPUT
            return self

        # An explicit response_format must not contradict the related fields.
        if (
            self.workflow is not None
            and self.response_format != ResponseFormat.WORKFLOW
        ):
            raise ValueError(
                "response_format must be WORKFLOW when a workflow is provided"
            )
        if self.output_json_schema is not None and self.response_format not in (
            ResponseFormat.STRUCTURED_OUTPUT,
        ):
            raise ValueError(
                "response_format must be STRUCTURED_OUTPUT when an "
                "output_json_schema is provided"
            )
        return self


class BatchInferenceRequest(BaseModel):
    """A batch of requests run with bounded concurrency, preserving order."""

    requests: list[InferenceRequest]
    max_concurrency: int = 8


class BatchInferenceResponse(BaseModel):
    """The ordered batch result with success/failure counts and elapsed time."""

    responses: list[InferenceResponse]
    success_count: int
    failure_count: int
    elapsed_ms: float


class ModelPrice(BaseModel):
    """Per-model token pricing in USD per 1K tokens (input/output split)."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0

    def cost(self, *, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost for a usage split."""
        return (input_tokens / 1000.0) * self.input_per_1k + (
            output_tokens / 1000.0
        ) * self.output_per_1k


class GatewayModelConfig(BaseModel):
    """A single gateway-routed model entry (api format + provider model name)."""

    api_format: str
    model_name: str


class GatewayRuntimeConfig(BaseModel):
    """Routing configuration + pricing for :class:`GatewayClientManager`."""

    region: str = "us"
    base_url: str | None = None
    model_registry: dict[str, GatewayModelConfig] = {}
    #: Optional ``model_key`` (or resolved model name) -> price table. The natural
    #: home for $/token so the inference layer can fill ``InferenceResponse.cost``.
    model_prices: dict[str, ModelPrice] = {}

    def price_for(self, *, model_key: str, model_name: str | None = None) -> ModelPrice:
        """Look up a price by model_key, falling back to the resolved name, else 0."""
        if model_key in self.model_prices:
            return self.model_prices[model_key]
        if model_name and model_name in self.model_prices:
            return self.model_prices[model_name]
        return ModelPrice()


# Field-name fragments that hint a string field should carry prose seed text.
_PROSE_FIELD_HINTS: tuple[str, ...] = (
    "title",
    "summary",
    "rationale",
    "content",
    "text",
    "headline",
    "key_takeaway",
)


def _looks_like_prose(field_name: str) -> bool:
    """True when a field name suggests free-form prose worth seeding."""
    lowered = field_name.lower()
    return any(hint in lowered for hint in _PROSE_FIELD_HINTS)


def synthesize_from_schema(schema: dict[str, Any], *, seed_text: str = "") -> Any:
    """Produce a minimal VALID instance of a JSON schema (offline structured output).

    Supports ``object``/``array``/``string``/``number``/``integer``/``boolean``,
    ``enum``, ``const``, ``required``, ``default`` (validated against local
    constraints before use), ``additionalProperties`` dict-objects with
    ``minProperties``, the numeric/string/array size constraints (``minimum``,
    ``maximum``, ``minLength``, ``maxLength``, ``minItems``), and local
    ``$ref``/``$defs`` (so nested pydantic models round-trip). ``seed_text``
    (truncated) is injected into string fields whose name hints at prose, so the
    offline stub can emit a plausible, schema-valid structured result without a
    model.
    """
    defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    return _synth(schema, seed_text=seed_text, field_name="", defs=defs)


def _default_satisfies(schema: dict[str, Any], default: Any) -> bool:
    """Best-effort check that a schema ``default`` clears the local constraints.

    Only the cheap, local constraints are checked (enum/const membership and the
    obvious size/range bounds). When the default would violate one of these we
    fall through to synthesizing a fresh, constraint-honoring value instead of
    blindly returning an invalid default.
    """
    if "const" in schema and default != schema["const"]:
        return False
    enum_values = schema.get("enum")
    if enum_values is not None and default not in enum_values:
        return False
    if isinstance(default, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(default) < min_len:
            return False
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(default) > max_len:
            return False
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and default < minimum:
            return False
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and default > maximum:
            return False
    if isinstance(default, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(default) < min_items:
            return False
    return True


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``#/$defs/Name`` reference against the root ``$defs`` map."""
    name = ref.rsplit("/", 1)[-1]
    target = defs.get(name)
    return target if isinstance(target, dict) else {}


def _synth_string(schema: dict[str, Any], *, seed_text: str, field_name: str) -> str:
    """Synthesize a string that clears ``minLength``/``maxLength``."""
    if seed_text and _looks_like_prose(field_name):
        value = seed_text[:280]
    else:
        value = field_name or "value"
    min_len = schema.get("minLength")
    if isinstance(min_len, int) and len(value) < min_len:
        # Pad deterministically to clear the floor.
        value = (value + "x" * min_len)[:min_len] if value else "x" * min_len
    max_len = schema.get("maxLength")
    if isinstance(max_len, int) and len(value) > max_len:
        value = value[:max_len]
    return value


def _synth_integer(schema: dict[str, Any]) -> int:
    """Synthesize an integer honoring ``minimum``/``maximum`` (and exclusive)."""
    minimum = schema.get("minimum")
    exclusive_min = schema.get("exclusiveMinimum")
    maximum = schema.get("maximum")
    value = 0
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        value = max(
            value, int(minimum) if float(minimum).is_integer() else int(minimum) + 1
        )
    if isinstance(exclusive_min, (int, float)) and not isinstance(exclusive_min, bool):
        value = max(value, int(exclusive_min) + 1)
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        value = min(value, int(maximum))
    return value


def _synth_number(schema: dict[str, Any]) -> float:
    """Synthesize a float honoring ``minimum``/``maximum`` (and exclusive)."""
    minimum = schema.get("minimum")
    exclusive_min = schema.get("exclusiveMinimum")
    maximum = schema.get("maximum")
    value = 0.0
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        value = max(value, float(minimum))
    if isinstance(exclusive_min, (int, float)) and not isinstance(exclusive_min, bool):
        value = max(value, float(exclusive_min) + 1.0)
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        value = min(value, float(maximum))
    return value


def _synth(
    schema: Any, *, seed_text: str, field_name: str, defs: dict[str, Any]
) -> Any:
    """Recursive worker for :func:`synthesize_from_schema`."""
    if not isinstance(schema, dict):
        return None

    # Resolve a local $ref against the root $defs before anything else.
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _synth(
            _resolve_ref(ref, defs),
            seed_text=seed_text,
            field_name=field_name,
            defs=defs,
        )

    # ``const`` is the most specific constraint: it pins the only valid value.
    if "const" in schema:
        return schema["const"]

    # ``enum`` is checked BEFORE ``default`` so an enum-violating default
    # (e.g. {enum:[a,b], default:z}) never leaks an invalid value.
    enum_values = schema.get("enum")
    if enum_values:
        if "default" in schema and schema["default"] in enum_values:
            return schema["default"]
        return enum_values[0]

    # A ``default`` is honored only when it satisfies the local constraints.
    if "default" in schema and _default_satisfies(schema, schema["default"]):
        return schema["default"]

    schema_type = schema.get("type")
    # ``type`` may be a list (e.g. ["string", "null"]); pick the first non-null.
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if non_null else "null"

    # Composed schemas: synthesize from the first available subschema.
    if schema_type is None:
        for key in ("anyOf", "oneOf", "allOf"):
            options = schema.get(key)
            if options:
                return _synth(
                    options[0],
                    seed_text=seed_text,
                    field_name=field_name,
                    defs=defs,
                )
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"

    if schema_type == "object":
        properties: dict[str, Any] = schema.get("properties", {}) or {}
        required: list[str] = list(schema.get("required", []) or [])
        # Always synthesize required keys; include all properties for richness.
        keys = list(dict.fromkeys(list(properties.keys()) + required))
        result: dict[str, Any] = {}
        for key in keys:
            prop_schema = properties.get(key, {})
            result[key] = _synth(
                prop_schema, seed_text=seed_text, field_name=key, defs=defs
            )
        # Dict-typed objects (no fixed properties, free-form additionalProperties)
        # with minProperties>0 need at least one synthesized entry to be valid.
        min_props = int(schema.get("minProperties", 0) or 0)
        additional = schema.get("additionalProperties")
        if len(result) < min_props:
            value_schema = additional if isinstance(additional, dict) else {}
            i = 0
            while len(result) < min_props:
                synth_key = f"key_{i}"
                if synth_key not in result:
                    result[synth_key] = _synth(
                        value_schema,
                        seed_text=seed_text,
                        field_name=synth_key,
                        defs=defs,
                    )
                i += 1
        return result

    if schema_type == "array":
        items_schema = schema.get("items", {})
        min_items = int(schema.get("minItems", 1) or 0)
        count = max(min_items, 1)
        return [
            _synth(
                items_schema,
                seed_text=seed_text,
                field_name=field_name,
                defs=defs,
            )
            for _ in range(count)
        ]

    if schema_type == "string":
        return _synth_string(schema, seed_text=seed_text, field_name=field_name)

    if schema_type == "integer":
        return _synth_integer(schema)

    if schema_type == "number":
        return _synth_number(schema)

    if schema_type == "boolean":
        return False

    if schema_type == "null":
        return None

    # Untyped / unknown: a seeded string is the safest minimal value.
    if seed_text and _looks_like_prose(field_name):
        return seed_text[:280]
    return None
