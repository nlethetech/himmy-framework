"""Tools kernel: data shapes for tool definitions, invocations, and results."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from opensims.core.ids import new_uuid
from opensims.entities.records import stable_id_for

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from opensims.entities.records import EntityRecord


class ToolBackendKind(str, Enum):
    """Where a tool actually executes."""

    LOCAL = "LOCAL"
    HTTP = "HTTP"


class HttpAuthMode(str, Enum):
    """How an HTTP connector authenticates against its upstream.

    ``BASIC`` reads the env var as raw ``user:pass`` and base64-encodes it here;
    ``PREENCODED_BASIC`` treats the env var as an already-base64-encoded credential
    and passes it through verbatim (for secrets stored pre-encoded).
    """

    NONE = "NONE"
    BEARER = "BEARER"
    HEADER = "HEADER"
    BASIC = "BASIC"
    PREENCODED_BASIC = "PREENCODED_BASIC"


class HttpAuthConfig(BaseModel):
    """Declarative, env-backed auth for an HTTP tool.

    The secret value is never stored on the model; ``env_var`` names the
    environment variable read at execution time.
    """

    mode: HttpAuthMode = HttpAuthMode.NONE
    env_var: str | None = None
    header_name: str | None = None


class HttpToolConfig(BaseModel):
    """Declarative description of an HTTP REST connector.

    Path placeholders (``{name}``), query args, body args, and header args are
    all resolved from the invocation's ``args`` at execution time.
    """

    base_url_env_var: str
    method: str = "GET"
    path_template: str = "/"
    auth: HttpAuthConfig = HttpAuthConfig()
    query_arg_names: list[str] = []
    body_arg_names: list[str] = []
    header_arg_names: list[str] = []
    timeout_seconds: float = 15.0


class ToolDefinition(BaseModel):
    """The serializable catalog entry for one tool.

    The local Python handler is deliberately kept OUT of this model so the
    definition stays JSON-serializable and projectable to an entity record;
    the :class:`ToolRegistry` owns a separate ``name -> handler`` map.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    kind: ToolBackendKind
    description: str = ""
    args_json_schema: dict[str, Any] = {}
    output_json_schema: dict[str, Any] | None = None
    requires_approval: bool = False
    timeout_seconds: float | None = None
    retry_hints: dict[str, Any] = {}
    sequential: bool = False
    http_config: HttpToolConfig | None = None
    #: Arg keys whose values are secrets; redacted from emitted TOOL_CALLED args.
    sensitive_arg_names: list[str] = []
    metadata: dict[str, Any] = {}

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this definition into its ``EntityRecord`` (kind ``tool_definition``)."""
        from opensims.entities.records import EntityRecord

        stable_id = stable_id_for(self.name, namespace="tool_definition")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="tool_definition",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


class ToolInvocation(BaseModel):
    """A single request to execute a tool with concrete arguments."""

    tool_call_id: str = Field(default_factory=new_uuid)
    tool_name: str
    args: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class ToolPolicyDecision(BaseModel):
    """The verdict returned by a pre-execution policy hook."""

    allow: bool = True
    reason: str | None = None
    transformed_args: dict[str, Any] | None = None


class ToolErrorCode(str, Enum):
    """Normalized failure codes surfaced on a tool execution result."""

    NOT_FOUND = "NOT_FOUND"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    UNKNOWN = "UNKNOWN"


class ToolExecutionResult(BaseModel):
    """The outcome of one tool execution (success, failure, or denial)."""

    tool_call_id: str
    tool_name: str
    outcome: str
    result: Any = None
    error_code: ToolErrorCode | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = {}
