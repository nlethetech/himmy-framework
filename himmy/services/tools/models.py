"""Tools kernel: data shapes for tool definitions, invocations, and results."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from himmy.core.ids import new_uuid

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.records import EntityRecord


class ToolBackendKind(str, Enum):
    """Where a tool actually executes."""

    LOCAL = "LOCAL"
    HTTP = "HTTP"


class HttpAuthMode(str, Enum):
    """How an HTTP connector authenticates against its upstream.

    ``BEARER`` sends ``Authorization: Bearer <secret>``. ``HEADER`` writes the secret
    verbatim to ``header_name`` (a custom API-key header). ``API_KEY_QUERY`` appends
    the secret as a query parameter named by ``query_param`` (some APIs key off the
    URL, e.g. ``?api_key=...``). ``BASIC`` reads the secret as the password and pairs
    it with ``username`` (or reads the secret as raw ``user:pass`` when no username is
    set) and base64-encodes it here; ``PREENCODED_BASIC`` treats the secret as an
    already-base64-encoded credential and passes it through verbatim.
    """

    NONE = "NONE"
    BEARER = "BEARER"
    HEADER = "HEADER"
    API_KEY_QUERY = "API_KEY_QUERY"
    BASIC = "BASIC"
    PREENCODED_BASIC = "PREENCODED_BASIC"


class HttpAuthConfig(BaseModel):
    """Declarative, secrets-backed auth for an HTTP tool.

    The secret value is never stored on the model; ``env_var`` names the credential,
    resolved at execution time through the secrets layer (``get_secret``) — so it can
    live in a vault / cloud secret manager / file, not just an environment variable.
    """

    mode: HttpAuthMode = HttpAuthMode.NONE
    env_var: str | None = None  # secret NAME resolved via get_secret (never a literal)
    header_name: str | None = None  # for HEADER mode
    query_param: str | None = None  # for API_KEY_QUERY mode
    username: str | None = None  # for BASIC mode (the password is the secret)


class HttpPaginationMode(str, Enum):
    """How the connector follows pagination across multiple upstream pages.

    ``NONE`` returns one page. ``CURSOR`` reads a next-page token from the JSON body
    (``cursor_path``) and re-sends it as a query param (``cursor_param``). ``PAGE``
    increments a 1-based page number (``page_param``) until a page comes back empty or
    short. ``LINK_HEADER`` follows the RFC 5988 ``Link: <…>; rel="next"`` header. In
    every mode the records under ``items_path`` are concatenated, capped at
    ``max_pages`` so a buggy/hostile upstream can't loop forever.
    """

    NONE = "NONE"
    CURSOR = "CURSOR"
    PAGE = "PAGE"
    LINK_HEADER = "LINK_HEADER"


class HttpPaginationConfig(BaseModel):
    """Declarative pagination for an HTTP REST connector (bounded, opt-in)."""

    mode: HttpPaginationMode = HttpPaginationMode.NONE
    #: JSON path (dotted) to the list of records on each page (e.g. ``data.items``).
    items_path: str = ""
    #: CURSOR: dotted path to the next-page token in the body; the query param to send.
    cursor_path: str = ""
    cursor_param: str = "cursor"
    #: PAGE: the query param carrying the 1-based page number.
    page_param: str = "page"
    #: Hard ceiling on pages followed — a DoS / infinite-loop guard.
    max_pages: int = 20


class HttpToolConfig(BaseModel):
    """Declarative description of an HTTP REST connector.

    Path placeholders (``{name}``), query args, body args, and header args are
    all resolved from the invocation's ``args`` at execution time.
    """

    # The base URL comes from ``base_url_env_var`` (a secret/config NAME, preferred)
    # when set; otherwise the literal ``base_url`` (handy for simple public endpoints).
    # At least one must resolve at call time.
    base_url_env_var: str = ""
    base_url: str = ""
    method: str = "GET"
    path_template: str = "/"
    auth: HttpAuthConfig = HttpAuthConfig()
    query_arg_names: list[str] = []
    body_arg_names: list[str] = []
    header_arg_names: list[str] = []
    timeout_seconds: float = 15.0
    #: Egress allow-list (host names). Empty ⇒ any PUBLIC host (the SSRF guard still
    #: blocks private/loopback/reserved). Pin this to the API's host for defence-in-depth.
    egress_allow_hosts: list[str] = []
    #: Only for a trusted INTERNAL API: skip the public-IP SSRF check (still requires the
    #: host be reachable). Leave False for anything reachable from untrusted prompts.
    allow_private_hosts: bool = False
    #: Static query params always sent (e.g. an API version) — merged under model args.
    static_query: dict[str, str] = {}
    pagination: HttpPaginationConfig = HttpPaginationConfig()
    #: Arg whose value is an idempotency key for a side-effecting (non-GET) call. When
    #: set it is sent as the ``Idempotency-Key`` header so a safe retry can't double-act.
    idempotency_arg: str | None = None


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
    #: Whether the tool only reads (True) or also mutates state (False). ``None`` lets
    #: the model-facing description infer intent from the tool's name. Surfaced to
    #: small models so a look-up never lands on a write tool (reader/writer confusion).
    read_only: bool | None = None
    #: Whether ``read_only`` is an AUTHORITATIVE author assertion of no side effect
    #: (safe to re-fire on a TIMEOUT), as opposed to a mere INFERENCE from the tool's
    #: HTTP method or name. Only an authoritative ``read_only=True`` may re-fire a
    #: timed-out call: a GET/HEAD endpoint can still have server-side side effects (an
    #: analytics beacon, ``GET /trigger``), so method-derived read-only is a
    #: parallelism hint ONLY — never a licence to duplicate a timed-out invocation.
    #: Defaults False so any tool not explicitly asserted read-only is treated as
    #: side-effecting for retry purposes (fail closed).
    read_only_authoritative: bool = False
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
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.name,
            namespace="tool_definition",
            kind="tool_definition",
            version=version,
            metadata=metadata,
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
    #: A capability/RBAC denial: the run principal does NOT hold the tool's capability
    #: (``tool:<name>:invoke`` / ``:write``). DISTINCT from ``POLICY_BLOCKED`` — which the
    #: ``requires_approval`` gate uses — so a capability denial is a HARD failure the model
    #: sees, never mistaken for a resumable human-approval checkpoint (a re-run would deny it
    #: again with the same unchanged principal, wedging the run; red-team r6).
    CAPABILITY_DENIED = "CAPABILITY_DENIED"
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
