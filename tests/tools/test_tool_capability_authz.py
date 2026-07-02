"""Tests for P0 tool capability authorization (the confused-deputy fix).

Covers the four guarantees the work package introduces:

* a principal LACKING a tool's capability is DENIED that tool at runtime;
* an ANONYMOUS / all_tenants (offline) principal still runs EVERY tool (no-op bypass);
* the read/write intent split — a write tool needs the extra ``tool:<name>:write`` grant;
* a spawned sub-agent inherits the parent's gate verbatim (attenuate, never amplify).
"""

from __future__ import annotations

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.auth.rbac import DEFAULT_POLICY, AccessPolicy
from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
)
from himmy.services.tools.capability import ToolCapabilityAuthorizer
from tests.conftest import run_async


def _registry() -> ToolRegistry:
    """A registry with a read tool (``weather_get``) and a write tool (``send_email``)."""
    registry = ToolRegistry()

    def weather(args: dict) -> dict:
        return {"temp": 20}

    def send(args: dict) -> dict:
        return {"sent": True}

    # ``read_only=True`` is the EXPLICIT author flag — the only thing that waives the
    # ``:write`` sub-grant under the r2-hardened gate (name-inference no longer waives it,
    # since first-token-wins mis-classifies read-verb-prefixed writers like get_or_create).
    register_local_tool(
        registry,
        name="weather_get",
        handler=weather,
        description="reads weather",
        read_only=True,
    )
    register_local_tool(
        registry,
        name="send_email",
        handler=send,
        description="sends an email",
        read_only=False,
    )
    return registry


def _policy() -> AccessPolicy:
    return AccessPolicy.from_mapping(
        {
            "reader": ["tool:weather_get:invoke"],
            "mailer": ["tool:send_email:invoke", "tool:send_email:write"],
            "anytool": ["tool:*"],
            "admin": ["*:*"],
        }
    )


def _bound(principal: Principal) -> ToolService:
    authz = ToolCapabilityAuthorizer.from_principal(principal, _policy())
    return ToolService(_registry(), tool_authorizer=authz)


def test_anonymous_offline_runs_every_tool() -> None:
    """ANONYMOUS (all_tenants) bypasses the gate entirely — every tool allowed (no-op)."""
    svc = _bound(ANONYMOUS)
    for name in ("weather_get", "send_email"):
        res = run_async(svc.execute(ToolInvocation(tool_name=name)))
        assert res.outcome == "success", name


def test_no_authorizer_is_byte_unchanged() -> None:
    """With no authorizer wired at all the tool service behaves exactly as before."""
    svc = ToolService(_registry())  # no tool_authorizer
    res = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"


def test_principal_without_capability_is_denied() -> None:
    """A tenant-bound principal lacking a tool's capability is DENIED at runtime."""
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    svc = _bound(reader)
    # Granted the read tool.
    ok = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
    assert ok.outcome == "success"
    # Denied the write tool it has no grant for — deny-by-default.
    denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert denied.outcome == "denied"
    # red-team r6: a capability/RBAC denial carries the DISTINCT ``CAPABILITY_DENIED`` code
    # (not the approval gate's ``POLICY_BLOCKED``), so the runtime never mistakes it for a
    # resumable human-approval checkpoint.
    assert denied.error_code is ToolErrorCode.CAPABILITY_DENIED


def test_invoke_without_write_grant_denies_write_tool() -> None:
    """An invoke-only grant is NOT enough to call a side-effecting (write) tool."""
    # A role granting only invoke on the write tool, but not the write capability.
    policy = AccessPolicy.from_mapping({"halfgrant": ["tool:send_email:invoke"]})
    p = Principal(
        subject="u", roles=frozenset({"halfgrant"}), tenant_ids=frozenset({"ws1"})
    )
    svc = ToolService(
        _registry(), tool_authorizer=ToolCapabilityAuthorizer.from_principal(p, policy)
    )
    denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert denied.outcome == "denied"
    assert denied.error_code is ToolErrorCode.CAPABILITY_DENIED


def test_full_grant_allows_write_tool() -> None:
    """A role with both invoke + write may call the side-effecting tool."""
    mailer = Principal(
        subject="u", roles=frozenset({"mailer"}), tenant_ids=frozenset({"ws1"})
    )
    svc = _bound(mailer)
    ok = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert ok.outcome == "success"


def test_admin_and_wildcard_allow_all() -> None:
    """``admin`` (*:*) and ``tool:*`` both reach every tool (read AND write)."""
    for role in ("admin", "anytool"):
        p = Principal(
            subject="u", roles=frozenset({role}), tenant_ids=frozenset({"ws1"})
        )
        svc = _bound(p)
        for name in ("weather_get", "send_email"):
            res = run_async(svc.execute(ToolInvocation(tool_name=name)))
            assert res.outcome == "success", f"{role}:{name}"


def test_enforcing_authorizer_without_policy_fails_closed() -> None:
    """An enforcing authorizer with no policy denies everything (fail CLOSED)."""
    authz = ToolCapabilityAuthorizer(enforce=True, roles=frozenset({"reader"}), policy=None)
    svc = ToolService(_registry(), tool_authorizer=authz)
    denied = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
    assert denied.outcome == "denied"


def test_attenuate_returns_subset_gate() -> None:
    """A sub-agent's inherited gate cannot exceed the parent's (attenuate, never amplify)."""
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    parent = ToolCapabilityAuthorizer.from_principal(reader, _policy())
    child = parent.attenuate()
    # The child grants exactly what the parent grants — same roles, same enforcement.
    assert child.enforce is True
    assert child.roles == parent.roles
    assert child.is_authorized("weather_get", True) is True
    assert child.is_authorized("send_email", False) is False


def test_default_policy_operator_holds_tool_capability() -> None:
    """REGRESSION: the SHIPPED DEFAULT_POLICY must authorize operator for read+write tools.

    The routine/connector SERVICE principals AND operator humans run as ``operator``; if
    the default policy carried no ``tool:*`` grant, enabling an authenticator would deny
    EVERY tool to all of them. This pins the real policy (not a fabricated one) so the
    regression can never silently return.
    """
    op = Principal(
        subject="u", roles=frozenset({"operator"}), tenant_ids=frozenset({"ws1"})
    )
    authz = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    assert authz.enforce is True
    # A read tool and a (side-effecting) write tool are BOTH authorized for operator.
    assert authz.is_authorized("weather_get", True) is True
    assert authz.is_authorized("send_email", False) is True


def test_ambiguously_named_writer_requires_write_grant() -> None:
    """An ambiguously-named tool fails CLOSED to writer: invoke-only is NOT enough.

    ``process_payment`` has no ``read_only`` flag and a name ``classify_read_only`` cannot
    classify (-> None). An invoke-only grant must NOT reach it — it additionally requires
    ``tool:process_payment:write`` — so an operator granting a per-tool ``:invoke`` for a
    read-only reach cannot unknowingly hand write reach to an ambiguously-named writer.
    """
    invoke_only = AccessPolicy.from_mapping(
        {"halfgrant": ["tool:process_payment:invoke"]}
    )
    p = Principal(
        subject="u", roles=frozenset({"halfgrant"}), tenant_ids=frozenset({"ws1"})
    )
    gate = ToolCapabilityAuthorizer.from_principal(p, invoke_only)
    # read_only=None (no flag) + ambiguous name → treated as a writer → DENIED on invoke-only.
    assert gate.is_authorized("process_payment", None) is False
    # Granting the write capability too lets it through.
    full = AccessPolicy.from_mapping(
        {"full": ["tool:process_payment:invoke", "tool:process_payment:write"]}
    )
    pf = Principal(
        subject="u", roles=frozenset({"full"}), tenant_ids=frozenset({"ws1"})
    )
    assert (
        ToolCapabilityAuthorizer.from_principal(pf, full).is_authorized(
            "process_payment", None
        )
        is True
    )


def test_read_verb_prefixed_writer_requires_write_grant() -> None:
    """red-team r2: a read-verb-PREFIXED writer must NOT be waived to invoke-only.

    ``classify_read_only`` is first-token-wins, so a side-effecting tool whose name merely
    STARTS with a read verb is INFERRED read-only even though it mutates state:
    ``get_or_create_invoice`` (first token ``get``, but ``create`` is a later token) →
    ``True``; ``check_payment`` (first token ``check``) → ``True``. Before the fix the gate
    trusted that inference to waive the ``:write`` sub-grant, so a role holding only
    ``tool:<name>:invoke`` (expecting a look-up) could fire the write tool. After the fix
    ONLY an explicit author ``read_only=True`` waives ``:write``; a name-inferred verdict
    never does, so these fail CLOSED on invoke-only and require the ``:write`` grant.
    """
    from himmy.services.tools.access import classify_read_only

    # Sanity: these names ARE inferred read-only by the classifier (the mis-classification).
    assert classify_read_only("get_or_create_invoice") is True
    assert classify_read_only("check_payment") is True

    for name in ("get_or_create_invoice", "check_payment"):
        invoke_only = AccessPolicy.from_mapping({"halfgrant": [f"tool:{name}:invoke"]})
        p = Principal(
            subject="u", roles=frozenset({"halfgrant"}), tenant_ids=frozenset({"ws1"})
        )
        gate = ToolCapabilityAuthorizer.from_principal(p, invoke_only)
        # read_only=None (no explicit flag) → name-inference does NOT waive → DENIED.
        assert gate.is_authorized(name, None) is False, name
        # The write sub-grant lets it through.
        full = AccessPolicy.from_mapping(
            {"full": [f"tool:{name}:invoke", f"tool:{name}:write"]}
        )
        pf = Principal(
            subject="u", roles=frozenset({"full"}), tenant_ids=frozenset({"ws1"})
        )
        assert (
            ToolCapabilityAuthorizer.from_principal(pf, full).is_authorized(name, None)
            is True
        ), name
        # An EXPLICIT author read_only=True DOES waive the write grant (invoke-only OK).
        assert gate.is_authorized(name, True) is True, name


def test_method_derived_get_still_needs_write_grant() -> None:
    """A method-derived read-only HTTP ``GET`` must NOT waive the ``:write`` sub-grant.

    ``register_http_tool`` DERIVES ``read_only=True`` for a bare ``GET`` (no explicit
    author flag) but leaves ``read_only_authoritative`` ``False`` — the endpoint may still
    be side-effecting (``GET /trigger``). ``authorize_definition`` must therefore mirror the
    retry (:meth:`ToolService._is_retry_side_effect_safe`) and parallelism
    (:func:`himmy.services.inference.local._is_parallel_safe`) gates: only an AUTHORITATIVE
    ``read_only=True`` waives ``:write``. So an invoke-only role is DENIED the GET connector
    and needs ``tool:<name>:write`` — while an EXPLICIT (authoritative) ``read_only=True``
    GET is invoke-only reachable.
    """
    from himmy.services.tools import HttpToolConfig, register_http_tool

    reg = ToolRegistry()
    derived = register_http_tool(
        reg, name="trigger_get", http_config=HttpToolConfig(base_url="https://x", method="GET")
    )
    explicit = register_http_tool(
        reg,
        name="lookup_get",
        http_config=HttpToolConfig(base_url="https://x", method="GET"),
        read_only=True,
    )
    # Sanity: the derived GET IS inferred read-only but NOT authoritative.
    assert derived.read_only is True and derived.read_only_authoritative is False
    assert explicit.read_only is True and explicit.read_only_authoritative is True

    invoke_only = AccessPolicy.from_mapping(
        {"half": ["tool:trigger_get:invoke", "tool:lookup_get:invoke"]}
    )
    p = Principal(subject="u", roles=frozenset({"half"}), tenant_ids=frozenset({"ws1"}))
    gate = ToolCapabilityAuthorizer.from_principal(p, invoke_only)
    # Method-derived GET → not authoritative → fails CLOSED without the :write grant.
    assert gate.authorize_definition(derived) is False
    # Authoritative author read_only=True → waived → invoke-only reaches it.
    assert gate.authorize_definition(explicit) is True
    # The :write sub-grant lets the side-effecting GET through.
    full = AccessPolicy.from_mapping(
        {"full": ["tool:trigger_get:invoke", "tool:trigger_get:write"]}
    )
    pf = Principal(subject="u", roles=frozenset({"full"}), tenant_ids=frozenset({"ws1"}))
    assert (
        ToolCapabilityAuthorizer.from_principal(pf, full).authorize_definition(derived)
        is True
    )


def test_declarative_connector_get_still_needs_write_grant() -> None:
    """A declarative-spec (YAML) GET connector must NOT waive the ``:write`` sub-grant.

    ``register_connector_specs`` fronts every declared tool as a LOCAL tool with a
    METHOD-DERIVED ``read_only`` (``GET``/``HEAD`` → ``True``). That inference must register
    ``read_only_authoritative=False`` so a side-effecting ``GET /trigger`` fails CLOSED at the
    capability gate exactly like ``register_http_tool`` — an invoke-only role cannot fire it.
    """
    from himmy.connectors.spec import ConnectorSpec

    reg = ToolRegistry()
    spec = ConnectorSpec(
        name="beacon",
        description="A side-effecting GET beacon.",
        base_url="https://x",
        egress_allow_hosts=["x"],
        tools=[{"name": "trigger", "method": "GET", "path": "/trigger"}],
    )
    spec.build(fetcher=None).register_tools(reg)
    derived = reg.get("trigger")
    assert derived.read_only is True and derived.read_only_authoritative is False

    invoke_only = AccessPolicy.from_mapping({"half": ["tool:trigger:invoke"]})
    p = Principal(subject="u", roles=frozenset({"half"}), tenant_ids=frozenset({"ws1"}))
    gate = ToolCapabilityAuthorizer.from_principal(p, invoke_only)
    # Method-derived GET → not authoritative → fails CLOSED without :write.
    assert gate.authorize_definition(derived) is False
    # The :write sub-grant lets the side-effecting GET through.
    full = AccessPolicy.from_mapping(
        {"full": ["tool:trigger:invoke", "tool:trigger:write"]}
    )
    pf = Principal(subject="u", roles=frozenset({"full"}), tenant_ids=frozenset({"ws1"}))
    assert (
        ToolCapabilityAuthorizer.from_principal(pf, full).authorize_definition(derived)
        is True
    )


def test_from_actor_rebuilds_enforcing_gate() -> None:
    """A persisted actor descriptor rebuilds the enforcing gate (dispatch-recovery path)."""
    actor = {"subject": "u", "roles": ["reader"], "tool_authz_enforce": True}
    authz = ToolCapabilityAuthorizer.from_actor(actor, _policy())
    assert authz.enforce is True
    assert authz.is_authorized("weather_get", True) is True
    assert authz.is_authorized("send_email", False) is False
    # An actor without the enforce flag (offline / all_tenants) is a pass-through.
    offline = ToolCapabilityAuthorizer.from_actor({"subject": "anon"}, _policy())
    assert offline.enforce is False
    assert offline.is_authorized("send_email", False) is True


# ------------------------------------------- scope narrowing follows into the gate
def test_scope_narrows_tool_capability() -> None:
    """A token scoped to a single tool's invoke cannot reach OTHER tools its role grants.

    The operator role holds ``tool:*`` (every tool). A token whose OAuth scope is
    ``tool:weather_get:invoke`` must narrow that down to JUST the read tool — it can no
    longer invoke ``send_email``, exactly mirroring the route-layer narrowing. Before the
    fix the synthetic probe carried no scopes, so the gate ignored them (fail-open).
    """
    op = Principal(
        subject="u",
        roles=frozenset({"operator"}),
        tenant_ids=frozenset({"ws1"}),
        scopes=frozenset({"tool:weather_get:invoke"}),
    )
    gate = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    assert gate.scopes == frozenset({"tool:weather_get:invoke"})
    assert gate.is_authorized("weather_get", True) is True
    # The write tool the operator role WOULD reach is narrowed away by the scope.
    assert gate.is_authorized("send_email", False) is False


def test_read_scope_cannot_invoke_any_tool() -> None:
    """A 'run:read'-scoped operator token cannot invoke ANY tool (scope omits 'tool')."""
    op = Principal(
        subject="u",
        roles=frozenset({"operator"}),
        tenant_ids=frozenset({"ws1"}),
        scopes=frozenset({"run:read"}),
    )
    gate = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    # Route layer narrows run:write away; the tool gate must ALSO deny side-effecting tools.
    assert gate.is_authorized("send_email", False) is False
    assert gate.is_authorized("weather_get", True) is False


def test_oidc_resource_scope_does_not_lock_out_tools() -> None:
    """A garbage IdP resource scope must NOT lock the operator out of every tool."""
    op = Principal(
        subject="u",
        roles=frozenset({"operator"}),
        tenant_ids=frozenset({"ws1"}),
        scopes=frozenset({"openid", "api://himmy/access_as_user"}),
    )
    gate = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    # No permission-scope recognized → no narrowing → operator keeps its tool:* reach.
    assert gate.is_authorized("weather_get", True) is True
    assert gate.is_authorized("send_email", False) is True


def test_scope_survives_actor_roundtrip() -> None:
    """Scopes serialize into actor_metadata and rebuild the gate (dispatch-recovery)."""
    op = Principal(
        subject="u",
        roles=frozenset({"operator"}),
        tenant_ids=frozenset({"ws1"}),
        scopes=frozenset({"tool:weather_get:invoke"}),
    )
    actor = op.actor_metadata()
    assert actor["scopes"] == ["tool:weather_get:invoke"]
    rebuilt = ToolCapabilityAuthorizer.from_actor(actor, DEFAULT_POLICY)
    assert rebuilt.scopes == frozenset({"tool:weather_get:invoke"})
    assert rebuilt.is_authorized("weather_get", True) is True
    assert rebuilt.is_authorized("send_email", False) is False


def test_all_tenants_scoped_token_still_enforces_tool_reach() -> None:
    """Red-team r3: a scope-narrowed all_tenants admin token must NOT regain full tools.

    An operator mints a deliberately-attenuated delegated token (admin role for
    all_tenants reach, but ``scopes=[run:read, run:write]`` and NO tool scope) so it may
    START runs but never invoke side-effecting tools. The route layer ALREADY narrows its
    tool reach (the ``tool`` resource is uncovered by its scopes), so the per-run tool gate
    must enforce the SAME narrowing — else a run it starts could invoke any tool
    (attenuation escape). Only a truly scope-LESS all_tenants principal is the pass-through.
    """
    tok = Principal.build(
        subject="ci-bot",
        roles=["admin"],
        scopes=["run:read", "run:write"],
        all_tenants=True,
        auth_method="oidc",
    )
    # Sanity: the route layer honors the scope (token may create a run, not invoke tools).
    assert DEFAULT_POLICY.authorize(tok, "run", "write") is True
    assert DEFAULT_POLICY.authorize(tok, "tool", "send_email:write") is False
    gate = ToolCapabilityAuthorizer.from_principal(tok, DEFAULT_POLICY)
    assert gate.enforce is True  # the escape: previously False (non-enforcing)
    # Every tool denied — the scope covers no tool resource.
    assert gate.is_authorized("send_email", False) is False
    assert gate.is_authorized("weather_get", True) is False


def test_scopeless_all_tenants_principal_is_byte_unchanged_passthrough() -> None:
    """A truly scope-LESS all_tenants principal (offline/ANONYMOUS) stays non-enforcing."""
    for principal in (ANONYMOUS, Principal(subject="shared", all_tenants=True)):
        gate = ToolCapabilityAuthorizer.from_principal(principal, DEFAULT_POLICY)
        assert gate.enforce is False
        assert gate.is_authorized("send_email", False) is True
        assert gate.is_authorized("weather_get", True) is True


def test_all_tenants_scoped_token_enforces_across_actor_roundtrip() -> None:
    """The dispatch-recovery path also enforces a scope-narrowed all_tenants token.

    ``actor_metadata`` must carry ``tool_authz_enforce`` for a scoped all_tenants token so
    the leased-dispatch rebuild (``from_actor``) does not re-open the escape across a
    process boundary.
    """
    tok = Principal.build(
        subject="ci-bot",
        roles=["admin"],
        scopes=["run:read", "run:write"],
        all_tenants=True,
        auth_method="oidc",
    )
    actor = tok.actor_metadata()
    assert actor.get("tool_authz_enforce") is True
    rebuilt = ToolCapabilityAuthorizer.from_actor(actor, DEFAULT_POLICY)
    assert rebuilt.enforce is True
    assert rebuilt.is_authorized("send_email", False) is False
