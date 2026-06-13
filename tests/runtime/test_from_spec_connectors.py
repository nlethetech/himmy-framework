"""T1.4(a) — outbound connectors wired into the agent runtime from an AgentSpec.

An ``agent.yaml`` may declare a ``connectors:`` block; ``build_runtime_for_spec`` binds
each named OUTBOUND connector's tools into the run's tool registry — but ONLY when the
connector is enabled for the ``outbound`` surface AND configured (capability OK). These
tests cover, fully offline (no network):

* an ENABLED + CONFIGURED connector binds its tools into the registry;
* a connector that is configured but NOT enabled binds nothing (default-deny);
* a connector that is enabled but NOT configured (missing credential) binds nothing —
  skipped cleanly, never an error;
* an unknown connector name is skipped, not a crash;
* the connector's audit lands on the RUN's audit spine (so a tool call it makes shares
  the run's lineage registry).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import ConnectorService
from himmy.runtime.from_spec import (
    _register_outbound_connectors,
    build_runtime_for_spec,
)

_TOUCHED = [
    "HIMMY_GITHUB_TOKEN",
    "HIMMY_GITHUB_ALLOWED_REPOS",
    "HIMMY_SLACK_BOT_TOKEN",
    "HIMMY_SLACK_ALLOWED_CHANNELS",
    "HIMMY_CONNECTOR_GITHUB_OUTBOUND_ENABLED",
    "HIMMY_CONNECTOR_SLACK_OUTBOUND_ENABLED",
]


@pytest.fixture
def writable_backend(tmp_path: Path):
    """A writable FileSecrets backend so configure/enable can persist."""
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    yield
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _tool_names(registry: object) -> set[str]:
    return {d.name for d in registry.list()} if registry is not None else set()


def test_enabled_configured_connector_binds_its_tools(writable_backend: None) -> None:
    svc = ConnectorService()
    svc.configure(
        "github",
        {"HIMMY_GITHUB_TOKEN": "ghp_x", "HIMMY_GITHUB_ALLOWED_REPOS": "o/a"},
    )
    svc.enable("github", "outbound")

    spec = AgentSpec(name="a", connectors=["github"], provider="stub")
    _runtime, registry = build_runtime_for_spec(spec)
    names = _tool_names(registry)
    # the GitHub connector contributes its read/act tools
    assert any(n.startswith("github_") for n in names)
    assert "github_get_repo" in names


def test_configured_but_disabled_binds_nothing(writable_backend: None) -> None:
    svc = ConnectorService()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_x"})
    # NOT enabled for outbound

    spec = AgentSpec(name="a", connectors=["github"], provider="stub")
    _runtime, registry = build_runtime_for_spec(spec)
    assert not any(n.startswith("github_") for n in _tool_names(registry))


def test_enabled_but_unconfigured_skips_cleanly(writable_backend: None) -> None:
    svc = ConnectorService()
    svc.enable("github", "outbound")  # enabled but NO token configured

    spec = AgentSpec(name="a", connectors=["github"], provider="stub")
    # must not raise — a missing credential is skipped cleanly
    _runtime, registry = build_runtime_for_spec(spec)
    assert not any(n.startswith("github_") for n in _tool_names(registry))


def test_unknown_connector_name_is_skipped(writable_backend: None) -> None:
    spec = AgentSpec(name="a", connectors=["does-not-exist"], provider="stub")
    _runtime, registry = build_runtime_for_spec(spec)
    # registry exists (connectors block is truthy) but holds nothing
    assert _tool_names(registry) == set()


def test_inbound_connector_name_is_not_bound_as_a_tool(writable_backend: None) -> None:
    """A webhook (inbound) named in connectors: is the BFF's job, not an outbound tool."""
    svc = ConnectorService()
    svc.configure("webhook", {"HIMMY_WEBHOOK_SIGNING_SECRET": "a-long-secret"})
    svc.enable("webhook", "outbound")  # even if mis-enabled outbound, it has no tools

    spec = AgentSpec(name="a", connectors=["webhook"], provider="stub")
    _runtime, registry = build_runtime_for_spec(spec)
    assert _tool_names(registry) == set()


def test_register_outbound_audits_onto_the_run_spine(writable_backend: None) -> None:
    """The connector's register/skip audits onto the spine passed in (the run's registry)."""
    from himmy.entities.registry import EntityRegistry
    from himmy.services.tools.registry import ToolRegistry

    svc = ConnectorService()
    svc.configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_x"})
    svc.enable("github", "outbound")

    spine = EntityRegistry()
    registry = ToolRegistry(entity_registry=spine)
    registered = _register_outbound_connectors(registry, ["github"], audit=spine)
    assert any(n.startswith("github_") for n in registered)
    # a connector_action event landed on the spine (the registration audit)
    from himmy.connectors.sdk import CONNECTOR_EVENT_TYPE
    from himmy.services.audit.log import SecurityAuditLog

    events = SecurityAuditLog(spine).recent(event_type=CONNECTOR_EVENT_TYPE)
    assert events, "expected a connector registration audit on the run spine"
    assert any(e.resource == "github" and e.action == "register" for e in events)
