"""Regression test for the google_pack approval gate (security finding #5).

``gmail_send`` and ``gcal_create`` mutate the real connected Google account. They
MUST be approval-gated by default (``requires_approval=True``) so a prompt-injection
email cannot make the agent silently send mail or create calendar events. The gate
is released only when ``google_allow_send`` is explicitly set on the config.

Before the fix both tools registered without ``requires_approval``, defaulting to
``False`` — these assertions failed.
"""

from __future__ import annotations

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.google_pack import register_google_pack


def _registered(config: ToolkitConfig) -> ToolRegistry:
    registry = ToolRegistry()
    register_google_pack(registry, config)
    return registry


def test_google_mutators_require_approval_by_default() -> None:
    registry = _registered(ToolkitConfig())

    send = registry.get("gmail_send")
    create = registry.get("gcal_create")
    assert send is not None
    assert create is not None

    assert send.requires_approval is True
    assert create.requires_approval is True


def test_google_reads_are_not_approval_gated() -> None:
    registry = _registered(ToolkitConfig())

    inbox = registry.get("gmail_inbox")
    events = registry.get("gcal_events")
    assert inbox is not None
    assert events is not None

    # Read-only tools never need a human checkpoint.
    assert inbox.requires_approval is False
    assert events.requires_approval is False


def test_google_allow_send_is_a_real_field_defaulting_to_false() -> None:
    # The opt-out must be a real, declared ToolkitConfig field — not a phantom
    # attribute — so the documented HIMMY_GOOGLE_ALLOW_SEND knob actually exists
    # and defaults to the gated posture.
    assert "google_allow_send" in ToolkitConfig.model_fields
    assert ToolkitConfig().google_allow_send is False


def test_google_allow_send_releases_the_gate() -> None:
    # Flip the real field (no getattr/setattr trickery): only an explicit opt-out
    # may release the approval gate.
    config = ToolkitConfig(google_allow_send=True)
    registry = _registered(config)

    send = registry.get("gmail_send")
    create = registry.get("gcal_create")
    assert send is not None
    assert create is not None

    assert send.requires_approval is False
    assert create.requires_approval is False


def test_google_allow_send_from_env(monkeypatch) -> None:
    # The documented env knob must wire into the config and release the gate; with
    # it unset (or falsey) the mutators stay approval-gated.
    monkeypatch.setenv("HIMMY_GOOGLE_ALLOW_SEND", "true")
    assert ToolkitConfig.from_env().google_allow_send is True

    monkeypatch.delenv("HIMMY_GOOGLE_ALLOW_SEND", raising=False)
    assert ToolkitConfig.from_env().google_allow_send is False
