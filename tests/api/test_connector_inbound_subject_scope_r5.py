"""Per-sender memory/KB isolation for the inbound connector handler (rbac r5).

``_build_inbound_handler`` keeps a per-sender :class:`ChatThread` (conversational isolation) but
historically built ONE runtime shared by every sender, threading only the connector's fixed
service-principal workspace as ``subject``. So distinct external senders shared one memory subject
(``t:<workspace>:default``) and one KB scope — sender A's ``remember`` surfaced in sender B's
``recall`` (a within-connector cross-end-user data bleed).

The fix builds a runtime PER SENDER with ``subject_scope=sender_id``, namespacing each sender's
memory/KB to ``t:<workspace>:s:<sender>`` (the combined-token scheme the per-user run path uses).
This test spies on ``build_runtime_for_spec`` and asserts each sender gets its own subject_scope.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import himmy.runtime.from_spec as from_spec_mod
from himmy.api.connector_inbound import _build_inbound_handler

_AGENT_YAML = """\
name: greeter
provider: stub
instructions:
  - hi
"""


class _Runtime:
    async def run_task_detailed(self, persona, task, thread, *, llm_config):  # type: ignore[no-untyped-def]
        class _R:
            output_text = "ok"
            thread = None

        return _R()


def test_each_sender_gets_its_own_subject_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(_AGENT_YAML)

    seen: list[str | None] = []
    real = from_spec_mod.build_runtime_for_spec

    def _spy(spec, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs.get("subject_scope"))
        # No tool packs declared → registry is None (single-task path), avoid real wiring.
        return _Runtime(), None

    monkeypatch.setattr(from_spec_mod, "build_runtime_for_spec", _spy)
    assert real is not None

    handle = _build_inbound_handler(str(agent_path), app=None)
    asyncio.run(handle("111", "hi"))
    asyncio.run(handle("222", "hi"))
    # Re-delivery from the same sender reuses its runtime (no rebuild).
    asyncio.run(handle("111", "again"))

    # Sender 111 and 222 each got their OWN subject_scope; 111's second delivery did not rebuild.
    assert seen == ["111", "222"]
    assert len(set(seen)) == 2
