"""Every built-in tool declares an explicit read/write intent (no heuristic guessing)."""

from __future__ import annotations

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit import BUILTIN_PACKS, ToolkitConfig, register_packs

# Tools whose intent is genuinely method-dependent and left unset on purpose.
_INTENTIONALLY_UNSET = {"http_request"}

# A few anchors we assert explicitly so a future change can't silently flip them.
_EXPECTED = {
    "read_file": True,
    "write_file": False,
    "sql_query": True,
    "run_python": False,
    "calculator": True,
    "current_time": True,
    "weather": True,
    "kb_search": True,
    "kb_ingest": False,
    "recall": True,
    "remember": False,
    "send_email": False,
    "todo_read": True,
    "todo_write": False,
}


def _all_defs():
    reg = ToolRegistry()
    register_packs(reg, list(BUILTIN_PACKS), ToolkitConfig.from_env())
    return {d.name: d for d in reg.list()}


def test_every_builtin_has_explicit_intent_except_dual_use() -> None:
    defs = _all_defs()
    unset = [
        n
        for n, d in defs.items()
        if d.read_only is None and n not in _INTENTIONALLY_UNSET
    ]
    assert not unset, f"built-in tools missing an explicit read_only: {unset}"


def test_known_anchors_classified_correctly() -> None:
    defs = _all_defs()
    for name, expected in _EXPECTED.items():
        assert name in defs, f"{name} not registered"
        assert defs[name].read_only is expected, (
            f"{name} read_only={defs[name].read_only}, expected {expected}"
        )


def test_http_request_left_unset_on_purpose() -> None:
    defs = _all_defs()
    assert defs["http_request"].read_only is None
