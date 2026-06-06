"""`himmy init --template <name>` scaffolds a working specialised agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from himmy.cli.commands import _TEMPLATES, cmd_init
from himmy.config.agent_spec import load_agent_spec


def _init(tmp_path: Path, template: str) -> int:
    return cmd_init(
        argparse.Namespace(
            directory=str(tmp_path), force=False, team=False, template=template
        )
    )


def test_three_templates_are_available() -> None:
    assert set(_TEMPLATES) == {"helpdesk", "analyst", "researcher"}


def test_helpdesk_scaffolds_a_knowledge_agent_plus_docs(tmp_path: Path) -> None:
    assert _init(tmp_path, "helpdesk") == 0
    assert (tmp_path / "agent.yaml").exists()
    assert (tmp_path / "docs" / "handbook.md").exists()
    spec = load_agent_spec(tmp_path / "agent.yaml")
    assert spec.knowledge == ["./docs"]
    assert spec.tools == ["kb_search"]


def test_analyst_scaffolds_a_declarative_http_tool(tmp_path: Path) -> None:
    _init(tmp_path, "analyst")
    spec = load_agent_spec(tmp_path / "agent.yaml")
    assert spec.http_tools and spec.http_tools[0].name == "exchange_rate"
    assert spec.http_tools[0].query == ["base", "symbols"]


def test_researcher_scaffolds_a_skill_agent(tmp_path: Path) -> None:
    _init(tmp_path, "researcher")
    spec = load_agent_spec(tmp_path / "agent.yaml")
    assert "web_research" in spec.skills


def test_every_template_produces_a_valid_loadable_spec(tmp_path: Path) -> None:
    for i, name in enumerate(_TEMPLATES):
        d = tmp_path / str(i)
        _init(d, name)
        load_agent_spec(d / "agent.yaml")  # must not raise
