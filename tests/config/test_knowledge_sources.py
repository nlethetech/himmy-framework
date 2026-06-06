"""No-code knowledge: `knowledge: [./docs]` auto-ingests files/dirs into the KB."""

from __future__ import annotations

from pathlib import Path

from himmy.cli.commands import _ingest_knowledge
from himmy.config.agent_spec import AgentSpec
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit import ToolkitConfig, register_packs
from tests.conftest import run_async


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_packs(registry, ["knowledge"], ToolkitConfig())
    return registry


def test_ingests_a_directory_of_text_docs(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("the prize goat is named Himmy", encoding="utf-8")
    (tmp_path / "b.txt").write_text("the tractor is red", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")  # non-text, ignored
    registry = _registry()
    n = run_async(_ingest_knowledge(registry, [str(tmp_path)]))
    assert n == 2  # the .md and .txt, not the .png
    # and the ingested content is searchable
    out = run_async(registry.handler_for("kb_search")({"query": "goat name"}))
    assert out["results"]


def test_ingests_a_single_file(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("a single note", encoding="utf-8")
    assert run_async(_ingest_knowledge(_registry(), [str(f)])) == 1


def test_missing_source_is_skipped_not_fatal(tmp_path: Path) -> None:
    assert run_async(_ingest_knowledge(_registry(), [str(tmp_path / "nope")])) == 0


def test_agent_spec_loads_knowledge_field() -> None:
    spec = AgentSpec.model_validate({"name": "a", "knowledge": ["./docs", "faq.md"]})
    assert spec.knowledge == ["./docs", "faq.md"]
