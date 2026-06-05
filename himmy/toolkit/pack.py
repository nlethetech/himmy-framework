"""Toolkit pack catalog: named bundles of built-in tools the CLI/spec can resolve.

A :class:`ToolPack` couples a registrar (``register(registry, config)``) with the names
of the tools it provides, so an ``agent.yaml`` can say ``tool_packs: [web, files]`` and
:func:`register_packs` wires them onto a :class:`ToolRegistry` in one call. The catalog
is the single source of truth listed by ``himmy tools``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from himmy.core import HimmyError
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.agentic import AGENTIC_TOOL_NAMES, register_agentic_pack
from himmy.toolkit.code import register_code_pack
from himmy.toolkit.comms import register_comms_pack
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.data import register_data_pack
from himmy.toolkit.datasources import register_datasources_pack
from himmy.toolkit.documents import register_documents_pack
from himmy.toolkit.files import register_files_pack
from himmy.toolkit.knowledge import register_knowledge_pack
from himmy.toolkit.memory import register_memory_pack
from himmy.toolkit.nepal import NEPAL_TOOL_NAMES, register_nepal_pack
from himmy.toolkit.utils import register_utils_pack
from himmy.toolkit.web import register_web_pack


class UnknownToolPackError(HimmyError):
    """Raised when a requested pack name is not in the catalog."""


@dataclass(frozen=True)
class ToolPack:
    """A named bundle of built-in tools plus its registrar."""

    name: str
    description: str
    tool_names: tuple[str, ...]
    register: Callable[[ToolRegistry, ToolkitConfig], None]


#: The built-in pack catalog, keyed by pack name.
BUILTIN_PACKS: dict[str, ToolPack] = {
    "web": ToolPack(
        "web",
        "Reach the open web: search, fetch a page, or make an HTTP request.",
        ("web_search", "web_fetch", "http_request"),
        register_web_pack,
    ),
    "files": ToolPack(
        "files",
        "Read, write, and list files under a sandboxed root.",
        ("read_file", "write_file", "list_dir"),
        register_files_pack,
    ),
    "data": ToolPack(
        "data",
        "Run read-only SQL queries against SQLite or Postgres.",
        ("sql_query",),
        register_data_pack,
    ),
    "code": ToolPack(
        "code",
        "Execute Python in a resource-limited sandbox.",
        ("run_python",),
        register_code_pack,
    ),
    "utils": ToolPack(
        "utils",
        "Everyday helpers: a safe calculator and the current time.",
        ("calculator", "current_time"),
        register_utils_pack,
    ),
    "knowledge": ToolPack(
        "knowledge",
        "Build and semantically search the agent's own knowledge base (RAG).",
        ("kb_ingest", "kb_search"),
        register_knowledge_pack,
    ),
    "documents": ToolPack(
        "documents",
        "Extract text from PDF / text / Markdown / CSV / Excel files under the root.",
        ("read_document",),
        register_documents_pack,
    ),
    "comms": ToolPack(
        "comms",
        "Reach the outside world: send email or POST to a webhook (approval-gated).",
        ("send_email", "send_webhook"),
        register_comms_pack,
    ),
    "data-sources": ToolPack(
        "data-sources",
        "Keyless public data: weather, geocoding, and Wikipedia summaries.",
        ("weather", "geocode", "wikipedia"),
        register_datasources_pack,
    ),
    "memory": ToolPack(
        "memory",
        "Durable long-term memory: remember facts and recall them semantically.",
        ("remember", "recall"),
        register_memory_pack,
    ),
    "nepal": ToolPack(
        "nepal",
        "Nepal: Bikram Sambat dates, NPR/Devanagari formatting, NRB forex.",
        NEPAL_TOOL_NAMES,
        register_nepal_pack,
    ),
    "agentic": ToolPack(
        "agentic",
        "Be agentic: ask the human, keep a scratchpad, manage a todo list.",
        AGENTIC_TOOL_NAMES,
        register_agentic_pack,
    ),
}


def resolve_packs(names: Iterable[str]) -> list[ToolPack]:
    """Resolve pack ``names`` to :class:`ToolPack`s, raising on any unknown name."""
    resolved: list[ToolPack] = []
    for name in names:
        pack = BUILTIN_PACKS.get(name)
        if pack is None:
            raise UnknownToolPackError(
                f"unknown tool pack {name!r}; known packs: "
                f"{', '.join(sorted(BUILTIN_PACKS))}"
            )
        resolved.append(pack)
    return resolved


def register_packs(
    registry: ToolRegistry,
    names: Iterable[str],
    config: ToolkitConfig | None = None,
) -> ToolRegistry:
    """Register every tool of each named pack onto ``registry`` and return it."""
    cfg = config or ToolkitConfig()
    for pack in resolve_packs(names):
        pack.register(registry, cfg)
    return registry


__all__ = [
    "ToolPack",
    "BUILTIN_PACKS",
    "resolve_packs",
    "register_packs",
    "UnknownToolPackError",
]
