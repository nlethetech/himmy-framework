"""Seeded synthetic-data factory for the load/profiling harness.

:class:`SyntheticDataFactory` builds deterministic entity DAGs directly on an
in-memory :class:`~himmy.entities.registry.EntityRegistry` so the load tests have a
non-trivial lineage graph to traverse without any runtime, network, or DB.

Determinism is total and reproducible:

* every record's ``stable_id`` is derived from a fixed seed + an incrementing
  per-kind counter (``stable_id_for``), **never** from wall-clock time or
  unseeded randomness — so two factories with the same seed emit byte-identical
  graphs, and the resulting ``record_id``s (a UUID5 of ``kind:stable_id:version``)
  are stable across processes and machines;
* the graph shape (personas -> prompts -> snapshots, with typed
  ``uses_persona`` / ``built_from`` links) is a pure function of the requested
  counts.

The factory returns the populated registry plus the records it created so a test
can pick traversal roots and assert on graph size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from himmy.entities.records import EntityRecord, stable_id_for
from himmy.entities.registry import EntityRegistry

#: Namespace prefix for every synthetic ``stable_id`` so derived ids never collide
#: with real spine records and are obviously test data.
_NAMESPACE = "loadtest"

#: The default seed; tests may override it but the same seed always reproduces the
#: same graph (ids, edges, ordering).
DEFAULT_SEED = 1729


@dataclass
class SyntheticGraph:
    """The product of one :meth:`SyntheticDataFactory.build_graph` call.

    Holds the in-memory :class:`EntityRegistry` plus flat lists of the records
    created (by kind), the root record ids suitable for a lineage traversal, and the
    total edge count — everything a load/profiling test needs to choose roots and
    assert on size without re-walking the graph.
    """

    registry: EntityRegistry
    personas: list[EntityRecord] = field(default_factory=list)
    prompts: list[EntityRecord] = field(default_factory=list)
    snapshots: list[EntityRecord] = field(default_factory=list)
    root_ids: list[str] = field(default_factory=list)
    edge_count: int = 0

    @property
    def node_count(self) -> int:
        """Total number of records (personas + prompts + snapshots) in the graph."""
        return len(self.personas) + len(self.prompts) + len(self.snapshots)


class SyntheticDataFactory:
    """Builds reproducible entity DAGs on a fresh in-memory registry.

    Seeded by construction (``seed``); a fresh factory with the same seed produces an
    identical graph. Ids are derived from ``stable_id_for`` over a deterministic
    per-kind counter (see :meth:`_next_stable_id`), so nothing about the output
    depends on the wall clock or process randomness.
    """

    def __init__(self, *, seed: int = DEFAULT_SEED) -> None:
        """Initialise with a fixed seed; the per-kind id counter starts at zero."""
        self._seed = seed
        self._counters: dict[str, int] = {}

    @property
    def seed(self) -> int:
        """The seed this factory derives all of its deterministic ids from."""
        return self._seed

    def _next_stable_id(self, kind: str) -> str:
        """Return the next deterministic ``stable_id`` for ``kind``.

        The key is ``"{seed}:{kind}:{counter}"`` with a per-kind incrementing
        counter — fully deterministic and free of wall-clock / RNG input — fed
        through :func:`stable_id_for` so the result is a stable UUID5.
        """
        index = self._counters.get(kind, 0)
        self._counters[kind] = index + 1
        key = f"{self._seed}:{kind}:{index}"
        return stable_id_for(key, namespace=_NAMESPACE)

    def build_graph(
        self,
        *,
        personas: int = 3,
        prompts_per_persona: int = 4,
        snapshots_per_prompt: int = 2,
    ) -> SyntheticGraph:
        """Build a persona -> prompt -> snapshot DAG with typed links, deterministically.

        For each persona a fan-out of prompts is created and linked
        ``prompt --uses_persona--> persona``; for each prompt a fan-out of evidence
        snapshots is created and linked ``snapshot --built_from--> prompt``. The
        persona records are returned as the traversal ``root_ids`` (a downward walk
        from a persona reaches its prompts and their snapshots).

        The total node count is
        ``personas * (1 + prompts_per_persona * (1 + snapshots_per_prompt))`` and is a
        pure function of the arguments + seed.
        """
        registry = EntityRegistry()
        graph = SyntheticGraph(registry=registry)

        for _ in range(max(0, personas)):
            persona = self._persona_record()
            registry.register(persona)
            graph.personas.append(persona)

            for _ in range(max(0, prompts_per_persona)):
                prompt = self._prompt_record(persona_stable_id=persona.stable_id)
                registry.register(prompt)
                graph.prompts.append(prompt)
                registry.link(
                    from_record_id=prompt.record_id,
                    to_record_id=persona.record_id,
                    relation="uses_persona",
                )
                graph.edge_count += 1

                for _ in range(max(0, snapshots_per_prompt)):
                    snapshot = self._snapshot_record(prompt_stable_id=prompt.stable_id)
                    registry.register(snapshot)
                    graph.snapshots.append(snapshot)
                    registry.link(
                        from_record_id=snapshot.record_id,
                        to_record_id=prompt.record_id,
                        relation="built_from",
                    )
                    graph.edge_count += 1

            graph.root_ids.append(persona.record_id)

        return graph

    def build_graph_of_size(self, target_nodes: int) -> SyntheticGraph:
        """Build a DAG with *at least* ``target_nodes`` records, deterministically.

        A convenience for the profiling test that wants "roughly N nodes": picks a
        fixed fan-out and grows the persona count until the node total reaches the
        target. The exact node count is deterministic for a given ``target_nodes``.
        """
        prompts_per_persona = 4
        snapshots_per_prompt = 2
        per_persona = 1 + prompts_per_persona * (1 + snapshots_per_prompt)
        personas = max(1, -(-max(1, target_nodes) // per_persona))  # ceil-div
        return self.build_graph(
            personas=personas,
            prompts_per_persona=prompts_per_persona,
            snapshots_per_prompt=snapshots_per_prompt,
        )

    def _persona_record(self) -> EntityRecord:
        """Create one deterministic ``persona`` record."""
        stable_id = self._next_stable_id("persona")
        index = self._counters["persona"] - 1
        return EntityRecord.create(
            stable_id=stable_id,
            version=1,
            kind="persona",
            payload={
                "name": f"Analyst-{index}",
                "description": "Synthetic load-test persona.",
                "instructions": ["Be precise.", "Cite evidence."],
            },
            metadata={"seed": self._seed, "synthetic": True, "index": index},
        )

    def _prompt_record(self, *, persona_stable_id: str) -> EntityRecord:
        """Create one deterministic ``prompt`` record bound to a persona."""
        stable_id = self._next_stable_id("prompt")
        index = self._counters["prompt"] - 1
        return EntityRecord.create(
            stable_id=stable_id,
            version=1,
            kind="prompt",
            payload={
                "text": f"Summarize outlook #{index}.",
                "persona_stable_id": persona_stable_id,
            },
            metadata={"seed": self._seed, "synthetic": True, "index": index},
        )

    def _snapshot_record(self, *, prompt_stable_id: str) -> EntityRecord:
        """Create one deterministic ``context_snapshot`` record bound to a prompt."""
        stable_id = self._next_stable_id("context_snapshot")
        index = self._counters["context_snapshot"] - 1
        return EntityRecord.create(
            stable_id=stable_id,
            version=1,
            kind="context_snapshot",
            payload={
                "fields": {"company": f"ACME-{index}", "score": index % 7},
                "prompt_stable_id": prompt_stable_id,
            },
            metadata={"seed": self._seed, "synthetic": True, "index": index},
        )


__all__ = [
    "DEFAULT_SEED",
    "SyntheticDataFactory",
    "SyntheticGraph",
]
