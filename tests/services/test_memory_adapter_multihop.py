"""Tests for relational recall expansion + adapter multi-hop injection.

Covers ``recall(max_hops=...)`` expanding hits with graph-related memories, and the
:class:`MemoryContextAdapter` injecting a deduped ``related_memories`` section when
``max_hops`` is set — while the default (``None``) reproduces today's exact output.
"""

from __future__ import annotations

from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryContextAdapter,
    MemoryService,
)
from himmy.services.memory.store import RELATES_TO
from tests.conftest import run_async


def _linked_corpus() -> tuple[MemoryService, str]:
    """A subject whose top hit is linked to a related (off-query) fact."""
    svc = MemoryService(InMemoryMemoryStore())
    hit = svc.remember("the orchard has apple and pear trees", subject_id="u")
    # A related fact that would NOT surface on the orchard query on its own.
    related = svc.remember("pollination is handled by the bees", subject_id="u")
    svc.link(hit.memory_id, related.memory_id, relation=RELATES_TO)
    return svc, "u"


def test_recall_max_hops_off_is_unchanged() -> None:
    """recall() with max_hops=None (default) returns only the semantic hits."""
    svc, subject = _linked_corpus()
    hits = run_async(svc.recall("apple pear orchard trees", subject_id=subject))
    texts = [h.record.text for h in hits]
    assert texts == ["the orchard has apple and pear trees"]


def test_recall_max_hops_expands_with_related() -> None:
    """recall(max_hops=1) appends the graph-related memory after the semantic hit."""
    svc, subject = _linked_corpus()
    hits = run_async(
        svc.recall("apple pear orchard trees", subject_id=subject, max_hops=1)
    )
    texts = [h.record.text for h in hits]
    assert texts[0] == "the orchard has apple and pear trees"  # semantic hit leads
    assert "pollination is handled by the bees" in texts  # related appended
    # The related fact is injected with a 0.0 similarity (it is a graph neighbour).
    related_hit = next(h for h in hits if "pollination" in h.record.text)
    assert related_hit.similarity == 0.0


def test_recall_max_hops_dedups_related_against_hits() -> None:
    """A fact that is BOTH a semantic hit and a graph neighbour is not duplicated."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("the orchard has apple trees", subject_id="u")
    b = svc.remember("the orchard has pear trees", subject_id="u")
    svc.link(a.memory_id, b.memory_id)  # both also match the query semantically
    hits = run_async(
        svc.recall("orchard apple pear trees", subject_id="u", top_k=5, max_hops=1)
    )
    ids = [h.record.memory_id for h in hits]
    assert len(ids) == len(set(ids))  # no duplicates
    assert set(ids) == {a.memory_id, b.memory_id}


def test_adapter_max_hops_none_reproduces_exact_output() -> None:
    """With max_hops unset the adapter output is byte-for-byte the historical render."""
    svc, subject = _linked_corpus()
    baseline = MemoryContextAdapter(svc, subject_id=subject)
    field = run_async(baseline.fetch("memory", {"query": "apple pear orchard trees"}))
    assert field is not None
    assert field.value["rendered_text"] == "- the orchard has apple and pear trees"
    assert "related_memories" not in field.value


def test_adapter_max_hops_injects_related_section() -> None:
    """With max_hops set the adapter injects a deduped related_memories section."""
    svc, subject = _linked_corpus()
    adapter = MemoryContextAdapter(svc, subject_id=subject, max_hops=1)
    field = run_async(adapter.fetch("memory", {"query": "apple pear orchard trees"}))
    assert field is not None
    related = field.value["related_memories"]
    assert [r["text"] for r in related] == ["pollination is handled by the bees"]
    # The rendered text carries both the main hit and the related section.
    rendered = field.value["rendered_text"]
    assert "- the orchard has apple and pear trees" in rendered
    assert "Related memories:" in rendered
    assert "- pollination is handled by the bees" in rendered


def test_adapter_max_hops_no_related_links_omits_section() -> None:
    """A hit with no graph neighbours injects no related section even with max_hops."""
    svc = MemoryService(InMemoryMemoryStore())
    svc.remember("the orchard has apple and pear trees", subject_id="u")
    adapter = MemoryContextAdapter(svc, subject_id="u", max_hops=2)
    field = run_async(adapter.fetch("memory", {"query": "apple pear orchard trees"}))
    assert field is not None
    assert "related_memories" not in field.value
    assert field.value["rendered_text"] == "- the orchard has apple and pear trees"


def test_adapter_related_dedups_across_multiple_hits() -> None:
    """Two hits sharing one neighbour inject that neighbour once (deduped)."""
    svc = MemoryService(InMemoryMemoryStore())
    a = svc.remember("orchard apple trees thrive", subject_id="u")
    b = svc.remember("orchard pear trees thrive", subject_id="u")
    shared = svc.remember("irrigation runs on schedule", subject_id="u")
    svc.link(a.memory_id, shared.memory_id)
    svc.link(b.memory_id, shared.memory_id)
    adapter = MemoryContextAdapter(svc, subject_id="u", top_k=5, max_hops=1)
    field = run_async(adapter.fetch("memory", {"query": "orchard trees thrive"}))
    assert field is not None
    related = field.value["related_memories"]
    assert [r["text"] for r in related] == ["irrigation runs on schedule"]
