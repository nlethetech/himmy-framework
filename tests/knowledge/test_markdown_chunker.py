"""Behavioral tests for the heading-aware MarkdownAwareChunker."""

from __future__ import annotations

from himmy.services.knowledge.chunker import MarkdownAwareChunker, SemanticChunker


def test_splits_on_header_boundaries() -> None:
    """Each H1/H2 section becomes its own chunk; sections never merge across a header."""
    doc = (
        "# Intro\n"
        "Welcome to the guide.\n\n"
        "## Setup\n"
        "Install the package first.\n\n"
        "## Usage\n"
        "Then call the function.\n"
    )
    chunker = MarkdownAwareChunker(max_chars=800)
    chunks = chunker.chunk(doc)
    texts = [t for _, _, t in chunks]
    # Three sections -> the per-section content stays separated.
    assert any("Welcome to the guide" in t and "Install" not in t for t in texts)
    assert any(
        "Install the package" in t and "call the function" not in t for t in texts
    )
    assert any("call the function" in t and "Install" not in t for t in texts)


def test_offsets_index_original_document() -> None:
    """Returned (start, end) offsets slice the ORIGINAL text back to the chunk text."""
    doc = "# A\nfirst section body\n\n## B\nsecond section body\n"
    chunker = MarkdownAwareChunker(max_chars=800)
    for start, end, text in chunker.chunk(doc):
        assert doc[start:end] == text


def test_deep_header_below_split_levels_stays_in_section() -> None:
    """A header deeper than header_split_levels does not start a new chunk."""
    doc = (
        "# Title\n"
        "Top content.\n\n"
        "#### Tiny\n"
        "This deep subsection should remain in the H1 section.\n"
    )
    # Only split on H1/H2; H4 is ordinary content.
    chunker = MarkdownAwareChunker(max_chars=2000, header_split_levels=(1, 2))
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Top content" in chunks[0][2]
    assert "deep subsection" in chunks[0][2]


def test_long_section_still_windowed() -> None:
    """A single oversized section is still split by the inner window logic."""
    body = "sentence. " * 300  # well over max_chars
    doc = f"# Big\n{body}"
    chunker = MarkdownAwareChunker(max_chars=200, overlap=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(len(t) <= 200 for _, _, t in chunks)


def test_no_headers_matches_semantic_chunker() -> None:
    """A document with no headers degrades to plain SemanticChunker behaviour."""
    doc = "Just plain prose. " * 40
    md = MarkdownAwareChunker(max_chars=200, overlap=20)
    plain = SemanticChunker(max_chars=200, overlap=20)
    assert md.chunk(doc) == plain.chunk(doc)


def test_empty_document() -> None:
    """An empty document yields no chunks."""
    assert MarkdownAwareChunker().chunk("") == []


def test_invalid_split_levels_rejected() -> None:
    """Out-of-range or empty header_split_levels is a construction error."""
    for bad in ((), (0,), (7,)):
        try:
            MarkdownAwareChunker(header_split_levels=bad)
        except ValueError:
            pass
        else:  # pragma: no cover - must raise
            raise AssertionError(f"expected ValueError for levels={bad}")
