"""Property tests: knowledge chunkers cover the source, respect bounds, no empties.

Hypothesis fuzzes the chunking invariants retrieval depends on: with the default
``min_new_chars`` every character of the source is covered gap-free, offsets
always slice the ORIGINAL document back to the chunk text, no chunk is empty,
window bounds hold, the markdown-aware chunker never lets a chunk straddle a
split-level header, and headerless documents degrade to exactly the plain
window chunker. Skips gracefully when ``hypothesis`` is not installed
(dev-only dependency).
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from himmy.services.knowledge.chunker import (  # noqa: E402
    MarkdownAwareChunker,
    SemanticChunker,
)

# Moderate example counts, no deadline, derandomized => fast and seed-stable in CI.
_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True, database=None)

#: Mirror of the chunker's ATX header detector (the test oracle).
_HEADER_RE = re.compile(r"^(#{1,6})\s+\S")

_TEXTS = st.text(max_size=2000)


@st.composite
def _window_configs(draw: st.DrawFn) -> tuple[int, int]:
    """A valid (max_chars, overlap) pair: overlap strictly below max_chars."""
    max_chars = draw(st.integers(min_value=8, max_value=200))
    overlap = draw(st.integers(min_value=0, max_value=max_chars - 1))
    return max_chars, overlap


def _split_header_positions(doc: str) -> list[int]:
    """Document offsets of qualifying (H1/H2/H3) header line starts.

    Lines are delimited exactly as the chunker delimits them
    (``str.splitlines``, which also splits on ``\\r`` etc.), so the oracle and
    the implementation agree on what a "line start" is.
    """
    positions: list[int] = []
    offset = 0
    for line in doc.splitlines(keepends=True):
        match = _HEADER_RE.match(line)
        if match is not None and len(match.group(1)) in (1, 2, 3):
            positions.append(offset)
        offset += len(line)
    return positions


_LINE_CHARS = st.characters(codec="utf-8", exclude_characters="\n")
_MD_LINES = st.one_of(
    st.builds(
        lambda level, body: "#" * level + " " + body,
        st.integers(min_value=1, max_value=6),
        st.text(alphabet=_LINE_CHARS, min_size=1, max_size=30),
    ),
    st.text(alphabet=_LINE_CHARS, max_size=40),
)
_MD_DOCS = st.lists(_MD_LINES, max_size=20).map("\n".join)


@_SETTINGS
@given(text=_TEXTS, config=_window_configs())
def test_default_chunking_covers_source_without_gaps(
    text: str, config: tuple[int, int]
) -> None:
    """Default config: every character is covered, slices match, bounds hold."""
    max_chars, overlap = config
    chunks = SemanticChunker(max_chars=max_chars, overlap=overlap).chunk(text)
    if not text:
        assert chunks == []
        return
    covered = 0
    for start, end, chunk_text in chunks:
        assert end > start  # no empty chunks
        assert text[start:end] == chunk_text  # offsets index the original
        assert end - start <= max_chars  # window bound respected
        assert start <= covered  # contiguous: no gap before this chunk
        covered = max(covered, end)
    assert covered == len(text)  # full coverage through the tail


@_SETTINGS
@given(
    text=_TEXTS,
    config=_window_configs(),
    min_new_chars=st.integers(min_value=1, max_value=40),
)
def test_min_new_chars_preserves_head_tail_and_relaxed_bound(
    text: str, config: tuple[int, int], min_new_chars: int
) -> None:
    """Any ``min_new_chars``: head/tail always covered, no empties, bound relaxed
    by at most ``min_new_chars - 1`` (the documented tail fold)."""
    max_chars, overlap = config
    chunker = SemanticChunker(
        max_chars=max_chars, overlap=overlap, min_new_chars=min_new_chars
    )
    chunks = chunker.chunk(text)
    if not text:
        assert chunks == []
        return
    assert chunks
    assert chunks[0][0] == 0
    assert max(end for _, end, _ in chunks) == len(text)
    for start, end, chunk_text in chunks:
        assert end > start
        assert text[start:end] == chunk_text
        assert end - start <= max_chars + min_new_chars - 1


@_SETTINGS
@given(doc=_MD_DOCS, config=_window_configs())
def test_markdown_chunks_cover_source_and_never_straddle_split_headers(
    doc: str, config: tuple[int, int]
) -> None:
    """Markdown-aware: full gap-free coverage AND no chunk crosses an H1/H2/H3."""
    max_chars, overlap = config
    chunker = MarkdownAwareChunker(max_chars=max_chars, overlap=overlap)
    chunks = chunker.chunk(doc)
    if not doc:
        assert chunks == []
        return
    headers = _split_header_positions(doc)
    covered = 0
    for start, end, chunk_text in chunks:
        assert end > start
        assert doc[start:end] == chunk_text
        assert start <= covered
        covered = max(covered, end)
        # A split-level header may only ever sit at a chunk's START: one strictly
        # inside the span means the chunk straddled a section boundary.
        assert not any(start < position < end for position in headers)
    assert covered == len(doc)


@_SETTINGS
@given(
    text=st.text(
        alphabet=st.characters(codec="utf-8", exclude_characters="#"), max_size=1500
    ),
    config=_window_configs(),
)
def test_headerless_documents_degrade_to_the_plain_chunker(
    text: str, config: tuple[int, int]
) -> None:
    """With no headers, the markdown chunker is exactly the semantic chunker."""
    max_chars, overlap = config
    markdown = MarkdownAwareChunker(max_chars=max_chars, overlap=overlap)
    plain = SemanticChunker(max_chars=max_chars, overlap=overlap)
    assert markdown.chunk(text) == plain.chunk(text)
