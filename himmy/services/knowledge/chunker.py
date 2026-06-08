"""Knowledge kernel: chunkers that segment documents for embedding.

:class:`SemanticChunker` is the offline default — a char-window splitter on natural
boundaries. :class:`MarkdownAwareChunker` is the opt-in, structure-aware upgrade
(2026 RAG best practice): it splits on markdown headers FIRST so a chunk never
straddles a section boundary, then applies the same char-window logic within each
section. Both return ``(start, end, text)`` triples whose offsets index the
ORIGINAL document, so parent-document windows still resolve.
"""

from __future__ import annotations

import re


class SemanticChunker:
    """Splits text into overlapping character windows on natural boundaries.

    Each chunk targets ``max_chars`` with ``overlap`` carried into the next chunk
    so context is not severed mid-thought. Boundaries prefer paragraph/sentence
    breaks within the window before falling back to a hard cut.
    """

    def __init__(
        self,
        *,
        max_chars: int = 800,
        overlap: int = 100,
        min_new_chars: int = 1,
    ) -> None:
        """Configure target chunk size, inter-chunk overlap, and the minimum new span.

        ``min_new_chars`` is the smallest amount of *non-overlapping* content a
        follow-on chunk must contribute; a window that would add fewer than this many
        new characters is skipped so the chunker never emits near-duplicate
        micro-chunks (which would double embed cost and pollute top-k with noise).
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap must be in [0, max_chars)")
        if min_new_chars < 1:
            raise ValueError("min_new_chars must be >= 1")
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_new_chars = min_new_chars

    def chunk(self, text: str) -> list[tuple[int, int, str]]:
        """Return ``(start, end, text)`` triples covering ``text``.

        ``start``/``end`` are character offsets into the original document, enabling
        windowed retrieval back from the parent document. The realized overlap is
        clamped to at most half of the cut chunk's length, so a boundary cut well
        before ``start + max_chars`` cannot produce a follow-on window that is almost
        entirely overlap.
        """
        if not text:
            return []
        chunks: list[tuple[int, int, str]] = []
        n = len(text)
        start = 0
        prev_end = -1
        while start < n:
            end = min(start + self.max_chars, n)
            if end < n:
                end = self._find_boundary(text, start, end)
            chunk_text = text[start:end]
            # Skip a window that adds too little new (non-overlapping) content
            # relative to its predecessor — avoids near-duplicate micro-chunks.
            # Never skip the final window: that would drop tail coverage. When the
            # tail is too small to stand alone, fold it into the previous chunk so
            # the document stays fully covered without a near-duplicate micro-chunk.
            if prev_end >= 0 and (end - prev_end) < self.min_new_chars:
                if end >= n:
                    if chunks:
                        p_start, _, _ = chunks[-1]
                        chunks[-1] = (p_start, end, text[p_start:end])
                    else:  # pragma: no cover - prev_end>=0 implies a prior chunk
                        chunks.append((start, end, chunk_text))
                    break
                start = end
                continue
            chunks.append((start, end, chunk_text))
            prev_end = end
            if end >= n:
                break
            # Clamp the realized overlap to half the cut chunk so the next window
            # is never dominated by carry-over context.
            effective_overlap = min(self.overlap, (end - start) // 2)
            next_start = end - effective_overlap
            # Guarantee forward progress even with large overlap on short cuts.
            start = next_start if next_start > start else end
        return chunks

    def _find_boundary(self, text: str, start: int, end: int) -> int:
        """Prefer a paragraph/sentence/whitespace boundary near ``end``."""
        # Search backwards from end toward start for a clean break.
        window = text[start:end]
        for sep in ("\n\n", "\n", ". ", " "):
            idx = window.rfind(sep)
            # Only honor a boundary that keeps the chunk reasonably full.
            if idx != -1 and idx >= self.max_chars // 2:
                return start + idx + len(sep)
        return end


#: A markdown ATX header line: 1-6 ``#`` then a space then the heading text.
_HEADER_RE = re.compile(r"^(#{1,6})\s+\S")


class MarkdownAwareChunker:
    """Heading-aware chunker: split on markdown headers, then char-window each section.

    A header at a level in ``header_split_levels`` starts a new section, so a chunk
    never merges content across (say) two H1s or H2s — keeping each chunk topically
    coherent, which measurably lifts retrieval precision over a blind char-window
    split. Within a section the proven :class:`SemanticChunker` window logic runs, so
    a long section is still split into overlapping windows and offsets stay valid
    into the original document (parent-doc windows resolve unchanged).

    Documents with no headers degrade to exactly the :class:`SemanticChunker`
    behaviour, so this is a safe drop-in for mixed corpora.
    """

    def __init__(
        self,
        *,
        max_chars: int = 800,
        overlap: int = 100,
        min_new_chars: int = 1,
        header_split_levels: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        """Configure window sizing (see :class:`SemanticChunker`) + split levels.

        ``header_split_levels`` lists the markdown header depths that begin a new
        section (default H1/H2/H3). A header at a deeper, non-listed level stays
        inside its current section (it is ordinary content there).
        """
        if not header_split_levels:
            raise ValueError("header_split_levels must list at least one level")
        if any(lvl < 1 or lvl > 6 for lvl in header_split_levels):
            raise ValueError("header_split_levels entries must be in [1, 6]")
        self._levels = frozenset(header_split_levels)
        self._inner = SemanticChunker(
            max_chars=max_chars, overlap=overlap, min_new_chars=min_new_chars
        )

    @property
    def max_chars(self) -> int:
        """Target chunk size, delegated to the inner window chunker."""
        return self._inner.max_chars

    def chunk(self, text: str) -> list[tuple[int, int, str]]:
        """Return ``(start, end, text)`` triples, never crossing a split header.

        Sections are cut at qualifying header boundaries; each section is then
        windowed by the inner :class:`SemanticChunker`, with the inner offsets
        shifted back into the original document so the triples index ``text``.
        """
        if not text:
            return []
        sections = self._split_sections(text)
        chunks: list[tuple[int, int, str]] = []
        for sec_start, sec_end in sections:
            section_text = text[sec_start:sec_end]
            for start, end, _ in self._inner.chunk(section_text):
                abs_start = sec_start + start
                abs_end = sec_start + end
                chunks.append((abs_start, abs_end, text[abs_start:abs_end]))
        return chunks

    def _split_sections(self, text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` spans split at qualifying header line starts."""
        boundaries = [0]
        offset = 0
        for line in text.splitlines(keepends=True):
            match = _HEADER_RE.match(line)
            if match is not None and len(match.group(1)) in self._levels:
                if offset != 0 and offset not in boundaries:
                    boundaries.append(offset)
            offset += len(line)
        boundaries.append(len(text))
        spans: list[tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end > start:
                spans.append((start, end))
        return spans


__all__ = ["SemanticChunker", "MarkdownAwareChunker"]
