"""Knowledge kernel: the SemanticChunker that segments documents for embedding."""

from __future__ import annotations


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


__all__ = ["SemanticChunker"]
