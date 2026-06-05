"""Nepal kernel: Nepali language support for cross-script RAG.

Real Nepali text mixes **Devanagari** (नेपाल), **Romanized Nepali** (nepal), and
English. Most embedders treat those as unrelated, so a query in one script misses
documents in another. This module normalizes across scripts:

* :func:`transliterate` — Devanagari → Roman (handles consonant + matra + halant,
  so नेपाल → ``nepala``), passing Roman/English through untouched.
* :func:`normalize_nepali` — transliterate + lowercase + token-final schwa deletion,
  so ``नेपाल`` / ``Nepal`` / ``nepal`` all normalize to ``nepal``.
* :class:`NepaliEmbedder` — wraps any embedder and normalizes text before
  embedding, so retrieval matches across scripts (great with the offline
  :class:`~himmy.services.knowledge.embedder.DeterministicEmbedder`; even better
  with a real multilingual model).
"""

from __future__ import annotations

import re
from typing import Any

# Devanagari consonants -> Roman base (no inherent vowel yet).
_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "w",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
}
# Independent vowels.
_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "u",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
# Dependent vowel signs (matras) that follow a consonant.
_MATRAS = {
    "ा": "aa",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}
_HALANT = "्"
_NASAL = {"ं": "n", "ँ": "n", "ः": "h"}
_DIGITS = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def is_devanagari(text: str) -> bool:
    """True when ``text`` contains any Devanagari character."""
    return bool(_DEVANAGARI_RE.search(text))


def transliterate(text: str) -> str:
    """Transliterate Devanagari to Roman; Roman/English passes through unchanged.

    A consonant takes its following matra as its vowel, drops the vowel on a halant,
    and otherwise carries the inherent ``a`` (so क → ``ka``, क् → ``k``, के → ``ke``).
    """
    chars = list(text)
    out: list[str] = []
    i = 0
    n = len(chars)
    while i < n:
        char = chars[i]
        if char in _CONSONANTS:
            base = _CONSONANTS[char]
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt in _MATRAS:
                out.append(base + _MATRAS[nxt])
                i += 2
            elif nxt == _HALANT:
                out.append(base)
                i += 2
            else:
                out.append(base + "a")
                i += 1
        elif char in _VOWELS:
            out.append(_VOWELS[char])
            i += 1
        elif char in _NASAL:
            out.append(_NASAL[char])
            i += 1
        elif char in _DIGITS:
            out.append(_DIGITS[char])
            i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def normalize_nepali(text: str) -> str:
    """Script-fold text so Devanagari/Romanized/English forms compare equal.

    Transliterates, lowercases, tokenizes, and deletes each token's trailing
    inherent-``a`` schwa (Nepali drops the word-final schwa), so ``नेपाल`` and
    ``Nepal`` and ``nepal`` all fold to ``nepal``.
    """
    roman = transliterate(text).lower()
    folded: list[str] = []
    for token in _TOKEN_RE.findall(roman):
        # Collapse repeated vowels (matras double them: नेपाल -> nepaal -> nepal),
        # then delete the word-final inherent-a schwa.
        token = re.sub(r"([aeiou])\1+", r"\1", token)
        folded.append(re.sub(r"a+$", "", token) or token)
    return " ".join(folded)


class NepaliEmbedder:
    """Wrap an embedder so it embeds the script-folded form (cross-script RAG).

    Conforms to :class:`~himmy.services.knowledge.embedder.EmbedderProtocol`. The
    base embedder defaults to the offline ``DeterministicEmbedder``; pass a real
    multilingual model for semantic Nepali retrieval.
    """

    def __init__(self, base: Any = None, *, dim: int = 64) -> None:
        """Wrap ``base`` (default: a DeterministicEmbedder of dimension ``dim``)."""
        if base is None:
            from himmy.services.knowledge.embedder import DeterministicEmbedder

            base = DeterministicEmbedder(dim=dim)
        self._base = base
        self.dim = getattr(base, "dim", dim)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query after folding it across scripts."""
        return await self._base.embed_query(normalize_nepali(text))

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents after folding each across scripts."""
        return await self._base.embed_documents([normalize_nepali(t) for t in texts])


__all__ = [
    "is_devanagari",
    "transliterate",
    "normalize_nepali",
    "NepaliEmbedder",
]
