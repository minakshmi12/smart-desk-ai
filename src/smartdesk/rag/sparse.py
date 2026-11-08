"""
BM25 keyword search (the "sparse" half of hybrid retrieval).

WHY THIS FILE EXISTS
--------------------
Dense embeddings capture meaning but can miss exact tokens: an error code
("809"), a port ("4500"), a product name. BM25 ranks by exact word overlap
weighted by rarity, so it catches those. It is written here (about 60 lines)
instead of imported so tokenization is under our control and IDF is never negative.
"""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it me my of on or "
    "our so that the their there this to was we what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """
    Lower-case words and numbers, minus stopwords, with plural "s" stripped.

    The same function is used for documents and queries, so any imperfection
    (e.g. "this" -> "thi", which a stopword rule happens to prevent) is consistent
    on both sides. Digits are kept: "error 809" must match "809".
    """
    tokens = []
    for word in _WORD_RE.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


@dataclass(frozen=True)
class SparseDoc:
    """One indexed chunk: its id, text and metadata (title is indexed with the text)."""

    id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class SparseHit:
    """A BM25 result: the chunk and its (unbounded, higher-is-better) score."""

    doc: SparseDoc
    score: float


class BM25Index:
    """In-memory inverted index with Okapi BM25 scoring."""

    def __init__(self, docs: Sequence[SparseDoc], k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            docs: the chunks to index.
            k1: how quickly repeated terms stop adding score (typical 1.2-2.0).
            b: how strongly long chunks are penalised (0 = not at all, 1 = fully).
        """
        self._docs = list(docs)
        self._k1, self._b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        self._lengths: list[int] = []
        for i, d in enumerate(self._docs):
            tokens = tokenize(f"{d.metadata.get('title', '')} {d.text}")
            self._lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((i, tf))
        self._avg_len = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._docs)

    def _idf(self, docs_with_term: int) -> float:
        """Lucene-style IDF: ln(1 + (N-n+0.5)/(n+0.5)), always >= 0 (rarer = higher)."""
        n_total = len(self._docs)
        return math.log(1 + (n_total - docs_with_term + 0.5) / (docs_with_term + 0.5))

    def search(self, query: str, k: int, department: str | None = None) -> list[SparseHit]:
        """
        Top-k chunks by BM25, best first; [] if the query has no indexable terms.

        `department` restricts results to that metadata value. Ties are broken by
        chunk id so results are deterministic.
        """
        terms = set(tokenize(query))
        if not terms or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(len(postings))
            for i, tf in postings:
                if department and self._docs[i].metadata.get("department") != department:
                    continue
                norm = 1 - self._b + self._b * self._lengths[i] / self._avg_len
                scores[i] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._docs[kv[0]].id))
        return [SparseHit(self._docs[i], s) for i, s in ranked[:k]]