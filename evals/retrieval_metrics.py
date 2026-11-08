"""Pure retrieval metrics (no network)."""

from collections.abc import Iterable, Sequence


def ranked_sources(chunks) -> list[str]:
    """Distinct document sources in rank order (relevance is judged per document)."""
    return list(dict.fromkeys(c.metadata.get("source", "") for c in chunks))


def first_relevant_rank(sources: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first relevant document, or None if it was not retrieved."""
    wanted = set(relevant)
    for rank, source in enumerate(sources, start=1):
        if source in wanted:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float | None:
    """Share of questions whose relevant document appears within the top k (None if no data)."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else None


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float | None:
    """Mean of 1/rank (0 for a miss): rewards putting the right document first."""
    return sum(1 / r if r else 0.0 for r in ranks) / len(ranks) if ranks else None