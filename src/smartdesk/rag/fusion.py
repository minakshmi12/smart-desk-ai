"""Reciprocal Rank Fusion: merge several ranked lists into one."""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Combine rankings of ids into [(id, score)], best first.

    Each list adds 1/(k + rank) per item (rank starts at 1), so an item ranked
    well by SEVERAL retrievers beats one ranked first by only one.

    WHY RRF AND NOT SCORE MIXING: BM25 scores (unbounded) and cosine distances
    (0-2, lower is better) are on different scales; ranks are comparable.
    k=60 is the value from the original paper; larger k flattens the differences.

    An id repeated inside one list counts once. Ties are broken by best rank,
    then id, so results are deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))