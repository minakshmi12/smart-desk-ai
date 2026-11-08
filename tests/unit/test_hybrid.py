"""HybridRetriever with a fake Chroma collection (no network)."""

import math

import pytest

from smartdesk.config import get_settings
from smartdesk.rag.hybrid import HybridRetriever, cosine_distance
from smartdesk.rag.retriever import RetrievedChunk


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """Dummy key; small pool so dense search deliberately misses some chunks."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("CANDIDATE_POOL", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


ROWS = {  # id -> (text, metadata, embedding)
    "a": ("Install the VPN client from the portal", {"department": "it", "title": "Setup"}, [1.0, 0.0]),
    "b": ("Contractors must request VPN access", {"department": "it", "title": "Access"}, [1.0, 0.1]),
    "c": ("Annual leave is 20 days", {"department": "hr", "title": "Leave"}, [0.9, 0.3]),
    "d": ("If you see error 809 restart the client and open UDP 4500",
          {"department": "it", "title": "Errors"}, [0.2, 1.0]),   # semantically far from the query
}


class FakeCollection:
    """Dense search returns `order`; get() serves documents or embeddings."""

    def __init__(self, rows=None, order=("a", "b", "c", "d")):
        self.rows, self.order = dict(rows or ROWS), list(order)
        self.where, self.full_gets = None, 0

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.where = where
        ids = [i for i in self.order
               if not where or self.rows[i][1]["department"] == where["department"]][:n_results]
        return {"ids": [ids], "documents": [[self.rows[i][0] for i in ids]],
                "metadatas": [[self.rows[i][1] for i in ids]],
                "distances": [[0.1 * (n + 1) for n in range(len(ids))]]}

    def get(self, ids=None, include=None):
        sel = list(ids) if ids is not None else list(self.rows)
        out = {"ids": sel}
        if "documents" in include:
            self.full_gets += 1
            out["documents"] = [self.rows[i][0] for i in sel]
            out["metadatas"] = [self.rows[i][1] for i in sel]
        if "embeddings" in include:
            out["embeddings"] = [self.rows[i][2] for i in sel]
        return out


class FakeLLM:
    """Embeds every query as [1, 0] and counts calls."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def _retriever(col=None, **kw):
    return HybridRetriever(llm=FakeLLM(), collection=col or FakeCollection(), **kw)


def test_cosine_distance_basics():
    """Identical = 0, orthogonal = 1, opposite = 2, zero vector = 1."""
    assert cosine_distance([1, 0], [2, 0]) == pytest.approx(0.0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert cosine_distance([1, 0], [-1, 0]) == pytest.approx(2.0)
    assert cosine_distance([0, 0], [1, 0]) == 1.0


def test_keyword_hit_that_dense_search_missed_is_retrieved_with_a_real_distance():
    """
    Dense search (pool 2) returns a, b; BM25 finds d via "809". d must reach the
    result, with its exact cosine distance computed from its stored embedding.
    WHY IT MATTERS: this is the whole point of hybrid retrieval, and the gate needs the distance.
    """
    trace = _retriever(mode="hybrid").trace("error 809")
    assert [i for i, _ in trace.dense] == ["a", "b"]            # dense missed d
    d = next(c for c in trace.chunks if c.id == "d")
    assert d.distance == pytest.approx(1 - 0.2 / math.sqrt(1.04)) and d.sparse_score > 0


def test_sparse_mode_uses_bm25_only_but_still_reports_distances():
    """No dense list in sparse mode; the chunk still has a cosine distance."""
    trace = _retriever(mode="sparse").trace("error 809")
    assert trace.dense == [] and [c.id for c in trace.chunks] == ["d"]
    assert 0 <= trace.chunks[0].distance <= 2


def test_reranker_decides_final_order_and_top_k_is_respected():
    """A reversing reranker flips the fused order; at most TOP_K chunks come back."""
    class Reverse:
        def rerank(self, query, candidates, top_k):
            return list(reversed(candidates))[:top_k]

    base = _retriever(mode="hybrid").search("error 809")
    flipped = _retriever(mode="hybrid", reranker=Reverse()).search("error 809")
    assert len(base) <= 3 and [c.id for c in flipped] == [c.id for c in reversed(base)][:3]


def test_per_call_top_k_overrides_configured_result_limit():
    """A per-query K overrides TOP_K without mutating the retriever default."""
    retriever = _retriever(mode="hybrid")
    assert len(retriever.search("error 809", top_k=1)) == 1
    assert len(retriever.search("error 809", top_k=3)) == 3


def test_department_filter_applies_to_both_retrievers():
    """department="hr" reaches Chroma's filter and restricts BM25 too."""
    col = FakeCollection()
    chunks = _retriever(col, mode="hybrid").search("error 809 leave", department="hr")
    assert col.where == {"department": "hr"} and [c.id for c in chunks] == ["c"]


def test_empty_index_returns_nothing_without_embedding():
    """An empty collection yields [] and never calls the embedding API."""
    llm = FakeLLM()
    assert HybridRetriever(llm=llm, collection=FakeCollection(rows={}), mode="hybrid").search("x") == []
    assert llm.calls == 0


def test_bm25_index_is_cached_and_rebuilt_when_the_collection_changes():
    """Repeated searches reuse the index; adding a chunk triggers one rebuild."""
    col, r = FakeCollection(), None
    r = _retriever(col, mode="hybrid")
    r.search("error 809"); r.search("leave")
    assert col.full_gets == 1
    col.rows["e"] = ("Printer toner order form", {"department": "it", "title": "Printers"}, [0.5, 0.5])
    assert [c.id for c in r.search("toner")][:1] == ["e"] and col.full_gets == 2


def test_dense_is_not_a_hybrid_mode():
    """Plain dense search belongs to Retriever; asking HybridRetriever for it is an error."""
    with pytest.raises(ValueError):
        _retriever(mode="dense")