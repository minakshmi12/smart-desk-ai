"""Search-method selection: display summaries, MultiRetriever dispatch, chain pass-through."""

import pytest

from smartdesk.agent.nodes.kb_answer import make_kb_answer_node
from smartdesk.config import get_settings
from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.chain import RAGChain, RAGResult
from smartdesk.rag.confidence import EscalationReason
from smartdesk.rag.display import summarize_chunks
from smartdesk.rag.hybrid import MODE_INFO, MultiRetriever
from smartdesk.rag.retriever import RetrievedChunk


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """Dummy key and a known threshold."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.6")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chunk(text, distance, **kw):
    return RetrievedChunk(text, {"title": "VPN Setup", "source": "it/vpn.md"}, distance, **kw)


def test_summary_flags_gate_citation_and_rounds_scores():
    """Chunks past the threshold are marked not-passed; only the cited object is marked cited."""
    a, b = _chunk("good", 0.2, sparse_score=3.12345, rerank_score=0.9), _chunk("far", 0.9)
    rows = summarize_chunks([a, b], 0.6, cited=[a])
    assert rows[0]["passed_gate"] and rows[0]["cited"] and rows[0]["bm25"] == 3.123
    assert not rows[1]["passed_gate"] and not rows[1]["cited"] and rows[1]["bm25"] is None
    assert rows[0]["text"] == "good" and rows[0]["rank"] == 1


def test_multi_retriever_uses_requested_default_and_rejects_unknown_modes():
    """mode picks the retriever; None uses the default; an unknown mode raises."""
    class Fake:
        def __init__(self, name):
            self.name, self.calls = name, []

        def search(self, query, department=None, top_k=None):
            self.calls.append((query, department, top_k))
            return [self.name]

    mr = MultiRetriever(llm=object(), default_mode="hybrid")
    mr._retrievers = {m: Fake(m) for m in MODE_INFO}
    assert mr.search("q") == ["hybrid"]
    assert mr.search("q", "it", mode="dense") == ["dense"]
    assert mr._retrievers["dense"].calls == [("q", "it", None)]
    assert mr.search("q", mode="sparse", top_k=8) == ["sparse"]
    assert mr._retrievers["sparse"].calls == [("q", None, 8)]
    with pytest.raises(ValueError):
        mr.search("q", mode="bogus")
    assert set(MODE_INFO) == {"dense", "sparse", "hybrid"}


class _Retriever:
    default_mode = "hybrid"

    def __init__(self, chunks):
        self.chunks, self.kwargs = chunks, None

    def search(self, query, department=None, **kw):
        self.kwargs = kw
        return self.chunks


class _LLM:
    def __init__(self, grounded):
        self.grounded = grounded

    def parse(self, messages, schema):
        return self.grounded

    def chat(self, messages, temperature=0.2):
        return "standalone"


def test_chain_passes_mode_only_when_set_and_describes_the_search():
    """
    mode reaches the retriever only if given; the result carries mode, timings and
    per-chunk gate/citation flags.
    """
    chunks = [_chunk("Restart the client.", 0.2), _chunk("noise", 0.9)]
    ret = _Retriever(chunks)
    chain = RAGChain(retriever=ret, llm=_LLM(GroundedAnswer(
        answerable=True, answer="Restart", used_sources=[1])))

    r = chain.answer("q")
    assert ret.kwargs == {} and r.retrieval["mode"] == "hybrid"
    rows = r.retrieval["chunks"]
    assert [x["passed_gate"] for x in rows] == [True, False]
    assert rows[0]["cited"] and not rows[1]["cited"]
    assert r.retrieval["seconds_total"] >= r.retrieval["seconds_retrieval"] >= 0
    assert r.contexts == ["Restart the client."]
    assert r.retrieval["top_k"] == 3

    r2 = chain.answer("q", mode="dense", top_k=8)
    assert ret.kwargs == {"mode": "dense", "top_k": 8}
    assert r2.retrieval["mode"] == "dense" and r2.retrieval["reranker"] == "none"
    assert r2.retrieval["top_k"] == 8


def test_kb_node_forwards_top_k_from_per_turn_config():
    """The graph node forwards the API's per-turn K to the RAG chain."""
    class Chain:
        options = None

        def answer(self, question, history, **options):
            self.options = options
            return RAGResult(
                answerable=False, answer="I don't know", standalone_question=question,
                reason=EscalationReason.LLM_INSUFFICIENT,
            )

    chain = Chain()
    node = make_kb_answer_node(chain)
    node({"messages": [{"role": "user", "content": "question"}]},
         {"configurable": {"retrieval_mode": "sparse", "top_k": 8}})
    assert chain.options == {"mode": "sparse", "top_k": 8}


def test_escalated_results_still_describe_the_search():
    """When nothing passes the gate the agent escalates, and the UI can still show why."""
    chain = RAGChain(retriever=_Retriever([_chunk("x", 0.95)]), llm=_LLM(None))
    r = chain.answer("q")
    assert not r.answerable and r.retrieval["chunks"][0]["passed_gate"] is False