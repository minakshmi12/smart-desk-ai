"""Rerankers: ordering, dropping, clamping and fallbacks (fake LLM and model)."""

from smartdesk.rag.rerank import (
    CrossEncoderReranker, LLMReranker, NoReranker, PassageScore, RerankResult,
)
from smartdesk.rag.retriever import RetrievedChunk


def _chunks(n=3):
    """Helper: n candidates c1..cn."""
    return [RetrievedChunk(f"text {i}", {"title": f"D{i}", "source": f"s{i}"}, 0.3, id=f"c{i}")
            for i in range(1, n + 1)]


class FakeLLM:
    """Returns canned passage scores, or raises."""

    def __init__(self, scores=None, fail=False):
        self.scores, self.fail = scores or {}, fail

    def parse(self, messages, schema):
        if self.fail:
            raise RuntimeError("down")
        return RerankResult(scores=[PassageScore(passage=p, score=s) for p, s in self.scores.items()])


def _ids(chunks):
    return [c.id for c in chunks]


def test_llm_reranker_orders_by_score_and_normalises():
    """Scores 2/9/5 give order c2, c3, c1, with rerank_score 0.9 on the best."""
    out = LLMReranker(FakeLLM({1: 2, 2: 9, 3: 5}), min_score=0).rerank("q", _chunks(), 3)
    assert _ids(out) == ["c2", "c3", "c1"] and out[0].rerank_score == 0.9


def test_unrelated_passages_are_dropped_but_unscored_ones_are_kept():
    """
    Score 0 is dropped at min_score=1; a passage the model omitted is kept, after scored ones.
    WHY IT MATTERS: noise must not reach the answer prompt, but an omission is not a verdict.
    """
    out = LLMReranker(FakeLLM({1: 0, 2: 7}), min_score=1).rerank("q", _chunks(), 5)
    assert _ids(out) == ["c2", "c3"]


def test_out_of_range_numbers_are_clamped_or_ignored():
    """Score 15 becomes 10, -3 becomes 0, and an invented passage 99 is ignored."""
    out = LLMReranker(FakeLLM({1: 15, 2: -3, 99: 8}), min_score=0).rerank("q", _chunks(2), 2)
    assert [(c.id, c.rerank_score) for c in out] == [("c1", 1.0), ("c2", 0.0)]


def test_failures_and_empty_replies_fall_back_to_the_fused_order():
    """
    A failed call or a reply with no usable scores keeps the original order.
    WHY IT MATTERS: a flaky reranker must degrade quality, not break answers.
    """
    assert _ids(LLMReranker(FakeLLM(fail=True)).rerank("q", _chunks(), 2)) == ["c1", "c2"]
    assert _ids(LLMReranker(FakeLLM({})).rerank("q", _chunks(), 2)) == ["c1", "c2"]
    assert LLMReranker(FakeLLM()).rerank("q", [], 3) == []


def test_cross_encoder_orders_by_model_score_and_no_reranker_truncates():
    """A fake model's scores decide the order; NoReranker just keeps the first top_k."""
    class Model:
        def predict(self, pairs):
            return [0.1, 0.9, 0.5]

    assert _ids(CrossEncoderReranker(model=Model()).rerank("q", _chunks(), 2)) == ["c2", "c3"]
    assert _ids(NoReranker().rerank("q", _chunks(), 2)) == ["c1", "c2"]