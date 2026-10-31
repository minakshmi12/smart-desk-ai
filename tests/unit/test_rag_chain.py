"""
Tests for the RAG chain, the retriever mapping, and the escalation edge.

WHY THIS FILE EXISTS
--------------------
Verifies the wiring end to end with fakes: right chunks reach the prompt, GPT-4o
is skipped when retrieval is hopeless, and an unanswerable question ends in a
ticket offer with the draft saved for the next turn.
"""

import pytest

from smartdesk.agent.graph import build_graph, route_after_kb
from smartdesk.agent.prompts import DONT_KNOW, build_answer_messages
from smartdesk.config import get_settings
from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.chain import RAGChain, RAGResult
from smartdesk.rag.confidence import EscalationReason
from smartdesk.rag.retriever import RetrievedChunk, Retriever
from fakes import ScriptedLLM
from smartdesk.tickets.mock_provider import MockTicketProvider


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """Fake settings: a dummy key and a known threshold/top_k. Cache cleared around each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.6")
    monkeypatch.setenv("TOP_K", "3")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLM:
    """
    Records calls and returns canned results.
    `parse` returns the GroundedAnswer given at construction; `chat` returns the
    rewrite; `embed` returns a dummy vector.
    """

    def __init__(self, grounded=None, rewrite="standalone question"):
        self.grounded, self.rewrite = grounded, rewrite
        self.parse_calls, self.chat_calls = [], []

    def parse(self, messages, schema):
        self.parse_calls.append(messages)
        return self.grounded

    def chat(self, messages, temperature=0.2):
        self.chat_calls.append(messages)
        return self.rewrite

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeRetriever:
    """Returns fixed chunks and remembers which query it was given."""

    def __init__(self, chunks):
        self.chunks, self.queries = chunks, []

    def search(self, query, department=None):
        self.queries.append(query)
        return self.chunks


def _chunk(text, distance, title="VPN Setup"):
    """Helper: a chunk with a title and source, so labels are predictable."""
    return RetrievedChunk(text, {"title": title, "source": "it/vpn.md"}, distance)


def _chain(chunks, grounded=None):
    """Helper: build a RAGChain with fakes; returns (chain, llm, retriever)."""
    llm, ret = FakeLLM(grounded), FakeRetriever(chunks)
    return RAGChain(retriever=ret, llm=llm), llm, ret


# -------------------------------------------------------------- chain
def test_answerable_question_returns_answer_and_cited_sources():
    """
    Good retrieval + grounded model reply = an answered result with citations.
    WHY IT MATTERS: the happy path, including mapping passage numbers to labels.
    """
    chain, _, _ = _chain(
        [_chunk("Restart the client.", 0.2), _chunk("Open UDP 500.", 0.3, "Network Ports")],
        GroundedAnswer(answerable=True, answer="Restart the client.", used_sources=[1]),
    )
    r = chain.answer("How do I fix VPN error 809?")
    assert r.answerable and r.answer == "Restart the client."
    assert r.sources == ["VPN Setup (it/vpn.md)"]
    assert len(r.consulted) == 2


def test_hopeless_retrieval_skips_the_llm_entirely():
    """
    If every chunk is too far away, GPT-4o is never called.
    WHY IT MATTERS: saves cost and prevents the model from improvising an answer.
    """
    chain, llm, _ = _chain([_chunk("Unrelated.", 0.95)])
    r = chain.answer("What's the CEO's salary?")
    assert not r.answerable and r.reason is EscalationReason.LOW_SIMILARITY
    assert r.answer == DONT_KNOW
    assert llm.parse_calls == []
    assert r.consulted == ["VPN Setup (it/vpn.md)"]  # still passed on for the ticket


def test_model_dont_know_becomes_escalation():
    """
    A model "I don't know" is normalised and flagged for escalation.
    WHY IT MATTERS: the second line of defence when retrieval looked fine but wasn't.
    """
    chain, _, _ = _chain(
        [_chunk("Install the client.", 0.4)],
        GroundedAnswer(answerable=False, answer="whatever", used_sources=[]),
    )
    r = chain.answer("Can I use VPN from Mars?")
    assert not r.answerable and r.reason is EscalationReason.LLM_INSUFFICIENT
    assert r.answer == DONT_KNOW


def test_only_relevant_passages_reach_the_prompt():
    """
    A far-away chunk is excluded from the numbered context sent to GPT-4o.
    WHY IT MATTERS: noise in the prompt is how wrong answers get blended in.
    """
    chain, llm, _ = _chain(
        [_chunk("GOOD passage", 0.2), _chunk("NOISE passage", 0.9)],
        GroundedAnswer(answerable=True, answer="ok", used_sources=[1]),
    )
    chain.answer("q")
    user_msg = llm.parse_calls[0][1]["content"]
    assert "GOOD passage" in user_msg and "NOISE passage" not in user_msg
    assert llm.parse_calls[0][0]["role"] == "system"


def test_answer_prompt_distinguishes_leave_allowance_from_personal_balance():
    """The model should answer policy entitlement without claiming account balance."""
    messages = build_answer_messages(
        "How many casual leave do I have?",
        [_chunk("Annual casual leave allowance is 12 days per calendar year.", 0.2)],
    )

    assert "answer with that allowance" in messages[0]["content"]
    assert "personal remaining balance is not available" in messages[0]["content"]


def test_follow_up_is_rewritten_before_search_but_first_message_is_not():
    """
    With history, the retriever receives the rewritten question; without, no rewrite call.
    WHY IT MATTERS: "what about Windows?" is unsearchable on its own, and the
    first message shouldn't pay for a pointless extra call.
    """
    g = GroundedAnswer(answerable=True, answer="ok", used_sources=[1])
    chain, llm, ret = _chain([_chunk("x", 0.2)], g)
    chain.answer("How do I set up VPN?")
    assert llm.chat_calls == [] and ret.queries == ["How do I set up VPN?"]

    history = [{"role": "user", "content": "VPN on Mac?"}, {"role": "assistant", "content": "..."}]
    chain.answer("what about Windows?", history)
    assert ret.queries[-1] == "standalone question"


# ---------------------------------------------------------- retriever
def test_retriever_maps_chroma_output_and_caps_n_results():
    """
    Chroma's nested result dict becomes RetrievedChunks; n_results never exceeds the index size.
    WHY IT MATTERS: asking Chroma for more results than exist can raise errors,
    and the mapping is the contract the confidence rules rely on.
    """

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kw):
            self.kw = kw
            return {
                "documents": [["a", "b"]],
                "metadatas": [[{"title": "A"}, {"title": "B"}]],
                "distances": [[0.2, 0.5]],
            }

    col = FakeCollection()
    hits = Retriever(llm=FakeLLM(), collection=col).search("q", department="it")
    assert [(h.text, h.distance) for h in hits] == [("a", 0.2), ("b", 0.5)]
    assert col.kw["n_results"] == 2          # min(top_k=3, count=2)
    assert col.kw["where"] == {"department": "it"}


def test_retriever_returns_empty_for_empty_index():
    """
    An empty collection yields [] without calling the embedding API.
    WHY IT MATTERS: a fresh install should escalate gracefully, not crash.
    """

    class Empty:
        def count(self):
            return 0

    assert Retriever(llm=FakeLLM(), collection=Empty()).search("q") == []


# -------------------------------------------------------------- graph
class FakeChain:
    """A chain stub returning a preset RAGResult."""

    def __init__(self, result):
        self.result = result

    def answer(self, question, history=None, department=None):
        return self.result


def test_route_after_kb_follows_escalate_flag():
    """
    The conditional edge sends escalations to offer_ticket and everything else to the end.
    WHY IT MATTERS: this one function IS the escalation policy at graph level.
    """
    assert route_after_kb({"escalate": True}) == "offer_ticket"
    assert route_after_kb({"escalate": False}) == "end"
    assert route_after_kb({}) == "end"


def test_unanswerable_question_ends_with_ticket_offer_and_saved_draft():
    """
    Full graph run: "I don't know" -> offer message, awaiting set, draft stored.
    WHY IT MATTERS: the draft is what later lets the human agent see what was
    already tried; `awaiting` is what lets the next turn understand "yes".
    """
    result = RAGResult(
        answerable=False, answer=DONT_KNOW, standalone_question="Can I expense a standing desk?",
        consulted=["Expenses (hr/expenses.md)"], best_distance=0.55,
        reason=EscalationReason.LLM_INSUFFICIENT,
    )
    graph = build_graph(chain=FakeChain(result), llm=ScriptedLLM(), provider=MockTicketProvider())
    cfg = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"messages": [{"role": "user", "content": "standing desk expense?"}]}, cfg)

    assert out["awaiting"] == "ticket_confirmation"
    assert out["ticket_draft"]["question"] == "Can I expense a standing desk?"
    assert out["ticket_draft"]["consulted_sources"] == ["Expenses (hr/expenses.md)"]
    assert "create one" in out["messages"][-1]["content"]


def test_answered_question_does_not_offer_a_ticket():
    """
    A grounded answer ends the turn with the answer and sources, and no ticket offer.
    WHY IT MATTERS: the offer must appear only on escalation, never alongside an answer.
    """
    result = RAGResult(
        answerable=True, answer="Restart the client.", standalone_question="q",
        sources=["VPN Setup (it/vpn.md)"], consulted=["VPN Setup (it/vpn.md)"], best_distance=0.2,
    )
    graph = build_graph(chain=FakeChain(result), llm=ScriptedLLM(), provider=MockTicketProvider())
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "vpn 809"}]},
        {"configurable": {"thread_id": "t2"}},
    )
    assert out.get("awaiting") is None
    assert "Restart the client." in out["messages"][-1]["content"]
    assert "Sources:" in out["messages"][-1]["content"]