"""
Tests for the HTTP layer (no network, no API key, no real LLM).

WHY THIS FILE EXISTS
--------------------
Verifies the API contract: input validation, conversation ids, error hygiene, the
lock, and, in the last test, that a ticket flow really spans separate HTTP calls.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from smartdesk.api.locks import ThreadLocks
from smartdesk.api.main import create_app
from smartdesk.config import get_settings
from smartdesk.rag.chain import RAGResult
from smartdesk.rag.confidence import EscalationReason
from smartdesk.rag.vector_store import get_collection
from smartdesk.tickets.mock_provider import MockTicketProvider
from tests.fakes import ScriptedLLM
from contextlib import contextmanager
from smartdesk.api.locks import LockTimeout


class BusyLocks:
    """A lock backend whose conversation is always busy."""

    @contextmanager
    def hold(self, key):
        raise LockTimeout(key)
        yield  # unreachable; makes this a generator for @contextmanager


def test_health_reports_degraded_for_an_empty_knowledge_base():
    """
    With no ingested documents the service is up but "degraded", HTTP 200,
    and /health names the storage backend in use.
    """
    res = _client(FakeAgent()).get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "degraded", "ticket_provider": "mock",
                          "checkpointer": "memory", "kb_chunks": 0}


def test_busy_conversation_returns_409_with_retry_after_and_skips_the_agent():
    """
    If the conversation's lock can't be obtained, the client gets a retryable 409.

    WHY IT MATTERS: a stuck turn must not become a 500 (which alerts and
    retry-storms), and the agent must not run without the lock.
    """
    agent = FakeAgent()
    client = TestClient(create_app(agent=agent, locks=BusyLocks()))
    res = client.post("/chat", json={"message": "hi", "thread_id": "t1"})
    assert res.status_code == 409 and res.headers["retry-after"] == "2"
    assert agent.calls == []


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Dummy key and a temp vector store, so /health never touches real data."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "store"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeAgent:
    """
    Stand-in for the compiled graph: records calls, returns a canned reply,
    and can be told to fail with a recognisable internal message.
    """

    def __init__(self, reply="hello", awaiting=None, fail=False):
        self.reply, self.awaiting, self.fail = reply, awaiting, fail
        self.calls = []

    def invoke(self, payload, config):
        self.calls.append((payload, config))
        if self.fail:
            raise RuntimeError("secret internal detail")
        return {
            "messages": [payload["messages"][0], {"role": "assistant", "content": self.reply}],
            "awaiting": self.awaiting,
        }


def _client(agent):
    """Helper: a TestClient over an app that uses `agent`."""
    return TestClient(create_app(agent=agent))


# ---------------------------------------------------------------- /chat
def test_new_conversation_gets_a_thread_id_and_only_the_new_message_is_sent():
    """
    No thread_id in -> a fresh one out; the agent receives just this message.
    WHY IT MATTERS: history is restored by the checkpointer, so the API must not
    resend (or trust) client-supplied history.
    """
    agent = FakeAgent(reply="Hi there")
    body = _client(agent).post("/chat", json={"message": "  hello  "}).json()

    assert body["reply"] == "Hi there" and len(body["thread_id"]) == 32
    payload, config = agent.calls[0]
    assert payload == {"messages": [{"role": "user", "content": "hello"}]}   # trimmed
    assert config == {"configurable": {"thread_id": body["thread_id"]}}


def test_existing_thread_id_is_reused_and_awaiting_is_returned():
    """
    A supplied thread_id reaches the agent unchanged; the pending-question hint is passed on.
    WHY IT MATTERS: this is how multi-turn flows (consent, email) continue.
    """
    agent = FakeAgent(awaiting="ticket_confirmation")
    body = _client(agent).post("/chat", json={"message": "yes", "thread_id": "abc-123"}).json()
    assert body["thread_id"] == "abc-123" and body["awaiting"] == "ticket_confirmation"
    assert agent.calls[0][1]["configurable"]["thread_id"] == "abc-123"


def test_top_k_is_passed_to_the_agent_for_this_message():
    """The selected result count travels through graph config without changing the thread key."""
    agent = FakeAgent()
    body = _client(agent).post("/chat", json={"message": "vpn setup", "top_k": 8}).json()
    configurable = agent.calls[0][1]["configurable"]
    assert configurable == {"thread_id": body["thread_id"], "top_k": 8}


def test_top_k_one_is_accepted():
    """The minimum supported Top-K option reaches the agent config."""
    agent = FakeAgent()
    body = _client(agent).post("/chat", json={"message": "vpn setup", "top_k": 1}).json()
    assert body["thread_id"] and agent.calls[0][1]["configurable"]["top_k"] == 1


def test_top_k_rejects_values_outside_the_ui_choices():
    """Top-K is bounded to the supported user-facing options."""
    agent = FakeAgent()
    response = _client(agent).post("/chat", json={"message": "vpn setup", "top_k": 10})
    assert response.status_code == 422 and agent.calls == []


def test_source_view_displays_indexed_document_page_safely():
    """The source link shows indexed passage text for the requested source/page, escaped as HTML."""
    collection = get_collection()
    collection.upsert(
        ids=["it/guide.pdf#2#0", "it/guide.pdf#3#0"],
        documents=["Page two <script>alert('x')</script>", "Page three details"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        metadatas=[
            {"source": "it/guide.pdf", "title": "VPN Guide", "page": 2, "chunk_index": 0},
            {"source": "it/guide.pdf", "title": "VPN Guide", "page": 3, "chunk_index": 0},
        ],
    )

    response = _client(FakeAgent()).get("/sources", params={"source": "it/guide.pdf", "page": 2})

    assert response.status_code == 200 and "VPN Guide" in response.text
    assert "Page two &lt;script&gt;" in response.text and "Page three details" not in response.text


def test_source_view_returns_404_for_unknown_source():
    """The source viewer serves only documents present in the indexed KB."""
    response = _client(FakeAgent()).get("/sources", params={"source": "../../.env"})
    assert response.status_code == 404


@pytest.mark.parametrize("payload", [
    {"message": ""},
    {"message": "   "},
    {"message": "x" * 4001},
    {"message": "hi", "thread_id": "../etc/passwd"},
    {"message": "hi", "thread_id": "has space"},
    {"message": "hi", "thread_id": "a" * 101},
    {},
])
def test_invalid_requests_are_rejected_without_calling_the_agent(payload):
    """
    Blank/oversized messages and unsafe thread ids get a 422 and cost nothing.
    WHY IT MATTERS: protects LLM spend and keeps odd strings out of lock/storage keys.
    """
    agent = FakeAgent()
    assert _client(agent).post("/chat", json=payload).status_code == 422
    assert agent.calls == []


def test_unexpected_failure_returns_a_generic_500_without_internal_details():
    """
    An exception escaping the agent becomes a safe 500.
    WHY IT MATTERS: stack traces and internal messages must never reach the client.
    """
    res = _client(FakeAgent(fail=True)).post("/chat", json={"message": "hi"})
    assert res.status_code == 500
    assert "secret internal detail" not in res.text and "went wrong" in res.json()["detail"]


# -------------------------------------------------------------- /health
def test_health_reports_degraded_for_an_empty_knowledge_base():
    """
    With no ingested documents the service is up but "degraded", HTTP 200.
    WHY IT MATTERS: a load balancer shouldn't kill a server that re-ingestion would fix.
    """
    res = _client(FakeAgent()).get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "degraded", "ticket_provider": "mock", "kb_chunks": 0}


# ---------------------------------------------------------------- locks
def test_requests_for_the_same_thread_never_overlap():
    """
    Five threads using one conversation id run strictly one at a time.
    WHY IT MATTERS: this is what prevents two overlapping "yes" messages from
    creating two tickets.
    """
    locks, guard = ThreadLocks(), threading.Lock()
    state = {"active": 0, "peak": 0}

    def work():
        with locks.hold("t1"):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1

    threads = [threading.Thread(target=work) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 1


def test_different_threads_do_not_block_each_other():
    """
    A long-held lock on one conversation doesn't delay another.
    WHY IT MATTERS: one slow turn must not stall unrelated employees.
    """
    locks, done = ThreadLocks(), threading.Event()

    def other():
        with locks.hold("b"):
            done.set()

    with locks.hold("a"):
        threading.Thread(target=other).start()
        assert done.wait(timeout=2)


# ------------------------------------------------- real graph over HTTP
def test_ticket_flow_spans_separate_http_calls_and_threads_are_isolated():
    """
    Offer -> "yes, email" creates a ticket across two requests; another
    conversation is unaffected.

    WHY IT MATTERS: proves the checkpointer + thread_id wiring works through the
    API (not just in graph tests), and that one employee's "yes" can't confirm
    another employee's pending offer.
    """
    from smartdesk.agent.graph import build_graph

    class Chain:
        def answer(self, question, history=None, department=None):
            return RAGResult(
                answerable=False, answer="I don't know",
                standalone_question="Can I expense a standing desk?",
                consulted=["Expenses (hr/expenses.md)"], best_distance=0.55,
                reason=EscalationReason.LLM_INSUFFICIENT,
            )

    provider = MockTicketProvider()
    client = _client(build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider))

    first = client.post("/chat", json={"message": "Can I expense a standing desk?"}).json()
    assert first["awaiting"] == "ticket_confirmation"

    done = client.post("/chat", json={"thread_id": first["thread_id"],
                                      "message": "yes, jane@corp.com"}).json()
    assert "HELP-1" in done["reply"] and done["awaiting"] is None
    assert len(provider.find_tickets_by_email("jane@corp.com")) == 1

    # A different conversation saying "yes" has nothing to confirm.
    other = client.post("/chat", json={"thread_id": "someone-else", "message": "yes"}).json()
    assert len(provider.find_tickets_by_email("jane@corp.com")) == 1
    assert other["awaiting"] == "ticket_confirmation"   # treated as a fresh question, offered again