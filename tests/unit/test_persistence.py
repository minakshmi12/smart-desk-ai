"""
Proves conversations survive a server restart (SQLite, real files, no network).

WHY THIS FILE EXISTS
--------------------
Unit tests can't show that the whole agent state (the pending question, the
ticket draft) serialises and restores correctly. This runs a conversation across
two separate "server lifetimes" that share only the database file.
"""

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

from smartdesk.agent.graph import build_graph  # noqa: E402
from smartdesk.api.persistence import open_persistence  # noqa: E402
from smartdesk.config import get_settings  # noqa: E402
from smartdesk.rag.chain import RAGResult  # noqa: E402
from smartdesk.rag.confidence import EscalationReason  # noqa: E402
from smartdesk.tickets.mock_provider import MockTicketProvider  # noqa: E402
from tests.fakes import ScriptedLLM  # noqa: E402


@pytest.fixture(autouse=True)
def sqlite_env(monkeypatch, tmp_path):
    """Point CHECKPOINTER at a throwaway SQLite file and reset the settings cache."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHECKPOINTER", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ckpt.sqlite"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class Chain:
    """RAG stub that never knows the answer, so every question escalates."""

    def answer(self, question, history=None, department=None):
        return RAGResult(
            answerable=False, answer="I don't know",
            standalone_question="Can I expense a standing desk?",
            consulted=["Expenses (hr/expenses.md)"], best_distance=0.55,
            reason=EscalationReason.LLM_INSUFFICIENT,
        )


def _turn(agent, cfg, text):
    """Send one employee message and return the full resulting state."""
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, cfg)


def test_pending_ticket_offer_survives_a_restart():
    """
    Offer in "process 1"; "yes, email" in a brand-new "process 2" creates the ticket.

    WHY IT MATTERS: with the in-memory saver a restart (or a deploy) between the
    offer and the answer lost the draft, and the employee's "yes" meant nothing.
    The email check also proves the state was restored from the file, not memory.
    """
    provider, cfg = MockTicketProvider(), {"configurable": {"thread_id": "t1"}}

    with open_persistence() as p:  # first server lifetime
        agent = build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider,
                            checkpointer=p.checkpointer)
        assert _turn(agent, cfg, "Can I expense a standing desk?")["awaiting"] == "ticket_confirmation"

    with open_persistence() as p:  # new connection, new graph objects
        agent = build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider,
                            checkpointer=p.checkpointer)
        out = _turn(agent, cfg, "yes, jane@corp.com")
        assert "HELP-1" in out["messages"][-1]["content"]
        assert agent.get_state(cfg).values["email"] == "jane@corp.com"
    assert len(provider.find_tickets_by_email("jane@corp.com")) == 1


def test_conversations_in_the_same_file_stay_separate():
    """
    One employee's pending offer can't be confirmed from another thread_id.
    WHY IT MATTERS: persistence must not merge conversations.
    """
    provider = MockTicketProvider()
    with open_persistence() as p:
        agent = build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider,
                            checkpointer=p.checkpointer)
        _turn(agent, {"configurable": {"thread_id": "a"}}, "Can I expense a standing desk?")
        _turn(agent, {"configurable": {"thread_id": "b"}}, "yes, bob@corp.com")
    assert provider.find_tickets_by_email("bob@corp.com") == []