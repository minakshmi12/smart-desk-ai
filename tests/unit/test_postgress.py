"""
Real-Postgres checks: shared state and cross-instance locking.

Run with:  docker compose up -d postgres
           TEST_DATABASE_URL=postgresql://smartdesk:smartdesk@localhost:5432/smartdesk pytest tests/integration/test_postgres.py
"""

import os
import threading
import time
import uuid

import pytest

pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from smartdesk.agent.graph import build_graph  # noqa: E402
from smartdesk.api.persistence import open_persistence  # noqa: E402
from smartdesk.config import get_settings  # noqa: E402
from smartdesk.tickets.mock_provider import MockTicketProvider  # noqa: E402
from tests.fakes import ScriptedLLM  # noqa: E402
from tests.integration.test_persistence import Chain, _turn  # noqa: E402

DB = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="set TEST_DATABASE_URL to run Postgres tests")


@pytest.fixture(autouse=True)
def pg_env(monkeypatch):
    """Select the Postgres backend for the test database."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", DB or "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_second_instance_continues_a_conversation_started_on_the_first():
    """
    Two instances share one database: instance A makes the offer, instance B handles "yes".

    WHY IT MATTERS: this is the load-balancer scenario (the next request lands on
    a different worker). A random thread_id keeps re-runs from colliding.
    """
    provider, cfg = MockTicketProvider(), {"configurable": {"thread_id": uuid.uuid4().hex}}
    with open_persistence() as a, open_persistence() as b:
        agent_a = build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider, checkpointer=a.checkpointer)
        agent_b = build_graph(chain=Chain(), llm=ScriptedLLM(), provider=provider, checkpointer=b.checkpointer)
        assert _turn(agent_a, cfg, "Can I expense a standing desk?")["awaiting"] == "ticket_confirmation"
        assert "HELP-1" in _turn(agent_b, cfg, "yes, jane@corp.com")["messages"][-1]["content"]


def test_advisory_lock_excludes_across_instances():
    """
    Turns for one conversation never overlap even when they run on different instances.

    WHY IT MATTERS: the real-database proof of what the fake-server tests assume.
    """
    key, state, guard = uuid.uuid4().hex, {"active": 0, "peak": 0}, threading.Lock()
    with open_persistence() as a, open_persistence() as b:
        def work(locks):
            with locks.hold(key):
                with guard:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with guard:
                    state["active"] -= 1

        threads = [threading.Thread(target=work, args=(x.locks,)) for x in (a, b, a, b)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert state["peak"] == 1