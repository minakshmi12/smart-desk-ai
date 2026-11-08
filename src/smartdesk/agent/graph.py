"""
Wires router, nodes and edges into the runnable agent.

WHY THIS FILE EXISTS
--------------------
This is the one place that shows the agent's whole control flow. Every edge is a
plain Python function over state, so routing is deterministic and testable
without an LLM.
"""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from smartdesk.agent.nodes.chitchat import make_chitchat_node
from smartdesk.agent.nodes.kb_answer import make_kb_answer_node
from smartdesk.agent.nodes.offer_ticket import offer_ticket_node
from smartdesk.agent.nodes.ticket_create import (
    ask_email_node,
    cancel_ticket_node,
    make_create_ticket_node,
    make_start_ticket_node,
)
from smartdesk.agent.nodes.ticket_status import make_ticket_status_node
from smartdesk.agent.router import make_router_node
from smartdesk.agent.state import AgentState
from smartdesk.llm.client import LLMClient
from smartdesk.rag.chain import RAGChain
from smartdesk.tickets.factory import get_ticket_provider

# Every node the router may select. Must match the names added below.
ROUTES = ["kb_answer", "chitchat", "start_ticket", "ask_email",
          "create_ticket", "ticket_status", "cancel_ticket"]


def pick_route(state: AgentState) -> str:
    """Conditional edge after the router: follow the `route` it wrote this turn."""
    return state.get("route") or "kb_answer"


def route_after_kb(state: AgentState) -> Literal["offer_ticket", "end"]:
    """
    THE ESCALATION EDGE: unanswered -> offer a ticket; answered -> finish.
    Deliberately trivial: all the intelligence lives in rag/confidence.py.
    """
    return "offer_ticket" if state.get("escalate") else "end"


def build_graph(chain=None, llm=None, provider=None, checkpointer=None):
    """
    Compile the agent.

    Args:
        chain: RAG chain (tests pass a fake).
        llm: used by the router, summariser and chit-chat (tests pass a scripted fake).
        provider: ticket platform (defaults to TICKET_PROVIDER from config; tests
            pass a mock so no network is touched).
        checkpointer: stores state between turns by thread_id. REQUIRED for the
            ticket flow: `awaiting`, `ticket_draft` and `email` must survive
            between messages. MemorySaver is in-process (dev only); use a
            database-backed saver in production so restarts keep conversations.

    Usage:
        graph = build_graph()
        cfg = {"configurable": {"thread_id": "employee-42"}}
        out = graph.invoke({"messages": [{"role": "user", "content": "..."}]}, cfg)
        print(out["messages"][-1]["content"])
    """
    llm = llm or LLMClient()
    provider = provider or get_ticket_provider()
    chain = chain or RAGChain(llm=llm)

    g = StateGraph(AgentState)
    g.add_node("router", make_router_node(llm))
    g.add_node("kb_answer", make_kb_answer_node(chain))
    g.add_node("offer_ticket", offer_ticket_node)
    g.add_node("chitchat", make_chitchat_node(llm))
    g.add_node("start_ticket", make_start_ticket_node(llm))
    g.add_node("ask_email", ask_email_node)
    g.add_node("create_ticket", make_create_ticket_node(provider))
    g.add_node("ticket_status", make_ticket_status_node(provider))
    g.add_node("cancel_ticket", cancel_ticket_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", pick_route, {r: r for r in ROUTES})
    g.add_conditional_edges(
        "kb_answer", route_after_kb, {"offer_ticket": "offer_ticket", "end": END}
    )
    for node in ("offer_ticket", "chitchat", "start_ticket", "ask_email",
                 "create_ticket", "ticket_status", "cancel_ticket"):
        g.add_edge(node, END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())

