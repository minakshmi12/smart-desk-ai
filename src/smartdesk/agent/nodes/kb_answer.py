"""
Graph node for the Knowledge-Base Query branch.

WHY THIS FILE EXISTS
--------------------
Adapts the RAG chain to the graph: reads the latest user message, calls the
chain, and writes its outcome into state. The node does NOT decide where to go
next: it only records `escalate`; the conditional edge in graph.py routes.
"""

import logging

from smartdesk.agent.state import AgentState
from smartdesk.rag.chain import RAGChain
from smartdesk.rag.confidence import EscalationReason

logger = logging.getLogger(__name__)


def make_kb_answer_node(chain: RAGChain):
    """
    Build the node function with the chain injected.

    WHY A FACTORY: LangGraph nodes receive only `state`. Closing over the chain
    lets tests plug in a fake without patching globals.
    """

    def kb_answer(state: AgentState) -> dict:
        """
        Answer from the KB, or flag the turn for escalation.

        Returns only the keys it changes (LangGraph merges them into state).
        `escalate` is set on every run so a True from an earlier turn can never
        leak into this one.

        ANY exception (OpenAI outage, model refusal, Chroma failure) becomes
        an escalation with reason ERROR. WHY: the employee still gets a useful
        path (a ticket) instead of a stack trace, and Jira may be up even when
        OpenAI isn't.
        """
        question = state["messages"][-1]["content"]
        history = state["messages"][:-1]

        try:
            result = chain.answer(question, history)
        except Exception:
            logger.exception("RAG chain failed; escalating")
            return {
                "escalate": True,
                "escalation_reason": EscalationReason.ERROR.value,
                "answer": "",
                "sources": [],
                "ticket_draft": {
                    "question": question,
                    "original_message": question,
                    "reason": EscalationReason.ERROR.value,
                    "consulted_sources": [],
                    "best_distance": None,
                },
            }

        if not result.answerable:
            return {
                "escalate": True,
                "escalation_reason": result.reason.value,
                "answer": result.answer,
                "sources": [],
                "ticket_draft": {
                    "question": result.standalone_question,
                    "original_message": question,
                    "reason": result.reason.value,
                    "consulted_sources": result.consulted,
                    "best_distance": result.best_distance,
                },
            }

        # Show titles so the employee can verify the answer. Deduplicate while
        # keeping order (one document may be cited via several chunks).
        text = result.answer
        if result.sources:
            text += "\n\nSources: " + "; ".join(dict.fromkeys(result.sources))
        return {
            "escalate": False,
            "escalation_reason": None,
            "answer": result.answer,
            "sources": result.sources,
            "ticket_draft": None,
            "messages": [{"role": "assistant", "content": text}],
        }

    return kb_answer