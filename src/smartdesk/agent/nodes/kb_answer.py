"""
Graph node for the Knowledge-Base Query branch.

Reads the newest user message, calls the chain with the search method chosen for
THIS request (graph config "retrieval_mode"), and records the outcome in state.
It does not decide where to go next: it only sets `escalate`.
"""

import logging

from langchain_core.runnables import RunnableConfig  # installed with langgraph; type only

from smartdesk.agent.state import AgentState
from smartdesk.rag.chain import RAGChain
from smartdesk.rag.confidence import EscalationReason

logger = logging.getLogger(__name__)


def make_kb_answer_node(chain: RAGChain):
    """Build the node with the chain injected (tests plug in a fake)."""

    def kb_answer(state: AgentState, config: RunnableConfig) -> dict:
        """
        Answer from the KB, or flag the turn for escalation.

        The search method comes from config["configurable"]["retrieval_mode"], set
        per request by the API, so it applies to this turn only and is never stored.
        Any exception becomes an escalation with reason ERROR.
        """
        question = state["messages"][-1]["content"]
        history = state["messages"][:-1]
        configurable = config.get("configurable") or {}
        mode = configurable.get("retrieval_mode")
        top_k = configurable.get("top_k")

        try:
            options = {key: value for key, value in (("mode", mode), ("top_k", top_k))
                       if value is not None}
            result = chain.answer(question, history, **options)
        except Exception:
            logger.exception("RAG chain failed; escalating")
            return {
                "escalate": True, "escalation_reason": EscalationReason.ERROR.value,
                "answer": "", "sources": [], "retrieval": None,
                "ticket_draft": {
                    "question": question, "original_message": question,
                    "reason": EscalationReason.ERROR.value,
                    "consulted_sources": [], "best_distance": None,
                },
            }

        if not result.answerable:
            return {
                "escalate": True, "escalation_reason": result.reason.value,
                "answer": result.answer, "sources": [], "retrieval": result.retrieval,
                "ticket_draft": {
                    "question": result.standalone_question, "original_message": question,
                    "reason": result.reason.value, "consulted_sources": result.consulted,
                    "best_distance": result.best_distance,
                },
            }

        text = result.answer
        if result.sources:
            text += "\n\nSources: " + "; ".join(dict.fromkeys(result.sources))
        return {
            "escalate": False, "escalation_reason": None,
            "answer": result.answer, "sources": result.sources,
            "retrieval": result.retrieval, "ticket_draft": None,
            "messages": [{"role": "assistant", "content": text}],
        }

    return kb_answer