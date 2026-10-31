"""Node for greetings, thanks and "what can you do?" (no retrieval, no tools)."""

import logging

from smartdesk.agent.prompts import CHITCHAT_FALLBACK, build_chitchat_messages
from smartdesk.agent.state import AgentState

logger = logging.getLogger(__name__)


def make_chitchat_node(llm):
    """Build the chit-chat node (LLM injected)."""

    def chitchat(state: AgentState) -> dict:
        """
        Reply briefly. WHY A SEPARATE BRANCH: without it, "thanks!" would trigger a
        vector search, fail confidence checks, and offer the employee a ticket.
        A static fallback is used if the LLM call fails.
        """
        try:
            reply = llm.chat(build_chitchat_messages(state["messages"]), temperature=0.3)
        except Exception:
            logger.exception("Chit-chat reply failed; using fallback")
            reply = ""
        return {"messages": [{"role": "assistant", "content": reply.strip() or CHITCHAT_FALLBACK}]}

    return chitchat