"""
Graph node for the escalation transition: offer a ticket, don't create one.

WHY THIS FILE EXISTS
--------------------
The transition from "can't answer" to "ticket" needs the employee's consent
(creating a ticket is a write, and it will notify humans) and their email. This
node performs the first half: it tells them it doesn't know, asks permission,
and parks the context in state so the next turn can continue the flow.
"""

from smartdesk.agent.prompts import OFFER_TICKET_DEFAULT, OFFER_TICKET_ERROR
from smartdesk.agent.state import AgentState
from smartdesk.rag.confidence import EscalationReason


def offer_ticket_node(state: AgentState) -> dict:
    """
    Ask the employee whether to raise a ticket, and set `awaiting`.

    Setting awaiting="ticket_confirmation" is what makes the NEXT message
    ("yes please", "no thanks") get interpreted as an answer to this offer.
    The ticket_draft was already filled by kb_answer and is left untouched.

    The wording differs for ERROR (a system problem) versus the "I don't know"
    reasons, so the bot doesn't claim ignorance when the real issue is an outage.
    """
    is_error = state.get("escalation_reason") == EscalationReason.ERROR.value
    text = OFFER_TICKET_ERROR if is_error else OFFER_TICKET_DEFAULT
    return {
        "awaiting": "ticket_confirmation",
        "messages": [{"role": "assistant", "content": text}],
    }