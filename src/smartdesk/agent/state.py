"""
Conversation state that flows through the graph and persists between turns.

WHY THIS FILE EXISTS
--------------------
The ticket flow spans several turns (offer -> consent -> email -> create), so
anything needed later must live in state. Values are plain Python types so any
LangGraph checkpointer can serialise them.
"""

import operator
from typing import Annotated, Literal, TypedDict

# After this many unusable replies to "what's your email?", the agent stops asking.
MAX_EMAIL_ATTEMPTS = 3


class TicketDraft(TypedDict, total=False):
    """
    A ticket that has been proposed but not yet created.

    WHY STORED IN STATE: the context (question, documents already checked) exists
    only at escalation time; it must survive until the employee consents and
    gives their email, so the human agent never has to re-ask.
    """

    title: str                    # optional; make_title() falls back to the question
    question: str                 # standalone version of what the employee needs
    original_message: str         # their exact words
    reason: str                   # EscalationReason value, or "user_requested"
    consulted_sources: list[str]  # documents already checked
    best_distance: float | None


class AgentState(TypedDict, total=False):
    """
    Shared graph state.

    messages: chat history. The `operator.add` reducer means nodes RETURN only new
        messages and LangGraph appends them.
    route: the router's decision for THIS turn (which node runs next).
    email: the employee's validated, normalised email once known. Reused for
        later tickets and status checks, so they are never asked twice.
    ticket_id: ticket the employee asked about (status flow).
    pending_route: what to do once the email arrives: "create_ticket" or "ticket_status".
    email_attempts: unusable replies so far (caps the email loop).
    answer / sources / escalate / escalation_reason: result of the last KB lookup.
    ticket_draft: see TicketDraft.
    awaiting: what the NEXT user message is a reply to. The router reads this first.
    """

    messages: Annotated[list[dict], operator.add]
    route: str
    email: str | None
    ticket_id: str | None
    pending_route: Literal["create_ticket", "ticket_status"] | None
    email_attempts: int
    answer: str
    sources: list[str]
    escalate: bool
    escalation_reason: str | None
    ticket_draft: TicketDraft | None
    awaiting: Literal["ticket_confirmation", "email"] | None
    
