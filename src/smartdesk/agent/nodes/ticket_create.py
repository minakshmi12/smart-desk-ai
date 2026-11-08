"""
Nodes for the Ticket Creation branch, including the human-in-the-loop steps.

Flow (each arrow is a separate user turn):
  [offer_ticket | start_ticket] -> employee says yes -> ask_email (if unknown)
                                -> employee gives email -> create_ticket
Nothing is written to Jira until create_ticket runs, and that is only reachable
after an explicit "yes" (see router._resolve_confirmation).
"""

import logging

from smartdesk.agent.prompts import (
    TICKET_FAILED_MESSAGE,
    TICKET_LOST_MESSAGE,
    ask_email_message,
    build_summary_messages,
    cancel_message,
    confirm_ticket_message,
    ticket_created_message,
)
from smartdesk.agent.state import MAX_EMAIL_ATTEMPTS, AgentState, TicketDraft
from smartdesk.llm.schemas import TicketSummary
from smartdesk.tickets.base import TicketProvider
from smartdesk.tickets.tools import build_ticket_description, create_support_ticket

logger = logging.getLogger(__name__)

# State fields reset when a ticket flow ends (success, cancel, or unrecoverable error).
_CLEARED = {
    "awaiting": None, "ticket_draft": None, "pending_route": None,
    "email_attempts": 0, "escalate": False,
}

# Human-readable escalation reasons for the ticket body.
REASON_TEXT = {
    "no_results": "No relevant document was found in the knowledge base.",
    "low_similarity": "Documents were found, but none were close enough to the question.",
    "llm_insufficient": "Relevant documents were found, but they did not contain the answer.",
    "ungrounded": "The assistant could not support an answer with the documents.",
    "error": "The assistant hit a technical error while searching.",
    "user_requested": "The employee asked for a ticket directly.",
}


def make_title(draft: TicketDraft) -> str:
    """
    Produce a valid single-line ticket title from a draft.

    Prefers an explicit title, then the question, then the raw message. Newlines
    are collapsed (Jira rejects them in summaries), length is capped at 100, and
    a floor of 3 characters satisfies TicketCreateRequest's validator.
    """
    raw = draft.get("title") or draft.get("question") or draft.get("original_message") or ""
    one_line = " ".join(raw.split())
    if len(one_line) > 100:
        one_line = one_line[:97].rstrip() + "..."
    return one_line if len(one_line) >= 3 else "Support request"


def make_start_ticket_node(llm):
    """Build the 'employee asked for a ticket directly' node (LLM injected)."""

    def start_ticket(state: AgentState) -> dict:
        """
        Draft a ticket from the conversation and ask for confirmation.

        WHY CONFIRM EVEN THOUGH THEY ASKED: the employee asked for *a* ticket,
        not for this exact wording. Showing the title and details lets them
        catch a bad summary before it reaches the support team.

        If summarising fails, the employee's own message becomes the details,
        so an LLM outage doesn't block ticket creation.
        """
        user_text = state["messages"][-1]["content"]
        try:
            summary = llm.parse(build_summary_messages(state["messages"]), TicketSummary)
            title, problem = summary.title, summary.problem
        except Exception:
            logger.exception("Ticket summary failed; using the employee's message")
            title, problem = "", user_text

        draft: TicketDraft = {
            "question": problem,
            "original_message": user_text,
            "reason": "user_requested",
            "consulted_sources": [],
            "best_distance": None,
        }
        if title.strip():
            draft["title"] = title
        return {
            "ticket_draft": draft,
            "awaiting": "ticket_confirmation",
            "escalate": False,
            "messages": [{"role": "assistant",
                          "content": confirm_ticket_message(make_title(draft), problem)}],
        }

    return start_ticket


def ask_email_node(state: AgentState) -> dict:
    """
    Ask for the employee's email and set awaiting="email".

    Reached from three places: consent given without a known email, a status
    request without one, and a retry after an unusable reply. The wording
    (first ask vs retry, create vs status) comes from state.
    """
    text = ask_email_message(state.get("pending_route"), state.get("email_attempts", 0))
    return {"awaiting": "email", "messages": [{"role": "assistant", "content": text}]}


def make_create_ticket_node(provider: TicketProvider):
    """Build the node that writes the ticket (provider injected)."""

    def create_ticket(state: AgentState) -> dict:
        """
        Create the ticket from the saved draft and the collected email.

        Outcomes:
        - success         -> confirmation with ID and link; flow state cleared.
        - provider error  -> keep the draft and stay in "ticket_confirmation" so a
                             simple "yes" retries without re-entering anything.
        - invalid input   -> clear and ask the employee to start over (a retry
                             can't fix bad data).
        - missing data    -> same; defensive, e.g. if state was lost between turns.
        """
        draft, email = state.get("ticket_draft"), state.get("email")
        if not draft or not email:
            return {**_CLEARED, "messages": [{"role": "assistant", "content": TICKET_LOST_MESSAGE}]}

        description = build_ticket_description(
            question=draft.get("question") or draft.get("original_message", ""),
            original_message=draft.get("original_message", ""),
            reason=REASON_TEXT.get(draft.get("reason", ""), draft.get("reason", "unknown")),
            consulted_sources=draft.get("consulted_sources", []),
            best_distance=draft.get("best_distance"),
            history=state["messages"],
        )
        result = create_support_ticket(provider, email, make_title(draft), description)

        if result.ok:
            return {**_CLEARED, "messages": [
                {"role": "assistant", "content": ticket_created_message(result.ticket, email)}]}
        if result.error == "provider_error":
            return {"awaiting": "ticket_confirmation",
                    "messages": [{"role": "assistant", "content": TICKET_FAILED_MESSAGE}]}
        return {**_CLEARED, "messages": [{"role": "assistant", "content": TICKET_LOST_MESSAGE}]}

    return create_ticket


def cancel_ticket_node(state: AgentState) -> dict:
    """
    End a pending flow without doing anything, and clear its state.

    Handles both "the employee said no / cancel" and "we gave up after
    MAX_EMAIL_ATTEMPTS unusable replies", plus the status-lookup variant.
    """
    gave_up = state.get("email_attempts", 0) >= MAX_EMAIL_ATTEMPTS
    text = cancel_message(state.get("pending_route"), gave_up)
    return {**_CLEARED, "ticket_id": None, "messages": [{"role": "assistant", "content": text}]}
