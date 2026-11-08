"""
Node for the Ticket Status branch (a read-only operation).

The router guarantees an email is known before routing here (or sends the
employee through ask_email first), so this node needs no consent step.
"""

from smartdesk.agent.prompts import STATUS_ERROR_MESSAGE, ask_email_message
from smartdesk.agent.state import AgentState
from smartdesk.tickets.base import TicketProvider
from smartdesk.tickets.models import Ticket, TicketStatus
from smartdesk.tickets.tools import StatusResult, lookup_tickets

_STATUS_LABEL = {
    TicketStatus.OPEN: "Open (not started yet)",
    TicketStatus.IN_PROGRESS: "In progress",
    TicketStatus.DONE: "Done",
}


def format_ticket(t: Ticket) -> str:
    """
    One ticket as a short text block.

    Shows the normalised status AND the platform's own label ("Waiting for
    Vendor"), because the label is what employees recognise from Jira emails.
    """
    lines = [f"{t.id}: {t.title}",
             f"  Status: {_STATUS_LABEL[t.status]} (\"{t.raw_status}\")"]
    if t.updated_at:
        lines.append(f"  Last updated: {t.updated_at:%Y-%m-%d}")
    if t.url:
        lines.append(f"  {t.url}")
    return "\n".join(lines)


def build_status_reply(result: StatusResult, email: str, ticket_id: str | None) -> str:
    """
    Turn a StatusResult into the reply text.

    The "not_found" wording is identical for "doesn't exist" and "belongs to
    someone else", so the reply never confirms another person's ticket exists.
    """
    if result.outcome == "found":
        head = "Here's what I found:" if len(result.tickets) == 1 else "Here are your most recent tickets:"
        return head + "\n\n" + "\n\n".join(format_ticket(t) for t in result.tickets)
    if result.outcome == "none":
        return (f"I couldn't find any tickets linked to {email}. If you raised one under "
                "a different address, tell me that address.")
    if result.outcome == "not_found":
        return (f"I couldn't find ticket {ticket_id} linked to {email}. Please check the ID, "
                "or ask me to list your recent tickets.")
    return STATUS_ERROR_MESSAGE


def make_ticket_status_node(provider: TicketProvider):
    """Build the status node (provider injected)."""

    def ticket_status(state: AgentState) -> dict:
        """
        Look up one ticket or the employee's recent ones, and reply.

        If the email is somehow missing or invalid (defensive), fall back to
        asking for it instead of failing.
        """
        email, ticket_id = state.get("email"), state.get("ticket_id")
        result = lookup_tickets(provider, email, ticket_id=ticket_id, limit=5)

        if result.outcome == "invalid_email":
            return {"awaiting": "email", "pending_route": "ticket_status", "email_attempts": 0,
                    "messages": [{"role": "assistant",
                                  "content": ask_email_message("ticket_status", 0)}]}
        return {
            "awaiting": None, "pending_route": None, "ticket_id": None, "email_attempts": 0,
            "messages": [{"role": "assistant",
                          "content": build_status_reply(result, email, ticket_id)}],
        }

    return ticket_status
