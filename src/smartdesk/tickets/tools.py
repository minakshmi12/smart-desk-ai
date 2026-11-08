"""
Agent-facing ticket operations: create a ticket, look tickets up.

WHY THIS FILE EXISTS
--------------------
TicketProvider is the platform contract (it raises exceptions). The agent wants
something simpler: "do this, tell me the outcome". These functions
1. validate input,
2. call the provider,
3. convert every failure into a result object (they NEVER raise),
so graph nodes are plain if/else on the result and can be tested without a graph.
They are also the natural thing to expose as LLM tools later.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from smartdesk.tickets.base import TicketNotFoundError, TicketProvider, TicketProviderError
from smartdesk.tickets.models import Ticket, TicketCreateRequest
from smartdesk.utils.validation import is_valid_email

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreateResult:
    """
    Outcome of create_support_ticket.

    ok=True  -> `ticket` is set.
    ok=False -> `error` says why:
        "invalid_input"  : bad email/title (a bug or bad data; retrying won't help)
        "provider_error" : platform down/rejected it (retrying later may work)
    `detail` is for logs only and must never be shown to the employee (it can
    contain internal messages).
    """

    ok: bool
    ticket: Ticket | None = None
    error: Literal["invalid_input", "provider_error"] | None = None
    detail: str = ""


@dataclass(frozen=True)
class StatusResult:
    """
    Outcome of lookup_tickets.

    outcome:
      "found"         -> `tickets` has 1+ entries (newest first when listing)
      "none"          -> the person has no tickets (normal, not an error)
      "not_found"     -> a specific ID doesn't exist OR belongs to someone else
                         (deliberately indistinguishable: no existence leaks)
      "invalid_email" -> email missing or malformed
      "error"         -> platform failure
    """

    outcome: Literal["found", "none", "not_found", "invalid_email", "error"]
    tickets: list[Ticket] = field(default_factory=list)


def create_support_ticket(
    provider: TicketProvider, requester_email: str, title: str, description: str
) -> CreateResult:
    """
    Create a ticket. Never raises.

    Args:
        provider: any TicketProvider (Jira in production, mock in tests).
        requester_email: validated and normalised by TicketCreateRequest.
        title: 3-200 chars, single line (Jira rejects newlines in summaries).
        description: full context for the human agent (see build_ticket_description).

    WHY VALIDATION HAPPENS HERE: building TicketCreateRequest runs the email and
    title validators, so a malformed request is rejected before any HTTP call.
    """
    try:
        request = TicketCreateRequest(
            requester_email=requester_email, title=title, description=description
        )
    except ValidationError as exc:
        logger.warning("Rejected ticket request: %s", exc)
        return CreateResult(ok=False, error="invalid_input", detail=str(exc))

    try:
        ticket = provider.create_ticket(request)
    except TicketProviderError as exc:
        logger.error("Ticket creation failed: %s", exc)
        return CreateResult(ok=False, error="provider_error", detail=str(exc))
    return CreateResult(ok=True, ticket=ticket)


def lookup_tickets(
    provider: TicketProvider,
    requester_email: str | None,
    ticket_id: str | None = None,
    limit: int = 5,
) -> StatusResult:
    """
    Look up one ticket (if `ticket_id` is given) or list the requester's most
    recent ones. Never raises.

    The email is ALWAYS required, even with an ID: it is the ownership check that
    stops people reading each other's tickets by guessing IDs.

    WHY not_found COVERS "belongs to someone else": the provider raises the same
    exception for both, so the reply can't confirm that another person's ticket exists.
    """
    if not requester_email or not is_valid_email(requester_email):
        return StatusResult("invalid_email")
    try:
        if ticket_id:
            return StatusResult("found", [provider.get_ticket(ticket_id, requester_email)])
        tickets = provider.find_tickets_by_email(requester_email, limit=limit)
    except TicketNotFoundError:  # must be caught BEFORE its parent class
        return StatusResult("not_found")
    except TicketProviderError:
        logger.exception("Ticket lookup failed")
        return StatusResult("error")
    return StatusResult("found", tickets) if tickets else StatusResult("none")


def build_ticket_description(
    *,
    question: str,
    original_message: str,
    reason: str,
    consulted_sources: list[str],
    best_distance: float | None,
    history: list[dict],
    max_history: int = 6,
    max_chars: int = 500,
) -> str:
    """
    Assemble the ticket body so the human agent doesn't have to re-ask anything.

    Contents: the question, the employee's exact words, why the bot gave up,
    which documents were already checked, and the last few chat messages.

    WHY EACH MESSAGE IS FLATTENED AND TRUNCATED: Jira descriptions are built
    one paragraph per line (see _adf), and a pasted log or stack trace could
    otherwise bloat or break the ticket.

    Keyword-only arguments, so call sites stay readable and the tickets
    package doesn't depend on the agent's state types.
    """
    lines = [f"Question: {question}"]
    if original_message and original_message != question:
        lines.append(f"Employee's exact words: {original_message}")
    lines.append(f"Why it was escalated: {reason}")

    if consulted_sources:
        lines.append("Documents already checked:")
        lines.extend(f"- {s}" for s in dict.fromkeys(consulted_sources))
    else:
        lines.append("Documents already checked: none")
    if best_distance is not None:
        lines.append(f"Closest document match (cosine distance, lower is closer): {best_distance:.2f}")

    recent = [m for m in history if m.get("role") in ("user", "assistant")][-max_history:]
    if recent:
        lines += ["", "Recent conversation:"]
        for m in recent:
            who = "Employee" if m["role"] == "user" else "SmartDesk"
            lines.append(f"{who}: {' '.join(m['content'].split())[:max_chars]}")

    lines += ["", "Created automatically by SmartDesk AI."]
    return "\n".join(lines)