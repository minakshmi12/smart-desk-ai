"""
In-memory ticket provider for development and tests.

WHY THIS FILE EXISTS
--------------------
Lets you build and test the full agent flow (including integration tests and
evals) without Jira credentials, network, or cleanup of junk tickets. Because it
implements the same interface as JiraProvider, passing tests against the mock is
meaningful evidence the agent logic is right.
"""

from datetime import datetime, timezone
from itertools import count

from smartdesk.tickets.base import TicketNotFoundError, TicketProvider
from smartdesk.tickets.models import Ticket, TicketCreateRequest, TicketStatus
from smartdesk.utils.validation import normalize_email


class MockTicketProvider(TicketProvider):
    """Stores tickets in a dict. Data is lost when the process exits (by design)."""

    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._counter = count(1)  # HELP-1, HELP-2, ... like a real tracker

    def create_ticket(self, request: TicketCreateRequest) -> Ticket:
        """Create a ticket in memory with status OPEN."""
        ticket_id = f"HELP-{next(self._counter)}"
        now = datetime.now(timezone.utc)
        ticket = Ticket(
            id=ticket_id,
            title=request.title,
            status=TicketStatus.OPEN,
            raw_status="To Do",
            url=f"https://mock.local/browse/{ticket_id}",
            requester_email=request.requester_email,
            created_at=now,
            updated_at=now,
        )
        self._tickets[ticket_id] = ticket
        return ticket

    def get_ticket(self, ticket_id: str, requester_email: str) -> Ticket:
        """Return the ticket only if the email matches; otherwise act as 'not found'."""
        ticket = self._tickets.get(ticket_id.upper())
        if ticket is None or ticket.requester_email != normalize_email(requester_email):
            raise TicketNotFoundError(ticket_id)
        return ticket

    def find_tickets_by_email(self, requester_email: str, limit: int = 10) -> list[Ticket]:
        """Return this requester's tickets, newest first."""
        email = normalize_email(requester_email)
        mine = [t for t in self._tickets.values() if t.requester_email == email]
        mine.sort(key=lambda t: t.created_at, reverse=True)
        return mine[:limit]

    # ---- test helper (not part of the TicketProvider interface) ----
    def set_status(self, ticket_id: str, status: TicketStatus, raw_status: str) -> None:
        """
        Simulate a human agent moving the ticket along.

        WHY: lets integration tests cover "I created a ticket yesterday, is it done?"
        by changing the status between two agent calls.
        """
        t = self._tickets[ticket_id]
        self._tickets[ticket_id] = t.model_copy(
            update={
                "status": status,
                "raw_status": raw_status,
                "updated_at": datetime.now(timezone.utc),
            }
        )