"""
The contract every ticket platform must satisfy.

WHY THIS FILE EXISTS
--------------------
This is the most important seam in the project. The agent nodes depend on
`TicketProvider`, never on Jira directly. That gives you:
- swap Jira -> Asana/Notion by writing one new class, with no agent changes,
- develop and test everything with MockTicketProvider and no network,
- one place defining what "create / read / list" mean for tickets.
"""

from abc import ABC, abstractmethod

from smartdesk.tickets.models import Ticket, TicketCreateRequest


class TicketProviderError(Exception):
    """
    The platform failed (network down, bad credentials, rate limit...).

    WHY A CUSTOM EXCEPTION: nodes catch this single type to respond gracefully
    ("I couldn't reach the ticket system, please try again") without knowing
    whether httpx, requests, or an SDK raised the original error.
    """


class TicketNotFoundError(TicketProviderError):
    """
    No ticket with that ID exists *for this requester*.

    WHY A SEPARATE SUBCLASS: "not found" is an expected user situation (typo in
    the ID), not a system fault. The status node handles it with a friendly
    message instead of an apology about outages. It is also raised when the
    ticket exists but belongs to someone else, so the existence of other
    people's tickets is never revealed.
    """


class TicketProvider(ABC):
    """
    Abstract interface for creating and reading support tickets.

    The three methods map one-to-one onto the agent's two ticket capabilities:
    - create_ticket                   -> the Ticket Creation (write) branch
    - get_ticket / find_tickets_by_email -> the Ticket Status (read) branch
    """

    @abstractmethod
    def create_ticket(self, request: TicketCreateRequest) -> Ticket:
        """
        Create a ticket on the platform and return it (including its new ID).

        Implementations MUST record the requester's email in a way that
        find_tickets_by_email can later search on.

        Raises:
            TicketProviderError: platform unreachable or rejected the request.
        """

    @abstractmethod
    def get_ticket(self, ticket_id: str, requester_email: str) -> Ticket:
        """
        Fetch one ticket by ID, but only if it belongs to `requester_email`.

        WHY requester_email IS REQUIRED: this is the privacy check. The employee
        types the ID, and nothing else stops them from typing a colleague's.

        Raises:
            TicketNotFoundError: no such ticket, or it belongs to someone else.
            TicketProviderError: platform failure.
        """

    @abstractmethod
    def find_tickets_by_email(self, requester_email: str, limit: int = 10) -> list[Ticket]:
        """
        List the requester's most recent tickets, newest first.

        WHY: handles "what's the status of my ticket?" when the employee doesn't
        remember an ID. Returns an empty list (not an error) if they have none.

        Raises:
            TicketProviderError: platform failure.
        """