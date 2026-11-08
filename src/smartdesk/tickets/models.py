"""
Provider-neutral data models for tickets.

WHY THIS FILE EXISTS
--------------------
Jira, Asana, and Notion all describe a ticket differently (Jira has "status
categories", Asana has "completed" booleans, Notion has free-form select
properties). The agent should not know any of that. These models are the common
language: each provider translates its own format into these, and the agent only
ever sees these.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from smartdesk.utils.validation import is_valid_email, normalize_email


class TicketStatus(str, Enum):
    """
    Normalized ticket status shared by all providers.

    WHY AN ENUM: Jira workflows are customizable ("Waiting for Vendor",
    "Triaged", ...), so raw status names are unpredictable. Mapping them into
    three stable buckets lets the agent give consistent answers and lets you
    test status logic without a live Jira. The original name is kept separately
    in Ticket.raw_status so no information is lost.

    Inheriting from `str` makes values JSON-serializable for the API layer.
    """

    OPEN = "open"                # not started yet
    IN_PROGRESS = "in_progress"  # someone is working on it
    DONE = "done"                # resolved or closed


class TicketCreateRequest(BaseModel):
    """
    Everything needed to create a ticket: the input to TicketProvider.create_ticket().

    WHY A MODEL (not loose arguments): the ticket_create node assembles these
    fields across several conversation turns (email first, then confirmation).
    One validated object makes it clear when the draft is complete and
    guarantees providers never receive a malformed request.
    """

    requester_email: str = Field(description="Employee's email; links the ticket to them")
    title: str = Field(min_length=3, max_length=200, description="Short summary line")
    description: str = Field(
        description=(
            "Full context: the employee's question, what the KB search found, and "
            "relevant conversation history, so the human agent doesn't re-ask."
        )
    )

    @field_validator("requester_email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        """Reject malformed emails at construction time and store a normalized form."""
        if not is_valid_email(v):
            raise ValueError(f"Invalid email address: {v!r}")
        return normalize_email(v)


class Ticket(BaseModel):
    """
    A ticket as the agent sees it, regardless of which platform stores it.

    Returned by create_ticket / get_ticket / find_tickets_by_email.
    """

    id: str                                # e.g. "HELP-123"
    title: str
    status: TicketStatus                   # normalized bucket, used for logic
    raw_status: str                        # platform's own label, shown to the user
                                           # ("Waiting for Vendor")
    url: str | None = None                 # link the employee can open
    requester_email: str | None = None     # None if the platform can't return it
    created_at: datetime | None = None
    updated_at: datetime | None = None