"""
Unit tests for the ticket layer (models, validation, MockTicketProvider, Jira helpers).

WHY THIS FILE EXISTS
--------------------
The agent nodes trust the TicketProvider contract: emails are normalized, a
ticket is only readable by its owner, and status changes become visible. If any
of those guarantees silently break, the bot could leak another employee's ticket
or report stale status. These tests pin those guarantees down.

WHY THE MOCK PROVIDER IS TESTED HERE
------------------------------------
MockTicketProvider implements the same interface as JiraProvider. Integration
tests and evals run against the mock, so the mock must behave like the real
thing. These tests keep it honest.

WHAT IS NOT COVERED HERE
------------------------
JiraProvider's HTTP behavior (retries, 404 handling, ADF payloads). That needs
a fake httpx transport and belongs in a separate test file.

RUN WITH: pytest tests/unit/test_ticket_providers.py -v
"""

import pytest

from smartdesk.tickets.base import TicketNotFoundError
from smartdesk.tickets.jira_provider import requester_label
from smartdesk.tickets.mock_provider import MockTicketProvider
from smartdesk.tickets.models import TicketCreateRequest, TicketStatus


def _req(email="Jane@Corp.com", title="VPN fails on Mac"):
    """
    Helper: build a valid TicketCreateRequest with sensible defaults.

    WHY IT EXISTS: almost every test needs a request object. A helper keeps
    the tests short and lets each one override only what it cares about
    (usually the email). The default email is deliberately written with mixed
    case ("Jane@Corp.com") so normalization is exercised in every test that
    uses it, not only the one that tests it explicitly.

    Not a test itself: the leading underscore keeps pytest from collecting it.
    """
    return TicketCreateRequest(
        requester_email=email, title=title, description="Error 809"
    )


def test_email_is_normalized_and_validated():
    """
    Emails are lowercased on the way in, and malformed ones are rejected.

    WHAT IT CHECKS:
    1. "Jane@Corp.com" becomes "jane@corp.com" after constructing the request.
    2. "not-an-email" raises ValueError (pydantic's ValidationError subclasses it).

    WHY IT MATTERS: ownership checks and "find my tickets" compare emails as
    strings. If normalization breaks, an employee who types "Jane@Corp.com"
    today and "jane@corp.com" tomorrow would appear to be two different people
    and could not see their own tickets. If validation breaks, a ticket could be
    created with an address nobody can be reached at.

    LAYER UNDER TEST: TicketCreateRequest's field validator, which relies on
    utils/validation.py.
    """
    assert _req().requester_email == "jane@corp.com"
    with pytest.raises(ValueError):
        _req("not-an-email")


def test_owner_can_read_but_others_cannot():
    """
    A ticket is readable by its creator and by nobody else.

    WHAT IT CHECKS:
    1. Jane creates a ticket, and fetching it with her email succeeds.
    2. Fetching the same ticket ID with a different email raises
       TicketNotFoundError (not a different error, and not a successful result).

    WHY IT MATTERS: this is the privacy guarantee. Employees type ticket IDs
    themselves, and IDs like HELP-1, HELP-2 are guessable. Without this check
    anyone could read colleagues' tickets, which may describe HR matters.
    Raising "not found" rather than "forbidden" is intentional: it avoids
    confirming that the ticket exists.

    IF IT FAILS: the agent's status node would leak other people's ticket
    details. Treat a failure here as a security bug, not a cosmetic one.
    """
    p = MockTicketProvider()
    t = p.create_ticket(_req())
    assert p.get_ticket(t.id, "jane@corp.com").id == t.id
    with pytest.raises(TicketNotFoundError):
        p.get_ticket(t.id, "someone-else@corp.com")  # privacy check


def test_status_updates_are_visible():
    """
    When a human agent changes a ticket's status, the next lookup shows it.

    WHAT IT CHECKS: after create_ticket (status OPEN), calling set_status(...)
    to DONE with the label "Resolved" is reflected in the result of
    find_tickets_by_email, and the status is the DONE enum member.

    WHY IT MATTERS: the "check ticket status" branch is only useful if it
    reports current state. This simulates the real-world flow: the employee
    raises a ticket, a human works on it, and the employee asks again later.
    It also verifies that set_status replaces the stored ticket rather than
    leaving a stale copy behind.

    NOTE: set_status is a test helper on the mock, not part of the
    TicketProvider interface. Real providers get their status changes from the
    platform itself.
    """
    p = MockTicketProvider()
    t = p.create_ticket(_req())
    p.set_status(t.id, TicketStatus.DONE, "Resolved")
    assert p.find_tickets_by_email("jane@corp.com")[0].status is TicketStatus.DONE


def test_requester_label_is_stable_and_jql_safe():
    """
    The Jira label derived from an email is predictable and contains safe characters.

    WHAT IT CHECKS: " Jane@Corp.com " (with stray spaces and capitals) becomes
    exactly "req-jane_at_corp.com".

    WHY IT MATTERS: JiraProvider links tickets to people through this label and
    later searches for it with JQL. Two properties are required:
    - STABLE: the same person must always produce the same label, however they
      typed their email, or "find my tickets" misses tickets created earlier.
    - SAFE: Jira labels cannot contain spaces, and the label is embedded inside
      a JQL string. Replacing "@" and other odd characters keeps both Jira and
      the query happy.

    IF IT FAILS: tickets could be created but become unfindable, or lookups
    could return nothing for existing users.
    """
    assert requester_label(" Jane@Corp.com ") == "req-jane_at_corp.com"


# ----------------------------------------------------------------------
# Additional tests covering gaps in the original four
# ----------------------------------------------------------------------
def test_created_ticket_starts_open_with_sequential_ids():
    """
    New tickets begin as OPEN and get incrementing IDs (HELP-1, HELP-2, ...).

    WHAT IT CHECKS: two tickets created in a row have IDs "HELP-1" and
    "HELP-2", and the first one's status is TicketStatus.OPEN.

    WHY IT MATTERS: the status node and the evals assume a freshly created
    ticket reads as "open". The ID format also matters because the router
    extracts IDs like "HELP-123" from user messages, and the mock should look
    like the real tracker.
    """
    p = MockTicketProvider()
    first = p.create_ticket(_req())
    second = p.create_ticket(_req())
    assert (first.id, second.id) == ("HELP-1", "HELP-2")
    assert first.status is TicketStatus.OPEN


def test_ticket_id_lookup_is_case_insensitive():
    """
    "help-1" finds the same ticket as "HELP-1".

    WHAT IT CHECKS: get_ticket with a lowercase ID returns the ticket.

    WHY IT MATTERS: employees type IDs in whatever case they like. The LLM
    router may also pass the ID through unchanged. The provider should absorb
    that so users never see "ticket not found" over capitalization.
    """
    p = MockTicketProvider()
    t = p.create_ticket(_req())
    assert p.get_ticket("help-1", "jane@corp.com").id == t.id


def test_unknown_ticket_id_raises_not_found():
    """
    Asking for an ID that was never created raises TicketNotFoundError.

    WHAT IT CHECKS: get_ticket("HELP-999", ...) on an empty provider raises.

    WHY IT MATTERS: the status node relies on this specific exception to reply
    with "I couldn't find that ticket, could you double-check the ID?" instead
    of crashing or treating it as a system outage.
    """
    p = MockTicketProvider()
    with pytest.raises(TicketNotFoundError):
        p.get_ticket("HELP-999", "jane@corp.com")


def test_find_tickets_returns_only_own_tickets_newest_first():
    """
    Listing returns the requester's tickets only, ordered newest first.

    WHAT IT CHECKS: Jane creates two tickets and Bob creates one. Jane's list
    has exactly two entries, in reverse creation order, and none of Bob's.

    WHY IT MATTERS: two guarantees in one test.
    - ISOLATION: the same privacy rule as get_ticket, applied to listing.
    - ORDERING: when an employee says "what's the status of my ticket?"
      without an ID, the agent will usually report the most recent one, so
      index 0 must be the newest.
    """
    p = MockTicketProvider()
    old = p.create_ticket(_req("jane@corp.com", title="Old issue"))
    new = p.create_ticket(_req("jane@corp.com", title="New issue"))
    p.create_ticket(_req("bob@corp.com", title="Bob's issue"))

    mine = p.find_tickets_by_email("jane@corp.com")
    assert [t.id for t in mine] == [new.id, old.id]


def test_find_tickets_returns_empty_list_when_none_exist():
    """
    An employee with no tickets gets an empty list, not an error.

    WHAT IT CHECKS: find_tickets_by_email for an unknown person returns [].

    WHY IT MATTERS: "you don't have any tickets yet" is a normal answer, not a
    failure. The status node distinguishes an empty list (friendly message)
    from TicketProviderError (system problem), so the two must not be mixed up.
    """
    p = MockTicketProvider()
    assert p.find_tickets_by_email("nobody@corp.com") == []


