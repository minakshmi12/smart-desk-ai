"""
Independent tests for the ticket tools (no agent, no graph, no LLM).

WHY THIS FILE EXISTS
--------------------
The tools are the only code between the agent and the ticket platform. These
tests prove they validate input, never raise, and never leak other people's
tickets, using the mock provider and small failing subclasses.
"""

import pytest

from smartdesk.tickets.base import TicketProviderError
from smartdesk.tickets.mock_provider import MockTicketProvider
from smartdesk.tickets.models import TicketCreateRequest, TicketStatus
from smartdesk.tickets.tools import build_ticket_description, create_support_ticket, lookup_tickets


class SpyProvider(MockTicketProvider):
    """Counts create calls, to prove invalid input never reaches the platform."""

    creates = 0

    def create_ticket(self, request):
        self.creates += 1
        return super().create_ticket(request)


class DownProvider(MockTicketProvider):
    """Every operation fails like an unreachable platform."""

    def create_ticket(self, request):
        raise TicketProviderError("down")

    def get_ticket(self, ticket_id, requester_email):
        raise TicketProviderError("down")

    def find_tickets_by_email(self, requester_email, limit=10):
        raise TicketProviderError("down")


def _seed(provider, email="jane@corp.com", title="VPN fails"):
    """Create a ticket straight on the provider (bypassing the tool)."""
    return provider.create_ticket(
        TicketCreateRequest(requester_email=email, title=title, description="d")
    )


# ------------------------------------------------------------- creation
def test_create_succeeds_and_normalises_email():
    """
    A valid request returns ok with the new ticket linked to the normalised email.
    WHY IT MATTERS: the happy path every successful escalation relies on.
    """
    r = create_support_ticket(MockTicketProvider(), "Jane@Corp.com", "VPN fails", "Error 809")
    assert r.ok and r.ticket.id == "HELP-1" and r.ticket.requester_email == "jane@corp.com"


@pytest.mark.parametrize("email,title", [("not-an-email", "VPN fails"), ("jane@corp.com", "ab")])
def test_invalid_input_is_rejected_before_the_provider_is_called(email, title):
    """
    A bad email or too-short title returns invalid_input and makes no platform call.
    WHY IT MATTERS: protects the platform from junk, and distinguishes a data
    problem (don't retry) from an outage (retry).
    """
    spy = SpyProvider()
    r = create_support_ticket(spy, email, title, "d")
    assert not r.ok and r.error == "invalid_input" and spy.creates == 0


def test_provider_failure_becomes_a_result_not_an_exception():
    """
    A platform outage returns provider_error instead of raising.
    WHY IT MATTERS: the agent must be able to say "try again" instead of crashing the chat.
    """
    r = create_support_ticket(DownProvider(), "jane@corp.com", "VPN fails", "d")
    assert not r.ok and r.error == "provider_error" and r.ticket is None


# --------------------------------------------------------------- lookup
def test_lookup_by_id_returns_the_ticket_for_its_owner():
    """
    The owner can fetch their ticket by ID, in any letter case.
    WHY IT MATTERS: employees type "help-1" as often as "HELP-1".
    """
    p = MockTicketProvider()
    t = _seed(p)
    r = lookup_tickets(p, "Jane@Corp.com", ticket_id="help-1")
    assert r.outcome == "found" and r.tickets[0].id == t.id


def test_lookup_by_id_for_someone_elses_ticket_is_not_found():
    """
    Another person's ticket is reported exactly like a missing one.
    WHY IT MATTERS: the privacy guarantee at the tool level. A failure is a data leak.
    """
    p = MockTicketProvider()
    _seed(p)
    assert lookup_tickets(p, "bob@corp.com", ticket_id="HELP-1").outcome == "not_found"
    assert lookup_tickets(p, "jane@corp.com", ticket_id="HELP-99").outcome == "not_found"


def test_lookup_by_email_lists_own_tickets_newest_first():
    """
    Listing returns only the requester's tickets, newest first.
    WHY IT MATTERS: this answers "what's the status of my ticket?" with no ID.
    """
    p = MockTicketProvider()
    old, new = _seed(p, title="Old issue"), _seed(p, title="New issue")
    _seed(p, email="bob@corp.com", title="Bob's issue")
    r = lookup_tickets(p, "jane@corp.com")
    assert r.outcome == "found" and [t.id for t in r.tickets] == [new.id, old.id]


def test_lookup_reflects_status_changes():
    """
    A status change made by a human agent shows up on the next lookup.
    WHY IT MATTERS: stale status is the main way a status bot loses trust.
    """
    p = MockTicketProvider()
    t = _seed(p)
    p.set_status(t.id, TicketStatus.DONE, "Resolved")
    assert lookup_tickets(p, "jane@corp.com", ticket_id=t.id).tickets[0].raw_status == "Resolved"


def test_lookup_with_no_tickets_and_bad_email_and_outage():
    """
    "none", "invalid_email" and "error" are three distinct outcomes.
    WHY IT MATTERS: each needs a different reply (no tickets / ask again / try later).
    """
    assert lookup_tickets(MockTicketProvider(), "new@corp.com").outcome == "none"
    assert lookup_tickets(MockTicketProvider(), "nope").outcome == "invalid_email"
    assert lookup_tickets(MockTicketProvider(), None).outcome == "invalid_email"
    assert lookup_tickets(DownProvider(), "jane@corp.com").outcome == "error"


# ---------------------------------------------------------- description
def test_description_carries_context_for_the_human_agent():
    """
    The body names the question, reason, documents checked and recent chat.
    WHY IT MATTERS: this is what lets the human agent avoid re-asking.
    """
    text = build_ticket_description(
        question="Can I expense a standing desk?", original_message="standing desk expense?",
        reason="Documents did not contain the answer.", consulted_sources=["Expenses (hr/expenses.md)"] * 2,
        best_distance=0.55,
        history=[{"role": "user", "content": "standing desk expense?"},
                 {"role": "assistant", "content": "I don't know"}],
    )
    assert "Question: Can I expense a standing desk?" in text
    assert text.count("- Expenses (hr/expenses.md)") == 1      # de-duplicated
    assert "0.55" in text and "Employee: standing desk expense?" in text


def test_description_limits_history_and_flattens_messages():
    """
    Only the last 6 messages are included, each flattened and truncated.
    WHY IT MATTERS: pasted logs must not bloat or break the ticket, and system
    messages must never leak into it.
    """
    history = [{"role": "system", "content": "SECRET PROMPT"}] + [
        {"role": "user", "content": f"msg {i}\nline2 " + "x" * 900} for i in range(10)
    ]
    text = build_ticket_description(
        question="q", original_message="q", reason="r", consulted_sources=[],
        best_distance=None, history=history,
    )
    assert "SECRET PROMPT" not in text and "msg 3" not in text and "msg 4" in text
    assert all(len(line) < 520 for line in text.splitlines())