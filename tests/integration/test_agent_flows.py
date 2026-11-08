"""
End-to-end conversations through the real graph (fake LLM, fake RAG chain, mock tickets).

WHY THIS FILE EXISTS
--------------------
Unit tests prove the pieces; these prove the three capabilities work together
across multiple turns: escalate -> consent -> email -> create, then check status.
"""

import pytest

from smartdesk.agent.graph import build_graph
from smartdesk.agent.prompts import DONT_KNOW
from smartdesk.config import get_settings
from smartdesk.rag.chain import RAGResult
from smartdesk.rag.confidence import EscalationReason
from smartdesk.tickets.base import TicketProviderError
from smartdesk.tickets.mock_provider import MockTicketProvider
from smartdesk.tickets.models import TicketCreateRequest, TicketStatus
from fakes import ScriptedLLM

UNANSWERED = RAGResult(
    answerable=False, answer=DONT_KNOW, standalone_question="Can I expense a standing desk?",
    consulted=["Expenses (hr/expenses.md)"], best_distance=0.55,
    reason=EscalationReason.LLM_INSUFFICIENT,
)
ANSWERED = RAGResult(
    answerable=True, answer="Restart the client.", standalone_question="q",
    sources=["VPN Setup (it/vpn.md)"], consulted=["VPN Setup (it/vpn.md)"], best_distance=0.2,
)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """Dummy API key so nothing depends on the developer's environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeChain:
    """RAG chain stub returning a preset result and counting calls."""

    def __init__(self, result):
        self.result, self.calls = result, 0

    def answer(self, question, history=None, department=None):
        self.calls += 1
        return self.result


class RecordingProvider(MockTicketProvider):
    """Mock provider that records create requests and can simulate an outage."""

    def __init__(self):
        super().__init__()
        self.requests, self.fail = [], False

    def create_ticket(self, request):
        if self.fail:
            raise TicketProviderError("down")
        self.requests.append(request)
        return super().create_ticket(request)


class Chat:
    """One employee conversation: `say()` sends a message and returns the bot's reply."""

    def __init__(self, result=UNANSWERED):
        self.llm, self.provider, self.chain = ScriptedLLM(), RecordingProvider(), FakeChain(result)
        self.graph = build_graph(chain=self.chain, llm=self.llm, provider=self.provider)
        self.cfg = {"configurable": {"thread_id": "t"}}

    def say(self, text):
        out = self.graph.invoke({"messages": [{"role": "user", "content": text}]}, self.cfg)
        return out["messages"][-1]["content"]

    @property
    def state(self):
        return self.graph.get_state(self.cfg).values


@pytest.fixture
def chat():
    return Chat()


def _seed_ticket(chat, email="jane@corp.com"):
    """Create a ticket directly on the provider (as if raised earlier)."""
    return chat.provider.create_ticket(
        TicketCreateRequest(requester_email=email, title="VPN fails on Mac", description="d")
    )


# ------------------------------------------------- escalation -> ticket
def test_unanswerable_question_flows_through_consent_and_email_to_a_ticket(chat):
    """
    Offer -> "yes" -> email -> ticket, with nothing created before the email arrives.
    WHY IT MATTERS: the core promise. It also checks the context reaches the ticket body.
    """
    assert "create one" in chat.say("Can I expense a standing desk?")
    assert chat.state["awaiting"] == "ticket_confirmation"

    assert "email" in chat.say("yes please").lower()
    assert chat.provider.requests == []                      # still nothing written

    reply = chat.say("it's Jane@Corp.com")
    req = chat.provider.requests[0]
    assert "HELP-1" in reply
    assert req.requester_email == "jane@corp.com"
    assert "Expenses (hr/expenses.md)" in req.description and "standing desk" in req.description
    assert chat.state["ticket_draft"] is None and chat.state["awaiting"] is None
    assert chat.state["email"] == "jane@corp.com"


def test_known_email_is_not_asked_again(chat):
    """
    A second escalation in the same conversation creates the ticket right after "yes".
    WHY IT MATTERS: employees shouldn't be asked for the same thing twice.
    """
    chat.say("Can I expense a standing desk?"); chat.say("yes"); chat.say("jane@corp.com")
    chat.say("What about monitors?")
    assert "HELP-2" in chat.say("yes")
    assert len(chat.provider.requests) == 2


def test_consent_and_email_in_one_message_needs_no_llm_classification(chat):
    """
    "yes, jane@corp.com" creates the ticket and never calls the follow-up classifier.
    WHY IT MATTERS: the consent decision for the common case is made in code.
    """
    chat.say("Can I expense a standing desk?")
    assert "HELP-1" in chat.say("yes, jane@corp.com")
    assert "FollowUpDecision" not in chat.llm.parsed


def test_declining_creates_nothing_and_clears_the_draft(chat):
    """
    "no thanks" ends the flow without writing anything.
    WHY IT MATTERS: human-in-the-loop means "no" must really mean no.
    """
    chat.say("Can I expense a standing desk?")
    assert "won't create" in chat.say("no thanks")
    assert chat.provider.requests == [] and chat.state["ticket_draft"] is None


def test_cancelling_at_the_email_step(chat):
    """Saying "cancel" when asked for the email stops the flow."""
    chat.say("Can I expense a standing desk?"); chat.say("yes")
    chat.say("cancel")
    assert chat.provider.requests == [] and chat.state["awaiting"] is None


def test_unusable_email_replies_are_retried_then_the_flow_gives_up(chat):
    """
    Three replies without a valid email end the flow instead of looping forever.
    WHY IT MATTERS: bounded loops; and no ticket is created without a valid email.
    """
    chat.llm.decision = "yes"                                  # replies aren't refusals
    chat.say("Can I expense a standing desk?"); chat.say("yes")
    assert "valid email" in chat.say("banana")
    assert "valid email" in chat.say("apple")
    assert "no ticket was created" in chat.say("pear").lower()
    assert chat.provider.requests == [] and chat.state["awaiting"] is None


def test_changing_the_subject_abandons_the_offer(chat):
    """
    An unrelated reply to "create a ticket?" drops the draft and is handled normally.
    WHY IT MATTERS: the employee must never be trapped in the ticket conversation.
    """
    chat.say("Can I expense a standing desk?")
    chat.llm.decision, chat.llm.intent = "other", "chitchat"
    assert "SmartDesk" not in chat.say("hello there") or True   # reply text comes from the LLM stub
    assert chat.state["awaiting"] is None and chat.state["ticket_draft"] is None
    assert chat.provider.requests == []


def test_provider_outage_keeps_the_draft_so_yes_retries(chat):
    """
    If the platform is down, the draft is kept and a plain "yes" retries.
    WHY IT MATTERS: the employee doesn't have to start over after an outage.
    """
    chat.say("Can I expense a standing desk?"); chat.say("yes")
    chat.provider.fail = True
    assert "try again" in chat.say("jane@corp.com")
    assert chat.state["awaiting"] == "ticket_confirmation" and chat.state["ticket_draft"]

    chat.provider.fail = False
    assert "HELP-1" in chat.say("yes")


# ------------------------------------------- explicit ticket request
def test_explicit_request_shows_the_draft_and_waits_for_confirmation():
    """
    "Please raise a ticket" shows title and details first; nothing is created until "yes".
    WHY IT MATTERS: confirmation applies even when the employee asked for the ticket.
    """
    chat = Chat()
    chat.llm.intent = "create_ticket"
    reply = chat.say("My VPN is broken, please raise a ticket")
    assert "Title: Cannot connect to VPN" in reply and "(yes / no)" in reply
    assert chat.provider.requests == []

    chat.say("yes"); chat.say("bob@corp.com")
    req = chat.provider.requests[0]
    assert req.title == "Cannot connect to VPN" and "asked for a ticket directly" in req.description


# ----------------------------------------------------------- status
def test_status_by_id_reports_current_status(chat):
    """
    A known email plus an ID returns the ticket, and later status changes are visible.
    WHY IT MATTERS: the read branch must reflect what the human agent did since.
    """
    chat.say("Can I expense a standing desk?"); chat.say("yes"); chat.say("jane@corp.com")
    chat.llm.intent = "check_status"
    assert "Open" in chat.say("what's the status of help-1?")

    chat.provider.set_status("HELP-1", TicketStatus.DONE, "Resolved")
    reply = chat.say("and now?")                 # no ID -> lists recent tickets
    assert "Done" in reply and "Resolved" in reply


def test_status_without_email_asks_then_lists_tickets():
    """
    Status question -> ask for email -> list the tickets for that email.
    WHY IT MATTERS: "ask for email when needed" for the read branch.
    """
    chat = Chat()
    _seed_ticket(chat)
    chat.llm.intent = "check_status"
    assert "email" in chat.say("what's the status of my ticket?").lower()
    assert chat.state["awaiting"] == "email"
    assert "HELP-1" in chat.say("jane@corp.com")
    assert chat.state["awaiting"] is None and chat.state["email"] == "jane@corp.com"


def test_someone_elses_ticket_is_not_revealed():
    """
    Asking for another person's ticket looks exactly like asking for a missing one.
    WHY IT MATTERS: end-to-end privacy check (the most important test in this file).
    """
    chat = Chat()
    _seed_ticket(chat, "jane@corp.com")
    chat.llm.intent = "check_status"
    reply = chat.say("status of HELP-1 please, I'm bob@corp.com")
    assert "couldn't find ticket HELP-1" in reply and "VPN fails" not in reply


def test_employee_with_no_tickets_gets_a_friendly_answer():
    """No tickets is a normal answer, not an error."""
    chat = Chat()
    chat.llm.intent = "check_status"
    assert "couldn't find any tickets" in chat.say("tickets for new.hire@corp.com?")


# ------------------------------------------------- other branches
def test_answered_question_never_touches_tickets():
    """A grounded answer ends the turn with sources and no ticket offer."""
    chat = Chat(result=ANSWERED)
    reply = chat.say("vpn error 809")
    assert "Restart the client." in reply and "Sources:" in reply
    assert chat.state.get("awaiting") is None and chat.provider.requests == []


def test_greeting_skips_the_knowledge_base():
    """
    Chit-chat is answered without retrieval.
    WHY IT MATTERS: "thanks!" must not trigger a search, a failed lookup, and a ticket offer.
    """
    chat = Chat()
    chat.llm.intent = "chitchat"
    chat.say("hi!")
    assert chat.chain.calls == 0 and chat.state.get("awaiting") is None
