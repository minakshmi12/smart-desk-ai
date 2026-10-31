"""
Decides which node handles each incoming message.

WHY THIS FILE EXISTS
--------------------
Routing is what makes three capabilities behave like one agent. Priority order:
1. If the agent asked a question last turn (state.awaiting), the message is
   treated as the ANSWER to it. This is deterministic code, because "yes" only
   means something in context, and an LLM must not decide whether consent was given.
2. Otherwise the LLM classifies the intent (kb_query / create_ticket /
   check_status / chitchat).
3. Anything an LLM could get wrong in a harmful way (emails, ticket IDs, obvious
   yes/no) is extracted by regex and code instead.

The router node only DECIDES (writes `route` plus a few state fields); the nodes
it points to do the work.
"""

import logging
import re
from typing import Literal

from smartdesk.agent.prompts import build_followup_messages, build_intent_messages
from smartdesk.agent.state import MAX_EMAIL_ATTEMPTS, AgentState
from smartdesk.llm.schemas import FollowUpDecision, IntentResult
from smartdesk.utils.validation import is_valid_email, normalize_email

logger = logging.getLogger(__name__)

Decision = Literal["yes", "no", "other"]

# Loose candidate pattern; is_valid_email() is the real gatekeeper.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# Jira-style key, e.g. HELP-123 (case-insensitive; upper-cased on extraction).
_TICKET_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]{1,9}-\d{1,7})\b")

_YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "yes please",
        "please do", "go ahead", "do it", "sounds good", "yes go ahead", "sure thing"}
_NO = {"no", "n", "nope", "no thanks", "no thank you", "cancel", "never mind", "nevermind",
       "dont", "not now", "stop", "forget it", "no dont"}

_INTENT_TO_ROUTE = {
    "kb_query": "kb_answer",
    "chitchat": "chitchat",
    "create_ticket": "start_ticket",
    "check_status": "ticket_status",
}

# Cleared whenever a pending question is abandoned (the employee moved on).
_RESET = {"awaiting": None, "ticket_draft": None, "pending_route": None, "email_attempts": 0}


# ----------------------------------------------------------------------
# Deterministic helpers (pure functions: unit-tested without any LLM)
# ----------------------------------------------------------------------
def extract_email(text: str) -> str | None:
    """
    Return the first valid email in `text`, normalised, or None.

    WHY REGEX, NOT THE LLM: a regex can only return something that is literally
    in the message. An LLM can "helpfully" invent or complete an address, and a
    wrong address means the ticket is linked to the wrong person.
    """
    for match in _EMAIL_RE.finditer(text):
        if is_valid_email(match.group()):
            return normalize_email(match.group())
    return None


def extract_ticket_id(text: str) -> str | None:
    """
    Return the first ticket key in `text` (upper-cased), or None.

    Emails are removed first so "help-12@corp.com" isn't mistaken for ticket
    HELP-12. The ID becomes part of a URL later, so it is validated again in
    JiraProvider; this function just finds candidates.
    """
    match = _TICKET_ID_RE.search(_EMAIL_RE.sub(" ", text))
    return match.group(1).upper() if match else None


def _normalize(text: str) -> str:
    """Lower-case, drop punctuation, collapse spaces: "Yes!!" -> "yes"."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def quick_decision(text: str) -> Decision | None:
    """
    Settle obvious replies without an LLM call.

    Returns "yes"/"no" only when the WHOLE (normalised) reply is a known phrase;
    anything longer or mixed ("yes, but also...") returns None so the LLM
    judges it. Exact matching trades a few extra LLM calls for never
    misreading a mixed message as consent.
    """
    normalized = _normalize(text)
    if normalized in _YES:
        return "yes"
    if normalized in _NO:
        return "no"
    return None


def classify_followup(llm, messages: list[dict], expecting: str) -> Decision:
    """
    Ask the LLM whether a reply is yes / no / other. Falls back to "other"
    if the call fails, which abandons the pending flow instead of guessing at consent.
    """
    try:
        return llm.parse(build_followup_messages(messages, expecting), FollowUpDecision).decision
    except Exception:
        logger.exception("Follow-up classification failed; treating reply as unrelated")
        return "other"


def _last_text(state: AgentState) -> str:
    """Content of the newest message (always the employee's)."""
    return state["messages"][-1]["content"]


# ----------------------------------------------------------------------
# The three situations the router can be in
# ----------------------------------------------------------------------
def _route_fresh(state: AgentState, llm) -> dict:
    """
    No pending question: classify the intent and pick a branch.

    - LLM failure falls back to the KB branch. If OpenAI is down the KB branch
      fails too and escalates with reason ERROR, which offers a ticket.
    - An email stated in a ticket-related message is saved; one in a plain KB
      question is NOT (it may be about someone else: "why can't bob@x.com log in?").
    - A status request without a known email goes to ask_email first, remembering
      the ticket ID so it can resume afterwards.
    """
    text = _last_text(state)
    try:
        intent = llm.parse(build_intent_messages(state["messages"]), IntentResult).intent
    except Exception:
        logger.exception("Intent classification failed; defaulting to kb_query")
        intent = "kb_query"

    route = _INTENT_TO_ROUTE[intent]
    if route in ("kb_answer", "chitchat"):
        return {"route": route}

    stated_email = extract_email(text)
    update = {"email": stated_email} if stated_email else {}
    if route == "start_ticket":
        return {"route": route, **update}

    # check_status
    ticket_id = extract_ticket_id(text)
    if not (stated_email or state.get("email")):
        return {"route": "ask_email", "pending_route": "ticket_status",
                "ticket_id": ticket_id, "email_attempts": 0}
    return {"route": "ticket_status", "ticket_id": ticket_id, **update}


def _resolve_confirmation(state: AgentState, llm) -> dict:
    """
    The agent asked "create a ticket?": interpret the answer.

    yes   -> create now if the email is known (or given in the same message,
             e.g. "yes, it's jane@corp.com"), otherwise ask for it.
    no    -> cancel.
    other -> the employee moved on: drop the draft and route the message normally.
    """
    text = _last_text(state)
    # Strip emails first so "yes, jane@corp.com" still counts as a plain "yes".
    decision = quick_decision(_EMAIL_RE.sub(" ", text)) or classify_followup(
        llm, state["messages"], "confirmation"
    )
    if decision == "yes":
        email = extract_email(text) or state.get("email")
        if email:
            return {"route": "create_ticket", "email": email}
        return {"route": "ask_email", "pending_route": "create_ticket", "email_attempts": 0}
    if decision == "no":
        return {"route": "cancel_ticket"}
    return {**_RESET, **_route_fresh(state, llm)}


def _resolve_email(state: AgentState, llm) -> dict:
    """
    The agent asked for an email: interpret the answer.

    A valid email resumes whatever was pending (create or status). Otherwise:
    cancel / unrelated message are honoured, and anything else re-asks until
    MAX_EMAIL_ATTEMPTS is reached, so the conversation can't loop forever.
    """
    email = extract_email(_last_text(state))
    if email:
        return {"route": state.get("pending_route") or "create_ticket",
                "email": email, "email_attempts": 0}

    decision = quick_decision(_last_text(state)) or classify_followup(llm, state["messages"], "email")
    if decision == "no":
        return {"route": "cancel_ticket"}
    if decision == "other":
        return {**_RESET, **_route_fresh(state, llm)}

    attempts = state.get("email_attempts", 0) + 1
    if attempts >= MAX_EMAIL_ATTEMPTS:
        return {"route": "cancel_ticket", "email_attempts": attempts}
    return {"route": "ask_email", "email_attempts": attempts}


def make_router_node(llm):
    """
    Build the router node with the LLM injected (tests pass a scripted fake).

    The returned function checks `awaiting` FIRST, then falls back to intent
    classification. It returns only state updates; it never writes messages.
    """

    def router(state: AgentState) -> dict:
        """Resolve the pending question if there is one, else classify the intent."""
        awaiting = state.get("awaiting")
        if awaiting == "ticket_confirmation":
            return _resolve_confirmation(state, llm)
        if awaiting == "email":
            return _resolve_email(state, llm)
        return _route_fresh(state, llm)

    return router
