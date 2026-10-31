"""
Structured-output schemas used with LLMClient.parse().

WHY THIS FILE EXISTS
--------------------
Each class here is a "contract" for one decision the LLM makes. Keeping them
together, separate from the code that uses them, makes the LLM's outputs
explicit and testable. The Field descriptions are sent to the model as
instructions, so they matter as much as the prompts do.
"""

from typing import Literal

from pydantic import BaseModel, Field



class IntentResult(BaseModel):
    """Router decision for a FRESH message (no pending question). See router.py."""

    intent: Literal["kb_query", "create_ticket", "check_status", "chitchat"]
    reasoning: str = Field(description="One short sentence explaining the choice")


class FollowUpDecision(BaseModel):
    """
    How the employee answered a question the agent just asked ("create a ticket?"
    or "what's your email?").

    WHY THREE VALUES: "other" is what lets the employee change their mind or ask
    something new mid-flow without getting stuck in the ticket conversation.
    """

    decision: Literal["yes", "no", "other"]
    reasoning: str = Field(description="One short sentence explaining the choice")


class TicketSummary(BaseModel):
    """
    A support-ticket title and problem statement distilled from the chat.
    Used only when the employee asks for a ticket directly (when the bot escalates
    on its own, the draft already exists). Length limits are enforced in code
    because OpenAI's strict structured outputs don't support them.
    """

    title: str = Field(description="One line, under 100 characters, no line breaks")
    problem: str = Field(
        description="2-4 plain sentences a support agent can act on. Only facts "
        "stated in the conversation; never invent error codes, names or steps."
    )

    # Not used for control flow. It is for debugging and evals: when a message
    # is misrouted, this tells you why. Asking the model to explain also tends
    # to improve classification accuracy slightly.
    reasoning: str = Field(description="One short sentence explaining the choice")


class EmailExtraction(BaseModel):
    """
    Result of pulling an email address out of a free-text message.

    WHY IT EXISTS:
    Before creating a ticket we must collect the employee's email, and people
    answer in many ways: "sure, it's jane@corp.com", "jane@corp.com thanks!",
    "use my work email". The LLM is good at finding the address in messy text.

    IMPORTANT: this only *extracts*. Whether the address is valid is decided
    by code in utils/validation.py, not by the model. LLMs can wrongly
    approve malformed addresses, and a bad email means the ticket can't be
    linked to the right person.
    """

    # Optional on purpose: if no email is present the model must return null
    # rather than guess or fabricate one. A None result tells the ticket node
    # to ask the user again.
    email: str | None = Field(
        default=None, description="Email address in the message, or null if none"
    )


class GroundedAnswer(BaseModel):
    """
    The generator's structured reply: an answer plus a self-assessment.

    WHY ONE CALL INSTEAD OF "ANSWER THEN GRADE": it halves latency and cost on
    the happy path. The trade-off is that a model grading its own answer is
    slightly less independent than a separate grader, which is why
    confidence.verify_generation adds code-level checks on top (citations
    must exist). If evals show too many wrong answers, add a separate grader.
    """

    answerable: bool = Field(
        description="True only if the numbered passages contain enough information "
        "to fully answer the question. False otherwise."
    )
    answer: str = Field(
        description="The answer, written only from the passages; or exactly "
        "\"I don't know\" when answerable is false."
    )
    used_sources: list[int] = Field(
        description="Numbers of the passages the answer relies on, e.g. [1, 3]. "
        "Empty when answerable is false."
    )
