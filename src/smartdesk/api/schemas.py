"""
Request and response models for the HTTP API.

WHY THIS FILE EXISTS
--------------------
These are the public contract with whatever frontend (web widget, Slack bridge,
mobile app) calls the agent. They are separate from the agent's internal state
on purpose: the internal state can change freely without breaking clients, and
internal fields (the employee's email, the ticket draft, routing flags) are
never exposed by accident.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


MAX_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    """
    One employee message.

    thread_id: identifies the conversation. Omit it on the first message; the
        response contains the one the server created, and the client sends it back
        on every later message. WHY THE PATTERN: the id is used as a lock key and
        a storage key, so only a conservative character set is accepted.
    """

    message: str = Field(description="What the employee typed")
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.:\-]{1,100}$",
        description="Conversation id from a previous response; omit to start a new one",
    )

    @field_validator("message")
    @classmethod
    def _check_message(cls, v: str) -> str:
        """
        Trim the message and enforce 1..4000 characters.

        WHY A CAP: every message is sent to an LLM (and copied into tickets), so an
        unbounded body is a cost and abuse risk. Blank messages are rejected
        because they would waste an LLM call and confuse the yes/no logic.
        """
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
        return v


class ChatResponse(BaseModel):
    """
    The agent's reply to one message.

    awaiting: what the agent's reply is asking for, if anything:
        "ticket_confirmation" -> a yes/no question (a UI can show Yes / No buttons)
        "email"               -> the employee should type their email
        None                  -> no pending question
    This is purely a UI hint. The agent decides from its own stored state, not
    from anything the client sends back.
    """

    thread_id: str
    reply: str
    awaiting: Literal["ticket_confirmation", "email"] | None = None


class HealthResponse(BaseModel):
    """
    Service status for load balancers and humans.

    status: "ok" when the knowledge base has content, "degraded" when it is empty
        or unreachable (the agent still runs, but would escalate every question).
    """

    status: Literal["ok", "degraded"]
    ticket_provider: str
    kb_chunks: int | None
