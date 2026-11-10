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
    top_k: optional per-message maximum for retrieved passages; None uses the server setting.
    """

    message: str = Field(description="What the employee typed")
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.:\-]{1,100}$",
        description="Conversation id from a previous response; omit to start a new one",
    )
    retrieval_mode: str | None = Field(
        default=None, pattern=r"^[a-z_]{1,20}$",
        description="Search method for this message (see GET /retrieval/modes); default is the server's",
    )
    top_k: Literal[1, 3, 5, 8] | None = Field(
        default=None, description="Maximum retrieved passages for this message; default is the server setting",
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
    retrieval: dict | None = None   # how the answer was found (mode, scores, timings); None for non-KB turns


class HealthResponse(BaseModel):
    """
    Service status for load balancers and humans.

    status: "ok" when the knowledge base has content, "degraded" when it is empty
        or unreachable (the agent still runs, but would escalate every question).
    """

    status: Literal["ok", "degraded"]
    ticket_provider: str
    checkpointer: str
    kb_chunks: int | None
class ModeInfo(BaseModel):
    """One search method, as shown in the UI."""

    id: str
    label: str
    description: str


class ModesResponse(BaseModel):
    """Every search method the server supports, plus the default and the reranker in use."""

    default: str
    reranker: str
    modes: list[ModeInfo]


class CompareRequest(BaseModel):
    """A question to run through every search method (no answer is generated)."""

    query: str = Field(min_length=1, max_length=500)
    department: str | None = Field(default=None, pattern=r"^(it|hr|general)$")


class ModeResult(BaseModel):
    """What one search method returned (error is set if that method failed)."""

    mode: str
    label: str
    seconds: float
    chunks: list[dict]
    error: str | None = None


class CompareResponse(BaseModel):
    """Results of every method for one question."""

    query: str
    threshold: float
    results: list[ModeResult]


class EvalRunRequest(BaseModel):
    """How many recent chats to evaluate (capped to bound judge cost)."""

    limit: int = Field(default=20, ge=1, le=50)