"""
HTTP endpoints: POST /chat and GET /health.

WHY THIS FILE EXISTS
--------------------
The routes are deliberately thin: validate input, pick the conversation, run one
agent turn, shape the output. All decision-making stays in the agent so it can
be reused by the CLI, tests, or another front end.
"""

from asyncio import locks
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from smartdesk.api.locks import ThreadLocks
from smartdesk.api.schemas import ChatRequest, ChatResponse, HealthResponse
from smartdesk.config import get_settings
from smartdesk.rag.vector_store import get_collection
from fastapi import APIRouter, Depends, HTTPException, Request

from smartdesk.api.locks import ConversationLocks, LockTimeout

BUSY_TEXT = "Your previous message is still being processed. Please wait a moment and try again."

logger = logging.getLogger(__name__)
router = APIRouter()

SERVER_ERROR_TEXT = "Sorry, something went wrong on our side. Please try again in a moment."


def get_agent(request: Request) -> Any:
    """
    Dependency: the compiled agent graph.

    WHY A DEPENDENCY: tests replace it with a fake via create_app(agent=...).
    Raises 503 if the agent isn't built, so a half-started server fails clearly.
    """
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    return agent


def get_locks(request: Request) -> ConversationLocks:
    """Dependency: the lock backend chosen at startup (in-process or Postgres)."""
    return request.app.state.locks




@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    agent: Annotated[Any, Depends(get_agent)],
    locks: Annotated[ThreadLocks, Depends(get_locks)],
) -> ChatResponse:
    """
    Run one conversation turn.

    STEPS
    1. Use the client's thread_id, or create a random one for a new conversation.
    2. Take that conversation's lock, so overlapping requests run one at a time.
    3. Send ONLY the new message: the graph's checkpointer restores earlier
       messages, the pending question, the draft, and the known email by thread_id.
    4. Return the agent's last message and the pending-question hint.

    WHY `def`, NOT `async def`: the agent makes blocking calls (OpenAI, Chroma,
    Jira). FastAPI runs `def` endpoints in a worker thread, so a slow turn doesn't
    freeze other users.

    ERRORS: the nodes already turn most failures (LLM down, Jira down) into
    friendly replies. Anything that still escapes (e.g. the checkpointer failing)
    is logged with its traceback and returned as a generic 500, so internal
    details are never sent to the client.
    """
    thread_id = body.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [{"role": "user", "content": body.message}]}

    try:
        with locks.hold(thread_id):
            out = agent.invoke(payload, config)
        reply = out["messages"][-1]["content"]
    except LockTimeout:
        logger.warning("Conversation busy (thread_id=%s)", thread_id)
        raise HTTPException(status_code=409, detail=BUSY_TEXT, headers={"Retry-After": "2"})
    except Exception:
        logger.exception("Agent turn failed (thread_id=%s)", thread_id)
        raise HTTPException(status_code=500, detail=SERVER_ERROR_TEXT)

    return ChatResponse(thread_id=thread_id, reply=reply, awaiting=out.get("awaiting"))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Report that the server is up and whether the knowledge base has content.

    Always returns HTTP 200 while the process can answer. A missing or empty index
    is reported as "degraded" rather than failing the check, because the agent
    still works in that state (it escalates every question) and restarting the
    server would not fix it: re-running ingestion would.

    NOTE: get_collection() creates an empty collection if none exists, so the
    first call on a fresh install has that small side effect.
    """
    try:
        chunks: int | None = get_collection().count()
    except Exception:
        logger.exception("Health check could not read the vector store")
        chunks = None
    return HealthResponse(
        status="ok" if chunks else "degraded",
        ticket_provider=get_settings().ticket_provider,
        checkpointer=get_settings().checkpointer,
        kb_chunks=chunks,
    )
@router.get("/architecture")
def architecture(agent: Annotated[Any, Depends(get_agent)]) -> dict:
    """Mermaid source of the compiled LangGraph, generated from the real graph."""
    return {"mermaid": agent.get_graph().draw_mermaid()}