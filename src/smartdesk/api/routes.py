"""
HTTP endpoints.

POST /chat, GET /health, GET /architecture          chat and status
GET /retrieval/modes, POST /retrieval/compare       search methods (UI selector and comparison)
GET /evaluation/status, POST /evaluation/run        LangSmith evaluation of recent chats
"""

import logging
import math
import time
import uuid
from html import escape
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from smartdesk.api.evaluation import (
    EvaluationBusy, EvaluationCooldown, EvaluationService, EvaluationUnavailable)
from smartdesk.api.locks import ConversationLocks, LockTimeout
from smartdesk.api.schemas import (
    ChatRequest, ChatResponse, CompareRequest, CompareResponse, EvalRunRequest,
    HealthResponse, ModeInfo, ModeResult, ModesResponse)
from smartdesk.config import get_settings
from smartdesk.rag.display import summarize_chunks
from smartdesk.rag.hybrid import MODE_INFO
from smartdesk.rag.vector_store import get_collection

logger = logging.getLogger(__name__)
router = APIRouter()

SERVER_ERROR_TEXT = "Sorry, something went wrong on our side. Please try again in a moment."
BUSY_TEXT = "Your previous message is still being processed. Please wait a moment and try again."


def get_agent(request: Request) -> Any:
    """Dependency: the compiled agent graph (503 if not built)."""
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    return agent


def get_locks(request: Request) -> ConversationLocks:
    """Dependency: the lock backend chosen at startup."""
    return request.app.state.locks


def get_retriever(request: Request) -> Any:
    """Dependency: the multi-method retriever (503 if not built)."""
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not ready")
    return retriever


def get_evaluator(request: Request) -> EvaluationService:
    """Dependency: the live-evaluation service."""
    return request.app.state.evaluator


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    agent: Annotated[Any, Depends(get_agent)],
    locks: Annotated[ConversationLocks, Depends(get_locks)],
) -> ChatResponse:
    """
    Run one conversation turn.

    The search method (`retrieval_mode`) and passage count (`top_k`) are passed to the
    graph as per-request config, so they apply to this message only. Retrieval details are returned only when THIS
    turn went through the KB branch (state keeps the last value, so `route` is checked).
    KB turns are also recorded for the Evaluation page; a recording failure never
    fails the turn.
    """
    mode = body.retrieval_mode
    if mode and mode not in MODE_INFO:
        raise HTTPException(status_code=422, detail=f"Unknown retrieval_mode: {mode}")

    thread_id = body.thread_id or uuid.uuid4().hex
    configurable = {"thread_id": thread_id}
    if mode:
        configurable["retrieval_mode"] = mode
    if body.top_k is not None:
        configurable["top_k"] = body.top_k
    payload = {"messages": [{"role": "user", "content": body.message}]}

    try:
        with locks.hold(thread_id):
            out = agent.invoke(payload, {"configurable": configurable})
        reply = out["messages"][-1]["content"]
    except LockTimeout:
        logger.warning("Conversation busy (thread_id=%s)", thread_id)
        raise HTTPException(status_code=409, detail=BUSY_TEXT, headers={"Retry-After": "2"})
    except Exception:
        logger.exception("Agent turn failed (thread_id=%s)", thread_id)
        raise HTTPException(status_code=500, detail=SERVER_ERROR_TEXT)

    retrieval = out.get("retrieval") if out.get("route") == "kb_answer" else None
    if retrieval:
        try:
            request.app.state.interactions.record(
                retrieval, out.get("answer", ""), answered=not out.get("escalate"),
                reason=out.get("escalation_reason"))
        except Exception:
            logger.exception("Could not record the interaction")
    return ChatResponse(thread_id=thread_id, reply=reply, awaiting=out.get("awaiting"),
                        retrieval=retrieval)


@router.get("/retrieval/modes", response_model=ModesResponse)
def retrieval_modes(request: Request) -> ModesResponse:
    """Every search method the code supports (the UI builds its selector from this)."""
    s = get_settings()
    default = getattr(request.app.state.retriever, "default_mode", None) or s.retrieval_mode
    return ModesResponse(
        default=default, reranker=s.reranker,
        modes=[ModeInfo(id=k, **v) for k, v in MODE_INFO.items()],
    )


@router.post("/retrieval/compare", response_model=CompareResponse)
def retrieval_compare(
    body: CompareRequest, retriever: Annotated[Any, Depends(get_retriever)]
) -> CompareResponse:
    """
    Run one question through EVERY search method and return each ranking.

    No answer is generated, but each method embeds the question and the keyword and
    hybrid methods also call the reranker, so this costs a few small API calls.
    A failing method is reported in its own result instead of failing the request.
    """
    threshold = get_settings().confidence_threshold
    results = []
    for mode, info in MODE_INFO.items():
        start, error, chunks = time.perf_counter(), None, []
        try:
            chunks = retriever.search(body.query, body.department, mode=mode)
        except Exception:
            logger.exception("Compare: %s search failed", mode)
            error = "This search method failed. See the API logs."
        results.append(ModeResult(
            mode=mode, label=info["label"], seconds=round(time.perf_counter() - start, 3),
            chunks=summarize_chunks(chunks, threshold), error=error))
    return CompareResponse(query=body.query, threshold=threshold, results=results)


@router.get("/sources", response_class=HTMLResponse)
def view_source(source: str, page: int | None = None) -> HTMLResponse:
    """Display indexed text for a source document, optionally limited to a PDF page."""
    collection = get_collection()
    result = collection.get(where={"source": source}, include=["documents", "metadatas"])
    chunks = list(zip(result["documents"], result["metadatas"]))
    if page is not None:
        chunks = [(text, meta) for text, meta in chunks if meta.get("page") == page]
    if not chunks:
        raise HTTPException(status_code=404, detail="Source document not found in the knowledge base")

    chunks.sort(key=lambda item: (item[1].get("page") or 0, item[1].get("chunk_index") or 0))
    title = escape(chunks[0][1].get("title", "Knowledge-base source"))
    source_label = escape(source)
    sections = []
    previous_page = object()
    for text, metadata in chunks:
        chunk_page = metadata.get("page")
        heading = ""
        if chunk_page is not None and chunk_page != previous_page:
            heading = f"<h2>Page {int(chunk_page)}</h2>"
        previous_page = chunk_page
        sections.append(f"{heading}<pre>{escape(text)}</pre>")
    content = "\n".join(sections)
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;"
        "color:#17212b}h1{font-size:1.6rem}h2{font-size:1.1rem;border-bottom:1px solid #ccd5dc;padding-bottom:6px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:16px;border-radius:6px}"
        ".source{color:#52616b;font-size:.9rem}</style></head><body>"
        f"<h1>{title}</h1><p class='source'>{source_label}</p>{content}</body></html>"
    )


@router.get("/evaluation/status")
def evaluation_status(evaluator: Annotated[EvaluationService, Depends(get_evaluator)]) -> dict:
    """Is LangSmith configured, how many chats are waiting, and the dashboard links."""
    return evaluator.status()


@router.post("/evaluation/run")
def evaluation_run(
    body: EvalRunRequest, evaluator: Annotated[EvaluationService, Depends(get_evaluator)]
) -> dict:
    """Evaluate recent chats in LangSmith; returns scores and the experiment URL."""
    try:
        return evaluator.run(body.limit)
    except EvaluationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except EvaluationBusy:
        raise HTTPException(status_code=409, detail="An evaluation is already running.")
    except EvaluationCooldown as exc:
        raise HTTPException(
            status_code=429, detail=str(exc),
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))})
    except Exception:
        logger.exception("Live evaluation failed")
        raise HTTPException(status_code=502, detail="The evaluation could not be completed. See the API logs.")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Server up; "degraded" when the knowledge base is empty or unreadable."""
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