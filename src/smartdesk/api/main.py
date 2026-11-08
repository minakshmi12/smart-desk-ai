"""
Application factory: `uvicorn smartdesk.api.main:create_app --factory`.

No module-level `app`, so importing this module never needs an API key.
"""

import logging
from contextlib import ExitStack, asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smartdesk.api.evaluation import EvaluationService
from smartdesk.api.interactions import InteractionLog
from smartdesk.api.locks import ConversationLocks, ThreadLocks
from smartdesk.api.routes import router
from smartdesk.config import get_settings
from smartdesk.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    agent: Any | None = None, locks: ConversationLocks | None = None,
    retriever: Any | None = None, evaluator: Any | None = None,
) -> FastAPI:
    """
    Build the FastAPI application.

    Args:
        agent / locks / retriever / evaluator: injectable for tests. In production
            leave them None and the real ones are built at startup.

    The retriever is created here (not inside the graph) so the compare endpoint
    and the chat agent share the same BM25 indexes and reranker.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Export .env (LangSmith reads os.environ), configure logging, build storage + agent."""
        load_dotenv()  # must run before LLMClient is created (it reads LANGSMITH_TRACING)
        configure_logging(get_settings().log_level)
        with ExitStack() as stack:
            if app.state.agent is None:
                from smartdesk.agent.graph import build_graph
                from smartdesk.api.persistence import open_persistence
                from smartdesk.llm.client import LLMClient
                from smartdesk.rag.chain import RAGChain
                from smartdesk.rag.hybrid import build_retriever

                persistence = stack.enter_context(open_persistence())
                llm = LLMClient()
                app.state.retriever = build_retriever(llm)
                chain = RAGChain(retriever=app.state.retriever, llm=llm)
                app.state.locks = persistence.locks
                app.state.agent = build_graph(
                    chain=chain, llm=llm, checkpointer=persistence.checkpointer)
            yield

    app = FastAPI(title="SmartDesk AI", version="0.2.0", lifespan=lifespan)
    app.state.agent = agent
    app.state.locks = locks or ThreadLocks()
    app.state.retriever = retriever
    app.state.interactions = InteractionLog()
    app.state.evaluator = evaluator or EvaluationService(app.state.interactions)

    origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()] \
        if agent is None else []
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    app.include_router(router)
    return app