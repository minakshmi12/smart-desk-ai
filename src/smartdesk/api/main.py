"""
Application factory: `uvicorn smartdesk.api.main:create_app --factory`.

WHY A FACTORY WITH NO MODULE-LEVEL `app`
----------------------------------------
Building the app reads settings (which require OPENAI_API_KEY). Doing that at
import time made merely importing this module (for example during test
collection in CI) fail without a key. uvicorn's --factory mode calls create_app()
only when the server actually starts, once per worker process.
"""

import logging
from contextlib import ExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smartdesk.api.locks import ConversationLocks, ThreadLocks
from smartdesk.api.routes import router
from smartdesk.config import get_settings
from smartdesk.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(agent: Any | None = None, locks: ConversationLocks | None = None) -> FastAPI:
    """
    Build the FastAPI application.

    Args:
        agent: a ready-made agent (anything with .invoke(payload, config)). Tests
            pass a fake. In production leave it None: the real graph, its
            checkpointer and the matching lock are built at startup.
        locks: lock backend for tests. In production the lock comes from
            open_persistence(), so it always matches the checkpointer.

    WHY EVERYTHING REAL IS BUILT IN THE LIFESPAN: connection pools must be opened
    once per process and closed on shutdown, and the checkpointer must be the SAME
    object for every request. A misconfiguration (bad DATABASE_URL, missing key,
    incomplete Jira settings) stops the server from starting rather than failing
    on an employee's first message.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Configure logging; build and later close storage + agent (production path)."""
        configure_logging(get_settings().log_level)
        with ExitStack() as stack:
            if app.state.agent is None:
                from smartdesk.agent.graph import build_graph  # lazy: heavy imports
                from smartdesk.api.persistence import open_persistence

                persistence = stack.enter_context(open_persistence())
                app.state.locks = persistence.locks
                app.state.agent = build_graph(checkpointer=persistence.checkpointer)
            yield  # pools/connections close when the server stops

    app = FastAPI(title="SmartDesk AI", version="0.1.0", lifespan=lifespan)
    app.state.agent = agent
    app.state.locks = locks or ThreadLocks()  # replaced by the real one in the lifespan

    # CORS is needed only when a browser on ANOTHER origin calls this API.
    origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()] \
        if agent is None else []
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    app.include_router(router)
    return app