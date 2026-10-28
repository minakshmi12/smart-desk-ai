"""FastAPI application entry point."""

from fastapi import FastAPI

from smartdesk.api.routes import router

app = FastAPI(title="SmartDesk AI")
app.include_router(router)
