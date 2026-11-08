"""
Central configuration for SmartDesk AI.

WHY THIS FILE EXISTS
--------------------
Every other module needs settings (API keys, thresholds, which ticket
platform to use). If each module reads os.environ on its own, you get
scattered magic strings, no validation, and no single place to see what
the app can be configured with. This file is that single place.

Values come from environment variables or a local `.env` file, so secrets
never live in source code and the same code runs in dev, test, and prod
with different settings.
"""

from functools import lru_cache
from typing import Literal


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, model_validator

class Settings(BaseSettings):
    """
    Typed, validated application settings.

    WHY A CLASS (not plain os.getenv calls):
    - Type coercion: CHUNK_SIZE="800" in .env becomes the int 800.
    - Fail-fast: if OPENAI_API_KEY is missing, the app crashes at startup
      with a clear error, instead of failing mid-conversation.
    - Autocomplete and a documented list of every setting.

    Field names map to env vars case-insensitively
    (llm_model <-> LLM_MODEL).
    """

    # `extra="ignore"` lets .env contain unrelated variables without errors.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- OpenAI ----
    # SecretStr hides the value in logs/reprs, so the key is not leaked by
    # an accidental print(settings). Call .get_secret_value() to read it.
    openai_api_key: SecretStr
    llm_model: str = "gpt-4o"                       # model for chat + structured output
    embedding_model: str = "text-embedding-3-small" # model for turning text into vectors

    # ---- RAG ----
    vector_store_dir: str = "data/vector_store"  # where Chroma persists the index
    chunk_size: int = 800        # characters per chunk: small = precise, large = more context
    chunk_overlap: int = 120     # shared text between neighbouring chunks, so answers
                                 # that straddle a boundary aren't cut in half
    top_k: int = 3               # default maximum chunks returned per query
    confidence_threshold: float = 0.60  # "answer vs escalate" cutoff; tune with evals/
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    candidate_pool: int = Field(default=20, ge=1)
    rerank_candidates: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    reranker: Literal["none", "llm", "cross_encoder"] = "llm"
    rerank_model: str = "gpt-4o-mini"
    rerank_min_score: int = Field(default=1, ge=0, le=10)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    judge_model: str = "gpt-4o"

    # ---- Tickets ----
    # Literal restricts this to known values, so a typo like "jria" fails at startup.
    ticket_provider: Literal["mock", "jira", "asana", "notion"] = "mock"
    jira_base_url: str = "Task"
    jira_email: str = ""
    jira_api_token: SecretStr = SecretStr("")
    jira_project_key: str = "HELP"
    jira_issue_type: str = "Task"

        # API
    log_level: str = "INFO"
    # Comma-separated browser origins allowed to call the API, e.g.
    # "https://intranet.corp.com". Empty = no CORS headers (same-origin or server-to-server only).
    cors_origins: str = ""

    # ---- Persistence ----
    # memory   = lost on restart, single process (dev only)
    # sqlite   = durable, single process
    # postgres = durable, safe with many workers/hosts
    checkpointer: Literal["memory", "sqlite", "postgres"] = "memory"
    sqlite_path: str = "data/checkpoints.sqlite"
    database_url: SecretStr = SecretStr("")
    # Optional separate DSN for the lock pool when the main URL uses transaction pooling.
    lock_database_url: SecretStr = SecretStr("")
    postgres_pool_size: int = Field(default=10, ge=1)
    lock_pool_size: int = Field(default=20, ge=1)
    lock_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _require_database_url_for_postgres(self) -> "Settings":
        """Require a DSN when PostgreSQL persistence is selected."""
        if self.checkpointer == "postgres" and not self.database_url.get_secret_value():
            raise ValueError("DATABASE_URL is required when CHECKPOINTER=postgres")
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Return the application's Settings, creating it only once.

    WHY lru_cache:
    Parsing .env and validating on every call is wasteful, and you want every
    module to see the *same* settings object. The cache makes this a cheap
    singleton.

    WHY A FUNCTION (not a module-level `settings = Settings()`):
    Creating Settings at import time would crash any import (including in
    tests) when env vars are missing. A function defers creation until first
    use, and tests can override env vars and call get_settings.cache_clear().
    """
    return Settings()
