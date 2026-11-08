"""
Builds the conversation store and the matching lock from configuration.

WHY THIS FILE EXISTS
--------------------
The checkpointer (where state is saved) and the lock (who may run a turn) must
agree on scope. A shared database with a per-process lock is the classic
scale-out bug. Producing both here, from one setting, makes that pairing
impossible to get wrong. It mirrors tickets/factory.py.

WHY A CONTEXT MANAGER: Postgres pools and SQLite connections must be closed
cleanly on shutdown, and Postgres setup must happen once at startup. The API
lifespan enters this once and leaves it when the server stops.
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartdesk.api.locks import ConversationLocks, ThreadLocks
from smartdesk.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Key for the startup lock that serialises schema setup across instances.
SETUP_LOCK_KEY = "smartdesk:checkpoint-setup"


@dataclass(frozen=True)
class Persistence:
    """
    The pair the API needs, plus a label for logs and /health.

    checkpointer: pass to build_graph(checkpointer=...).
    locks: pass to the app; routes hold it around every turn.
    kind: "memory" | "sqlite" | "postgres".
    """

    checkpointer: Any
    locks: ConversationLocks
    kind: str


def _missing(extra: str) -> RuntimeError:
    """Build the error shown when an optional dependency group isn't installed."""
    return RuntimeError(
        f"CHECKPOINTER={extra} needs extra packages: pip install 'smartdesk-ai[{extra}]'"
    )


@contextmanager
def _open_memory(s: Settings) -> Iterator[Persistence]:
    """
    In-process storage for development and tests.

    Logs a warning because forgetting to switch this off is the most likely
    production mistake: conversations vanish on restart and are invisible to
    other workers.
    """
    from langgraph.checkpoint.memory import MemorySaver

    logger.warning(
        "CHECKPOINTER=memory: conversations are lost on restart and not shared "
        "between workers. Use sqlite (single process) or postgres (scale-out) in production."
    )
    yield Persistence(MemorySaver(), ThreadLocks(s.lock_timeout_seconds), "memory")


@contextmanager
def _open_sqlite(s: Settings) -> Iterator[Persistence]:
    """
    Durable single-process storage in a local file.

    check_same_thread=False: FastAPI runs endpoints on worker threads, and the
    saver serialises access to the connection with its own lock.
    WAL mode lets you inspect the file with the sqlite3 CLI while the server runs.
    The lock stays in-process, so this is for ONE server process only.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise _missing("sqlite") from exc

    path = Path(s.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn)
        saver.setup()
        yield Persistence(saver, ThreadLocks(s.lock_timeout_seconds), "sqlite")
    finally:
        conn.close()


@contextmanager
def _open_postgres(s: Settings) -> Iterator[Persistence]:
    """
    Shared storage + cluster-wide locks for running many workers/hosts.

    TWO POOLS (see PostgresAdvisoryLocks for why): one for conversation state,
    one for locks. Total connections per instance = POSTGRES_POOL_SIZE +
    LOCK_POOL_SIZE; multiply by instance count and keep it under the server's
    max_connections.

    pool.wait() makes a wrong DSN or unreachable database fail startup within
    10 seconds instead of on the first chat message.

    SCHEMA SETUP UNDER A LOCK: saver.setup() creates/migrates tables. Several
    instances starting at once (a rolling deploy) could race on that, so it runs
    under the same advisory-lock mechanism, one instance at a time.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        from smartdesk.api.pg_locks import PostgresAdvisoryLocks
    except ImportError as exc:
        raise _missing("postgres") from exc

    dsn = s.database_url.get_secret_value()
    lock_dsn = s.lock_database_url.get_secret_value() or dsn

    with ExitStack() as stack:
        # The saver requires autocommit connections that return dict rows.
        # prepare_threshold=0 avoids prepared-statement clashes behind poolers.
        store_pool = stack.enter_context(ConnectionPool(
            dsn, min_size=1, max_size=s.postgres_pool_size, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        ))
        # The lock pool uses default tuple rows: _release reads row[0].
        lock_pool = stack.enter_context(ConnectionPool(
            lock_dsn, min_size=1, max_size=s.lock_pool_size, open=False,
            kwargs={"autocommit": True},
        ))
        store_pool.wait(timeout=10)
        lock_pool.wait(timeout=10)

        locks = PostgresAdvisoryLocks(lock_pool, timeout=s.lock_timeout_seconds)
        saver = PostgresSaver(store_pool)
        with locks.hold(SETUP_LOCK_KEY):
            saver.setup()
        yield Persistence(saver, locks, "postgres")


_OPENERS = {"memory": _open_memory, "sqlite": _open_sqlite, "postgres": _open_postgres}


@contextmanager
def open_persistence(settings: Settings | None = None) -> Iterator[Persistence]:
    """
    Open the storage selected by CHECKPOINTER; close it when the `with` exits.

    Args:
        settings: injectable for tests; defaults to get_settings().

    Usage:
        with open_persistence() as p:
            agent = build_graph(checkpointer=p.checkpointer)
            ...serve requests using p.locks...
    """
    s = settings or get_settings()
    with _OPENERS[s.checkpointer](s) as persistence:
        logger.info("Conversation storage ready: %s", persistence.kind)
        yield persistence