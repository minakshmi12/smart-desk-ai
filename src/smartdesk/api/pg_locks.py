"""
Cross-process conversation locks using PostgreSQL advisory locks.

WHY THIS FILE EXISTS
--------------------
With several workers or hosts, an in-process lock can't stop two instances from
running the same conversation's turn at once. Postgres is already required for
the shared checkpointer, so it also arbitrates the lock.

HOW IT WORKS
------------
A session-level advisory lock is taken on a connection checked out of a dedicated
pool and held for the whole turn.
- Crash-safe: if the process dies, its connections close and Postgres releases the
  lock. No stuck conversations, no expiry timers to tune.
- Needs a connection that maps 1:1 to a Postgres session for the duration. That is
  why it must NOT go through a transaction-mode pooler such as PgBouncer.

Imported lazily (only when CHECKPOINTER=postgres), so the postgres extra is
optional for everyone else.
"""

import hashlib
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from psycopg_pool import ConnectionPool, PoolTimeout

from smartdesk.api.locks import LockTimeout

logger = logging.getLogger(__name__)


def lock_id_for(key: str) -> int:
    """
    Map a conversation id to the signed 64-bit integer Postgres advisory locks use.

    WHY SHA-256 (not Python's hash()): hash() is randomised per process, so two
    instances would compute different numbers for the same conversation and never
    exclude each other. A cryptographic digest is identical everywhere. Two
    different keys colliding in 64 bits is astronomically unlikely, and the only
    effect would be two conversations briefly sharing a lock.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class PostgresAdvisoryLocks:
    """Hands out one cluster-wide lock per conversation id."""

    def __init__(self, pool: ConnectionPool, timeout: float = 30.0, poll_interval: float = 0.05) -> None:
        """
        Args:
            pool: a DEDICATED pool (autocommit, default tuple rows). WHY DEDICATED:
                each in-flight turn keeps one connection checked out for its whole
                duration. If the checkpointer shared this pool, enough concurrent
                turns would hold every connection, and each would then wait for
                one to save its state: a deadlock.
            timeout: seconds to wait (for a connection AND for the lock) before
                raising LockTimeout.
            poll_interval: pause between lock attempts. We poll with
                pg_try_advisory_lock instead of blocking in the database so the
                deadline is always honoured, with no server-side settings needed.
        """
        self._pool = pool
        self._timeout = timeout
        self._poll = poll_interval

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        """
        Wait for the cluster-wide lock on `key`, hold it for the `with` body.

        ORDER OF OPERATIONS AND WHY
        1. Check out a connection (getconn, not `with pool.connection()`). A pool
           timeout is converted to LockTimeout HERE ONLY. Wrapping the whole
           `with` body would misreport an unrelated PoolTimeout from the
           checkpointer's pool as "conversation busy".
        2. Acquire the lock; a failure here still returns the connection.
        3. Run the body. On exit, unlock. If the unlock can't be confirmed, CLOSE
           the connection instead of pooling it: a pooled connection that still
           holds the session lock would block that conversation forever. Closing
           makes Postgres drop the lock.

        Raises:
            LockTimeout: no connection or no lock within the timeout.
        """
        deadline = time.monotonic() + self._timeout
        lock_id = lock_id_for(key)
        try:
            conn = self._pool.getconn(timeout=self._timeout)
        except PoolTimeout as exc:
            raise LockTimeout(key) from exc
        try:
            self._acquire(conn, lock_id, key, deadline)
            try:
                yield
            finally:
                if not self._release(conn, lock_id):
                    conn.close()
        finally:
            self._pool.putconn(conn)

    def _acquire(self, conn, lock_id: int, key: str, deadline: float) -> None:
        """
        Poll pg_try_advisory_lock until it succeeds or the deadline passes.

        A connection checked out of the pool belongs to exactly one caller, so two
        threads in this process can never "share" a session and accidentally pass
        Postgres' re-entrant lock check: they always contend like separate clients.
        """
        while True:
            row = conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,)).fetchone()
            if row[0]:
                return
            if time.monotonic() >= deadline:
                raise LockTimeout(key)
            time.sleep(self._poll)

    @staticmethod
    def _release(conn, lock_id: int) -> bool:
        """
        Unlock; return True only if Postgres confirmed it released our lock.

        Never raises: this runs in a `finally`, and masking the real error (or a
        successful turn) with an unlock failure would be worse than the failure.
        A dead connection already freed the lock server-side, but we still report
        False so the caller closes it.
        """
        try:
            row = conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,)).fetchone()
            return bool(row and row[0])
        except Exception:
            logger.exception("Could not release advisory lock; closing the connection")
            return False