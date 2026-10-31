"""
Tests for PostgresAdvisoryLocks against a fake Postgres (no database needed).

WHY THIS FILE EXISTS
--------------------
The lock logic (polling, timeouts, connection hygiene) is where the subtle bugs
are, and it can be checked without a server by faking Postgres' one relevant
behaviour: a lock id can be held by one session at a time, and closing a session
frees its locks. Real-database behaviour is covered by the gated tests in
tests/integration/test_postgres.py.
"""

import threading
import time

import pytest

pytest.importorskip("psycopg_pool")
from psycopg_pool import PoolTimeout  # noqa: E402

from smartdesk.api.locks import LockTimeout  # noqa: E402
from smartdesk.api.pg_locks import PostgresAdvisoryLocks, lock_id_for  # noqa: E402


class FakeServer:
    """The shared 'database': which session currently holds each advisory lock id."""

    def __init__(self):
        self.held: dict[int, "FakeConn"] = {}
        self.mutex = threading.Lock()


class _Cursor:
    """Minimal cursor: fetchone() returns the canned row."""

    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    """
    One fake Postgres session.

    execute() understands only the two advisory-lock statements. close() frees
    every lock the session holds, as a real disconnect does. `fail_unlock` makes
    the unlock statement raise, simulating a broken connection.
    """

    def __init__(self, server, fail_unlock=False):
        self.server, self.fail_unlock, self.closed = server, fail_unlock, False

    def execute(self, sql, params):
        (lock_id,) = params
        with self.server.mutex:
            if "pg_try_advisory_lock" in sql:
                owner = self.server.held.get(lock_id)
                if owner is None or owner is self:
                    self.server.held[lock_id] = self
                    return _Cursor((True,))
                return _Cursor((False,))
            if self.fail_unlock:
                raise RuntimeError("connection broke")
            ok = self.server.held.get(lock_id) is self
            self.server.held.pop(lock_id, None) if ok else None
            return _Cursor((ok,))

    def close(self):
        self.closed = True
        with self.server.mutex:
            for k in [k for k, v in self.server.held.items() if v is self]:
                del self.server.held[k]


class FakePool:
    """Hands out fake sessions on one server and records what comes back."""

    def __init__(self, server, fail_unlock=False, exhausted=False):
        self.server, self.fail_unlock, self.exhausted = server, fail_unlock, exhausted
        self.returned: list[FakeConn] = []

    def getconn(self, timeout=None):
        if self.exhausted:
            raise PoolTimeout("pool exhausted")
        return FakeConn(self.server, self.fail_unlock)

    def putconn(self, conn):
        self.returned.append(conn)


def _instance(server, **pool_kw):
    """Helper: a lock object with its own pool, i.e. one 'server instance'."""
    pool = FakePool(server, **pool_kw)
    return PostgresAdvisoryLocks(pool, timeout=0.2, poll_interval=0.005), pool


def test_lock_id_is_stable_signed_64_bit_and_distinct():
    """
    The same key always maps to the same in-range number; different keys differ.
    WHY IT MATTERS: instances must agree on the number or they never exclude each other.
    """
    a = lock_id_for("thread-1")
    assert a == lock_id_for("thread-1") and a != lock_id_for("thread-2")
    assert -(2**63) <= a < 2**63


def test_two_instances_never_run_the_same_conversation_at_once():
    """
    Locks objects with separate pools (separate processes) still exclude each other.
    WHY IT MATTERS: this is the whole point of the shared lock. It is what stops
    two instances from both acting on one "yes" and creating two tickets.
    """
    server = FakeServer()
    a, _ = _instance(server)
    b, _ = _instance(server)
    a.__dict__["_timeout"] = b.__dict__["_timeout"] = 5.0
    state, guard = {"active": 0, "peak": 0}, threading.Lock()

    def work(locks):
        with locks.hold("t1"):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1

    threads = [threading.Thread(target=work, args=(x,)) for x in (a, b, a, b)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 1


def test_different_conversations_do_not_block_each_other():
    """
    Holding one conversation doesn't delay another.
    WHY IT MATTERS: one slow turn must not stall unrelated employees.
    """
    server = FakeServer()
    a, _ = _instance(server)
    b, _ = _instance(server)
    with a.hold("one"):
        with b.hold("two"):
            assert len(server.held) == 2


def test_busy_conversation_times_out_and_returns_its_connection():
    """
    A waiter gives up with LockTimeout and does not leak its pooled connection.
    WHY IT MATTERS: leaked connections would shrink the pool until the app froze.
    """
    server = FakeServer()
    a, _ = _instance(server)
    b, pool_b = _instance(server)
    with a.hold("t1"):
        with pytest.raises(LockTimeout):
            with b.hold("t1"):
                pass
    assert len(pool_b.returned) == 1 and not pool_b.returned[0].closed


def test_lock_is_released_when_the_turn_raises():
    """
    An exception inside the body still unlocks, and the connection is reused.
    WHY IT MATTERS: one failed turn must not lock the conversation out.
    """
    server = FakeServer()
    a, pool = _instance(server)
    with pytest.raises(ValueError):
        with a.hold("t1"):
            raise ValueError("boom")
    assert server.held == {}
    assert len(pool.returned) == 1 and not pool.returned[0].closed


def test_connection_is_closed_not_pooled_when_unlock_fails():
    """
    If the unlock can't be confirmed, the session is closed so Postgres frees the lock.
    WHY IT MATTERS: a pooled connection still holding the lock would block that
    conversation until the process restarted.
    """
    server = FakeServer()
    a, pool = _instance(server, fail_unlock=True)
    with a.hold("t1"):
        assert len(server.held) == 1
    assert pool.returned[0].closed and server.held == {}


def test_exhausted_pool_becomes_lock_timeout():
    """No free connection within the timeout is reported as "conversation busy"."""
    a, _ = _instance(FakeServer(), exhausted=True)
    with pytest.raises(LockTimeout):
        with a.hold("t1"):
            pass


def test_a_pool_timeout_inside_the_turn_is_not_mistaken_for_a_busy_lock():
    """
    A PoolTimeout raised by the BODY (e.g. the checkpointer's pool) propagates unchanged.
    WHY IT MATTERS: reporting a database outage as "your message is still being
    processed" would send clients into pointless retries and hide the real fault.
    """
    a, _ = _instance(FakeServer())
    with pytest.raises(PoolTimeout):
        with a.hold("t1"):
            raise PoolTimeout("checkpointer pool exhausted")