"""
Per-conversation locks (in-process implementation + the shared interface).

WHY THIS FILE EXISTS
--------------------
The agent is stateful: a turn reads the saved state, runs, then saves it. If two
requests for the SAME conversation overlap (double-click, client retry, two tabs),
both read the same "awaiting confirmation" state and both act on it, which could
create two tickets. Serialising turns per conversation removes that race.

TWO IMPLEMENTATIONS, ONE INTERFACE
----------------------------------
- ThreadLocks (this file): correct within ONE process. Used for memory/sqlite.
- PostgresAdvisoryLocks (pg_locks.py): correct across processes and hosts.
Routes depend only on ConversationLocks, so switching is a config change.
"""

import threading
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol


class LockTimeout(Exception):
    """
    Could not get the conversation's lock in time.

    WHY A CUSTOM EXCEPTION: it means "another turn for this conversation is still
    running (or stuck)". The API maps it to HTTP 409 so the client can retry,
    instead of treating it as a server crash (500). Carries only the thread id,
    never message content.
    """


class ConversationLocks(Protocol):
    """
    What the routes need from any lock backend.

    WHY A PROTOCOL: structural typing means tests can pass any object with a
    `hold` method, with no inheritance required.
    """

    def hold(self, key: str) -> AbstractContextManager[None]:
        """
        Context manager: wait for `key`'s lock, hold it for the `with` body.

        Raises:
            LockTimeout: the lock could not be obtained within the timeout.
        """
        ...


class _Entry:
    """
    Holder for one lock.

    WHY NOT STORE threading.Lock DIRECTLY: lock objects can't be weakly
    referenced, but instances of a normal class can. Weak references let unused
    locks disappear automatically.
    """

    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        self.lock = threading.Lock()


class ThreadLocks:
    """
    One lock per conversation id, valid within a single process.

    WHY WEAK VALUES: a long-running server sees many conversations. A plain dict
    of locks would grow forever; here an entry is dropped as soon as nobody holds
    or waits on it.

    SCOPE: does NOT protect against a second worker process or host. Use
    PostgresAdvisoryLocks for that.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """
        Args:
            timeout: seconds to wait for a busy conversation before raising
                LockTimeout. WHY BOUNDED: an unbounded wait behind a stuck turn
                would pile up blocked threads until the server stops responding.
        """
        self._timeout = timeout
        self._guard = threading.Lock()  # protects the registry itself
        self._entries: "weakref.WeakValueDictionary[str, _Entry]" = weakref.WeakValueDictionary()

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        """
        Wait up to `timeout` seconds for this conversation's lock, hold it for the body.

        The entry is looked up (or created) under the registry guard so two
        threads can't create two different locks for the same key. The local
        variable `entry` keeps the entry alive for the whole `with` block.

        Raises:
            LockTimeout: the lock stayed busy for the whole timeout.
        """
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
        if not entry.lock.acquire(timeout=self._timeout):
            raise LockTimeout(key)
        try:
            yield
        finally:
            entry.lock.release()