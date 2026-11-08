"""
Tests for the in-process lock's timeout behaviour.

WHY THIS FILE EXISTS
--------------------
Mutual exclusion is covered in test_api.py. These tests cover the new guarantee:
a busy conversation fails fast with LockTimeout instead of blocking forever.
"""

import threading

from smartdesk.api.locks import LockTimeout, ThreadLocks


def test_waiting_longer_than_the_timeout_raises_lock_timeout():
    """
    A second caller for a held conversation gets LockTimeout, and the lock is
    still usable afterwards.

    WHY IT MATTERS: without a bound, requests queued behind a stuck turn would
    pile up threads until the whole server stopped responding.
    """
    locks, outcome = ThreadLocks(timeout=0.05), []

    def contender():
        try:
            with locks.hold("a"):
                outcome.append("entered")
        except LockTimeout:
            outcome.append("timeout")

    with locks.hold("a"):
        t = threading.Thread(target=contender)
        t.start()
        t.join()

    assert outcome == ["timeout"]
    with locks.hold("a"):  # released properly, not left locked by the failed attempt
        pass