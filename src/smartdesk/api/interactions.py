"""
In-memory log of recent KB turns, used by the UI's "Run evaluation" button.

SCOPE: per process, lost on restart, capped at `maxlen` entries (oldest dropped).
With several workers each has its own log. It stores questions, answers and the
retrieved passages, so it holds employee text in RAM.
"""

import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Interaction:
    """One recorded KB turn. `evaluated` flips once it has been sent to LangSmith."""

    id: str
    question: str            # standalone version
    answer: str
    answered: bool           # False = the agent escalated
    mode: str | None         # search method used
    top_k: int               # maximum number of passages requested
    contexts: list[str]      # passage texts the LLM was shown
    reason: str | None       # escalation reason, if any
    total_seconds: float
    retrieval_seconds: float
    created_at: str
    evaluated: bool = False


class InteractionLog:
    """Thread-safe ring buffer of Interactions."""

    def __init__(self, maxlen: int = 200) -> None:
        self._items: deque[Interaction] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, retrieval: dict, answer: str, answered: bool, reason: str | None = None) -> Interaction:
        """Store a turn from the `retrieval` dict the chain produced. Contexts are the chunks that passed the gate."""
        item = Interaction(
            id=uuid.uuid4().hex,
            question=retrieval.get("question", ""),
            answer=answer,
            answered=answered,
            mode=retrieval.get("mode"),
            top_k=int(retrieval.get("top_k", 3)),
            contexts=[c["text"] for c in retrieval.get("chunks", []) if c.get("passed_gate")],
            reason=reason,
            total_seconds=float(retrieval.get("seconds_total") or 0.0),
            retrieval_seconds=float(retrieval.get("seconds_retrieval") or 0.0),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._items.append(item)
        return item

    def pending(self, limit: int) -> list[Interaction]:
        """The newest `limit` not-yet-evaluated turns, oldest first."""
        with self._lock:
            return [i for i in self._items if not i.evaluated][-limit:]

    def mark_evaluated(self, ids) -> None:
        """Flag turns as evaluated so the next run skips them."""
        wanted = set(ids)
        with self._lock:
            for i in self._items:
                if i.id in wanted:
                    i.evaluated = True

    def counts(self) -> tuple[int, int]:
        """(recorded, pending)."""
        with self._lock:
            return len(self._items), sum(not i.evaluated for i in self._items)