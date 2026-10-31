"""
Runs the real chain and router over the eval cases, timing each one.

WHY THIS FILE EXISTS
--------------------
Separates "collect raw outcomes" from "score them", so scoring can be re-done
or changed without paying for another run.

WHAT LATENCY MEANS HERE: time spent inside RAGChain.answer (embedding, search,
GPT-4o). It excludes the router, the API, and Jira. Cases run one at a time so
timings aren't distorted by concurrency.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evals.schemas import QACase, RoutingCase
from smartdesk.rag.chain import RAGResult

logger = logging.getLogger(__name__)


class TimedRetriever:
    """
    Wraps a retriever and remembers how long its last search took.

    WHY A WRAPPER: the chain does retrieval and generation in one call. Timing
    the search separately shows which half to optimise, without modifying the
    chain. Generation time is (total - retrieval).
    """

    def __init__(self, inner) -> None:
        """Args: inner: anything with search(query, department=None)."""
        self._inner = inner
        self.last_seconds = 0.0

    def search(self, query: str, department: str | None = None):
        """Delegate to the wrapped retriever, recording the elapsed time (even on error)."""
        start = time.perf_counter()
        try:
            return self._inner.search(query, department)
        finally:
            self.last_seconds = time.perf_counter() - start


@dataclass
class QAOutcome:
    """
    The raw result of one question.

    result is None when the chain raised; `error` then holds the exception CLASS
    NAME only (messages can contain request details). Errored cases are reported
    separately and excluded from metrics, so an OpenAI outage can't masquerade as
    "the agent escalated correctly".
    """

    case: QACase
    result: RAGResult | None
    error: str | None
    total_seconds: float
    retrieval_seconds: float

    @property
    def answered(self) -> bool:
        """True if the chain produced an answer (rather than escalating or erroring)."""
        return self.result is not None and self.result.answerable

    @property
    def llm_seconds(self) -> float:
        """Approximate time outside retrieval (rewrite + answer generation)."""
        return max(0.0, self.total_seconds - self.retrieval_seconds)


def run_one(chain, retriever: TimedRetriever, case: QACase) -> QAOutcome:
    """
    Ask ONE question and time it. Never raises; failures are recorded by class name.
    Used by run_qa (offline) and by the LangSmith target (experiment mode), so
    both paths measure exactly the same thing.
    """
    retriever.last_seconds = 0.0
    result, error = None, None
    start = time.perf_counter()
    try:
        result = chain.answer(case.question, department=case.department)
    except Exception as exc:
        logger.exception("Case %s failed", case.id)
        error = type(exc).__name__
    return QAOutcome(case, result, error, time.perf_counter() - start, retriever.last_seconds)


def run_qa(chain, retriever, cases, on_progress=None) -> list[QAOutcome]:
    """Ask every question once, sequentially (used with --skip-judge)."""
    outcomes = []
    for i, case in enumerate(cases, start=1):
        outcome = run_one(chain, retriever, case)
        outcomes.append(outcome)
        if on_progress:
            on_progress(i, len(cases), outcome)
    return outcomes


@dataclass
class RoutingOutcome:
    """The router's decision for one message (actual=None if it errored)."""

    case: RoutingCase
    actual: str | None
    error: str | None
    seconds: float


def run_routing(router, cases: Sequence[RoutingCase]) -> list[RoutingOutcome]:
    """
    Route each message as a FRESH message (no pending question, no known email).

    NOTE: the router falls back to kb_answer if its LLM call fails, so an outage
    shows up as routing misses on non-KB cases rather than as errors.
    """
    outcomes = []
    for case in cases:
        state = {"messages": [{"role": "user", "content": case.message}]}
        actual, error = None, None
        start = time.perf_counter()
        try:
            actual = router(state).get("route")
        except Exception as exc:
            logger.exception("Routing case %s failed", case.id)
            error = type(exc).__name__
        outcomes.append(RoutingOutcome(case, actual, error, time.perf_counter() - start))
    return outcomes