"""
The RAG chain: question -> retrieve -> check -> generate -> verify.

Returns either a grounded answer or a clear "escalate" result, plus a
`retrieval` dict describing what the search found (for the UI and live evaluation).
"""

import logging
import time
from dataclasses import dataclass, field

from smartdesk.agent.prompts import DONT_KNOW, build_answer_messages, build_rewrite_messages
from smartdesk.config import get_settings
from smartdesk.llm.client import LLMClient
from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.confidence import EscalationReason, assess_retrieval, verify_generation
from smartdesk.rag.display import summarize_chunks
from smartdesk.rag.hybrid import build_retriever

logger = logging.getLogger(__name__)

_HISTORY_TURNS = 6  # messages of history used for follow-up rewriting


@dataclass(frozen=True)
class RAGResult:
    """
    ANSWERED -> answerable=True, answer=<text>, sources=<cited>, reason=None
    ESCALATE -> answerable=False, answer="I don't know", reason=<why>

    consulted: every retrieved label (passed to the human agent on escalation).
    contexts: passage texts shown to the LLM (used by evals).
    retrieval: JSON-friendly description of the search (mode, reranker, timings,
        per-chunk scores and gate/citation flags); None if the chain failed early.
    """

    answerable: bool
    answer: str
    standalone_question: str
    sources: list[str] = field(default_factory=list)
    consulted: list[str] = field(default_factory=list)
    best_distance: float | None = None
    reason: EscalationReason | None = None
    contexts: list[str] = field(default_factory=list)
    retrieval: dict | None = None


class RAGChain:
    """Retrieval-augmented answering with built-in escalation signals."""

    def __init__(self, retriever=None, llm: LLMClient | None = None) -> None:
        """Both injectable for tests; the default retriever supports every search method."""
        self._llm = llm or LLMClient()
        self._retriever = retriever or build_retriever(self._llm)

    def answer(
        self, question: str, history: list[dict] | None = None,
        department: str | None = None, mode: str | None = None, top_k: int | None = None,
    ) -> RAGResult:
        """
        Run the full chain for one question.

        `mode` and `top_k` select the search method and result count for this call.
        They are passed only when set, so simple retrievers remain compatible.
        """
        started = time.perf_counter()
        settings = get_settings()
        standalone = self._standalone(question, history or [])

        t0 = time.perf_counter()
        search_options = {key: value for key, value in (("mode", mode), ("top_k", top_k))
                  if value is not None}
        chunks = self._retriever.search(standalone, department, **search_options)
        retrieval_seconds = time.perf_counter() - t0
        consulted = [c.label for c in chunks]
        effective_mode = mode or getattr(self._retriever, "default_mode", None)

        def retrieval_info(cited=()) -> dict:
            """Built at return time so seconds_total includes generation."""
            return {
                "mode": effective_mode,
                "reranker": settings.reranker if effective_mode in ("sparse", "hybrid") else "none",
                "question": standalone,
                "threshold": settings.confidence_threshold,
                "top_k": settings.top_k if top_k is None else top_k,
                "seconds_total": round(time.perf_counter() - started, 3),
                "seconds_retrieval": round(retrieval_seconds, 3),
                "chunks": summarize_chunks(chunks, settings.confidence_threshold, cited),
            }

        assessment = assess_retrieval(chunks)
        if assessment.should_escalate:
            logger.info("Escalating (%s); best distance=%s", assessment.reason, assessment.best_distance)
            return self._escalate(standalone, assessment.reason, consulted,
                                  assessment.best_distance, retrieval_info())

        relevant = assessment.relevant
        grounded = self._llm.parse(build_answer_messages(standalone, relevant), GroundedAnswer)

        reason = verify_generation(grounded, len(relevant))
        if reason is not None:
            logger.info("Escalating (%s) after generation", reason)
            return self._escalate(standalone, reason, consulted,
                                  assessment.best_distance, retrieval_info())

        cited = sorted(set(grounded.used_sources))
        cited_chunks = [relevant[n - 1] for n in cited]
        return RAGResult(
            answerable=True,
            answer=grounded.answer.strip(),
            standalone_question=standalone,
            sources=[c.label for c in cited_chunks],
            consulted=consulted,
            contexts=[c.text for c in relevant],
            best_distance=assessment.best_distance,
            retrieval=retrieval_info(cited_chunks),
        )

    def _standalone(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up into a standalone question; fall back to the original on failure."""
        recent = [m for m in history if m.get("role") in ("user", "assistant")][-_HISTORY_TURNS:]
        if not recent:
            return question
        try:
            rewritten = self._llm.chat(build_rewrite_messages(question, recent), temperature=0)
            return rewritten.strip() or question
        except Exception:
            logger.exception("Question rewrite failed; using the original question")
            return question

    @staticmethod
    def _escalate(question, reason, consulted, best_distance, retrieval) -> RAGResult:
        """Build the 'cannot answer' result with the standard 'I don't know' text."""
        return RAGResult(
            answerable=False, answer=DONT_KNOW, standalone_question=question,
            consulted=consulted, best_distance=best_distance, reason=reason,
            retrieval=retrieval,
        )