"""
The RAG chain: question -> retrieve -> check -> generate -> verify.

WHY THIS FILE EXISTS
--------------------
Wires the retriever, the confidence rules, and GPT-4o into one callable with a
simple contract: give it a question, get back either a grounded answer or a
clear "escalate" result. The agent nodes never see Chroma or prompts.
"""

import logging
from dataclasses import dataclass, field

from smartdesk.agent.prompts import (
    DONT_KNOW,
    build_answer_messages,
    build_rewrite_messages,
)
from smartdesk.llm.client import LLMClient
from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.confidence import (
    EscalationReason,
    assess_retrieval,
    verify_generation,
)
from smartdesk.rag.retriever import Retriever

logger = logging.getLogger(__name__)

_HISTORY_TURNS = 6  # messages of history used for follow-up rewriting



@dataclass(frozen=True)
class RAGResult:
    """
    What the chain returns. Exactly one of two shapes:

    ANSWERED  -> answerable=True,  answer=<text>, sources=<cited>, reason=None
    ESCALATE  -> answerable=False, answer="I don't know", reason=<why>

    consulted lists EVERYTHING retrieved, even when escalating. WHY: the human
    agent who picks up the ticket can see which documents were already checked
    and doesn't repeat that work.
    """

    answerable: bool
    answer: str
    standalone_question: str
    sources: list[str] = field(default_factory=list)     # citations actually used
    consulted: list[str] = field(default_factory=list)   # all retrieved labels
    best_distance: float | None = None
    reason: EscalationReason | None = None
    contexts: list[str] = field(default_factory=list)  # passage texts shown to the LLM (for evals)


class RAGChain:
    """Retrieval-augmented answering with built-in escalation signals."""

    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None) -> None:
        """
        Args:
            retriever: injectable for tests (a fake returning canned chunks).
            llm: injectable for tests. Shared with the retriever so there is a
                single OpenAI client.
        """
        self._llm = llm or LLMClient()
        self._retriever = retriever or Retriever(self._llm)

    def answer(
        self, question: str, history: list[dict] | None = None, department: str | None = None
    ) -> RAGResult:
        """
        Run the full chain for one question.

        STEPS
        1. Rewrite follow-ups into a standalone question (only if history exists).
        2. Retrieve candidate chunks.
        3. Checkpoint 1 (assess_retrieval): escalate without calling GPT-4o if
           nothing relevant was found.
        4. Generate a structured, citation-bearing answer from relevant chunks only.
        5. Checkpoint 2 (verify_generation): escalate if the model said "I don't
           know" or its citations don't check out.

        Typical cost on the happy path: one embedding + one GPT-4o call
        (+ one small rewrite call on follow-up turns).
        """
        standalone = self._standalone(question, history or [])
        chunks = self._retriever.search(standalone, department)
        consulted = [c.label for c in chunks]

        assessment = assess_retrieval(chunks)
        if assessment.should_escalate:
            logger.info("Escalating (%s); best distance=%s", assessment.reason, assessment.best_distance)
            return self._escalate(standalone, assessment.reason, consulted, assessment.best_distance)

        relevant = assessment.relevant
        grounded = self._llm.parse(build_answer_messages(standalone, relevant), GroundedAnswer)

        reason = verify_generation(grounded, len(relevant))
        if reason is not None:
            logger.info("Escalating (%s) after generation", reason)
            return self._escalate(standalone, reason, consulted, assessment.best_distance)

        cited = sorted(set(grounded.used_sources))
        return RAGResult(
            answerable=True,
            answer=grounded.answer.strip(),
            standalone_question=standalone,
            sources=[relevant[n - 1].label for n in cited],
            consulted=consulted,
            contexts=[c.text for c in relevant],
            best_distance=assessment.best_distance,
        )
    # ------------------------------------------------------------------
    def _standalone(self, question: str, history: list[dict]) -> str:
        """
        Turn a follow-up like "what about Windows?" into a searchable question.

        WHY: embeddings of "what about Windows?" match nothing useful. Skipped
        entirely when there is no history (saves a call on the first message).
        If the rewrite call fails, fall back to the original question: a worse
        search is better than a failed turn.
        """
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
    def _escalate(question, reason, consulted, best_distance) -> RAGResult:
        """Build the 'cannot answer' result with the standard 'I don't know' text."""
        return RAGResult(
            answerable=False,
            answer=DONT_KNOW,
            standalone_question=question,
            consulted=consulted,
            best_distance=best_distance,
            reason=reason,
        )