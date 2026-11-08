"""
The single place that decides: answer, or escalate to a ticket?

WHY THIS FILE EXISTS
--------------------
"Gracefully handle questions it cannot answer" is the core promise of the
product. If that decision were scattered across prompts, nodes, and the chain,
tuning it would mean hunting through the codebase. Here, every rule is a small,
pure, unit-testable function with no LLM or network access.

TWO CHECKPOINTS
---------------
1. assess_retrieval   -> before the LLM call (cheap, skips GPT-4o when hopeless)
2. verify_generation  -> after the LLM call (checks the model's own verdict)
"""

from dataclasses import dataclass
from enum import Enum

from smartdesk.config import get_settings
from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.retriever import RetrievedChunk


class EscalationReason(str, Enum):
    """
    Why the agent gave up on answering.

    WHY AN ENUM: the reason is (a) shown to the human agent in the ticket,
    (b) logged for evals, so you can see which signal causes most escalations,
    and (c) used to pick the wording of the offer. Free-text reasons would make
    all three unreliable.
    """

    NO_RESULTS = "no_results"            # nothing retrieved at all
    LOW_SIMILARITY = "low_similarity"    # retrieved, but all too far from the question
    LLM_INSUFFICIENT = "llm_insufficient"  # model says the context doesn't answer it
    UNGROUNDED = "ungrounded"            # model claimed an answer but cited nothing valid
    ERROR = "error"                      # something failed (API down, refusal...)


@dataclass(frozen=True)
class RetrievalAssessment:
    """
    Outcome of checkpoint 1.

    relevant:      chunks that passed the distance filter, closest first. Only
                   these are shown to the LLM, so weak matches can't distract it.
    reason:        None means "proceed to the LLM"; otherwise the escalation reason.
    best_distance: distance of the closest chunk (None if nothing retrieved).
                   Kept for logging and threshold tuning.
    """

    relevant: list[RetrievedChunk]
    reason: EscalationReason | None
    best_distance: float | None

    @property
    def should_escalate(self) -> bool:
        """True when the chain should skip the LLM and offer a ticket."""
        return self.reason is not None


def assess_retrieval(
    chunks: list[RetrievedChunk], max_distance: float | None = None
) -> RetrievalAssessment:
    """
    Checkpoint 1: is there anything in the search results worth showing the LLM?

    Args:
        chunks: raw search results (any order).
        max_distance: cutoff; defaults to CONFIDENCE_THRESHOLD. A chunk exactly
            at the cutoff is kept (<=).

    WHY FILTER, NOT JUST CHECK THE BEST CHUNK: if the top hit is good but the
    4th is noise, passing the noise along invites the model to blend unrelated
    text into its answer. Only passages under the cutoff reach the prompt.
    """
    limit = get_settings().confidence_threshold if max_distance is None else max_distance
    if not chunks:
        return RetrievalAssessment([], EscalationReason.NO_RESULTS, None)

    ordered = sorted(chunks, key=lambda c: c.distance)
    best = ordered[0].distance
    relevant = [c for c in ordered if c.distance <= limit]
    if not relevant:
        return RetrievalAssessment([], EscalationReason.LOW_SIMILARITY, best)
    return RetrievalAssessment(relevant, None, best)


def verify_generation(grounded: GroundedAnswer, n_passages: int) -> EscalationReason | None:
    """
    Checkpoint 2: can we trust the model's answer?

    Returns None if the answer is acceptable, otherwise the escalation reason.

    RULES
    - The model said it can't answer          -> LLM_INSUFFICIENT.
    - It said it can, but cited no passages   -> UNGROUNDED.
    - It cited a passage number that doesn't
      exist (e.g. [7] with 3 passages)        -> UNGROUNDED.

    WHY THE CITATION RULES: they turn "I'm sure" into something checkable. A
    confident answer with no valid citation is the signature of the model
    answering from its own memory, which is exactly what this bot must not do
    (it could state a policy the company doesn't have).
    """
    if not grounded.answerable:
        return EscalationReason.LLM_INSUFFICIENT
    cited = set(grounded.used_sources)
    if not cited or any(n < 1 or n > n_passages for n in cited):
        return EscalationReason.UNGROUNDED
    return None
