"""
Unit tests for the answer-vs-escalate rules (pure functions, no LLM, no network).

WHY THIS FILE EXISTS
--------------------
These rules ARE the product's safety net. A regression here means the bot either
answers things it shouldn't (wrong policy quoted) or escalates things it could
answer (ticket noise). Each rule gets its own test.
"""

import pytest

from smartdesk.llm.schemas import GroundedAnswer
from smartdesk.rag.confidence import EscalationReason, assess_retrieval, verify_generation
from smartdesk.rag.retriever import RetrievedChunk


def _chunk(distance, text="t"):
    """Helper: a RetrievedChunk with the given distance."""
    return RetrievedChunk(text=text, metadata={"title": "Doc", "source": "it/doc.md"}, distance=distance)


def test_no_results_escalates():
    """
    An empty search result escalates with NO_RESULTS and no best distance.
    WHY IT MATTERS: an empty index or filter must never reach the LLM.
    """
    a = assess_retrieval([], max_distance=0.6)
    assert a.should_escalate and a.reason is EscalationReason.NO_RESULTS
    assert a.best_distance is None


def test_all_chunks_too_far_escalates_with_best_distance():
    """
    When every chunk is past the cutoff, escalate LOW_SIMILARITY and report the closest.
    WHY IT MATTERS: the best distance is what you look at when tuning the threshold.
    """
    a = assess_retrieval([_chunk(0.9), _chunk(0.7)], max_distance=0.6)
    assert a.reason is EscalationReason.LOW_SIMILARITY
    assert a.best_distance == 0.7


def test_only_relevant_chunks_are_kept_closest_first():
    """
    Chunks past the cutoff are dropped; the rest are ordered closest first.
    WHY IT MATTERS: noisy passages must not reach the prompt.
    """
    a = assess_retrieval([_chunk(0.5, "b"), _chunk(0.9, "x"), _chunk(0.2, "a")], max_distance=0.6)
    assert not a.should_escalate
    assert [c.text for c in a.relevant] == ["a", "b"]


def test_chunk_exactly_at_threshold_is_kept():
    """
    The cutoff is inclusive (<=).
    WHY IT MATTERS: pins down boundary behaviour so tuning is predictable.
    """
    assert assess_retrieval([_chunk(0.6)], max_distance=0.6).relevant


def test_model_saying_dont_know_escalates():
    """
    answerable=False maps to LLM_INSUFFICIENT.
    WHY IT MATTERS: this is the "I don't know" path required by the spec.
    """
    g = GroundedAnswer(answerable=False, answer="I don't know", used_sources=[])
    assert verify_generation(g, 3) is EscalationReason.LLM_INSUFFICIENT


@pytest.mark.parametrize("sources", [[], [4], [0]])
def test_answer_without_valid_citation_is_ungrounded(sources):
    """
    A "confident" answer citing nothing, or a non-existent passage, escalates.
    WHY IT MATTERS: it catches the model answering from memory instead of context.
    """
    g = GroundedAnswer(answerable=True, answer="Do X", used_sources=sources)
    assert verify_generation(g, 3) is EscalationReason.UNGROUNDED


def test_valid_grounded_answer_passes():
    """
    A confident answer citing existing passages is accepted (returns None).
    WHY IT MATTERS: guards against rules so strict that good answers escalate.
    """
    g = GroundedAnswer(answerable=True, answer="Do X", used_sources=[1, 3])
    assert verify_generation(g, 3) is None
