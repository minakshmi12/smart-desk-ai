"""
Show retrieval stage by stage, then the grounded answer.

Usage: python scripts/demo_retrieval.py ["question" ...] [--department it]

For each question it prints the dense, BM25, fused and reranked rankings, then
answers using EXACTLY those final chunks (so the output proves which passages the
model saw), with citations, or the escalation reason. Questions the documents
don't cover should end in "I don't know", not an invented answer.
"""

import argparse

from smartdesk.llm.client import LLMClient
from smartdesk.rag.chain import RAGChain
from smartdesk.rag.hybrid import HybridRetriever, build_reranker

DEFAULTS = [
    "How do I fix VPN error 809 on my Mac?",       # exact-token question
    "What ports must be open for the VPN?",         # paraphrase question
    "Can I expense a standing desk?",               # not in the documents
]


class _Fixed:
    """Retriever stub that returns the chunks we already retrieved."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, department=None):
        return self.chunks


def _rows(title, rows, labels, fmt):
    """Print the top 5 rows of one stage."""
    print(f"  {title}")
    for rank, (id_, score) in enumerate(rows[:5], start=1):
        print(f"    {rank}. {labels.get(id_, id_):<48} {fmt(score)}")
    if not rows:
        print("    (none)")


def main() -> None:
    """Run the demo for each question."""
    parser = argparse.ArgumentParser(description="Demonstrate hybrid retrieval.")
    parser.add_argument("queries", nargs="*", default=DEFAULTS)
    parser.add_argument("--department")
    args = parser.parse_args()

    llm = LLMClient()
    retriever = HybridRetriever(llm, mode="hybrid", reranker=build_reranker())
    for q in args.queries:
        trace = retriever.trace(q, args.department)
        print(f"\n{'=' * 78}\nQUESTION: {q}")
        _rows("dense (cosine distance, lower is better)", trace.dense, trace.labels, lambda s: f"{s:.3f}")
        _rows("BM25 keyword (higher is better)", trace.sparse, trace.labels, lambda s: f"{s:.2f}")
        _rows("fused (RRF)", trace.fused, trace.labels, lambda s: f"{s:.4f}")
        print("  FINAL after rerank (what the LLM may see)")
        for rank, c in enumerate(trace.chunks, start=1):
            rr = "n/a" if c.rerank_score is None else f"{c.rerank_score:.2f}"
            print(f"    {rank}. {c.label:<48} distance {c.distance:.3f} | rerank {rr}")

        result = RAGChain(retriever=_Fixed(trace.chunks), llm=llm).answer(q)
        print(f"  passages given to the model after the confidence gate: {len(result.contexts)}")
        if result.answerable:
            print(f"  ANSWER: {result.answer}\n  CITED:  {'; '.join(result.sources)}")
        else:
            print(f"  ESCALATED ({result.reason.value}): {result.answer}")


if __name__ == "__main__":
    main()