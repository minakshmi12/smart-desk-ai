"""JSON-friendly summaries of retrieved chunks, shared by the chain and the compare endpoint."""

from collections.abc import Iterable

from smartdesk.rag.retriever import RetrievedChunk


def summarize_chunks(
    chunks: Iterable[RetrievedChunk], threshold: float, cited: Iterable[RetrievedChunk] = ()
) -> list[dict]:
    """
    One dict per chunk, in rank order, for the API and the UI.

    passed_gate: distance <= threshold, i.e. the chunk was allowed into the LLM prompt.
    cited: the model used this chunk (matched by object identity, so two chunks from
        the same document are not confused). Scores a retriever did not compute are None.
    """
    cited_ids = {id(c) for c in cited}
    rows = []
    for rank, c in enumerate(chunks, start=1):
        rows.append({
            "rank": rank,
            "label": c.label,
            "source": c.metadata.get("source"),
            "page": c.metadata.get("page"),
            "text": c.text,
            "distance": round(c.distance, 4),
            "bm25": None if c.sparse_score is None else round(c.sparse_score, 3),
            "fused": None if c.fused_score is None else round(c.fused_score, 5),
            "rerank": None if c.rerank_score is None else round(c.rerank_score, 3),
            "passed_gate": c.distance <= threshold,
            "cited": id(c) in cited_ids,
        })
    return rows