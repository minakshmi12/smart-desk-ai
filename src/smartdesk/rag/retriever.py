"""
Dense (vector) search over the knowledge base, and the shared hit type.

WHY THIS FILE EXISTS
--------------------
Turns a question into the nearest stored chunks. It does NOT decide whether they
are good enough (rag/confidence.py does) and does not do keyword search or
reranking (rag/hybrid.py does). RetrievedChunk is shared by all retrievers.
"""

from dataclasses import dataclass

from smartdesk.config import get_settings
from smartdesk.llm.client import LLMClient
from smartdesk.rag.vector_store import get_collection


def label_for(metadata: dict) -> str:
    """Citation text from chunk metadata, e.g. 'Leave Policy (hr/leave_policy.pdf, p.2)'."""
    title = metadata.get("title", "Unknown document")
    page = metadata.get("page")
    where = metadata.get("source", "")
    return f"{title} ({where}{f', p.{page}' if page else ''})"


@dataclass(frozen=True)
class RetrievedChunk:
    """
    One search hit.

    distance: cosine distance to the question (0 = same direction, LOWER IS BETTER).
        Hybrid search computes it for every returned chunk, so the confidence gate
        works the same for all retrieval modes.
    The optional fields are filled by the hybrid pipeline and are None otherwise:
    id (chunk id), sparse_score (BM25), fused_score (RRF), rerank_score (0-1 from
    the LLM reranker; raw model score for a cross-encoder).
    """

    text: str
    metadata: dict
    distance: float
    id: str = ""
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def title(self) -> str:
        """Human-friendly document name, used in citations."""
        return self.metadata.get("title", "Unknown document")

    @property
    def label(self) -> str:
        """Citation text, e.g. 'Leave Policy (hr/leave_policy.pdf, p.2)'."""
        return label_for(self.metadata)


class Retriever:
    """Dense retrieval: embeds a query and returns the closest chunks from Chroma."""

    def __init__(self, llm: LLMClient | None = None, collection=None) -> None:
        """
        Args:
            llm: used only for `embed`. Injectable so tests need no API key.
            collection: optional Chroma collection (or a fake) for tests.
        """
        self._llm = llm or LLMClient()
        self._collection = collection
        self._top_k = get_settings().top_k

    def search(
        self, query: str, department: str | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """
        Return up to `top_k` chunks, closest first; [] for an empty index.

        The collection is reopened per search so a re-ingest (which recreates it)
        is picked up by a long-running server without a restart.
        """
        collection = self._collection if self._collection is not None else get_collection()
        total = collection.count()
        if total == 0:
            return []

        [vector] = self._llm.embed([query])
        result_limit = self._top_k if top_k is None else top_k
        res = collection.query(
            query_embeddings=[vector],
            n_results=min(result_limit, total),
            where={"department": department} if department else None,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        ids = (res.get("ids") or [[""] * len(docs)])[0]
        return [
            RetrievedChunk(text=t, metadata=m, distance=d, id=i)
            for i, t, m, d in zip(ids, docs, res["metadatas"][0], res["distances"][0])
        ]