"""
Semantic search over the knowledge base.

WHY THIS FILE EXISTS
--------------------
Turns a question into the most relevant stored chunks. It deliberately does NOT
decide whether those chunks are good enough: that judgment lives in
rag/confidence.py, so the retriever stays a plain search component.
"""

from dataclasses import dataclass

from smartdesk.config import get_settings
from smartdesk.llm.client import LLMClient
from smartdesk.rag.vector_store import get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """
    One search hit.

    `distance` is Chroma's cosine distance: 0 = same direction, larger = less
    similar. LOWER IS BETTER. Everything downstream (confidence checks) uses
    this convention, so it is kept raw rather than converted to a "score".
    """

    text: str
    metadata: dict
    distance: float

    @property
    def title(self) -> str:
        """Human-friendly document name, used in citations."""
        return self.metadata.get("title", "Unknown document")

    @property
    def label(self) -> str:
        """Citation text, e.g. 'Leave Policy (hr/leave_policy.pdf, p.2)'."""
        page = self.metadata.get("page")
        where = self.metadata.get("source", "")
        return f"{self.title} ({where}{f', p.{page}' if page else ''})"


class Retriever:
    """Embeds a query and returns the closest chunks from the vector store."""

    def __init__(self, llm: LLMClient | None = None, collection=None) -> None:
        """
        Args:
            llm: used only for `embed`. Injectable so tests need no API key.
            collection: optional Chroma collection (or a fake). WHY: tests
                inject one; in production it stays None and the collection is
                opened fresh on each search.
        """
        self._llm = llm or LLMClient()
        self._collection = collection
        self._top_k = get_settings().top_k

    def search(self, query: str, department: str | None = None) -> list[RetrievedChunk]:
        """
        Return up to `top_k` chunks, closest first.

        Args:
            query: a standalone question (follow-ups are rewritten before this).
            department: optional "it" / "hr" filter. WHY: a leave question
                shouldn't be able to match an IT document that happens to share
                words. Left None, the whole index is searched.

        Returns:
            [] if the index is empty or nothing matches the filter. An empty
            result is a normal outcome that triggers escalation, not an error.

        WHY THE COLLECTION IS OPENED PER SEARCH: after you re-run ingestion
        (which deletes and recreates the collection), a handle cached at server
        startup would point at the deleted one. Reopening keeps a long-running
        API server correct without a restart.
        """
        collection = self._collection if self._collection is not None else get_collection()
        total = collection.count()
        if total == 0:
            return []

        [vector] = self._llm.embed([query])
        res = collection.query(
            query_embeddings=[vector],
            n_results=min(self._top_k, total),
            where={"department": department} if department else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(text=t, metadata=m, distance=d)
            for t, m, d in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
    