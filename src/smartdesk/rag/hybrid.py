"""
Hybrid retrieval: dense + BM25, fused with RRF, then reranked.

WHY THIS FILE EXISTS
--------------------
Combines the two retrievers and the reranker behind the same `search(query,
department)` interface as the plain Retriever, so RAGChain, the evals and the
timing wrapper need no changes. `trace()` exposes every stage for demos and debugging.
"""

import logging
import math
import threading
from dataclasses import dataclass, field

from smartdesk.config import get_settings
from smartdesk.llm.client import LLMClient
from smartdesk.rag.fusion import reciprocal_rank_fusion
from smartdesk.rag.rerank import CrossEncoderReranker, LLMReranker, NoReranker, Reranker
from smartdesk.rag.retriever import RetrievedChunk, Retriever, label_for
from smartdesk.rag.sparse import BM25Index, SparseDoc
from smartdesk.rag.vector_store import get_collection, read_index_version

logger = logging.getLogger(__name__)

MAX_DISTANCE = 2.0  # largest possible cosine distance; used if an embedding can't be fetched


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity, clamped to [0, 2]; 1.0 if either vector is all zeros."""
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return min(MAX_DISTANCE, max(0.0, 1.0 - dot / (na * nb)))


def _distances_for(collection, ids: list[str], vector: list[float]) -> dict[str, float]:
    """
    Exact cosine distance from the query to chunks the dense search did not return.

    WHY: BM25-only hits have no distance, but the confidence gate needs one. Chroma
    stores each chunk's embedding, so we compute it ourselves. Ids whose embedding
    can't be fetched are simply absent (the caller uses MAX_DISTANCE).
    """
    if not ids:
        return {}
    res = collection.get(ids=list(ids), include=["embeddings"])
    embeddings = res.get("embeddings")
    if embeddings is None:  # note: may be a numpy array, so never use `if embeddings:`
        return {}
    return {
        id_: cosine_distance(vector, [float(x) for x in emb])
        for id_, emb in zip(res["ids"], embeddings)
    }


@dataclass
class RetrievalTrace:
    """
    Every stage of one search, for demos, debugging and tests.

    dense: (chunk id, cosine distance) best first.  sparse: (id, BM25 score).
    fused: (id, RRF score).  chunks: the final reranked result (what the LLM sees,
    before the confidence gate).  labels: id -> citation text.
    """

    query: str
    mode: str
    dense: list[tuple[str, float]] = field(default_factory=list)
    sparse: list[tuple[str, float]] = field(default_factory=list)
    fused: list[tuple[str, float]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


class HybridRetriever:
    """Dense and/or BM25 retrieval, RRF fusion, optional reranking."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        collection=None,
        reranker: Reranker | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            llm: used only for `embed` (injectable for tests).
            collection: optional Chroma collection or fake for tests.
            reranker: defaults to no reranking; build_retriever() supplies the configured one.
            mode: "sparse" or "hybrid" (defaults to RETRIEVAL_MODE). Plain "dense"
                belongs to Retriever. The query is embedded in every mode, because
                the confidence gate needs a cosine distance for each chunk.
        """
        s = get_settings()
        self._llm = llm or LLMClient()
        self._collection = collection
        self._reranker = reranker or NoReranker()
        self._mode = mode or s.retrieval_mode
        if self._mode not in ("sparse", "hybrid"):
            raise ValueError("HybridRetriever mode must be 'sparse' or 'hybrid'; use Retriever for dense")
        self._top_k, self._pool, self._rrf_k = s.top_k, s.candidate_pool, s.rrf_k
        self._rerank_n = max(s.rerank_candidates, s.top_k)
        self._lock = threading.Lock()
        self._index: BM25Index | None = None
        self._fingerprint: tuple | None = None

    def search(
        self, query: str, department: str | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks, best first ([] for an empty index)."""
        return self.trace(query, department, top_k=top_k).chunks

    def trace(
        self, query: str, department: str | None = None, top_k: int | None = None
    ) -> RetrievalTrace:
        """
        Run the full pipeline and return every stage.

        1. Embed the query; dense top-N from Chroma (skipped in sparse mode).
        2. BM25 top-N, filtered by department.
        3. RRF over the two rankings; keep the best candidates.
        4. Fill in cosine distance for candidates dense search didn't return.
        5. Rerank to top_k.
        """
        collection = self._collection if self._collection is not None else get_collection()
        total = collection.count()
        trace = RetrievalTrace(query=query, mode=self._mode)
        if total == 0:
            return trace
        result_limit = self._top_k if top_k is None else top_k
        [vector] = self._llm.embed([query])

        known: dict[str, tuple[str, dict]] = {}   # id -> (text, metadata)
        distance: dict[str, float] = {}
        sparse_score: dict[str, float] = {}

        if self._mode == "hybrid":
            res = collection.query(
                query_embeddings=[vector], n_results=min(self._pool, total),
                where={"department": department} if department else None,
                include=["documents", "metadatas", "distances"],
            )
            for id_, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0]):
                known[id_], distance[id_] = (text, meta), dist
                trace.dense.append((id_, dist))

        for hit in self._sparse_index(collection, total).search(query, self._pool, department):
            known.setdefault(hit.doc.id, (hit.doc.text, hit.doc.metadata))
            sparse_score[hit.doc.id] = hit.score
            trace.sparse.append((hit.doc.id, hit.score))

        trace.fused = reciprocal_rank_fusion(
            [[i for i, _ in trace.dense], [i for i, _ in trace.sparse]], self._rrf_k
        )
        fused_score = dict(trace.fused)
        top_ids = [i for i, _ in trace.fused[: max(self._rerank_n, result_limit)]]
        distance.update(_distances_for(collection, [i for i in top_ids if i not in distance], vector))

        candidates = [
            RetrievedChunk(
                text=known[i][0], metadata=known[i][1], distance=distance.get(i, MAX_DISTANCE),
                id=i, sparse_score=sparse_score.get(i), fused_score=fused_score[i],
            )
            for i in top_ids
        ]
        trace.labels = {i: label_for(meta) for i, (_, meta) in known.items()}
        trace.chunks = self._reranker.rerank(query, candidates, result_limit)
        return trace
    def _sparse_index(self, collection, total: int) -> BM25Index:
            """
            The BM25 index, rebuilt only when the collection changes.
    
            Fingerprint = (chunk count, ingest version stamp). Built from Chroma's
            stored text, so it can never disagree with the vector index. Under a lock
            so two request threads don't build it twice. Holds all chunk text in RAM,
            which is fine for thousands of chunks; for far more, use a search server.
            """
            fingerprint = (total, read_index_version())
            with self._lock:
                if self._index is None or self._fingerprint != fingerprint:
                    data = collection.get(include=["documents", "metadatas"])
                    docs = [SparseDoc(i, t, m or {}) for i, t, m in
                            zip(data["ids"], data["documents"], data["metadatas"])]
                    self._index, self._fingerprint = BM25Index(docs), fingerprint
                    logger.info("Built BM25 index over %d chunks", len(docs))
                return self._index
MODE_INFO = {
    "dense": {
        "label": "Semantic (dense vectors)",
        "description": "Embeds the question and finds the chunks closest in meaning "
                       "(Chroma, cosine distance). Best for paraphrased questions.",
    },
    "sparse": {
        "label": "Keyword (BM25)",
        "description": "Ranks chunks by exact word overlap, then reranks. "
                       "Best for error codes and product names.",
    },
    "hybrid": {
        "label": "Hybrid (semantic + keyword)",
        "description": "Runs both, merges the rankings with Reciprocal Rank Fusion, then reranks.",
    },
}
MODES = tuple(MODE_INFO)


class MultiRetriever:
    """
    Every search method behind one object; the method is chosen per call.

    WHY: the UI lets the employee pick a method per message, and the compare page
    runs all of them. Retrievers are built lazily (the BM25 index and reranker are
    only created if that method is used) and share one reranker. Adding a method =
    add it to MODE_INFO and to _get().

    NOTE: "sparse" and "hybrid" each hold their own BM25 index in RAM.
    """

    def __init__(self, llm: LLMClient | None = None, default_mode: str | None = None) -> None:
        """default_mode falls back to RETRIEVAL_MODE."""
        self._llm = llm or LLMClient()
        self.default_mode = default_mode or get_settings().retrieval_mode
        self._retrievers: dict = {}
        self._reranker = None
        self._lock = threading.Lock()

    def _get(self, mode: str):
        """Build (once) and return the retriever for `mode`."""
        with self._lock:
            retriever = self._retrievers.get(mode)
            if retriever is None:
                if mode == "dense":
                    retriever = Retriever(self._llm)
                else:
                    if self._reranker is None:
                        self._reranker = build_reranker()
                    retriever = HybridRetriever(self._llm, mode=mode, reranker=self._reranker)
                self._retrievers[mode] = retriever
            return retriever

    def search(
        self, query: str, department: str | None = None,
        mode: str | None = None, top_k: int | None = None,
    ):
        """
        Search with `mode` (default: the configured one).

        Raises:
            ValueError: unknown mode (the API rejects these earlier with a 422).
        """
        mode = mode or self.default_mode
        if mode not in MODE_INFO:
            raise ValueError(f"Unknown retrieval mode: {mode!r}")
        options = {"top_k": top_k} if top_k is not None else {}
        return self._get(mode).search(query, department, **options)




    


def build_reranker() -> Reranker:
    """Create the reranker selected by RERANKER (none / llm / cross_encoder)."""
    s = get_settings()
    if s.reranker == "llm":
        return LLMReranker(LLMClient(model=s.rerank_model), min_score=s.rerank_min_score)
    if s.reranker == "cross_encoder":
        return CrossEncoderReranker(model_name=s.cross_encoder_model)
    return NoReranker()


def build_retriever(llm: LLMClient | None = None) -> MultiRetriever:
    """The retriever used by the app: all methods available, RETRIEVAL_MODE is the default."""
    return MultiRetriever(llm)