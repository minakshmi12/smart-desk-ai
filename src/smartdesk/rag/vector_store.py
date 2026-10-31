"""
Single place that knows how the Chroma collection is created and opened.

WHY THIS FILE EXISTS
--------------------
Both sides of RAG touch the same collection: ingest.py writes to it and
retriever.py (next step) reads from it. If each configured Chroma on its own,
they could end up with different distance metrics, and similarity scores would
silently mean different things. One module removes that risk.
"""

import chromadb
from chromadb.api.models.Collection import Collection

from smartdesk.config import get_settings

COLLECTION_NAME = "smartdesk_kb"


def get_collection(reset: bool = False) -> Collection:
    """
    Open (or create) the knowledge-base collection.

    Args:
        reset: if True, delete the existing collection first and return an
            empty one. Used by a full re-ingest so documents that were edited
            or removed don't leave stale chunks behind.

    WHY COSINE DISTANCE: OpenAI embeddings are compared by angle, not length.
    Chroma then reports distance = 1 - cosine_similarity, so LOWER is better
    (0 = identical direction). The confidence threshold in config.py must be
    interpreted with this in mind; we will pin down its exact meaning when we
    write rag/confidence.py and tune it with evals/.

    WHY A PERSISTENT CLIENT: the index survives restarts, so the API server
    doesn't re-embed every document on startup (slow and costs money).

    NOTE: the "hnsw:space" metadata key is how Chroma 0.5/0.6 selects the metric.
    Newer major versions expose it as a configuration object; if you upgrade
    Chroma, check its docs for the equivalent.
    """
    client = chromadb.PersistentClient(path=get_settings().vector_store_dir)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Collection didn't exist yet. The exception type differs between
            # Chroma versions, so we catch broadly; "nothing to delete" is fine.
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )