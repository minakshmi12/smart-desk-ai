"""
Build the searchable knowledge base: load -> chunk -> embed -> store.

WHY THIS FILE EXISTS
--------------------
Retrieval quality is decided mostly here, before any question is asked. How
documents are cut into chunks determines whether the right passage can be found,
and whether it contains enough context to answer. Keeping the pipeline in one
module makes it easy to re-run after documents change.

WHY CHUNKING AT ALL
-------------------
A whole policy PDF is too broad to match a specific question ("how many days of
sick leave?"), and its embedding is a blurry average of everything inside.
Small focused chunks produce sharper matches and keep the prompt sent to GPT-4o
short and cheap.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from smartdesk.config import get_settings
from smartdesk.llm.client import LLMClient
from smartdesk.rag.loaders import SourceDocument, load_directory
from smartdesk.rag.vector_store import get_collection, write_index_version

logger = logging.getLogger(__name__)

_BATCH = 100  # chunks per Chroma write; matches LLMClient.embed's batch size


@dataclass(frozen=True)
class Chunk:
    """
    One searchable passage: the unit that is embedded, stored, and retrieved.

    WHY `id` IS DETERMINISTIC: it is built from source, page, and position
    (e.g. "hr/leave_policy.pdf#2#0"), so the same file always produces the same
    IDs. That makes runs reproducible and lets answers reference exactly where
    information came from.
    """

    id: str
    text: str            # the passage shown to the LLM and the employee
    embed_text: str      # what is actually embedded (see _to_chunks for why it differs)
    metadata: dict


@dataclass
class IngestReport:
    """
    Summary returned by ingest(), so callers (CLI, tests, future admin API)
    can show what happened without parsing log lines.
    """

    documents: int = 0                 # pages/files loaded
    chunks: int = 0                    # chunks stored
    by_department: dict[str, int] = field(default_factory=dict)  # chunk count per dept


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Cut an oversized paragraph into overlapping fixed-size windows.

    WHY IT EXISTS: normally we split on paragraph boundaries, but a single huge
    paragraph (a pasted table, a wall of text) can exceed chunk_size by itself.
    This is the last-resort fallback. It may cut mid-word; that's acceptable
    because it only applies to unusually long paragraphs.

    Each window starts `chunk_size - overlap` characters after the previous
    one, so neighbouring windows share `overlap` characters. We stop as soon
    as a window reaches the end, which avoids a final window that is entirely
    contained in the previous one.
    """
    step = chunk_size - overlap
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return pieces


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    preserve_paragraphs: bool = False,
) -> list[str]:
    """
    Split text into chunks of at most `chunk_size` characters, respecting
    paragraph boundaries where possible.

    ALGORITHM
    1. Split into paragraphs on blank lines. Paragraphs larger than chunk_size
       are hard-split (see _hard_split).
    2. Greedily pack paragraphs into a chunk until the next one wouldn't fit.
    3. When starting a new chunk, carry over the last `overlap` characters of
       the previous chunk (trimmed to a word boundary) as lead-in context.

    WHY PARAGRAPH-AWARE: a rule like "Employees get 20 days of leave" stays
    next to its exceptions instead of being cut mid-sentence, which gives
    better matches and better answers than blind fixed-size slicing.

    WHY OVERLAP: if the answer straddles a chunk boundary, the overlap ensures
    one of the two chunks still contains the full sentence.

    Guarantees: no chunk is longer than `chunk_size`; empty input gives [].

    Raises:
        ValueError: if overlap >= chunk_size (the splitter would never advance).
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= chunk_size else _hard_split(p, chunk_size, overlap))

    if preserve_paragraphs:
        return [
            piece
            for paragraph in paragraphs
            for piece in (
                [paragraph]
                if len(paragraph) <= chunk_size
                else _hard_split(paragraph, chunk_size, overlap)
            )
        ]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1:]  # don't start the lead-in mid-word
        # Only use the lead-in if the result still respects chunk_size.
        current = f"{tail}\n\n{piece}" if tail and len(tail) + 2 + len(piece) <= chunk_size else piece

    if current:
        chunks.append(current)
    return chunks


def _to_chunks(doc: SourceDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """
    Convert one SourceDocument into Chunk objects with IDs and metadata.

    WHY `embed_text` DIFFERS FROM `text`: we embed "Title\\n\\npassage" but store
    only the passage. A chunk reading "Restart the client and re-enter your
    credentials." is ambiguous alone; with the title "Vpn Setup Mac" in front,
    the embedding knows what it's about and matches questions better. The
    employee and GPT-4o still see only the clean passage.
    """
    chunks = []
    pieces = split_text(
        doc.text,
        chunk_size,
        overlap,
        preserve_paragraphs=doc.source.lower().endswith(".json"),
    )
    for i, piece in enumerate(pieces):
        meta = {**doc.metadata(), "chunk_index": i}
        chunk_id = f"{doc.source}#{doc.page or 0}#{i}"
        chunks.append(
            Chunk(id=chunk_id, text=piece, embed_text=f"{doc.title}\n\n{piece}", metadata=meta)
        )
    return chunks


def ingest(
    raw_dir: str | Path = "data/raw",
    reset: bool = True,
    llm: LLMClient | None = None,
) -> IngestReport:
    """
    Run the full pipeline and persist the result to the vector store.

    Args:
        raw_dir: folder containing the source documents.
        reset: if True (default), rebuild the index from scratch. WHY DEFAULT
            TRUE: with upserts alone, a document that was edited to be shorter
            or deleted would leave stale chunks behind, and the bot could quote
            policy that no longer exists. For a knowledge base of this size a
            full rebuild takes seconds and is always correct.
        llm: optional LLMClient. WHY INJECTABLE: tests pass a fake so no OpenAI
            call or API key is needed.

    SAFETY ORDER: everything is loaded, chunked, and embedded in memory BEFORE
    the old index is touched. If loading finds nothing or the embedding API
    fails halfway, the existing working index is left intact instead of being
    wiped and replaced with nothing.

    Raises:
        FileNotFoundError: raw_dir doesn't exist.
        ValueError: no usable text found (refuses to wipe the index).
    """
    s = get_settings()
    docs = load_directory(raw_dir)

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(_to_chunks(doc, s.chunk_size, s.chunk_overlap))
    if not chunks:
        raise ValueError(f"No text could be extracted from {raw_dir}; index left unchanged.")

    llm = llm or LLMClient()
    vectors = llm.embed([c.embed_text for c in chunks])

    collection = get_collection(reset=reset)
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i : i + _BATCH]
        collection.upsert(  # upsert (not add) so reset=False re-runs don't hit duplicate IDs
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i : i + _BATCH],
            metadatas=[c.metadata for c in batch],
        )
    write_index_version()   # tells running servers to rebuild their BM25 index
    report = IngestReport(documents=len(docs), chunks=len(chunks))
    for c in chunks:
        dept = c.metadata["department"]
        report.by_department[dept] = report.by_department.get(dept, 0) + 1
    logger.info("Ingested %d chunks from %d documents", report.chunks, report.documents)
    return report