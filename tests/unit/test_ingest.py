"""
Unit tests for loaders and the ingestion pipeline (no network, no OpenAI key).

WHY THIS FILE EXISTS
--------------------
Retrieval can only be as good as the chunks it searches. These tests lock in the
properties the rest of the system relies on: chunks respect size limits, overlap
really overlaps, bad files don't crash ingestion, and a failed ingest never
destroys a working index.

RUN WITH: pytest tests/unit/test_ingest.py -v
"""

import pytest
from docx import Document as DocxDocument

from smartdesk.config import get_settings
from smartdesk.rag.ingest import _to_chunks, ingest, split_text
from smartdesk.rag.loaders import SourceDocument, load_directory
from smartdesk.rag.vector_store import get_collection


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """
    Give each test its own fake settings and its own empty vector store folder.

    WHY: Chroma persists to disk. Without a per-test temp directory, tests would
    see each other's data (and your real index). The settings cache is cleared
    before and after so values never leak between tests.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "store"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLM:
    """
    Stand-in for LLMClient that returns deterministic vectors without any API call.

    WHY IT EXISTS: ingest() only needs `.embed(list[str])`. Vectors are derived
    from text length so they're non-zero (cosine distance is undefined for a
    zero vector) and the same text always gets the same vector.
    """

    def embed(self, texts):
        return [[1.0, len(t) / 100 + 0.1] for t in texts]


def _write(root, rel, content):
    """Helper: create a text file (and its folders) under `root`."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------- split_text
def test_short_text_is_a_single_chunk():
    """
    Text smaller than chunk_size is returned unchanged as one chunk.
    WHY IT MATTERS: short FAQ entries must not be fragmented.
    """
    assert split_text("Reset your password at the portal.", 800, 100) == [
        "Reset your password at the portal."
    ]


def test_empty_text_gives_no_chunks():
    """
    Empty or whitespace-only input yields [].
    WHY IT MATTERS: avoids embedding (and paying for) empty strings.
    """
    assert split_text("  \n\n  ", 800, 100) == []


def test_no_chunk_exceeds_chunk_size():
    """
    Every chunk is <= chunk_size, for both normal paragraphs and one giant paragraph.

    WHY IT MATTERS: the size limit is the contract that keeps prompts and
    embeddings within budget. The test mixes a long unbroken paragraph with
    short ones to exercise the hard-split path and the packing path.
    """
    text = "Short intro.\n\n" + ("word " * 400) + "\n\nShort outro."
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)


def test_hard_split_windows_overlap():
    """
    Consecutive windows of an oversized paragraph share exactly `overlap` characters.

    WHY IT MATTERS: overlap is what stops an answer from being cut in half at
    a chunk boundary. This verifies the first window's tail equals the second
    window's head.
    """
    chunks = split_text("word " * 500, chunk_size=200, overlap=50)
    assert chunks[0][-50:] == chunks[1][:50]


def test_paragraphs_are_packed_together_when_they_fit():
    """
    Small paragraphs share a chunk instead of one chunk each.

    WHY IT MATTERS: tiny one-line chunks match poorly and waste retrieval slots.
    """
    chunks = split_text("one\n\ntwo\n\nthree", chunk_size=100, overlap=10)
    assert chunks == ["one\n\ntwo\n\nthree"]


def test_json_paragraphs_can_be_kept_in_separate_chunks():
    """JSON policy records can be chunked independently for more precise retrieval."""
    doc = SourceDocument(
        text="category: Casual Leave | rule: Annual allowance is 12 days.\n\n"
        "category: Sick Leave | rule: Annual allowance is 8 days.",
        source="hr/policy.json",
        department="hr",
        title="Policy",
    )

    chunks = _to_chunks(doc, chunk_size=800, overlap=100)
    assert len(chunks) == 2
    assert "12 days" in chunks[0].text
    assert "8 days" in chunks[1].text


def test_invalid_overlap_is_rejected():
    """
    overlap >= chunk_size raises ValueError.

    WHY IT MATTERS: such a configuration would make the splitter loop forever or
    never advance; a clear error at startup beats a hang.
    """
    with pytest.raises(ValueError):
        split_text("anything", chunk_size=100, overlap=100)


# ------------------------------------------------------------------ loaders
def test_department_comes_from_folder_and_source_is_relative(tmp_path):
    """
    A file in it/ is tagged department "it" with a root-relative source path;
    a file outside known folders is tagged "general".

    WHY IT MATTERS: department drives retrieval filtering, and the relative
    source is what answers cite. A misplaced file should still be searchable.
    """
    raw = tmp_path / "raw"
    _write(raw, "it/vpn_setup.md", "# VPN\nInstall the client.")
    _write(raw, "misc/notes.txt", "Some notes.")
    docs = {d.source: d for d in load_directory(raw)}
    assert docs["it/vpn_setup.md"].department == "it"
    assert docs["it/vpn_setup.md"].title == "Vpn Setup"
    assert docs["misc/notes.txt"].department == "general"


def test_json_files_are_loaded_as_readable_text(tmp_path):
    """JSON documents are parsed and retain their content and source metadata."""
    raw = tmp_path / "raw"
    _write(
        raw,
        "hr/policy.json",
        '{"question":"How much leave?","answer":"20 days annually."}',
    )

    doc = load_directory(raw)[0]
    assert doc.source == "hr/policy.json"
    assert doc.department == "hr"
    assert "question: How much leave?" in doc.text
    assert "answer: 20 days annually." in doc.text


def test_json_keeps_policy_record_fields_together(tmp_path):
    """Structured policy records become compact searchable prose."""
    raw = tmp_path / "raw"
    _write(
        raw,
        "hr/leave.json",
        '{"leave_policy_entries":[{"id":"LP-001","category":"Casual Leave",'
        '"rule":"Annual casual leave allowance is fixed at 12 days per calendar year."}]}',
    )

    doc = load_directory(raw)[0]
    assert "leave policy entries" in doc.text
    assert "category: Casual Leave" in doc.text
    assert "rule: Annual casual leave allowance is fixed at 12 days per calendar year." in doc.text
    assert "{\n" not in doc.text


def test_unsupported_empty_and_hidden_files_are_skipped(tmp_path):
    """
    .exe, empty files, and dotfiles produce no documents and no exception.

    WHY IT MATTERS: real document folders contain junk (.DS_Store, installers,
    blank drafts). Ingestion must be tolerant.
    """
    raw = tmp_path / "raw"
    _write(raw, "it/tool.exe", "binary")
    _write(raw, "it/empty.md", "   ")
    _write(raw, "it/.DS_Store", "x")
    _write(raw, "it/real.md", "Real content.")
    assert [d.source for d in load_directory(raw)] == ["it/real.md"]


def test_docx_headings_and_tables_are_extracted(tmp_path):
    """
    Word headings become "# " lines and table rows become "a | b" lines.

    WHY IT MATTERS: HR policies are often Word files with leave-entitlement
    tables. If tables were dropped, the bot would be unable to answer exactly
    the questions employees ask most.
    """
    raw = tmp_path / "raw"
    (raw / "hr").mkdir(parents=True)
    d = DocxDocument()
    d.add_heading("Leave Policy", level=1)
    d.add_paragraph("Employees accrue leave monthly.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Annual leave"
    t.rows[0].cells[1].text = "20 days"
    d.save(raw / "hr" / "leave.docx")

    text = load_directory(raw)[0].text
    assert "# Leave Policy" in text
    assert "Annual leave | 20 days" in text


def test_missing_directory_raises(tmp_path):
    """
    A wrong raw-data path is a loud error, not "0 documents".
    WHY IT MATTERS: a typo in the path should never look like a successful run.
    """
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "nope")


# ------------------------------------------------------------------- ingest
def test_ingest_stores_chunks_with_metadata(tmp_path):
    """
    End-to-end: files in, chunks with correct text/metadata stored in Chroma.

    WHAT IT CHECKS: the report counts match, stored chunks carry department,
    source, and title, and the stored text is the clean passage (without the
    title prefix that is only used for embedding).

    WHY IT MATTERS: the retriever and the answer prompt depend on this exact
    shape; this is the contract between ingestion and retrieval.
    """
    raw = tmp_path / "raw"
    _write(raw, "it/vpn.md", "Install the VPN client.\n\nRestart on error 809.")
    _write(raw, "hr/leave.md", "Employees get 20 days of annual leave.")

    report = ingest(raw, llm=FakeLLM())
    assert report.documents == 2
    assert report.by_department == {"it": 1, "hr": 1}

    stored = get_collection().get(include=["documents", "metadatas"])
    assert len(stored["ids"]) == report.chunks == 2
    hr = next(m for m in stored["metadatas"] if m["department"] == "hr")
    assert hr["source"] == "hr/leave.md" and hr["title"] == "Leave"
    assert "Employees get 20 days of annual leave." in stored["documents"]


def test_reingest_with_reset_removes_stale_chunks(tmp_path):
    """
    After deleting a document and re-ingesting, its chunks are gone.

    WHY IT MATTERS: this is the "bot quotes a policy that no longer exists"
    bug. Full-rebuild-by-default is what prevents it.
    """
    raw = tmp_path / "raw"
    old = _write(raw, "hr/old_policy.md", "Old policy text.")
    _write(raw, "hr/new_policy.md", "New policy text.")
    ingest(raw, llm=FakeLLM())

    old.unlink()
    ingest(raw, llm=FakeLLM())
    assert get_collection().count() == 1


def test_empty_folder_does_not_wipe_existing_index(tmp_path):
    """
    Ingesting a folder with no usable text raises and leaves the old index intact.

    WHY IT MATTERS: pointing the script at the wrong (empty) folder must not
    destroy a working knowledge base. This is the safety-order guarantee.
    """
    good = tmp_path / "good"
    _write(good, "it/vpn.md", "Install the VPN client.")
    ingest(good, llm=FakeLLM())

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ingest(empty, llm=FakeLLM())
    assert get_collection().count() == 1