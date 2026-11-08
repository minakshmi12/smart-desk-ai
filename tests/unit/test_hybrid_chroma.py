"""Hybrid retrieval over a REAL Chroma store (no network: embeddings are faked)."""

import pytest

from smartdesk.config import get_settings
from smartdesk.rag.hybrid import HybridRetriever
from smartdesk.rag.ingest import ingest


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Temp vector store, small candidate pool, dummy key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("CANDIDATE_POOL", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLM:
    """Embeddings depend only on text length, so dense ranking is meaningless on purpose."""

    def embed(self, texts):
        return [[1.0, len(t) / 100 + 0.1] for t in texts]


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_keyword_chunk_is_found_with_a_real_distance_and_new_content_is_picked_up(tmp_path):
    """
    With a pool of 1 dense hit, the "809" chunk is reached via BM25 and gets a distance
    computed from Chroma's stored embedding. After an edit that keeps the chunk COUNT
    the same, the SAME retriever instance finds the new text.
    WHY IT MATTERS: exercises the real Chroma `get(include=["embeddings"])` shape and the
    ingest version stamp that invalidates the BM25 index.
    """
    raw = tmp_path / "raw"
    _write(raw, "it/vpn.md", "Install the VPN client from the software portal.")
    _write(raw, "it/errors.md", "If you see error 809 restart the client and open UDP 4500.")
    _write(raw, "hr/leave.md", "Employees get 20 days of annual leave.")
    ingest(raw, llm=FakeLLM())

    r = HybridRetriever(llm=FakeLLM(), mode="hybrid")
    hit = next(c for c in r.search("error 809") if "809" in c.text)
    assert 0 <= hit.distance <= 2 and hit.sparse_score > 0

    _write(raw, "hr/leave.md", "Employees get bravo days of annual leave.")   # same chunk count
    ingest(raw, llm=FakeLLM())
    assert any("bravo" in c.text for c in r.search("bravo"))