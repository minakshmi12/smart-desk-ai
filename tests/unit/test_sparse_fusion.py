"""BM25 and RRF: pure logic, no network."""

from smartdesk.rag.fusion import reciprocal_rank_fusion
from smartdesk.rag.sparse import BM25Index, SparseDoc, tokenize


def _doc(id_, text, dept="it"):
    """Helper: an indexable chunk."""
    return SparseDoc(id_, text, {"department": dept, "title": ""})


def test_tokenize_keeps_codes_and_drops_stopwords_and_plurals():
    """
    "809" survives, stopwords go, "Macs" -> "mac".
    WHY IT MATTERS: exact tokens like error codes are the reason BM25 exists.
    """
    assert tokenize("How do I fix VPN error 809 on my Macs?") == ["fix", "vpn", "error", "809", "mac"]


def test_exact_token_ranks_first_and_unrelated_chunks_are_not_returned():
    """
    Only the chunk containing "809" matches "error 809".
    WHY IT MATTERS: this is the exact-match behaviour dense search lacks.
    """
    idx = BM25Index([_doc("a", "Install the client from the software portal"),
                     _doc("b", "If you see error 809 restart the client"),
                     _doc("c", "Request VPN access through your manager")])
    hits = idx.search("error 809", 3)
    assert [h.doc.id for h in hits] == ["b"] and hits[0].score > 0


def test_rare_term_outranks_common_term_and_scores_are_never_negative():
    """
    "809" (rare) beats "client" (in every chunk); a term in ALL chunks still scores > 0.
    WHY IT MATTERS: the Lucene IDF avoids negative scores on small corpora.
    """
    idx = BM25Index([_doc("a", "client setup"), _doc("b", "client error 809"), _doc("c", "client portal")])
    assert idx.search("client 809", 3)[0].doc.id == "b"
    assert all(h.score > 0 for h in idx.search("client", 3))


def test_department_filter_and_empty_queries():
    """A department restricts results; stopword-only or empty queries return nothing."""
    idx = BM25Index([_doc("a", "leave policy", "hr"), _doc("b", "leave of absence form", "it")])
    assert [h.doc.id for h in idx.search("leave", 5, department="hr")] == ["a"]
    assert idx.search("how do I", 5) == [] and idx.search("", 5) == []


def test_rrf_rewards_agreement_and_breaks_ties_deterministically():
    """
    An item both retrievers rank well beats one ranked first by only one; ties are stable;
    a repeated id counts once.
    """
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    assert [i for i, _ in fused] == ["a", "c", "b"]
    assert [i for i, _ in reciprocal_rank_fusion([["x"], ["y"]])] == ["x", "y"]
    assert reciprocal_rank_fusion([["a", "a"]])[0][1] == 1 / 61