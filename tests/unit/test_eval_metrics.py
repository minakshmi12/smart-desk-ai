"""
Tests for the eval suite's own logic (no network, no LLM, no Ragas).

WHY THIS FILE EXISTS
--------------------
If the metrics are wrong, every decision made from them is wrong, and nobody
notices because the numbers still look plausible. The Ragas wrapper itself is
NOT tested here (it needs a live judge); everything around it is.
"""

import pytest

from evals.metrics import (
    GateRecord, check_gates, escalation_by_category, escalation_report,
    keyword_hit_rate, latency_summary, percentile, recommend_threshold,
    routing_report, sweep,
)
from evals.runner import TimedRetriever, run_qa, run_routing
from evals.schemas import QACase, RoutingCase, load_qa_cases
from smartdesk.rag.chain import RAGResult


# ---------------------------------------------------------------- latency
def test_percentile_interpolates_and_handles_edges():
    """p50 of 1..4 is 2.5; single values and empty input behave; bad p raises."""
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 0) == 1 and percentile([4, 1, 3, 2], 100) == 4
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_latency_summary_for_data_and_for_nothing():
    """Empty input gives None values (not zeros), so "no data" is never read as "instant"."""
    assert latency_summary([1.0, 3.0])["mean"] == 2.0
    assert latency_summary([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}


# ------------------------------------------------------------- escalation
def test_escalation_counts_and_rates():
    """
    Each (should, did) pair lands in the right cell; rates use the right denominators.
    WHY IT MATTERS: false-answer rate must be over should-ESCALATE cases only.
    """
    r = escalation_report([(True, True), (True, False), (False, False), (False, False), (False, True)])
    assert (r.answered_correctly, r.over_escalated, r.escalated_correctly, r.false_answered) == (1, 1, 2, 1)
    assert r.false_answer_rate == pytest.approx(1 / 3)
    assert r.over_escalation_rate == 0.5
    assert r.accuracy == 0.6


def test_rates_are_none_without_data():
    """No should-escalate cases means the false-answer rate is None, not 0%."""
    r = escalation_report([(True, True)])
    assert r.false_answer_rate is None and escalation_report([]).accuracy is None


def test_breakdown_by_category():
    """A weak category stays visible instead of being averaged away."""
    by = escalation_by_category([("near_miss", False, True), ("near_miss", False, False),
                                 ("answerable", True, True)])
    assert by["near_miss"].false_answer_rate == 0.5 and by["answerable"].accuracy == 1.0


def test_keyword_hit_rate_requires_all_keywords_case_insensitively():
    """An answer missing any required keyword counts as a miss."""
    assert keyword_hit_rate([("Restart the client; open UDP 4500", ["restart", "4500"]),
                             ("Restart it", ["restart", "4500"])]) == 0.5
    assert keyword_hit_rate([]) is None


# ----------------------------------------------------------------- sweep
RECORDS = [
    GateRecord(should_answer=True, llm_answered=True, best_distance=0.3),    # good match
    GateRecord(should_answer=False, llm_answered=True, best_distance=0.7),   # LLM wrongly answers a near miss
    GateRecord(should_answer=False, llm_answered=False, best_distance=0.9),  # LLM declines
    GateRecord(should_answer=False, llm_answered=True, best_distance=None),  # nothing retrieved
]


def test_sweep_tightens_and_loosens_the_gate():
    """
    Low threshold: good case over-escalates. Middle: all correct. High: the near
    miss slips through as a false answer. A None distance is never answered.
    """
    low, mid, high = sweep(RECORDS, [0.2, 0.5, 0.8])
    assert low.report.over_escalation_rate == 1.0 and low.report.false_answer_rate == 0
    assert mid.report.accuracy == 1.0
    assert high.report.false_answered == 1


def test_recommend_threshold_is_largest_safe_value():
    """Picks 0.5: the largest threshold with zero false answers."""
    assert recommend_threshold(sweep(RECORDS, [0.2, 0.5, 0.8])) == 0.5
    assert recommend_threshold(sweep(RECORDS, [0.8])) is None      # none qualifies
    assert recommend_threshold([]) is None


# --------------------------------------------------------------- routing
def test_routing_report_lists_mismatches():
    """Accuracy plus the exact (id, expected, actual) of every miss."""
    r = routing_report([("a", "kb_answer", "kb_answer"), ("b", "chitchat", "kb_answer")])
    assert r.accuracy == 0.5 and r.mismatches == [("b", "chitchat", "kb_answer")]


# ----------------------------------------------------------------- gates
def test_gates_pass_fail_and_fail_closed():
    """
    Violations fail; a set limit on an unmeasured metric also fails; no limits means no failures.
    WHY IT MATTERS: a CI gate that passes when Ragas silently didn't run is worthless.
    """
    base = dict(false_answer_rate=0.0, faithfulness=0.9, relevancy=0.9, p95_seconds=3.0)
    assert check_gates(**base) == []
    assert check_gates(**base, min_faithfulness=0.95)[0].startswith("faithfulness")
    assert check_gates(**{**base, "false_answer_rate": 0.2}, max_false_answer_rate=0.0)
    assert "not measured" in check_gates(**{**base, "faithfulness": None}, min_faithfulness=0.8)[0]


# ----------------------------------------------------------------- loading
def test_loader_validates_input(tmp_path):
    """Valid files load; bad JSON reports its line; missing keys and duplicate ids are rejected."""
    good = tmp_path / "q.jsonl"
    good.write_text('{"id":"a","question":"q","should_answer":true,"must_mention":["x"]}\n\n')
    assert load_qa_cases(good)[0].must_mention == ("x",)

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a"\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_qa_cases(bad)

    missing = tmp_path / "m.jsonl"
    missing.write_text('{"id":"a","question":"q"}\n')
    with pytest.raises(ValueError, match="should_answer"):
        load_qa_cases(missing)

    dup = tmp_path / "d.jsonl"
    dup.write_text('{"id":"a","question":"q","should_answer":true}\n' * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_qa_cases(dup)


# ------------------------------------------------------------------ runner
class _Inner:
    """Retriever stub that returns nothing."""

    def search(self, query, department=None):
        return []


class _Chain:
    """Chain stub that searches, then answers or raises depending on the question."""

    def __init__(self, retriever):
        self.retriever = retriever

    def answer(self, question, history=None, department=None):
        self.retriever.search(question)
        if "boom" in question:
            raise RuntimeError("private detail")
        return RAGResult(answerable=True, answer="ok", standalone_question=question, contexts=["c"])


def test_run_qa_times_cases_and_records_errors_without_leaking_messages():
    """
    A failing case is recorded by exception class only, and later cases still run.
    WHY IT MATTERS: one outage mustn't abort a paid run, and error text could contain secrets.
    """
    retriever = TimedRetriever(_Inner())
    cases = [QACase("a", "fine", True), QACase("b", "boom", True)]
    first, second = run_qa(_Chain(retriever), retriever, cases)

    assert first.answered and first.error is None
    assert 0 <= first.retrieval_seconds <= first.total_seconds
    assert second.error == "RuntimeError" and not second.answered and second.result is None


def test_run_routing_records_actual_route_and_errors():
    """The router's `route` is captured; an exception becomes an error, not a crash."""
    def router(state):
        if "boom" in state["messages"][0]["content"]:
            raise RuntimeError("x")
        return {"route": "kb_answer"}

    ok, bad = run_routing(router, [RoutingCase("1", "hi", "chitchat"), RoutingCase("2", "boom", "x")])
    assert ok.actual == "kb_answer" and bad.actual is None and bad.error == "RuntimeError"