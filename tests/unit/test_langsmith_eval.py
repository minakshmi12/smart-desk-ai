"""
Tests for the LangSmith eval wiring (no network, no account, no real LLM).

WHY THIS FILE EXISTS
--------------------
Covers everything WE wrote: judge scoring, evaluator rules, dataset naming and
cleanup, aggregation, and the target/collected wiring. It does NOT cover the real
`langsmith.evaluate()`; the fake `evaluate` below encodes our ASSUMPTION about
its call and result shape, so run one real experiment to confirm.
"""

from types import SimpleNamespace

import pytest

from evals.judges import (
    ClaimCheck, FaithfulnessVerdict, RelevanceVerdict, faithfulness_score,
    relevance_score, unsupported_claims,
)
from evals.langsmith_eval import (
    JUDGE_FAILED, NOT_APPLICABLE, dataset_name_for, ensure_dataset, false_answer,
    keyword_hit, latency_seconds, make_judge_evaluators, over_escalation,
    run_langsmith_experiment, summarise, escalation_correct,
)
from evals.runner import TimedRetriever
from evals.schemas import QACase
from smartdesk.rag.chain import RAGResult

GOOD = {"answered": True, "answer": "Restart the client and open UDP 4500", "contexts": ["p"],
        "total_seconds": 1.5, "retrieval_seconds": 0.2, "error": None}
ESCALATED = {**GOOD, "answered": False, "answer": "", "contexts": []}
SHOULD_ANSWER = {"should_answer": True, "must_mention": ["restart", "4500"]}
SHOULD_ESCALATE = {"should_answer": False, "must_mention": []}
INPUTS = {"case_id": "a", "question": "q"}


# ------------------------------------------------------------------ judges
def test_faithfulness_is_the_share_of_supported_claims():
    """2 of 3 claims supported = 0.67; the unsupported one is listed; no claims = None."""
    v = FaithfulnessVerdict(claims=[ClaimCheck(claim="a", supported=True),
                                    ClaimCheck(claim="b", supported=True),
                                    ClaimCheck(claim="c", supported=False)])
    assert faithfulness_score(v) == pytest.approx(2 / 3)
    assert unsupported_claims(v) == ["c"]
    assert faithfulness_score(FaithfulnessVerdict(claims=[])) is None


def test_relevance_labels_map_to_scores():
    """The three labels score 1.0, 0.5 and 0.0."""
    scores = [relevance_score(RelevanceVerdict(verdict=v, reasoning="r"))
              for v in ("fully_addresses", "partially_addresses", "does_not_address")]
    assert scores == [1.0, 0.5, 0.0]


# -------------------------------------------------------------- evaluators
def test_escalation_evaluators_score_each_outcome_correctly():
    """
    correct/false-answer/over-escalation follow the confusion-matrix definitions.
    WHY IT MATTERS: the experiment's mean of `false_answer` IS the false-answer rate.
    """
    assert escalation_correct(INPUTS, GOOD, SHOULD_ANSWER)["score"] == 1.0
    assert escalation_correct(INPUTS, GOOD, SHOULD_ESCALATE)["score"] == 0.0
    assert false_answer(INPUTS, GOOD, SHOULD_ESCALATE)["score"] == 1.0
    assert false_answer(INPUTS, ESCALATED, SHOULD_ESCALATE)["score"] == 0.0
    assert over_escalation(INPUTS, ESCALATED, SHOULD_ANSWER)["score"] == 1.0
    assert false_answer(INPUTS, GOOD, SHOULD_ANSWER)["comment"].startswith(NOT_APPLICABLE)


def test_keyword_and_latency_evaluators():
    """All keywords must appear; latency passes through; errored cases are excluded."""
    assert keyword_hit(INPUTS, GOOD, SHOULD_ANSWER)["score"] == 1.0
    assert keyword_hit(INPUTS, {**GOOD, "answer": "Restart it"}, SHOULD_ANSWER)["score"] == 0.0
    assert latency_seconds(INPUTS, GOOD, SHOULD_ANSWER)["score"] == 1.5
    errored = {**GOOD, "error": "RuntimeError"}
    assert latency_seconds(INPUTS, errored, SHOULD_ANSWER)["score"] is None
    assert escalation_correct(INPUTS, errored, SHOULD_ANSWER)["score"] is None


class FakeJudge:
    """Returns canned verdicts by schema; can be told to fail."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def parse(self, messages, schema):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        if schema is FaithfulnessVerdict:
            return FaithfulnessVerdict(claims=[ClaimCheck(claim="x", supported=True)])
        return RelevanceVerdict(verdict="fully_addresses", reasoning="on topic")


def test_judges_score_only_answered_should_answer_cases_without_calling_the_llm_otherwise():
    """
    Escalations and should-escalate cases never reach the judge.
    WHY IT MATTERS: saves judge cost, and prevents scoring an "I don't know".
    """
    judge = FakeJudge()
    faith, rel = make_judge_evaluators(judge)
    assert faith(INPUTS, GOOD, SHOULD_ANSWER)["score"] == 1.0
    assert rel(INPUTS, GOOD, SHOULD_ANSWER)["score"] == 1.0
    assert faith(INPUTS, ESCALATED, SHOULD_ANSWER)["comment"].startswith(NOT_APPLICABLE)
    assert rel(INPUTS, GOOD, SHOULD_ESCALATE)["comment"].startswith(NOT_APPLICABLE)
    assert judge.calls == 2


def test_judge_failure_becomes_a_comment_not_an_exception():
    """
    A failing judge call yields score None + "judge failed", never a crash.
    WHY IT MATTERS: one flaky call must not abort a paid experiment.
    """
    faith, rel = make_judge_evaluators(FakeJudge(fail=True))
    for ev in (faith, rel):
        r = ev(INPUTS, GOOD, SHOULD_ANSWER)
        assert r["score"] is None and r["comment"].startswith(JUDGE_FAILED)


# ----------------------------------------------------------------- dataset
CASES = [QACase("a", "good question", True, must_mention=("restart",)),
         QACase("b", "unknown thing", False, category="near_miss")]


class FakeClient:
    """In-memory stand-in for langsmith.Client (only the methods we call)."""

    def __init__(self, fail_examples=False):
        self.datasets, self.deleted, self.examples = {}, [], []
        self.fail_examples = fail_examples

    def has_dataset(self, dataset_name):
        return dataset_name in self.datasets

    def create_dataset(self, dataset_name, description=""):
        self.datasets[dataset_name] = SimpleNamespace(id=f"id-{dataset_name}")
        return self.datasets[dataset_name]

    def create_examples(self, inputs, outputs, metadata, dataset_id):
        if self.fail_examples:
            raise RuntimeError("upload failed")
        self.examples = [SimpleNamespace(inputs=i, outputs=o, metadata=m)
                         for i, o, m in zip(inputs, outputs, metadata)]

    def delete_dataset(self, dataset_id):
        self.deleted.append(dataset_id)
        self.datasets = {k: v for k, v in self.datasets.items() if v.id != dataset_id}

    def list_examples(self, dataset_name):
        return list(self.examples)


def test_dataset_name_changes_with_content_and_is_created_only_once():
    """
    Same questions -> same name (no duplicates); an edited set -> a new dataset.
    WHY IT MATTERS: old experiments must keep pointing at the questions they ran on.
    """
    edited = [*CASES[:1], QACase("b", "different", False)]
    assert dataset_name_for(CASES) == dataset_name_for(list(CASES))
    assert dataset_name_for(CASES) != dataset_name_for(edited)

    client = FakeClient()
    name = ensure_dataset(client, CASES)
    client.examples.clear()
    assert ensure_dataset(client, CASES) == name and client.examples == []   # not re-uploaded


def test_failed_upload_deletes_the_half_built_dataset():
    """
    If examples can't be uploaded the empty dataset is removed and the error surfaces.
    WHY IT MATTERS: otherwise the next run would find an empty dataset and "pass".
    """
    client = FakeClient(fail_examples=True)
    with pytest.raises(RuntimeError):
        ensure_dataset(client, CASES)
    assert client.deleted and client.datasets == {}


# -------------------------------------------------------------- experiment
class _Results:
    """Iterable result object with an experiment name, like ExperimentResults."""

    experiment_name = "smartdesk-abc123"

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class _Chain:
    """Answers "good question"; escalates anything else."""

    def answer(self, question, history=None, department=None):
        if question == "good question":
            return RAGResult(True, "Restart the client and open UDP 4500", question, contexts=["p"])
        return RAGResult(False, "I don't know", question)


class _Inner:
    def search(self, query, department=None):
        return []


def test_experiment_runs_target_collects_outcomes_and_summarises_judges():
    """
    Our wiring end to end with a fake `evaluate`: the target runs each case, outcomes
    are collected, metadata is forwarded, and judge feedback is averaged per case.
    """
    seen = {}

    def fake_evaluate(target, data, evaluators, **kw):
        seen.update(kw)
        rows = []
        for ex in data:
            out = target(ex.inputs)
            res = [ev(inputs=ex.inputs, outputs=out, reference_outputs=ex.outputs)
                   for ev in evaluators]
            rows.append({"example": ex, "evaluation_results": {
                "results": [SimpleNamespace(**r) for r in res]}})
        return _Results(rows)

    client, collected, retriever = FakeClient(), {}, TimedRetriever(_Inner())
    scores = run_langsmith_experiment(
        chain=_Chain(), retriever=retriever, cases=CASES, judge=FakeJudge(), collected=collected,
        metadata={"confidence_threshold": 0.6}, client=client, evaluate_fn=fake_evaluate,
    )
    assert seen["max_concurrency"] == 1 and seen["metadata"] == {"confidence_threshold": 0.6}
    assert collected["a"].answered and not collected["b"].answered
    assert scores.faithfulness == 1.0 and scores.answer_relevancy == 1.0
    assert scores.evaluated == 1 and scores.failed == 0       # only case "a" is judgeable
    assert scores.experiment_name == "smartdesk-abc123"


def test_limit_runs_only_the_requested_cases_against_the_full_dataset():
    """
    `cases` can be a slice while the dataset is built from `all_cases`.
    WHY IT MATTERS: --limit must not create a new dataset per limit value.
    """
    seen = []

    def fake_evaluate(target, data, evaluators, **kw):
        seen.extend(e.inputs["case_id"] for e in data)
        return _Results([])

    client = FakeClient()
    run_langsmith_experiment(
        chain=_Chain(), retriever=TimedRetriever(_Inner()), cases=CASES[:1], all_cases=CASES,
        judge=FakeJudge(), collected={}, client=client, evaluate_fn=fake_evaluate,
    )
    assert seen == ["a"] and len(client.datasets) == 1
    assert len(client.examples) == 2


def test_summarise_counts_failures_separately_from_not_applicable():
    """Judge errors count as failed; "not applicable" rows are ignored entirely."""
    def row(case_id, key, score, comment):
        return {"example": SimpleNamespace(inputs={"case_id": case_id}),
                "evaluation_results": {"results": [SimpleNamespace(key=key, score=score, comment=comment)]}}

    s = summarise([row("a", "faithfulness", 0.5, "x"),
                   row("b", "faithfulness", None, f"{JUDGE_FAILED}: RuntimeError"),
                   row("c", "faithfulness", None, f"{NOT_APPLICABLE}: escalated"),
                   row("d", "latency_seconds", 2.0, None)])
    assert s.faithfulness == 0.5 and s.evaluated == 2 and s.failed == 1