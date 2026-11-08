"""
Runs the QA set as a LangSmith experiment: dataset, target, evaluators, summary.

WHY THIS FILE EXISTS
--------------------
Isolates everything that touches LangSmith (the one dependency here that needs an
account, uploads data, and changes API between versions). It is imported lazily
by run_evals, so the rest of the suite works without it.

HOW IT FITS TOGETHER
--------------------
1. ensure_dataset   uploads the questions once, as a dataset named by content hash.
2. target           runs the real RAG chain on one question (timed) and returns a
                    plain dict. It also stashes the full QAOutcome in `collected`
                    so the existing local metrics keep working.
3. evaluators       score each result: code metrics + two LLM judges. Each score
                    appears in the LangSmith UI as feedback on that run.
4. summarise        averages the judge scores for the local report.

PRIVACY: questions, answers AND the retrieved passages (your internal documents)
are sent to LangSmith. Use only with an account/region your company approves, or
a self-hosted LangSmith.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from evals.judges import (
    FaithfulnessVerdict, RelevanceVerdict, build_faithfulness_messages,
    build_relevance_messages, faithfulness_score, relevance_score, unsupported_claims,
)
from evals.runner import QAOutcome, TimedRetriever, run_one
from evals.schemas import QACase

logger = logging.getLogger(__name__)

DATASET_PREFIX = "smartdesk-qa"
NOT_APPLICABLE = "not applicable"   # comment prefix: the metric doesn't apply to this case
JUDGE_FAILED = "judge failed"       # comment prefix: the judge call errored


# ------------------------------------------------------------------ dataset
def dataset_name_for(cases: list[QACase]) -> str:
    """
    Name a dataset after a hash of its content, e.g. "smartdesk-qa-3fa81c20b7".

    WHY A HASH: an edited question set becomes a NEW dataset, so old experiments
    keep pointing at exactly the questions they were run on, and there is no
    fragile "sync existing examples" logic. Cost: old datasets accumulate; delete
    them in the UI when you no longer need them.
    """
    payload = json.dumps([asdict(c) for c in cases], sort_keys=True)
    return f"{DATASET_PREFIX}-{hashlib.sha256(payload.encode()).hexdigest()[:10]}"


def ensure_dataset(client, cases: list[QACase]) -> str:
    """
    Create the dataset if it doesn't exist yet; return its name.

    Inputs hold what the target needs (id, question, department); reference
    outputs hold the ground truth (should_answer, must_mention) that evaluators
    compare against. If uploading examples fails, the half-built dataset is
    deleted, so the next run doesn't find an empty dataset and "succeed" on nothing.
    """
    name = dataset_name_for(cases)
    if client.has_dataset(dataset_name=name):
        return name
    dataset = client.create_dataset(
        dataset_name=name, description="SmartDesk knowledge-base questions with expected behaviour"
    )
    try:
        client.create_examples(
            inputs=[{"case_id": c.id, "question": c.question, "department": c.department}
                    for c in cases],
            outputs=[{"should_answer": c.should_answer, "must_mention": list(c.must_mention)}
                     for c in cases],
            metadata=[{"category": c.category} for c in cases],
            dataset_id=dataset.id,
        )
    except Exception:
        client.delete_dataset(dataset_id=dataset.id)
        raise
    return name


# --------------------------------------------------------------- evaluators
def _result(key: str, score: float | None, comment: str | None = None) -> dict:
    """Build the dict LangSmith expects from an evaluator: {"key", "score", "comment"}."""
    out: dict[str, Any] = {"key": key, "score": score}
    if comment:
        out["comment"] = comment
    return out


def _not_applicable(key: str, why: str) -> dict:
    """A None score with a recognisable comment, so summaries can tell 'n/a' from 'failed'."""
    return _result(key, None, f"{NOT_APPLICABLE}: {why}")


def escalation_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """1.0 if the agent answered exactly when it should have, else 0.0."""
    if outputs.get("error"):
        return _not_applicable("escalation_correct", "case errored")
    return _result("escalation_correct",
                   float(bool(outputs["answered"]) == bool(reference_outputs["should_answer"])))


def false_answer(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    On should-ESCALATE cases: 1.0 if the agent answered anyway (the worst failure).
    The mean over the experiment is the false-answer rate; lower is better.
    """
    if outputs.get("error") or reference_outputs["should_answer"]:
        return _not_applicable("false_answer", "only for questions that should escalate")
    return _result("false_answer", 1.0 if outputs["answered"] else 0.0)


def over_escalation(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """On should-ANSWER cases: 1.0 if the agent escalated instead (a needless ticket)."""
    if outputs.get("error") or not reference_outputs["should_answer"]:
        return _not_applicable("over_escalation", "only for questions that should be answered")
    return _result("over_escalation", 0.0 if outputs["answered"] else 1.0)


def keyword_hit(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """1.0 if the answer contains EVERY required keyword (case-insensitive)."""
    keywords = reference_outputs.get("must_mention") or []
    if (outputs.get("error") or not outputs["answered"]
            or not reference_outputs["should_answer"] or not keywords):
        return _not_applicable("keyword_hit", "needs an answered question with must_mention")
    text = outputs["answer"].lower()
    return _result("keyword_hit", float(all(k.lower() in text for k in keywords)))


def latency_seconds(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Total seconds spent in the RAG chain (embedding, search, GPT-4o)."""
    if outputs.get("error"):
        return _not_applicable("latency_seconds", "case errored")
    return _result("latency_seconds", float(outputs["total_seconds"]))


def retrieval_seconds(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Seconds spent in the vector search alone."""
    if outputs.get("error"):
        return _not_applicable("retrieval_seconds", "case errored")
    return _result("retrieval_seconds", float(outputs["retrieval_seconds"]))


CODE_EVALUATORS = [escalation_correct, false_answer, over_escalation, keyword_hit,
                   latency_seconds, retrieval_seconds]


def _judgeable(outputs: dict, reference_outputs: dict) -> bool:
    """Judges only score answered, should-answer cases that have passages to check against."""
    return bool(outputs.get("answered") and not outputs.get("error")
                and reference_outputs.get("should_answer") and outputs.get("contexts"))


def make_judge_evaluators(judge) -> list[Callable]:
    """
    Build the two LLM-judge evaluators around `judge`.

    `judge` is anything with parse(messages, schema), normally LLMClient(model=...).
    A judge failure becomes a None score with a "judge failed" comment instead of
    an exception: one flaky call must not abort a paid experiment.
    """

    def faithfulness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """Share of the answer's claims supported by the passages shown to the model."""
        if not _judgeable(outputs, reference_outputs):
            return _not_applicable("faithfulness", "needs an answered should-answer case")
        try:
            verdict = judge.parse(
                build_faithfulness_messages(inputs["question"], outputs["answer"], outputs["contexts"]),
                FaithfulnessVerdict)
        except Exception as exc:
            logger.exception("Faithfulness judge failed")
            return _result("faithfulness", None, f"{JUDGE_FAILED}: {type(exc).__name__}")
        bad = unsupported_claims(verdict)
        return _result("faithfulness", faithfulness_score(verdict),
                       ("unsupported: " + "; ".join(bad)) if bad else "all claims supported")

    def answer_relevancy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """Does the answer address the question (1 / 0.5 / 0)?"""
        if not _judgeable(outputs, reference_outputs):
            return _not_applicable("answer_relevancy", "needs an answered should-answer case")
        try:
            verdict = judge.parse(
                build_relevance_messages(inputs["question"], outputs["answer"]), RelevanceVerdict)
        except Exception as exc:
            logger.exception("Relevance judge failed")
            return _result("answer_relevancy", None, f"{JUDGE_FAILED}: {type(exc).__name__}")
        return _result("answer_relevancy", relevance_score(verdict), verdict.reasoning)

    return [faithfulness, answer_relevancy]


# ------------------------------------------------------------------ summary
@dataclass(frozen=True)
class JudgeScores:
    """
    Judge results for the local report (same shape the old Ragas wrapper returned).

    faithfulness / answer_relevancy: means over scored cases (0-1), None if none.
    evaluated: cases a judge was applicable to.
    failed:    applicable cases that got no score (judge error, or no claims found).
               A high number makes the means weak evidence.
    per_case:  case id -> {"faithfulness": x, "answer_relevancy": y}.
    experiment_name: look it up under the dataset's Experiments tab in LangSmith.
    """

    faithfulness: float | None
    answer_relevancy: float | None
    evaluated: int
    failed: int
    per_case: dict[str, dict[str, float | None]] = field(default_factory=dict)
    experiment_name: str | None = None


def _mean(values: list[float]) -> float | None:
    """Mean, or None for an empty list."""
    return sum(values) / len(values) if values else None


def summarise(rows: Iterable[dict], experiment_name: str | None = None) -> JudgeScores:
    """
    Aggregate judge feedback from the rows `evaluate()` yields.

    Each row is expected to look like {"example": <has .inputs>, "evaluation_results":
    {"results": [<has .key, .score, .comment>]}}. That is the shape LangSmith
    documents for ExperimentResults; if your version differs, this is the one
    function to adjust.
    """
    values: dict[str, list[float]] = {"faithfulness": [], "answer_relevancy": []}
    per_case: dict[str, dict[str, float | None]] = {}
    evaluated, failed = set(), 0
    for row in rows:
        case_id = row["example"].inputs["case_id"]
        for r in row["evaluation_results"]["results"]:
            if r.key not in values or (getattr(r, "comment", None) or "").startswith(NOT_APPLICABLE):
                continue
            evaluated.add(case_id)
            per_case.setdefault(case_id, {})[r.key] = r.score
            if r.score is None:
                failed += 1
            else:
                values[r.key].append(float(r.score))
    return JudgeScores(
        faithfulness=_mean(values["faithfulness"]),
        answer_relevancy=_mean(values["answer_relevancy"]),
        evaluated=len(evaluated), failed=failed, per_case=per_case,
        experiment_name=experiment_name,
    )


# -------------------------------------------------------------- experiment
def run_langsmith_experiment(
    *,
    chain,
    retriever: TimedRetriever,
    cases: list[QACase],
    judge,
    collected: dict[str, QAOutcome],
    all_cases: list[QACase] | None = None,
    experiment_prefix: str = "smartdesk",
    metadata: dict | None = None,
    on_case: Callable[[QAOutcome], None] | None = None,
    client=None,
    evaluate_fn: Callable | None = None,
) -> JudgeScores:
    """
    Run `cases` through the chain as a LangSmith experiment and return judge scores.

    Args:
        collected: filled with the full QAOutcome per case id AS CASES RUN. The
            caller owns it, so if LangSmith fails halfway the finished cases are
            still available for the local metrics.
        all_cases: the full question set used to name/create the dataset, so a
            --limit run reuses the same dataset instead of creating one per limit.
        metadata: stored on the experiment (threshold, models...) so you can
            compare experiments in the UI by configuration.
        client / evaluate_fn: injectable for tests; default to the real ones.

    max_concurrency=1 keeps the target single-threaded because TimedRetriever
    keeps its timing in one shared variable. Judge calls may still overlap the
    next case slightly, which can inflate latency a little; for clean latency
    numbers run with --skip-judge.
    """
    if client is None:
        from langsmith import Client
        client = Client()
    if evaluate_fn is None:
        from langsmith import evaluate as evaluate_fn

    dataset_name = ensure_dataset(client, all_cases or cases)
    order = {c.id: i for i, c in enumerate(cases)}
    examples = sorted(
        (e for e in client.list_examples(dataset_name=dataset_name)
         if e.inputs["case_id"] in order),
        key=lambda e: order[e.inputs["case_id"]],
    )
    by_id = {c.id: c for c in cases}

    def target(inputs: dict) -> dict:
        """Run one question through the real chain; return a JSON-friendly summary."""
        outcome = run_one(chain, retriever, by_id[inputs["case_id"]])
        collected[outcome.case.id] = outcome
        if on_case:
            on_case(outcome)
        r = outcome.result
        return {
            "answered": outcome.answered,
            "answer": r.answer if r else "",
            "reason": r.reason.value if r and r.reason else None,
            "best_distance": r.best_distance if r else None,
            "contexts": r.contexts if r else [],
            "total_seconds": outcome.total_seconds,
            "retrieval_seconds": outcome.retrieval_seconds,
            "error": outcome.error,
        }

    results = evaluate_fn(
        target,
        data=examples,
        evaluators=[*CODE_EVALUATORS, *make_judge_evaluators(judge)],
        experiment_prefix=experiment_prefix,
        metadata=metadata or {},
        max_concurrency=1,
    )
    return summarise(results, getattr(results, "experiment_name", None))