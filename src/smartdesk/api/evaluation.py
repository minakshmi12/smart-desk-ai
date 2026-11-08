"""
Evaluate recent chats in LangSmith (behind the UI's "Run evaluation" button).

HOW IT WORKS
1. Take the newest un-evaluated turns from the InteractionLog.
2. Upload them as examples to the dataset "smartdesk-live" (created on first use).
3. Run a LangSmith experiment whose target just REPLAYS the recorded answer (no
   chain re-run, no extra retrieval cost). Evaluators: faithfulness and answer
   relevancy (LLM judges), latency, retrieval latency, escalation.
4. Return the experiment URL so the UI can send the user to the dashboard.

Live traffic has no ground truth, so the judges score every ANSWERED turn;
escalated turns are counted but not judged.

PRIVACY: questions, answers and retrieved passages are uploaded to LangSmith.
"""

import importlib.util
import logging
import os
import threading
import time
from collections import Counter, defaultdict

from smartdesk.api.interactions import InteractionLog
from smartdesk.config import get_settings

logger = logging.getLogger(__name__)

LIVE_DATASET = "smartdesk-live"


class EvaluationUnavailable(Exception):
    """LangSmith is not configured (message is safe to show the user)."""


class EvaluationBusy(Exception):
    """An evaluation is already running."""


class EvaluationCooldown(Exception):
    """Too soon after the previous run; `retry_after` is in seconds."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"Wait {retry_after:.0f}s before running another evaluation.")
        self.retry_after = retry_after


def mean_by_mode(per_case: dict, modes: dict, key: str) -> dict[str, float]:
    """Average a judge score per search method. Cases without a score are skipped."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for case_id, scores in per_case.items():
        value = scores.get(key)
        if value is not None:
            buckets[modes.get(case_id) or "unknown"].append(value)
    return {m: sum(v) / len(v) for m, v in sorted(buckets.items())}


def escalated(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """1.0 when the agent escalated instead of answering (the mean is the escalation rate)."""
    return {"key": "escalated", "score": 0.0 if outputs["answered"] else 1.0}


class EvaluationService:
    """Status + run, with a cooldown and a single-run lock to bound judge cost."""

    def __init__(self, log: InteractionLog, *, client_factory=None, judge_factory=None,
                 evaluate_fn=None, cooldown: float = 30.0, clock=time.monotonic) -> None:
        """The factories, evaluate_fn and clock are injectable for tests."""
        self._log, self._client_factory = log, client_factory
        self._judge_factory, self._evaluate_fn = judge_factory, evaluate_fn
        self._cooldown, self._clock = cooldown, clock
        self._lock = threading.Lock()
        self._last_run: float | None = None
        self._last_url: str | None = None
        self._project_url_cache: str | None = None

    # ------------------------------------------------------------ helpers
    def _availability(self) -> str | None:
        """None if usable, else a user-safe reason."""
        if not os.getenv("LANGSMITH_API_KEY"):
            return "LANGSMITH_API_KEY is not set on the API server."
        if self._client_factory is None and importlib.util.find_spec("langsmith") is None:
            return "The langsmith package is not installed (pip install -e '.[evals]')."
        return None

    def _client(self):
        if self._client_factory:
            return self._client_factory()
        from langsmith import Client
        return Client()

    def _evaluate(self):
        if self._evaluate_fn:
            return self._evaluate_fn
        from langsmith import evaluate
        return evaluate

    def _judge(self):
        if self._judge_factory:
            return self._judge_factory()
        from smartdesk.llm.client import LLMClient
        return LLMClient(model=get_settings().judge_model)

    def _project_url(self) -> str | None:
        """Dashboard URL of the tracing project (needs at least one trace to exist)."""
        if self._project_url_cache:
            return self._project_url_cache
        try:
            project = self._client().read_project(
                project_name=os.getenv("LANGSMITH_PROJECT", "default"))
            self._project_url_cache = getattr(project, "url", None)
        except Exception:
            logger.info("LangSmith tracing project not found yet")
        return self._project_url_cache

    # ------------------------------------------------------------- public
    def status(self) -> dict:
        """Whether evaluation is available, how many chats are recorded, and dashboard links."""
        reason = self._availability()
        recorded, pending = self._log.counts()
        return {
            "enabled": reason is None, "reason": reason,
            "recorded": recorded, "pending": pending,
            "tracing": os.getenv("LANGSMITH_TRACING", "").lower() == "true",
            "project_url": self._project_url() if reason is None else None,
            "last_experiment_url": self._last_url,
        }

    def run(self, limit: int) -> dict:
        """
        Evaluate up to `limit` recent chats.

        Raises: EvaluationUnavailable, EvaluationBusy, EvaluationCooldown; any other
        exception (network, SDK drift) propagates for the route to log.
        """
        reason = self._availability()
        if reason:
            raise EvaluationUnavailable(reason)
        if not self._lock.acquire(blocking=False):
            raise EvaluationBusy()
        try:
            if self._last_run is not None:
                wait = self._cooldown - (self._clock() - self._last_run)
                if wait > 0:
                    raise EvaluationCooldown(wait)
            items = self._log.pending(limit)
            if not items:
                return {"evaluated": 0, "judged": 0, "failed": 0, "faithfulness": None,
                        "answer_relevancy": None, "by_mode": {}, "experiment": None,
                        "url": self._last_url or self._project_url(),
                        "message": "No new chats to evaluate. Ask some questions on the Chat page first."}
            self._last_run = self._clock()
            return self._run_experiment(items)
        finally:
            self._lock.release()

    def _run_experiment(self, items) -> dict:
        try:
            from evals.langsmith_eval import (
                latency_seconds, make_judge_evaluators, retrieval_seconds, summarise)
        except ImportError as exc:
            raise EvaluationUnavailable(
                "The evals package is not importable. Start the API from the project root.") from exc

        client = self._client()
        if client.has_dataset(dataset_name=LIVE_DATASET):
            dataset = client.read_dataset(dataset_name=LIVE_DATASET)
        else:
            dataset = client.create_dataset(
                dataset_name=LIVE_DATASET,
                description="Recent SmartDesk chats (no ground truth; judged for quality)")
        client.create_examples(
            inputs=[{"case_id": i.id, "question": i.question} for i in items],
            outputs=[{"should_answer": True} for _ in items],   # lets the judges apply to answered turns
            metadata=[{"retrieval_mode": i.mode, "top_k": i.top_k, "answered": i.answered}
                      for i in items],
            dataset_id=dataset.id,
        )
        by_id = {i.id: i for i in items}
        examples = [e for e in client.list_examples(dataset_name=LIVE_DATASET)
                    if e.inputs.get("case_id") in by_id]
        if not examples:
            raise RuntimeError("Uploaded examples are not visible in LangSmith yet; try again.")

        def target(inputs: dict) -> dict:
            """Replay the recorded outcome (no chain re-run)."""
            it = by_id[inputs["case_id"]]
            return {"answered": it.answered, "answer": it.answer, "reason": it.reason,
                    "contexts": it.contexts, "total_seconds": it.total_seconds,
                    "retrieval_seconds": it.retrieval_seconds, "error": None}

        results = self._evaluate()(
            target, data=examples,
            evaluators=[latency_seconds, retrieval_seconds, escalated,
                        *make_judge_evaluators(self._judge())],
            experiment_prefix="smartdesk-live",
            metadata={"source": "ui", "batch_size": len(items)},
            max_concurrency=4,
        )
        scores = summarise(results, getattr(results, "experiment_name", None))
        self._log.mark_evaluated([i.id for i in items])

        url = None
        try:
            url = getattr(client.read_project(project_name=scores.experiment_name), "url", None)
        except Exception:
            logger.info("Could not read the experiment URL")
        url = url or getattr(dataset, "url", None)
        self._last_url = url or self._last_url

        modes = {i.id: i.mode for i in items}
        top_ks = {i.id: str(i.top_k) for i in items}
        return {
            "evaluated": len(items), "judged": scores.evaluated, "failed": scores.failed,
            "faithfulness": scores.faithfulness, "answer_relevancy": scores.answer_relevancy,
            "by_mode": {
                "faithfulness": mean_by_mode(scores.per_case, modes, "faithfulness"),
                "answer_relevancy": mean_by_mode(scores.per_case, modes, "answer_relevancy"),
                "chats": dict(Counter(i.mode or "unknown" for i in items)),
            },
            "by_top_k": {
                "faithfulness": mean_by_mode(scores.per_case, top_ks, "faithfulness"),
                "answer_relevancy": mean_by_mode(scores.per_case, top_ks, "answer_relevancy"),
                "chats": dict(Counter(str(i.top_k) for i in items)),
            },
            "experiment": scores.experiment_name, "url": url,
            "message": f"Evaluated {len(items)} chats.",
        }