"""Interaction log, live evaluation service and the new API endpoints (no network)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from evals.judges import ClaimCheck, FaithfulnessVerdict, RelevanceVerdict
from smartdesk.api.evaluation import (
    EvaluationBusy, EvaluationCooldown, EvaluationService, EvaluationUnavailable, mean_by_mode)
from smartdesk.api.interactions import InteractionLog
from smartdesk.api.main import create_app
from smartdesk.config import get_settings
from smartdesk.rag.retriever import RetrievedChunk


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """Dummy OpenAI and LangSmith settings."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "proj")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _retrieval(mode="hybrid", passed=True, seconds=1.0, top_k=3):
    return {"mode": mode, "question": "vpn 809", "threshold": 0.6, "reranker": "llm",
            "top_k": top_k,
            "seconds_total": seconds, "seconds_retrieval": 0.4,
            "chunks": [{"rank": 1, "label": "VPN (it/vpn.md)", "text": "Restart the client",
                        "distance": 0.2, "bm25": None, "fused": None, "rerank": None,
                        "passed_gate": passed, "cited": passed}]}


# ------------------------------------------------------------------- log
def test_log_records_contexts_marks_evaluated_and_caps_size():
    """Only gate-passing passages become contexts; evaluated turns leave `pending`; maxlen drops the oldest."""
    log = InteractionLog(maxlen=2)
    a = log.record(_retrieval(), "Restart.", answered=True)
    log.record(_retrieval(passed=False), "I don't know", answered=False, reason="low_similarity")
    assert a.contexts == ["Restart the client"] and a.top_k == 3 and log.counts() == (2, 2)
    log.mark_evaluated([a.id])
    assert log.counts() == (2, 1)
    log.record(_retrieval(), "x", answered=True)           # evicts the oldest
    assert log.counts()[0] == 2 and len(log.pending(10)) == 2
    assert [i.answer for i in log.pending(1)] == ["x"]     # newest `limit`, oldest first


def test_mean_by_mode_groups_and_skips_missing_scores():
    """Per-method averages ignore None scores."""
    per_case = {"a": {"faithfulness": 1.0}, "b": {"faithfulness": 0.5}, "c": {"faithfulness": None}}
    assert mean_by_mode(per_case, {"a": "dense", "b": "dense", "c": "hybrid"}, "faithfulness") == {"dense": 0.75}


# --------------------------------------------------------------- service
class _Judge:
    def parse(self, messages, schema):
        if schema is FaithfulnessVerdict:
            return FaithfulnessVerdict(claims=[ClaimCheck(claim="x", supported=True)])
        return RelevanceVerdict(verdict="fully_addresses", reasoning="ok")


class _Client:
    """Minimal langsmith.Client stand-in."""

    def __init__(self):
        self.dataset, self.examples, self.created = None, [], 0

    def has_dataset(self, dataset_name):
        return self.dataset is not None

    def create_dataset(self, dataset_name, description=""):
        self.dataset = SimpleNamespace(id="d1", url="https://smith.test/ds")
        return self.dataset

    def read_dataset(self, dataset_name):
        return self.dataset

    def create_examples(self, inputs, outputs, metadata, dataset_id):
        self.created += 1
        self.examples += [SimpleNamespace(inputs=i, outputs=o, metadata=m)
                          for i, o, m in zip(inputs, outputs, metadata)]

    def list_examples(self, dataset_name):
        return list(self.examples)

    def read_project(self, project_name):
        return SimpleNamespace(url=f"https://smith.test/{project_name}")


class _Results:
    experiment_name = "smartdesk-live-abc"

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def _fake_evaluate(target, data, evaluators, **kw):
    rows = []
    for ex in data:
        out = target(ex.inputs)
        res = [ev(inputs=ex.inputs, outputs=out, reference_outputs=ex.outputs) for ev in evaluators]
        rows.append({"example": ex, "evaluation_results": {"results": [SimpleNamespace(**r) for r in res]}})
    return _Results(rows)


def _service(log, clock):
    client = _Client()
    svc = EvaluationService(log, client_factory=lambda: client, judge_factory=_Judge,
                            evaluate_fn=_fake_evaluate, cooldown=30, clock=lambda: clock[0])
    return svc, client


def test_run_evaluates_pending_chats_groups_by_mode_and_respects_cooldown():
    """
    Two answered chats (dense, hybrid) and one escalation are evaluated once; judges skip
    the escalation; the URL points at the experiment; an immediate rerun is refused, and
    a rerun after the cooldown has nothing to do and uploads nothing.
    """
    log, clock = InteractionLog(), [0.0]
    log.record(_retrieval("dense"), "A", answered=True)
    log.record(_retrieval("hybrid"), "B", answered=True)
    log.record(_retrieval("hybrid", passed=False), "I don't know", answered=False, reason="low_similarity")
    svc, client = _service(log, clock)

    out = svc.run(10)
    assert (out["evaluated"], out["judged"]) == (3, 2)
    assert out["faithfulness"] == 1.0 and out["answer_relevancy"] == 1.0
    assert out["by_mode"]["faithfulness"] == {"dense": 1.0, "hybrid": 1.0}
    assert out["by_mode"]["chats"] == {"dense": 1, "hybrid": 2}
    assert out["by_top_k"]["chats"] == {"3": 3}
    assert out["url"] == "https://smith.test/smartdesk-live-abc" and log.counts() == (3, 0)

    with pytest.raises(EvaluationCooldown):
        svc.run(10)
    clock[0] = 31
    assert svc.run(10)["evaluated"] == 0 and client.created == 1


def test_unconfigured_langsmith_is_reported_not_run(monkeypatch):
    """Without LANGSMITH_API_KEY status says why, and run raises EvaluationUnavailable."""
    monkeypatch.delenv("LANGSMITH_API_KEY")
    svc, _ = _service(InteractionLog(), [0.0])
    assert svc.status()["enabled"] is False and "LANGSMITH_API_KEY" in svc.status()["reason"]
    with pytest.raises(EvaluationUnavailable):
        svc.run(5)


# ------------------------------------------------------------------- API
class _Agent:
    def __init__(self, out):
        self.out, self.configs = out, []

    def invoke(self, payload, config):
        self.configs.append(config)
        return {"messages": [payload["messages"][0], {"role": "assistant", "content": "Restart the client."}],
                **self.out}


KB_TURN = {"route": "kb_answer", "retrieval": _retrieval(), "answer": "Restart the client.",
           "escalate": False, "awaiting": None}


def test_chat_passes_the_chosen_mode_returns_retrieval_and_records_it():
    """The mode reaches the graph config; KB turns return and record retrieval; other turns do not."""
    agent = _Agent(KB_TURN)
    app = create_app(agent=agent)
    client = TestClient(app)
    body = client.post("/chat", json={"message": "vpn 809", "retrieval_mode": "dense", "top_k": 8}).json()
    assert agent.configs[0]["configurable"]["retrieval_mode"] == "dense"
    assert agent.configs[0]["configurable"]["top_k"] == 8
    assert body["retrieval"]["mode"] == "hybrid" and app.state.interactions.counts() == (1, 1)

    stale = TestClient(create_app(agent=_Agent({**KB_TURN, "route": "chitchat"})))
    assert stale.post("/chat", json={"message": "hi"}).json()["retrieval"] is None


def test_unknown_mode_is_rejected_and_modes_are_listed():
    """A mode outside the registry gets 422; /retrieval/modes lists every registered method."""
    client = TestClient(create_app(agent=_Agent(KB_TURN)))
    assert client.post("/chat", json={"message": "x", "retrieval_mode": "magic"}).status_code == 422
    data = client.get("/retrieval/modes").json()
    assert {m["id"] for m in data["modes"]} == {"dense", "sparse", "hybrid"}


def test_compare_runs_every_mode_and_isolates_failures():
    """One failing method is reported in its own result, not as a failed request."""
    class Multi:
        def search(self, query, department=None, mode=None):
            if mode == "sparse":
                raise RuntimeError("boom")
            return [RetrievedChunk("Restart", {"title": "VPN", "source": "it/vpn.md"}, 0.2, id="a")]

    res = TestClient(create_app(agent=_Agent({}), retriever=Multi())).post(
        "/retrieval/compare", json={"query": "vpn 809"}).json()
    by_mode = {r["mode"]: r for r in res["results"]}
    assert set(by_mode) == {"dense", "sparse", "hybrid"}
    assert by_mode["sparse"]["error"] and not by_mode["dense"]["error"]
    assert by_mode["dense"]["chunks"][0]["passed_gate"] is True


@pytest.mark.parametrize("exc,status", [
    (EvaluationBusy(), 409), (EvaluationCooldown(4.2), 429), (EvaluationUnavailable("nope"), 503)])
def test_evaluation_errors_map_to_http_statuses(exc, status):
    """Busy -> 409, cooldown -> 429 with Retry-After, not configured -> 503."""
    class Eval:
        def run(self, limit):
            raise exc

    res = TestClient(create_app(agent=_Agent({}), evaluator=Eval())).post("/evaluation/run", json={"limit": 5})
    assert res.status_code == status
    if status == 429:
        assert res.headers["retry-after"] == "5"