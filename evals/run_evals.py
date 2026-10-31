"""
Command-line entry point: `python -m evals.run_evals`.

Runs the QA set (as a LangSmith experiment unless --skip-judge), optionally
routing, prints a report, saves JSON, and exits non-zero if a gate is violated.

Exit codes: 0 ok | 1 a gate failed | 2 setup problem

Examples
  python -m evals.run_evals --limit 5 --skip-judge       # offline smoke run
  python -m evals.run_evals --sweep --skip-judge          # find CONFIDENCE_THRESHOLD
  python -m evals.run_evals --max-false-answer-rate 0 --min-faithfulness 0.85 --min-relevancy 0.8
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv

from evals.metrics import (
    GateRecord, check_gates, escalation_by_category, escalation_report,
    keyword_hit_rate, latency_summary, recommend_threshold, routing_report, sweep,
)
from evals.runner import TimedRetriever, run_qa, run_routing
from evals.schemas import load_qa_cases, load_routing_cases
from smartdesk.config import get_settings

SWEEP_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
LOOSE_THRESHOLD = 2.0  # cosine distance maximum: every retrieved chunk reaches the LLM


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    p = argparse.ArgumentParser(description="Evaluate SmartDesk AI.")
    p.add_argument("--qa", default="evals/questions.jsonl", help="Knowledge-base question set")
    p.add_argument("--routing", default="evals/routing.jsonl", help="Routing test set")
    p.add_argument("--limit", type=int, help="Only run the first N QA cases (cheaper)")
    p.add_argument("--threshold", type=float, help="Override CONFIDENCE_THRESHOLD for this run")
    p.add_argument("--sweep", action="store_true",
                   help="Run with a loose threshold and print a threshold-sweep table")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip LangSmith entirely (no upload, no LLM-judged metrics)")
    p.add_argument("--skip-routing", action="store_true")
    p.add_argument("--judge-model", default="gpt-4o", help="Model used as judge")
    p.add_argument("--experiment-prefix", default="smartdesk", help="LangSmith experiment name prefix")
    p.add_argument("--out", help="Report path (default evals/reports/<timestamp>.json)")
    p.add_argument("--max-false-answer-rate", type=float)
    p.add_argument("--min-faithfulness", type=float)
    p.add_argument("--min-relevancy", type=float)
    p.add_argument("--max-p95-seconds", type=float)
    return p


def _pct(x: float | None) -> str:
    """Format a 0-1 value as a percentage, or 'n/a'."""
    return "n/a" if x is None else f"{x:.0%}"


def _sec(x: float | None) -> str:
    """Format seconds, or 'n/a'."""
    return "n/a" if x is None else f"{x:.2f}s"


def _score(x: float | None) -> str:
    """Format a 0-1 score to 2 decimals, or 'n/a'."""
    return "n/a" if x is None else f"{x:.2f}"


def _version(package: str) -> str | None:
    """Installed version of a package, or None."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _print_case(done: int, total: int, o) -> None:
    """One progress line per finished case."""
    status = "ERROR " + o.error if o.error else ("answered" if o.answered else "escalated")
    print(f"  [{done}/{total}] {o.case.id:<10} {status:<18} {o.total_seconds:.2f}s")


def main(argv: list[str] | None = None) -> int:
    """Run the suite. Returns 0 on success, 1 if a gate fails, 2 on setup problems."""
    # The LangSmith SDK reads os.environ only; this exports .env into it.
    # Must run before settings or LLMClient are created.
    load_dotenv()

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Threshold override must be set BEFORE settings are first read.
    if args.sweep:
        os.environ["CONFIDENCE_THRESHOLD"] = str(LOOSE_THRESHOLD)
    elif args.threshold is not None:
        os.environ["CONFIDENCE_THRESHOLD"] = str(args.threshold)
    get_settings.cache_clear()
    settings = get_settings()

    from smartdesk.agent.router import make_router_node
    from smartdesk.llm.client import LLMClient
    from smartdesk.rag.chain import RAGChain
    from smartdesk.rag.retriever import Retriever
    from smartdesk.rag.vector_store import get_collection

    if get_collection().count() == 0:
        print("The knowledge base is empty. Run: python scripts/ingest_docs.py")
        return 2

    all_cases = load_qa_cases(args.qa)
    cases = all_cases[: args.limit]
    llm = LLMClient()
    retriever = TimedRetriever(Retriever(llm))
    chain = RAGChain(retriever=retriever, llm=llm)

    print(f"Running {len(cases)} QA cases (threshold={settings.confidence_threshold}) ...")
    collected: dict = {}
    judge_scores, judge_error = None, None
    if args.skip_judge:
        for o in run_qa(chain, retriever, cases, on_progress=_print_case):
            collected[o.case.id] = o
    else:
        if not os.getenv("LANGSMITH_API_KEY"):
            print("LANGSMITH_API_KEY is not set. Set it in .env, or use --skip-judge.")
            return 2
        try:
            from evals.langsmith_eval import run_langsmith_experiment
            judge_scores = run_langsmith_experiment(
                chain=chain, retriever=retriever, cases=cases, all_cases=all_cases,
                judge=LLMClient(model=args.judge_model), collected=collected,
                experiment_prefix=args.experiment_prefix,
                metadata={"confidence_threshold": settings.confidence_threshold,
                          "top_k": settings.top_k, "chunk_size": settings.chunk_size,
                          "llm_model": settings.llm_model, "judge_model": args.judge_model},
                on_case=lambda o: _print_case(len(collected), len(cases), o),
            )
        except Exception as exc:  # network, auth, SDK version drift
            judge_error = f"{type(exc).__name__}: {exc}"

    outcomes = [collected[c.id] for c in cases if c.id in collected]
    if not outcomes:
        print(f"No cases completed. {judge_error or ''}")
        return 2
    valid = [o for o in outcomes if o.error is None]
    errored = [o for o in outcomes if o.error is not None]

    # ---- code-based metrics
    esc = escalation_report((o.case.should_answer, o.answered) for o in valid)
    by_cat = escalation_by_category((o.case.category, o.case.should_answer, o.answered) for o in valid)
    kw = keyword_hit_rate(
        (o.result.answer, o.case.must_mention)
        for o in valid if o.answered and o.case.should_answer and o.case.must_mention
    )
    lat_all = latency_summary([o.total_seconds for o in valid])
    lat_answered = latency_summary([o.total_seconds for o in valid if o.answered])
    lat_retrieval = latency_summary([o.retrieval_seconds for o in valid])

    # ---- threshold sweep
    sweep_rows = recommended = None
    if args.sweep:
        sweep_rows = sweep(
            [GateRecord(o.case.should_answer, o.result.answerable, o.result.best_distance)
             for o in valid],
            SWEEP_THRESHOLDS,
        )
        recommended = recommend_threshold(sweep_rows)

    # ---- routing
    routing = None
    if not args.skip_routing:
        rcases = load_routing_cases(args.routing)
        routing_outcomes = run_routing(make_router_node(llm), rcases)
        routing = routing_report((o.case.id, o.case.expected_route, o.actual)
                                 for o in routing_outcomes)

    # ---- print
    print("\n=== Escalation decision ===")
    print(f"accuracy {_pct(esc.accuracy)} | false-answer rate {_pct(esc.false_answer_rate)} "
          f"| over-escalation rate {_pct(esc.over_escalation_rate)}  (n={esc.total})")
    for cat, r in by_cat.items():
        print(f"  {cat:<14} accuracy {_pct(r.accuracy)} (n={r.total})")
    bad_answers = [o.case.id for o in valid if o.answered and not o.case.should_answer]
    over_esc = [o.case.id for o in valid if not o.answered and o.case.should_answer]
    if bad_answers:
        print(f"  ANSWERED BUT SHOULD HAVE ESCALATED: {', '.join(bad_answers)}")
    if over_esc:
        print(f"  ESCALATED BUT COULD HAVE ANSWERED:  {', '.join(over_esc)}")
    if errored:
        print(f"  ERRORED (excluded from metrics): {', '.join(o.case.id for o in errored)}")

    print("\n=== Answer quality (answered, should-answer cases) ===")
    print(f"keyword hit rate: {_pct(kw)}")
    if judge_scores:
        print(f"faithfulness: {_score(judge_scores.faithfulness)} | answer relevancy: "
              f"{_score(judge_scores.answer_relevancy)} (n={judge_scores.evaluated}, "
              f"judge failures={judge_scores.failed}, judge={args.judge_model})")
        print(f"  LangSmith experiment: {judge_scores.experiment_name}")
        worst = sorted((s["faithfulness"], cid) for cid, s in judge_scores.per_case.items()
                       if s.get("faithfulness") is not None)[:3]
        if worst:
            print("  lowest faithfulness: " + ", ".join(f"{cid} ({v:.2f})" for v, cid in worst))
    elif judge_error:
        print(f"LangSmith FAILED: {judge_error}")
    else:
        print("Judge skipped.")

    print("\n=== Latency (RAG chain only) ===")
    for label, s in (("all cases", lat_all), ("answered only", lat_answered),
                     ("retrieval step", lat_retrieval)):
        print(f"{label:<15} p50 {_sec(s['p50'])} | p95 {_sec(s['p95'])} | "
              f"mean {_sec(s['mean'])} | max {_sec(s['max'])} (n={s['n']})")

    if routing:
        print("\n=== Routing ===")
        print(f"accuracy {_pct(routing.accuracy)} ({routing.correct}/{routing.total})")
        for cid, exp, act in routing.mismatches:
            print(f"  {cid}: expected {exp}, got {act}")

    if sweep_rows:
        print("\n=== Threshold sweep (approximate) ===")
        print("threshold  false-answer  over-escalation  accuracy")
        for row in sweep_rows:
            r = row.report
            print(f"   {row.threshold:.2f}      {_pct(r.false_answer_rate):>6}        "
                  f"{_pct(r.over_escalation_rate):>6}        {_pct(r.accuracy):>6}")
        print(f"largest threshold with 0% false answers: {recommended}")

    # ---- save
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "confidence_threshold": settings.confidence_threshold, "top_k": settings.top_k,
            "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
            "llm_model": settings.llm_model, "embedding_model": settings.embedding_model,
            "judge_model": args.judge_model, "langsmith_version": _version("langsmith"),
        },
        "escalation": {**esc.__dict__, "accuracy": esc.accuracy,
                       "false_answer_rate": esc.false_answer_rate,
                       "over_escalation_rate": esc.over_escalation_rate},
        "keyword_hit_rate": kw,
        "judge": judge_scores.__dict__ if judge_scores else {"error": judge_error},
        "latency": {"all": lat_all, "answered": lat_answered, "retrieval": lat_retrieval},
        "routing": routing.__dict__ if routing else None,
        "cases": [
            {"id": o.case.id, "category": o.case.category, "should_answer": o.case.should_answer,
             "answered": o.answered, "error": o.error,
             "reason": o.result.reason.value if o.result and o.result.reason else None,
             "best_distance": o.result.best_distance if o.result else None,
             "total_seconds": round(o.total_seconds, 3),
             "retrieval_seconds": round(o.retrieval_seconds, 3)}
            for o in outcomes
        ],
    }
    out = Path(args.out or f"evals/reports/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved to {out}")

    # ---- gates
    failures = check_gates(
        false_answer_rate=esc.false_answer_rate,
        faithfulness=judge_scores.faithfulness if judge_scores else None,
        relevancy=judge_scores.answer_relevancy if judge_scores else None,
        p95_seconds=lat_all["p95"],
        max_false_answer_rate=args.max_false_answer_rate,
        min_faithfulness=args.min_faithfulness,
        min_relevancy=args.min_relevancy,
        max_p95_seconds=args.max_p95_seconds,
    )
    for f in failures:
        print(f"GATE FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())