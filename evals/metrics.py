"""
Pure metric functions (no network, no LLM): fully unit-testable.

WHY THIS FILE EXISTS
--------------------
Metrics are where evals quietly go wrong (an off-by-one percentile, a divide by
zero). Keeping them pure and tested means a number in a report is trustworthy,
and any change in it comes from the agent, not the evaluator.

Rates return None (not 0.0) when their denominator is empty. "No data" and
"zero failures" must never look the same.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _ratio(num: int, den: int) -> float | None:
    """num/den, or None when there is nothing to divide by."""
    return num / den if den else None


# ------------------------------------------------------------- latency
def percentile(values: Sequence[float], p: float) -> float | None:
    """
    The p-th percentile (0-100) using linear interpolation; None for no data.

    WHY NOT numpy: one function doesn't justify a dependency.
    NOTE: with fewer than ~20 samples, p95 is basically the maximum. Treat it
    as indicative only.
    """
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(seconds: Sequence[float]) -> dict:
    """Summarise durations as n, mean, p50, p95 and max (None values when empty)."""
    if not seconds:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(seconds),
        "mean": sum(seconds) / len(seconds),
        "p50": percentile(seconds, 50),
        "p95": percentile(seconds, 95),
        "max": max(seconds),
    }


# ---------------------------------------------------------- escalation
@dataclass(frozen=True)
class EscalationReport:
    """
    Confusion counts for the answer-vs-escalate decision.

    answered_correctly: should answer, did   (good)
    escalated_correctly: should escalate, did (good)
    over_escalated: should answer, escalated  (costs a needless ticket)
    false_answered: should escalate, ANSWERED (risk: a confident wrong answer)
    """

    answered_correctly: int = 0
    escalated_correctly: int = 0
    over_escalated: int = 0
    false_answered: int = 0

    @property
    def total(self) -> int:
        """Number of cases counted."""
        return (self.answered_correctly + self.escalated_correctly
                + self.over_escalated + self.false_answered)

    @property
    def accuracy(self) -> float | None:
        """Share of cases where the decision matched the label."""
        return _ratio(self.answered_correctly + self.escalated_correctly, self.total)

    @property
    def false_answer_rate(self) -> float | None:
        """Of the should-escalate cases, the share that got an answer. Lower is better."""
        return _ratio(self.false_answered, self.false_answered + self.escalated_correctly)

    @property
    def over_escalation_rate(self) -> float | None:
        """Of the should-answer cases, the share that escalated. Lower is better."""
        return _ratio(self.over_escalated, self.over_escalated + self.answered_correctly)


def escalation_report(pairs: Iterable[tuple[bool, bool]]) -> EscalationReport:
    """Build a report from (should_answer, answered) pairs."""
    c = defaultdict(int)
    for should, did in pairs:
        key = {(True, True): "answered_correctly", (False, False): "escalated_correctly",
               (True, False): "over_escalated", (False, True): "false_answered"}[(should, did)]
        c[key] += 1
    return EscalationReport(**c)


def escalation_by_category(items: Iterable[tuple[str, bool, bool]]) -> dict[str, EscalationReport]:
    """
    Per-category reports from (category, should_answer, answered) triples.

    WHY: an overall 90% can hide a 40% score on "near_miss" questions, which is
    exactly where users get confidently wrong answers.
    """
    buckets: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for category, should, did in items:
        buckets[category].append((should, did))
    return {k: escalation_report(v) for k, v in sorted(buckets.items())}


def keyword_hit_rate(items: Iterable[tuple[str, Sequence[str]]]) -> float | None:
    """
    Share of (answer, must_mention) pairs whose answer contains EVERY keyword.

    Case-insensitive substring match. Deliberately crude: it catches an answer
    that omits the key fact ("restart the client") without paying for a judge.
    """
    results = [all(k.lower() in answer.lower() for k in keywords) for answer, keywords in items]
    return _ratio(sum(results), len(results))


# ------------------------------------------------------ threshold sweep
@dataclass(frozen=True)
class GateRecord:
    """
    What is needed to replay the retrieval gate at a different threshold.

    llm_answered: the model's verdict when it was allowed to see everything.
    best_distance: distance of the closest chunk (None = nothing retrieved).
    """

    should_answer: bool
    llm_answered: bool
    best_distance: float | None


@dataclass(frozen=True)
class SweepRow:
    """Escalation results at one candidate threshold."""

    threshold: float
    report: EscalationReport


def sweep(records: Sequence[GateRecord], thresholds: Sequence[float]) -> list[SweepRow]:
    """
    Replay the answer/escalate decision at each threshold.

    A case counts as answered at threshold t if the model answered AND its best
    chunk is within t. This is an APPROXIMATION: a tighter threshold would also
    remove weaker chunks from the prompt, which could change the model's answer.
    It is valid only if the records came from a run with a very loose threshold
    (run_evals --sweep does this).
    """
    rows = []
    for t in thresholds:
        pairs = [
            (r.should_answer,
             r.llm_answered and r.best_distance is not None and r.best_distance <= t)
            for r in records
        ]
        rows.append(SweepRow(t, escalation_report(pairs)))
    return rows


def recommend_threshold(rows: Sequence[SweepRow], max_false_answer_rate: float = 0.0) -> float | None:
    """
    The LARGEST threshold whose false-answer rate stays within the limit.

    Larger thresholds answer more questions, so this minimises over-escalation
    subject to the safety cap. None if no row qualifies (or no data).
    It sits right at the edge of your worst wrong match, so with a small question
    set, choose a value somewhat below it.
    """
    ok = [r.threshold for r in rows
          if r.report.false_answer_rate is not None
          and r.report.false_answer_rate <= max_false_answer_rate]
    return max(ok) if ok else None


# -------------------------------------------------------------- routing
@dataclass(frozen=True)
class RoutingReport:
    """Routing accuracy plus the (id, expected, actual) of every miss."""

    correct: int
    total: int
    mismatches: list[tuple[str, str, str | None]]

    @property
    def accuracy(self) -> float | None:
        """Share of messages routed to the expected node."""
        return _ratio(self.correct, self.total)


def routing_report(items: Iterable[tuple[str, str, str | None]]) -> RoutingReport:
    """Build a RoutingReport from (case_id, expected, actual) triples."""
    items = list(items)
    misses = [i for i in items if i[1] != i[2]]
    return RoutingReport(len(items) - len(misses), len(items), misses)


# ---------------------------------------------------------------- gates
def check_gates(
    *,
    false_answer_rate: float | None,
    faithfulness: float | None,
    relevancy: float | None,
    p95_seconds: float | None,
    max_false_answer_rate: float | None = None,
    min_faithfulness: float | None = None,
    min_relevancy: float | None = None,
    max_p95_seconds: float | None = None,
) -> list[str]:
    """
    Compare results with optional limits; return a list of failure messages.

    FAILS CLOSED: if a limit is set but the metric wasn't measured (e.g. Ragas
    was skipped), that is a failure. A gate that silently passes when it can't
    see the number is worse than no gate.
    """
    failures = []

    def gate(name, value, limit, worse):
        if limit is None:
            return
        if value is None:
            failures.append(f"{name}: not measured, but a limit of {limit} was set")
        elif worse(value, limit):
            failures.append(f"{name}: {value:.3f} violates limit {limit}")

    gate("false_answer_rate", false_answer_rate, max_false_answer_rate, lambda v, l: v > l)
    gate("faithfulness", faithfulness, min_faithfulness, lambda v, l: v < l)
    gate("answer_relevancy", relevancy, min_relevancy, lambda v, l: v < l)
    gate("p95_seconds", p95_seconds, max_p95_seconds, lambda v, l: v > l)
    return failures