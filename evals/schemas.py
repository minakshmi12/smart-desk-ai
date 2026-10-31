"""
Eval case definitions and JSONL loading.

WHY THIS FILE EXISTS
--------------------
The question set is the most valuable asset in this suite. Loading it through
one validating function catches typos (a missing field, a duplicate id) before
a paid run starts, instead of halfway through it.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QACase:
    """
    One knowledge-base question with its expected behaviour.

    should_answer: True if the documents contain the answer. This is the ground
        truth for the escalate-vs-answer decision.
    category: free-text tag used to break results down (e.g. "answerable",
        "near_miss", "off_topic", "injection"). "near_miss" (related topic, but
        not actually covered) is the category that exposes weak thresholds.
    must_mention: words a correct answer must contain (case-insensitive); a cheap,
        deterministic correctness check. Empty = not checked.
    department: optional retrieval filter ("it" / "hr").
    """

    id: str
    question: str
    should_answer: bool
    category: str = "general"
    must_mention: tuple[str, ...] = ()
    department: str | None = None


@dataclass(frozen=True)
class RoutingCase:
    """
    One fresh message and the graph node the router should pick.

    expected_route: one of kb_answer, chitchat, start_ticket, ask_email.
        (A status question with no known email routes to ask_email.)
    """

    id: str
    message: str
    expected_route: str


def _read_jsonl(path: str | Path) -> list[dict]:
    """
    Parse a JSONL file into dicts, skipping blank lines.

    Raises:
        ValueError: with the file and line number if a line isn't valid JSON,
        so a typo is fixed in seconds rather than hunted for.
    """
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return rows


def _check(rows: list[dict], required: tuple[str, ...], path) -> None:
    """Raise ValueError if any row lacks a required key or ids are not unique."""
    seen: set[str] = set()
    for row in rows:
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"{path}: case {row.get('id', '?')!r} is missing {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        seen.add(row["id"])


def load_qa_cases(path: str | Path) -> list[QACase]:
    """Load and validate the knowledge-base question set."""
    rows = _read_jsonl(path)
    _check(rows, ("id", "question", "should_answer"), path)
    return [
        QACase(
            id=r["id"], question=r["question"], should_answer=bool(r["should_answer"]),
            category=r.get("category", "general"),
            must_mention=tuple(r.get("must_mention", [])),
            department=r.get("department"),
        )
        for r in rows
    ]


def load_routing_cases(path: str | Path) -> list[RoutingCase]:
    """Load and validate the routing test set."""
    rows = _read_jsonl(path)
    _check(rows, ("id", "message", "expected_route"), path)
    return [RoutingCase(r["id"], r["message"], r["expected_route"]) for r in rows]