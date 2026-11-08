"""Terminal chat for local testing: `python cli.py` (after running scripts/ingest_docs.py)."""

import uuid

from anyio import Path

from smartdesk.agent.graph import build_graph
from pathlib import Path


def main() -> None:
    """Run a chat loop; one thread_id per session keeps state between turns."""
    graph = build_graph()
    

    Path("graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print("Graph saved to graph.png")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("SmartDesk AI (type 'exit' to quit)")
    while (text := input("you> ").strip()) not in {"exit", "quit"}:
        if text:
            out = graph.invoke({"messages": [{"role": "user", "content": text}]}, cfg)
            print("bot>", out["messages"][-1]["content"], "\n")


if __name__ == "__main__":
    main()