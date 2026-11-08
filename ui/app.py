"""
SmartDesk chat UI (Streamlit). A thin client: it only calls the FastAPI service.

Pages: Chat (with a search-method selector and per-answer retrieval details),
Compare search (every method side by side), Evaluation (LangSmith), Architecture.
"""

import os
from urllib.parse import urlencode

import httpx
import streamlit as st
import streamlit.components.v1 as components

from animated_arch import ANIMATED_HTML

API = os.getenv("SMARTDESK_API_URL", "http://localhost:8000")
LANGSMITH_HOME = "https://smith.langchain.com"

st.set_page_config(page_title="SmartDesk AI", page_icon="💬", layout="wide")
ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("messages", [])
ss.setdefault("awaiting", None)
ss.setdefault("compare", None)
ss.setdefault("last_eval", None)

FALLBACK_MODES = {"default": "hybrid", "reranker": "unknown", "modes": [
    {"id": "hybrid", "label": "Hybrid (semantic + keyword)", "description": ""},
    {"id": "dense", "label": "Semantic (dense vectors)", "description": ""},
    {"id": "sparse", "label": "Keyword (BM25)", "description": ""},
]}

ARCHITECTURE = """
flowchart TD
  U[Employee] --> UI[Streamlit UI]
  UI -->|"POST /chat: message, thread_id, search method"| API[FastAPI]
  API -->|"one turn at a time per thread"| R{Router}
  R -->|"pending question: code decides"| F[Follow-up handling]
  R -->|"fresh message: GPT-4o classifies intent"| I{Intent}
  I -->|kb_query| KB[RAG chain]
  I -->|chitchat| CC[Chit-chat]
  I -->|create_ticket| D[Draft the ticket]
  I -->|check_status| S[Ticket status]
  KB --> M{Search method chosen in the UI}
  M -->|semantic| DV[(Chroma vectors)]
  M -->|keyword| BM[BM25 index]
  M -->|hybrid| H[Both, fused with RRF, then reranked]
  H --> DV
  H --> BM
  KB --> G{Relevant and grounded?}
  G -->|yes| A[Answer with sources]
  G -->|no| O[Offer a ticket]
  O --> C{Employee confirms?}
  D --> C
  F --> C
  C -->|email unknown| E[Ask for email]
  C -->|email known| T[Create ticket]
  E --> T
  T --> J[(Jira)]
  S --> J
  API --- P[(Conversation state and locks)]
  API -.->|"recent chats"| LS[(LangSmith evaluation)]
"""

DESIGN_NOTES = """
- **Answers only from documents.** The model must cite numbered passages. No valid
  citation, or a distant match, means it says "I don't know" and offers a ticket.
- **Choose how it searches.** Semantic, keyword or hybrid per message, with every
  passage's scores shown, so retrieval is never a black box.
- **Code decides, the model assists.** Consent, email validation and ticket IDs are
  handled in code, so the model can never create a ticket on its own.
- **Human in the loop.** A ticket is only created after an explicit yes.
- **Measured.** Recent chats can be scored for faithfulness, relevancy and latency in LangSmith.
"""


# ------------------------------------------------------------------ API helpers
def api_get(path: str, timeout: float = 15):
    """GET and return JSON, or None on any failure."""
    try:
        r = httpx.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, body: dict, timeout: float = 60):
    """POST. Returns (data, None) or (None, user-facing error text)."""
    try:
        r = httpx.post(f"{API}{path}", json=body, timeout=timeout)
    except httpx.HTTPError:
        return None, "I can't reach the SmartDesk service right now. Please try again shortly."
    if r.status_code == 200:
        return r.json(), None
    try:
        detail = r.json().get("detail")
    except Exception:
        detail = None
    if r.status_code in (409, 429, 503) and isinstance(detail, str) and detail:
        return None, detail
    return None, "Sorry, something went wrong. Please try again."


@st.cache_data(ttl=60)
def load_modes():
    """Search methods published by the API (so new methods appear automatically)."""
    return api_get("/retrieval/modes")


@st.cache_data(ttl=300)
def fetch_live_mermaid():
    """Mermaid source generated from the compiled graph, or None."""
    data = api_get("/architecture", timeout=10)
    return data["mermaid"] if data else None


# ----------------------------------------------------------------- shared views
def chunk_rows(chunks: list[dict]) -> list[dict]:
    """Table rows for st.dataframe."""
    return [{
        "#": c["rank"], "Document": c["label"], "Distance": c["distance"],
        "BM25": c["bm25"], "Fused": c["fused"], "Rerank": c["rerank"],
        "Passed gate": "yes" if c["passed_gate"] else "no",
        "Used in answer": "yes" if c["cited"] else "",
    } for c in chunks]


def show_retrieval(r: dict, labels: dict) -> None:
    """Render how an answer was found: method, timings, scores and passages."""
    st.caption(
        f"Search: **{labels.get(r['mode'], r['mode'])}** | reranker: {r['reranker']} | "
        f"Top-K: {r.get('top_k', 'default')} | "
        f"{r['seconds_total']}s total, {r['seconds_retrieval']}s retrieval | "
        f"distance threshold {r['threshold']}")
    if r["chunks"]:
        st.dataframe(chunk_rows(r["chunks"]), hide_index=True)
        st.caption("Distance: lower is closer. BM25, Fused and Rerank are empty when a method does not compute them.")
        for c in r["chunks"]:
            st.markdown(f"**Passage {c['rank']}: {c['label']}**")
            st.text(c["text"])
    else:
        st.caption("No passages were retrieved.")


# --------------------------------------------------------------------- sidebar
modes_info = load_modes() or FALLBACK_MODES
mode_ids = [m["id"] for m in modes_info["modes"]]
mode_labels = {m["id"]: m["label"] for m in modes_info["modes"]}
mode_help = {m["id"]: m["description"] for m in modes_info["modes"]}
if ss.get("mode") not in mode_ids:
    ss["mode"] = modes_info["default"] if modes_info["default"] in mode_ids else mode_ids[0]

page = st.sidebar.radio("Page", ["Chat", "Compare search", "Evaluation", "Architecture"])
st.sidebar.radio("Search method", mode_ids, key="mode", format_func=lambda i: mode_labels[i])
st.sidebar.caption(mode_help.get(ss["mode"], ""))
st.sidebar.selectbox(
    "Passages to retrieve", [1, 3, 5, 8], index=1, key="top_k",
    help="Maximum passages to consider. The confidence gate can filter out weak matches.",
)
if st.sidebar.button("New conversation"):
    ss.thread_id, ss.messages, ss.awaiting = None, [], None
    st.rerun()


# ------------------------------------------------------------------------ pages
def chat_page() -> None:
    """Conversation, yes/no buttons, and retrieval details under each KB answer."""
    st.title("SmartDesk AI")
    st.caption(f"IT and HR help desk. Search method: {mode_labels[ss.mode]} | Top-K: {ss.top_k}")

    prompt = st.chat_input("Ask an IT or HR question...")
    text = prompt or ss.pop("quick", None)
    if text:
        ss.messages.append({"role": "user", "content": text})
        body = {"message": text, "retrieval_mode": ss.mode, "top_k": ss.top_k,
                **({"thread_id": ss.thread_id} if ss.thread_id else {})}
        with st.spinner("Thinking..."):
            data, error = api_post("/chat", body)
        if data:
            ss.thread_id, ss.awaiting = data["thread_id"], data["awaiting"]
            ss.messages.append({"role": "assistant", "content": data["reply"],
                                "retrieval": data.get("retrieval")})
        else:
            ss.messages.append({"role": "assistant", "content": error})

    for message_index, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.text(m["content"])
            else:
                st.markdown(m["content"].replace("\n", "  \n"))
                if m.get("retrieval"):
                    cited = [c for c in m["retrieval"].get("chunks", []) if c.get("cited")]
                    if cited:
                        st.markdown("**Sources**")
                        for source_index, chunk in enumerate(cited, start=1):
                            source = chunk.get("source")
                            if not source:
                                continue
                            params = {"source": source}
                            if chunk.get("page") is not None:
                                params["page"] = chunk["page"]
                            st.link_button(
                                chunk["label"],
                                f"{API}/sources?{urlencode(params)}",
                                key=f"source-{message_index}-{source_index}",
                            )
                    with st.expander("How this was found"):
                        show_retrieval(m["retrieval"], mode_labels)

    if ss.awaiting == "ticket_confirmation":
        yes, no, _ = st.columns([2, 2, 6])
        if yes.button("Yes, create it", type="primary"):
            ss.quick = "yes"
            st.rerun()
        if no.button("No thanks"):
            ss.quick = "no"
            st.rerun()
    elif ss.awaiting == "email":
        st.caption("Type your work email address in the box below.")


def compare_page() -> None:
    """Run one question through every search method and show the rankings side by side."""
    st.title("Compare search methods")
    st.caption("Retrieval only: no answer is generated. Shows what each method would hand to the model.")
    query = st.text_input("Question", placeholder="e.g. vpn error 809")
    dept = st.selectbox("Department filter", ["all", "it", "hr"])
    if st.button("Compare all methods", type="primary") and query.strip():
        with st.spinner("Searching with every method..."):
            data, error = api_post("/retrieval/compare", {
                "query": query.strip(), **({"department": dept} if dept != "all" else {})}, timeout=120)
        ss.compare = data or {"error": error}

    result = ss.compare
    if not result:
        return
    if "error" in result:
        st.error(result["error"])
        return
    st.caption(f"Distance threshold for the confidence gate: {result['threshold']}")
    cols = st.columns(len(result["results"]))
    for col, r in zip(cols, result["results"]):
        with col:
            st.subheader(r["label"])
            st.caption(f"{r['seconds']}s | {len(r['chunks'])} passages")
            if r["error"]:
                st.error(r["error"])
            elif r["chunks"]:
                st.dataframe(chunk_rows(r["chunks"]), hide_index=True)
                top = r["chunks"][0]
                st.caption(f"Top result: {top['label']}")
                with st.expander("Top passage"):
                    st.text(top["text"])
            else:
                st.caption("No passages.")


def evaluation_page() -> None:
    """Run LangSmith evaluation on recent chats and jump to the dashboard."""
    st.title("Evaluation")
    st.caption("Scores recent chats for faithfulness, answer relevancy and latency in LangSmith.")
    status = api_get("/evaluation/status", timeout=20)
    if status is None:
        st.error("Could not reach the API.")
        return
    if not status["enabled"]:
        st.warning(status["reason"])
        st.markdown("Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_TRACING=true`) in the API's "
                    "`.env`, then restart the API.")
        return

    a, b, c = st.columns(3)
    a.metric("Chats recorded", status["recorded"])
    b.metric("Waiting for evaluation", status["pending"])
    c.metric("Live tracing", "on" if status["tracing"] else "off")
    if not status["tracing"]:
        st.caption("Set LANGSMITH_TRACING=true on the API to also see live traces of every call.")

    limit = st.slider("Chats to evaluate", 1, 50, 20)
    if st.button("Run evaluation on recent chats", type="primary", disabled=status["pending"] == 0):
        with st.spinner("Uploading chats and running the judges (this can take a minute)..."):
            data, error = api_post("/evaluation/run", {"limit": limit}, timeout=300)
        ss.last_eval = data or {"error": error}
    if status["pending"] == 0:
        st.caption("Nothing new to evaluate. Ask questions on the Chat page first.")

    result = ss.last_eval
    if result and "error" in result:
        st.error(result["error"])
    elif result:
        st.success(result["message"])
        m1, m2, m3 = st.columns(3)
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        m1.metric("Faithfulness", fmt(result["faithfulness"]))
        m2.metric("Answer relevancy", fmt(result["answer_relevancy"]))
        m3.metric("Judged / evaluated", f"{result['judged']} / {result['evaluated']}")
        if result["failed"]:
            st.caption(f"{result['failed']} judge calls failed and are excluded from the averages.")
        by = result.get("by_mode") or {}
        if by.get("chats"):
            st.markdown("**By search method**")
            st.dataframe([{
                "Method": mode_labels.get(m, m), "Chats": n,
                "Faithfulness": by["faithfulness"].get(m), "Relevancy": by["answer_relevancy"].get(m),
            } for m, n in by["chats"].items()], hide_index=True)
        by_k = result.get("by_top_k") or {}
        if by_k.get("chats"):
            st.markdown("**By Top-K**")
            st.dataframe([{
                "Top-K": int(k), "Chats": n,
                "Faithfulness": by_k["faithfulness"].get(k),
                "Relevancy": by_k["answer_relevancy"].get(k),
            } for k, n in by_k["chats"].items()], hide_index=True)
        if result.get("url"):
            st.link_button("Open this evaluation in LangSmith", result["url"], type="primary")

    st.divider()
    last = (result or {}).get("url") or status.get("last_experiment_url")
    if last and not (result and result.get("url")):
        st.link_button("Open the latest evaluation in LangSmith", last)
    st.link_button("Open live traces", status.get("project_url") or LANGSMITH_HOME)


def render_mermaid(source: str, height: int = 900) -> None:
    """Render Mermaid in the browser (loads Mermaid from a CDN, so needs internet)."""
    components.html(
        f"""<pre class="mermaid">{source}</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad: false }});
const diagram = document.querySelector(".mermaid");
try {{
    const {{ svg }} = await mermaid.render("smartdesk-diagram", diagram.textContent);
    diagram.innerHTML = svg;
}} catch (error) {{
    diagram.textContent = `Diagram could not be rendered: ${{error.message}}`;
}}
</script>""", height=height, scrolling=True)


def architecture_page() -> None:
    """Animated flow, hand-written diagram, and the diagram generated from the code."""
    st.title("How SmartDesk works")
    view = st.radio(
        "Architecture view", ["Animated flow", "Diagram (Mermaid)", "Live graph"],
        horizontal=True, label_visibility="collapsed",
    )
    if view == "Animated flow":
        components.html(ANIMATED_HTML, height=860, scrolling=True)
    elif view == "Diagram (Mermaid)":
        render_mermaid(ARCHITECTURE)
        with st.expander("Diagram source (if it doesn't render)"):
            st.code(ARCHITECTURE, language="text")
    else:
        source = fetch_live_mermaid()
        if source:
            st.caption("Generated from the compiled LangGraph, so it always matches the code.")
            render_mermaid(source, 700)
        else:
            st.info("Could not load /architecture from the API.")
    st.subheader("Design decisions")
    st.markdown(DESIGN_NOTES)


{"Chat": chat_page, "Compare search": compare_page,
 "Evaluation": evaluation_page, "Architecture": architecture_page}[page]()