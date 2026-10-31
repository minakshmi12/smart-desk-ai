"""
SmartDesk chat UI (Streamlit). A thin client: it only calls the FastAPI service.

WHY A THIN CLIENT: the API owns the agent, the per-conversation lock and the
conversation storage. The UI keeps just the thread_id and a copy of the messages
for display. Streamlit reruns this whole script on every interaction, so anything
that must persist lives in st.session_state.

LIMITS: a browser refresh starts a new conversation (new session, no thread_id).
The employee's email is typed by them, not authenticated.
"""

import os

import httpx
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("SMARTDESK_API_URL", "http://localhost:8000")

st.set_page_config(page_title="SmartDesk AI", page_icon="💬")
ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("messages", [])
ss.setdefault("awaiting", None)

ARCHITECTURE = """
flowchart TD
  U[Employee] --> UI[Streamlit UI]
  UI -->|"POST /chat with thread_id"| API[FastAPI]
  API -->|"one turn at a time per thread"| R{Router}
  R -->|"pending question: code decides yes, no, email"| F[Follow-up handling]
  R -->|"fresh message: GPT-4o classifies intent"| I{Intent}
  I -->|kb_query| KB[RAG chain]
  I -->|chitchat| CC[Chit-chat]
  I -->|create_ticket| D[Draft the ticket]
  I -->|check_status| S[Ticket status]
  KB --> V[(Chroma vector store)]
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
  API --- P[(Postgres: conversation state and locks)]
"""

DESIGN_NOTES = """
- **Answers only from documents.** The model must cite numbered passages. No valid
  citation, or a distant match, means it says "I don't know" and offers a ticket.
- **Code decides, the model assists.** Consent (yes/no), email validation and ticket
  IDs are handled in code, so the model can never create a ticket on its own.
- **Human in the loop.** A ticket is only created after an explicit yes, and the
  employee sees the title and details first.
- **Privacy.** Tickets are only visible to the email they were raised with, and
  someone else's ticket looks exactly like a missing one.
- **Resilient.** If OpenAI or Jira is down, the employee still gets a clear next step.
"""


def call_api(message: str, thread_id: str | None) -> tuple[dict | None, str | None]:
    """POST one message. Returns (data, None) on success or (None, user-facing error)."""
    body = {"message": message, **({"thread_id": thread_id} if thread_id else {})}
    try:
        r = httpx.post(f"{API}/chat", json=body, timeout=60)
    except httpx.HTTPError:
        return None, "I can't reach the SmartDesk service right now. Please try again shortly."
    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 409:
        return None, r.json().get("detail", "Please wait a moment and try again.")
    return None, "Sorry, something went wrong. Please try again."


def chat_page() -> None:
    """Render the conversation, the yes/no buttons, and handle new input."""
    st.title("SmartDesk AI")
    st.caption("IT and HR help desk. I answer from company documents and raise a ticket if I can't.")

    # Handle input BEFORE rendering so the new messages appear in this run.
    prompt = st.chat_input("Ask an IT or HR question...")
    text = prompt or ss.pop("quick", None)
    if text:
        ss.messages.append({"role": "user", "content": text})
        with st.spinner("Thinking..."):
            data, error = call_api(text, ss.thread_id)
        if data:
            ss.thread_id, ss.awaiting = data["thread_id"], data["awaiting"]
            ss.messages.append({"role": "assistant", "content": data["reply"]})
        else:
            ss.messages.append({"role": "assistant", "content": error})

    for m in ss.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.text(m["content"])  # plain text: employee input is never rendered as markdown
            else:
                st.markdown(m["content"].replace("\n", "  \n"))  # keep line breaks

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


def architecture_page() -> None:
    """Render the architecture diagram and the key design decisions."""
    st.title("How SmartDesk works")
    # Mermaid is loaded by the BROWSER from a CDN, so this page needs internet access.
    components.html(
        f"""<pre class="mermaid">{ARCHITECTURE}</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad: true }});
</script>""",
        height=900, scrolling=True,
    )
    with st.expander("Diagram source (if it doesn't render)"):
        st.code(ARCHITECTURE, language="text")
    st.subheader("Design decisions")
    st.markdown(DESIGN_NOTES)


page = st.sidebar.radio("Page", ["Chat", "Architecture"])
if st.sidebar.button("New conversation"):
    ss.thread_id, ss.messages, ss.awaiting = None, [], None
    st.rerun()

chat_page() if page == "Chat" else architecture_page()