# SmartDesk AI

An IT and HR help desk agent. It answers employee questions from a curated knowledge base (RAG), offers to raise a Jira ticket when it can't answer, and reports the status of tickets an employee raised earlier.

- **Answers only from documents.** Answers must cite retrieved passages. If nothing relevant is found, or the model can't ground its answer, it says "I don't know" and offers a ticket.
- **Human in the loop.** A ticket is created only after an explicit "yes" and a valid email. The employee sees the title and details first.
- **Code decides, the model assists.** Consent, email validation and ticket IDs are handled in code, not by the LLM.
- **Private by default.** A ticket is visible only to the email it was raised with. Someone else's ticket looks exactly like a missing one.

## Architecture

```
Employee -> Streamlit UI -> FastAPI (POST /chat, per-conversation lock)
                                |
                              Router  (pending question? code decides; else GPT-4o classifies intent)
        +----------+-----------+------------+-------------+
     kb_answer  chitchat  start_ticket  ask_email/create_ticket  ticket_status
        |                                        |                     |
  Chroma + GPT-4o                              Jira                  Jira
        |
  not answerable -> offer_ticket
```

State (pending question, ticket draft, known email) is stored per `thread_id` by a LangGraph checkpointer: memory, SQLite or Postgres.

## Requirements

- Python 3.11 or newer
- An OpenAI API key (GPT-4o and `text-embedding-3-small`)
- Optional: a Jira Cloud project (otherwise use the built-in mock provider), Docker (for local Postgres), a LangSmith account (for evals)

## Setup

```bash
git clone <your-repo-url> smartdesk-ai && cd smartdesk-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Core + tests. Add extras as needed (see table below).
pip install -e ".[dev]"

cp .env.example .env        # then edit .env and set OPENAI_API_KEY
```

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff | running tests |
| `sqlite` | `langgraph-checkpoint-sqlite` | `CHECKPOINTER=sqlite` |
| `postgres` | `langgraph-checkpoint-postgres`, `psycopg[binary,pool]` | `CHECKPOINTER=postgres` |
| `evals` | `langsmith` | LLM-judged evals |
| `ui` | `streamlit` | the chat UI |

Everything at once: `pip install -e ".[dev,sqlite,postgres,evals,ui]"`

## Add your documents and build the index

Put source documents under `data/raw/`. The first folder is the department:

```
data/raw/it/vpn_setup.md
data/raw/hr/leave_policy.pdf
```

Supported formats: `.pdf`, `.docx`, `.md`, `.txt`. Scanned (image-only) PDFs are skipped because they need OCR.

```bash
python scripts/ingest_docs.py                 # full rebuild (default)
python scripts/ingest_docs.py --raw-dir DIR   # different source folder
python scripts/ingest_docs.py --no-reset      # add/update without wiping (stale chunks may remain)
```

Re-run it whenever documents change, or after changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL`. Until it has run, every question escalates to a ticket offer and `/health` reports `degraded`.

## Environment variables

Settings are read from the environment or a `.env` file in the project root.

### OpenAI

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Chat and structured-output model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

### RAG

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_DIR` | `data/vector_store` | Where Chroma persists the index |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Characters shared between neighbouring chunks |
| `TOP_K` | `5` | Chunks retrieved per question |
| `CONFIDENCE_THRESHOLD` | `0.6` | Maximum cosine distance for a chunk to count as relevant (lower is closer). Tune with `evals`. |

### Tickets

| Variable | Default | Description |
|---|---|---|
| `TICKET_PROVIDER` | `mock` | `mock` (in-memory) or `jira` |
| `JIRA_BASE_URL` | | e.g. `https://your-org.atlassian.net` |
| `JIRA_EMAIL` | | Atlassian account email used for API auth |
| `JIRA_API_TOKEN` | | Token from id.atlassian.com/manage-profile/security/api-tokens |
| `JIRA_PROJECT_KEY` | `HELP` | Project that receives tickets |
| `JIRA_ISSUE_TYPE` | `Task` | Must exist in the project |

`JIRA_*` values are required only when `TICKET_PROVIDER=jira`. Missing values stop the server at startup.

### API and persistence

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CORS_ORIGINS` | empty | Comma-separated browser origins allowed to call the API |
| `CHECKPOINTER` | `memory` | `memory`, `sqlite` or `postgres` (also selects the lock type) |
| `SQLITE_PATH` | `data/checkpoints.sqlite` | SQLite file (single process only) |
| `DATABASE_URL` | | `postgresql://user:pass@host:5432/db`, required for `postgres` |
| `LOCK_DATABASE_URL` | empty | Direct Postgres DSN for locks. Required if `DATABASE_URL` goes through PgBouncer in transaction mode. |
| `POSTGRES_POOL_SIZE` | `10` | Connections for conversation state |
| `LOCK_POOL_SIZE` | `20` | Max in-flight turns per process |
| `LOCK_TIMEOUT_SECONDS` | `30` | Wait for a busy conversation before returning HTTP 409 |

| `CHECKPOINTER` | Survives restart | Several workers or hosts |
|---|---|---|
| `memory` | no | no (development only) |
| `sqlite` | yes | no (one process) |
| `postgres` | yes | yes (advisory locks) |

### UI

| Variable | Default | Description |
|---|---|---|
| `SMARTDESK_API_URL` | `http://localhost:8000` | API address used by the Streamlit app |

### Evals and tracing (LangSmith)

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_API_KEY` | | Required for experiments (not for `--skip-judge`) |
| `LANGSMITH_PROJECT` | `smartdesk-evals` | Tracing project name |
| `LANGSMITH_TRACING` | `false` | `true` also traces every OpenAI call made through `LLMClient` |
| `LANGSMITH_ENDPOINT` | US default | Set `https://eu.api.smith.langchain.com` for an EU workspace |

### Tests only

| Variable | Description |
|---|---|
| `TEST_DATABASE_URL` | Enables the real-Postgres integration tests |

## Running

### Terminal chat

```bash
python cli.py
```

Uses the mock ticket provider unless `TICKET_PROVIDER=jira`.

### API server

```bash
uvicorn smartdesk.api.main:create_app --factory --port 8000 --reload
```

Use `--factory`. There is no module-level `app`, so importing the module never requires an API key. For production use `--workers 1` with `memory` or `sqlite`; use `postgres` before running more than one worker.

| Endpoint | Description |
|---|---|
| `POST /chat` | Body `{"message": "...", "thread_id": "optional"}`. Returns `{thread_id, reply, awaiting}`. Omit `thread_id` on the first message and send the returned one afterwards. |
| `GET /health` | `ok` or `degraded` (empty knowledge base), the ticket provider and the checkpointer in use |
| `GET /architecture` | Mermaid source of the compiled agent graph (if you added this route) |

`awaiting` is `"ticket_confirmation"`, `"email"` or `null`, and tells a UI what the agent just asked for.

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "How do I fix VPN error 809 on my Mac?"}'
```

### Chat UI

```bash
uvicorn smartdesk.api.main:create_app --factory --port 8000   # terminal 1
streamlit run ui/app.py                                       # terminal 2
```

The UI has a chat page (with Yes/No buttons for the ticket offer) and an architecture page with an animated flow, a Mermaid diagram and the live graph.

### Postgres for scale-out (local)

```bash
docker compose up -d postgres
export CHECKPOINTER=postgres
export DATABASE_URL=postgresql://smartdesk:smartdesk@localhost:5432/smartdesk
uvicorn smartdesk.api.main:create_app --factory --port 8000
```

### Jira

Create a throwaway sandbox project first (for example key `HELPTEST`), then:

```bash
export TICKET_PROVIDER=jira JIRA_BASE_URL=... JIRA_EMAIL=... JIRA_API_TOKEN=... JIRA_PROJECT_KEY=HELPTEST
python scripts/jira_smoke_test.py you@corp.com
```

This creates one real ticket and reads it back by ID and by email. It is the only check that proves your credentials, project key and issue type work.

## Tests

```bash
pytest                                          # unit and integration tests; no network, no API key
pytest tests/unit/test_ticket_providers.py -v   # a single file

# Real Postgres tests (skipped without the variable)
docker compose up -d postgres
TEST_DATABASE_URL=postgresql://smartdesk:smartdesk@localhost:5432/smartdesk \
  pytest tests/integration/test_postgres.py -v
```

## Evaluations

Questions live in `evals/questions.jsonl` (knowledge base) and `evals/routing.jsonl` (routing). Rewrite them to match your own documents. A question labelled `should_answer: false` becomes wrong the moment your documents cover it.

```bash
python -m evals.run_evals --limit 5 --skip-judge     # offline smoke run, no LangSmith
python -m evals.run_evals --limit 3                  # first real LangSmith experiment
python -m evals.run_evals --sweep --skip-judge       # find a CONFIDENCE_THRESHOLD
python -m evals.run_evals --threshold 0.5            # try a threshold for this run only

# CI-style gate: exit code 1 if any limit is violated
python -m evals.run_evals --max-false-answer-rate 0 --min-faithfulness 0.85 \
  --min-relevancy 0.8 --max-p95-seconds 6

python scripts/langsmith_check.py                    # show which datasets/experiments your key sees
```

| Metric | How it is computed |
|---|---|
| Faithfulness | LLM judge: share of the answer's claims supported by the retrieved passages |
| Answer relevancy | LLM judge: fully, partially or not addressing the question (1 / 0.5 / 0) |
| False-answer rate | Of the questions that should escalate, how many got an answer (lower is better) |
| Over-escalation rate | Of the answerable questions, how many became ticket offers |
| Keyword hit rate | Does the answer contain the required facts |
| Routing accuracy | Does the router pick the expected branch |
| Latency p50 / p95 | Time inside the RAG chain (embedding, search, GPT-4o) |

Exit codes: `0` ok, `1` a gate failed or the LangSmith run failed, `2` setup problem (empty knowledge base, missing `LANGSMITH_API_KEY`).

Experiments upload questions, answers and retrieved document passages to LangSmith. Check that this is acceptable for your documents before running against real HR content.

## Project layout

```
smartdesk-ai/
├── cli.py                     terminal chat
├── pyproject.toml
├── .env.example
├── docker-compose.yml         local Postgres
├── data/raw/{it,hr}/          source documents
├── data/vector_store/         Chroma index (gitignored)
├── scripts/                   ingest_docs.py, jira_smoke_test.py, langsmith_check.py
├── src/smartdesk/
│   ├── config.py              typed settings
│   ├── llm/                   OpenAI wrapper, structured-output schemas
│   ├── rag/                   loaders, ingest, retriever, confidence rules, chain
│   ├── tickets/               provider interface, Jira, mock, tools
│   ├── agent/                 router, state, prompts, nodes, graph
│   ├── api/                   FastAPI app, routes, locks, persistence
│   └── utils/                 validation, logging
├── ui/                        Streamlit app and animated architecture view
├── evals/                     question sets, metrics, judges, LangSmith runner
└── tests/                     unit, integration, shared fakes
```

## Known limitations

- **The email is not authenticated.** Anyone can type a colleague's email and see their tickets. For production, take identity from SSO or your chat platform and set `state["email"]` before the first turn.
- **No rate limiting.** Every message costs LLM calls. Add limits at your gateway.
- **Personal data is stored.** With `sqlite` or `postgres`, emails and chat text are persisted. Restrict access, encrypt at rest and set a retention period. Chat history is also copied into the Jira ticket.
- **A timed-out Jira create can still produce a duplicate ticket.** There is no idempotency key.
- **Scanned PDFs are not indexed** (no OCR).
- **`CONFIDENCE_THRESHOLD` needs real data.** Tune it with `--sweep` on 50+ real employee questions before trusting it.
- **Some code was written without being run** (the Streamlit UI, the LangSmith evaluation call, the Postgres lock against a live database). Run the smoke checks above after setup.
