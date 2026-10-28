# SmartDesk AI

SmartDesk AI is a support assistant scaffold for IT and HR knowledge questions, ticket creation, and ticket status checks.

## Project layout

- `src/smartdesk/`: application package
- `data/raw/`: source documents for ingestion
- `data/vector_store/`: local persisted index (ignored by Git)
- `tests/`: unit and integration test suites
- `evals/`: answer-quality evaluation data and runner

The existing datasets under `smart_desk_modules/data_sets/` are retained separately; move or select source documents for ingestion as the ingestion workflow is implemented.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Run the API with `uvicorn smartdesk.api.main:app --reload` after the API implementation is added. The application modules are currently structural placeholders.
