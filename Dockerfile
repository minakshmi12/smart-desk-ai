FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencies first, so rebuilds are fast when only code changes
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[postgres,evals,ui]"

COPY evals ./evals
COPY scripts ./scripts
COPY ui ./ui

# Non-root user; /app/data/store becomes the persistent volume
RUN useradd -m app && mkdir -p /app/data/store && chown -R app /app/data
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "smartdesk.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]