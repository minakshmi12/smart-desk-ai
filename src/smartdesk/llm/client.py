"""LLM and embedding client wrappers."""
"""
Single gateway to OpenAI.

WHY THIS FILE EXISTS
--------------------
The rest of the codebase (router, RAG, nodes) should never import `openai`
directly. Routing every model call through this one class gives you:
- one place for retries and error handling,
- one place to swap models/providers or add logging and cost tracking,
- easy mocking in tests (replace LLMClient with a fake; no network needed).
"""

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from smartdesk.config import get_settings
import logging
import os

logger = logging.getLogger(__name__)

# T stands for "some Pydantic model class". It lets parse() promise that if you
# pass in IntentResult you get back an IntentResult (not a vague BaseModel),
# which keeps type checkers and autocomplete accurate.
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Thin wrapper around the OpenAI SDK exposing the three things SmartDesk needs:
    free-text chat, structured (schema-validated) output, and embeddings.

    WHY "THIN": it adds reliability (retries) and a stable interface, but no
    business logic. Prompts, routing, and decisions live elsewhere.
    """

    def __init__(self, model: str | None = None) -> None:
            """
            Args:
                model: override the chat model. WHY: the eval judge can use a different
                    model from the one that writes the answers. Embeddings are unaffected.
    
            Tracing: when LANGSMITH_TRACING=true and `langsmith` is installed, the
            OpenAI client is wrapped so calls appear as traces. If the package is
            missing we log a warning and carry on untraced. The wrapper patches the
            client in place, so any method it doesn't recognise still works, just untraced.
            """
            s = get_settings()
            client = OpenAI(api_key=s.openai_api_key.get_secret_value())
            if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
                try:
                    from langsmith.wrappers import wrap_openai
                    client = wrap_openai(client)
                except ImportError:
                    logger.warning("LANGSMITH_TRACING=true but langsmith is not installed")
            self._client = client
            self._model = model or s.llm_model
            self._embed_model = s.embedding_model

    # Retry rationale: rate limits (429) and transient network/5xx errors are
    # normal with LLM APIs. Three attempts with exponential backoff (1s, 2s, 4s,
    # capped at 8s) smooth these over without the user seeing an error.
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        """
        Send a conversation to GPT-4o and return the reply as plain text.

        Args:
            messages: OpenAI-format list, e.g.
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "How do I set up VPN?"}]
            temperature: randomness. Low default (0.2) because a help desk
                should give consistent, factual answers rather than creative ones.

        Used by: the kb_answer node to write the final answer from retrieved chunks.
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        # `content` can be None (e.g. tool calls), so fall back to an empty string.
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def parse(self, messages: list[dict], schema: type[T]) -> T:
        """
        Ask GPT-4o for output that conforms to a Pydantic schema, and return
        it as a validated object instead of a string.

        WHY THIS EXISTS:
        Parsing free-text replies like "I think the intent is create_ticket"
        is fragile. With structured outputs the API guarantees the response
        matches the schema, so the router can do `result.intent == "check_status"`
        with no regex or JSON repair.

        Args:
            messages: same format as chat().
            schema: a Pydantic class (IntentResult, EmailExtraction, AnswerGrade).

        Returns:
            An instance of `schema`.

        Raises:
            ValueError: if the model refuses to answer (e.g. a safety refusal),
            in which case there is no parsed object. Callers should treat this
            as a signal to fall back (e.g. escalate to a ticket).

        temperature=0 because classification and extraction should be
        deterministic: the same message should always route the same way.
        """
        resp = self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=schema,
            temperature=0,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:  # model refusal
            raise ValueError(resp.choices[0].message.refusal or "No parsed output")
        return parsed

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert texts into embedding vectors (lists of floats) for semantic search.

        WHY THIS EXISTS:
        Both sides of RAG need it. At ingestion, every document chunk is embedded
        and stored. At query time, the employee's question is embedded and compared
        to those stored vectors to find the most relevant chunks.

        Args:
            texts: strings to embed. Output order matches input order.

        WHY BATCHING IN GROUPS OF 100:
        A large document set can produce thousands of chunks. Sending them in one
        request risks hitting size/token limits, while one request per chunk is
        slow. Batches of 100 balance both.
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self._client.embeddings.create(model=self._embed_model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


    
