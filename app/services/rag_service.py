"""Conversational RAG: query reformulation, retrieval, and grounded answers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.models.schemas import SourceChunk
from app.services.embedder import EmbeddingService
from app.services.memory_service import ChatMessage, MemoryService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's indexed "
    "documents. Answer using ONLY the provided context. If the context does not "
    "contain the answer, say so plainly instead of guessing. Cite the sources you "
    "use with their bracketed labels, for example [1]. Never invent facts, "
    "filenames, or citations."
)

REFORMULATION_PROMPT = (
    "Rewrite the user's follow-up question into a standalone search query that "
    "makes sense without the conversation. Keep it short and specific. Return "
    "only the rewritten query, with no preamble or quotation marks."
)

NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the indexed documents to answer "
    "that question. Try ingesting a document that covers this topic, or rephrase "
    "the question."
)

# Only recent turns are needed to resolve pronouns and ellipsis.
_REFORMULATION_WINDOW = 6


class LLMError(Exception):
    """Raised when the language model cannot produce a usable response."""


class RetrievalError(Exception):
    """Raised when Qdrant cannot serve the similarity search."""


@dataclass(frozen=True, slots=True)
class RagResult:
    """Outcome of one RAG turn."""

    answer: str
    sources: list[SourceChunk]
    standalone_query: str


class RagService:
    """Retrieve context from Qdrant and answer with an OpenAI-compatible LLM."""

    def __init__(
        self,
        *,
        memory: MemoryService,
        embedder: EmbeddingService,
        qdrant: AsyncQdrantClient,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._qdrant = qdrant
        self._client = client
        self._model = model or settings.llm_model
        self._temperature = settings.llm_temperature if temperature is None else temperature
        self._top_k = top_k or settings.rag_top_k

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.llm_api_key.strip():
                raise LLMError(
                    "LLM_API_KEY is not configured; cannot generate an answer. "
                    "Set it in .env and restart the API."
                )
            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=120.0,
                max_retries=2,
            )
        return self._client

    async def answer(self, *, session_id: str, question: str) -> RagResult:
        """Answer ``question`` with retrieved context and persist the exchange.

        Raises:
            MemoryStoreError: Redis is unavailable.
            EmbeddingError: the embedding provider failed.
            RetrievalError: Qdrant is unavailable.
            LLMError: the chat provider failed or returned nothing usable.
        """
        history = await self._memory.get_history(session_id)
        standalone_query = (
            await self._reformulate(question, history) if history else question
        )

        vectors = await self._embedder.embed_texts([standalone_query])
        if not vectors:
            raise RetrievalError("Could not embed the search query.")
        sources = await self._search(vectors[0])

        if not sources:
            answer = NO_CONTEXT_ANSWER
        else:
            answer = await self._complete(question, history, sources)

        await self._memory.append_exchange(session_id, question, answer)
        logger.info(
            "Answered session %s (history=%d, sources=%d, query=%r).",
            session_id,
            len(history),
            len(sources),
            standalone_query,
        )
        return RagResult(answer=answer, sources=sources, standalone_query=standalone_query)

    async def _reformulate(self, question: str, history: Sequence[ChatMessage]) -> str:
        """Rewrite a follow-up into a standalone query, falling back on failure."""
        recent = history[-_REFORMULATION_WINDOW:]
        transcript = "\n".join(f"{message.role}: {message.content}" for message in recent)
        prompt = (
            f"Conversation so far:\n{transcript}\n\n"
            f"Follow-up question: {question}\n\n"
            "Standalone search query:"
        )

        try:
            response = await self._get_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": REFORMULATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=128,
            )
        except OpenAIError as exc:
            logger.warning("Query reformulation failed; using the original question: %s", exc)
            return question

        content = response.choices[0].message.content or ""
        rewritten = _clean_query(content)
        return rewritten or question

    async def _search(self, vector: list[float]) -> list[SourceChunk]:
        """Query Qdrant and map scored points to attributed source chunks."""
        try:
            response = await self._qdrant.query_points(
                collection_name=settings.qdrant_collection,
                query=vector,
                limit=self._top_k,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001 - Qdrant raises transport-specific errors
            logger.warning("Qdrant search failed: %s", exc)
            raise RetrievalError("The vector store is temporarily unavailable.") from exc

        sources: list[SourceChunk] = []
        for point in response.points:
            source = _to_source_chunk(point)
            if source is not None:
                sources.append(source)
        return sources

    async def _complete(
        self,
        question: str,
        history: Sequence[ChatMessage],
        sources: Sequence[SourceChunk],
    ) -> str:
        """Generate a grounded answer from the retrieved context."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for message in history:
            if message.role == "assistant":
                messages.append({"role": "assistant", "content": message.content})
            else:
                messages.append({"role": "user", "content": message.content})
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{_format_context(sources)}\n\nQuestion: {question}",
            }
        )

        try:
            response = await self._get_client().chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except OpenAIError as exc:
            logger.warning("Chat completion failed: %s", exc)
            raise LLMError(f"LLM provider request failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("The language model returned an empty response.")
        return content


def _clean_query(raw: str) -> str:
    """Normalise a reformulated query emitted by a small local model."""
    value = raw.strip().strip('"').strip("'").strip()
    lowered = value.lower()
    for prefix in ("standalone search query:", "search query:", "query:"):
        if lowered.startswith(prefix):
            value = value[len(prefix) :].strip().strip('"').strip("'").strip()
            break
    # Keep only the first line in case the model adds commentary.
    value = value.splitlines()[0].strip() if value else ""
    return value[:2_000]


def _format_context(sources: Sequence[SourceChunk]) -> str:
    """Render retrieved chunks with stable, citable source labels."""
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        label = source.filename or "unknown source"
        blocks.append(
            f"[{index}] filename={label!r}, chunk={source.chunk_index}, "
            f"score={source.score:.4f}\n{source.text}"
        )
    return "\n\n".join(blocks)


def _to_source_chunk(point: models.ScoredPoint) -> SourceChunk | None:
    """Map one Qdrant point to a ``SourceChunk``, skipping invalid payloads."""
    payload = point.payload or {}
    raw_document_id = payload.get("document_id")
    text = payload.get("text")

    if not isinstance(raw_document_id, str) or not isinstance(text, str) or not text.strip():
        logger.warning("Skipping Qdrant point %s with an incomplete payload.", point.id)
        return None

    try:
        document_id = UUID(raw_document_id)
    except ValueError:
        logger.warning("Skipping Qdrant point %s with an invalid document_id.", point.id)
        return None

    raw_filename = payload.get("filename")
    filename = raw_filename if isinstance(raw_filename, str) else None

    raw_chunk_index = payload.get("chunk_index")
    chunk_index = (
        raw_chunk_index
        if isinstance(raw_chunk_index, int) and raw_chunk_index >= 0
        else 0
    )

    return SourceChunk(
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=max(-1.0, min(1.0, float(point.score))),
    )
