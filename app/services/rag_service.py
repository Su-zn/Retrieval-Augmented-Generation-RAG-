"""Conversational RAG: query reformulation, retrieval, and grounded answers."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.models.schemas import InterviewBookingRead, SourceChunk
from app.services.booking_service import (
    BOOK_INTERVIEW_TOOL,
    BOOK_INTERVIEW_TOOL_NAME,
    BookingService,
    BookingValidationError,
)
from app.services.embedder import EmbeddingService
from app.services.memory_service import ChatMessage, MemoryService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's indexed "
    "documents. Answer using ONLY the provided context. If the context is empty or "
    "does not contain the answer, say so plainly instead of guessing. Cite the "
    "sources you use with their bracketed labels, for example [1]. Never invent "
    "facts, filenames, or citations. You can also book interviews with the "
    "book_interview tool. Call it only when the user has supplied a full name, "
    "email address, date (YYYY-MM-DD), and time (HH:MM); never invent or guess "
    "missing values, ask the user for them instead. Interview dates must be today "
    "or later. After a successful booking, confirm the details."
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
    booking: InterviewBookingRead | None = None


@dataclass(frozen=True, slots=True)
class _Completion:
    """Internal result of the tool-enabled completion loop."""

    answer: str
    booking: InterviewBookingRead | None = None
    booking_attempted: bool = False


class RagService:
    """Retrieve context from Qdrant and answer with an OpenAI-compatible LLM."""

    def __init__(
        self,
        *,
        memory: MemoryService,
        embedder: EmbeddingService,
        qdrant: AsyncQdrantClient,
        booking_service: BookingService | None = None,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._qdrant = qdrant
        self._booking_service = booking_service
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

        When a :class:`BookingService` is configured, the completion is tool
        enabled; a validated ``book_interview`` call is persisted and returned in
        the result. With no retrieved context and no booking attempt, the answer
        stays deterministic instead of giving the model a chance to hallucinate.

        Raises:
            MemoryStoreError: Redis is unavailable.
            EmbeddingError: the embedding provider failed.
            RetrievalError: Qdrant is unavailable.
            LLMError: the chat provider failed or returned nothing usable.
            BookingPersistenceError: a validated booking could not be stored.
        """
        history = await self._memory.get_history(session_id)
        standalone_query = (
            await self._reformulate(question, history) if history else question
        )

        vectors = await self._embedder.embed_texts([standalone_query])
        if not vectors:
            raise RetrievalError("Could not embed the search query.")
        sources = await self._search(vectors[0])

        if not sources and self._booking_service is None:
            answer = NO_CONTEXT_ANSWER
            booking: InterviewBookingRead | None = None
        else:
            completion = await self._complete(
                session_id=session_id,
                question=question,
                history=history,
                sources=sources,
            )
            if (
                not sources
                and completion.booking is None
                and not completion.booking_attempted
            ):
                # No documents matched and the model did not try to book: keep the
                # deterministic refusal even though the completion was tool-enabled.
                answer = NO_CONTEXT_ANSWER
                booking = None
            else:
                answer = completion.answer
                booking = completion.booking

        await self._memory.append_exchange(session_id, question, answer)
        logger.info(
            "Answered session %s (history=%d, sources=%d, booking=%s, query=%r).",
            session_id,
            len(history),
            len(sources),
            booking.id if booking is not None else "none",
            standalone_query,
        )
        return RagResult(
            answer=answer,
            sources=sources,
            standalone_query=standalone_query,
            booking=booking,
        )

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
        *,
        session_id: str,
        question: str,
        history: Sequence[ChatMessage],
        sources: Sequence[SourceChunk],
    ) -> _Completion:
        """Generate a grounded answer, executing ``book_interview`` when requested."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for history_message in history:
            if history_message.role == "assistant":
                messages.append({"role": "assistant", "content": history_message.content})
            else:
                messages.append({"role": "user", "content": history_message.content})

        context = _format_context(sources) or "(no document context matched this question)"
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        )

        client = self._get_client()
        try:
            if self._booking_service is None:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                )
            else:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    tools=[BOOK_INTERVIEW_TOOL],
                )
        except OpenAIError as exc:
            logger.warning("Chat completion failed: %s", exc)
            raise LLMError(f"LLM provider request failed: {exc}") from exc

        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls or self._booking_service is None:
            content = (assistant_message.content or "").strip()
            if not content:
                raise LLMError("The language model returned an empty response.")
            return _Completion(answer=content)

        messages.append(_assistant_tool_message(assistant_message))
        booking: InterviewBookingRead | None = None
        attempted = False
        last_error: str | None = None

        for call in assistant_message.tool_calls:
            if call.function.name != BOOK_INTERVIEW_TOOL_NAME:
                tool_content = f"Unknown tool '{call.function.name}'."
            elif booking is not None:
                tool_content = "Only one interview can be booked per turn."
            else:
                attempted = True
                try:
                    conversation_text = "\n".join(
                        [message.content for message in history] + [question]
                    )
                    booking = await self._booking_service.book(
                        session_id=session_id,
                        arguments=call.function.arguments,
                        conversation_text=conversation_text,
                    )
                    tool_content = json.dumps(
                        {
                            "status": "booked",
                            "booking": booking.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                except BookingValidationError as exc:
                    last_error = str(exc)
                    logger.info(
                        "Rejected booking for session %s: %s", session_id, last_error
                    )
                    tool_content = f"Booking rejected. {last_error}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_content,
                }
            )

        # Second pass: let the model turn the tool result into a natural reply.
        try:
            follow_up = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except OpenAIError as exc:
            logger.warning("Follow-up completion after tool call failed: %s", exc)
            if booking is not None:
                return _Completion(_booking_confirmation(booking), booking, attempted)
            if attempted:
                return _Completion(_booking_clarification(last_error), None, attempted)
            raise LLMError(f"LLM provider request failed: {exc}") from exc

        content = (follow_up.choices[0].message.content or "").strip()
        if not content:
            if booking is not None:
                content = _booking_confirmation(booking)
            elif attempted:
                content = _booking_clarification(last_error)
            else:
                raise LLMError(
                    "The language model returned an empty response after the tool call."
                )
        return _Completion(answer=content, booking=booking, booking_attempted=attempted)


def _assistant_tool_message(
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    """Rebuild the assistant tool-call message for the follow-up request."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls or []
        ],
    }


def _booking_confirmation(booking: InterviewBookingRead) -> str:
    """Deterministic confirmation used when the follow-up completion fails."""
    return (
        f"Your interview is booked for {booking.interview_date.isoformat()} at "
        f"{booking.interview_time.strftime('%H:%M')} for {booking.candidate_name}. "
        f"A confirmation will be sent to {booking.candidate_email}."
    )


def _booking_clarification(error: str | None) -> str:
    """Deterministic clarification used when the follow-up completion fails."""
    prefix = "I couldn't book the interview yet."
    if error:
        prefix = f"{prefix} {error}"
    return (
        f"{prefix} Please provide your full name, email address, interview date "
        "(YYYY-MM-DD), and time (HH:MM)."
    )


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
