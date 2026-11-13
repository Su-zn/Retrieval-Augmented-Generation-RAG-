"""Direct async embedding generation through the OpenAI-compatible SDK."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, OpenAIError

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated or violate the vector schema."""


class EmbeddingService:
    """Generate dense vectors for text batches without LangChain abstractions."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._client = client
        self._model = model or settings.embedding_model
        self._dimension = dimension or settings.embedding_dimension
        self._batch_size = batch_size or settings.embedding_batch_size

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.llm_api_key.strip():
                raise EmbeddingError(
                    "LLM_API_KEY is not configured; cannot generate embeddings. "
                    "Set it in .env and restart the API."
                )
            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=60.0,
                max_retries=2,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving input order.

        Raises:
            EmbeddingError: on provider failures, count mismatches, or when the
                returned vectors do not match ``EMBEDDING_DIMENSION``.
        """
        if not texts:
            return []

        client = self._get_client()
        vectors: list[list[float]] = []

        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            try:
                response = await client.embeddings.create(model=self._model, input=batch)
            except OpenAIError as exc:
                logger.warning("Embedding request failed: %s", exc)
                raise EmbeddingError(f"Embedding provider request failed: {exc}") from exc

            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise EmbeddingError(
                    f"Embedding provider returned {len(ordered)} vectors "
                    f"for a batch of {len(batch)}."
                )

            for item in ordered:
                vector = [float(value) for value in item.embedding]
                if len(vector) != self._dimension:
                    raise EmbeddingError(
                        f"Embedding model '{self._model}' returned {len(vector)} dimensions, "
                        f"but EMBEDDING_DIMENSION={self._dimension}. "
                        "Align the setting with the model or recreate the Qdrant collection."
                    )
                vectors.append(vector)

        return vectors


_default_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """FastAPI dependency returning the shared embedding service."""
    return _default_service
