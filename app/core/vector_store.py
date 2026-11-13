"""Async Qdrant client lifecycle and collection bootstrap."""

import logging

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    """Return the process-wide Qdrant client, or raise if uninitialised."""
    if _qdrant_client is None:
        raise RuntimeError("Qdrant client has not been initialised. Call init_qdrant() first.")
    return _qdrant_client


async def init_qdrant() -> None:
    """Create the Qdrant client and ensure the collection exists.

    Transport-level failures are logged instead of aborting startup so that
    ``/docs`` stays reachable; vector-dependent endpoints translate the outage
    into a structured HTTP 503. Misconfiguration (for example a collection
    dimension mismatch) still fails fast.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

    try:
        await ensure_collection()
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - any transport failure is non-fatal here
        logger.warning("Qdrant is not reachable at startup: %s", exc)


async def ensure_collection() -> None:
    """Create the target collection (cosine distance) when it is missing.

    The vector size is taken from ``EMBEDDING_DIMENSION`` so the deployment can
    switch embedding models without code changes.
    """
    client = get_qdrant()
    collection = settings.qdrant_collection

    try:
        if not await client.collection_exists(collection_name=collection):
            await client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dimension=%d, distance=cosine).",
                collection,
                settings.embedding_dimension,
            )
            return

        info = await client.get_collection(collection_name=collection)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, models.VectorParams) and (
            vectors_config.size != settings.embedding_dimension
        ):
            raise RuntimeError(
                f"Qdrant collection '{collection}' has dimension {vectors_config.size}, "
                f"but EMBEDDING_DIMENSION={settings.embedding_dimension}. "
                "Recreate the collection or fix EMBEDDING_DIMENSION."
            )
        logger.info("Using existing Qdrant collection '%s'.", collection)
    except UnexpectedResponse as exc:
        raise RuntimeError(f"Qdrant collection setup failed: {exc}") from exc


async def close_qdrant() -> None:
    """Close the Qdrant client and its underlying HTTP session."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None
        logger.info("Qdrant client closed.")
