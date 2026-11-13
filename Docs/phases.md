# Implementation Phases

Execute these phases sequentially. Update `Memory.md` after completing each phase.

## Phase 1: Project Scaffolding & Infrastructure
- Initialize project structure (`app/`, `core/`, `models/`, `services/`, `api/`).
- Create `pyproject.toml` and `requirements.txt` with pinned dependencies (`fastapi`, `uvicorn[standard]`, `pydantic-settings`, `sqlalchemy[asyncio]`, `asyncpg`, `aiosqlite`, `redis`, `qdrant-client`, `openai`, `pypdf`, `python-multipart`, `email-validator`).
- Create `docker-compose.yml` defining `qdrant` (port 6333), `redis` (port 6379), and `api` (port 8000), plus `.env.example`.
- Implement `app/core/config.py`, `database.py`, `redis_client.py`, and `vector_store.py` with FastAPI lifespan startup/shutdown hooks.

## Phase 2: Database Models & Pydantic Schemas
- Implement SQLAlchemy ORM models in `app/models/domain.py`:
  - `DocumentMetadata`: `id` (UUID), `filename` (str), `content_type` (str), `chunking_strategy` (str), `chunk_count` (int), `created_at` (datetime).
  - `InterviewBooking`: `id` (UUID), `session_id` (str), `candidate_name` (str), `candidate_email` (str), `interview_date` (date), `interview_time` (time), `created_at` (datetime).
- Implement Pydantic schemas in `app/models/schemas.py` for ingestion responses, chat requests/responses, and `InterviewBookingToolArgs`.

## Phase 3: Document Ingestion Pipeline
- Implement `app/services/extractor.py` supporting `.pdf` and `.txt`.
- Implement `app/services/chunker.py` with a clean `BaseChunker` protocol/ABC and two concrete implementations:
  1. `FixedSizeChunker(chunk_size=500, chunk_overlap=50)`
  2. `SemanticRecursiveChunker(max_chunk_size=800)` (splits on paragraphs `\n\n`, then sentences `. `, preserving semantic coherence).
- Implement `app/services/embedder.py` using `AsyncOpenAI.embeddings.create`.
- Wire up `POST /api/v1/ingest` in `app/api/v1/endpoints/ingest.py` to store vectors in Qdrant and metadata in SQL.

## Phase 4: Conversational RAG & Redis Memory
- Implement `app/services/memory_service.py` to store/retrieve message lists in Redis (`chat:history:{session_id}`) with a sliding window cap (e.g., last 20 messages) and 24h TTL.
- Implement `app/services/rag_service.py`:
  - Query reformulation step: if chat history exists, ask LLM to rewrite follow-up into a standalone search query.
  - Embed standalone query -> search Qdrant (`limit=5`) -> format retrieved chunks with source attribution.

## Phase 5: LLM Tool Calling for Interview Booking
- Implement `app/services/booking_service.py` with OpenAI-compatible tool schema (`book_interview`) requiring `name`, `email`, `date`, and `time`.
- Integrate tool execution loop into `rag_service.py`:
  - Pass `tools=[BOOK_INTERVIEW_TOOL]` to chat completion.
  - When `tool_calls` are returned, validate arguments, insert `InterviewBooking` into SQL DB, append the `tool` role message, and run a second LLM completion to generate a natural confirmation response.

## Phase 6: Verification, Testing & Documentation
- Write pytest integration/unit tests (`tests/`) covering chunkers, ingestion endpoint, multi-turn Redis memory, and interview booking tool execution.
- Verify `docker-compose up --build` starts cleanly and `/docs` Swagger UI works end-to-end.
- Write a clear, concise `README.md` with architecture diagram, setup instructions, environment variables, and `curl` examples.