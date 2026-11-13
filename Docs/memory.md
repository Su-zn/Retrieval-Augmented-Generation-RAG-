---

### 6. `Memory.md` (Agent State Tracker for OpenCode)

```markdown
# Project Memory & Progress Log

## Current Status
- **Active Phase:** Phase 3 (Document Ingestion Pipeline)
- **Completed Phases:** Phase 1 (Scaffolding & Infrastructure), Phase 2
  (Database Models & Pydantic Schemas)
- **Blockers:** None. Docker Desktop 29.9.3 (Compose v5.7.0) is installed and
  the full stack has been built and verified locally.

## Key Architectural Decisions
- **Vector Store:** Qdrant (`AsyncQdrantClient`) using cosine distance and dynamic vector dimension matching the configured embedding model (default `1536` for `text-embedding-3-small`, configurable via `EMBEDDING_DIMENSION` env var so local/free models also work).
- **Relational DB:** SQLAlchemy 2.0 async configured via `DATABASE_URL` (defaults to `sqlite+aiosqlite:///./app.db` for instant local run, and `postgresql+asyncpg://...` inside `docker-compose.yml`).
- **LLM Client:** OpenAI-compatible `AsyncOpenAI` client reading `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and `EMBEDDING_MODEL` from `.env` so any provider (OpenAI, OpenRouter, Groq) works without code changes.
- **Configuration:** one cached `Settings` object (`app/core/config.py`, `pydantic-settings`) shared process-wide; `ENVIRONMENT` accepts `local | development | docker | staging | production | test`. Blank optional env values (`QDRANT_API_KEY`, `LLM_BASE_URL`) are normalised to `None`.
- **Startup resilience:** database failures are fatal (metadata/booking writes are critical); Redis and Qdrant *transport* failures are logged as warnings so `/docs` remains reachable, with capability endpoints responsible for returning structured 503s. A Qdrant collection dimension mismatch still fails fast with an actionable `RuntimeError`.
- **Vector collection bootstrap:** the collection is created automatically on startup with cosine distance and `EMBEDDING_DIMENSION`; existing collections are reused and their dimension validated.
- **Session management:** `async_sessionmaker(expire_on_commit=False)`; `get_db()` yields a session, rolls back on exception, and services own commits. SQLite uses `check_same_thread=False`, plus `StaticPool` for in-memory URLs.
- **Upload limit:** `MAX_UPLOAD_BYTES` (default 10 MiB) lives in settings for Phase 3 enforcement.
- **Docker:** `api` waits on healthy `qdrant`, `redis`, and `postgres`; the container runs as a non-root user and exposes a `/health` liveness probe used by the compose healthcheck.
- **ORM modelling (Phase 2):** SQLAlchemy 2.0 `DeclarativeBase` with `Mapped`/`mapped_column`. Primary keys use the portable `sqlalchemy.Uuid` type (`default=uuid.uuid4`) so SQLite and PostgreSQL share one schema. `created_at` is timezone-aware `DateTime(timezone=True)` with `server_default=func.now()`. `InterviewBooking.session_id` is indexed for per-session lookups.
- **API schemas (Phase 2):** Pydantic v2 only, with a shared `ChunkingStrategy = Literal["fixed", "semantic"]`. `ChatRequest` trims whitespace and rejects blank inputs; `InterviewBookingToolArgs` keeps calendar date/time as strings (LLM JSON payloads) but enforces `YYYY-MM-DD` / `HH:MM` patterns plus real-calendar validation; `InterviewBookingRead` uses `from_attributes=True`; `ChatResponse` carries optional `sources` and `booking`.

## Completed Checklist
- [x] Phase 1: Scaffolding, Docker Compose, Core Clients (Qdrant, Redis, SQLAlchemy)
- [x] Phase 2: ORM Models (`DocumentMetadata`, `InterviewBooking`) & Pydantic Schemas
- [ ] Phase 3: Document Ingestion API (`/api/v1/ingest`) + Fixed & Semantic Chunkers
- [ ] Phase 4: Redis Chat Memory + Custom Multi-Turn RAG Service
- [ ] Phase 5: LLM Tool Calling (`book_interview`) + Booking Persistence
- [ ] Phase 6: End-to-End Verification, Pytest Suite & `README.md`

## Phase 1 — Delivered Artifacts
- `pyproject.toml` (project metadata, pinned deps, ruff/mypy/pytest config),
  `requirements.txt` (runtime, fully pinned), `requirements-dev.txt`.
- `Dockerfile` (python:3.12-slim, non-root `appuser`, uvicorn entrypoint),
  `docker-compose.yml` (qdrant `6333`/`6334`, redis `6379`, postgres `5432`,
  api `8000`, named volumes + healthchecks), `.env.example` (every setting
  documented), `.gitignore`, `.dockerignore`.
- `app/core/config.py`, `app/core/database.py`, `app/core/redis_client.py`,
  `app/core/vector_store.py`.
- `app/main.py` with async `lifespan` startup/shutdown hooks, app factory, and
  `/health`; `app/api/v1/router.py` ready for endpoint registration.
- Package skeleton: `app/`, `core/`, `models/`, `services/`, `api/v1/endpoints/`
  (all `__init__.py` files present, `app.services`/`app.models` documented).
- Verification performed: `ruff check app` clean, `mypy app` clean (13 files),
  TestClient lifespan test starts/stops cleanly and returns HTTP 200 from
  `/health`; settings override test passed (`ENVIRONMENT=docker`, blank
  `QDRANT_API_KEY` -> `None`); compose YAML parses with all four services.
- Extended verification (29 behavioral + 35 infrastructure checks, all passed):
  settings validators, SQLite engine/session commit-rollback paths, Redis
  unreachable-server handling, Qdrant collection create/reuse/dimension-mismatch/
  outage paths via a fake client, `/health` + `/docs` + OpenAPI, compose
  topology/healthchecks/dependency ordering, pinned dependency parity,
  `.env.example` coverage, and a forbidden-library scan (no faiss/chromadb/
  langchain/RetrievalQA references). A live uvicorn smoke test served `/health`
  and `/docs` with HTTP 200.
- Docker verification (completed): `docker compose build` succeeded and
  `docker compose up -d` brought up qdrant, redis, postgres, and api with all
  four reporting **healthy**. Startup logs confirm Postgres connectivity, Redis
  ping, and automatic creation of the `documents` Qdrant collection
  (dimension 1536, cosine). Live checks: `/health` 200 (env=docker), `/docs`
  200, Qdrant `/collections` lists `documents`, `redis-cli ping` -> PONG,
  Postgres 16.16 reachable. The stack is now verified end-to-end; no outstanding
  Docker issues.
- Dependency pin correction: `openai==1.59.0` does not exist on PyPI; pinned to
  `openai==1.59.2` (the docs only require the OpenAI-compatible SDK).

## Phase 2 — Delivered Artifacts
- `app/models/domain.py`: `DocumentMetadata` (`id`, `filename`, `content_type`,
  `chunking_strategy`, `chunk_count`, `created_at`) and `InterviewBooking`
  (`id`, `session_id`, `candidate_name`, `candidate_email`, `interview_date`,
  `interview_time`, `created_at`), both on the shared `Base`, with portable
  `Uuid` primary keys, indexed `session_id`, and `created_at` server defaults.
- `app/models/schemas.py`: `ChunkingStrategy`, `IngestionResponse`,
  `ChatRequest`, `SourceChunk`, `InterviewBookingRead`, `ChatResponse`, and
  `InterviewBookingToolArgs` (with `as_booking_values()` for Phase 5).
- `app/core/database.py`: `init_db()` now imports `app.models.domain` so both
  tables are registered on `Base.metadata` before `create_all`.
- Verification performed: `ruff check app` clean; `mypy app` clean (15 files);
  in-memory SQLite round-trip inserted and read back a document + booking
  (UUID parsing, `created_at` defaults, date/time fidelity); Pydantic positive
  and negative cases for email, impossible dates (`2026-02-30`), invalid times
  (`25:00`), blank names, and blank chat fields; TestClient lifespan startup
  still returns HTTP 200 from `/health` with the new models registered.

## Notes for Next Step
- Phase 3: implement `app/services/extractor.py` (`pypdf` PDF parsing wrapped in
  `asyncio.to_thread`, UTF-8 TXT decoding with graceful `UnicodeDecodeError`
  handling), then `chunker.py` (`BaseChunker` ABC + `FixedSizeChunker(500, 50)`
  + `SemanticRecursiveChunker(800)`), `embedder.py` (`AsyncOpenAI` batch
  embeddings), and `POST /api/v1/ingest`.
- Enforce `MAX_UPLOAD_BYTES` and `.pdf`/`.txt` extension + MIME checks in the
  endpoint (or a thin service) and raise `HTTPException(400)` for empty files or
  unsupported formats; translate Qdrant/OpenAI outages to 502/503.
- Use a single `document_id` (UUID) as the Qdrant point ID namespace (for
  example `uuid5` per chunk or Qdrant integer IDs) and store `document_id`,
  `chunk_index`, `text`, and `filename` in the point payload so Phase 4 can
  attribute sources.
- Persist `DocumentMetadata` with `chunk_count=len(chunks)` and
  `chunking_strategy` exactly as received, after vectors are upserted; keep the
  handler thin and business logic inside services.
- Do not add `RetrievalQAChain`/LangChain chains, FAISS, or ChromaDB at any
  point; all retrieval and completion logic stays explicit and hand-written.
```
